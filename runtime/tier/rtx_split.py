"""RTX PRO 6000 as a per-layer expert co-processor for GLM-5.3 EXL3 (hecate; runtime patch imported before the model loads).

Placement (EXL3_RTX_SPLIT = json from tools/placement/tier_placement.py): per main MoE layer, the routed experts are permuted so the GB300's set
is [0, first) and the RTX's band is [first, 256). The GB300 module is shrunk to its slice exactly like ExLlamaV3's CPU split
(the kernels mask the tail; cpu_post_load permutes the router to match); the tail experts load straight onto the RTX.

Per layer and forward (hooks cpu_split_submit / cpu_split_combine, called by BlockSparseMLP.forward right after routing and
after the local expert kernels):
  GB300 stream: D2H of the MoE input, selection and routing weights into pinned slots, flag ISSUE = gen
  RTX stream (enqueued at submit, no host thread): wait ISSUE >= gen, H2D, ExLlamaV3 coop kernel over the RTX band
               (min_expert = first: picks outside the band contribute nothing), D2H of the partial sums, flag DONE = gen
  GB300 stream at combine: wait DONE >= gen, H2D of the partials, add into the routed sum
One slot set serves every layer: stream order (the GB300 waits for DONE before the next layer's D2H) serializes them."""
import json, os, re, types
import torch
import torch.nn.functional as F
from cuda.bindings import driver as cu
import exllamav3.modules.block_sparse_mlp as bsm
from exllamav3.ext import exllamav3_ext as ext

PLACE = json.load(open(os.environ["EXL3_RTX_SPLIT"]))
RDEV = int(os.environ.get("EXL3_RTX_DEVICE", "1"))
CAP = int(os.environ.get("EXL3_RTX_CAP_ROWS", "2048"))
CHUNK = 32                                   # coop kernel rows per launch (scratch sized for 32 rows x top-8)
ZEROCOPY = os.environ.get("EXL3_RTX_ZEROCOPY", "1") == "1"   # kernels read/write the pinned slots in place (no copy-engine hops)
GEQ = cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
WDEF = cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT
_S = None
HANDOFF = None                               # set by tier_split: one kernel writes the slots + releases ISSUE
META = False                                 # set by tier_split: also write (layer, rows) into flags[4..5] before ISSUE
META_LAYER = None
_orig_split_load = bsm.BlockSparseMLP.cpu_maybe_split_load
print(f"rtx_split: placement for {len(PLACE)} layers, RTX = cuda:{RDEV}", flush=True)


def _ck(r):
    if isinstance(r, tuple):
        err, *rest = r
    else:
        err, rest = r, []
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return rest[0] if len(rest) == 1 else (rest or None)


class _Shared:
    def __init__(self, H, I, topk, gdev):
        self.H, self.topk = H, topk
        self.g, self.r = torch.device(gdev), torch.device(f"cuda:{RDEV}")
        self.nb_y, self.nb_s, self.nb_w, self.nb_o = CAP * H * 2, CAP * topk * 8, CAP * topk * 2, CAP * H * 4
        tot = self.nb_y + self.nb_s + self.nb_w + self.nb_o + 256
        for d in (self.g, self.r):          # make sure both primary contexts exist before the driver calls
            torch.zeros(1, device=d); torch.cuda.synchronize(d)
        with torch.cuda.device(self.g):
            self.hbuf = int(_ck(cu.cuMemHostAlloc(tot, cu.CU_MEMHOSTALLOC_PORTABLE | cu.CU_MEMHOSTALLOC_DEVICEMAP)))
        self.hy = self.hbuf; self.hs = self.hy + self.nb_y; self.hw = self.hs + self.nb_s; self.ho = self.hw + self.nb_w
        self.hf = (self.ho + self.nb_o + 63) // 64 * 64
        self.fdev = {}
        for d in (self.g, self.r):
            with torch.cuda.device(d):
                self.fdev[d.index] = int(_ck(cu.cuMemHostGetDevicePointer(self.hbuf, 0))) + (self.hf - self.hbuf)
        import ctypes
        self.flags = (ctypes.c_uint32 * 16).from_address(self.hf)
        # zero-copy views of the pinned slots on both devices (UVA: device pointer == host pointer)
        class _CAI:
            def __init__(s, ptr, shape, ts): s.__cuda_array_interface__ = {"shape": shape, "typestr": ts, "data": (ptr, False), "version": 3}
        self.v = {}
        for d in (self.g, self.r):
            with torch.cuda.device(d):
                base = int(_ck(cu.cuMemHostGetDevicePointer(self.hbuf, 0))); off = lambda a: base + (a - self.hbuf)
                self.v[d.index] = types.SimpleNamespace(
                    y=torch.as_tensor(_CAI(off(self.hy), (CAP, H), "<f2"), device=d),
                    s=torch.as_tensor(_CAI(off(self.hs), (CAP, topk), "<i8"), device=d),
                    w=torch.as_tensor(_CAI(off(self.hw), (CAP, topk), "<f2"), device=d),
                    o=torch.as_tensor(_CAI(off(self.ho), (CAP, H), "<f4"), device=d))
        for i in range(16): self.flags[i] = 0
        self.gen = 0
        with torch.cuda.device(self.r):
            r = self.r; n = 256
            self.x = torch.empty(CAP, H, dtype=torch.half, device=r)
            self.sel = torch.empty(CAP, topk, dtype=torch.long, device=r)
            self.rw = torch.empty(CAP, topk, dtype=torch.half, device=r)
            self.out = torch.empty(CAP, H, dtype=torch.float, device=r)
            self.had_g = torch.zeros(n, H, dtype=torch.half, device=r); self.had_u = torch.zeros_like(self.had_g)
            self.gu_g = torch.zeros(n, I, dtype=torch.float, device=r); self.gu_u = torch.zeros_like(self.gu_g)
            self.act = torch.zeros(n, I, dtype=torch.half, device=r); self.dout = torch.zeros(n, H, dtype=torch.float, device=r)
            self.ctr = torch.zeros(n * (I // 128) + CHUNK * (H // 128) + 2 + (n + 1) + n, dtype=torch.int32, device=r)
            self.stream = torch.cuda.Stream(r)
        self.part = torch.empty(CAP, H, dtype=torch.float, device=self.g)
        print(f"rtx_split: shared slots {tot / 2**20:.0f} MiB pinned, RTX scratch ready", flush=True)


def _layer_of(key):
    m = re.search(r"layers\.(\d+)\.", key + ".")
    return int(m.group(1)) if m else -1


def rtx_split_load(self, device, **kwargs):
    L = _layer_of(self.key)
    if str(L) not in PLACE or device is None or torch.device(device).type != "cuda" or self.cpu_split_first is not None:
        return _orig_split_load(self, device, **kwargs)
    global _S
    p = PLACE[str(L)]; perm, first = p["perm"], p["first"]
    assert len(perm) == self.num_experts and self.gated
    self._split_perm = perm; self._split_seeded = True; self._split_dynamic = False; self._split_map = None; self._split_spares = 0
    self.gates = [self.gates[e] for e in perm]; self.ups = [self.ups[e] for e in perm]; self.downs = [self.downs[e] for e in perm]
    tail = (self.gates[first:], self.ups[first:], self.downs[first:])
    tailset = set(tail[0] + tail[1] + tail[2])
    self._split_saved = None
    self.gates, self.ups, self.downs = self.gates[:first], self.ups[:first], self.downs[:first]
    self.modules = [m for m in self.modules if m not in tailset]
    self.num_local_experts = first; self.routing_first = 0; self.routing_last = first; self.cpu_split_first = first
    H, I = tail[0][0].in_features, tail[0][0].out_features
    if _S is None:
        _S = _Shared(H, I, self.num_experts_per_tok, device)
    stc, rdev = self.config.stc, _S.r
    keep, tabs, Ks = [], [], []
    with torch.cuda.device(rdev):
        for lins in tail:      # gate, up, down
            tr, su, sv = [], [], []
            for lin in lins:
                t = stc.get_tensor(lin.key + ".trellis", rdev, no_defer=True, arena=False)
                s1 = stc.get_tensor(lin.key + ".suh", rdev, no_defer=True, arena=False)
                s2 = stc.get_tensor(lin.key + ".svh", rdev, no_defer=True, arena=False)
                assert stc.get_tensor(lin.key + ".mul1", None, optional=True) is not None, "RTX split expects mul1 experts"
                keep += [t, s1, s2]; tr.append(t.data_ptr()); su.append(s1.data_ptr()); sv.append(s2.data_ptr())
                Ks.append(t.numel() * t.element_size() * 8 / (lin.in_features * lin.out_features))
            tabs += [torch.tensor(x, dtype=torch.int64, device=rdev) for x in (tr, su, sv)]
    kg, ku, kd = Ks[0], Ks[len(tail[0])], Ks[2 * len(tail[0])]
    self._rtx = types.SimpleNamespace(first=first, E=len(perm), tabs=tabs, keep=keep, K=(float(kg), float(ku), float(kd)), H=H)
    self.cpu_split_submit = types.MethodType(rtx_submit, self)
    self.cpu_split_combine = types.MethodType(rtx_combine, self)
    self.cpu_unload = types.MethodType(lambda s: None, self)
    print(f" -- RTX split: {self.key} [{first}..{len(perm)}) of {len(perm)} on cuda:{RDEV} (K {kg:.0f}/{ku:.0f}/{kd:.0f})", flush=True)


PF_ROWS = int(os.environ.get("EXL3_RTX_PREFILL_ROWS", "64"))
_PF_CHECK = [int(os.environ.get("EXL3_RTX_PREFILL_CHECK", "0"))]   # above this many rows: per-expert dequant + GEMM prefill


def _rtx_prefill(self, X, SEL, RW, OUT, bsz, counts):
    """Prefill on the RTX (> PF_ROWS rows), on S.stream: the coop decode kernel in 32-row chunks would stream every RTX
    expert's weights once per chunk; instead each expert is dequantized to FP16 once (trellis decode + Hadamards, as
    LinearEXL3.get_weight_tensor) and runs three GEMMs over all of its rows. counts (host ints, from the GB300 side)
    are the rows per RTX expert; the RTX sorts the same selection with the same stable argsort, so no sync here."""
    S, R = _S, self._rtx; first = R.first; nt = R.E - first; K8 = SEL.shape[-1]; H = R.H
    fs = SEL[:bsz].reshape(-1)
    order = torch.argsort(fs, stable=True)
    n_rtx = sum(counts); start = fs.numel() - n_rtx
    seg = order[start:]
    rows_all = seg // K8; w_all = RW[:bsz].reshape(-1)[seg].float()
    out = OUT[:bsz]; out.zero_()
    # groups of experts: each matrix of the group reconstructed in place into one reused stacked buffer (trellis decode +
    # Hadamards, as LinearEXL3.get_weight_tensor), then three grouped GEMMs per group
    from itertools import accumulate
    G = int(os.environ.get("EXL3_PREFILL_GROUP", "24"))
    nz = [(j, c) for j, c in enumerate(counts) if c]
    MAT = R.keep[1].numel() * R.keep[2].numel()          # elements of one expert matrix (gate / up / down alike)
    WB = getattr(S, "_pf_wbuf", None)
    if WB is None or WB.numel() < G * MAT or WB.device != X.device:
        WB = S._pf_wbuf = torch.empty(G * MAT, dtype=torch.half, device=X.device)
    def stackw(i0, grp):
        """[len(grp), in, out] fp16 view of WB holding matrix i0 + j of every expert j in grp"""
        _, su0, sv0 = R.keep[3 * (i0 + grp[0][0]): 3 * (i0 + grp[0][0]) + 3]
        W = WB[: len(grp) * su0.numel() * sv0.numel()].view(len(grp), su0.numel(), sv0.numel())
        for k, (j, _) in enumerate(grp):
            t, su, sv = R.keep[3 * (i0 + j): 3 * (i0 + j) + 3]
            ext.reconstruct_had_slice(W[k], t, su, sv, int(round(R.K[(i0 + j) // nt])), False, True, 0)
        return W
    xs = X[rows_all]; p0 = 0
    for g0 in range(0, len(nz), G):
        grp = nz[g0:g0 + G]; cs = [c for _, c in grp]; Pg = sum(cs)
        offs = torch.tensor(list(accumulate(cs)), dtype=torch.int32, pin_memory=True).to(X.device, non_blocking=True)
        x = xs[p0:p0 + Pg]
        g_ = torch._grouped_mm(x, stackw(0, grp), offs=offs)
        a = F.silu(g_) * torch._grouped_mm(x, stackw(nt, grp), offs=offs); del g_
        d = torch._grouped_mm(a.half(), stackw(2 * nt, grp), offs=offs); del a
        out.index_add_(0, rows_all[p0:p0 + Pg], d.float() * w_all[p0:p0 + Pg, None])
        p0 += Pg


def rtx_submit(self, y, bsz, selected_experts, routing_weights):
    S, R = _S, self._rtx
    assert bsz <= CAP, f"rtx_split: {bsz} rows > CAP {CAP}"
    pf_counts = None
    if bsz > PF_ROWS:     # rows per RTX expert, from the GB300 copy of the selection (one host sync per layer)
        pf_counts = torch.bincount(selected_experts.reshape(-1), minlength=R.E)[R.first:].tolist()
    S.gen += 1; G = S.gen
    y = y.contiguous(); sel = selected_experts.contiguous(); rw = routing_weights.contiguous()
    assert y.dtype == torch.half and sel.dtype == torch.long and rw.dtype == torch.half, (y.dtype, sel.dtype, rw.dtype)
    gs = cu.CUstream(torch.cuda.current_stream(y.device).cuda_stream)
    zc = ZEROCOPY and bsz > 1
    if zc and HANDOFF is not None:
        vg = S.v[S.g.index]
        HANDOFF(y.view(bsz, -1), sel.view(bsz, -1), rw.view(bsz, -1), vg.y.data_ptr(), vg.s.data_ptr(), vg.w.data_ptr(),
                int(S.fdev[S.g.index]) + 0, G, bsz, (int(S.fdev[S.g.index]) + 16) if META else 0, META_LAYER(self) if META else 0, 0, 0, 0)
    else:
        if zc:      # GB300 kernels write straight into the pinned slots over C2C
            vg = S.v[S.g.index]; vg.y[:bsz].copy_(y); vg.s[:bsz].copy_(sel); vg.w[:bsz].copy_(rw)
        else:
            _ck(cu.cuMemcpyDtoHAsync(S.hy, y.data_ptr(), bsz * S.H * 2, gs))
            _ck(cu.cuMemcpyDtoHAsync(S.hs, sel.data_ptr(), bsz * S.topk * 8, gs))
            _ck(cu.cuMemcpyDtoHAsync(S.hw, rw.data_ptr(), bsz * S.topk * 2, gs))
        if META:
            _ck(cu.cuStreamWriteValue32(gs, S.fdev[S.g.index] + 16, META_LAYER(self), WDEF))
            _ck(cu.cuStreamWriteValue32(gs, S.fdev[S.g.index] + 20, bsz, WDEF))
        _ck(cu.cuStreamWriteValue32(gs, S.fdev[S.g.index] + 0, G, WDEF))
    with torch.cuda.device(S.r), torch.cuda.stream(S.stream):
        rs = cu.CUstream(S.stream.cuda_stream)
        _ck(cu.cuStreamWaitValue32(rs, S.fdev[S.r.index] + 0, G, GEQ))
        if zc:  # the coop kernel reads its inputs and writes its partials in place over PCIe
            vr = S.v[S.r.index]; X, SEL, RW, OUT = vr.y, vr.s, vr.w, vr.o
        else:
            _ck(cu.cuMemcpyHtoDAsync(S.x.data_ptr(), S.hy, bsz * S.H * 2, rs))
            _ck(cu.cuMemcpyHtoDAsync(S.sel.data_ptr(), S.hs, bsz * S.topk * 8, rs))
            _ck(cu.cuMemcpyHtoDAsync(S.rw.data_ptr(), S.hw, bsz * S.topk * 2, rs))
            X, SEL, RW, OUT = S.x, S.sel, S.rw, S.out
        T = R.tabs
        if pf_counts is not None:
            if zc:     # stage the mapped slots on the device once; accumulate on the device; copy the sum back
                S.x[:bsz].copy_(X[:bsz]); S.sel[:bsz].copy_(SEL[:bsz]); S.rw[:bsz].copy_(RW[:bsz])
                _rtx_prefill(self, S.x, S.sel, S.rw, S.out, bsz, pf_counts)
                OUT[:bsz].copy_(S.out[:bsz])
            else:
                _rtx_prefill(self, X, SEL, RW, OUT, bsz, pf_counts)
            if _PF_CHECK[0] > 0:     # EXL3_RTX_PREFILL_CHECK: compare with the coop kernel on the first calls
                _PF_CHECK[0] -= 1
                ref = torch.empty_like(OUT[:bsz])
                Xs, Ss, Ws = (S.x, S.sel, S.rw) if zc else (X, SEL, RW)
                for c in range(0, bsz, CHUNK):
                    n = min(CHUNK, bsz - c)
                    ext.exl3_moe_coop(Xs[c:c + n], Ss[c:c + n], Ws[c:c + n], R.first, R.E, R.H,
                                      T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], None, None, None,
                                      R.K[0], R.K[1], R.K[2], False, True, 0, 0.0, True,
                                      S.had_g, S.had_u, S.gu_g, S.gu_u, S.act, S.dout, S.ctr, ref[c:c + n], None, None)
                got = OUT[:bsz].to(ref.device)
                print(f" -- rtx prefill check {self.key}: {bsz} rows, rel diff vs coop "
                      f"{((got - ref).norm() / ref.norm().clamp_min(1e-12)).item():.2e}", flush=True)
        for c in (range(0, bsz, CHUNK) if pf_counts is None else ()):
            n = min(CHUNK, bsz - c)
            ext.exl3_moe_coop(X[c:c + n], SEL[c:c + n], RW[c:c + n], R.first, R.E, R.H,
                              T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], None, None, None,
                              R.K[0], R.K[1], R.K[2], False, True, 0, 0.0, True,
                              S.had_g, S.had_u, S.gu_g, S.gu_u, S.act, S.dout, S.ctr, OUT[c:c + n], None, None)
        if not zc:
            _ck(cu.cuMemcpyDtoHAsync(S.ho, S.out.data_ptr(), bsz * S.H * 4, rs))
        _ck(cu.cuStreamWriteValue32(rs, S.fdev[S.r.index] + 4, G, WDEF))
    return None, ("rtx", G, bsz, zc)


def rtx_combine(self, final_hidden_states, cpu_partial, cpu_pending, shape):
    if not (isinstance(cpu_pending, tuple) and cpu_pending and cpu_pending[0] == "rtx"):
        return final_hidden_states
    S = _S; _, G, bsz, zc = cpu_pending
    gs = cu.CUstream(torch.cuda.current_stream(final_hidden_states.device).cuda_stream)
    _ck(cu.cuStreamWaitValue32(gs, S.fdev[S.g.index] + 4, G, GEQ))
    f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
    if zc:      # add straight from the pinned partials over C2C
        f2 += S.v[S.g.index].o[:bsz]
    else:
        _ck(cu.cuMemcpyHtoDAsync(S.part.data_ptr(), S.ho, bsz * S.H * 4, gs))
        f2 += S.part[:bsz]
    return final_hidden_states


bsm.BlockSparseMLP.cpu_maybe_split_load = rtx_split_load
