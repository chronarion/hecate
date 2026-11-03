"""Hecate tier split for GLM-5.3 on the DGX Station GB300 (imported before ExLlamaV3 loads the model).

Every main MoE layer's 256 routed experts are permuted (router rows included) into three tiers:
    [0, hot)          int4 (GPTQ from BF16) in GB300 HBM, run by the moe4 kernels (decode <= 8 rows; prefill: moe4 in
                      8-row chunks up to 64 rows, else the moe4p grouped GEMMs)
    [hot, hot + far)  int4 in pinned Grace memory: decode on the Grace cores (cpuband.cpp, 60 threads); prefill on the GB300
                      (moe4p reads them in place over C2C). With EXL3_TIER_RTX_I4=k the first k of them also live in RTX VRAM
                      and run there as int4 in decode (rtx_graph.py), freeing HBM when the hot band is made smaller
    [first, 256)      3-bit EXL3 trellis on the RTX PRO 6000 (rtx_split.py loads it; rtx_graph.py runs it in decode as one
                      CUDA graph per step that waits on per-layer ISSUE flags in pinned host memory)
The MTP layer has no RTX band: all its experts are int4 in HBM (identity order).

Decode (<= 8 rows) per MoE layer, on the GB300 stream: one handoff kernel writes the layer's MoE input / selection / routing
weights into its pinned mailbox and releases the RTX and Grace jobs; the shared expert runs into the hot-band accumulator
(EXL3_TIER_SH_FAST); moe4 adds the hot band; one combine kernel spins on the RTX DONE flag (and the Grace CPU_DONE),
adds both partials and zeroes the accumulator.

Placement file (EXL3_TIER_SPLIT): {layer: {"perm": [256 ids], "hot": n, "far": m, "first": n + m}} (tools/placement).
int4 files (EXL3_TIER_U4): layer{L}.safetensors / layer{L}.r*.safetensors with "{id}.{gate,up,down}.{q,s}"."""
import glob, json, os, sys, types
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.cpp_extension import load as _load_ext
import exllamav3.modules.block_sparse_mlp as bsm
import rtx_split as rs                     # RTX band load, router permutation, per-layer (prefill) RTX path
import rtx_graph as rg                     # decode RTX band as one CUDA graph per step

PLACE = json.load(open(os.environ["EXL3_TIER_SPLIT"]))
U4DIR = os.environ["EXL3_TIER_U4"]
MOE4_SRC = os.environ.get("EXL3_MOE4_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe4"))
BUILD = os.environ.get("EXL3_TIER_BUILD", "/root/.cache/hecate-tier")   # extension build dirs (writable)
def _bd(d):
    os.makedirs(d, exist_ok=True); return d
CPU_THREADS = int(os.environ.get("EXL3_TIER_CPU_THREADS", "60"))
CPU_CORE0 = int(os.environ.get("EXL3_TIER_CPU_CORE0", "12"))   # workers on cores CORE0.., the server's own threads below
SH_FAST = os.environ.get("EXL3_TIER_SH_FAST", "1") == "1"       # shared expert straight into the hot-band accumulator
PREFILL_MOE4 = os.environ.get("EXL3_TIER_PREFILL_MOE4", "1") == "1"
_PF_GROUP = int(os.environ.get("EXL3_PREFILL_GROUP", "24"))     # experts per batched-dequant group in long prefill
# EXL3_TIER_PREFILL_MOE4P=1 (default): prefill above 64 rows on the moe4p grouped GEMMs (moe4/moe4p.cu, int4 decoded in the
# GEMM, the Grace tier read in place), at most EXL3_TIER_P4_NPOS picks per call (scratch ~44 KB per pick); 0 = batched
# bf16 dequant + grouped GEMMs
PF4 = os.environ.get("EXL3_TIER_PREFILL_MOE4P", "1") == "1"
# decode: the handoff kernel (MoE input + routing into the RTX / Grace slots, ISSUE release) runs on a side stream, so the
# GB300's shared expert and hot band start without it; joined before the combine (the only GB300 consumer of its effects)
HSIDE = os.environ.get("EXL3_TIER_HSIDE", "0") == "1"
# EXL3_TIER_COMBINE_NORM=1: a decode MoE layer whose output goes to the next block's fused input norm leaves its combine
# to that norm: cn.cu waits for the bands, adds the partials, updates the residual and normalizes in one kernel
COMBINE_NORM = os.environ.get("EXL3_TIER_COMBINE_NORM", "0") == "1"
# EXL3_CN_PDL=1 (with COMBINE_NORM): that kernel launches as a programmatic dependent of moe4 B (combine_norm_pdl): it
# waits for the RTX DONE and the Grace CPU_DONE itself (no wait_reset_k), loads both partials, the residual and the norm
# weight while B drains and reads the hot band only after griddepcontrol.wait; the layer skips the zero-filled routed
# sum it never reads (BlockSparseMLP.tier_nofill). Bit-identical.
CN_PDL = COMBINE_NORM and os.environ.get("EXL3_CN_PDL", "0") == "1"
# EXL3_TIER_SH_MOE4=1: decode rows <= 8 run the shared expert inside moe4 (its int8 gate / up units in phase A's stream-K
# list, act with weight 1, its down units in phase B) instead of the separate gate+up and down GEMVs before moe4
SH_MOE4 = os.environ.get("EXL3_TIER_SH_MOE4", "0") == "1"
# EXL3_MOE4_XF=1: the post-attention norm + router kernel (EXL3_NORM_ROUTE handoff) also writes the MoE input in moe4's
# B-fragment order, and moe4 phase A reads that copy (two 16-byte loads per unit instead of eight 4-byte ones)
MOE4_XF = os.environ.get("EXL3_MOE4_XF", "0") == "1"
NPOS4 = int(os.environ.get("EXL3_TIER_P4_NPOS", "16384"))
_VG = {"cap": False}   # set by exl3_vgraph while it captures a whole-forward graph (the RTX graph is launched outside)
H_, I_ = 6144, 2048

moe4 = _load_ext(name="moe4", sources=[f"{MOE4_SRC}/moe4.cu"], build_directory=_bd(f"{BUILD}/build"), verbose=False,
                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a", "--use_fast_math"])
moe4p = _load_ext(name="moe4p", sources=[f"{MOE4_SRC}/moe4p.cu"], build_directory=_bd(f"{BUILD}/build_moe4p"), verbose=False,
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a", "--use_fast_math"]) if PF4 else None
# EXL3_PREFILL_GB300_ROWS > 0: prefill chunks above this many rows run every routed expert on the GB300's tensor cores
# (_prefill_gb300: moe4t for the int4 tiers, moe3t for the RTX band from its pinned Grace copy); the RTX idles
moe4t = moe3t = None
if rs.GB_ROWS > 0:
    _f = ["-O3", "-gencode=arch=compute_103a,code=sm_103a", "--use_fast_math"]
    moe4t = _load_ext(name="moe4t", sources=[f"{MOE4_SRC}/moe4t.cu"], build_directory=_bd(f"{BUILD}/build_moe4t"), verbose=False,
                      extra_cuda_cflags=_f)
    moe3t = _load_ext(name="moe3t", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe3", "moe3t.cu")],
                      build_directory=_bd(f"{BUILD}/build_moe3t"), verbose=False, extra_cuda_cflags=_f)
cn = _load_ext(name="cn", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "cn.cu")],
               build_directory=_bd(f"{BUILD}/build_cn"), verbose=False,
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a", "--use_fast_math"]) if COMBINE_NORM else None
_cb = _load_ext(name="cpuband", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpuband.cpp")],
                build_directory=_bd(f"{BUILD}/build_cpuband"), verbose=False, extra_cflags=["-O3", "-mcpu=neoverse-v2"])
os.sched_setaffinity(0, set(range(CPU_CORE0)))     # the main thread and its later children stay off the Grace workers
rs.META = True; rs.META_LAYER = lambda m: m._L
rs.HANDOFF = moe4.handoff
_CPU = types.SimpleNamespace(started=False, out=None)

RTX_I4 = int(os.environ.get("EXL3_TIER_RTX_I4", "0") or 0)
if RTX_I4:
    moe4r = _load_ext(name="moe4r", sources=[f"{MOE4_SRC}/moe4.cu"], build_directory=_bd(f"{BUILD}/build_moe4r"),
                      verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"])
    rg.MOE4R = moe4r
_T = None   # GB300-side shared state of the hot band
print(f"tier_split: placement for {len(PLACE)} layers, int4 from {U4DIR}"
      + (f", {RTX_I4} far experts per layer as int4 on the RTX" if RTX_I4 else ""), flush=True)


def _unpack_moe4(q, N, K):
    """moe4 fragment-order int4 -> codes [N, K] uint8 (inverse of the packing)"""
    c = torch.stack((q & 15, q >> 4), dim=-1).view(N // 64, K // 64, 4, 8, 4, 4, 2, 2, 2)
    return c.permute(0, 2, 8, 3, 1, 5, 7, 4, 6).reshape(N, K)


def _rowmajor_i4(q, N, K):
    c = _unpack_moe4(q.view(-1), N, K)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()


def _cpu_start():
    """start the Grace pool: its own ISSUE / DONE / meta lines; x / sel / rw come from each layer's mailbox"""
    S = rs._S
    _CPU.out = torch.zeros(8, H_, dtype=torch.float, pin_memory=True)
    with torch.cuda.device(S.g):
        _CPU.ctl = int(rs._ck(rs.cu.cuMemHostAlloc(256, rs.cu.CU_MEMHOSTALLOC_PORTABLE | rs.cu.CU_MEMHOSTALLOC_DEVICEMAP)))
        _CPU.ctl_dev = int(rs._ck(rs.cu.cuMemHostGetDevicePointer(_CPU.ctl, 0)))
    import ctypes; ctypes.memset(_CPU.ctl, 0, 256); _CPU.gen = 0
    base, stride, oy, os_, ow = rg.slot_layout()
    _cb.start(_CPU.ctl, _CPU.ctl + 128, _CPU.ctl + 64, 0, 0, 0, _CPU.out.data_ptr(), CPU_THREADS, CPU_CORE0, base, stride, oy, os_, ow)
    _CPU.started = True
    print(f"tier_split: Grace band pool started ({CPU_THREADS} threads on cores {CPU_CORE0}..{CPU_CORE0 + CPU_THREADS - 1})", flush=True)


def _far(m):
    h = getattr(m, "_hot", None)
    return h is not None and h.far > 0


def _cpu_partial(self, dev, bsz):
    """decode: make the stream wait for this layer's Grace job (CPU_DONE >= gen, then reset to 0, so a stale larger
    generation, e.g. a captured graph's reserved one, can never satisfy a later wait); returns the partial's address"""
    if not _far(self):
        return 0
    g = getattr(self, "_cpu_gen", None)
    if g is None or bsz > 8:
        return 0
    self._cpu_gen = None
    rg._fk.wait_reset(_CPU.ctl_dev + 64, g, 0, dev.index if dev.index is not None else 0)
    return _CPU.out.data_ptr()


def _cpu_wait_args(self, bsz):
    """EXL3_CN_PDL: (partial address, CPU_DONE address, generation) of this layer's Grace job for combine_norm_pdl, which
    waits for CPU_DONE >= generation and resets it itself; (0, 0, 0) if the layer has none"""
    if not _far(self):
        return 0, 0, 0
    g = getattr(self, "_cpu_gen", None)
    if g is None or bsz > 8:
        return 0, 0, 0
    self._cpu_gen = None
    return _CPU.out.data_ptr(), _CPU.ctl_dev + 64, g


def _tier_nofill(pend):
    """EXL3_CN_PDL: the decode layers whose combine (tier_combine, "graph") never reads the routed sum before zeroing it"""
    return isinstance(pend, tuple) and len(pend) == 2 and isinstance(pend[0], tuple) and pend[0][0] == "graph"


class _HotShared:
    def __init__(self, dev):
        self.dev = torch.device(dev)
        self.scr = moe4.scratch(self.dev)
        self.y = torch.zeros(8, H_, dtype=torch.float, device=self.dev)   # hot-band accumulator (decode rows)
        self.cctr = torch.zeros(1, dtype=torch.int32, device=self.dev)    # combine_norm_pdl's CTA counter (EXL3_CN_PDL)
        self.xf = torch.zeros(8, H_, dtype=torch.half, device=self.dev)   # the decode MoE input in fragment order (EXL3_MOE4_XF)
        sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self.blocks, self.blocks_a = sms * 2, sms * 3
        self.h_stream = torch.cuda.Stream(self.dev) if HSIDE else None
        self.bytes = self.far_bytes = self.rtx4_bytes = 0


def tier_split_load(self, device, **kwargs):
    L = rs._layer_of(self.key)
    if str(L) not in PLACE or device is None or torch.device(device).type != "cuda" or self.cpu_split_first is not None:
        return rs._orig_split_load(self, device, **kwargs)
    global _T
    p = PLACE[str(L)]; perm, hot, far, first = p["perm"], p.get("hot", 0), p.get("far", 0), p["first"]
    nortx = first >= len(perm)
    if nortx:
        # no RTX band (the MTP layer): identity order, split hooks enabled without rtx_split
        assert perm == list(range(len(perm))), "no-RTX tier layers keep the checkpoint order"
        self._split_perm = perm; self._split_seeded = True; self._split_dynamic = False; self._split_map = None
        self._split_spares = 0; self._split_saved = None
        self.cpu_split_first = first; self.routing_first = 0; self.routing_last = first
        self.cpu_unload = types.MethodType(lambda s_: None, self)
        if rs._S is None:
            rs._S = rs._Shared(self.gates[0].in_features, self.gates[0].out_features, self.num_experts_per_tok, device)
    else:
        rs.PLACE[str(L)] = {"perm": perm, "first": first}
        rs.rtx_split_load(self, device, **kwargs)                 # RTX band + router permutation
    self._nortx = nortx
    nb = hot + far
    assert hot > 0 and first == nb, f"layer {L}: Hecate placements have no GB300 trellis band (first == hot + far)"
    # shrink the GB300 module to the (empty) trellis band: the int4 tiers' trellis never loads
    keepset = set(self.gates[:nb] + self.ups[:nb] + self.downs[:nb])
    self.gates, self.ups, self.downs = self.gates[nb:], self.ups[nb:], self.downs[nb:]
    self.modules = [m for m in self.modules if m not in keepset]
    self.num_local_experts = 0; self.routing_first = nb; self.routing_last = first

    def _load_local_empty(*a, _m=self, **k):
        # ExLlamaV3's forward then takes its "empty slice" branch (zeros) and runs the shared expert on its own
        _m.bc = None; _m.bc_sh_exp = False; _m.is_quantized = False; _m.support_quant_paths = False
        _m.support_bc_bszn = False; _m.support_fused = False; _m.fused_mode_buffers = None; _m.experts_cfg = None
    self.load_local = _load_local_empty
    if _T is None:
        _T = _HotShared(device)
    keep = []; tab = {k: [] for k in ("gate_w", "up_w", "down_w", "gate_s", "up_s", "down_s")}
    files = [safe_open(fn, "pt", device="cpu") for fn in sorted(glob.glob(f"{U4DIR}/layer{L}.safetensors") + glob.glob(f"{U4DIR}/layer{L}.r*.safetensors"))]
    where = {k: f for f in files for k in f.keys()}
    get = lambda k: where[k].get_tensor(k)
    for e in perm[:hot]:
        for m in ("gate", "up", "down"):
            q = get(f"{e}.{m}.q").to(_T.dev); s = get(f"{e}.{m}.s").to(_T.dev)
            keep += [q, s]; tab[m + "_w"].append(q.data_ptr()); tab[m + "_s"].append(s.data_ptr())
            _T.bytes += q.numel() + s.numel() * 4
    keep_far = []
    if far:
        # one pinned buffer per layer, 256-byte slots (the caching host allocator would round each tensor up to 2^n)
        src = [get(f"{e}.{m}.{t}") for e in perm[hot:nb] for m in ("gate", "up", "down") for t in ("q", "s")]
        offs, o = [], 0
        for t in src: offs.append(o); o += (t.numel() * t.element_size() + 255) // 256 * 256
        buf = torch.empty(o, dtype=torch.uint8, pin_memory=True); keep_far = [buf]
        for t, o0 in zip(src, offs):
            v = buf[o0: o0 + t.numel() * t.element_size()].view(t.dtype).view(t.shape); v.copy_(t); keep.append(v)
        for i in range(0, len(src), 2):      # pointer tables of the cold tier (read in place by moe4 in short prefill)
            m = ("gate", "up", "down")[(i // 2) % 3]
            tab[m + "_w"].append(keep[6 * hot + i].data_ptr()); tab[m + "_s"].append(keep[6 * hot + i + 1].data_ptr())
        _T.far_bytes += buf.numel()
        k4 = min(RTX_I4, far) if not nortx else 0
        if k4:
            rdev = rs._S.r      # the first k far experts also in RTX VRAM (same int4 weights)
            r4keep = [t.to(rdev) for t in keep[6 * hot: 6 * hot + 6 * k4]]
            r4tab = {n: torch.tensor([r4keep[6 * j + o].data_ptr() for j in range(k4)], dtype=torch.int64, device=rdev)
                     for n, o in (("gate_w", 0), ("gate_s", 1), ("up_w", 2), ("up_s", 3), ("down_w", 4), ("down_s", 5))}
            self._rtx4 = types.SimpleNamespace(k=k4, lo=hot, keep=r4keep, tab=r4tab)
            _T.rtx4_bytes += sum(t.numel() * t.element_size() for t in r4keep)
        cg, cu_, cd, sg, su, sd = [], [], [], [], [], []
        for j in range(k4, far):     # the Grace pool: row-major int4 copies of its experts
            k6 = keep[6 * hot + 6 * j: 6 * hot + 6 * j + 6]
            cg.append(_rowmajor_i4(k6[0], I_, H_)); sg.append(k6[1].float().contiguous())
            cu_.append(_rowmajor_i4(k6[2], I_, H_)); su.append(k6[3].float().contiguous())
            cd.append(_rowmajor_i4(k6[4], H_, I_)); sd.append(k6[5].float().contiguous())
        keep_far += cg + cu_ + cd + sg + su + sd
        _cb.add_layer(L, hot + k4, len(rg.MODS), cg, cu_, cd, sg, su, sd)   # slot = this layer's index in rg.MODS
    del files, where
    self._hot = types.SimpleNamespace(n=nb, hot=hot, far=far, keep=keep, keep_far=keep_far,
                                      tab={k: torch.tensor(v, dtype=torch.int64, device=_T.dev) for k, v in tab.items()})
    self._L = L
    if not nortx:
        rg.MODS.append(self)
    self.cpu_split_submit = types.MethodType(tier_submit, self)
    self.nr_handoff = types.MethodType(tier_nr_handoff, self)     # (EXL3_NORM_ROUTE)
    self.cpu_split_combine = types.MethodType(tier_combine, self)
    if CN_PDL:
        self.tier_nofill = _tier_nofill
    if SH_FAST and getattr(self, "shared_experts", None) is not None:
        _mf = self.forward
        def _fwd(x, params, out_dtype=None, _m=self, _f=_mf):
            sh0 = _m.shared_experts
            _m._sh_fast = getattr(sh0, "bc", None) is not None and getattr(sh0.bc, "run_bszN", None) is not None
            _m._prm = params if (COMBINE_NORM and _m._sh_fast and out_dtype is None) else None
            try:
                return _f(x, params, out_dtype)
            finally:
                _m.shared_experts = sh0; _m._sh_fast = False; _m._prm = None
        self.forward = _fwd
    print(f" -- tier split: {self.key} int4 [0..{hot}) GB300" + (f", int4 [{hot}..{nb}) Grace" if far else "")
          + f", [{first}..{len(perm)}) RTX trellis; int4 {_T.bytes / 1e9:.1f} GB HBM"
          + (f" + {_T.far_bytes / 1e9:.1f} GB Grace" if far else "")
          + (f" + {_T.rtx4_bytes / 1e9:.1f} GB RTX" if _T.rtx4_bytes else "") + " so far", flush=True)


_DQ_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
// Batched dequant of moe4 int4 matrices (fragment order, SPW 4: uint4 i = ((gr * kb_n + kb) * 4 + s) * 32 + lane holds
// a 16 x 64 tile: rows (gr * 4 + s) * 16 + lane / 4 + 8 * rh, cols kb * 64 + t * 16 + h * 8 + (lane % 4) * 2 + c) into
// out[g] = bf16 [N, K] row-major times the row scale. Block = 8 warps = one 16-row strip x 512 columns, staged in shared
// memory, written as whole 1 KB row segments.
__global__ void dq_k(const long* __restrict__ qp, const long* __restrict__ sp, __nv_bfloat16* __restrict__ out, int N, int K)
{
    __shared__ __nv_bfloat16 tile[16][512 + 8];
    const int g = blockIdx.z, strip = blockIdx.x, kgrp = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, kb_n = K / 64, kb = kgrp * 8 + warp;
    const uint4* q = (const uint4*) qp[g]; const float* sc = (const float*) sp[g];
    const int gr = strip >> 2, s = strip & 3, row0 = strip * 16;
    if (kb < kb_n)
    {
        const uint4 v = q[((size_t) (gr * kb_n + kb) * 4 + s) * 32 + lane];
        const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
        const int gg = lane >> 2, qq = lane & 3;
        const float s0 = sc[row0 + gg], s1 = sc[row0 + gg + 8];
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib)
            {
                const int c = nib >> 2, h = (nib >> 1) & 1, rh = nib & 1;
                const float val = (float) (int) (((ww[t] >> (4 * nib)) & 15) - 8) * (rh ? s1 : s0);
                tile[gg + 8 * rh][warp * 64 + t * 16 + h * 8 + qq * 2 + c] = __float2bfloat16(val);
            }
    }
    __syncthreads();
    const int cols = min(512, K - kgrp * 512);
    __nv_bfloat16* o = out + (size_t) g * N * K + (size_t) row0 * K + kgrp * 512;
    for (int idx = threadIdx.x; idx < 16 * (cols / 8); idx += blockDim.x)
    {
        const int r = idx / (cols / 8), c8 = idx % (cols / 8);
        *(uint4*) (o + (size_t) r * K + c8 * 8) = *(const uint4*) (&tile[r][c8 * 8]);
    }
}
void dequant_batched(at::Tensor qp, at::Tensor sp, at::Tensor out, int64_t N, int64_t K)
{
    const int G = qp.numel();
    dim3 grid(N / 16, (K / 64 + 7) / 8, G);
    dq_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(qp.data_ptr<long>(), sp.data_ptr<long>(),
        (__nv_bfloat16*) out.data_ptr(), (int) N, (int) K);
}
"""
_dq = None
def _dqext():
    global _dq
    if _dq is None:
        from torch.utils.cpp_extension import load_inline
        _dq = load_inline(name="moe4dq", cpp_sources="void dequant_batched(at::Tensor qp, at::Tensor sp, at::Tensor out, int64_t N, int64_t K);",
                          cuda_sources=_DQ_SRC, functions=["dequant_batched"], build_directory=_bd(f"{BUILD}/build_moe4dq"),
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"])
    return _dq


def _dq_stack(ks, j, N, K, dev):
    """bf16 [G, N, K] of matrix j (0 gate, 1 up, 2 down) of the experts in ks, one launch"""
    # pinned: a pageable host->device copy blocks the host until the stream reaches it (no overlap)
    qp = torch.tensor([k[2 * j].data_ptr() for k in ks], dtype=torch.int64, pin_memory=True).to(dev, non_blocking=True)
    sp = torch.tensor([k[2 * j + 1].data_ptr() for k in ks], dtype=torch.int64, pin_memory=True).to(dev, non_blocking=True)
    out = torch.empty((len(ks), N, K), dtype=torch.bfloat16, device=dev)
    _dqext().dequant_batched(qp, sp, out, N, K)
    return out


def _hot_prefill(self, y, sel, rw):
    """> 8 rows, both int4 tiers on the GB300. Up to 64 rows: the moe4 decode kernels in 8-row chunks (weights stream
    once per chunk, the Grace tier read in place). Longer: the moe4p grouped GEMMs (_hot_prefill4), or with
    EXL3_TIER_PREFILL_MOE4P=0 the bf16 dequant path (_hot_prefill_dq)."""
    Hh = self._hot; T = _T; R = y.shape[0]
    if R <= 64 and PREFILL_MOE4:
        out = torch.empty(R, H_, dtype=torch.float, device=y.device)
        sel = sel.contiguous(); rw = rw.contiguous()
        for c in range(0, R, 8):
            n = min(8, R - c)
            yh = out[c:c + n]; yh.zero_()
            moe4.forward(y[c:c + n], sel[c:c + n], rw[c:c + n], Hh.n, Hh.tab["gate_w"], Hh.tab["up_w"], Hh.tab["down_w"],
                         Hh.tab["gate_s"], Hh.tab["up_s"], Hh.tab["down_s"], *T.scr, yh, T.blocks, T.blocks_a, 0)
        return out
    out = torch.zeros(y.shape[0], H_, dtype=torch.float, device=y.device)
    if moe4p is not None:
        _hot_prefill4(self, y, sel, rw, out)
        if _P4_CHECK[0] > 0:     # EXL3_TIER_PREFILL_CHECK: compare with the bf16 dequant path on the first calls
            _P4_CHECK[0] -= 1
            ref = _hot_prefill_dq(self, y, sel, rw, torch.zeros_like(out))
            print(f" -- int4 prefill check {self.key}: {R} rows, rel diff vs bf16 dequant "
                  f"{((out - ref).norm() / ref.norm().clamp_min(1e-12)).item():.2e}", flush=True)
        return out
    return _hot_prefill_dq(self, y, sel, rw, out)


_P4_CHECK = [int(os.environ.get("EXL3_TIER_PREFILL_CHECK", "0"))]


def _hot_prefill_dq(self, y, sel, rw, out):
    """per group of experts one batched dequant per matrix into a stacked bf16 buffer and three grouped GEMMs (the Grace
    tier copied to HBM per group)"""
    Hh = self._hot
    # (token, slot) pairs routed to the int4 tiers, sorted by expert once: one host sync for the per-expert counts
    K8 = sel.shape[-1]; fs = sel.reshape(-1)
    idx = (fs < Hh.n).nonzero(as_tuple=True)[0]
    if idx.numel() == 0: return out
    es = fs[idx]; order = torch.argsort(es, stable=True); es = es[order]; idx = idx[order]
    rows_all = idx // K8; w_all = rw.reshape(-1)[idx].float()
    counts = torch.bincount(es, minlength=Hh.n).tolist()
    from itertools import accumulate
    nz = [(e, c) for e, c in enumerate(counts) if c]
    xs = y[rows_all].to(torch.bfloat16); p0 = 0
    for g0 in range(0, len(nz), _PF_GROUP):
        grp = nz[g0:g0 + _PF_GROUP]; cs = [c for _, c in grp]; Pg = sum(cs)
        offs = torch.tensor(list(accumulate(cs)), dtype=torch.int32, pin_memory=True).to(y.device, non_blocking=True)
        ks = [[t.to(y.device, non_blocking=True) for t in Hh.keep[6 * e: 6 * e + 6]] for e, _ in grp]
        x = xs[p0:p0 + Pg]
        Wg = _dq_stack(ks, 0, I_, H_, y.device).transpose(1, 2)
        g_ = torch._grouped_mm(x, Wg, offs=offs); del Wg
        Wu = _dq_stack(ks, 1, I_, H_, y.device).transpose(1, 2)
        a = F.silu(g_) * torch._grouped_mm(x, Wu, offs=offs); del Wu, g_
        Wd = _dq_stack(ks, 2, H_, I_, y.device).transpose(1, 2)
        d = torch._grouped_mm(a, Wd, offs=offs); del Wd, a
        out.index_add_(0, rows_all[p0:p0 + Pg], d.float() * w_all[p0:p0 + Pg, None])
        p0 += Pg
    return out


def _hot_prefill4(self, y, sel, rw, out):
    """> 64 rows on the moe4p kernels: the int4 tiers' picks sorted by expert (one host sync for the counts), row blocks
    of <= 64 picks per expert, the int4 decoded inside the GEMMs (hot experts from HBM, cold ones read in place from Grace
    memory); nothing is dequantized to bf16"""
    Hh = self._hot; K8 = sel.shape[-1]; fs = sel.reshape(-1)
    order = torch.argsort(fs, stable=True)                   # the int4 tiers' ids [0, n) sort first
    allc = getattr(self, "_pf_counts", None); self._pf_counts = None     # set by this call's rtx_submit (same selection)
    counts = allc[:Hh.n] if allc is not None else torch.bincount(fs, minlength=Hh.n)[:Hh.n].tolist()
    n4 = sum(counts)
    if n4 == 0:
        return out
    seg = order[:n4]
    tok = (seg // K8).int(); eid = fs[seg].int(); wt = rw.reshape(-1)[seg].float()
    scr = getattr(_T, "p4scr", None)
    if scr is None:
        scr = _T.p4scr = moe4p.scratch(y.device, max(NPOS4, rs.CAP))
    t = Hh.tab
    for p0, p1, rbt in rs.pick_groups(counts, scr[0].shape[1]):
        moe4p.forward(y, tok[p0:p1], eid[p0:p1], wt[p0:p1], torch.from_numpy(rbt).pin_memory().to(y.device, non_blocking=True),
                      t["gate_w"], t["up_w"], t["down_w"], t["gate_s"], t["up_s"], t["down_s"], *scr, out)
    return out


def _dev32(a, dev):
    return torch.from_numpy(a).pin_memory().to(dev, non_blocking=True)


def _prefill_gb300(self, y, sel, rw):
    """Long prefill chunk with every routed expert on the GB300's tensor cores (one host sync for the per-expert counts):
    the RTX band on moe3t from its pinned Grace copy (rtx_split._host_band, read in place over C2C) on a second stream with
    its own scratch, overlapping the int4 tiers on moe4t (hot experts from HBM, cold ones read in place over C2C) on the
    current stream; both add into one fp32 output. Row blocks of <= 256 picks."""
    Hh, R, T = self._hot, self._rtx, _T
    K8 = sel.shape[-1]; fs = sel.reshape(-1)
    order = torch.argsort(fs, stable=True)
    allc = torch.bincount(fs, minlength=R.E).tolist()
    out = torch.zeros(y.shape[0], H_, dtype=torch.float, device=y.device)
    cur = torch.cuda.current_stream(y.device)
    ps = max(NPOS4, rs.CAP)
    if getattr(T, "p4scr", None) is None:
        T.p4scr = (moe4p or moe4t).scratch(y.device, ps)
    if getattr(T, "gb", None) is None:
        T.gb = types.SimpleNamespace(stream=torch.cuda.Stream(y.device), scr=moe3t.scratch(y.device, ps))
    c3 = allc[R.first:]; n3 = sum(c3)
    if n3:
        T.gb.stream.wait_stream(cur)
        with torch.cuda.stream(T.gb.stream):
            seg = order[fs.numel() - n3:]
            tok = (seg // K8).int(); eid = (fs[seg] - R.first).int(); wt = rw.reshape(-1)[seg].float()
            for p0, p1, rbt in rs.pick_groups(c3, ps, 256, True):
                moe3t.forward(y, tok[p0:p1], eid[p0:p1], wt[p0:p1], _dev32(rbt, y.device), *R.htabs, *T.gb.scr, out)
    c4 = allc[:Hh.n]; n4 = sum(c4)
    if n4:
        gu, ar, d = T.p4scr[0], T.p4scr[1], T.p4scr[2]
        seg = order[:n4]
        tok = (seg // K8).int(); eid = fs[seg].int(); wt = rw.reshape(-1)[seg].float()
        t = Hh.tab
        for p0, p1, rbt in rs.pick_groups(c4, ps, 256, True):
            moe4t.forward(y, tok[p0:p1], eid[p0:p1], wt[p0:p1], _dev32(rbt, y.device), t["gate_w"], t["up_w"], t["down_w"],
                          t["gate_s"], t["up_s"], t["down_s"], gu, ar, d.view(1, ps, H_), out)
    if n3:
        cur.wait_stream(T.gb.stream)
    return out


def _pre_handoff(self, bsz):
    """decode handoff bookkeeping: the step's RTX graph on the first tier layer, this layer's Grace job generation;
    returns the Grace job (meta, layer, flag, gen) or None"""
    L = self._L
    if self is rg.MODS[0] and not _VG["cap"]:         # (a captured verify graph launches it before each replay)
        rg.launch(bsz)                                # the whole step's RTX band (waits on each layer's ISSUE)
    if not _far(self):
        return None
    if _VG["cap"]:                                    # reserved range in captured graphs: never equal to an eager job
        self._cpu_gen = 0x40000000 | L
    else:
        _CPU.gen += 1; self._cpu_gen = _CPU.gen
    return (_CPU.ctl_dev + 128, L, _CPU.ctl_dev, self._cpu_gen)


def tier_nr_handoff(self, bsz):
    """EXL3_NORM_ROUTE: the slot pointers and flags of this layer's decode handoff, for the post-attention norm +
    router kernel to write and release itself (tier_submit then skips its handoff); None if this layer has none"""
    if self._nortx or bsz > 8 or not hasattr(self, "_gslot"):
        return None
    if not _CPU.started and _far(self):
        _cpu_start()
    cpu = _pre_handoff(self, bsz)
    meta, lay, f2, g2 = cpu if cpu is not None else (0, 0, 0, 0)
    v = rg.slot_view(self)
    self._nr_handed = bsz
    ho = (v["y"].data_ptr(), v["s"].data_ptr(), v["w"].data_ptr(), v["issue"], 1, meta, lay, f2, g2, v["done"])
    return ho + (_T.xf.data_ptr(),) if MOE4_XF else ho


def _sh_i8(sh):
    """EXL3_TIER_SH_MOE4: (gate_w, up_w, down_w, gate_s, up_s, down_s) device pointers of the shared expert's int8 copies
    (dense GEMV fragment order, 4 strips per group), or None when its linears do not all have one"""
    out = []
    for names in (("gates", "gate", "gate_proj"), ("ups", "up", "up_proj"), ("downs", "down", "down_proj")):
        mod = next((getattr(sh, nm) for nm in names if getattr(sh, nm, None) is not None), None)
        if isinstance(mod, (list, tuple)):           # GatedMLP keeps one Linear per slice
            mod = mod[0] if len(mod) == 1 else None
        w8 = getattr(getattr(mod, "inner", None), "w8", None)
        if w8 is None or w8[4] != 4:
            print(f" -- tier: shared expert not in moe4 (no int8 copy for {names[0]}: {type(mod).__name__}; "
                  f"attributes {[a for a in vars(sh) if not a.startswith('_')][:12]})", flush=True)
            return None
        out.append(w8)
    (gq, gs), (uq, us), (dq, ds) = [(w[0], w[1]) for w in out]
    global _SH_SAID
    if not _SH_SAID:
        _SH_SAID = True
        print(f" -- tier: shared expert inside moe4 (int8 gate / up / down, {gs.numel()} x {uq.numel() // gs.numel()})", flush=True)
    return (gq.data_ptr(), uq.data_ptr(), dq.data_ptr(), gs.data_ptr(), us.data_ptr(), ds.data_ptr())


_SH_SAID = False


def tier_submit(self, y, bsz, selected_experts, routing_weights):
    L = self._L
    xf = False
    if not _CPU.started and _far(self):
        _cpu_start()                                  # before the first handoff of any size (the combine waits CPU_DONE)
    if self._nortx:
        pend = ("nortx", 0, bsz)
    elif bsz <= 8 and getattr(self, "_nr_handed", 0) == bsz:
        self._nr_handed = 0                           # the post-attention norm + router kernel did the handoff
        pend = ("graph", 0, bsz)
        xf = MOE4_XF                                  # ... and wrote the fragment-order input (EXL3_MOE4_XF)
    elif bsz <= 8:
        cpu = _pre_handoff(self, bsz)
        if HSIDE:
            hs = _T.h_stream
            hy, hsel, hrw = y.contiguous(), selected_experts.contiguous(), routing_weights.contiguous()
            hs.wait_stream(torch.cuda.current_stream(hy.device))
            with torch.cuda.stream(hs):
                rg.handoff(self, hy, hsel, hrw, bsz, moe4, cpu)
            self._h_join = (hy, hsel, hrw)            # kept alive until the join (no allocator reuse under the copy)
        else:
            rg.handoff(self, y.contiguous(), selected_experts.contiguous(), routing_weights.contiguous(), bsz, moe4, cpu)
        pend = ("graph", 0, bsz)
    elif moe3t is not None and bsz > rs.GB_ROWS and getattr(self._rtx, "htabs", None) is not None:
        pend = ("gb300", 0, bsz)                      # long prefill chunk: every expert on the GB300, the RTX idles
    else:
        _, pend = rs.rtx_submit(self, y, bsz, selected_experts, routing_weights)    # prefill: per-layer RTX path
    T, Hh = _T, self._hot
    sh = self.shared_experts
    y = y.contiguous().view(-1, y.shape[-1])
    if bsz <= 8:
        yh = T.y[:bsz]                                # left zeroed by the previous layer's combine
        shp = ()
        if getattr(self, "_sh_fast", False) and sh is not None:
            if SH_MOE4:
                if not hasattr(self, "_sh_ptrs"):
                    self._sh_ptrs = _sh_i8(sh)
                shp = self._sh_ptrs or ()
            if not shp:
                sh.bc.run_bszN(y.view(1, bsz, -1), yh.view(1, bsz, -1))   # shared expert into the accumulator (overwrite)
            self.shared_experts = None                # restored by the forward wrapper
        if xf:
            moe4.forward(T.xf[:bsz], selected_experts.view(bsz, -1).contiguous(), routing_weights.view(bsz, -1).contiguous(),
                         Hh.hot if Hh.far else Hh.n, Hh.tab["gate_w"], Hh.tab["up_w"], Hh.tab["down_w"],
                         Hh.tab["gate_s"], Hh.tab["up_s"], Hh.tab["down_s"], *T.scr, yh, T.blocks, T.blocks_a, 0, *shp, xf=1)
        else:
            moe4.forward(y, selected_experts.view(bsz, -1).contiguous(), routing_weights.view(bsz, -1).contiguous(),
                         Hh.hot if Hh.far else Hh.n, Hh.tab["gate_w"], Hh.tab["up_w"], Hh.tab["down_w"],
                         Hh.tab["gate_s"], Hh.tab["up_s"], Hh.tab["down_s"], *T.scr, yh, T.blocks, T.blocks_a, 0, *shp)
        hot = ("main", bsz)
    elif pend[0] == "gb300":
        hot = ("pre", _prefill_gb300(self, y, selected_experts.view(bsz, -1), routing_weights.view(bsz, -1)))
    else:
        hot = ("pre", _hot_prefill(self, y, selected_experts.view(bsz, -1), routing_weights.view(bsz, -1)))
    return None, (pend, hot)


def tier_combine(self, final_hidden_states, cpu_partial, cpu_pending, shape):
    if not (isinstance(cpu_pending, tuple) and len(cpu_pending) == 2 and isinstance(cpu_pending[1], tuple)):
        return final_hidden_states
    pend, hot = cpu_pending
    dev, rows = final_hidden_states.device, pend[2]
    f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
    if pend[0] == "nortx":                         # MTP layer: hot band only
        if hot[0] == "main" and final_hidden_states.is_contiguous():
            moe4.combine(f2, 0, _T.y, rows, 0, 0)
        else:
            f2 += _T.y[:rows] if hot[0] == "main" else hot[1]
            if hot[0] == "main": _T.y[:rows].zero_()
        return final_hidden_states
    if pend[0] == "gb300":                         # both bands are in the prefill output
        f2 += hot[1]
        return final_hidden_states
    if pend[0] == "graph":                         # one kernel: spin on DONE_L (+ CPU_DONE), add both partials, zero T.y
        if getattr(self, "_h_join", None) is not None:   # the side-stream handoff (EXL3_TIER_HSIDE)
            torch.cuda.current_stream(dev).wait_stream(_T.h_stream)
            self._h_join = None
        o_ptr, done = rg.wait_done(self, dev)
        prm = getattr(self, "_prm", None)
        if prm is not None and prm.get("mlp_defer_ok") and f2.dtype == torch.float and f2.is_contiguous() and \
                self.latent_out is None and not self.routed_post_norm and not self.tp_reduce and self.shared_experts is None:
            if CN_PDL:     # f2 is not zero-filled (tier_nofill): the fused kernel ignores it, the fallback zeroes it
                cw = _cpu_wait_args(self, rows)
                def can(norm, x, _r=rows):
                    w = getattr(norm, "weight", None)
                    return (w is not None and getattr(norm, "groups", 1) == 1 and not getattr(norm, "span_heads", False) and
                            w.numel() == H_ and w.dtype in (torch.half, torch.bfloat16) and w.is_contiguous() and
                            x.dtype == torch.float and x.is_contiguous() and x.numel() == _r * H_)
                def run(_f=f2, _o=o_ptr, _r=rows, _cw=cw, _d=done, _dev=dev):
                    if _cw[1]:
                        rg._fk.wait_reset(_cw[1], _cw[2], 0, _dev.index if _dev.index is not None else 0)
                    _f.zero_()
                    moe4.combine(_f, _o, _T.y, _r, _cw[0], _d)
                def fused(x, norm, _o=o_ptr, _r=rows, _cw=cw, _d=done):
                    xn = torch.empty((_r, H_), dtype=torch.half, device=x.device)
                    cn.combine_norm_pdl(0, _o, _T.y, _r, _cw[0], _d, _cw[1], _cw[2], _T.cctr, x.view(_r, H_), norm.weight.data,
                                        xn, float(norm.rms_norm_eps), float(norm.constant_bias), float(norm.constant_scale), 1)
                    return xn.view(x.shape)
                prm["resid_pending_combine"] = (can, run, fused)
                return final_hidden_states
            o2 = _cpu_partial(self, dev, rows)
            # EXL3_TIER_COMBINE_NORM: the next block's input norm runs this combine (runtime/patched/transformer.py)
            def can(norm, x, _r=rows):
                w = getattr(norm, "weight", None)
                return (w is not None and getattr(norm, "groups", 1) == 1 and not getattr(norm, "span_heads", False) and
                        w.numel() == H_ and w.dtype in (torch.half, torch.bfloat16) and w.is_contiguous() and
                        x.dtype == torch.float and x.is_contiguous() and x.numel() == _r * H_)
            def run(_f=f2, _o=o_ptr, _r=rows, _o2=o2, _d=done):
                moe4.combine(_f, _o, _T.y, _r, _o2, _d)
            def fused(x, norm, _f=f2, _o=o_ptr, _r=rows, _o2=o2, _d=done):
                xn = torch.empty((_r, H_), dtype=torch.half, device=_f.device)
                cn.combine_norm(_f, _o, _T.y, _r, _o2, _d, x.view(_r, H_), norm.weight.data, xn, float(norm.rms_norm_eps),
                                float(norm.constant_bias), float(norm.constant_scale))
                return xn.view(x.shape)
            prm["resid_pending_combine"] = (can, run, fused)
            return final_hidden_states
        o2 = _cpu_partial(self, dev, rows)
        if CN_PDL:
            f2.zero_()
        moe4.combine(f2, o_ptr, _T.y, rows, o2, done)
        return final_hidden_states
    final_hidden_states = rs.rtx_combine(self, final_hidden_states, cpu_partial, pend, shape)
    f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
    f2 += hot[1]
    return final_hidden_states


bsm.BlockSparseMLP.cpu_maybe_split_load = tier_split_load

# keep the (unquantized) token embedding in HBM: on the CPU every MTP draft pass would move its argmax ids to the host,
# a blocking sync that drains the host's run-ahead (+1.9 GB HBM)
if os.environ.get("EXL3_EMBED_GPU", "1") == "1":
    import exllamav3.modules.embedding as _emb
    _ei = _emb.Embedding.__init__
    def _emb_init(self, *a, **k):
        _ei(self, *a, **k)
        if self.caps.get("prefer_cpu"):
            self.caps.update({"prefer_cpu": False, "x_cpu": False})
    _emb.Embedding.__init__ = _emb_init
