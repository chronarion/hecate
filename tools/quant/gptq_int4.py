"""Reference-guided requant of the int4 expert bands of the served GLM-5.3 stack + the stock KLD reference, in one BF16
streaming pass (stock zai-org/GLM-5.3-BF16; same module loop as kld_ref.py).

Per module, two state streams run through the BF16 weights:
  A = the frozen KLD corpus (corpus.json): reference logits at the head (REF/{wid}.safetensors) + route traces.
  Q = the calibration corpus (calib.json, disjoint), propagated through the *quantized* experts: every MoE layer is
      calibrated on the states the int4 model actually produces up to that layer (causal correction).
Per MoE layer L (experts of the int4 bands: placement perm[:144] hot -> layer{L}.safetensors, perm[144:171] cold ->
layer{L}.r229.safetensors, same format as requant_u4.py: uniform int4 per-row MSE-best scale, moe4 fragment order):
  1. capture the MoE input of every Q token and its routing (the layer's MoE is skipped in this pass)
  2. per expert H = X^T X over the tokens routed to it; batched act-order GPTQ of gate and up (shared H)
  3. down calibrated on the reconstructed (quantized) gate/up output: H_down = A^T A, A = silu(X Gq^T) * (X Uq^T); GPTQ
  4. write the files; copy the dequantized int4 weights into the module; run the layer on Q -> next layer's Q states
   python -I kld_gptq.py --a /kbf16 --corpus C --calib Q --ref REF --out OUT --stage STAGE --u4out U4 --place P [--smoke]"""
import sys, os, json, time, argparse
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from exllamav3 import Config, Model
from exllamav3.util.memory import free_mem

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
P = argparse.ArgumentParser()
for a in ("a", "corpus", "calib", "ref", "out", "stage", "u4out", "place", "check_u4"):
    P.add_argument("--" + a, default="")
P.add_argument("--smoke", action="store_true")
P.add_argument("--damp", type=float, default=0.01)
P.add_argument("--windows", default=""); P.add_argument("--calib_n", type=int, default=0)
args = P.parse_args()
for d in (args.ref, args.out, args.u4out): os.makedirs(d, exist_ok=True)
dev = torch.device("cuda:0")
W = json.load(open(args.corpus))
if args.windows: W = [w for w in W if w["id"] in args.windows.split(",")]
CQ = json.load(open(args.calib))
if args.calib_n: CQ = CQ[:args.calib_n]
PLACE = json.load(open(args.place))
U4F = torch.tensor((2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.7, 4.0), device=dev)
NHOT, NBAND = 144, 171

cfg = Config.from_directory(args.a)
model = Model.from_config(cfg)

# routing hook: "ref" records the KLD windows' routes, "cap" records MoE input + routes and aborts the MoE
MODE = {"m": "off"}; CAP = []; ROUTES = {}
class _Captured(Exception): pass
for mod in model.modules:
    for sub in getattr(mod, "modules", []):
        if type(sub).__name__ == "BlockSparseMLP":
            f0 = sub.routing_fn
            def f(bsz, rcfg, z, params, f0=f0, key=sub.key):
                s, w = f0(bsz, rcfg, z, params)
                if MODE["m"] == "cap":
                    CAP.append((z.view(-1, z.shape[-1]).half().clone(), s.view(-1, s.shape[-1]).clone()))
                    raise _Captured()
                if MODE["m"] == "ref":
                    CAP.append((s.to(torch.int16).cpu(), w.half().cpu()))
                return s, w
            sub.routing_fn = f


def scales_mse(Wt):
    """per-row MSE-best scale over U4F (requant_u4.quant_u4), batched: Wt [..., N, K] -> [..., N]"""
    s0 = Wt.pow(2).mean(-1).sqrt().clamp_min(1e-12)
    best_e = best_s = None
    for f in U4F:
        s = s0 / f
        e = (Wt - torch.round(Wt / s[..., None]).clamp_(-8, 7) * s[..., None]).pow_(2).sum(-1)
        if best_e is None: best_e, best_s = e, s
        else: m = e < best_e; best_e = torch.where(m, e, best_e); best_s = torch.where(m, s, best_s)
    return best_s


def pack_u4(codes, SPW=4):
    N, K = codes.shape
    c = codes.view(N // (16 * SPW), SPW, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 8, 6, 2).contiguous().view(-1, 2)
    return (c[:, 0] | (c[:, 1] << 4)).contiguous()


def gptq(Wt, H, s, damp):
    """batched act-order GPTQ, uniform int4 (q in -8..7) with fixed per-row scale s.
    Wt [E, N, K] fp32 (modified), H [E, K, K] fp32 (consumed), s [E, N] -> int8 codes [E, N, K]"""
    E, N, K = Wt.shape
    d = H.diagonal(dim1=1, dim2=2)
    dead = d <= 0
    if dead.any():
        H.diagonal(dim1=1, dim2=2)[dead] = 1.0
        Wt.masked_fill_(dead[:, None, :], 0.0)
    perm = torch.argsort(H.diagonal(dim1=1, dim2=2), dim=1, descending=True)
    inv = torch.argsort(perm, dim=1)
    Wt = Wt.gather(2, perm[:, None, :].expand(E, N, K))
    H = H.gather(1, perm[:, :, None].expand(E, K, K)).gather(2, perm[:, None, :].expand(E, K, K))
    H.diagonal(dim1=1, dim2=2).add_(damp * H.diagonal(dim1=1, dim2=2).mean(1, keepdim=True))
    L = torch.linalg.cholesky(H); del H
    Hi = torch.cholesky_inverse(L); del L
    Hi = torch.linalg.cholesky(Hi, upper=True)
    Q = torch.empty((E, N, K), dtype=torch.int8, device=Wt.device)
    inv_s = 1.0 / s
    B = 128
    for b0 in range(0, K, B):
        b1 = min(b0 + B, K)
        W1 = Wt[:, :, b0:b1].clone(); Err = torch.empty_like(W1); Hb = Hi[:, b0:b1, b0:b1]
        for i in range(b1 - b0):
            w = W1[:, :, i]
            q = torch.round(w * inv_s).clamp_(-8, 7)
            Q[:, :, b0 + i] = q.to(torch.int8)
            e = (w - q * s) / Hb[:, i, i][:, None]
            W1[:, :, i:] -= e[:, :, None] * Hb[:, i, i:][:, None, :]
            Err[:, :, i] = e
        if b1 < K:
            Wt[:, :, b1:] -= torch.bmm(Err, Hi[:, b0:b1, b1:])
    del Hi
    return Q.gather(2, inv[:, None, :].expand(E, N, K))


def hessians(Z, SEL, experts, K):
    H = torch.zeros((len(experts), K, K), dtype=torch.float, device=dev); n = []
    for i, e in enumerate(experts):
        X = Z[(SEL == e).any(-1)].float()
        n.append(X.shape[0])
        if X.shape[0]: torch.matmul(X.T, X, out=H[i])
    return H, n


def proxy_err(Wt, Wq, H):
    """sum over experts of tr(dW H dW^T) / tr(W H W^T): relative output error on the calibration tokens"""
    num = den = 0.0
    for i in range(Wt.shape[0]):
        dW = Wt[i] - Wq[i]
        num += ((dW @ H[i]) * dW).sum().item(); den += ((Wt[i] @ H[i]) * Wt[i]).sum().item()
    return (num / max(den, 1e-30)) ** 0.5


def expert_weights(mlp, experts, which):
    out = []
    for e in experts:
        lin = getattr(mlp, which)[e].inner; w = lin.weight
        out.append(w.T.float() if w.shape[0] != lin.out_features else w.float())   # [N_out, K_in]
    return torch.stack(out)


def set_expert_weights(mlp, experts, which, Wq):
    for i, e in enumerate(experts):
        lin = getattr(mlp, which)[e].inner; w = lin.weight
        w.copy_(Wq[i].T if w.shape[0] != lin.out_features else Wq[i])


def requant_layer(mlp, L, Z, SEL, EV, CH=64, RHO=0.3, NMIN=4096):
    t0 = time.time(); experts = PLACE[str(L)]["perm"][:NBAND]
    tr = ~EV; Ztr = Z[tr]; SELtr = SEL[tr]; Zev = Z[EV]; SELev = SEL[EV]
    g = torch.Generator(device=dev).manual_seed(L)
    # shrinkage target: the layer-wide input covariance over every training token (experts all read the same residual
    # stream; an expert's own covariance from ~1-3 tokens per input dimension is too noisy for GPTQ's H^-1)
    Sig = torch.zeros((Z.shape[1], Z.shape[1]), dtype=torch.float, device=dev)
    for c0 in range(0, Ztr.shape[0], 65536):
        zc = Ztr[c0:c0 + 65536].float(); Sig.addmm_(zc.T, zc)
    Sig /= Ztr.shape[0]
    Zp = Ztr[torch.randperm(Ztr.shape[0], device=dev, generator=g)[:8192]].float()   # sample for the down prior
    acc = {"gu_rtn": [0.0, 0.0], "gu_gptq": [0.0, 0.0], "down_rtn": [0.0, 0.0], "down_gptq": [0.0, 0.0]}
    def perr(key, Wt, Wq, H):
        for i in range(Wt.shape[0]):
            dW = Wt[i] - Wq[i]
            acc[key][0] += ((dW @ H[i]) * dW).sum().item(); acc[key][1] += ((Wt[i] @ H[i]) * Wt[i]).sum().item()
    Qgu, sgu, Qd, sd, n = [], [], [], [], []
    for c0 in range(0, len(experts), CH):
        ex = experts[c0:c0 + CH]
        Hg, nc = hessians(Ztr, SELtr, ex, Z.shape[1]); n += nc
        for i in range(len(ex)):
            Hg[i].mul_(1 - RHO).add_(Sig, alpha=RHO * max(nc[i], NMIN))
        Hge, _ = hessians(Zev, SELev, ex, Z.shape[1])
        Wgu = torch.cat([expert_weights(mlp, ex, "gates"), expert_weights(mlp, ex, "ups")], 1)
        s_ = scales_mse(Wgu)
        perr("gu_rtn", Wgu, torch.round(Wgu / s_[..., None]).clamp_(-8, 7) * s_[..., None], Hge)
        q_ = gptq(Wgu.clone(), Hg.clone(), s_, args.damp)
        Wq = q_.float() * s_[..., None]
        perr("gu_gptq", Wgu, Wq, Hge); del Hg, Hge
        I = Wgu.shape[1] // 2
        if c0 == 0 and args.check_u4 and os.path.exists(f"{args.check_u4}/layer{L}.safetensors"):
            old = load_file(f"{args.check_u4}/layer{L}.safetensors"); e0 = ex[0]
            qr = torch.round(Wgu[0, :I] / s_[0, :I, None]).clamp_(-8, 7)
            acc["pack_check"] = bool(torch.equal(pack_u4((qr.to(torch.int16) + 8).to(torch.uint8)).cpu(), old[f"{e0}.gate.q"])) and \
                bool(torch.allclose(s_[0, :I].cpu(), old[f"{e0}.gate.s"], rtol=1e-5))
        del Wgu
        # down: Hessian of the reconstructed (quantized) gate/up output
        Hd = torch.zeros((len(ex), I, I), dtype=torch.float, device=dev); Hde = torch.zeros_like(Hd)
        for i, e in enumerate(ex):
            act = lambda X: F.silu(X @ Wq[i, :I].T) * (X @ Wq[i, I:].T)
            X = Ztr[(SELtr == e).any(-1)].float()
            if X.shape[0]:
                A = act(X); torch.matmul(A.T, A, out=Hd[i])
            A = act(Zp); Hd[i].mul_(1 - RHO).add_(A.T @ A, alpha=RHO * max(n[c0 + i], NMIN) / Zp.shape[0])
            X = Zev[(SELev == e).any(-1)].float()
            if X.shape[0]:
                A = act(X); torch.matmul(A.T, A, out=Hde[i])
        Wd = expert_weights(mlp, ex, "downs"); sd_ = scales_mse(Wd)
        perr("down_rtn", Wd, torch.round(Wd / sd_[..., None]).clamp_(-8, 7) * sd_[..., None], Hde)
        qd_ = gptq(Wd.clone(), Hd.clone(), sd_, args.damp)
        Wdq = qd_.float() * sd_[..., None]
        perr("down_gptq", Wd, Wdq, Hde); del Hd, Hde, Wd
        set_expert_weights(mlp, ex, "gates", Wq[:, :I].half()); set_expert_weights(mlp, ex, "ups", Wq[:, I:].half())
        set_expert_weights(mlp, ex, "downs", Wdq.half()); del Wq, Wdq
        Qgu.append(q_); sgu.append(s_); Qd.append(qd_); sd.append(sd_)
    Qgu = torch.cat(Qgu); sgu = torch.cat(sgu); Qd = torch.cat(Qd); sd = torch.cat(sd); del Sig, Zp
    # files (same layout as requant_u4.py) and the module copy for the Q propagation
    hot, cold = {}, {}
    for i, e in enumerate(experts):
        dst = hot if i < NHOT else cold
        for m, q, s in (("gate", Qgu[i, :I], sgu[i, :I]), ("up", Qgu[i, I:], sgu[i, I:]), ("down", Qd[i], sd[i])):
            dst[f"{e}.{m}.q"] = pack_u4((q.to(torch.int16) + 8).to(torch.uint8)).cpu()
            dst[f"{e}.{m}.s"] = s.float().cpu().contiguous()
    for path, t in ((f"{args.u4out}/layer{L}.safetensors", hot), (f"{args.u4out}/layer{L}.r229.safetensors", cold)):
        save_file(t, path + ".tmp"); os.replace(path + ".tmp", path)
    nn = torch.tensor(n, dtype=torch.float)
    r = {"layer": L, "tokens_min": int(nn.min()), "tokens_median": int(nn.median()), "tokens_hot_min": int(nn[:NHOT].min()), "s": time.time() - t0}
    for k_ in ("gu_rtn", "gu_gptq", "down_rtn", "down_gptq"):
        r[k_] = (acc[k_][0] / max(acc[k_][1], 1e-30)) ** 0.5
    if "pack_check" in acc: r["pack_check"] = acc["pack_check"]
    return r


def wait_stage(i):
    if not args.stage: return
    t0 = time.time()
    while not os.path.exists(f"{args.stage}/ready.{i}"):
        time.sleep(0.2)
        if time.time() - t0 > 2400: raise RuntimeError(f"stage timeout module {i}")


ids_a = [torch.tensor(w["ids"], dtype=torch.long).view(1, -1) for w in W]
ids_q = [torch.tensor(w["ids"], dtype=torch.long).view(1, -1) for w in CQ]
pa = [{} for _ in W]; pq = [{} for _ in CQ]
sa = [model.prepare_inputs(x, p) for x, p in zip(ids_a, pa)]
sq = [model.prepare_inputs(x, p) for x, p in zip(ids_q, pq)]
LJ = open(f"{args.out}/gptq.jsonl", "a")
mods = list(enumerate(model.modules))
if args.smoke: mods = mods[:5] + mods[-2:]
T0 = time.time()
for idx, ma in mods:
    last = ma.key == "lm_head"
    wait_stage(idx); t0 = time.time()
    cfg.stc.begin_deferred_load(); ma.load(dev if not ma.caps.get("prefer_cpu") else "cpu"); cfg.stc.end_deferred_load()
    mlp = next((s for s in getattr(ma, "modules", []) if type(s).__name__ == "BlockSparseMLP"), None)
    # A: reference
    MODE["m"] = "ref" if mlp is not None else "off"
    for j, w in enumerate(W):
        pa[j]["dev_cache"] = None
        x = ma.forward(ma.prepare_for_device(sa[j], pa[j]), pa[j]).clone()
        if mlp is not None:
            ROUTES[f"{mlp.key}.{w['id']}.sel"] = torch.cat([c[0] for c in CAP]); ROUTES[f"{mlp.key}.{w['id']}.w"] = torch.cat([c[1] for c in CAP]); CAP.clear()
        if not last:
            sa[j] = x
        else:
            s0 = w["score_from"]; T = len(w["ids"])
            save_file({"logits": x[0, s0:T - 1].float().cpu().contiguous(), "target_ids": torch.tensor(w["ids"][s0 + 1:T], dtype=torch.long),
                       "positions": torch.arange(s0, T - 1, dtype=torch.int32)}, f"{args.ref}/{w['id']}.safetensors")
        del x
    rec = {"module": ma.key, "idx": idx}
    # Q: calibration stream
    if mlp is not None:
        L = int(ma.key.split(".")[2])
        MODE["m"] = "cap"; ev = []
        for j in range(len(CQ)):
            pq[j]["dev_cache"] = None
            n0 = len(CAP)
            try:
                ma.forward(ma.prepare_for_device(sq[j], pq[j]), pq[j])
            except _Captured:
                pass
            ev += [j % 10 == 9] * sum(c[0].shape[0] for c in CAP[n0:])
        MODE["m"] = "off"
        Z = torch.cat([c[0] for c in CAP]); SEL = torch.cat([c[1] for c in CAP]); CAP.clear()
        EV = torch.tensor(ev, dtype=torch.bool, device=dev)
        rec.update(requant_layer(mlp, L, Z, SEL, EV)); del Z, SEL, EV
        free_mem()
    if not last:
        for j in range(len(CQ)):
            pq[j]["dev_cache"] = None
            sq[j] = ma.forward(ma.prepare_for_device(sq[j], pq[j]), pq[j]).clone()
    ma.unload(); cfg.stc.close(); free_mem()
    if args.stage: open(f"{args.stage}/done.{idx}", "w").close()
    rec["s_total"] = time.time() - t0
    LJ.write(json.dumps(rec) + "\n"); LJ.flush()
    extra = ""
    if "gu_rtn" in rec:
        extra = (f" | tokens/expert min {rec['tokens_min']} med {rec['tokens_median']} | out err gate+up RTN {rec['gu_rtn']:.4f} -> GPTQ "
                 f"{rec['gu_gptq']:.4f}, down {rec['down_rtn']:.4f} -> {rec['down_gptq']:.4f} ({rec['s']:.0f} s)"
                 + (f" pack_check {rec['pack_check']}" if "pack_check" in rec else ""))
    print(f"{time.strftime('%H:%M:%S')} {idx:2d} {ma.key:20s} {rec['s_total']:5.1f} s{extra}", flush=True)

save_file(ROUTES, f"{args.out}/routes_a.safetensors")
print(f"gptq pass done in {(time.time() - T0) / 60:.1f} min", flush=True)
