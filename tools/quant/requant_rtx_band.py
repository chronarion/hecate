"""Requantize the RTX band of GLM-5.3 (85 middle-popularity routed experts per MoE layer, placement perm[171:256]) as 3-bit
EXL3 trellis from stock zai-org/GLM-5.3-BF16, so the whole served model is stock lineage.

One streaming pass over the BF16 model (stage.py fetches module i, this script loads it after STAGE/ready.i):
  - a calibration stream (calib2.json, the GPTQ v2 corpus) runs through the model as it is served: per MoE layer the int4
    bands (hot 144 + cold 27) are set to their dequantized GPTQ v2 weights, the RTX band to its new trellis weights, so
    each layer is calibrated on the states the served model actually produces (causal correction, as in kld_gptq.py);
  - per RTX-band expert: H = X^T X over its routed tokens, shrunk toward the layer-wide input covariance (rho 0.3, prior
    weight max(n, 4096)); gate and up quantized with that H, down with the Hessian of the *quantized* gate/up output;
    ExLlamaV3's own quantizer (quantize_exl3_batch: LDLQ trellis, K 3, mul1 codebook, output scales always), i.e. the
    same format and settings as the published 3.0 bpw EXL3 quant, only calibrated on stock;
  - output: OUT/layer{L}.safetensors with model.layers.L.mlp.experts.E.{gate,up,down}_proj.{trellis,suh,svh,mul1}.
   python -I requant_rtx_band.py --a /kbf16 --calib CALIB --stage STAGE --place PLACE --u4 U4 --out OUT [--smoke]"""
import sys, os, json, time, argparse
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from exllamav3 import Config, Model
from exllamav3.util.memory import free_mem
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
P = argparse.ArgumentParser()
for a in ("a", "calib", "stage", "place", "u4", "out"):
    P.add_argument("--" + a, default="")
P.add_argument("--smoke", action="store_true")
P.add_argument("--layers", default="")            # restrict the requant to these MoE layers (the stream still runs all)
P.add_argument("--calib_n", type=int, default=0)
P.add_argument("--chunk", type=int, default=16)   # experts per quantizer batch
args = P.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda:0")
CQ = json.load(open(args.calib))
if args.calib_n: CQ = CQ[:args.calib_n]
PLACE = json.load(open(args.place))
NBAND = 171                                       # perm[:171] int4 (hot 144 + cold 27), perm[171:] the RTX band
ONLY = {int(x) for x in args.layers.split(",")} if args.layers else None
RHO, NMIN = 0.3, 4096

cfg = Config.from_directory(args.a)
model = Model.from_config(cfg)

MODE = {"m": "off"}; CAP = []
class _Captured(Exception): pass
for mod in model.modules:
    for sub in getattr(mod, "modules", []):
        if type(sub).__name__ == "BlockSparseMLP":
            f0 = sub.routing_fn
            def f(bsz, rcfg, z, params, f0=f0):
                s, w = f0(bsz, rcfg, z, params)
                if MODE["m"] == "cap":
                    CAP.append((z.view(-1, z.shape[-1]).half().clone(), s.view(-1, s.shape[-1]).clone()))
                    raise _Captured()
                return s, w
            sub.routing_fn = f


def unpack_u4(q, s, N, K, SPW=4):
    """inverse of kld_gptq.pack_u4: moe4 fragment-order int4 codes + per-row scale -> fp32 [N, K]"""
    lo = (q & 15).to(torch.int16); hi = (q >> 4).to(torch.int16)
    # packed order (pack_u4's permute): (N/16SPW, K/64, SPW, 8, 4, 4, 2, 2) bytes, the nibble pair last
    c = torch.stack([lo, hi], -1).view(N // (16 * SPW), K // 64, SPW, 8, 4, 4, 2, 2, 2)
    c = c.permute(0, 2, 8, 3, 1, 5, 7, 4, 6).contiguous().view(N, K)
    return (c.float() - 8.0) * s.float()[:, None]


def set_expert(mlp, e, which, Wq):
    """Wq [N_out, K_in] -> the module's weight (the BF16 module keeps [K, N] or [N, K])"""
    lin = getattr(mlp, which)[e].inner; w = lin.weight
    w.copy_(Wq.T if w.shape[0] != lin.out_features else Wq)


def expert_w(mlp, e, which):
    lin = getattr(mlp, which)[e].inner; w = lin.weight
    return (w.T if w.shape[0] != lin.out_features else w).float()   # [N_out, K_in]


def h_data(H, n, key):
    return {"H": H, "count": max(n, 1), "finalized": False, "device": dev, "first_key": key}


def qargs(seed):
    return {"K": 3, "seed": seed, "devices": [dev.index or 0], "apply_out_scales": True, "mul1": True, "sigma_reg": 0.025}


def load_int4(mlp, L, experts):
    """GPTQ v2 int4 weights (as served) into the module for the propagation"""
    f = {}
    for fn in (f"{args.u4}/layer{L}.safetensors", f"{args.u4}/layer{L}.r229.safetensors"):
        f.update(load_file(fn, device="cuda:0"))
    I, H = mlp.ups[0].inner.out_features, mlp.ups[0].inner.in_features
    for e in experts:
        set_expert(mlp, e, "gates", unpack_u4(f[f"{e}.gate.q"], f[f"{e}.gate.s"], I, H).half())
        set_expert(mlp, e, "ups", unpack_u4(f[f"{e}.up.q"], f[f"{e}.up.s"], I, H).half())
        set_expert(mlp, e, "downs", unpack_u4(f[f"{e}.down.q"], f[f"{e}.down.s"], H, I).half())


def requant_band(mlp, L, Z, SEL):
    t0 = time.time(); perm = PLACE[str(L)]["perm"]; band = perm[NBAND:]
    g = torch.Generator(device=dev).manual_seed(L)
    Sig = torch.zeros((Z.shape[1], Z.shape[1]), dtype=torch.float, device=dev)
    for c0 in range(0, Z.shape[0], 65536):
        zc = Z[c0:c0 + 65536].float(); Sig.addmm_(zc.T, zc)
    Sig /= Z.shape[0]
    Zp = Z[torch.randperm(Z.shape[0], device=dev, generator=g)[:8192]].float()
    out, errs, ns = {}, [], []
    for c0 in range(0, len(band), args.chunk):
        ex = band[c0:c0 + args.chunk]
        Hs, nc = [], []
        for e in ex:
            X = Z[(SEL == e).any(-1)].float(); n = X.shape[0]; nc.append(n)
            H = X.T @ X if n else torch.zeros_like(Sig)
            Hs.append(H.mul_(1 - RHO).add_(Sig, alpha=RHO * max(n, NMIN)))
        ns += nc
        # gate + up: one Hessian per expert, shared by its two projections
        Ws, HD, QA = [], [], []
        for i, e in enumerate(ex):
            hd = h_data(Hs[i], nc[i], f"{mlp.key}.experts.{e}")
            for which in ("gates", "ups"):
                Ws.append(expert_w(mlp, e, which).T.contiguous())          # quantizer layout [in, out]
                HD.append(hd); QA.append(qargs(1000 * L + 2 * e + (which == "ups")))
        res = quantize_exl3_batch(Ws, HD, QA)
        Wq = {}
        for j, (perr, t) in enumerate(res):
            e = ex[j // 2]; which = ("gate_proj", "up_proj")[j % 2]
            for k_, v in t.items(): out[f"{mlp.key}.experts.{e}.{which}.{k_}"] = v.cpu()
            errs.append(perr)
        # the batch API returns only the packed tensors: reconstruct the stored weights the way the runtime does
        del Ws, HD, Hs
        for i, e in enumerate(ex):
            for which, mod in (("gate_proj", "gates"), ("up_proj", "ups")):
                Wq[(e, mod)] = dequant_exl3(out, f"{mlp.key}.experts.{e}.{which}", expert_w(mlp, e, mod).shape)
        # down: Hessian of the quantized gate/up output
        Ws, HD, QA = [], [], []
        for i, e in enumerate(ex):
            G, U = Wq[(e, "gates")], Wq[(e, "ups")]
            act = lambda X: F.silu(X @ G.T) * (X @ U.T)
            X = Z[(SEL == e).any(-1)].float()
            Hd = (lambda A: A.T @ A)(act(X)) if X.shape[0] else torch.zeros((G.shape[0], G.shape[0]), device=dev)
            A = act(Zp); Hd.mul_(1 - RHO).add_(A.T @ A, alpha=RHO * max(nc[i], NMIN) / Zp.shape[0])
            Ws.append(expert_w(mlp, e, "downs").T.contiguous()); HD.append(h_data(Hd, nc[i], f"{mlp.key}.experts.{e}.down"))
            QA.append(qargs(1000 * L + 600 + e))
        res = quantize_exl3_batch(Ws, HD, QA)
        for j, (perr, t) in enumerate(res):
            e = ex[j]
            for k_, v in t.items(): out[f"{mlp.key}.experts.{e}.down_proj.{k_}"] = v.cpu()
            errs.append(perr)
        del Ws, HD
        # module copy for the propagation: the band as stored
        for e in ex:
            set_expert(mlp, e, "gates", Wq[(e, "gates")].half()); set_expert(mlp, e, "ups", Wq[(e, "ups")].half())
            set_expert(mlp, e, "downs", dequant_exl3(out, f"{mlp.key}.experts.{e}.down_proj", expert_w(mlp, e, "downs").shape).half())
        del Wq
    path = f"{args.out}/layer{L}.safetensors"
    save_file(out, path + ".tmp"); os.replace(path + ".tmp", path)
    nn = torch.tensor(ns, dtype=torch.float); pe = torch.tensor(errs)
    return {"layer": L, "band": len(band), "tokens_min": int(nn.min()), "tokens_median": int(nn.median()),
            "proxy_err_mean": float(pe.mean()), "proxy_err_max": float(pe.max()), "s": time.time() - t0}


_DQ = {}
def dequant_exl3(t, key, shape_out_in):
    """the runtime's own reconstruction of a trellis linear (ExLlamaV3 LinearEXL3.get_weight_tensor) -> fp32 [out, in]"""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    N, K = shape_out_in
    lin = LinearEXL3(cfg, K, N, None, None, None, t[f"{key}.suh"].to(dev), t[f"{key}.svh"].to(dev), t[f"{key}.trellis"].to(dev),
                     t.get(f"{key}.mcg"), t.get(f"{key}.mul1", torch.tensor(0)).to(dev) if f"{key}.mul1" in t else None,
                     None, None)
    w = lin.get_weight_tensor().float()                                      # [in, out]
    return w.T.contiguous()


def wait_stage(i):
    if not args.stage: return
    t0 = time.time()
    while not os.path.exists(f"{args.stage}/ready.{i}"):
        time.sleep(0.2)
        if time.time() - t0 > 2400: raise RuntimeError(f"stage timeout module {i}")


ids_q = [torch.tensor(w["ids"], dtype=torch.long).view(1, -1) for w in CQ]
pq = [{} for _ in CQ]
sq = [model.prepare_inputs(x, p) for x, p in zip(ids_q, pq)]
LJ = open(f"{args.out}/requant.jsonl", "a")
mods = list(enumerate(model.modules))
if args.smoke: mods = mods[:5]             # embedding, dense layers 0-2, first MoE layer (the stage daemon fetches in order)
T0 = time.time()
for idx, ma in mods:
    last = ma.key == "lm_head"
    wait_stage(idx); t0 = time.time()
    cfg.stc.begin_deferred_load(); ma.load(dev if not ma.caps.get("prefer_cpu") else "cpu"); cfg.stc.end_deferred_load()
    mlp = next((s for s in getattr(ma, "modules", []) if type(s).__name__ == "BlockSparseMLP"), None)
    rec = {"module": ma.key, "idx": idx}
    if mlp is not None and str(int(ma.key.split(".")[2])) in PLACE and PLACE[str(int(ma.key.split(".")[2]))].get("first", 0) < 256:
        L = int(ma.key.split(".")[2]); perm = PLACE[str(L)]["perm"]
        load_int4(mlp, L, perm[:NBAND])
        done_path = f"{args.out}/layer{L}.safetensors"
        if os.path.exists(done_path):          # resume: this layer's band is already written; load it for the propagation
            t = load_file(done_path)
            for e in perm[NBAND:]:
                for which, mod in (("gate_proj", "gates"), ("up_proj", "ups"), ("down_proj", "downs")):
                    set_expert(mlp, e, mod, dequant_exl3(t, f"{mlp.key}.experts.{e}.{which}", expert_w(mlp, e, mod).shape).half())
            rec["resumed"] = True; del t
        elif ONLY is None or L in ONLY:
            MODE["m"] = "cap"
            for j in range(len(CQ)):
                pq[j]["dev_cache"] = None
                try:
                    ma.forward(ma.prepare_for_device(sq[j], pq[j]), pq[j])
                except _Captured:
                    pass
            MODE["m"] = "off"
            Z = torch.cat([c[0] for c in CAP]); SEL = torch.cat([c[1] for c in CAP]); CAP.clear()
            rec.update(requant_band(mlp, L, Z, SEL)); del Z, SEL
            free_mem()
    if not last:
        for j in range(len(CQ)):
            pq[j]["dev_cache"] = None
            sq[j] = ma.forward(ma.prepare_for_device(sq[j], pq[j]), pq[j]).clone()
    ma.unload(); cfg.stc.close(); free_mem()
    if args.stage: open(f"{args.stage}/done.{idx}", "w").close()
    rec["s_total"] = time.time() - t0
    LJ.write(json.dumps(rec) + "\n"); LJ.flush()
    extra = (f" | band {rec['band']} tokens/expert min {rec['tokens_min']} med {rec['tokens_median']} | proxy err mean "
             f"{rec['proxy_err_mean']:.4f} max {rec['proxy_err_max']:.4f} ({rec['s']:.0f} s)") if "band" in rec else ""
    print(f"{time.strftime('%H:%M:%S')} {idx:2d} {ma.key:20s} {rec['s_total']:6.1f} s{extra}", flush=True)
print(f"requant pass done in {(time.time() - T0) / 60:.1f} min", flush=True)
