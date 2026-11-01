"""Candidate pass of the GLM-5.3 KLD benchmark: a serving stack's deployed path, teacher-forced on the frozen corpus and
compared position by position with the reference logits written by kld_ref.py (KL(ref || candidate), full vocabulary,
fp32 log-softmax, fp64 sums). Same token IDs as the reference (corpus.json, sha256 per window checked).

KLD_MODE=decode (default): the context before a window's scored span goes through model.prefill in 2048-token chunks (the
  served chunk size, as a prompt would); every scored position comes from model.forward in KLD_R-row chunks against the FP16
  cache (R = 5 = the verify shape of MTP with 4 drafts; no drafting, so no acceptance effects).
KLD_MODE=prefill: one model.forward over the whole window (prompt-logit estimand of the lab's prefill workflow).
   python -I kld_cand.py MODEL_DIR REF_DIR CORPUS OUT
env: KLD_TIER=1 imports tier_split first (KLD_TIER_DIR, default /opt/hecate/tier: the served three-tier stack); KLD_SPLIT=240,0 (use_per_device);
     KLD_SAVE=wiki-0,chat-0 keeps fp32 candidate logits of those windows (runtime-repeat floor); KLD_WINDOWS=subset;
     KLD_ROUTES=routes_a.safetensors: B x Q cell (every MoE layer consumes the reference's selected experts and routing
     weights for the positions it processes instead of its own; the candidate's router still runs, its result is replaced)"""
import os, sys, json, time, hashlib, threading
if os.environ.get("KLD_TIER", "0") == "1":
    sys.path.insert(0, os.environ.get("KLD_TIER_DIR", "/opt/hecate/tier"))
    import tier_split  # noqa: F401
if os.environ.get("KLD_RUNTIME", "0") == "1":   # the served Hecate stack: folder loader + runtime (int8 base, graphs, router)
    sys.path.insert(0, os.environ.get("KLD_CORE_DIR", "/opt/hecate/core"))
    import hecate_loader  # noqa: F401
    import hecate_runtime  # noqa: F401
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3 import Config, Model, Cache

torch.set_grad_enabled(False)
mdir, refd, corpus, out = sys.argv[1:5]
MODE = os.environ.get("KLD_MODE", "decode"); R = int(os.environ.get("KLD_R", "5"))
SPLIT = [float(x) for x in os.environ.get("KLD_SPLIT", "240,0").split(",")]
SAVE = set(filter(None, os.environ.get("KLD_SAVE", "").split(",")))
os.makedirs(out + "/pos", exist_ok=True)
W = json.load(open(corpus))
if os.environ.get("KLD_WINDOWS"):
    W = [w for w in W if w["id"] in os.environ["KLD_WINDOWS"].split(",")]
for w in W:
    assert hashlib.sha256(json.dumps(w["ids"]).encode()).hexdigest() == w["sha256"], w["id"]
MAXT = max(len(w["ids"]) for w in W)

config = Config.from_directory(mdir)
model = Model.from_config(config)
CUR = {"wid": None, "p": 0, "calls": 0}
if os.environ.get("KLD_ROUTES"):
    from safetensors.torch import load_file
    RT = load_file(os.environ["KLD_ROUTES"]); nw = 0
    # the tier stack permutes each layer's experts (and its router) into placement order: map the reference's
    # original expert ids to that order (identity for the stock / published stacks)
    INV = {}
    if os.environ.get("EXL3_TIER_SPLIT"):
        for L, pl in json.load(open(os.environ["EXL3_TIER_SPLIT"])).items():
            INV[f"model.layers.{L}.mlp"] = torch.argsort(torch.tensor(pl["perm"]))
    for mod in model.modules:
        for sub in getattr(mod, "modules", []):
            if type(sub).__name__ == "BlockSparseMLP":
                f0 = sub.routing_fn
                def f(bsz, rcfg, z, params, f0=f0, key=sub.key):
                    s_, w_ = f0(bsz, rcfg, z, params)
                    if CUR["wid"] is None:          # load / warmup / graph capture: no window in flight
                        return s_, w_
                    a = CUR["p"]; rs_ = RT[f"{key}.{CUR['wid']}.sel"][a:a + bsz].long(); rw_ = RT[f"{key}.{CUR['wid']}.w"][a:a + bsz]
                    if key in INV: rs_ = INV[key][rs_]
                    assert rs_.shape[0] == bsz, (key, CUR, bsz)
                    CUR["calls"] += 1
                    s_.copy_(rs_.to(s_.device, non_blocking=True)); w_.copy_(rw_.to(w_.device, non_blocking=True))
                    return s_, w_
                sub.routing_fn = f; nw += 1
    print(f"B x Q: {nw} MoE layers replay the reference routes", flush=True)
KVQ = os.environ.get("KLD_CACHE", "")          # "k,v" bits: quantized KV cache as tabby builds it (cache_mode Q8 = 8,8)
if MODE != "decode":
    cache = None
elif KVQ:
    from exllamav3.cache import CacheLayer_quant
    kb, vb = (int(x) for x in KVQ.split(","))
    cache = Cache(model, max_num_tokens=(MAXT + 255) // 256 * 256 + 256, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
else:
    cache = Cache(model, max_num_tokens=(MAXT + 255) // 256 * 256 + 256)
t0 = time.time()
model.load(progressbar=False, use_per_device=SPLIT)
dev = torch.device("cuda:0")
print(f"loaded in {time.time() - t0:.0f} s; mode {MODE}, R {R}, split {SPLIT}, {len(W)} windows", flush=True)


def ref_load(w, box):
    with safe_open(f"{refd}/{w['id']}.safetensors", "pt", device="cpu") as f:
        box["logits"] = f.get_tensor("logits").pin_memory(); box["tgt"] = f.get_tensor("target_ids")


def stats(la, lb, tgt):
    out = {k: [] for k in ("kl", "jsd", "top1", "nll_a", "nll_b", "p1_a", "ent_a")}
    for c in range(0, la.shape[0], 256):
        a = torch.log_softmax(la[c:c + 256].to(dev, non_blocking=True).float(), -1); b = torch.log_softmax(lb[c:c + 256].float(), -1)
        t = tgt[c:c + 256].to(dev)
        pa_, pb_ = a.exp(), b.exp()
        out["kl"].append((pa_.double() * (a - b).double()).sum(-1))
        m = (0.5 * (pa_.double() + pb_.double())).clamp_min(1e-300).log()
        out["jsd"].append(0.5 * (pa_.double() * (a.double() - m)).sum(-1) + 0.5 * (pb_.double() * (b.double() - m)).sum(-1))
        out["top1"].append((a.argmax(-1) == b.argmax(-1)).double())
        out["nll_a"].append(-a.gather(1, t[:, None])[:, 0].double()); out["nll_b"].append(-b.gather(1, t[:, None])[:, 0].double())
        out["p1_a"].append(pa_.max(-1).values.double()); out["ent_a"].append(-(pa_.double() * a.double()).sum(-1))
    return {k: torch.cat(v).cpu() for k, v in out.items()}


summ = []
box = {}; th = threading.Thread(target=ref_load, args=(W[0], box)); th.start()
for i, w in enumerate(W):
    T = len(w["ids"]); s0 = w["score_from"]; n = T - 1 - s0
    ids = torch.tensor(w["ids"], dtype=torch.long).view(1, -1)
    t1 = time.time()
    if MODE == "decode":
        base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cache.max_num_tokens)}
        CUR["wid"] = w["id"]
        for c in range(0, s0, 2048):
            CUR["p"] = c
            model.prefill(input_ids=ids[:, c:min(c + 2048, s0)], params={**base, "past_len": c})
        cand = None
        PROF = int(os.environ.get("KLD_PROFILE", "0"))   # profile-only run: N verify-shape forwards after 60 warm ones
        for p in range(s0, T - 1, R):
            CUR["p"] = p
            if PROF and p == s0 + 60 * R:
                torch.cuda.synchronize()
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for q in range(PROF):
                        pp = p + q * R
                        model.forward(input_ids=ids[:, pp:pp + R], params={**base, "past_len": pp})
                    torch.cuda.synchronize()
                prof.export_chrome_trace(f"{out}/trace.json")
                open(f"{out}/profile.txt", "w").write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=80))
                print(f"profile written ({PROF} forwards)", flush=True)
                sys.exit(0)
            lg = model.forward(input_ids=ids[:, p:min(p + R, T)], params={**base, "past_len": p})[0]
            if cand is None: cand = torch.empty((n, lg.shape[-1]), dtype=torch.float, device=dev)
            k = min(R, T - 1 - p); cand[p - s0:p - s0 + k] = lg[:k].float()
    else:
        CUR["wid"] = w["id"]; CUR["p"] = 0
        cand = model.forward(input_ids=ids, params={})[0, s0:T - 1].float()
    torch.cuda.synchronize(); tf = time.time() - t1
    th.join(); ref = box
    if i + 1 < len(W):
        box = {}; th = threading.Thread(target=ref_load, args=(W[i + 1], box)); th.start()
    assert torch.equal(ref["tgt"], ids[0, s0 + 1:T]), w["id"]
    V = min(ref["logits"].shape[-1], cand.shape[-1])
    st = stats(ref["logits"][:, :V], cand[:, :V], ref["tgt"])
    save_file({k: v.contiguous() for k, v in st.items()}, f"{out}/pos/{w['id']}.safetensors")
    if w["id"] in SAVE:
        save_file({"logits": cand.cpu().contiguous()}, f"{out}/logits_{w['id']}.safetensors")
    s = {"id": w["id"], "V": V, "n": n, "fwd_s": tf, **{k: v.mean().item() for k, v in st.items()}}
    summ.append(s)
    print(f"{w['id']:8s} n={n} {tf:6.1f} s  KL {s['kl']:.5f}  JSD {s['jsd']:.5f}  top1 {s['top1']:.4f}  "
          f"NLL ref {s['nll_a']:.4f} cand {s['nll_b']:.4f}", flush=True)
    del cand
json.dump({"mode": MODE, "R": R, "kv": KVQ or "fp16", "split": SPLIT, "model": mdir, "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "KLD_", "CUDA_"))},
           "windows": summ}, open(out + "/summary.json", "w"), indent=1)
print(f"candidate pass done (route replay calls: {CUR['calls']})", flush=True)
if os.getuid() == 0:
    os.system(f"chown -R 1001:1001 {out}")
