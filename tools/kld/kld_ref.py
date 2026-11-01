"""Reference pass of the GLM-5.3 KLD benchmark (protocol: local-inference-lab/rtx6kpro kld/README.md), built on ExLlamaV3's
eval/model_diff.py loop: every module of both models runs over all corpus windows before the next module loads.

A (reference) = stock zai-org/GLM-5.3 BF16 (shards streamed by stage.py into sparse local files when the checkpoint does not
                fit on disk), ExLlamaV3 FP16 compute, no KV cache (one causal forward per window). --edits optionally
                replaces tensors of the reference (e.g. to score a fine-tuned variant against its own BF16).
B (candidate) = optional: any EXL3 model scored in the same pass (stock ExLlamaV3 forward, prefill shape, natural routes).

Writes
  REF/{wid}.safetensors      logits [n_scored, V] fp32 of A (key "logits") + "target_ids", "positions"
  OUT/routes_a.safetensors   A's selected experts (int16) / weights (fp16) per MoE layer per window, all tokens (B x Q replay)
  OUT/layers.jsonl           per module, per window: B-vs-A state rfn / cos error, route agreement
  OUT/pos_b/{wid}.safetensors per scored position: kl(A||B), jsd, top1 agree, nll A, nll B, A top-1 prob, A entropy
   python -I kld_ref.py --a /kbf16 [--edits EDITS.safetensors] [--b MODEL_B] --corpus /k/corpus/corpus.json
                       --ref /ref/... --out /k/runs/... --stage /k/stage"""
import sys, os, json, time, argparse
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.memory import free_mem

torch.set_grad_enabled(False)
P = argparse.ArgumentParser()
for a in ("a", "edits", "b", "corpus", "ref", "out", "stage"):
    P.add_argument("--" + a, required=a != "stage")
P.add_argument("--windows", default="")       # comma list of window ids (smoke tests)
P.add_argument("--max_modules", type=int, default=0)
P.add_argument("--smoke", action="store_true")   # modules 0-3 + norm + head
args = P.parse_args()
os.makedirs(args.ref, exist_ok=True); os.makedirs(args.out + "/pos_b", exist_ok=True)
dev = torch.device("cuda:0")

W = json.load(open(args.corpus))
if args.windows:
    W = [w for w in W if w["id"] in args.windows.split(",")]

cfg_a = Config.from_directory(args.a)
cfg_a.stc.add_tensor_files(args.edits)          # overrides the 119 edited keys (prints each)
cfg_b = Config.from_directory(args.b)
model_a = Model.from_config(cfg_a); model_b = Model.from_config(cfg_b)
assert [m.key for m in model_a.modules] == [m.key for m in model_b.modules]
tok = Tokenizer.from_config(cfg_a)

# route capture: wrap each BlockSparseMLP's routing function (called once or more per window forward)
CAP = {"a": {}, "b": {}}
def hook(model, tag):
    for mod in model.modules:
        for sub in getattr(mod, "modules", []):
            if type(sub).__name__ == "BlockSparseMLP":
                f0 = sub.routing_fn
                def f(bsz, rcfg, z, params, f0=f0, key=sub.key):
                    s, w = f0(bsz, rcfg, z, params)
                    CAP[tag].setdefault(key, []).append((s.to(torch.int16).cpu(), w.half().cpu()))
                    return s, w
                sub.routing_fn = f
hook(model_a, "a"); hook(model_b, "b")

ids = [torch.tensor(w["ids"], dtype=torch.long).view(1, -1) for w in W]
pa = [{} for _ in W]; pb = [{} for _ in W]
sa = [model_a.prepare_inputs(x, p) for x, p in zip(ids, pa)]
sb = [model_b.prepare_inputs(x, p) for x, p in zip(ids, pb)]
routes = {}
LJ = open(args.out + "/layers.jsonl", "a")


def wait_stage(i):
    if not args.stage: return
    t0 = time.time()
    while not os.path.exists(f"{args.stage}/ready.{i}"):
        time.sleep(0.2)
        if time.time() - t0 > 1800: raise RuntimeError(f"stage timeout module {i}")


def stats(la, lb, tgt):
    """per position, fp32 log-softmax, fp64 vocabulary sums"""
    out = {k: [] for k in ("kl", "jsd", "top1", "nll_a", "nll_b", "p1_a", "ent_a")}
    for c in range(0, la.shape[0], 256):
        a = torch.log_softmax(la[c:c + 256].float(), -1); b = torch.log_softmax(lb[c:c + 256].float(), -1)
        t = tgt[c:c + 256].to(a.device)
        pa_, pb_ = a.exp(), b.exp()
        out["kl"].append((pa_.double() * (a - b).double()).sum(-1))
        m = (0.5 * (pa_.double() + pb_.double())).clamp_min(1e-300).log()
        out["jsd"].append(0.5 * (pa_.double() * (a.double() - m)).sum(-1) + 0.5 * (pb_.double() * (b.double() - m)).sum(-1))
        out["top1"].append((a.argmax(-1) == b.argmax(-1)).double())
        out["nll_a"].append(-a.gather(1, t[:, None])[:, 0].double()); out["nll_b"].append(-b.gather(1, t[:, None])[:, 0].double())
        out["p1_a"].append(pa_.max(-1).values.double()); out["ent_a"].append(-(pa_.double() * a.double()).sum(-1))
    return {k: torch.cat(v).cpu() for k, v in out.items()}


T0 = time.time()
mods = list(zip(model_a.modules, model_b.modules))
if args.smoke: mods = mods[:4] + mods[-2:]
for idx, (ma, mb) in enumerate(mods):
    last = idx == len(mods) - 1
    wait_stage(idx); t0 = time.time()
    cfg_a.stc.begin_deferred_load(); ma.load(dev if not ma.caps.get("prefer_cpu") else "cpu"); cfg_a.stc.end_deferred_load()
    cfg_b.stc.begin_deferred_load(); mb.load(dev if not mb.caps.get("prefer_cpu") else "cpu"); cfg_b.stc.end_deferred_load()
    tl = time.time() - t0
    CAP["a"].clear(); CAP["b"].clear()
    rec = {"module": ma.key, "idx": idx, "win": []}
    for j, w in enumerate(W):
        pa[j]["dev_cache"] = None; pb[j]["dev_cache"] = None
        x = ma.prepare_for_device(sa[j], pa[j]); x = ma.forward(x, pa[j]).clone()
        y = mb.prepare_for_device(sb[j], pb[j]); y = mb.forward(y, pb[j]).clone()
        if not last:
            sa[j], sb[j] = x, y
            if idx > 0:
                xa = x[0].reshape(-1, x.shape[-1]).float(); yb = y[0].reshape(-1, y.shape[-1]).float()
                rfn = ((xa - yb).norm() / yb.norm()).item()
                cos = (1 - torch.nn.functional.cosine_similarity(xa, yb, dim=-1)).mean().item()
                rec["win"].append({"id": w["id"], "rfn": rfn, "cos": cos})
        else:
            s0 = w["score_from"]; T = len(w["ids"])
            la = x[0, s0:T - 1].float(); lb = y[0, s0:T - 1].float()
            V = min(la.shape[-1], lb.shape[-1]); la, lb = la[:, :V].contiguous(), lb[:, :V].contiguous()
            tgt = torch.tensor(w["ids"][s0 + 1:T], dtype=torch.long)
            st = stats(la, lb, tgt)
            save_file({k: v.contiguous() for k, v in st.items()}, f"{args.out}/pos_b/{w['id']}.safetensors")
            save_file({"logits": la.cpu().contiguous(), "target_ids": tgt.contiguous(),
                       "positions": torch.arange(s0, T - 1, dtype=torch.int32)}, f"{args.ref}/{w['id']}.safetensors")
            rec["win"].append({"id": w["id"], "V": V, "kl": st["kl"].mean().item(), "top1": st["top1"].mean().item(),
                               "nll_a": st["nll_a"].mean().item(), "nll_b": st["nll_b"].mean().item()})
            print(f"   {w['id']:8s} V={V} KL(A||B) {st['kl'].mean().item():.5f}  top1 {st['top1'].mean().item():.4f}  "
                  f"NLL A {st['nll_a'].mean().item():.4f} B {st['nll_b'].mean().item():.4f}", flush=True)
        # route traces of this window (all routing calls of this forward, concatenated over rows)
        for tag in ("a", "b"):
            for key, lst in CAP[tag].items():
                s = torch.cat([q[0].view(-1, q[0].shape[-1]) for q in lst]); wt = torch.cat([q[1].view(-1, q[1].shape[-1]) for q in lst])
                CAP[tag][key] = []
                if tag == "a":
                    routes[f"{key}.{w['id']}.sel"] = s.contiguous(); routes[f"{key}.{w['id']}.w"] = wt.contiguous()
                    CAP.setdefault("last_a", {})[key] = s
                else:
                    A = CAP["last_a"][key]
                    if A.shape == s.shape:
                        same = (A.sort(-1).values == s.sort(-1).values).all(-1).float().mean().item()
                        inter = (A.unsqueeze(-1) == s.unsqueeze(-2)).any(-1).float().sum(-1).mean().item()
                        rec["win"][-1].update({"route_same": same, "route_overlap": inter / A.shape[-1]})
    ma.unload(); cfg_a.stc.close(); free_mem()
    mb.unload(); cfg_b.stc.close(); free_mem()
    if args.stage: open(f"{args.stage}/done.{idx}", "w").close()
    LJ.write(json.dumps(rec) + "\n"); LJ.flush()
    ws = rec["win"]
    if ws and "rfn" in ws[0]:
        r = sum(v["rfn"] for v in ws) / len(ws); ro = [v["route_same"] for v in ws if "route_same" in v]
        print(f"{time.strftime('%H:%M:%S')} {idx:2d} {ma.key:20s} load {tl:5.1f} s, total {time.time() - t0:5.1f} s, "
              f"rfn(B vs A) {r:.5f}" + (f", same top-8 set {sum(ro) / len(ro):.4f}" if ro else ""), flush=True)
    else:
        print(f"{time.strftime('%H:%M:%S')} {idx:2d} {ma.key:20s} load {tl:5.1f} s, total {time.time() - t0:5.1f} s", flush=True)

save_file(routes, args.out + "/routes_a.safetensors")
print(f"reference pass done in {(time.time() - T0) / 60:.1f} min", flush=True)
