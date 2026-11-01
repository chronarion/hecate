"""Statistics for the GLM-5.3 KLD benchmark, as the rtx6kpro kld/README asks: micro mean / median / p95 / p99 / p99.9 / max,
macro mean over strata, JSD, top-1 agreement, per-stratum / per-partition / per-depth tables, reference-confidence buckets,
source-cluster (window) bootstrap CIs resampled within strata, paired deltas between candidates, worst contexts, and with
KLD_FLOOR=A,B the runtime-repeat floor (KL between the logits two runs cand-A / cand-B of one candidate both saved).
   python -I kld_report.py /k  CAND_LABEL[=DIR] ...   -> /k/report.md, /k/report.json"""
import sys, os, json, glob
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

K = sys.argv[1]
W = json.load(open(f"{K}/corpus/corpus.json")); WI = {w["id"]: w for w in W}
STRATA = list(dict.fromkeys(w["stratum"] for w in W))
tok = Tokenizer.from_file(f"{K}/bf16/tokenizer.json")
cands = []
for a in sys.argv[2:]:
    lab, d = (a.split("=", 1) + [None])[:2]
    cands.append((lab, d or f"{K}/runs/cand-{lab}/pos"))
G = torch.Generator().manual_seed(1234)
NB = 2000


def load(d):
    P = {}
    for w in W:
        f = f"{d}/{w['id']}.safetensors"
        if os.path.exists(f): P[w["id"]] = load_file(f)
    return P


def q(x, p): return torch.quantile(x.double(), p).item() if x.numel() < 16_000_000 else x.double().sort().values[int(p * (x.numel() - 1))].item()


def boot(P, key="kl", paired=None):
    """95% CI of the micro mean and the macro (stratum) mean; windows resampled with replacement within each stratum"""
    by = {s: [i for i in P if WI[i]["stratum"] == s] for s in STRATA}
    sums = {i: (P[i][key].double() - (paired[i][key].double() if paired else 0)).sum().item() for i in P}
    cnt = {i: P[i][key].numel() for i in P}
    mi, ma = [], []
    for _ in range(NB):
        tot = n = 0; sm = []
        for s, ids in by.items():
            if not ids: continue
            pick = [ids[j] for j in torch.randint(len(ids), (len(ids),), generator=G).tolist()]
            a = sum(sums[i] for i in pick); b = sum(cnt[i] for i in pick)
            tot += a; n += b; sm.append(a / b)
        mi.append(tot / n); ma.append(sum(sm) / len(sm))
    mi, ma = torch.tensor(mi), torch.tensor(ma)
    return (q(mi, .025), q(mi, .975)), (q(ma, .025), q(ma, .975))


def summary(P):
    allk = torch.cat([P[i]["kl"] for i in P]).double()
    r = {"n": allk.numel(), "mean": allk.mean().item(), "median": q(allk, .5), "p95": q(allk, .95), "p99": q(allk, .99),
         "p999": q(allk, .999), "max": allk.max().item(),
         "jsd": torch.cat([P[i]["jsd"] for i in P]).double().mean().item(),
         "top1": torch.cat([P[i]["top1"] for i in P]).double().mean().item(),
         "nll_ref": torch.cat([P[i]["nll_a"] for i in P]).double().mean().item(),
         "nll_cand": torch.cat([P[i]["nll_b"] for i in P]).double().mean().item()}
    r["strata"] = {}
    for s in STRATA:
        ids = [i for i in P if WI[i]["stratum"] == s]
        if ids:
            x = torch.cat([P[i]["kl"] for i in ids]).double()
            r["strata"][s] = {"n": x.numel(), "mean": x.mean().item(), "p99": q(x, .99),
                              "top1": torch.cat([P[i]["top1"] for i in ids]).double().mean().item()}
    r["macro"] = sum(v["mean"] for v in r["strata"].values()) / len(r["strata"])
    r["partition"] = {p: torch.cat([P[i]["kl"] for i in P if WI[i]["partition"] == p]).double().mean().item()
                      for p in ("analysis", "qualification")}
    # depth: position inside the window (long windows: their scored span is 6143-8190)
    dep = {}
    for i in P:
        s0 = WI[i]["score_from"]; pos = torch.arange(s0, s0 + P[i]["kl"].numel())
        for lo, hi, nm in ((0, 128, "0-127"), (128, 512, "128-511"), (512, 1024, "512-1023"), (1024, 2048, "1024-2047"), (2048, 1 << 30, "6143-8190")):
            m = (pos >= lo) & (pos < hi)
            if m.any(): dep.setdefault(nm, []).append(P[i]["kl"][m])
    r["depth"] = {k: torch.cat(v).double().mean().item() for k, v in dep.items()}
    conf = torch.cat([P[i]["p1_a"] for i in P]); kl = allk
    r["confidence"] = {}
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, .95), (.95, 1.01)):
        m = (conf >= lo) & (conf < hi)
        r["confidence"][f"{lo:.2f}-{min(hi, 1):.2f}"] = {"share": m.double().mean().item(), "mean": kl[m].mean().item() if m.any() else None}
    (r["ci_micro"], r["ci_macro"]) = boot(P)
    # worst contexts
    flat = [(P[i]["kl"][j].item(), i, j) for i in P for j in torch.topk(P[i]["kl"], min(3, P[i]["kl"].numel())).indices.tolist()]
    flat.sort(reverse=True); r["worst"] = []
    for v, i, j in flat[:8]:
        w = WI[i]; p = w["score_from"] + j
        r["worst"].append({"kl": v, "window": i, "pos": p, "context": tok.decode(w["ids"][max(0, p - 12):p + 1]),
                           "next": tok.decode([w["ids"][p + 1]])})
    return r


res = {}; loaded = {}
for lab, d in cands:
    P = load(d)
    if not P: print("no data for", lab, d); continue
    loaded[lab] = P; res[lab] = summary(P)
    res[lab]["windows"] = len(P)
labs = list(res)
# paired deltas vs the first candidate given (positions shared by both)
pairs = {}
for lab in labs[1:]:
    A, B = loaded[lab], loaded[labs[0]]
    common = {i: A[i] for i in A if i in B}
    d = torch.cat([A[i]["kl"].double() - B[i]["kl"].double() for i in common])
    ci, _ = boot(common, paired={i: B[i] for i in common})
    pairs[lab] = {"vs": labs[0], "delta_mean": d.mean().item(), "ci": ci}
# runtime-repeat floor: KL(run1 || run2) on saved candidate logits
floor = {}
for a, b in ([os.environ["KLD_FLOOR"].split(",")] if os.environ.get("KLD_FLOOR") else []):
    fa = sorted(glob.glob(f"{K}/runs/cand-{a}/logits_*.safetensors"))
    for f in fa:
        g = f.replace(f"cand-{a}", f"cand-{b}")
        if os.path.exists(g):
            x = load_file(f)["logits"]; y = load_file(g)["logits"]
            kl = []
            for c in range(0, x.shape[0], 256):
                la = torch.log_softmax(x[c:c + 256].float(), -1); lb = torch.log_softmax(y[c:c + 256].float(), -1)
                kl.append((la.exp().double() * (la - lb).double()).sum(-1))
            kl = torch.cat(kl)
            floor[os.path.basename(f)[7:-12]] = {"mean": kl.mean().item(), "max": kl.max().item(),
                                                  "identical_rows": (kl == 0).double().mean().item()}
json.dump({"candidates": res, "paired": pairs, "floor": floor}, open(f"{K}/report.json", "w"), indent=1)

f = lambda x: f"{x:.5f}" if isinstance(x, float) else str(x)
o = ["# GLM-5.3 KLD (reference: stock BF16; KL(ref || candidate), nats, full vocabulary)", ""]
o += ["| candidate | positions | mean | 95% CI (window bootstrap) | macro mean | median | p95 | p99 | p99.9 | max | JSD | top-1 | NLL ref / cand |",
      "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for lab in labs:
    r = res[lab]
    o.append(f"| {lab} | {r['n']} | **{r['mean']:.5f}** | {r['ci_micro'][0]:.5f}-{r['ci_micro'][1]:.5f} | {r['macro']:.5f} | {r['median']:.5f} | "
             f"{r['p95']:.4f} | {r['p99']:.4f} | {r['p999']:.3f} | {r['max']:.3f} | {r['jsd']:.5f} | {r['top1']:.4f} | {r['nll_ref']:.4f} / {r['nll_cand']:.4f} |")
o += ["", "Per stratum (mean KL / top-1):", "", "| candidate | " + " | ".join(STRATA) + " | analysis | qualification |", "|---|" + "---:|" * (len(STRATA) + 2)]
for lab in labs:
    r = res[lab]
    o.append(f"| {lab} | " + " | ".join(f"{r['strata'][s]['mean']:.4f} / {r['strata'][s]['top1']:.3f}" if s in r["strata"] else "-" for s in STRATA)
             + f" | {r['partition']['analysis']:.5f} | {r['partition']['qualification']:.5f} |")
o += ["", "Per depth (position in window, mean KL):", "", "| candidate | " + " | ".join(res[labs[0]]["depth"]) + " |", "|---|" + "---:|" * len(res[labs[0]]["depth"])]
for lab in labs:
    o.append(f"| {lab} | " + " | ".join(f"{v:.5f}" for v in res[lab]["depth"].values()) + " |")
o += ["", "By reference top-1 probability (share of positions, mean KL):", ""]
for lab in labs:
    o.append(f"- {lab}: " + ", ".join(f"[{k}] {v['share'] * 100:.1f}% {v['mean']:.4f}" if v["mean"] is not None else f"[{k}] 0%" for k, v in res[lab]["confidence"].items()))
if pairs:
    o += ["", "Paired deltas (same positions; CI from the window bootstrap):", ""]
    for lab, p in pairs.items():
        o.append(f"- {lab} - {p['vs']}: {p['delta_mean']:+.5f} (95% CI {p['ci'][0]:+.5f} .. {p['ci'][1]:+.5f})")
if floor:
    o += ["", "Runtime-repeat floor (KL between two runs of one candidate, same windows):", ""]
    for k, v in floor.items():
        o.append(f"- {k}: mean {v['mean']:.2e}, max {v['max']:.2e}, bit-identical rows {v['identical_rows'] * 100:.1f}%")
o += ["", "Worst contexts (per candidate):", ""]
for lab in labs:
    for w in res[lab]["worst"][:4]:
        o.append(f"- {lab} {w['window']}@{w['pos']}: KL {w['kl']:.2f}, ...{json.dumps(w['context'])[-80:]} -> {json.dumps(w['next'])}")
open(f"{K}/report.md", "w").write("\n".join(o) + "\n")
print("\n".join(o))
