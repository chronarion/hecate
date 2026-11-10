"""Decode routing statistics from an EXL3_ROUTE_LOG file (runtime/tier/rtx_graph.log_routing): how many distinct experts
each band gets per layer and step, the per-layer popularity of every expert, and a band-race model of exact int4 copies of
the most-routed cold (Grace) experts on the RTX.
Copy model: per layer, copies of the C most-routed cold experts (chosen on the first half of the steps, scored on the
second). A step's cold picks in the Grace band's order (first appearance over rows, then top-k order); the copied picks
that form a trailing run of that list (never the first pick) move to the RTX, so the Grace band's partial plus the moved
picks in order is the same fp32 sum. Band times from the 1.8 measurements: Grace job 8 / 86 / 150 / 254 us for 0-3 picks,
+90 per further pick; hot band ends 78.3 us after the handoff (+2.9 per distinct hot expert off the median); RTX 75.5 us
(+slope per RTX expert) + COST per moved pick. Window = max of the three, per layer.
   python3 -I route_stats.py ROUTES.bin PLACEMENT.json [--copies 0,3,5,7] [--cost 15,22] [--anyorder]
                             [--write-copies OUT.json]   (per layer: cold local ids, most often picked first)"""
import collections, json, struct, sys
import numpy as np

args = [x for x in sys.argv[1:] if not x.startswith("--")]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
COPIES = [int(x) for x in opt("--copies", "0,3,5,7").split(",")]
COSTS = [float(x) for x in opt("--cost", "15,22").split(",")]
ANY = "--anyorder" in sys.argv
data = open(args[0], "rb").read()
P = json.load(open(args[1]))
off, layers, steps = 0, None, []
while off + 4 <= len(data):
    tag, n = struct.unpack_from("<HH", data, off)
    if tag == 0xA55B:
        layers = np.frombuffer(data, np.int16, n, off + 4).tolist(); off += 4 + 2 * n
    elif tag == 0xA55A:
        R = struct.unpack_from("<H", data, off + 4)[0]
        k = 8
        s = np.frombuffer(data, np.int16, n * R * k, off + 6).reshape(n, R, k)
        steps.append(s); off += 6 + 2 * n * R * k
    else:
        raise SystemExit(f"bad record tag {tag:#x} at {off}")
steps = [s for s in steps if s.shape[1] == 4]
print(f"{len(steps)} four-row verify steps, {len(layers)} tier layers")
GR = [8, 86, 150, 254] + [254 + 90 * i for i in range(1, 40)]
band = {}
for i, L in enumerate(layers):
    p = P.get(str(L))
    if not p:
        continue
    band[i] = (p["hot"], p["first"])        # local ids: [0, hot) GB300, [hot, first) Grace, [first, 256) RTX
nh, nc, nr = [], [], []
pick_hist = collections.Counter()
for s in steps:
    for i, (h, f) in band.items():
        u = np.unique(s[i])
        nh.append(int((u < h).sum())); c = int(((u >= h) & (u < f)).sum()); nc.append(c); nr.append(int((u >= f).sum()))
        pick_hist[min(c, 4)] += 1
mh, mr = float(np.median(nh)), float(np.median(nr))
tot = sum(pick_hist.values())
print(f"distinct experts per layer-step: hot mean {np.mean(nh):.1f}, Grace mean {np.mean(nc):.2f}, RTX mean {np.mean(nr):.1f}")
print("Grace picks per layer-step: " + ", ".join(f"{k}{'+' if k == 4 else ''}: {100 * v / tot:.1f}%" for k, v in sorted(pick_hist.items())))
SR = 49.1 / max(mr, 1.0)


def order(s_layer, h, f):
    out = []
    for r in range(s_layer.shape[0]):
        for e in s_layer[r].tolist():
            if h <= e < f and e not in out:
                out.append(e)
    return out


half = len(steps) // 2
if "--write-copies" in sys.argv:
    out = {}
    for i, (h, f) in band.items():
        cnt = collections.Counter()
        for s in steps:
            cnt.update(set(e for e in s[i].ravel().tolist() if h <= e < f))
        out[str(layers[i])] = [e - h for e, _ in cnt.most_common()] + [e - h for e in range(h, f) if e not in cnt]
    json.dump(out, open(opt("--write-copies", ""), "w"))
    print(f"wrote the cold ranking of {len(out)} layers to {opt('--write-copies', '')}")
for C in COPIES:
    for cost in COSTS if C else COSTS[:1]:
        w0 = w1 = late = moved = 0.0
        n = 0
        for i, (h, f) in band.items():
            cnt = collections.Counter()
            for s in steps[:half]:
                cnt.update(set(e for e in s[i].ravel().tolist() if h <= e < f))
            top = set(e for e, _ in cnt.most_common(C))
            for s in steps[half:]:
                u = np.unique(s[i]); a = int((u < h).sum()); b = int((u >= f).sum())
                picks = order(s[i], h, f)
                if ANY:     # every copied pick but the first (changes the fp32 order of the Grace term)
                    m = sum(1 for e in picks[1:] if e in top)
                else:
                    m = 0
                    while m < len(picks) - 1 and picks[len(picks) - 1 - m] in top:
                        m += 1
                th = 78.3 + 2.9 * (a - mh); tr = 75.5 + SR * (b - mr); tg = GR[len(picks)] + 2.0
                w0 += max(th, tr, tg); w1 += max(th, tr + cost * m, GR[len(picks) - m] + 2.0)
                late += max(0.0, tg - max(th, tr)); moved += m; n += 1
        L = len(band)
        print(f"copies {C} per layer (RTX {cost:.0f} us per moved pick): MoE window {w0 / n:.1f} -> {w1 / n:.1f} us per layer, "
              f"{(w0 - w1) / n * L / 1000:+.2f} ms per step over {L} layers; Grace lateness now {late / n:.1f} us/layer; "
              f"moved picks {moved / n:.3f} per layer-step")
