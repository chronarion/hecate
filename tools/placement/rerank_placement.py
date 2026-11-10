"""Re-rank which int4 experts sit in GB300 HBM (hot) and which on the Grace (cold) from decode routing logs, keeping the
RTX band exactly as it is (its experts and their order, so the 3-bit RTX weights still match). Per layer, the hot + cold
experts (the int4 files hold both sets) are ordered by the number of verify steps that pick them (distinct per step);
the top HOT stay in HBM, the rest go to the Grace band, most-picked first.
Logs: EXL3_ROUTE_LOG files (runtime/tier/rtx_graph.log_routing; placement-local ids of the placement they were recorded
under), each with a weight (its steps are scaled so every log counts in proportion to its weight).
   python3 -I rerank_placement.py BASE.json OUT.json LOG.bin[:weight] [LOG2.bin[:weight] ...]
The routing changes which experts run where; the Grace band computes with 8-bit activations, the GB300 with fp16, so a
re-ranked placement needs a KLD check (tools/kld) before it ships."""
import collections, json, struct, sys
import numpy as np

base = json.load(open(sys.argv[1]))
logs = []
for a in sys.argv[3:]:
    path, _, w = a.partition(":")
    logs.append((path, float(w or 1.0)))


def read(path):
    data = open(path, "rb").read(); off, layers, steps = 0, None, []
    while off + 4 <= len(data):
        tag, n = struct.unpack_from("<HH", data, off)
        if tag == 0xA55B:
            layers = np.frombuffer(data, np.int16, n, off + 4).tolist(); off += 4 + 2 * n
        elif tag == 0xA55A:
            R = struct.unpack_from("<H", data, off + 4)[0]
            steps.append(np.frombuffer(data, np.int16, n * R * 8, off + 6).reshape(n, R, 8)); off += 6 + 2 * n * R * 8
        else:
            raise SystemExit(f"{path}: bad record tag {tag:#x} at {off}")
    return layers, steps


cnt = collections.defaultdict(lambda: np.zeros(256))           # layer -> global expert id -> weighted step count
for path, w in logs:
    layers, steps = read(path)
    if not steps:
        continue
    scale = w / len(steps)
    for i, L in enumerate(layers):
        p = base.get(str(L))
        if not p:
            continue
        perm = np.asarray(p["perm"])
        c = np.zeros(256)
        for s in steps:
            c[perm[np.unique(s[i])]] += 1
        cnt[L] += c * scale
    print(f"{path}: {len(steps)} steps, weight {w}", flush=True)
out, moved = {}, []
for k, p in base.items():
    L = int(k)
    if L not in cnt or "far" not in p or p["first"] >= len(p["perm"]):
        out[k] = p
        continue
    perm, hot, first = p["perm"], p["hot"], p["first"]
    pool = perm[:first]                                         # hot + cold (int4 in both files)
    order = sorted(pool, key=lambda e: (-cnt[L][e], pool.index(e)))
    newp = order + perm[first:]
    out[k] = dict(p, perm=newp)
    moved.append(len(set(order[:hot]) - set(perm[:hot])))
json.dump(out, open(sys.argv[2], "w"))
print(f"wrote {sys.argv[2]}: {len(moved)} layers re-ranked, experts swapped between HBM and Grace per layer: "
      f"mean {np.mean(moved):.1f}, max {max(moved)}", flush=True)
