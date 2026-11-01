"""KV-room placements from the served 144 / 85 / 27 layout: the coldest hot experts (ranks NHOT..143) move to the head of
the far band (pinned Grace int4, prefill on the GB300 from there; decode: the first EXL3_TIER_RTX_I4 of them as int4 on the
RTX, the rest on the Grace cores), freeing ~1.42 GB of GB300 HBM per rank for the KV cache. RTX band and cold band unchanged.
   python3 kv_placement.py rank.json base.json NHOT out.json"""
import json, sys
rank, base, NHOT, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), int(sys.argv[3]), sys.argv[4]
res = {}
for L, p in base.items():
    if L not in rank or p.get("first", 0) >= len(p["perm"]):
        res[L] = p; continue                       # MTP layer (no RTX band) unchanged
    r = rank[L]; hot, rtx, cold = r[:144], r[144:229], r[229:]
    assert p["perm"] == hot + cold + rtx and p["hot"] == 144 and p["far"] == 27, f"layer {L}: base is not the 144/85/27 layout"
    res[L] = {"perm": r[:NHOT] + r[NHOT:144] + cold + rtx, "hot": NHOT, "first": 171, "far": 171 - NHOT}
json.dump(res, open(out, "w"))
print(f"{out}: hot {NHOT}, far {171 - NHOT} (hot tail {144 - NHOT} + cold 27), RTX 85")
