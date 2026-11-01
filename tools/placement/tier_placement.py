"""Tier placement from rank.json (per-layer popularity): [hot NHOT int4 | coldest -> GB300 trellis band (pinned to Grace)
| next NRTX -> RTX]. Writes the placement (also valid as EXL3_RTX_SPLIT) and the pin map of the GB300 band (local ids).
With U4DIR and FALLBACK (an rtx_split placement), layers without layer{L}.safetensors keep the two-tier placement.
   python3 tier_placement.py rank.json NHOT NRTX out-placement.json out-pin.json [U4DIR FALLBACK]
FAR=1: the cold band is int4 in pinned Grace memory instead (placement key "far", read by the moe4 launch; empty pin map)."""
import json, os, sys
FAR = os.environ.get("FAR") == "1"
KEEP = int(os.environ.get("KEEP", "0"))
rank = json.load(open(sys.argv[1])); NHOT, NRTX = int(sys.argv[2]), int(sys.argv[3])
u4dir, fb = (sys.argv[6], json.load(open(sys.argv[7]))) if len(sys.argv) > 7 else (None, None)
place, pin = {}, {}
for L, r in rank.items():
    if u4dir and not os.path.exists(f"{u4dir}/layer{L}.safetensors"):
        place[L] = {"perm": fb[L]["perm"], "hot": 0, "first": fb[L]["first"]}; continue
    hot, rtx, cold = r[:NHOT], r[NHOT:NHOT + NRTX], r[NHOT + NRTX:]
    place[L] = {"perm": hot + cold + rtx, "hot": NHOT, "first": NHOT + len(cold)}
    if FAR: place[L]["far"] = len(cold) - KEEP
    pin[L] = list(range(KEEP)) if FAR else list(range(len(cold)))
json.dump(place, open(sys.argv[4], "w")); json.dump(pin, open(sys.argv[5], "w"))
print(f"{len(pin)} three-tier layers of {len(place)}: int4 {NHOT}, GB300 trellis (pinned) {256 - NHOT - NRTX}, RTX {NRTX}")
