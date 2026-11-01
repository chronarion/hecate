"""Lay out the BF16 shards as sparse files: real header bytes, data regions left as holes (filled per module later)."""
import os, sys, json, base64
H = json.load(open(sys.argv[1])); d = sys.argv[2]
for f, v in H.items():
    p = os.path.join(d, f)
    hb = base64.b64decode(v["hdr"])
    with open(p, "wb") as fp:
        fp.write(hb)
        fp.truncate(v["size"])
print(len(H), "files,", sum(v["size"] for v in H.values()) / 1e12, "TB apparent")
