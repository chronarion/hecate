"""Host-side streaming of the stock GLM-5.3 BF16 shards from a remote host (ssh) into local sparse copies (make_sparse.py),
for machines whose disk cannot hold the 1.5 TB BF16 checkpoint.

The reference harness (kld_ref.py, in a container) loads module i after STAGE/ready.i exists and writes STAGE/done.i after
unloading it; this daemon fetches modules in forward order, at most AHEAD modules past the last done one, and punches the
done modules' ranges back to holes so the local footprint stays ~AHEAD+1 layers (~60 GB).
Integrity: the remote side md5s the byte stream while sending (tee), the local side md5s what it writes; they must match.
   python3 stage.py daemon            (main pass)
   python3 stage.py keys REGEX        (fetch only the tensors whose key matches, e.g. for the edit pre-pass)"""
import os, sys, re, json, base64, struct, subprocess, time, hashlib
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get("BF16_HOST", "bf16-host")                  # ssh host holding zai-org/GLM-5.3-BF16
REMOTE = os.environ["BF16_REMOTE"]                                # snapshot directory on that host
LOCAL = os.environ.get("LOCAL", "bf16")
STAGE = os.environ.get("STAGE", "stage")
AHEAD = int(os.environ.get("AHEAD", "2"))
PAR = int(os.environ.get("PAR", "4"))
MODULES = ["model.embed_tokens."] + [f"model.layers.{L}." for L in range(78)] + ["model.norm.", "lm_head."]

TMAP = {}   # key -> (file, abs_start, abs_end)
for f, v in json.load(open(os.environ.get("BF16_HEADERS", "bf16_headers.json"))).items():
    hb = base64.b64decode(v["hdr"]); n = struct.unpack("<Q", hb[:8])[0]
    for k, e in json.loads(hb[8:8 + n]).items():
        if k != "__metadata__":
            a, b = e["data_offsets"]; TMAP[k] = (f, 8 + n + a, 8 + n + b)


def ranges(pred):
    """merged byte ranges of the matching tensors; only exactly adjacent ranges merge (a gap may hold another module)"""
    by = {}
    for k, (f, a, b) in TMAP.items():
        if pred(k) and b > a:
            by.setdefault(f, []).append((a, b))
    out = []
    for f, rs in sorted(by.items()):
        rs.sort(); cur = list(rs[0])
        for a, b in rs[1:]:
            if a == cur[1]: cur[1] = b
            else: out.append((f, cur[0], cur[1])); cur = [a, b]
        out.append((f, cur[0], cur[1]))
    return out


def fetch1(f, a, b):
    remote = (f"bash -c 'dd if={REMOTE}/{f} bs=4M iflag=skip_bytes,count_bytes skip={a} count={b - a} status=none "
              f"| tee >(md5sum 1>&2)'")
    for attempt in range(3):
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", HOST, remote], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fd = os.open(f"{LOCAL}/{f}", os.O_WRONLY); h = hashlib.md5(); pos = a
        while True:
            d = p.stdout.read(8 << 20)
            if not d: break
            os.pwrite(fd, d, pos); h.update(d); pos += len(d)
        os.close(fd); err = p.stderr.read().decode(); rc = p.wait()
        rmd5 = err.split()[0] if err.split() else ""
        if rc == 0 and pos == b and rmd5 == h.hexdigest():
            return b - a
        print(f"  !! retry {f} [{a},{b}) rc={rc} got={pos - a} md5 {rmd5} vs {h.hexdigest()} {err[-200:]}", flush=True)
        time.sleep(2)
    raise RuntimeError(f"fetch failed {f} {a} {b}")


def fetch(rs):
    parts = [(f, x, min(x + (1 << 30), b)) for f, a, b in rs for x in range(a, b, 1 << 30)]
    t0 = time.time()
    with ThreadPoolExecutor(PAR) as ex:
        n = sum(ex.map(lambda r: fetch1(*r), parts))
    return n, time.time() - t0


def punch(rs):
    for f, a, b in rs:
        subprocess.run(["fallocate", "--punch-hole", "--keep-size", "-o", str(a), "-l", str(b - a), f"{LOCAL}/{f}"], check=True)


def daemon():
    os.makedirs(STAGE, exist_ok=True)
    R = [ranges(lambda k, p=p: k.startswith(p)) for p in MODULES]
    punched = set()

    def maxdone():
        d = -1
        for i in range(len(MODULES)):
            if os.path.exists(f"{STAGE}/done.{i}"):
                d = i
                if i not in punched:
                    punch(R[i]); punched.add(i)
        return d

    tot = 0; T0 = time.time()
    for i, p in enumerate(MODULES):
        while i > maxdone() + AHEAD:
            time.sleep(0.5)
        if os.path.exists(f"{STAGE}/ready.{i}"):
            continue
        n, dt = fetch(R[i]); tot += n
        open(f"{STAGE}/ready.{i}", "w").write(f"{n} {dt:.1f}\n")
        print(f"{time.strftime('%H:%M:%S')} ready {i:2d} {p:22s} {n / 1e9:6.2f} GB in {dt:5.1f} s ({n / dt / 1e6:4.0f} MB/s), "
              f"total {tot / 1e9:7.1f} GB, avg {tot / (time.time() - T0) / 1e6:4.0f} MB/s", flush=True)
    while maxdone() < len(MODULES) - 1:
        time.sleep(1)
    print("stage daemon done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "daemon":
        daemon()
    elif sys.argv[1] == "keys":
        rx = re.compile(sys.argv[2]); rs = ranges(lambda k: rx.search(k) is not None)
        n, dt = fetch(rs)
        print(f"fetched {len(rs)} ranges, {n / 1e9:.2f} GB in {dt:.1f} s ({n / dt / 1e6:.0f} MB/s)", flush=True)
