"""Incremental prefill: time to first token when a cached conversation grows by N new tokens (an agent's tool result, a
pasted file, a follow-up message). A base document is sent once so its pages are cached; then each request is the base
plus a fresh increment of N tokens (distinct text every time, so only the base is a cache hit) with max_tokens 1.
   python3 -I incprefill.py LABEL [--base 30000] [--sizes 500,1000,2000,3000,4000,8000] [--reps 3]"""
import argparse, glob, json, os, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8103/v1"))
ap.add_argument("--base", type=int, default=30000)
ap.add_argument("--sizes", default="500,1000,2000,3000,4000,8000")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
SRC = sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)) + sorted(glob.glob("/usr/include/**/*.h", recursive=True))
pos = 0


def text(n_tokens):
    global pos
    out, chars = [], int(n_tokens * 4.3)
    while sum(len(x) for x in out) < chars:
        try:
            out.append(f"# file: {SRC[pos % len(SRC)]}\n" + open(SRC[pos % len(SRC)]).read())
        except Exception:
            pass
        pos += 1
    return "\n\n".join(out)[:chars]


def ask(messages):
    req = {"model": model, "messages": messages, "max_tokens": 1, "temperature": 0}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=3600))
    u = r["usage"]
    return time.time() - t, u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


base = [{"role": "system", "content": "You are a helpful assistant.\n\n" + text(a.base)}]
dt, pt, ct = ask(base + [{"role": "user", "content": "OK?"}])
print(f"{a.label} base {pt} tokens, cold {dt:.2f} s", flush=True)
for n in [int(x) for x in a.sizes.split(",")]:
    ts, rates = [], []
    for _ in range(a.reps):
        dt, pt, ct = ask(base + [{"role": "user", "content": text(n) + "\n\nSummarize the above in one line."}])
        new = pt - ct
        ts.append(dt); rates.append(new / dt)
    print(f"{a.label} +{n:5d} tokens: ttft {statistics.median(ts):6.3f} s (new {new}, cached {ct}), "
          f"{statistics.median(rates):6.0f} new tok/s", flush=True)
