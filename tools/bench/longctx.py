"""Long-context check for the GLM-5.3 EXL3 server: a prompt of ~N tokens (Python stdlib sources, then a question about
them), streamed; reports the actual prompt tokens, prefill time (TTFT) / prefill tok/s and decode tok/s at that depth.
Each length uses fresh text (no prefix-cache reuse between lengths).
   python3 -I longctx.py LABEL [TOKENS, default 32768,131072]"""
import glob, json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8103/v1"
model = json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
files = sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)) + sorted(glob.glob("/usr/lib/python3/dist-packages/**/*.py", recursive=True)) + sorted(glob.glob("/usr/include/**/*.h", recursive=True))
pos = int(os.environ.get("START", "0"))   # first source file (different START = different text, no prefix sharing)


def text(n_tokens):
    global pos
    out, chars = [], int(n_tokens * 3.3)    # ~3.3 chars per token for Python source with this tokenizer
    while sum(len(x) for x in out) < chars and pos < len(files):
        try: out.append(f"# file: {files[pos]}\n" + open(files[pos]).read())
        except Exception: pass
        pos += 1
    return "\n\n".join(out)[:chars]


label = sys.argv[1] if len(sys.argv) > 1 else "served"
for n in [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "32768,131072").split(",")]:
    msg = text(n - 200) + "\n\nWhich of the files above defines the most classes? Answer in two sentences."
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": int(os.environ.get("MAXTOK", "300")), "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{URL}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None; ans = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                ans.append(dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or "")
                if tf is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")): tf = time.time()
    t1 = time.time(); u = usage or {}
    pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    print(f"{label} ctx {pt}: prefill {tf - t0:.1f} s ({pt / max(1e-6, tf - t0):.0f} tok/s), "
          f"decode {(ct - 1) / max(1e-6, t1 - tf):.1f} tok/s over {ct} tokens | answer: " + "".join(ans)[-400:].replace(chr(10), " "), flush=True)
