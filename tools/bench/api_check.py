"""API behaviour checks against the Hecate server (:8103): exact max_tokens, stop strings, two concurrent streams, a client
cancel mid-stream followed by a fresh request, and a streamed answer (with the greedy run-to-run agreement as information).
   python3 -I api_check.py"""
import json, threading, time, urllib.request

URL = "http://127.0.0.1:8103/v1"
model = json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
fails = []


def post(body, stream=False, timeout=600):
    body = {"model": model, "temperature": 0, **body, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(f"{URL}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(content, **kw):
    r = json.load(post({"messages": [{"role": "user", "content": content}], **kw}))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), r["usage"], r["choices"][0].get("finish_reason")


def stream_text(content, n_chunks=None, **kw):
    out, usage = [], None
    with post({"messages": [{"role": "user", "content": content}], **kw}, stream=True) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                out.append((delta.get("reasoning_content") or "") + (delta.get("content") or ""))
            if n_chunks is not None and len(out) >= n_chunks:
                return "".join(out), None          # closes the connection mid-stream
    return "".join(out), usage


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)
    if not ok:
        fails.append(name)


# 1. max_tokens is exact
for n in (1, 5, 17, 64):
    c, rc, u, fr = chat("Write a long story about a lighthouse.", max_tokens=n)
    check(f"max_tokens {n}", u["completion_tokens"] == n and fr == "length", f"got {u['completion_tokens']} ({fr})")

# 2. stop strings end the output before the stop text
c, rc, u, fr = chat("Count from 1 to 30, separated by commas, with no other text.", max_tokens=1500, stop=["17"])
t = rc + c   # the stop string applies to everything generated, the reasoning included
check("stop string", "17" not in t and len(t) > 0 and fr == "stop", f"{fr}: {t[-40:]!r}")

# 3. two concurrent streams both complete and stay on topic
res = {}
def worker(k, prompt):
    res[k] = stream_text(prompt, max_tokens=300)
ts = [threading.Thread(target=worker, args=(0, "Explain photosynthesis in three sentences.")),
      threading.Thread(target=worker, args=(1, "Explain how a car engine works in three sentences."))]
[t.start() for t in ts]; [t.join() for t in ts]
check("concurrent streams", all(res[k][1] and res[k][1]["completion_tokens"] > 20 for k in res) and
      "photosynth" in res[0][0].lower() and "engine" in res[1][0].lower(),
      f"tokens {[res[k][1]['completion_tokens'] if res[k][1] else None for k in res]}")

# 4. cancel mid-stream, then a fresh request answers correctly
stream_text("Write a very long essay about the history of Rome.", n_chunks=40, max_tokens=2000)
time.sleep(1.0)
c, rc, u, fr = chat("What is 17*23? Answer with the number only.", max_tokens=1500)
check("request after cancel", "391" in c, repr(c[:40]))

# 5. streamed vs non-streamed greedy output of one prompt. The MoE kernels' fp32 atomics change summation order between
# runs, so greedy runs can part at a near-tie; a second non-streamed run gives the noise baseline (informational)
def common(a_, b_):
    n = 0
    while n < min(len(a_), len(b_)) and a_[n] == b_[n]:
        n += 1
    return n
p = "List the planets of the solar system in order, one per line."
c1, rc1, _, _ = chat(p, max_tokens=400)
c2, rc2, _, _ = chat(p, max_tokens=400)
s, us = stream_text(p, max_tokens=400)
a1, a2 = (rc1 + c1).strip(), (rc2 + c2).strip()
print(f"INFO  greedy agreement (common prefix chars): stream vs run1 {common(s.strip(), a1)} / {len(a1)}, "
      f"run2 vs run1 {common(a2, a1)} / {len(a1)}; all three name Neptune: {all('Neptune' in t for t in (s, a1, a2))}", flush=True)
check("stream output complete", us is not None and us["completion_tokens"] > 50 and "Neptune" in s, "")

print(f"api_check: {'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}", flush=True)
