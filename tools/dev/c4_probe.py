"""Draft acceptance at N parallel requests for one request shape (the llm-inference-bench C=4 dip: one token per step).
   python3 -I c4_probe.py N [--stream] [--ignore-eos] [--temp T] [--max-tokens 600] [--effort low|none]"""
import argparse, json, os, threading, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8103/v1"))
ap.add_argument("--stream", action="store_true")
ap.add_argument("--ignore-eos", action="store_true")
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--max-tokens", type=int, default=600)
ap.add_argument("--effort", default="none")
ap.add_argument("--bench", action="store_true", help="llm-inference-bench's prompt and continuous usage stats")
ap.add_argument("--ctx-words", type=int, default=0, help="a shared document of this many words before each question (long-context decode)")
ap.add_argument("--same", action="store_true", help="every request gets the same prompt (llm-inference-bench)")
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
BENCH = ("Write an extremely detailed, comprehensive encyclopedia article about the complete history of mathematics from "
         "ancient Mesopotamia to 2025. Cover every civilization, every major mathematician, every theorem, proof, and "
         "breakthrough. Include detailed biographical information, historical context, and mathematical explanations. "
         "Do not summarize - provide maximum detail on every topic.")
P = ["Write a long story about a lighthouse keeper.", "Explain how TCP congestion control works in detail.",
     "Describe the history of the Roman empire.", "Write a detailed guide to growing tomatoes.",
     "Explain how compilers optimize loops.", "Describe the water cycle in detail.",
     "Write a long essay about the ocean.", "Explain the theory of relativity for a student."]


def ask(i, out):
    q = BENCH if a.bench else P[0 if a.same else i % len(P)]
    if a.ctx_words:
        q = DOC + "\n\nSet the document aside. " + q
    req = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": a.max_tokens}
    if a.temp is not None:
        req["temperature"] = a.temp
    if a.ignore_eos:
        req["ignore_eos"] = True
    if a.effort != "none":
        req["template_vars"] = {"reasoning_effort": a.effort}
    if a.stream:
        req["stream"] = True; req["stream_options"] = {"include_usage": True, **({"continuous_usage_stats": True} if a.bench else {})}
    r = urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                      {"Content-Type": "application/json"}), timeout=1800)
    u = None
    if a.stream:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                d = json.loads(line[5:])
                if d.get("usage"):
                    u = d["usage"]
    else:
        u = json.load(r)["usage"]
    d = u.get("completion_tokens_details") or {}
    g, acc, rej = u["completion_tokens"], d.get("accepted_prediction_tokens") or 0, d.get("rejected_prediction_tokens") or 0
    out[i] = (g, acc, rej, u.get("completion_tokens_per_sec") or 0)


import random as _r
_g = _r.Random(7)
_W = "river stone lamp orbit garden copper silent harbor violet engine marble canyon whisper ledger falcon prism".split()
DOC = " ".join(_g.choice(_W) for _ in range(a.ctx_words)) if a.ctx_words else ""
out = [None] * a.n
th = [threading.Thread(target=ask, args=(i, out)) for i in range(a.n)]
for t in th: t.start()
for t in th: t.join()
tps = [(g - 1) / max(1, g - 1 - acc) for g, acc, rej, _ in out]
print(f"N={a.n} stream={a.stream} ignore_eos={a.ignore_eos} temp={a.temp} effort={a.effort} same={a.same} bench={a.bench} ctx_words={a.ctx_words}: gen {[o[0] for o in out]}, "
      f"tokens/step {min(tps):.2f}..{max(tps):.2f}, drafts accepted {sum(o[1] for o in out)}/"
      f"{sum(o[1] + o[2] for o in out)}, per-request {sum(o[3] for o in out) / a.n:.1f} tok/s", flush=True)
