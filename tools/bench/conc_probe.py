"""Concurrency probe: N parallel non-streamed requests (N = 1..8), tokens per decode step per request from the usage
block's draft counts, and greedy texts at N > 1 against the same prompts run alone (with an alone-vs-alone control,
since float noise already moves greedy text between runs). A batch shape whose drafts are never accepted, or whose
greedy texts drift from the alone runs much earlier than the control does, has a bug.
   python3 -I conc_probe.py [--n 1,2,3,4,5,6,8] [--temps 0,0.8] [--max-tokens 300]"""
import argparse, json, os, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8103/v1"))
ap.add_argument("--n", default="1,2,3,4,5,6,8")
ap.add_argument("--temps", default="0,0.8")
ap.add_argument("--max-tokens", type=int, default=300)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
P = ["Explain how a hash map handles collisions, comparing chaining and open addressing.",
     "Write a Python function that merges overlapping intervals and explain its complexity.",
     "Describe the water cycle in detail for a middle-school science class.",
     "Explain the difference between TCP and UDP and when to use each.",
     "Derive the formula for the sum of a geometric series and give two examples.",
     "Write a bash script that backs up a directory with rotation of the last seven copies.",
     "Compare the economic policies of Keynesian and Austrian schools of thought.",
     "Give a step-by-step recipe for sourdough bread, including starter maintenance."]


def ask(p, temp, out, i):
    req = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens, "temperature": temp,
           "template_vars": {"reasoning_effort": "low"}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    u = r["usage"]; d = u.get("completion_tokens_details") or {}
    g, acc = u["completion_tokens"], d.get("accepted_prediction_tokens") or 0
    m = r["choices"][0]["message"]
    out[i] = {"gen": g, "acc": acc, "rej": d.get("rejected_prediction_tokens") or 0, "tps": (g - 1) / max(1, g - 1 - acc),
              "rate": u.get("completion_tokens_per_sec") or 0, "text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")}


def batch(n, temp):
    out = [None] * n
    th = [threading.Thread(target=ask, args=(P[i % len(P)], temp, out, i)) for i in range(n)]
    for t in th: t.start()
    for t in th: t.join()
    return out


def prefix(x, y):
    k = 0
    for c1, c2 in zip(x, y):
        if c1 != c2: break
        k += 1
    return k


alone = {}
for temp in [float(t) for t in a.temps.split(",")]:
    if temp == 0:
        alone = {i: batch(1, 0)[0] for i in range(4)} if False else {}
        for i in range(8):
            alone[i] = [None]; ask(P[i], 0, alone[i], 0); alone[i] = alone[i][0]
        ctrl = []
        for i in range(4):
            o = [None]; ask(P[i], 0, o, 0); ctrl.append(prefix(alone[i]["text"], o[0]["text"]))
        print(f"greedy alone vs alone (control): common prefix chars {ctrl}", flush=True)
    for n in [int(x) for x in a.n.split(",")]:
        res = batch(n, temp)
        tps = [r["tps"] for r in res]; acc = sum(r["acc"] for r in res); prop = sum(r["acc"] + r["rej"] for r in res)
        line = (f"T={temp} N={n}: tokens/step per request {min(tps):.2f}..{max(tps):.2f} (mean {sum(tps) / n:.2f}), drafts "
                f"accepted {acc}/{prop}, per-request {sum(r['rate'] for r in res) / n:.1f} tok/s")
        if temp == 0 and n > 1:
            line += f", common prefix with the alone run (chars): {[prefix(res[i]['text'], alone[i % 8]['text']) for i in range(min(n, 8))]}"
        print(line, flush=True)
