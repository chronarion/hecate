"""Overload check: a batch client floods the server with more non-streamed requests than it can serve before the client
times out, while an interactive client sends a streamed request every --agent-every seconds. With the concurrency
defaults every running request must keep decoding at the 8-row speed (~30 tok/s), never at a few tokens per second;
requests the batch client abandons must be cancelled, not decoded; agents must start within a step or two; and the
server must be idle afterwards.
   python3 -I overload_check.py [--batch 48] [--batch-tokens 2000] [--timeout 240] [--agent-every 45]"""
import argparse, json, os, socket, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8103/v1"))
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--batch-tokens", type=int, default=2000)
ap.add_argument("--timeout", type=float, default=240)
ap.add_argument("--agent-every", type=float, default=45)
ap.add_argument("--prompt-tokens", type=int, default=0, help="prepend ~this many tokens of unique text to each batch prompt (no cache hits)")
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
T0 = time.time()
res, agents, lock = [], [], threading.Lock()


def post(req, timeout):
    return urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                         {"Content-Type": "application/json"}), timeout=timeout)


WORDS = ("river stone lamp orbit garden copper silent harbor violet engine marble canyon whisper ledger falcon prism "
         "meadow anchor ember quartz tundra velvet beacon cipher lantern summit glacier thicket harvest compass").split()


def filler(i, n):
    import random
    r = random.Random(1000 + i)
    return f"Document {i}:\n" + " ".join(r.choice(WORDS) for _ in range(n)) + "\n\n" if n else ""


def batch_req(i):
    req = {"model": model, "messages": [{"role": "user", "content": filler(i, a.prompt_tokens) + f"Write a detailed report, part {i}, on the history of "
                                         "bridges: engineering, materials, famous failures and what they taught."}],
           "max_tokens": a.batch_tokens, "temperature": 0.8, "template_vars": {"reasoning_effort": "low"}}
    t = time.time()
    try:
        u = json.load(post(req, a.timeout))["usage"]
        r = ("ok", time.time() - t, u["completion_tokens"], u.get("completion_tokens_per_sec") or 0)
    except (socket.timeout, TimeoutError):
        r = ("timeout", time.time() - t, 0, 0)
    except Exception as e:
        r = (f"error {type(e).__name__}: {e}", time.time() - t, 0, 0)
    with lock:
        res.append(r)


def agent_req(k):
    req = {"model": model, "messages": [{"role": "user", "content": "Explain what a race condition is, with a short example."}],
           "max_tokens": 300, "temperature": 0.8, "stream": True, "stream_options": {"include_usage": True},
           "template_vars": {"reasoning_effort": "low"}}
    t0 = time.time(); tf = None; n = 0
    with post(req, 600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            if tf is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                                  for c in d.get("choices", [])):
                tf = time.time()
    t1 = time.time()
    agents.append((t0 - T0, (tf or t1) - t0, (n - 1) / max(1e-6, t1 - (tf or t1))))
    print(f"  agent {k} at {t0 - T0:.0f} s: first token {(tf or t1) - t0:.2f} s, {(n - 1) / max(1e-6, t1 - (tf or t1)):.1f} tok/s",
          flush=True)


th = [threading.Thread(target=batch_req, args=(i,)) for i in range(a.batch)]
for t in th:
    t.start()
k = 0
while any(t.is_alive() for t in th):
    time.sleep(min(a.agent_every, 5) if k else 5)
    if time.time() - T0 >= 5 + k * a.agent_every:
        agent_req(k); k += 1
for t in th:
    t.join()
ok = [r for r in res if r[0] == "ok"]; to = [r for r in res if r[0] == "timeout"]; err = [r for r in res if r[0].startswith("error")]
rates = sorted(r[3] for r in ok)
print(f"batch: {len(ok)} finished, {len(to)} timed out at the client, {len(err)} errors {[e[0] for e in err[:3]]}", flush=True)
if rates:
    print(f"  finished requests' decode speed: min {rates[0]:.1f}, median {rates[len(rates) // 2]:.1f} tok/s; completion "
          f"time median {sorted(r[1] for r in ok)[len(ok) // 2]:.0f} s", flush=True)
if agents:
    print(f"agents: first token max {max(x[1] for x in agents):.2f} s, decode min {min(x[2] for x in agents):.1f} tok/s",
          flush=True)
time.sleep(10)
t = time.time()
o = [None]
def probe():
    u = json.load(post({"model": model, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 200,
                        "temperature": 0}, 600))["usage"]
    o[0] = u.get("completion_tokens_per_sec") or 0
probe()
print(f"after the flood: a single request decoded at {o[0]:.1f} tok/s (abandoned requests still running would slow it)",
      flush=True)
