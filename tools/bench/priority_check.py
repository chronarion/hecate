"""Interactive priority under batch load: a client floods the server with non-streamed requests (more than
HECATE_MAX_BATCH), then a streamed request arrives. With HECATE_PRIORITY=stream it must start within a few decode steps
(the batch keeps HECATE_INTERACTIVE_RESERVE slots free) and decode at near its unloaded speed; every batch request must
still finish.   python3 -I priority_check.py [--batch 12] [--batch-tokens 800]"""
import argparse, json, os, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8103/v1"))
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--batch-tokens", type=int, default=800)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
done = []


def batch_req(i):
    req = {"model": model, "messages": [{"role": "user", "content": f"Write a long story number {i} about a lighthouse."}],
           "max_tokens": a.batch_tokens, "temperature": 0.8, "template_vars": {"reasoning_effort": "low"}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=3600))
    done.append((time.time() - t, r["usage"]["completion_tokens"]))


def stream_req():
    req = {"model": model, "messages": [{"role": "user", "content": "Explain how TCP congestion control works."}],
           "max_tokens": 300, "temperature": 0.8, "stream": True, "stream_options": {"include_usage": True},
           "template_vars": {"reasoning_effort": "low"}}
    t0 = time.time(); tf = None; n = 0
    with urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                       {"Content-Type": "application/json"}), timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")) and tf is None:
                    tf = time.time()
    t1 = time.time()
    return (tf or t1) - t0, (n - 1) / max(1e-6, t1 - (tf or t1)), n


alone = stream_req()
print(f"streamed request alone: first token {alone[0]:.2f} s, {alone[1]:.1f} tok/s", flush=True)
th = [threading.Thread(target=batch_req, args=(i,)) for i in range(a.batch)]
for t in th:
    t.start()
time.sleep(4)
loaded = stream_req()
print(f"streamed request with {a.batch} non-streamed requests in flight: first token {loaded[0]:.2f} s, "
      f"{loaded[1]:.1f} tok/s", flush=True)
for t in th:
    t.join()
lat = sorted(x[0] for x in done)
print(f"batch: {len(done)} of {a.batch} finished, completion time median {lat[len(lat) // 2]:.0f} s, max {lat[-1]:.0f} s",
      flush=True)
