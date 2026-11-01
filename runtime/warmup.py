"""Startup warmup for the TabbyAPI GLM-5.3 EXL3 server (runs in the container next to main.py).
The first generation after start runs at about half speed (one-time kernel/cache setup); one throwaway
request takes that cost before real traffic."""
import json, time, urllib.request

URL = "http://127.0.0.1:8103/v1"
for _ in range(1800):
    try:
        model = json.load(urllib.request.urlopen(f"{URL}/models", timeout=3))["data"][0]["id"]; break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("warmup: server never came up")
t = time.time()
body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Explain, step by step, how a compiler turns source code into machine code."}],
                   "max_tokens": 300, "temperature": 0}).encode()
try:
    u = json.load(urllib.request.urlopen(urllib.request.Request(f"{URL}/chat/completions", body,
                                         {"Content-Type": "application/json"}), timeout=900))["usage"]
    print(f"warmup: {u['completion_tokens']} tokens in {time.time() - t:.1f} s", flush=True)
except Exception as e:  # warmup must never take the server down
    print(f"warmup: request failed: {e}", flush=True)
