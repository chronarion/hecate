"""Startup warmup for the Hecate server (runs in the container next to main.py).
The first generation after start runs at about half speed (one-time kernel/cache setup), and the first prompt past the
model's 2,048-token sparse-attention threshold compiles the long-context kernels (~40 s with an empty ./cache). Two
throwaway requests, a short chat and a ~10K-token prompt, take those costs before real traffic."""
import json, os, time, urllib.request

URL = "http://127.0.0.1:8103/v1"
for _ in range(1800):
    try:
        model = json.load(urllib.request.urlopen(f"{URL}/models", timeout=3))["data"][0]["id"]; break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("warmup: server never came up")


def ask(content, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                       "temperature": 0}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(f"{URL}/chat/completions", body,
                                            {"Content-Type": "application/json"}), timeout=900))["usage"]


out = []
t = time.time()
try:  # warmup must never take the server down
    u = ask("Explain, step by step, how a compiler turns source code into machine code.", 300)
    out.append(f"{u['completion_tokens']} tokens in {time.time() - t:.1f} s")
except Exception as e:
    out.append(f"request failed: {e}")
n = min(10000, int(os.environ.get("HECATE_MAX_SEQ_LEN", "262144")) - 512)   # two prefill chunks at the default 8192
if n > 2600:
    t = time.time()
    try:
        log = " ".join(f"Entry {i}: the keeper logged the tide at {i % 24:02d}:00 and the wind from the {'NESW'[i % 4]}." for i in range(n // 20))
        u = ask(log + "\n\nHow many entries are in the log? Answer with the number only.", 32)
        out.append(f"long prompt ({u['prompt_tokens']} tokens) in {time.time() - t:.1f} s")
    except Exception as e:
        out.append(f"long prompt failed: {e}")
print("warmup: " + "; ".join(out), flush=True)
