"""Extended QA against a running Hecate server (:8103), beyond qa.py's six known answers and api_check.py's API behaviour:
known answers (math, facts, code reading, Chinese), a generated function executed with tests in a network-less container,
JSON output, multi-turn recall, a long generation without loops, sampled decoding, concurrent requests (C = 4 and 8),
needle retrieval in ~30K and ~100K-token prompts (dense and sparse attention regimes), a malformed request followed by a
normal one, and GPU memory before / after. Greedy unless stated.   python3 -I qa_ext.py [--quick]"""
import json, re, subprocess, sys, threading, time, urllib.error, urllib.request

URL = "http://127.0.0.1:8103/v1"
QUICK = "--quick" in sys.argv
model = json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
results = []


def chat(messages, timeout=1800, **kw):
    body = {"model": model, "temperature": 0, "max_tokens": 1500, **kw, "messages": messages}
    req = urllib.request.Request(f"{URL}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), r["usage"], r["choices"][0].get("finish_reason")


def ask(q, **kw):
    return chat([{"role": "user", "content": q}], **kw)


def check(name, ok, info=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {info}", flush=True)


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        return {}


mem0 = gpu_mem()
t_start = time.time()

# 1. known answers
KNOWN = [
    ("What is 123 * 47? Answer with the number only.", ["5781"]),
    ("What is the square root of 1764? Number only.", ["42"]),
    ("How many days are in a leap year? Number only.", ["366"]),
    ("Which planet is closest to the Sun? One word.", ["mercury"]),
    ("What is the largest ocean on Earth? Two words at most.", ["pacific"]),
    ("In which year did the first human land on the Moon? Year only.", ["1969"]),
    ("What is the boiling point of water at sea level in degrees Celsius? Number only.", ["100"]),
    ("What does `len([3, 1, 4, 1, 5])` return in Python? Number only.", ["5"]),
    ("What does `sorted('banana')[0]` evaluate to in Python? Answer with the character only.", ["a"]),
    ("What is the time complexity of binary search? Big-O only.", ["log"]),
    ("中国的首都是哪座城市？只回答城市名。", ["北京"]),
    ("一年有几个月？只回答数字。", ["12", "十二"]),
    ("Translate to French: 'Thank you very much'. Translation only.", ["merci"]),
    ("If a train travels 60 km in 45 minutes, what is its speed in km/h? Number only.", ["80"]),
]
ok = 0
for q, ans in KNOWN:
    c, _, _ = ask(q)
    hit = any(a in c.lower() for a in ans)
    ok += hit
    if not hit:
        print(f"      miss: {q[:50]!r} -> {c.strip()[:60]!r}", flush=True)
check("known answers", ok >= len(KNOWN) - 1, f"{ok}/{len(KNOWN)}")

# 2. code: a generated function run against tests in a container without network
c, u, fr = ask("Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals given as a "
               "list of [start, end] pairs and returns them sorted by start. Reply with one ```python code block only.",
               max_tokens=8000)          # the reasoning comes first and can run past 2,500 tokens
m = re.search(r"```(?:python)?\n(.*?)```", c, re.S)
if not c.strip():
    print(f"      no answer text: {u['completion_tokens']} tokens, finish {fr} (all reasoning)", flush=True)
code = (m.group(1) if m else c) + """
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
assert merge_intervals([]) == []
print("tests ok")
"""
try:
    r = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", "--memory", "512m", "python:3.12-slim", "python3", "-"],
                       input=code, capture_output=True, text=True, timeout=180)
    check("generated code passes tests", "tests ok" in r.stdout, (r.stdout + r.stderr).strip()[-80:])
except Exception as e:
    check("generated code passes tests", False, repr(e)[:80])

# 3. JSON output
c, _, _ = ask('Return a JSON object with keys "name" (string), "age" (integer) and "tags" (list of two strings) describing '
              'a fictional person. Output the JSON only, no code fence.', max_tokens=800)
try:
    j = json.loads(c.strip().strip("`").removeprefix("json").strip())
    check("JSON output parses", isinstance(j.get("name"), str) and isinstance(j.get("age"), int) and len(j.get("tags", [])) == 2, str(j)[:60])
except Exception as e:
    check("JSON output parses", False, f"{e!r} {c[:60]!r}")

# 4. multi-turn recall
msgs = [{"role": "user", "content": "My favourite colour is teal and my cat is called Biscuit. Reply 'noted'."}]
c1, _, _ = chat(msgs, max_tokens=300)
msgs += [{"role": "assistant", "content": c1}, {"role": "user", "content": "What is 9 + 10? Number only."}]
c2, _, _ = chat(msgs, max_tokens=300)
msgs += [{"role": "assistant", "content": c2}, {"role": "user", "content": "What is my cat called, and my favourite colour? Short answer."}]
c3, _, _ = chat(msgs, max_tokens=300)
check("multi-turn recall", "biscuit" in c3.lower() and "teal" in c3.lower() and "19" in c2, repr(c3.strip()[:60]))

# 5. long generation: exact length or a clean stop, and no degenerate loop at the end
c, u, fr = ask("Write a detailed, multi-section technical guide to designing a distributed key-value store: replication, "
               "consistency models, partitioning, failure detection, compaction and operations.", max_tokens=3000)
tail = c[-600:]
loop = any(tail.count(tail[i:i + 40]) >= 6 for i in range(0, max(1, len(tail) - 40), 40)) if len(tail) > 200 else False
check("long generation (3000 tokens)", (u["completion_tokens"] == 3000 or fr == "stop") and u["completion_tokens"] > 1000 and not loop,
      f"{u['completion_tokens']} tokens, finish {fr}, loop {loop}")

# 6. sampled decoding (temperature 0.7, top_p 0.95)
ok = 0
for q, a in (("What is the capital of Japan? One word.", "tokyo"), ("What is 15 + 27? Number only.", "42"),
             ("Who painted the Mona Lisa? Name only.", "vinci")):
    c, _, _ = ask(q, temperature=0.7, top_p=0.95, max_tokens=800)
    ok += a in c.lower()
check("sampled decoding (T 0.7)", ok >= 2, f"{ok}/3")

# 7. concurrency: C = 4 and 8 distinct known-answer requests at once
CQ = [("What is 7 * 8? Number only.", "56"), ("Capital of Italy? One word.", "rome"), ("What is 100 / 4? Number only.", "25"),
      ("Chemical symbol for sodium? Symbol only.", "na"), ("How many legs does a spider have? Number only.", "8"),
      ("What is 2 ** 8? Number only.", "256"), ("Capital of Canada? One word.", "ottawa"), ("What is 13 * 3? Number only.", "39")]
for C in (4, 8):
    out = {}
    def w(i):
        try:
            out[i] = ask(CQ[i][0], max_tokens=800)[0]
        except Exception as e:
            out[i] = f"ERROR {e!r}"
    th = [threading.Thread(target=w, args=(i,)) for i in range(C)]
    [t.start() for t in th]; [t.join() for t in th]
    good = sum(CQ[i][1] in out[i].lower() for i in range(C))
    check(f"concurrent requests C={C}", good >= C - 1, f"{good}/{C}")

# 8. needle retrieval across the dense / sparse attention regimes
def needle(n_lines, depth, code):
    lines = [f"Record {i}: the shipment of {i * 7 % 997} crates left warehouse {i % 41} on schedule." for i in range(n_lines)]
    lines.insert(int(n_lines * depth), f"Important: the secret access code is {code}.")
    return "\n".join(lines) + "\n\nWhat is the secret access code mentioned in the text above? Answer with the number only."
for n_lines, depth, code, label in ((1600, 0.37, 48213, "needle ~30K tokens"), (5400, 0.61, 90377, "needle ~100K tokens")):
    if QUICK and n_lines > 2000:
        continue
    t0 = time.time()
    c, u, _ = ask(needle(n_lines, depth, code), max_tokens=600)
    check(label, str(code) in c, f"{u['prompt_tokens']} prompt tokens, {time.time() - t0:.0f} s, {c.strip()[:30]!r}")

# 9. malformed request, then a normal one
try:
    req = urllib.request.Request(f"{URL}/chat/completions", b'{"model": "x", "messages": [', {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60); bad = "accepted"
except urllib.error.HTTPError as e:
    bad = e.code
except Exception as e:
    bad = repr(e)[:40]
c, _, _ = ask("What is 6 * 7? Number only.", max_tokens=600)
check("malformed request rejected, next ok", isinstance(bad, int) and 400 <= bad < 500 and "42" in c, f"status {bad}")

# 10. memory after the suite
mem1 = gpu_mem()
growth = {k: mem1.get(k, 0) - mem0.get(k, 0) for k in mem0}
check("GPU memory stable", all(abs(v) < 2048 for v in growth.values()), f"MiB change {growth} (caching allocator)")

n_ok = sum(ok for _, ok in results)
print(f"qa_ext: {n_ok}/{len(results)} passed in {time.time() - t_start:.0f} s"
      + ("" if n_ok == len(results) else "; FAILED: " + ", ".join(n for n, ok in results if not ok)), flush=True)
