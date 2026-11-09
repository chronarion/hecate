"""System-RAM KV tier check for the Hecate server.

One long document (~DOC tokens of Python/C sources) with NEEDLES vault codes planted at even depths. Phases:
  cold     first question on the document: full prefill
  warm     more questions: the document's pages are still in the GPU pool (prefix hit)
  flood    distinct long prompts (FLOOD tokens in total, max_tokens 1) push the document out of the GPU pool
  return   more questions: with the tier on, the pages come back from system RAM; with it off, full prefill again
Every answer is checked against the planted code; TTFT and decode speed are logged per request.
   python3 -I kvtier.py LABEL     (env: DOC, NEEDLES, FLOOD, FLOOD_EACH, START, MODEL, URL, OUT)
FLOOD counts estimated tokens (~0.75 of them are real); make it well above the GPU pool (HECATE_CACHE_TOKENS)"""
import glob, json, os, random, sys, time, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8103/v1")
DOC = int(os.environ.get("DOC", "100000"))
NEEDLES = int(os.environ.get("NEEDLES", "16"))
FLOOD = int(os.environ.get("FLOOD", "600000"))
FLOOD_EACH = int(os.environ.get("FLOOD_EACH", "120000"))
OUT = os.environ.get("OUT", "kvtier-results")
label = sys.argv[1] if len(sys.argv) > 1 else "run"
model = os.environ.get("MODEL") or json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
files = sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)) + \
    sorted(glob.glob("/usr/include/**/*.h", recursive=True))
pos = int(os.environ.get("START", "0"))


def text(n_tokens):
    global pos
    out, chars = [], int(n_tokens * 3.3)
    while sum(len(x) for x in out) < chars:
        try: out.append(f"# file: {files[pos % len(files)]}\n" + open(files[pos % len(files)]).read())
        except Exception: pass
        pos += 1
    return "\n\n".join(out)[:chars]


rng = random.Random(1234)
names = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel",
         "lagoon", "marble", "nimbus", "onyx", "pewter", "quartz", "rowan", "sierra", "tundra"][:NEEDLES]
codes = {n: rng.randint(100000, 999999) for n in names}
body = text(DOC)
step = len(body) // (NEEDLES + 1)
parts, at = [], 0
for i, n in enumerate(names):
    cut = body.rfind("\n", 0, (i + 1) * step) + 1 or (i + 1) * step
    parts += [body[at:cut], f"# NOTE: the access code for vault {n} is {codes[n]}.\n"]
    at = cut
parts.append(body[at:])
doc = "".join(parts)


def ask(content, max_tokens=600):
    req = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
           "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
           "template_vars": {"reasoning_effort": "low"}}
    r = urllib.request.Request(f"{URL}/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = {}; rc, c = [], []
    with urllib.request.urlopen(r, timeout=7200) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                r_ = dl.get("reasoning_content") or dl.get("reasoning") or ""
                c_ = dl.get("content") or ""
                if (r_ or c_) and tf is None: tf = time.time()
                rc.append(r_); c.append(c_)
    t1 = time.time()
    tf = tf or t1
    ct = usage.get("completion_tokens", 0)
    return {"ttft": tf - t0, "decode": (ct - 1) / max(1e-6, t1 - tf), "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": ct, "reasoning": "".join(rc), "content": "".join(c), "total": t1 - t0}


os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/{label}.jsonl", "w")
fails = 0


def question(phase, n):
    global fails
    q = doc + f"\n\nWhat is the access code for vault {n}? Answer with the number only."
    res = ask(q)
    ok = str(codes[n]) in res["content"]
    fails += not ok
    res.update(phase=phase, vault=n, code=codes[n], ok=ok)
    log.write(json.dumps(res) + "\n"); log.flush()
    print(f"{label} {phase:6s} vault {n:8s} ctx {res['prompt_tokens']:6d} ttft {res['ttft']:6.2f} s  "
          f"decode {res['decode']:5.1f} tok/s  {'ok' if ok else 'WRONG'}  {res['content'].strip()[:40]!r}", flush=True)


order = names[:]
rng.shuffle(order)
k = max(1, (NEEDLES - 1) // 2)
question("cold", order[0])
for n in order[1:1 + k]:
    question("warm", n)
t0 = time.time(); left = FLOOD; i = 0
while left > 0:
    n = min(FLOOD_EACH, left)
    res = ask(text(n) + "\n\nReply with OK.", max_tokens=1)
    print(f"{label} flood  {i} ctx {res['prompt_tokens']} in {res['total']:.1f} s", flush=True)
    log.write(json.dumps({"phase": "flood", **{k_: res[k_] for k_ in ("prompt_tokens", "total")}}) + "\n"); log.flush()
    left -= n; i += 1
print(f"{label} flood total {time.time() - t0:.0f} s", flush=True)
for n in order[1 + k:]:
    question("return", n)
print(f"{label} done: {fails} wrong of {NEEDLES}", flush=True)
