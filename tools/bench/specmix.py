"""Decode speed and draft acceptance on a mix of everyday workloads, at the sampling settings clients actually send.

The default bench (speed.py) is short and greedy. Real traffic is mostly sampled (TabbyAPI's default is temperature 1.0),
reasons before it answers, carries long contexts and writes tool-call arguments that copy from them. This mix covers:
  reason   short prompts that make the model think (math, planning, analysis)
  code     write or fix code from a short brief
  longctx  questions over a long document of Python sources (sparse-attention regime)
  agent    multi-turn tool use over a small repository (read_file / write_file / search), arguments copy file text
  edit     rewrite a given file with small changes (output mostly copies the input)
  editlow  the same edits at reasoning effort "low", so the rewritten file arrives within the token budget
  chat     24 everyday requests (explanations, stories, letters, translation, small code), reasoning effort "low"
Each case runs once per seed; the same seed on two servers samples the same text (up to float noise), so arms pair up.
Per request it records decode tok/s (server timing), generated tokens, and the draft counts from the usage block, which
give tokens per decode step = (generated - 1) / (generated - 1 - accepted).
   python3 -I specmix.py LABEL [--suite reason,code,longctx,agent,edit] [--seeds 1,2] [--temperature T] [--top-p P]
                               [--greedy] [--max-tokens N] [--ctx 64000] [--out DIR]"""
import argparse, glob, json, os, statistics, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8103/v1"))
ap.add_argument("--suite", default="reason,code,longctx,agent,edit")
ap.add_argument("--max-tokens-copy", type=int, default=None, help="token budget for agent / edit / editlow (default --max-tokens)")
ap.add_argument("--seeds", default="1,2")
ap.add_argument("--temperature", type=float, default=None, help="omit to use the server default (TabbyAPI: 1.0)")
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--greedy", action="store_true", help="temperature 0, one pass (seeds ignored)")
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--ctx", type=int, default=64000, help="long-context document size in tokens (approx.)")
ap.add_argument("--out", default=os.environ.get("OUT", "/data/scratch/hecate/specmix"))
ap.add_argument("--save-prompts", action="store_true", help="store each request's messages in the log (replays)")
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
SRC = sorted(glob.glob("/usr/lib/python3.12/**/*.py", recursive=True))

REASON = [
    "A train leaves city A at 9:00 at 80 km/h; another leaves city B, 500 km away, at 10:30 at 120 km/h towards A. "
    "Where and when do they meet? Then redo it if the second train stops for 20 minutes halfway. Show your reasoning.",
    "I have a 3-node Proxmox cluster and want to migrate 14 VMs to new hardware with at most 5 minutes of downtime per "
    "VM. Think through the options carefully and give me a step-by-step migration plan with rollback points.",
    "Is it possible to tile an 8x8 chessboard with two opposite corners removed using 1x2 dominoes? Prove your answer, "
    "then generalize to removing any two squares.",
    "Estimate how many piano tuners work in Chicago. State every assumption, give a range, and say which assumption the "
    "answer is most sensitive to.",
    "Compare consistent hashing and rendezvous hashing for a cache cluster that scales from 10 to 200 nodes. Which would "
    "you pick, and what would make you change your mind?",
    "Explain why the sum of the reciprocals of the primes diverges, at the level of an undergraduate who knows calculus.",
]
CODE = [
    "Write a Python module implementing an LRU cache with TTL expiry, thread safety and hit/miss statistics. Include "
    "type hints, docstrings and a small pytest suite.",
    "Write a Rust function that parses a CSV line with quoted fields (RFC 4180), with unit tests for the edge cases.",
    "Here is a buggy binary search. Find every bug, explain each, and give a corrected version with tests:\n\n"
    "def bsearch(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) / 2\n"
    "        if a[mid] < x:\n            lo = mid\n        elif a[mid] > x:\n            hi = mid - 1\n"
    "        else:\n            return mid\n    return -1\n",
    "Write a bash script that backs up a directory to a remote host with rsync, keeps the last seven daily copies, "
    "logs to syslog and exits non-zero on any failure. Explain each design choice briefly.",
]
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Overwrite a file with new content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the repository for a regex; returns "
     "matching lines as path:line:text.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}},
     "required": ["pattern"]}}},
]
REPO = {f"lib/{os.path.basename(p)}": p for p in ("/usr/lib/python3.12/textwrap.py", "/usr/lib/python3.12/shlex.py",
                                                  "/usr/lib/python3.12/fnmatch.py", "/usr/lib/python3.12/glob.py")}
AGENT = [
    "In lib/textwrap.py, add type hints to the public functions wrap, fill, shorten, dedent and indent, without "
    "changing behaviour. Read the file first, then write the full updated file back.",
    "Find where shell-style quoting of a single argument is implemented in this repository, read that file, and add a "
    "docstring example to that function. Write the full updated file back.",
]
EDIT = ["/usr/lib/python3.12/fnmatch.py", "/usr/lib/python3.12/shlex.py"]
CHAT = [
    "Explain how a hash map handles collisions, comparing chaining and open addressing.",
    "Describe the water cycle in detail for a middle-school science class.",
    "Summarize the causes and consequences of the French Revolution.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between TCP and UDP and when to use each.",
    "Give a step-by-step recipe for sourdough bread, including starter maintenance.",
    "Compare Keynesian and Austrian economics.",
    "Translate this into French and explain the grammar: 'The children who were playing outside came home late.'",
    "Write a SQL query that finds the top three customers by revenue per region, with an explanation.",
    "Explain how CRISPR gene editing works and its main ethical debates.",
    "Write a haiku sequence about the four seasons, then explain the imagery.",
    "Debug this JavaScript: for (var i = 0; i < 5; i++) setTimeout(() => console.log(i), 100);",
    "What are the main differences between Rust's ownership model and C++ RAII?",
    "Plan a three-day itinerary for Kyoto for someone interested in gardens and food.",
    "Write a formal complaint letter to a landlord about a broken heater.",
    "Describe the life cycle of a star like the Sun.",
    "Give advice for a first-time manager running one-on-one meetings.",
    "Explain the Monty Hall problem and why switching is better.",
    "用中文解释一下量子纠缠是什么，并举一个日常生活中的比喻。",
    "Escribe un correo breve en español pidiendo una prórroga para entregar un informe.",
    "Erkläre auf Deutsch, wie eine Wärmepumpe funktioniert.",
    "Write a limerick about a cat who learns to code, then a short prose version.",
    "What should I consider when choosing between renting and buying a home?",
    "Summarize the plot of Pride and Prejudice in one paragraph, then list the main themes.",
]


def doc(n_tokens, start):
    out, chars, pos = [], int(n_tokens * 3.3), start
    while sum(len(x) for x in out) < chars:
        try:
            out.append(f"# file: {SRC[pos % len(SRC)]}\n" + open(SRC[pos % len(SRC)]).read())
        except Exception:
            pass
        pos += 1
    return "\n\n".join(out)[:chars]


def ask(messages, seed, tools=None, max_tokens=None, effort=None):
    req = {"model": model, "messages": messages, "max_tokens": max_tokens or a.max_tokens, "stream": True,
           "stream_options": {"include_usage": True}}
    if effort:
        req["template_vars"] = {"reasoning_effort": effort}
    if a.greedy:
        req["temperature"] = 0
    else:
        req["seed"] = seed
        if a.temperature is not None:
            req["temperature"] = a.temperature
        if a.top_p is not None:
            req["top_p"] = a.top_p
    if tools:
        req["tools"] = tools
    r = urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = {}; rc, c, calls = [], [], {}
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                r_ = dl.get("reasoning_content") or dl.get("reasoning") or ""
                c_ = dl.get("content") or ""
                for tc in dl.get("tool_calls") or []:
                    e = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "args": ""})
                    f = tc.get("function") or {}
                    e["name"] += f.get("name") or ""
                    e["args"] += f.get("arguments") or ""
                    e["id"] = e["id"] or tc.get("id")
                if (r_ or c_ or dl.get("tool_calls")) and tf is None:
                    tf = time.time()
                rc.append(r_); c.append(c_)
    t1 = time.time()
    det = usage.get("completion_tokens_details") or {}
    gen = usage.get("completion_tokens", 0)
    acc, rej = det.get("accepted_prediction_tokens", 0) or 0, det.get("rejected_prediction_tokens", 0) or 0
    steps = max(1, gen - 1 - acc)
    return {"prompt": usage.get("prompt_tokens", 0), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "gen": gen, "acc": acc, "rej": rej, "tps": (gen - 1) / steps if gen > 1 else 0.0,
            "rate": usage.get("completion_tokens_per_sec") or 0.0, "ttft": (tf or t1) - t0,
            "client_rate": (gen - 1) / max(1e-6, t1 - (tf or t1)), "reasoning": "".join(rc), "content": "".join(c),
            "calls": [calls[k] for k in sorted(calls)]}


def run_tool(call):
    try:
        args = json.loads(call["args"] or "{}")
    except Exception:
        return "error: arguments are not valid JSON"
    if call["name"] == "read_file":
        p = REPO.get(args.get("path", "").lstrip("./"))
        return open(p).read() if p else f"error: no such file {args.get('path')}"
    if call["name"] == "write_file":
        return f"wrote {len(args.get('content', ''))} bytes to {args.get('path')}"
    if call["name"] == "search":
        import re
        try:
            rx = re.compile(args.get("pattern", ""))
        except re.error as e:
            return f"error: {e}"
        hits = [f"{k}:{i + 1}:{ln}" for k, p in REPO.items() for i, ln in enumerate(open(p).read().splitlines())
                if rx.search(ln)]
        return "\n".join(hits[:40]) or "no matches"
    return f"error: unknown tool {call['name']}"


os.makedirs(a.out, exist_ok=True)
log = open(f"{a.out}/{a.label}.jsonl", "a")
seeds = [0] if a.greedy else [int(s) for s in a.seeds.split(",")]
suites = a.suite.split(",")
rows = []


def record(kind, case, seed, res, turn=0, messages=None):
    rec = {k: v for k, v in res.items() if k not in ("reasoning", "content", "calls")}
    if a.save_prompts and messages is not None:
        rec["messages"] = messages
    rec.update(kind=kind, case=case, seed=seed, turn=turn, label=a.label,
               text_head=(res["reasoning"] + "|" + res["content"])[:200],
               text=res["reasoning"] + "\n=====\n" + res["content"] +
               ("\n=====\n" + json.dumps(res["calls"]) if res["calls"] else ""))
    log.write(json.dumps(rec) + "\n"); log.flush()
    rows.append(rec)
    print(f"{a.label} {kind:7s} {case:2d} seed {seed} turn {turn} ctx {res['prompt']:6d} gen {res['gen']:5d} "
          f"tok/step {res['tps']:.2f} acc {res['acc']}/{res['acc'] + res['rej']} rate {res['rate']:.1f} tok/s "
          f"ttft {res['ttft']:.2f}", flush=True)


for seed in seeds:
    if "reason" in suites:
        for i, p in enumerate(REASON):
            m = [{"role": "user", "content": p}]; record("reason", i, seed, ask(m, seed), 0, m)
    if "code" in suites:
        for i, p in enumerate(CODE):
            m = [{"role": "user", "content": p}]; record("code", i, seed, ask(m, seed), 0, m)
    if "longctx" in suites:
        body = doc(a.ctx, 900)
        for i, q in enumerate(["Pick the two most complex functions in the files above. For each, explain what it does, "
                               "list its edge cases, and rewrite it more clearly with tests.",
                               "Which of the files above has the weakest error handling? Quote the three worst spots "
                               "and propose fixes as unified diffs."]):
            m = [{"role": "user", "content": body + "\n\n" + q}]; record("longctx", i, seed, ask(m, seed), 0, m)
    if "agent" in suites:
        listing = "\n".join(sorted(REPO))
        for i, task in enumerate(AGENT):
            msgs = [{"role": "system", "content": "You are a coding agent working in a repository with these files:\n"
                     + listing + "\nUse the tools to do the task; when done, reply with a short summary."},
                    {"role": "user", "content": task}]
            for turn in range(4):
                res = ask(msgs, seed + 1000 * turn, TOOLS, a.max_tokens_copy)
                record("agent", i, seed, res, turn, json.loads(json.dumps(msgs)))
                if not res["calls"]:
                    break
                msgs.append({"role": "assistant", "content": res["content"] or None, "tool_calls": [
                    {"id": c["id"] or f"call_{turn}_{k}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["args"]}} for k, c in enumerate(res["calls"])]})
                for k, c in enumerate(res["calls"]):
                    msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{turn}_{k}", "content": run_tool(c)})
    if "chat" in suites:
        for i, p in enumerate(CHAT):
            m = [{"role": "user", "content": p}]; record("chat", i, seed, ask(m, seed, None, None, "low"), 0, m)
    for kind, effort in (("edit", None), ("editlow", "low")):
        if kind not in suites:
            continue
        for i, p in enumerate(EDIT):
            src = open(p).read()
            msgs = [{"role": "user", "content": f"Here is {os.path.basename(p)}:\n\n```python\n{src}\n```\n\nAdd type "
                     "hints to every function, rename the module-level helper functions to start with an underscore "
                     "where they are private, and output the complete updated file in one code block."}]
            record(kind, i, seed, ask(msgs, seed, None, a.max_tokens_copy, effort), 0, msgs)


def summ(rs):
    g = sum(r["gen"] for r in rs)
    steps = sum(max(1, r["gen"] - 1 - r["acc"]) for r in rs if r["gen"] > 1)
    t = sum((r["gen"] - 1) / r["rate"] for r in rs if r["rate"] and r["gen"] > 1)
    acc, prop = sum(r["acc"] for r in rs), sum(r["acc"] + r["rej"] for r in rs)
    return (f"gen {g:6d}  tok/step {(g - len(rs)) / max(1, steps):.3f}  accepted {acc / max(1, prop):.3f}  "
            f"decode {(g - len(rs)) / max(1e-9, t):6.1f} tok/s")


print(f"### {a.label} summary (token-weighted)", flush=True)
for kind in [k for k in ("reason", "code", "longctx", "agent", "edit", "editlow", "chat") if k in suites]:
    rs = [r for r in rows if r["kind"] == kind]
    if rs:
        print(f"{a.label} {kind:7s} {summ(rs)}", flush=True)
print(f"{a.label} ALL     {summ(rows)}", flush=True)
