"""Frozen stratified KLD corpus for GLM-5.3 (token IDs fixed here; reference and candidates consume the same IDs).
Run in the exllamav3 image:  python -I build_corpus.py /k/bf16 /k/corpus /k/lab_tokens_2048.json"""
import sys, json, hashlib, os
from tokenizers import Tokenizer
import jinja2
mdir, cdir, labf = sys.argv[1:4]
tok = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
raw = json.load(open(os.path.join(cdir, "raw.json")))
tmpl = jinja2.Environment(loader=jinja2.FileSystemLoader(mdir), extensions=["jinja2.ext.loopcontrols"]).get_template("chat_template.jinja")
def chat(msgs): return tmpl.render(messages=msgs, add_generation_prompt=False, reasoning_effort="high")
W = []
def add(stratum, source, ids, T=2048, score=2047):
    assert len(ids) >= T, (stratum, source, len(ids))
    ids = ids[:T]
    W.append(dict(id=f"{stratum}-{sum(w['stratum'] == stratum for w in W)}", stratum=stratum, source=source, ids=ids,
                  score_from=T - 1 - score, sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
# wiki: the lab's frozen window (rtx6kpro benchmarks/data/glm52-kld-tokens-2048.json) + three later wikitext-2 test windows
lab = json.load(open(labf)); add("wiki", "rtx6kpro glm52-kld-tokens-2048.json (wikitext-2 test, tokens 0-2047)", lab)
wt = enc("\n\n".join(raw["wikitext_test"]))
for off in (8192, 16384, 24576): add("wiki", f"wikitext-2-raw-v1 test tokens {off}-{off + 2047}", wt[off:])
# code: four files, first 2048 tokens each
for f in ("argparse.py", "typing.py", "exl3_gemm.cu", "cuda_runtime_api.h"):
    add("code", f"{f} (first 2048 tokens)", enc(open(os.path.join(cdir, f)).read()))
# chat: multi-turn conversations chained from no_robots test (human-written), rendered with the GLM-5.3 template
nr = [m for m in raw["no_robots_test"]]
def chain(convs, i0, need):
    msgs, i = [], i0
    while True:
        c = [m for m in convs[i] if m["role"] in ("user", "assistant")]; i += 1
        msgs += c
        ids = enc(chat(msgs))
        if len(ids) >= need: return ids, i
k = 0
for j in range(4):
    ids, k2 = chain(nr, k, 2048); add("chat", f"no_robots test rows {k}-{k2 - 1} as one conversation", ids); k = k2
# math: gsm8k test, worked solution as the assistant's reasoning, final answer as content
gs = raw["gsm8k_test"]; k = 0
for j in range(2):
    msgs = []
    while True:
        g = gs[k]; k += 1
        sol, ans = g["a"].rsplit("####", 1)
        msgs += [{"role": "user", "content": g["q"]}, {"role": "assistant", "content": f"<think>{sol.strip()}</think>The answer is {ans.strip()}."}]
        ids = enc(chat(msgs))
        if len(ids) >= 2048: break
    add("math", f"gsm8k test rows {k - len(msgs) // 2}-{k - 1}, solution as reasoning", ids)
# zh: Chinese Wikipedia articles (20231101.zh rows 50000+)
zt = [a["title"] + "\n\n" + a["text"] for a in raw["zhwiki"]]
ids = enc("\n\n".join(zt)); add("zh", "wikipedia 20231101.zh rows 50000+ (part 1)", ids)
add("zh", "wikipedia 20231101.zh rows 50000+ (part 2)", ids[len(ids) // 2:])
# long: 8192-token windows (past the DSA indexer's top-2048 span), scoring the last 2048 positions
add("long", "wikitext-2-raw-v1 test tokens 40960-49151", wt[40960:], T=8192, score=2048)
py = "\n\n".join(open(os.path.join(cdir, f)).read() for f in ("dataclasses.py", "functools.py", "inspect.py"))
add("long", "dataclasses.py + functools.py + inspect.py (first 8192 tokens)", enc(py), T=8192, score=2048)
for i, w in enumerate(W): w["partition"] = "analysis" if i % 2 == 0 else "qualification"
json.dump(W, open(os.path.join(cdir, "corpus.json"), "w"))
for w in W: print(f"{w['id']:8s} {w['partition']:13s} T={len(w['ids']):5d} scored={len(w['ids']) - 1 - w['score_from']:5d} {w['source']}")
print("windows", len(W), "scored positions", sum(len(w["ids"]) - 1 - w["score_from"] for w in W))
print("corpus sha256", hashlib.sha256(open(os.path.join(cdir, "corpus.json"), "rb").read()).hexdigest())
