"""Known-answer QA against the EXL3 TabbyAPI server (:8103), same 6 questions as glm-nll.py."""
import json, urllib.request
URL = "http://127.0.0.1:8103/v1"
model = json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
QA = [("What is 17*23? Answer with the number only.", "391"), ("What is the capital of Australia? One word.", "canberra"),
      ("What is the chemical symbol for gold? Symbol only.", "au"), ("How many sides does a hexagon have? Number only.", "6"),
      ("Who wrote Pride and Prejudice? Name only.", "austen"), ("What is 2 to the power of 10? Number only.", "1024")]
ok = 0
for q, a in QA:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 1500, "temperature": 0}).encode()
    m = json.load(urllib.request.urlopen(urllib.request.Request(f"{URL}/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))["choices"][0]["message"]
    c = (m.get("content") or "").strip(); ok += a in c.lower()
    print(f"{'PASS' if a in c.lower() else 'FAIL'}  {q[:40]:40s} -> {c[:40]!r}")
print(f"QA {ok}/{len(QA)}")
