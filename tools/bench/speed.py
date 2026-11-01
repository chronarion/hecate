"""Decode speed over 10 fixed prompts, greedy, 400 tokens (MTP acceptance varies with the text; the mean over the prompts
is the comparison):
   python3 -I speed.py LABEL"""
import json, sys, urllib.request, statistics
URL = "http://127.0.0.1:8103/v1"
model = json.load(urllib.request.urlopen(f"{URL}/models"))["data"][0]["id"]
prompts = ["Explain how a hash map handles collisions, comparing chaining and open addressing.",
           "Write a Python function that merges overlapping intervals and explain its complexity.",
           "Describe the water cycle in detail for a middle-school science class.",
           "Summarize the causes and consequences of the French Revolution in five paragraphs.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Explain the difference between TCP and UDP and when to use each.",
           "Give a step-by-step recipe for sourdough bread, including starter maintenance.",
           "Derive the formula for the sum of a geometric series and give two examples.",
           "Write a bash script that backs up a directory with rotation of the last seven copies.",
           "Compare the economic policies of Keynesian and Austrian schools of thought."]
rates = []
for p in prompts:
    req = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}
    body = json.dumps(req).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{URL}/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    rates.append(r["usage"]["completion_tokens_per_sec"])
print(f"{sys.argv[1]}: mean {statistics.mean(rates):.2f} tok/s, median {statistics.median(rates):.2f} over {len(rates)} prompts: " + " ".join(f"{x:.1f}" for x in rates))
