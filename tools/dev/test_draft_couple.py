"""Coupled drafts (EXL3_DRAFT_COUPLE): the int4 draft head must add exactly the Gumbel noise ExLlamaV3's sampler adds.

With the draft input at zero every draft logit is 0, so the head's pick is the argmax of its noise alone; it must equal
the sampler's pick on all-zero logits for the same random word (fused tail, the TabbyAPI default stack, and the unfused
gumbel_noise_f32 path), for every batch row and draft pass. Rows with inverse temperature 0 must stay plain argmax.
A last check measures the match rate the coupling gives on a pair of close distributions.
   docker run --rm --device nvidia.com/gpu=<GB300> -v $PWD/runtime/core:/opt/hecate/core -e EXL3_DRAFT_COUPLE=1 \
     -e EXL3_TIER_BUILD=/tmp/b --entrypoint python3 hecate:latest /opt/hecate/core/../../tools/dev/test_draft_couple.py"""
import os
import random
import sys

import torch

sys.path.insert(0, "/opt/hecate/core")
os.environ.setdefault("EXL3_DRAFT_COUPLE", "1")
import exl3_draft_head as dh                                         # noqa: E402
from exllamav3.ext import exllamav3_ext as ext                       # noqa: E402
from exllamav3.generator.sampler import ComboSampler                  # noqa: E402

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
N, K, V = 154880, 128, 154856
w = torch.randint(0, 256, (N * K // 2,), dtype=torch.uint8, device=dev)
scale = torch.rand(N, device=dev)
keys = torch.zeros(8, dtype=torch.int64, device=dev)
ids = torch.zeros(8, dtype=torch.int64, device=dev)
mod = dh._ext_mod()


def head(M, step, rows):
    dh.stage(rows, dev)
    x = torch.zeros((M, K), dtype=torch.half, device=dev)
    out = torch.empty(M, dtype=torch.int64, device=dev)
    mod.head_i4_argmax(w, scale, x, keys, out, N, V, dh._nz(dev)["dev"], step)
    return out.tolist()


fused = ComboSampler(temperature=1.0)                  # the default request: T 1, no truncation -> fused Gumbel tail
assert type(fused.steps[-1]).__name__ == "SS_Fused", [type(s).__name__ for s in fused.steps]


def target(r):
    z = torch.zeros((1, 1, V), dtype=torch.float, device=dev)
    a = int(fused.forward(z.clone(), rand_u32=r).view(-1)[0])
    g = torch.empty_like(z.view(1, V))
    ext.gumbel_noise_f32(z.view(1, V), g, r)
    b = int(torch.argmax(g, dim=-1)[0])
    return a, b


rng = random.Random(7)
bad = 0
for trial in range(64):
    M = rng.randint(1, 8)
    step = rng.randint(0, 7)
    words = [[rng.randint(0, (1 << 32) - 1) for _ in range(8)] for _ in range(M)]
    temps = [0.0 if rng.random() < 0.2 else 1.0 for _ in range(M)]
    rows = [(words[b], temps[b]) if temps[b] > 0 else None for b in range(M)]
    got = head(M, step, rows)
    for b in range(M):
        if temps[b] == 0:
            want = 0                      # all logits tie at 0: the lowest id wins, as before
        else:
            a, g = target(words[b][step])
            assert a == g, ("fused vs gumbel_noise_f32 disagree", a, g)
            want = a
        if got[b] != want:
            bad += 1
            print(f"MISMATCH trial {trial} row {b} step {step}: head {got[b]} target {want}", flush=True)
print(f"noise identity: {64} trials, {bad} mismatches", flush=True)

# match rate on close distributions: target l, draft l + noise (sd 0.3 nats), top mass spread over a few tokens
torch.manual_seed(0)
hits_c = hits_a = 0
n = 400
for t in range(n):
    l = torch.full((V,), -30.0, device=dev)
    top = torch.randint(0, V, (6,), device=dev)
    l[top] = torch.randn(6, device=dev) * 1.2
    q = l.clone(); q[top] += torch.randn(6, device=dev) * 0.3
    r = rng.randint(0, (1 << 32) - 1)
    tz = l.view(1, 1, V).clone()
    tgt = int(fused.forward(tz, rand_u32=r).view(-1)[0])
    g = torch.empty((1, V), device=dev)
    ext.gumbel_noise_f32(q.view(1, V), g, r)
    hits_c += int(torch.argmax(g, dim=-1)[0]) == tgt
    hits_a += int(torch.argmax(q)) == tgt
print(f"match rate on close distributions: coupled {hits_c / n:.3f}, argmax draft {hits_a / n:.3f}", flush=True)
sys.exit(1 if bad else 0)
