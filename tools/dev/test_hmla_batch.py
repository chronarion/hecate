"""hmla for decode batches (docker/patches/exllamav3/hmla/apply_hmla_batch.py): one launch over B sequences x q_len rows
must give, row for row, exactly the result of B one-sequence launches (the kernel's arithmetic per row is unchanged; only
the block-table row it reads moves with the row's sequence). Random Q6 cache pages and selections, short rows padded
with -1 as the DSA top-k pads them, shapes (B, q_len) with B * q_len <= 8.
   docker run --rm --device nvidia.com/gpu=<GB300> -v $PWD:/h --entrypoint python3 IMAGE /h/tools/dev/test_hmla_batch.py"""
import random, sys
import torch
from exllamav3.ext import exllamav3_ext as ext

if not hasattr(ext, "hmla_attn_batch"):
    sys.exit("this image has no hmla_attn_batch (apply_hmla_batch.py)")
torch.manual_seed(0)
rng = random.Random(1)
dev = torch.device("cuda", 0)
H, DC, DR, PAGE, PAGES, KPAD = 64, 512, 64, 256, 96, 2048
slots = PAGES * PAGE
ckv = torch.randint(-2**31, 2**31 - 1, (slots, 96), dtype=torch.int32, device=dev)
ckv_s = (torch.rand(slots, 16, device=dev) * 0.01 + 0.005).half()
kpe = (torch.randn(slots, DR, device=dev) * 0.1).half()
bad = cases = 0
for B, Q in [(2, 1), (2, 4), (3, 2), (4, 2), (5, 1), (8, 1), (6, 1), (2, 2)]:
    for trial in range(4):
        R = B * Q
        lens = [rng.randint(300, 20000) for _ in range(B)]
        W = max((L + Q + PAGE - 1) // PAGE for L in lens)
        bt = torch.zeros(B, W, dtype=torch.int32)
        for b in range(B):
            bt[b] = torch.tensor(rng.sample(range(PAGES), W), dtype=torch.int32)
        idx = torch.full((R, KPAD), -1, dtype=torch.int32)
        for r in range(R):
            n = lens[r // Q] + r % Q + 1
            sel = rng.sample(range(n), min(n, KPAD))
            idx[r, :len(sel)] = torch.tensor(sorted(sel), dtype=torch.int32)
        bt, idx = bt.to(dev), idx.to(dev)
        q_lat = (torch.randn(H, R, DC, device=dev) * 0.5).half()
        q_pe = (torch.randn(R, H, DR, device=dev) * 0.5).half()
        for ns in (16, 8):
            out = torch.zeros(H, R, DC, dtype=torch.half, device=dev)
            ext.hmla_attn_batch(q_lat, q_pe, ckv, ckv_s, kpe, bt, idx, 0.0721, ns, out, B)
            ref = torch.zeros_like(out)
            for b in range(B):
                rows = slice(b * Q, (b + 1) * Q)
                o = torch.zeros(H, Q, DC, dtype=torch.half, device=dev)
                ext.hmla_attn_batch(q_lat[:, rows].contiguous(), q_pe[rows].contiguous(), ckv, ckv_s, kpe, bt[b:b + 1].contiguous(),
                                    idx[rows].contiguous(), 0.0721, ns, o, 1)
                ref[:, rows] = o
            torch.cuda.synchronize()
            cases += 1
            same = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            finite = bool(torch.isfinite(out.float()).all())
            if not same or not finite or out.abs().sum() == 0:
                bad += 1
                print(f"  B={B} q_len={Q} ns={ns} trial {trial}: {'' if same else 'differs '}"
                      f"{'' if finite else 'non-finite '}max |diff| {(out.float() - ref.float()).abs().max().item():.3e}", flush=True)
print(f"hmla batch vs one-sequence launches: {cases} cases, {bad} not bit-identical (or empty / non-finite)", flush=True)
sys.exit(1 if bad else 0)
