"""Grace band jobs with many distinct cold picks (runtime/tier/cpuband.cpp): every pick must be computed. Until 1.9 the band
kept at most 8 distinct cold experts per job and skipped the 9th and later ones, so their rows lost those experts.
A job's partial is each row's fp32 sum, from 0, of its picks' outputs in pick order (first appearance over rows, then
top-k order). A one-pick job's partial is exactly that pick's output (0 + y), so the reference for a many-pick job is
the one-pick jobs of its picks, summed in the same order. Random cold experts (row-major int4, fp32 scales), jobs of 1-8
rows with 1-20 distinct cold picks among other (hot / RTX) selections.
   docker run --rm --network none -v $PWD:/h --entrypoint python3 IMAGE /h/tools/dev/test_grace_picks.py /h/runtime/tier"""
import os, random, sys, time
import torch
from torch.utils.cpp_extension import load

tier = sys.argv[1] if len(sys.argv) > 1 else "/opt/hecate/tier"
B = os.environ.get("EXL3_TIER_BUILD", "/tmp/b")
os.makedirs(f"{B}/build_cpuband_t", exist_ok=True)
CB = load(name="cpuband_t", sources=[f"{tier}/cpuband.cpp"], build_directory=f"{B}/build_cpuband_t", verbose=False,
          extra_cflags=["-O3", "-mcpu=neoverse-v2"])
H, I, N, LO = 6144, 2048, 24, 144
torch.manual_seed(5)
ex = []
for e in range(N):
    g = torch.randint(0, 256, (I, H // 2), dtype=torch.uint8); u = torch.randint(0, 256, (I, H // 2), dtype=torch.uint8)
    d = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8)
    ex.append((g, u, d, torch.rand(I) * 0.01 + 1e-3, torch.rand(I) * 0.01 + 1e-3, torch.rand(H) * 0.01 + 1e-3))
CB.add_layer(0, LO, -1, [x[0] for x in ex], [x[1] for x in ex], [x[2] for x in ex], [x[3] for x in ex], [x[4] for x in ex], [x[5] for x in ex])
flags = torch.zeros(64, dtype=torch.int32)
xs = torch.zeros(8, H, dtype=torch.half); sel = torch.zeros(8, 8, dtype=torch.long); rw = torch.zeros(8, 8, dtype=torch.half)
out = torch.zeros(8, H)
fp = flags.data_ptr()
CB.start(fp, fp + 64, fp + 128, xs.data_ptr(), sel.data_ptr(), rw.data_ptr(), out.data_ptr(),
         int(os.environ.get("CB_THREADS", "8")), int(os.environ.get("CB_CORE0", "4")), 0, 0, 0, 0, 0)
time.sleep(0.2)
gen = 0


def job(R):
    global gen
    flags[16] = 0; flags[17] = R; gen += 1; flags[0] = gen
    t0 = time.perf_counter()
    while int(flags[32]) != gen:
        if time.perf_counter() - t0 > 30:
            raise RuntimeError("band job timed out")
    return out[:R].clone()


rng = random.Random(3)
bad = cases = over8 = 0
OTHER = [e for e in range(256) if not (LO <= e < LO + N)]
for trial in range(40):
    R = rng.randint(1, 8)
    k = rng.randint(1, min(20, 8 * R))
    cold = rng.sample(range(N), k)
    S = [[] for _ in range(R)]
    for i, e in enumerate(cold):                              # every pick in some row, <= 8 per row
        S[i % R].append(LO + e)
    for r in range(R):
        S[r] += rng.sample(OTHER, 8 - len(S[r]))
        rng.shuffle(S[r])
    W = [(torch.rand(8) * 0.4 + 0.01).half() for _ in range(R)]
    x = (torch.randn(R, H) * rng.choice([0.3, 1.0, 4.0])).half()
    xs[:R] = x
    for r in range(R):
        sel[r] = torch.tensor(S[r]); rw[r] = W[r]
    got = job(R)
    order = []
    for r in range(R):
        for e in S[r]:
            if LO <= e < LO + N and e not in order:
                order.append(e)
    ref = torch.zeros(R, H)
    for e in order:                                           # the one-pick job of e, then add in pick order
        sel[:R] = OTHER[0]; rw[:R] = 0
        for r in range(R):
            if e in S[r]:
                j = S[r].index(e); sel[r, 0] = e; rw[r, 0] = W[r][j]
        ref = ref + job(R)
    cases += 1; over8 += len(order) > 8
    if not torch.equal(ref.view(torch.int32), got.view(torch.int32)):
        bad += 1
        print(f"  trial {trial}: R={R}, {len(order)} distinct cold picks: "
              f"{int((ref.view(torch.int32) != got.view(torch.int32)).sum())} values differ", flush=True)
print(f"Grace band many-pick jobs vs their one-pick jobs summed in order: {cases} jobs ({over8} with more than 8 distinct "
      f"cold picks), {bad} not bit-identical", flush=True)
CB.stop()
sys.exit(1 if bad else 0)
