"""moe4 built for sm_120a (RTX) vs sm_103a (GB300) vs a torch fp32 reference on real far-band int4 experts of one layer.
Run in the served image with both GPUs: python3 r4test.py LAYER K"""
import json, os, sys, torch
from safetensors import safe_open
from torch.utils.cpp_extension import load
L, K = int(sys.argv[1]), int(sys.argv[2])
SRC = "/opt/moe4/moe4.cu"; B = "/work/build"
os.makedirs(f"{B}_g", exist_ok=True); os.makedirs(f"{B}_r", exist_ok=True)
mg = load(name="moe4", sources=[SRC], build_directory=f"{B}_g", extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a", "--use_fast_math"])
mr = load(name="moe4r", sources=[SRC], build_directory=f"{B}_r", extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"])
devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
names = [torch.cuda.get_device_name(d) for d in devs]
G = devs[[i for i, n in enumerate(names) if "GB300" in n][0]]; R = devs[[i for i, n in enumerate(names) if "RTX" in n][0]]
print("devices", names, "GB300", G, "RTX", R, flush=True)
P = json.load(open("/opt/rtx/tier-144-85-far27-mtp256.json"))[str(L)]
perm, hot = P["perm"], P["hot"]
import glob
fs = [safe_open(f, "pt", device="cpu") for f in sorted(glob.glob(f"/opt/u4/layer{L}.safetensors") + glob.glob(f"/opt/u4/layer{L}.r*.safetensors"))]
where = {k: f for f in fs for k in f.keys()}
H, I = 6144, 2048
ex = [{(m, t): where[f"{e}.{m}.{t}"].get_tensor(f"{e}.{m}.{t}") for m in ("gate", "up", "down") for t in ("q", "s")} for e in perm[hot:hot + K]]

def tabs(dev):
    keep, tab = [], {}
    for n, (m, t) in (("gate_w", ("gate", "q")), ("gate_s", ("gate", "s")), ("up_w", ("up", "q")), ("up_s", ("up", "s")),
                      ("down_w", ("down", "q")), ("down_s", ("down", "s"))):
        ts = [x[(m, t)].to(dev) for x in ex]; keep += ts
        tab[n] = torch.tensor([t_.data_ptr() for t_ in ts], dtype=torch.int64, device=dev)
    return keep, tab

torch.manual_seed(0)
worst = 0.0
for rows in (1, 3, 4, 5, 8):
    x = (torch.randn(rows, H) * 0.5).half()
    sel = torch.randint(0, 256, (rows, 8))
    for r in range(rows):                       # make sure the band gets picks: 1-3 per row in [hot, hot + K)
        for j in range(1 + r % 3):
            sel[r, j] = hot + (r + j) % K
    rw = torch.rand(rows, 8).half()
    # reference (fp32, dequantized with the kernel's own dequant on the GB300)
    ref = torch.zeros(rows, H)
    xf = x.float()
    for e in range(K):
        Wg = mg.dequant(ex[e][("gate", "q")].to(G), ex[e][("gate", "s")].to(G), I, H).float().cpu()
        Wu = mg.dequant(ex[e][("up", "q")].to(G), ex[e][("up", "s")].to(G), I, H).float().cpu()
        Wd = mg.dequant(ex[e][("down", "q")].to(G), ex[e][("down", "s")].to(G), H, I).float().cpu()
        for r in range(rows):
            w = sum(float(rw[r, j]) for j in range(8) if int(sel[r, j]) == hot + e)
            if w == 0: continue
            g = Wg @ xf[r]; u = Wu @ xf[r]
            ref[r] += w * (Wd @ (torch.nn.functional.silu(g) * u))
    out = {}
    for name, ext, dev in (("gb300", mg, G), ("rtx", mr, R)):
        keep, tab = tabs(dev)
        scr = ext.scratch(dev)
        y = torch.zeros(rows, H, device=dev)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        with torch.cuda.device(dev):
            for it in range(3):                 # repeated calls: scratch must self-clean
                y.zero_()
                ext.forward(x.to(dev), sel.to(dev).contiguous(), rw.to(dev), K, tab["gate_w"], tab["up_w"], tab["down_w"],
                            tab["gate_s"], tab["up_s"], tab["down_s"], *scr, y, sms * 2, sms * 3, hot)
            torch.cuda.synchronize(dev)
        out[name] = y.cpu()
        clean = all(float(t.float().abs().sum()) == 0 for t in scr[2:3] + scr[4:5])
        err = float((out[name] - ref).norm() / ref.norm().clamp_min(1e-9))
        worst = max(worst, err)
        print(f"rows {rows} {name:5s}: rel err vs fp32 ref {err:.2e}, counters clean {clean}", flush=True)
    print(f"rows {rows} rtx vs gb300 rel diff {float((out['rtx'] - out['gb300']).norm() / out['gb300'].norm()):.2e}", flush=True)
print("WORST", f"{worst:.2e}", "PASS" if worst < 2e-3 else "FAIL")
