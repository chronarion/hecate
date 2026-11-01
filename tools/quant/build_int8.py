"""int8 copies of GLM-5.3's non-expert linears straight from stock BF16, in the fragment order of Hecate's int8 GEMV
kernels (dense_i8.cu): per-row scale with an MSE-optimal clip (absmax x f, f searched), codes stored as q + 128.
One file per layer: OUT/{layer}.safetensors (OUT/lm_head.safetensors for the head) with "{key}.q" (uint8, packed),
"{key}.s" (fp32 [N]) and "{key}.spw" (int32). Existing files are updated in place (keys added or replaced).
   python -I build_int8.py BF16_DIR OUT [REGEX]
REGEX (default: the served set) selects linears by tensor name without the ".weight" suffix."""
import sys, os, json, re, torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

src, out = sys.argv[1], sys.argv[2]
RX = re.compile(sys.argv[3] if len(sys.argv) > 3 else
                r"^model\.layers\.\d+\.(self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj|indexer\.wq_b)|"
                r"mlp\.shared_experts\.(gate|up|down)_proj|eh_proj)$|^model\.layers\.[0-2]\.mlp\.(gate|up|down)_proj$|^lm_head$")
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
os.makedirs(out, exist_ok=True)
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
keys = [k[:-len(".weight")] for k in idx if k.endswith(".weight") and RX.match(k[:-len(".weight")])]
F = torch.tensor([1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79])


def quant_i8(W):
    """[N, K] fp32 -> fragment-ordered uint8 codes, fp32 row scales, strips per warp, relative weight error"""
    N, K = W.shape
    a = W.abs().amax(1).clamp_min(1e-12)
    best = None
    for f in F:
        s = a * f / 127.0
        e = ((torch.round(W / s[:, None]).clamp_(-128, 127) * s[:, None]) - W).pow(2).sum(1)
        best = (e, s) if best is None else (torch.where(e < best[0], e, best[0]), torch.where(e < best[0], s, best[1]))
    s = best[1]
    q = (torch.round(W / s[:, None]).clamp_(-128, 127) + 128).to(torch.uint8)
    rel = ((q.float() - 128) * s[:, None] - W).norm() / W.norm()
    spw = 4 if (N // 16) % 4 == 0 else 2 if (N // 16) % 2 == 0 else 1
    q = q.view(N // (16 * spw), spw, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 8, 2).contiguous().view(-1)
    return q, s.float().contiguous(), spw, rel.item()


by = {}
for k in keys:
    by.setdefault(k.split(".")[2] if k.startswith("model.layers") else "lm_head", []).append(k)
for layer, ks in by.items():
    path = f"{out}/{layer}.safetensors"
    buf = load_file(path) if os.path.exists(path) else {}
    for k in ks:
        with safe_open(f"{src}/{idx[k + '.weight']}", "pt", device="cpu") as f:
            W = f.get_tensor(k + ".weight").float()
        q, s, spw, rel = quant_i8(W)
        buf[f"{k}.q"], buf[f"{k}.s"], buf[f"{k}.spw"] = q, s, torch.tensor([spw], dtype=torch.int32)
        print(f"{k}: [{W.shape[0]} x {W.shape[1]}] rel err {rel:.4f}", flush=True)
    save_file(buf, path + ".tmp"); os.replace(path + ".tmp", path)
