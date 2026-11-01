"""Base model folder of the Hecate GLM-5.3 release, built from stock zai-org/GLM-5.3-BF16 only.

ExLlamaV3 loads a model folder in its EXL3 layout. In Hecate most linears never run from EXL3 weights: attention, shared
experts, dense MLPs, LM head run from int8 copies (the MTP eh_proj from fp16), and 171 of 256 routed experts per layer from int4.
The folder therefore holds:
  base.safetensors   every unquantized tensor the loader reads (embeddings, norms, router weight + correction bias,
                     DSA indexer wk / weights_proj / k_norm, kv_b_proj, MTP enorm / hnorm / shared_head norm, final norm),
                     cast from BF16 to the dtype of the EXL3 layout
  virtual.json       name -> {dtype, shape} of the EXL3 tensors that are never loaded (trellis / suh / svh / mul1 of the
                     int8- and int4-served linears); the runtime registers them as zero-memory placeholders
  config.json        stock config + EXL3 quantization_config; tokenizer, chat template, generation config, LICENSE
The warm RTX tier's trellis experts live in rtx-band/ (requant_rtx_band.py) and the int8 / int4 weights in int8/, int4/.
   python -I build_skeleton.py BF16_DIR TEMPLATE.json INT8_DIR OUT_DIR
TEMPLATE.json: {name: {dtype, shape}} of the EXL3 layout (tensor names and shapes only)."""
import sys, os, json, re, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file

src, tmpl_path, i8_dir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
T = json.load(open(tmpl_path))
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
QSUF = re.compile(r"\.(trellis|suh|svh|mul1|mcg)$")


def get(k):
    with safe_open(f"{src}/{idx[k]}", "pt", device="cpu") as f:
        return f.get_tensor(k)


# MTP eh_proj ships unquantized (fp16): its rows carry outliers at ~76x the row RMS, which per-row int8 cannot represent
# (17% relative error); it only runs in the draft passes (1 row), so fp16 costs ~15 us per step
FP16_LINEARS = ("model.layers.78.eh_proj",)

# 1. unquantized tensors, from BF16, in the layout's dtype
base = {}
for k in FP16_LINEARS:
    base[k + ".weight"] = get(k + ".weight").to(torch.float16).contiguous()
for k, v in T.items():
    if QSUF.search(k):
        continue
    t = get(k)
    assert list(t.shape) == v["shape"], f"{k}: stock {list(t.shape)} vs layout {v['shape']}"
    base[k] = t.to(DT[v["dtype"]]).contiguous()
save_file(base, f"{out}/base.safetensors", metadata={"format": "pt", "source": "zai-org/GLM-5.3-BF16"})
print(f"base.safetensors: {len(base)} tensors, {sum(t.numel() * t.element_size() for t in base.values()) / 1e9:.2f} GB")

# 2. virtual entries: every quantized tensor of the layout except the warm (RTX) band, whose real trellis ships in rtx-band/
place = json.load(open(os.environ.get("PLACEMENT", "placement/tier-144-85-far27-mtp256.json")))
band = set()
for L, p in place.items():
    if p.get("first", 256) < len(p["perm"]):
        band |= {f"model.layers.{L}.mlp.experts.{e}." for e in p["perm"][p["first"]:]}
EXP = re.compile(r"^(model\.layers\.\d+\.mlp\.experts\.\d+\.)")
virt = {k: v for k, v in T.items() if QSUF.search(k) and not ((m := EXP.match(k)) and m.group(1) in band)
        and not any(k.startswith(f + ".") for f in FP16_LINEARS)}
json.dump(virt, open(f"{out}/virtual.json", "w"))
print(f"virtual.json: {len(virt)} placeholder tensors (never loaded)")

# 3. config + tokenizer from the stock release
cfg = json.load(open(f"{src}/config.json"))
cfg["quantization_config"] = {"quant_method": "exl3", "version": "1.6.5", "bits": 3.0, "head_bits": 6, "codebook": "mul1",
                              "out_scales": "always", "hecate": {"int8": "int8/", "int4": "int4/", "warm_band": "rtx-band/",
                                                                 "virtual": "virtual.json"}}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "LICENSE"):
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
print("config + tokenizer copied")
