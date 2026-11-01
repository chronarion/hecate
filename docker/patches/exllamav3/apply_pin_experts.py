"""Cold experts in pinned Grace memory (hecate, ExLlamaV3 v1.6.5; needs apply_fp16_linear.py first). Runtime switch
EXL3_PIN_EXPERTS=/path/map.json ({"layer": [expert ids]}, unset = stock): those MoE layers load outside the slab arena;
after the weights land, the listed experts' trellis tensors move to pinned host memory behind zero-copy CUDA aliases
(pin_linears) and the layer's expert pointer tables are rebuilt (load_local), so the MoE kernels read cold experts in
place over NVLink-C2C. GLM-5.3 EXL3 on the GB300 alone with the 64 least-routed experts per layer (68 GB) pinned:
53.7 tok/s vs 52.2 for the GB300 + RTX split (10 prompts, MTP).
   python3 apply_pin_experts.py PKG_DIR"""
import pathlib, re, sys
p = pathlib.Path(sys.argv[1]) / "model/model_ls.py"
s = p.read_text()
HELPER = '# hecate: EXL3_PIN_EXPERTS=/path/map.json ({"layer": [expert ids]}) -- single-GPU study of experts streamed from Grace\n# memory: after a MoE layer loads, its listed experts move to pinned host memory behind zero-copy CUDA aliases (the\n# kernels read them over NVLink-C2C) and the layer\'s expert pointer tables are rebuilt\n_PIN_MAP = None\nif _os.environ.get("EXL3_PIN_EXPERTS"):\n    import json as _json\n    _PIN_MAP = {int(k): sorted(v) for k, v in _json.load(open(_os.environ["EXL3_PIN_EXPERTS"])).items()}\n_pin_bytes = [0]\n\n\ndef _layer_of(key):\n    parts = (key or "").split(".")\n    for i, p in enumerate(parts[:-1]):\n        if p == "layers" and parts[i + 1].isdigit():\n            return int(parts[i + 1])\n    return -1\n\n\ndef _pin_planned(module):\n    if _PIN_MAP is None:\n        return False\n    from ..modules.block_sparse_mlp import BlockSparseMLP\n    return any(isinstance(sm, BlockSparseMLP) and _layer_of(sm.key) in _PIN_MAP for sm in module)\n\n\ndef _pin_cold_experts(module):\n    if _PIN_MAP is None:\n        return\n    from ..modules.block_sparse_mlp import BlockSparseMLP\n    for sm in module:\n        if not isinstance(sm, BlockSparseMLP) or sm.device is None:\n            continue\n        cold = _PIN_MAP.get(_layer_of(sm.key))\n        if not cold:\n            continue\n        for e in cold:\n            for lin in ((sm.gates[e] if sm.gates else None), sm.ups[e], sm.downs[e]):\n                if lin is not None and lin.inner is not None:\n                    _pin_bytes[0] += lin.inner.trellis.numel() * lin.inner.trellis.element_size()\n                    lin.pin_linears()\n        sm.load_local()\n        torch.cuda.empty_cache()\n        print(f" -- pinned {len(cold)} experts of {sm.key}, {_pin_bytes[0] / 1e9:.1f} GB host-resident so far", flush=True)\n\n'
assert s.count("\nclass Model_LSMixin(ABC):") == 1
s = s.replace("\nclass Model_LSMixin(ABC):", "\n" + HELPER + "\nclass Model_LSMixin(ABC):")
s, n1 = re.subn(r"(?m)^(\s*)config\.stc\.begin_deferred_load\(arena = not pin\)$",
                r"\1config.stc.begin_deferred_load(arena = not (pin or _pin_planned(module)))", s)
s, n2 = re.subn(r"(?m)^(\s*)_fp16_linears\(module\)$", r"\1_fp16_linears(module)\n\1_pin_cold_experts(module)", s)
assert n1 == 2 and n2 == 2, (n1, n2)
p.write_text(s)
print("patched: EXL3_PIN_EXPERTS support")
