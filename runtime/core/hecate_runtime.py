"""Hecate runtime module (imported before the model loads; the entrypoint imports it after tier_split and hecate_loader).

  int8 base       the int8 copies of the EXL3_I8_LINEARS set (attention projections, shared experts, dense MLPs, LM head)
                  come from stock BF16 (EXL3_I8_SRC, tools/quant/build_int8.py). Decode (<= 8 rows) runs them through the
                  cluster-reduced int8 GEMVs; prefill dequantizes them on the fly, so prompt and decode share one base.
                  Their EXL3 trellis is never loaded (EXL3_SKIP_TRELLIS=1).
  tensor override EXL3_TENSOR_OVERRIDE: extra tensor files or directories (':'-joined, later ones win), e.g. the warm
                  RTX tier's trellis experts.
  also installed  EXL3_ROUTER_FUSED (exl3_router), EXL3_DRAFT_HEAD_I4 (exl3_draft_head), EXL3_VERIFY_GRAPH /
                  EXL3_DRAFT_GRAPH (exl3_vgraph), EXL3_DRAFT_MAX_ROWS (row cap of the decode fast path)."""
import os, re
import torch
from safetensors import safe_open

OVERRIDE = os.environ.get("EXL3_TENSOR_OVERRIDE", "")   # extra tensor files / directories (":"-joined), e.g. the warm RTX band
I8_SRC = os.environ.get("EXL3_I8_SRC", "")             # directory of the stock int8 copies ({layer}.safetensors)
_stats = {"i8_src": 0, "i8_mismatch": []}


_files = {}
def _i8_tensors(key):
    if not I8_SRC:
        return None
    m = re.match(r"model\.layers\.(\d+)\.", key)
    name = m.group(1) if m else ("lm_head" if key == "lm_head" else None)
    if name is None:
        return None
    path = f"{I8_SRC}/{name}.safetensors"
    if path not in _files:
        _files[path] = safe_open(path, "pt", device="cpu") if os.path.exists(path) else None
    f = _files[path]
    if f is None or f"{key}.q" not in f.keys():
        return None
    return f.get_tensor(f"{key}.q"), f.get_tensor(f"{key}.s"), int(f.get_tensor(f"{key}.spw")[0])


def _codes(q, N, K, spw):
    """fragment order -> plain uint8 codes [N, K]"""
    return q.view(N // (16 * spw), K // 32, spw, 8, 4, 2, 2, 2, 2).permute(0, 2, 8, 3, 1, 5, 6, 4, 7).reshape(N, K)


def _pack(codes):
    """plain uint8 codes [N, K] -> dense_i8 fragment order (as Linear.to_i8_copy), with its spw"""
    N, K = codes.shape
    spw = 4 if (N // 16) % 4 == 0 else 2 if (N // 16) % 2 == 0 else 1
    q = codes.view(N // (16 * spw), spw, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 8, 2).contiguous().view(-1)
    return q, spw


def unpack_i8(q, s, N, K, spw):
    """inverse of the dense_i8 fragment order -> fp16 [N, K] (the int8 weights as a plain matrix)"""
    qq = q.view(N // (16 * spw), K // 32, spw, 8, 4, 2, 2, 2, 2).permute(0, 2, 8, 3, 1, 5, 6, 4, 7).reshape(N, K)
    return ((qq.float() - 128.0) * s[:, None]).half()


def _patch_skip_trellis():
    """Do not load the trellis of linears that will run from the stock int8 copy (EXL3_I8_LINEARS keys present in
    EXL3_I8_SRC): a zero-stride placeholder of the right shape stands in (decode uses the int8 GEMV, prefill the hooked
    int8 dequant path; the fused gate/up MultiLinear of such MLPs is switched off after load). Saves ~10 GB of HBM."""
    from exllamav3.loader.safetensors import SafetensorsCollection
    from exllamav3.modules.mlp import GatedMLP
    rx = re.compile(os.environ.get("EXL3_I8_LINEARS", "")) if os.environ.get("EXL3_I8_LINEARS") else None
    keys = {}
    def has_i8(base):
        m = re.match(r"model\.layers\.(\d+)\.", base)
        name = m.group(1) if m else ("lm_head" if base == "lm_head" else None)
        if name is None: return False
        if name not in keys:
            path = f"{I8_SRC}/{name}.safetensors"
            keys[name] = set(safe_open(path, "pt", device="cpu").keys()) if os.path.exists(path) else set()
        return base + ".q" in keys[name]
    gt = SafetensorsCollection.get_tensor
    def get_tensor(self, key, device=None, *a, **k):
        if key.endswith(".trellis") and rx is not None:
            base = key[:-len(".trellis")]
            if rx.search(base) and has_i8(base):
                f = self.tensor_file_map.get(key)
                shape = self.file_headers[f][key]["shape"] if f else None
                if shape is not None:
                    _stats["trellis_skipped"] = _stats.get("trellis_skipped", 0) + 1
                    dev = device if device is not None else "cpu"
                    return torch.zeros(1, dtype=torch.int16, device=dev).expand(*shape)
        return gt(self, key, device, *a, **k)
    SafetensorsCollection.get_tensor = get_tensor
    ll = GatedMLP.load_local
    def load_local(self, *a, **k):
        r = ll(self, *a, **k)
        g0 = self.gates[0].inner if getattr(self, "gates", None) else None
        if g0 is not None and getattr(g0, "trellis", None) is not None and g0.trellis.stride() == (0, 0, 0):
            self.multi_gu = [None for _ in self.multi_gu]          # fused gate/up mgemm would read the placeholder
        return r
    GatedMLP.load_local = load_local


def _patch_linear():
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.quant.exl3 import LinearEXL3
    to_i8 = Linear.to_i8_copy

    def to_i8_copy(self):
        inner = self.inner
        t = _i8_tensors(self.key) if isinstance(inner, LinearEXL3) and inner.bc is not None and inner.w8 is None else None
        if t is None:
            return to_i8(self)
        q, s, spw = t
        fake = inner.trellis.stride() == (0, 0, 0)       # EXL3_SKIP_TRELLIS: never loaded (zero-memory placeholder)
        if fake:
            N, K = self.out_features, self.in_features
            dev = inner.trellis.device
        else:
            W = inner.get_weight_tensor().float().T.contiguous()          # trellis dequant [out, in], only for the check
            N, K = W.shape
        if s.numel() < N and q.numel() == s.numel() * K and (N - s.numel()) % 16 == 0:
            # the module pads its output rows (kv_a_proj_with_mqa: 576 -> 640): zero rows (code 128, scale 1) at the end
            N0 = s.numel(); c = _codes(q, N0, K, spw)
            c = torch.cat([c, torch.full((N - N0, K), 128, dtype=torch.uint8)])
            q, spw = _pack(c); s = torch.cat([s.float(), torch.ones(N - N0)])
        if q.numel() != N * K or s.numel() != N:
            _stats["i8_mismatch"].append((self.key, "shape"))
            if fake:
                raise RuntimeError(f"hecate: {self.key} has no trellis (EXL3_SKIP_TRELLIS) and no matching stock int8")
            return to_i8(self)
        if not fake:
            dev = W.device
        q = q.to(dev); s = s.to(dev).float()
        rel = 0.0 if fake else ((unpack_i8(q, s, N, K, spw).float() - W).norm() / W.norm()).item()
        if rel > 0.3:   # rows permuted / wrong tensor: keep the trellis-derived copy
            _stats["i8_mismatch"].append((self.key, rel))
            if not fake: del W
            print(f" !! hecate: {self.key} stock int8 does not match the trellis (rel {rel:.3f}); kept the trellis copy", flush=True)
            return to_i8(self)
        if not fake: del W
        y32 = torch.zeros(8 * N, dtype=torch.float, device=dev)
        ctr = torch.zeros(max(1, N // 16), dtype=torch.int, device=dev)
        inner.w8 = (q, s, y32, ctr, spw)
        inner.bc.set_i8(q, s, y32, ctr, spw)
        inner.w8_src = "bf16"; inner.w8_rel_vs_trellis = rel
        _stats["i8_src"] += 1
        _hook_linear(self)
        return q.numel() + s.numel() * 4
    Linear.to_i8_copy = to_i8_copy


def _hook_linear(lin):
    """A linear whose int8 copy comes from stock BF16. Inner level: > 8 rows run the stock int8 weights dequantized on the
    fly (prefill shares the decode base instead of the trellis); <= 8 rows keep the int8 GEMV. Linear.forward itself
    (input padding, padded-output trim, scales) stays in charge."""
    if getattr(lin, "_i8_hooked", False):
        return
    inner = lin.inner
    fi = inner.forward
    def inner_forward(x, params=None, out_dtype=None, _i=inner, _f=fi):
        w8 = _i.w8; K = x.shape[-1]; rows = x.numel() // K
        if rows <= 8:
            return _f(x, params, out_dtype)
        N = w8[1].numel()
        W = unpack_i8(w8[0], w8[1], N, K, w8[4])
        y = (x.reshape(-1, K).half() @ W.T).to(out_dtype or getattr(_i, "default_out_dtype", None) or torch.half)
        return y.view(*x.shape[:-1], N)
    inner.forward = inner_forward
    lin._i8_hooked = True


def _load():
    if I8_SRC:
        _patch_linear()
    if I8_SRC and os.environ.get("EXL3_SKIP_TRELLIS", "0") == "1":
        _patch_skip_trellis()
    if OVERRIDE:
        from exllamav3 import Config
        fd = Config.from_directory
        def from_directory(*a, **k):
            c = fd(*a, **k)
            for _p in OVERRIDE.split(":"):          # file or directory; several joined by ':' (later ones win)
                if _p: c.stc.add_tensor_files(_p, warn_if_override=False)
            return c
        Config.from_directory = staticmethod(from_directory)
    rows = int(os.environ.get("EXL3_DRAFT_MAX_ROWS", "0") or 0)
    if rows:
        # concurrency: cap bsz * (1 + drafts) at the decode fast path's row limit (moe4 / RTX graph / CPU band: 8)
        from exllamav3.generator import generator as _g
        gi = _g.Generator.__init__
        def __init__(self, *a, **k):
            k.setdefault("draft_max_rows", rows); gi(self, *a, **k)
        _g.Generator.__init__ = __init__
    if os.environ.get("EXL3_ROUTER_FUSED", "0") == "1":
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import exl3_router
        exl3_router.install()
    if os.environ.get("EXL3_DRAFT_HEAD_I4", "0") == "1":
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import exl3_draft_head
        exl3_draft_head.install()
    if os.environ.get("EXL3_VERIFY_GRAPH", "0") == "1":
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import exl3_vgraph
        exl3_vgraph.install()
    print(f" -- hecate runtime: int8 base from {I8_SRC or 'trellis'}, tensor override {OVERRIDE or 'off'}", flush=True)


_load()
