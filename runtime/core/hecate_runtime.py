"""Hecate runtime module (imported before the model loads; the entrypoint imports it after tier_split and hecate_loader).

  int8 base       the int8 copies of the EXL3_I8_LINEARS set (attention projections, shared experts, dense MLPs, LM head)
                  come from stock BF16 (EXL3_I8_SRC, tools/quant/build_int8.py). Decode (<= 8 rows) runs them through the
                  cluster-reduced int8 GEMVs; prefill dequantizes them on the fly, so prompt and decode share one base.
                  Their EXL3 trellis is never loaded (EXL3_SKIP_TRELLIS=1).
  tensor override EXL3_TENSOR_OVERRIDE: extra tensor files or directories (':'-joined, later ones win), e.g. the warm
                  RTX tier's trellis experts.
  also installed  EXL3_ROUTER_FUSED (exl3_router), EXL3_NORM_ROUTE (exl3_normroute, with the fused router),
                  EXL3_DRAFT_HEAD_I4 (exl3_draft_head), EXL3_VERIFY_GRAPH / EXL3_DRAFT_GRAPH (exl3_vgraph),
                  EXL3_DRAFT_MAX_ROWS (row cap of the decode fast path)."""
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


_UNPACK_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
// dense_i8 fragment order ([group][k-block of 32][strip][g][q][t][h][c][rh] bytes) -> fp16 [N][K] = (code - 128) * s[n]:
// one thread per 8 consecutive k of one row (one 16-byte store), the same fp32 math as the torch version
__global__ void unpack_i8_k(const uint8_t* __restrict__ q, const float* __restrict__ s, half* __restrict__ w, int N, int K, int spw)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, kc = K / 8;
    if (i >= N * kc) return;
    const int n = i / kc, k0 = (i % kc) * 8;
    const int grp = n / (16 * spw), si = (n % (16 * spw)) / 16, rh = (n % 16) / 8, g = n % 8;
    const int kb = k0 / 32, t = (k0 % 32) / 16, h = (k0 % 16) / 8;
    const size_t a = ((size_t) (grp * (K / 32) + kb) * spw + si) * 8 + g;
    const uint8_t* src = q + a * 64 + t * 8 + h * 4 + rh;          // + qq * 16 + c * 2
    const float sc = s[n];
    __align__(16) half o[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __float2half_rn(((float) src[(j >> 1) * 16 + (j & 1) * 2] - 128.f) * sc);
    *(uint4*) (w + (size_t) n * K + k0) = *(const uint4*) o;
}
void unpack_i8(at::Tensor q, at::Tensor s, at::Tensor w, int64_t N, int64_t K, int64_t spw)
{
    const int n = (int) (N * K / 8);
    unpack_i8_k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(q.data_ptr<uint8_t>(), s.data_ptr<float>(),
        (half*) w.data_ptr(), (int) N, (int) K, (int) spw);
}
"""
_unpack_ext = None


def unpack_i8(q, s, N, K, spw):
    """inverse of the dense_i8 fragment order -> fp16 [N, K] (the int8 weights as a plain matrix); one fused kernel on
    CUDA (prefill runs this per chunk for every int8 linear), the torch ops otherwise"""
    global _unpack_ext
    if q.is_cuda and os.environ.get("EXL3_I8_UNPACK_FUSED", "1") == "1":
        if _unpack_ext is None:
            from torch.utils.cpp_extension import load_inline
            bd = os.path.join(os.environ.get("EXL3_TIER_BUILD", "/tmp"), "build_unpack_i8"); os.makedirs(bd, exist_ok=True)
            _unpack_ext = load_inline(name="unpack_i8", cpp_sources="void unpack_i8(at::Tensor q, at::Tensor s, at::Tensor w, int64_t N, int64_t K, int64_t spw);",
                                      cuda_sources=_UNPACK_SRC, functions=["unpack_i8"], build_directory=bd,
                                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"])
        w = torch.empty((N, K), dtype=torch.half, device=q.device)
        _unpack_ext.unpack_i8(q, s.float().contiguous(), w, N, K, spw)
        return w
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


I8_GEMV_ROWS = int(os.environ.get("EXL3_I8_GEMV_ROWS", "64"))   # up to this many rows: the int8 GEMV in 8-row chunks


def _hook_linear(lin):
    """A linear whose int8 copy comes from stock BF16. Inner level: up to 8 rows the int8 GEMV; up to EXL3_I8_GEMV_ROWS
    rows the GEMV in 8-row chunks (a decode batch at concurrency 16-32 would otherwise dequantize the whole matrix, the LM
    head included, every step); more rows (prefill) run the stock int8 weights dequantized on the fly, so prefill shares
    the decode base instead of the trellis. Linear.forward itself (input padding, padded-output trim, scales) stays in
    charge."""
    if getattr(lin, "_i8_hooked", False):
        return
    inner = lin.inner
    fi = inner.forward
    def inner_forward(x, params=None, out_dtype=None, _i=inner, _f=fi):
        w8 = _i.w8; K = x.shape[-1]; rows = x.numel() // K
        if rows <= 8:
            return _f(x, params, out_dtype)
        N = w8[1].numel()
        if rows <= I8_GEMV_ROWS:
            x2 = x.reshape(rows, K)
            y = torch.cat([_f(x2[r:r + 8].unsqueeze(0), params, out_dtype).reshape(-1, N) for r in range(0, rows, 8)])
            return y.view(*x.shape[:-1], N)
        W = unpack_i8(w8[0], w8[1], N, K, w8[4])
        y = (x.reshape(-1, K).half() @ W.T).to(out_dtype or getattr(_i, "default_out_dtype", None) or torch.half)
        return y.view(*x.shape[:-1], N)
    inner.forward = inner_forward
    lin._i8_hooked = True


# hecate (concurrency): each request's class, visible to the generator's enqueue in the same task. Streamed requests
# (chat UIs, agents) are interactive, the rest batch; HECATE_PRIORITY=off treats every request alike (FIFO)
import contextvars
REQUEST = contextvars.ContextVar("hecate_request", default=None)


def patch_tabby():
    """wrap TabbyAPI's ExllamaV3Container.generate_gen so the job it creates is tagged with its request's class; called
    by the entrypoint once /opt/tabbyAPI is importable, before TabbyAPI's main runs"""
    if os.environ.get("HECATE_PRIORITY", "stream") == "off":
        return
    try:
        from backends.exllamav3.model import ExllamaV3Container as C
    except Exception as e:
        print(f" -- hecate: requests not classified ({type(e).__name__}: {e}); FIFO admission", flush=True)
        return
    gg = C.generate_gen

    async def generate_gen(self, request_id, prompt, params, *a, **k):
        # the job is created and enqueued while the generator runs up to its first result: tag only that step (set and
        # cleared in the same context; a stream can be closed later from another one)
        agen = gg(self, request_id, prompt, params, *a, **k)
        tok = REQUEST.set({"cls": 0 if getattr(params, "stream", False) else 1})
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:
            return
        finally:
            REQUEST.reset(tok)
        yield first
        async for r in agen:
            yield r
    C.generate_gen = generate_gen
    print(" -- hecate: streamed requests start ahead of non-streamed ones (HECATE_PRIORITY)", flush=True)


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
        if os.environ.get("EXL3_NORM_ROUTE", "0") == "1":
            import exl3_normroute
            exl3_normroute.install()
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
