"""FP16 copies of selected EXL3 linears (hecate, ExLlamaV3 v1.6.5). Runtime switch EXL3_FP16_LINEARS (key regex,
unset = stock behaviour): after a module's weights land, matching EXL3 linears on devices with compute capability
EXL3_FP16_LINEARS_CC (default 10.3 = GB300) are dequantized once to an FP16 [in, out] copy. BC_LinearEXL3 then runs
that copy through the same hgemm path as BC_LinearFP16, so the graph-captured C++ blocks (BC_MLA, BC_MLP, ...) that
hold BC_LinearEXL3 objects pick it up unchanged. Why: at the 4-row MTP verify shape the trellis GEMV chain has a
~17-25 us floor per small matrix on the GB300 vs ~5.5 us for cuBLAS FP16 (layer 10 attention + shared experts:
203 vs 78 us per layer). The trellis stays resident (prefill fallbacks, fused paths that read it directly).
   python3 apply_fp16_linear.py EXT_DIR PKG_DIR"""
import pathlib, sys
ext, pkg = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


# C++: optional FP16 weight on BC_LinearEXL3
sub(ext / "libtorch/linear.h", "    bool mul1;\n    at::Tensor xh;\n",
    "    bool mul1;\n    at::Tensor xh;\n    c10::optional<at::Tensor> w16;   // hecate: FP16 [in, out] copy, replaces the trellis kernels when set\n")
sub(ext / "libtorch/linear.h", "    at::Tensor run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32);\n};",
    "    at::Tensor run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32);\n"
    "    void set_fp16(c10::optional<at::Tensor> w) { w16 = std::move(w); }\n};")
sub(ext / "libtorch/linear.cpp", "void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)\n{\n",
    "void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)\n{\n"
    "    if (w16)\n    {\n"
    "        if (x.dtype() == y.dtype() && !graph)\n            at::matmul_out(y, x, w16.value());\n"
    "        else\n            hgemm_gr(x, w16.value(), y, graph);\n"
    "        if (bias)\n            add_gr(y, bias.value(), y, graph);\n"
    "        return;\n    }\n\n")
sub(ext / "libtorch/linear_bc.h", '.def("run_alloc", &BC_LinearEXL3::run_alloc);',
    '.def("run_alloc", &BC_LinearEXL3::run_alloc)\n.def("set_fp16", &BC_LinearEXL3::set_fp16);')

# C++: BC_MLA (graph-captured MLA decode block) calls exl3_gemm_gr on each projection's trellis directly. Route the
# projections through cuBLAS when an FP16 copy is set. The graph patcher re-points kernel arguments per call (x and y
# are external), which it cannot do for cuBLAS nodes, so FP16 projections read a static copy of x (copy2d, its src
# patched per call) and o_proj writes a static buffer that copy2d moves to y (dst patched); the launch params list
# drops the GEMM entries of converted projections. (Shared experts stay trellis: they run inside the cooperative MoE
# kernel, which reads the trellis pointers.)
sub(ext / "libtorch/mla_attention.h", "        at::Tensor o;         // (R, H * D_v) fp16\n",
    "        at::Tensor o;         // (R, H * D_v) fp16\n"
    "        at::Tensor x16, y16;  // hecate: staged input / output of FP16 projections (EXL3_FP16_LINEARS)\n")
mla = ext / "libtorch/mla_attention.cpp"
sub(mla, '#include "../quant/exl3_gemm.cuh"\n', '''#include "../quant/exl3_gemm.cuh"
#include "../hgemm.cuh"

// hecate: EXL3_FP16_LINEARS -- an FP16 copy on the BC_LinearEXL3 replaces the trellis GEMM
static inline void mla_gemm_gr(const at::Tensor& x, const std::shared_ptr<BC_LinearEXL3>& p, at::Tensor& y,
                               const at::Tensor& xh, Graph* graph)
{
    if (p->w16) { hgemm_gr(x, p->w16.value(), y, graph); return; }
    exl3_gemm_gr(x, p->trellis, y, p->suh, xh, p->svh, -1, p->mcg, p->mul1, 0, graph);
}
''')
sub(mla, "    // Q projections into the static buffers (direct, or LoRA down/norm/up)\n    if (q_a_proj)\n",
    "    // hecate: FP16 projections read a static copy of x (see apply_fp16_linear.py)\n"
    "    const bool fp16_q = q_a_proj ? (bool) q_a_proj->w16 : (bool) q_proj->w16;\n"
    "    const bool fp16_kv = (bool) kv_a_proj->w16;\n"
    "    if (fp16_q || fp16_kv)\n    {\n"
    "        if (!s.x16.defined() || s.x16.size(0) != R) s.x16 = at::empty({R, hidden_size}, x2.options());\n"
    "        copy2d_gr(x2, s.x16, graph);\n    }\n"
    "    const at::Tensor& x_q = fp16_q ? s.x16 : x2;\n"
    "    const at::Tensor& x_kv = fp16_kv ? s.x16 : x2;\n\n"
    "    // Q projections into the static buffers (direct, or LoRA down/norm/up)\n    if (q_a_proj)\n")
sub(mla, "exl3_gemm_gr(x2, q_a_proj->trellis, s.q_a, q_a_proj->suh, xh_x, q_a_proj->svh, -1, q_a_proj->mcg, q_a_proj->mul1, 0, graph);",
    "mla_gemm_gr(x_q, q_a_proj, s.q_a, xh_x, graph);")
sub(mla, "exl3_gemm_gr(s.q_a, q_proj->trellis, s.q_full, q_proj->suh, xh_a, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);",
    "mla_gemm_gr(s.q_a, q_proj, s.q_full, xh_a, graph);")
sub(mla, "exl3_gemm_gr(x2, q_proj->trellis, s.q_full, q_proj->suh, xh_x, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);",
    "mla_gemm_gr(x_q, q_proj, s.q_full, xh_x, graph);")
sub(mla, "exl3_gemm_gr(x2, kv_a_proj->trellis, s.ckv_kpe, kv_a_proj->suh, xh_x, kv_a_proj->svh, -1, kv_a_proj->mcg, kv_a_proj->mul1, 0, graph);",
    "mla_gemm_gr(x_kv, kv_a_proj, s.ckv_kpe, xh_x, graph);")
sub(mla, """exl3_gemm_gr(s.q_a, idx_wq_b->trellis, s.qidx, idx_wq_b->suh,
                         xh.view({-1}).narrow(0, 0, (int64_t) R * q_lora_rank).view({R, q_lora_rank}),
                         idx_wq_b->svh, -1, idx_wq_b->mcg, idx_wq_b->mul1, 0, graph);""",
    "mla_gemm_gr(s.q_a, idx_wq_b, s.qidx, xh.view({-1}).narrow(0, 0, (int64_t) R * q_lora_rank).view({R, q_lora_rank}), graph);")
sub(mla, "    exl3_gemm_gr(s.o, o_proj->trellis, y2, o_proj->suh, xh_o, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);\n",
    "    if (o_proj->w16)\n    {\n"
    "        if (!s.y16.defined() || s.y16.size(0) != R || s.y16.dtype() != y2.dtype()) s.y16 = at::empty_like(y2);\n"
    "        hgemm_gr(s.o, o_proj->w16.value(), s.y16, graph);\n"
    "        copy2d_gr(s.y16, y2, graph);\n    }\n    else\n"
    "        exl3_gemm_gr(s.o, o_proj->trellis, y2, o_proj->suh, xh_o, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);\n")
# launch params: staging copy src first, GEMM entries only for trellis projections, o_proj through the copy2d dst
sub(mla, "    params.emplace_back(GP_gemm_A, (void*) x.data_ptr());\n    if (q_a_proj)\n        params.emplace_back(GP_gemm_A, (void*) s.q_a.data_ptr());\n",
    "    const bool fp16_q = q_a_proj ? (bool) q_a_proj->w16 : (bool) q_proj->w16;\n"
    "    if (fp16_q || kv_a_proj->w16)\n        params.emplace_back(GP_copy2d_src, (void*) x.data_ptr());   // hecate: FP16 staging copy\n"
    "    if (!fp16_q)\n        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());\n"
    "    if (q_a_proj && !q_proj->w16)\n        params.emplace_back(GP_gemm_A, (void*) s.q_a.data_ptr());\n")
sub(mla, "    // Latent projection\n    params.emplace_back(GP_gemm_A, (void*) x.data_ptr());\n",
    "    // Latent projection\n    if (!kv_a_proj->w16)\n        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());\n")
sub(mla, "    // Output projection\n    params.emplace_back(GP_gemm_C, (void*) y.data_ptr());\n",
    "    // Output projection\n    params.emplace_back(o_proj->w16 ? GP_copy2d_dst : GP_gemm_C, (void*) y.data_ptr());\n")

# Python: LinearEXL3 routes every row count through the BC object once a copy exists
sub(pkg / "modules/quant/exl3.py", "        self._fused_reconstruct = None\n",
    "        self._fused_reconstruct = None\n        self.w16 = None   # hecate: FP16 copy (EXL3_FP16_LINEARS)\n")
sub(pkg / "modules/quant/exl3.py",
    '        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"\n',
    '        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"\n'
    "        if self.w16 is not None:\n"
    "            return self.bc.run_alloc(x, self.out_features, (out_dtype or self.default_out_dtype) == torch.float)\n")

# Python: Linear.to_fp16_copy
sub(pkg / "modules/linear.py", "    @override\n    def can_defer_load(self):\n",
    "    def to_fp16_copy(self) -> int:\n"
    '        """hecate: dequantize this EXL3 linear once to an FP16 copy used by its BC object (EXL3_FP16_LINEARS)."""\n'
    "        inner = self.inner\n"
    "        if not isinstance(inner, LinearEXL3) or inner.bc is None or inner.w16 is not None or inner.bias is not None:\n"
    "            return 0\n"
    "        w = inner.get_weight_tensor().half().contiguous()\n"
    "        inner.w16 = w\n"
    "        inner.bc.set_fp16(w)\n"
    "        return w.numel() * 2\n\n\n"
    "    @override\n    def can_defer_load(self):\n")

# Python: post-load hook in both single-device and autosplit loaders (after the deferred fills land, before the
# autosplit measuring forward, so the split accounts for the copies)
ls = pkg / "model/model_ls.py"
sub(ls, "\nclass ", '''
import os as _os, re as _re
_FP16_RE = _re.compile(_os.environ["EXL3_FP16_LINEARS"]) if _os.environ.get("EXL3_FP16_LINEARS") else None
_FP16_CC = tuple(int(v) for v in _os.environ.get("EXL3_FP16_LINEARS_CC", "10.3").split("."))
_fp16_bytes = {}


def _fp16_linears(module):
    """hecate: FP16 copies of the EXL3 linears matching EXL3_FP16_LINEARS on EXL3_FP16_LINEARS_CC devices."""
    if _FP16_RE is None:
        return
    from ..modules.linear import Linear
    for sm in module:
        if isinstance(sm, Linear) and sm.device is not None and _FP16_RE.search(sm.key):
            dev = torch.device(sm.device)
            if dev.type == "cuda" and torch.cuda.get_device_capability(dev) == _FP16_CC:
                n = sm.to_fp16_copy()
                if n:
                    _fp16_bytes[str(dev)] = _fp16_bytes.get(str(dev), 0) + n


class ''', count=1)
sub(ls, '''                    module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
                    if defer:
                        config.stc.end_deferred_load()
''', '''                    module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
                    if defer:
                        config.stc.end_deferred_load()
                _fp16_linears(module)
''')
sub(ls, '''                            module.load(load_device, max_chunk_size = max_chunk_size)
                            if defer:
                                config.stc.end_deferred_load()
''', '''                            module.load(load_device, max_chunk_size = max_chunk_size)
                            if defer:
                                config.stc.end_deferred_load()
                        _fp16_linears(module)
''')
print("patched: EXL3_FP16_LINEARS support")
