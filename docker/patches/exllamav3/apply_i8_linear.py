"""int8 copies of selected EXL3 linears (hecate; applies on top of apply_fp16_linear.py). Runtime switch
EXL3_I8_LINEARS (key regex, unset = unchanged): after a module's weights land, matching EXL3 linears on
EXL3_FP16_LINEARS_CC devices are dequantized once and re-encoded as int8 with a per-output-row scale, pre-swizzled in
mma fragment order (pack_i8). BC_LinearEXL3 and the BC_MLA projections then run quant/dense_i8.cu (LOP3-decode GEMV,
stream-K, bandwidth-bound) for 1..8 rows; more rows fall back to the trellis kernels (still resident). A linear with an
int8 copy gets no FP16 copy, so listing a key in both regexes replaces its FP16 copy (half the HBM, half the bytes).
Why: at the 4-row verify shape the FP16 projections are bandwidth-bound cuBLAS GEMVs (3.8-5.6 TB/s + split-K reduce
kernels, ~6.9 ms per step); int8 halves the bytes and drops the reduce launches.
   python3 apply_i8_linear.py EXT_DIR PKG_DIR SRC_DIR"""
import pathlib, shutil, sys
ext, pkg, srcdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


shutil.copy(srcdir / "dense_i8.cu", ext / "quant/dense_i8.cu")
(ext / "quant/dense_i8.cuh").write_text('''#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

// hecate: int8 decode GEMV for 1..8 rows (EXL3_I8_LINEARS), see dense_i8.cu
void dense_i8_gr(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
                 at::Tensor& ctr, int spw, Graph* graph);
void dense_i8(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
              at::Tensor& ctr, int64_t spw);
''')
sub(ext / "quant/dense_i8.cu", '#include "../util.h"\n', '#include "../util.h"\n#include "../util.cuh"\n')

# bindings: dense_i8 for tests
sub(ext / "bindings.cpp", '#include "hgemm.cuh"\n', '#include "hgemm.cuh"\n#include "quant/dense_i8.cuh"\n')
sub(ext / "bindings.cpp", '    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");\n',
    '    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");\n    m.def("dense_i8", &dense_i8, "dense_i8");\n')

# BC_LinearEXL3: optional int8 copy
sub(ext / "libtorch/linear.h", "    c10::optional<at::Tensor> w16;   // hecate: FP16 [in, out] copy, replaces the trellis kernels when set\n",
    "    c10::optional<at::Tensor> w16;   // hecate: FP16 [in, out] copy, replaces the trellis kernels when set\n"
    "    c10::optional<at::Tensor> w8, w8s, w8y32, w8ctr;   // hecate: int8 copy (packed), row scales, scratch\n"
    "    int w8spw = 4;\n")
sub(ext / "libtorch/linear.h", "    void set_fp16(c10::optional<at::Tensor> w) { w16 = std::move(w); }\n",
    "    void set_fp16(c10::optional<at::Tensor> w) { w16 = std::move(w); }\n"
    "    void set_i8(at::Tensor w, at::Tensor s, at::Tensor y32, at::Tensor ctr, int64_t spw)\n"
    "        { w8 = std::move(w); w8s = std::move(s); w8y32 = std::move(y32); w8ctr = std::move(ctr); w8spw = (int) spw; }\n"
    "    bool i8_ok(const at::Tensor& x) const { return w8 && x.numel() / x.size(-1) <= 8; }\n")
sub(ext / "libtorch/linear_bc.h", '.def("set_fp16", &BC_LinearEXL3::set_fp16);',
    '.def("set_fp16", &BC_LinearEXL3::set_fp16)\n.def("set_i8", &BC_LinearEXL3::set_i8);')
lin = ext / "libtorch/linear.cpp"
s = lin.read_text()
first_inc = s.index("#include")
lin.write_text(s[:first_inc] + '#include "../quant/dense_i8.cuh"\n' + s[first_inc:])
sub(lin, "void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)\n{\n",
    "void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)\n{\n"
    "    if (i8_ok(x))\n    {\n"
    "        dense_i8_gr(x, w8.value(), w8s.value(), y, w8y32.value(), w8ctr.value(), w8spw, graph);\n"
    "        if (bias) add_gr(y, bias.value(), y, graph);\n"
    "        return;\n    }\n")

# BC_MLA: int8 projections (same graph params as the trellis GEMM: x = GP_gemm_A, y = GP_gemm_C)
mla = ext / "libtorch/mla_attention.cpp"
sub(mla, '#include "../quant/exl3_gemm.cuh"\n', '#include "../quant/exl3_gemm.cuh"\n#include "../quant/dense_i8.cuh"\n')
sub(mla, "    if (p->w16) { hgemm_gr(x, p->w16.value(), y, graph); return; }\n",
    "    if (p->i8_ok(x)) { dense_i8_gr(x, p->w8.value(), p->w8s.value(), y, p->w8y32.value(), p->w8ctr.value(), p->w8spw, graph); return; }\n"
    "    if (p->w16) { hgemm_gr(x, p->w16.value(), y, graph); return; }\n")
sub(mla, "        copy2d_gr(s.y16, y2, graph);\n    }\n    else\n        exl3_gemm_gr(s.o, o_proj->trellis, y2,",
    "        copy2d_gr(s.y16, y2, graph);\n    }\n"
    "    else if (o_proj->i8_ok(s.o))\n"
    "        dense_i8_gr(s.o, o_proj->w8.value(), o_proj->w8s.value(), y2, o_proj->w8y32.value(), o_proj->w8ctr.value(), o_proj->w8spw, graph);\n"
    "    else\n        exl3_gemm_gr(s.o, o_proj->trellis, y2,")

# BC_GatedMLP (shared expert with EXL3_MOE_SHARED_COOP=0, dense MLPs): int8 gate / up / down when all three have
# copies; the graph args then take the per-linear (A, C) form instead of the fused mgemm's
mlp = ext / "libtorch/mlp.cpp"
sub(mlp, '#include "../quant/exl3_gemm.cuh"\n', '#include "../quant/exl3_gemm.cuh"\n#include "../quant/dense_i8.cuh"\n')
sub(mlp, "    at::Tensor down_xh_n = down_xh.slice(1, 0, num_tokens);\n\n    if (gu_ptrs_trellis)\n",
    "    at::Tensor down_xh_n = down_xh.slice(1, 0, num_tokens);\n"
    "    const bool i8 = gate && up && down && gate->i8_ok(x) && up->w8 && down->w8;   // hecate: EXL3_I8_LINEARS\n\n"
    "    if (i8)\n    {\n"
    "        at::Tensor g2 = gu_n.select(0, 0);\n        at::Tensor u2 = gu_n.select(0, 1);\n"
    "        dense_i8_gr(x, gate->w8.value(), gate->w8s.value(), g2, gate->w8y32.value(), gate->w8ctr.value(), gate->w8spw, graph);\n"
    "        dense_i8_gr(x, up->w8.value(), up->w8s.value(), u2, up->w8y32.value(), up->w8ctr.value(), up->w8spw, graph);\n"
    "        if (gate->bias) add_gr(g2, gate->bias.value(), g2, graph);\n"
    "        if (up->bias) add_gr(u2, up->bias.value(), u2, graph);\n"
    "    }\n    else if (gu_ptrs_trellis)\n")
sub(mlp, "    exl3_gemm_gr(a_n, down->trellis, d, down->suh, down_xh_n, down->svh, -1, down->mcg, down->mul1, 0, graph);\n    if (down->bias)\n        add_gr(d, down->bias.value(), d, graph);\n}\n",
    "    if (i8)\n        dense_i8_gr(a_n, down->w8.value(), down->w8s.value(), d, down->w8y32.value(), down->w8ctr.value(), down->w8spw, graph);\n"
    "    else\n        exl3_gemm_gr(a_n, down->trellis, d, down->suh, down_xh_n, down->svh, -1, down->mcg, down->mul1, 0, graph);\n"
    "    if (down->bias)\n        add_gr(d, down->bias.value(), d, graph);\n}\n")
sub(mlp, "        std::vector<PPTR> args;\n        if (gu_ptrs_trellis)\n        {\n            args.emplace_back(GP_mgemm_A, (void*) x.data_ptr());\n",
    "        std::vector<PPTR> args;\n        if (gu_ptrs_trellis && !(gate && up && down && gate->i8_ok(x) && up->w8 && down->w8))\n        {\n            args.emplace_back(GP_mgemm_A, (void*) x.data_ptr());\n")

# Python: BC_GatedMLP always gets the gate/up handles (the C++ prefers the fused mgemm unless all three have int8 copies)
mp = pkg / "modules/mlp.py"
sub(mp, "                    g0.inner.bc if mgu is None else None,\n                    u0.inner.bc if mgu is None else None,\n",
    "                    getattr(g0.inner, 'bc', None),   # hecate: handles kept for EXL3_I8_LINEARS\n"
    "                    getattr(u0.inner, 'bc', None),\n")

# Python: LinearEXL3 routes <= 8 rows through the BC object once an int8 copy exists
ex = pkg / "modules/quant/exl3.py"
sub(ex, "        self.w16 = None   # hecate: FP16 copy (EXL3_FP16_LINEARS)\n",
    "        self.w16 = None   # hecate: FP16 copy (EXL3_FP16_LINEARS)\n        self.w8 = None    # hecate: int8 copy (EXL3_I8_LINEARS)\n")
sub(ex, "        if self.w16 is not None:\n",
    "        if self.w8 is not None and x.numel() // x.shape[-1] <= 8:\n"
    "            return self.bc.run_alloc(x, self.out_features, (out_dtype or self.default_out_dtype) == torch.float)\n"
    "        if self.w16 is not None:\n")

# Python: Linear.to_i8_copy (+ to_fp16_copy skips linears that already have an int8 copy)
li = pkg / "modules/linear.py"
sub(li, "        if not isinstance(inner, LinearEXL3) or inner.bc is None or inner.w16 is not None or inner.bias is not None:\n            return 0\n        w = inner.get_weight_tensor().half().contiguous()\n",
    "        if not isinstance(inner, LinearEXL3) or inner.bc is None or inner.w16 is not None or inner.bias is not None:\n            return 0\n"
    "        if getattr(inner, 'w8', None) is not None:\n            return 0\n"
    "        w = inner.get_weight_tensor().half().contiguous()\n")
sub(li, "    def to_fp16_copy(self) -> int:\n", '''    def to_i8_copy(self) -> int:
        """hecate: dequantize this EXL3 linear once and re-encode it as int8 (per-row absmax scale) in the fragment
        order of quant/dense_i8.cu, used by its BC object for 1..8 rows (EXL3_I8_LINEARS)."""
        inner = self.inner
        if not isinstance(inner, LinearEXL3) or inner.bc is None or inner.w8 is not None or inner.bias is not None:
            return 0
        W = inner.get_weight_tensor().float().T.contiguous()      # [out, in]
        N, K = W.shape
        if N % 16 or K % 32:
            return 0
        spw = 4 if (N // 16) % 4 == 0 else 2 if (N // 16) % 2 == 0 else 1
        s = (W.abs().amax(1) / 127.0).clamp_min(1e-12)
        q = (torch.round(W / s[:, None]) + 128).clamp_(0, 255).to(torch.uint8)
        del W
        # [group][spw][rh][g][kb][t][h][q][c] -> [group][kb][spw][lane = (g, q)][word = (t, h)][byte = (c, rh)]
        q = q.view(N // (16 * spw), spw, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 8, 2).contiguous().view(-1)
        s = s.contiguous()
        y32 = torch.zeros(8 * N, dtype = torch.float, device = q.device)
        ctr = torch.zeros(max(1, N // 16), dtype = torch.int, device = q.device)
        inner.w8 = (q, s, y32, ctr, spw)
        inner.bc.set_i8(q, s, y32, ctr, spw)
        return q.numel() + s.numel() * 4

    def to_fp16_copy(self) -> int:
''')

# Python: loader hook (int8 first, then FP16 for the rest)
ls = pkg / "model/model_ls.py"
sub(ls, '_FP16_RE = _re.compile(_os.environ["EXL3_FP16_LINEARS"]) if _os.environ.get("EXL3_FP16_LINEARS") else None\n',
    '_FP16_RE = _re.compile(_os.environ["EXL3_FP16_LINEARS"]) if _os.environ.get("EXL3_FP16_LINEARS") else None\n'
    '_I8_RE = _re.compile(_os.environ["EXL3_I8_LINEARS"]) if _os.environ.get("EXL3_I8_LINEARS") else None\n')
sub(ls, "    if _FP16_RE is None:\n        return\n    from ..modules.linear import Linear\n    for sm in module:\n"
        "        if isinstance(sm, Linear) and sm.device is not None and _FP16_RE.search(sm.key):\n"
        "            dev = torch.device(sm.device)\n"
        "            if dev.type == \"cuda\" and torch.cuda.get_device_capability(dev) == _FP16_CC:\n"
        "                n = sm.to_fp16_copy()\n",
    "    if _FP16_RE is None and _I8_RE is None:\n        return\n    from ..modules.linear import Linear\n    for sm in module:\n"
    "        i8 = _I8_RE is not None and isinstance(sm, Linear) and _I8_RE.search(sm.key) is not None\n"
    "        if isinstance(sm, Linear) and sm.device is not None and (i8 or (_FP16_RE is not None and _FP16_RE.search(sm.key))):\n"
    "            dev = torch.device(sm.device)\n"
    "            if dev.type == \"cuda\" and torch.cuda.get_device_capability(dev) == _FP16_CC:\n"
    "                n = sm.to_i8_copy() if i8 else 0\n"
    "                if not n and _FP16_RE is not None and _FP16_RE.search(sm.key):\n"
    "                    n = sm.to_fp16_copy()\n")
print("patched: EXL3_I8_LINEARS support")
