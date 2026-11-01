"""hecate: final int8 GEMV source (dense_i8.cu: thread-block-cluster GEMVs, fused q_a + kv_a and gate + up + SiLU kernels,
an optional rank-4 output correction DenseI8LR that the release leaves unset) and the MLA split floor knob.
   python3 apply_fold.py EXT_DIR SRC_DIR   (EXT_DIR = exllamav3/exllamav3_ext; every replacement asserted)"""
import sys, shutil
ext, src = sys.argv[1:3]
shutil.copy(f"{src}/dense_i8.cu", f"{ext}/quant/dense_i8.cu")
shutil.copy(f"{src}/dense_i8.cuh", f"{ext}/quant/dense_i8.cuh")


def patch(name, reps):
    p = f"{ext}/{name}"; s = open(p).read()
    for old, new in reps:
        assert s.count(old) == 1, (name, old[:80], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched", name)


patch("libtorch/linear.h", [
    ("struct BC_LinearEXL3\n{", "#include \"../quant/dense_i8.cuh\"\n\nstruct BC_LinearEXL3\n{"),
    ("    int w8spw = 4;\n",
     "    int w8spw = 4;\n"
     "    c10::optional<DenseI8LR> lr;   // hecate: optional rank-4 output correction of the int8 path (unset in the release)\n"),
    ("    bool i8_ok(const at::Tensor& x) const { return w8 && x.numel() / x.size(-1) <= 8; }\n",
     "    bool i8_ok(const at::Tensor& x) const { return w8 && x.numel() / x.size(-1) <= 8; }\n"
     "    void set_lowrank(at::Tensor us, at::Tensor v, at::Tensor t, at::Tensor modes)\n"
     "        { lr = DenseI8LR{std::move(us), std::move(v), std::move(t), std::move(modes)}; }\n"
     "    void clear_lowrank() { lr.reset(); }\n"
     "    const DenseI8LR* lr_ptr() const { return lr ? &lr.value() : nullptr; }\n"),
])
patch("libtorch/linear.cpp", [
    ("        dense_i8_gr(x, w8.value(), w8s.value(), y, w8y32.value(), w8ctr.value(), w8spw, graph);\n",
     "        dense_i8_gr(x, w8.value(), w8s.value(), y, w8y32.value(), w8ctr.value(), w8spw, graph, lr_ptr());\n"),
])
patch("libtorch/mlp.cpp", [
    ("        dense_i8_gr(a_n, down->w8.value(), down->w8s.value(), d, down->w8y32.value(), down->w8ctr.value(), down->w8spw, graph);\n",
     "        dense_i8_gr(a_n, down->w8.value(), down->w8s.value(), d, down->w8y32.value(), down->w8ctr.value(), down->w8spw, graph, down->lr_ptr());\n"),
    # gate + up + SiLU product as one kernel (dense_i8_gu_act) when both are plain int8 copies (EXL3_I8_GU_FUSED, default on)
    ("using namespace torch::indexing;\n",
     "using namespace torch::indexing;\n\n"
     "static bool gu_fused_ok(const BC_GatedMLP* m)\n"
     "{\n"
     "    static const bool on = [] { const char* e = getenv(\"EXL3_I8_GU_FUSED\"); return !(e && *e == '0'); }();\n"
     "    return on && m->act_silu && m->act_limit == 0.0f && m->gate && m->up && !m->gate->bias && !m->up->bias &&\n"
     "           m->gate->w8spw == 4 && m->up->w8spw == 4 && !m->gate->lr && !m->up->lr;\n"
     "}\n"),
    ("    if (i8)\n"
     "    {\n"
     "        at::Tensor g2 = gu_n.select(0, 0);\n",
     "    const bool gu_fused = i8 && gu_fused_ok(this);\n"
     "    if (gu_fused)\n"
     "        dense_i8_gu_act_gr(x, gate->w8.value(), gate->w8s.value(), gate->w8y32.value(), gate->w8ctr.value(),\n"
     "                           up->w8.value(), up->w8s.value(), up->w8y32.value(), a_n, 4, graph);\n"
     "    else if (i8)\n"
     "    {\n"
     "        at::Tensor g2 = gu_n.select(0, 0);\n"),
    ("    if (act_silu)\n"
     "        silu_mul_gr(g, u, a_n, act_limit, graph);\n"
     "    else if (act_gelu)\n",
     "    if (gu_fused)\n"
     "        ;\n"
     "    else if (act_silu)\n"
     "        silu_mul_gr(g, u, a_n, act_limit, graph);\n"
     "    else if (act_gelu)\n"),
    ("            at::Tensor gu_n = slot_pair_view(gu, num_tokens);\n"
     "            // The gate/up GEMMs record their own GP_gemm_C sites ahead of the down projection's;\n",
     "          if (gate && up && down && gate->i8_ok(x) && up->w8 && down->w8 && gu_fused_ok(this))\n"
     "            args.emplace_back(GP_gemm_A, (void*) x.data_ptr());   // one gate/up/act kernel: x only\n"
     "          else {\n"
     "            at::Tensor gu_n = slot_pair_view(gu, num_tokens);\n"
     "            // The gate/up GEMMs record their own GP_gemm_C sites ahead of the down projection's;\n"),
    ("            args.emplace_back(GP_gemm_C, (void*) gu_n.select(0, 1).data_ptr());\n",
     "            args.emplace_back(GP_gemm_C, (void*) gu_n.select(0, 1).data_ptr());\n"
     "          }\n"),
])
patch("libtorch/mla_attention.cpp", [
    ("    if (p->i8_ok(x)) { dense_i8_gr(x, p->w8.value(), p->w8s.value(), y, p->w8y32.value(), p->w8ctr.value(), p->w8spw, graph); return; }\n",
     "    if (p->i8_ok(x)) { dense_i8_gr(x, p->w8.value(), p->w8s.value(), y, p->w8y32.value(), p->w8ctr.value(), p->w8spw, graph, p->lr_ptr()); return; }\n"),
    ("        dense_i8_gr(s.o, o_proj->w8.value(), o_proj->w8s.value(), y2, o_proj->w8y32.value(), o_proj->w8ctr.value(), o_proj->w8spw, graph);\n",
     "        dense_i8_gr(s.o, o_proj->w8.value(), o_proj->w8s.value(), y2, o_proj->w8y32.value(), o_proj->w8ctr.value(), o_proj->w8spw, graph, o_proj->lr_ptr());\n"),
])
# q_a and kv_a (both read x) as one two-segment int8 GEMV (dense_i8_dual) in the BC_MLA graph (EXL3_I8_QKV_DUAL, default on)
patch("libtorch/mla_attention.cpp", [
    ("static inline void mla_gemm_gr(",
     "static bool mla_dual_ok(const std::shared_ptr<BC_LinearEXL3>& qa, const std::shared_ptr<BC_LinearEXL3>& kva,\n"
     "                        const at::Tensor& x, const at::Tensor& yq, const at::Tensor& ykv)\n"
     "{\n"
     "    static const bool on = [] { const char* e = getenv(\"EXL3_I8_QKV_DUAL\"); return !(e && *e == '0'); }();\n"
     "    return on && qa && kva && qa->i8_ok(x) && kva->i8_ok(x) && !qa->w16 && !kva->w16 && qa->w8spw == 4 && kva->w8spw == 4 &&\n"
     "           !qa->lr && !kva->lr && yq.scalar_type() == at::kHalf && ykv.scalar_type() == at::kHalf &&\n"
     "           yq.is_contiguous() && ykv.is_contiguous() && yq.size(-1) % 64 == 0 && ykv.size(-1) % 64 == 0;\n"
     "}\n\n"
     "static inline void mla_gemm_gr("),
    ("        mla_gemm_gr(x_q, q_a_proj, s.q_a, xh_x, graph);\n",
     "        if (qkv_dual)\n"
     "            dense_i8_dual_gr(x2, q_a_proj->w8.value(), q_a_proj->w8s.value(), s.q_a, q_a_proj->w8y32.value(), q_a_proj->w8ctr.value(),\n"
     "                             kv_a_proj->w8.value(), kv_a_proj->w8s.value(), s.ckv_kpe, kv_a_proj->w8y32.value(),\n"
     "                             kv_a_proj->w8ctr.value(), graph);\n"
     "        else\n"
     "            mla_gemm_gr(x_q, q_a_proj, s.q_a, xh_x, graph);\n"),
    ("    at::Tensor x2 = x.view({R, hidden_size});\n",
     "    at::Tensor x2 = x.view({R, hidden_size});\n"
     "    const bool qkv_dual = mla_dual_ok(q_a_proj, kv_a_proj, x2, s.q_a, s.ckv_kpe);   // hecate\n"),
    ("        mla_gemm_gr(x_kv, kv_a_proj, s.ckv_kpe, xh_x, graph);\n",
     "        if (!qkv_dual) mla_gemm_gr(x_kv, kv_a_proj, s.ckv_kpe, xh_x, graph);   // else done with q_a\n"),
    ("    if (!kv_a_proj->w16)\n"
     "        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());\n",
     "    if (!kv_a_proj->w16 && !mla_dual_ok(q_a_proj, kv_a_proj, x.view({-1, x.size(-1)}), s.q_a, s.ckv_kpe))\n"
     "        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());\n"),
])

# MLA decode combine over EXL3_MLA_COMBINE_DSPLIT latent-width blocks (grid dim 1; the Triton kernel takes BLOCK_D)
patch("libtorch/mla_attention.cpp", [
    ("        s.k_combine->launch(s.programs, 1, 1, args, stream);\n",
     "        static const int cdsplit = [] { const char* e = getenv(\"EXL3_MLA_COMBINE_DSPLIT\"); return e && atoi(e) > 0 ? atoi(e) : 1; }();\n"
     "        s.k_combine->launch(s.programs, cdsplit, 1, args, stream);\n"),
])

patch("libtorch/linear_bc.h", [
    ('.def("set_i8", &BC_LinearEXL3::set_i8);',
     '.def("set_i8", &BC_LinearEXL3::set_i8)\n.def("set_lowrank", &BC_LinearEXL3::set_lowrank)\n.def("clear_lowrank", &BC_LinearEXL3::clear_lowrank);'),
])
patch("bindings.cpp", [
    ('    m.def("dense_i8", &dense_i8, "dense_i8");\n',
     '    m.def("dense_i8", &dense_i8, "dense_i8");\n    m.def("dense_i8_lr", &dense_i8_lr, "dense_i8_lr");\n'
     '    m.def("dense_i8_gu_act", &dense_i8_gu_act, "dense_i8_gu_act");\n'
     '    m.def("dense_i8_dual", &dense_i8_dual, "dense_i8_dual");\n'),
])
# MLA decode split floor (tiles of block_n per split) from EXL3_MLA_SPLIT_TILES (default 4 = stock rule). With
# 4 heads per program (EXL3_MLA_BLOCK_H_Q=4) one-tile splits halve the short-context split kernel (standalone sweep)
patch("libtorch/mla_attention.cpp", [
    ("    *num_splits = MAX(1, MIN(splits_cap, CEIL_DIVIDE(max_k_len, 4 * block_n)));\n",
     "    static const int tiles = [] { const char* e = getenv(\"EXL3_MLA_SPLIT_TILES\"); return e && atoi(e) > 0 ? atoi(e) : 4; }();\n"
     "    *num_splits = MAX(1, MIN(splits_cap, CEIL_DIVIDE(max_k_len, tiles * block_n)));\n"),
])
