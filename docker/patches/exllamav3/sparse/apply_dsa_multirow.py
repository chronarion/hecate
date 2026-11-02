"""hecate: BC_MLAttention.set_force_multirow(bool): single-sequence sparse (DSA) slots take the batched (MULTIROW) path, whose
scoring / top-k kernels read their scan widths and causal bounds from the device state array (filled from cache_seqlens by
dsa_seq_state) instead of host scalars patched into the module graph per call. A whole decode step in the sparse regime then
captures and replays as one CUDA graph (exl3_vgraph, EXL3_VERIFY_GRAPH_SPARSE); bc_mla.py sets it with EXL3_DSA_MULTIROW=1.
   python3 apply_dsa_multirow.py EXT_DIR"""
import pathlib, sys
ext = pathlib.Path(sys.argv[1])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


h = ext / "libtorch/mla_attention.h"
sub(h, "    bool needs_configure(int bsz, int q_len, int regime);\n",
    "    // hecate: single-sequence sparse slots on the batched (device-state) path, see apply_dsa_multirow.py\n"
    "    bool force_multirow = false;\n"
    "    void set_force_multirow(bool v) { force_multirow = v; }\n"
    "\n"
    "    bool needs_configure(int bsz, int q_len, int regime);\n")
sub(ext / "libtorch/mla_attention_bc.h", '.def("set_indexer", &BC_MLAttention::set_indexer,\n',
    '.def("set_force_multirow", &BC_MLAttention::set_force_multirow)\n.def("set_indexer", &BC_MLAttention::set_indexer,\n')
c = ext / "libtorch/mla_attention.cpp"
sub(c, "        bool multirow = bsz > 1;\n", "        bool multirow = bsz > 1 || force_multirow;\n", count=2)
sub(c, 'TORCH_CHECK(bsz == 1 || dsa_arr, "BC_MLAttention: batched sparse slot without state array");',
    'TORCH_CHECK((bsz == 1 && !force_multirow) || dsa_arr, "BC_MLAttention: batched sparse slot without state array");')
# the tcgen05 gather (hmla_sparse_gr, single sequence) records block table + indices only; the Triton gather's page-count
# param must then not be passed (it was only ever emitted for bsz > 1, where hmla is off)
sub(c, """        if (multirow)
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);   // gather
""", """        const bool hmla_on = hmla_scale > 0.f && bsz == 1 && quant_cache && cache_scales && num_q_heads == 64 &&
                             kv_lora_rank == 512 && qk_rope_head_dim == 64 && page_size == 256 &&
                             cache_ckv.numel() == cache_kpe.numel() / 64 * 96;   // as in run_gr
        if (multirow && !hmla_on)
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);   // gather
""")
print("apply_dsa_multirow: ok")
