"""hecate: tcgen05 sparse attention for GLM-5.3's MLA (hmla.cu). Adds `hmla_attn` to the extension and routes the DSA
gathered attention through it, for one sequence over the Q6 latent cache with GLM-5.3's MLA shapes (anything else keeps
the Triton kernels):
 - prefill (EXL3_HMLA_PREFILL=1, default): MLAAttention._attend_sparse, prompt chunks past index_topk tokens
 - decode (EXL3_HMLA_DECODE=1, default): the sparse regime of the BC MLA graph, in place of the Triton split + combine;
   each query row's key list is split across EXL3_HMLA_NS CTAs (default 16) and merged by a combine kernel
The dense regime (contexts up to index_topk) keeps the Triton kernels.
   python3 apply_hmla.py EXT_DIR SP_DIR SRC_DIR"""
import pathlib, shutil, sys
ext, sp, srcdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


shutil.copy(srcdir / "hmla.cu", ext / "hmla.cu")
(ext / "hmla.cuh").write_text('''#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include "graph.cuh"

// hecate: tcgen05 sparse attention for the MLA decode (graph-aware) and prefill, see hmla.cu
void hmla_sparse_gr(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& ckv, const at::Tensor& ckv_s,
                    const at::Tensor& kpe, const at::Tensor& block_table, const at::Tensor& indices, float scale, at::Tensor& out,
                    int ns, cudaStream_t stream, Graph* graph);
void hmla_attn(at::Tensor q_lat, at::Tensor q_pe, at::Tensor ckv, at::Tensor ckv_s, at::Tensor kpe, at::Tensor block_table,
               at::Tensor indices, double scale, int64_t ns, at::Tensor out);
''')
sub(ext / "bindings.cpp", '#include "hgemm.cuh"\n', '#include "hgemm.cuh"\n#include "hmla.cuh"\n')
sub(ext / "bindings.cpp", '    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");\n',
    '    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");\n    m.def("hmla_attn", &hmla_attn, "hmla_attn");\n')

# prefill: the DSA gathered attention of prompt chunks
sub(sp / "modules/mla_attn.py", '''        from .attention_fn.dsa_triton import dsa_attn
''', '''        if _HMLA_PREFILL and qc is not None and qc[1] == 6 and bsz == 1 and block_table.shape[0] == 1 and \\
                self.num_q_heads == 64 and self.kv_lora_rank == 512 and self.qk_rope_head_dim == 64:
            # hecate: tcgen05 kernel, one CTA per query row (all 64 heads), the Q6 rows dequantized in shared memory
            from ..ext import exllamav3_ext as _hx
            R_ = bsz * seqlen
            o_lat = torch.empty((self.num_q_heads, R_, self.kv_lora_rank), dtype = torch.half, device = q_lat.device)
            _hx.hmla_attn(q_lat.contiguous(), q_pe.reshape(R_, self.num_q_heads, self.qk_rope_head_dim).contiguous(),
                          ckv_cache, qc[0], kpe_cache, block_table, indices.contiguous(), float(self.sm_scale), 1, o_lat)
            return o_lat
        from .attention_fn.dsa_triton import dsa_attn
''')
sub(sp / "modules/mla_attn.py", "import os\n", 'import os\n_HMLA_PREFILL = os.environ.get("EXL3_HMLA_PREFILL", "1") == "1"   # hecate\n')

# decode: the sparse regime of BC_MLAttention
h = ext / "libtorch/mla_attention.h"
sub(h, "    bool needs_configure(int bsz, int q_len, int regime);\n",
    "    // hecate: tcgen05 sparse attention instead of the Triton split + combine (set_hmla; scale 0 = off)\n"
    "    float hmla_scale = 0.f;\n"
    "    int hmla_ns = 16;\n"
    "    void set_hmla(double scale, int ns) { hmla_scale = (float) scale; hmla_ns = ns; }\n"
    "\n"
    "    bool needs_configure(int bsz, int q_len, int regime);\n")
sub(ext / "libtorch/mla_attention_bc.h", '.def("set_indexer", &BC_MLAttention::set_indexer,\n',
    '.def("set_hmla", &BC_MLAttention::set_hmla)\n.def("set_indexer", &BC_MLAttention::set_indexer,\n')

c = ext / "libtorch/mla_attention.cpp"
sub(c, '#include "../dsa_topk.cuh"\n', '#include "../dsa_topk.cuh"\n#include "../hmla.cuh"\n')
sub(c, """    if (regime == 1)
    {
        // Sparse regime: score + select""", """    // hecate: the tcgen05 kernel covers one sequence over the Q6 cache with GLM-5.3's MLA shapes
    const bool hmla_on = hmla_scale > 0.f && bsz == 1 && quant_cache && cache_scales && num_q_heads == 64 &&
                         kv_lora_rank == 512 && qk_rope_head_dim == 64 && page_size == 256 &&
                         cache_ckv.numel() == cache_kpe.numel() / 64 * 96;

    if (regime == 1)
    {
        // Sparse regime: score + select""")
sub(c, """        {
            // Gathered attention: causality lives in the selection, V is the latent. The""", """        if (hmla_on)
        {
            at::Tensor idx_h = ext_indices ? ext_indices.value() : s.indices;
            hmla_sparse_gr(s.q_lat, s.q_pe, cache_ckv, cache_scales.value(), cache_kpe, block_table, idx_h, hmla_scale, s.o_lat,
                           hmla_ns, stream, graph);
            dbg("hmla_sparse");
        }
        else
        {
        {
            // Gathered attention: causality lives in the selection, V is the latent. The""")
sub(c, """        dbg("dsa_combine");
    }
    else
    {
""", """        dbg("dsa_combine");
        }
    }
    else
    {
""")

# (the decode switch, BC_MLAttention.set_hmla, is called from runtime/mla/bc_mla.py)
print("apply_hmla: ok")
