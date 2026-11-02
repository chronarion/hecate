"""hecate: side branch for the BC MLA decode graph, behind BC_MLAttention.set_kvq(mode) (called from runtime/mla/bc_mla.py
under EXL3_MLA_KVQ; 0 = off).
 - mode bit 0, fused cache append (mla_kvq.cu): for quantized latent caches with a rotated 64-wide rope key, one kernel
   does the kv_a norm, the key rope, the group quantization and the paged scatter, on a side stream forked after the
   fused q_a + kv_a projection and joined before the absorb kernel; the absorb kernel (compiled with the query rope)
   rotates the query rope columns with the sin/cos table the append kernel wrote. The staging, rope, quant and scatter
   nodes leave the attention block's critical path.
 - mode bit 1 (with bit 0), full-indexer layers: the indexer-key stages (x staging, wk GEMM, key norm, key rope, plane
   append) follow the append on the side stream; they join before the sparse regime's first consumer of the staged x
   (the weights GEMM) or at the end of the block.
The side-branch kernels are launched at their old place in host order, so the module graph's node order (and with it
the sequential parameter patcher's site order) is unchanged; only their dependencies move to the fork point.
   python3 apply_kvq.py EXT_DIR SRC_DIR"""
import pathlib, shutil, sys
ext, srcdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


shutil.copy(srcdir / "mla_kvq.cu", ext / "mla_kvq.cu")
(ext / "mla_kvq.cuh").write_text('''#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include "graph.cuh"

// hecate: fused MLA decode cache append for quantized latent caches, see mla_kvq.cu
bool mla_kvq_supported(int kv_lora_rank, int qk_rope_head_dim, int rope_style, int bits);
void mla_kvq_gr(const at::Tensor& ckv_kpe, const at::Tensor& kv_norm_w, const at::Tensor& inv_freq, uint32_t position,
                const c10::optional<at::Tensor>& positions, const c10::optional<at::Tensor>& position_ids, int pid_stride,
                int rope_style, float attn_factor, float eps, const at::Tensor& block_table, const at::Tensor& cache_seqlens,
                at::Tensor& cache_ckv, at::Tensor& cache_scales, at::Tensor& cache_kpe, at::Tensor& sincos, int R, int q_len,
                int page_size, cudaStream_t stream, Graph* graph);
bool mla_kvq_side(int device, cudaStream_t* side, cudaEvent_t* ev, bool may_create);
void mla_kvq(at::Tensor ckv_kpe, at::Tensor kv_norm_w, at::Tensor inv_freq, int64_t position, c10::optional<at::Tensor> positions,
             int64_t rope_style, double attn_factor, double eps, at::Tensor block_table, at::Tensor cache_seqlens,
             at::Tensor cache_ckv, at::Tensor cache_scales, at::Tensor cache_kpe, at::Tensor sincos, int64_t q_len);
''')
sub(ext / "bindings.cpp", '#include "hmla.cuh"\n', '#include "hmla.cuh"\n#include "mla_kvq.cuh"\n')
sub(ext / "bindings.cpp", '    m.def("hmla_attn", &hmla_attn, "hmla_attn");\n',
    '    m.def("hmla_attn", &hmla_attn, "hmla_attn");\n    m.def("mla_kvq", &mla_kvq, "mla_kvq");\n')

h = ext / "libtorch/mla_attention.h"
sub(h, "    bool needs_configure(int bsz, int q_len, int regime);\n",
    "    // hecate: side branch of the decode graph (set_kvq, see mla_kvq.cu): bit 0 fused cache append + absorb kernel\n"
    "    // with the query rope, bit 1 full-indexer key stages on the same branch\n"
    "    int kvq_mode = 0;\n"
    "    void set_kvq(int mode);\n"
    "\n"
    "    bool needs_configure(int bsz, int q_len, int regime);\n")
sub(ext / "libtorch/mla_attention_bc.h", '.def("set_indexer", &BC_MLAttention::set_indexer,\n',
    '.def("set_kvq", &BC_MLAttention::set_kvq)\n.def("set_indexer", &BC_MLAttention::set_indexer,\n')

c = ext / "libtorch/mla_attention.cpp"
sub(c, '#include "../hmla.cuh"\n', '#include "../hmla.cuh"\n#include "../mla_kvq.cuh"\n#include <c10/cuda/CUDAStream.h>\n')

sub(c, """void BC_MLAttention::run_gr
(""", """void BC_MLAttention::set_kvq(int mode)
{
    if (mode)
    {
        const int bits = quant_cache ? (int) (cache_ckv.size(-1) / (kv_lora_rank / 32)) : 0;
        TORCH_CHECK((mode & 1) && quant_cache && cache_scales.has_value() && rope_active && rotate_dims == 1 &&
                    l4_scaling_beta <= 0.0f && mla_kvq_supported(kv_lora_rank, qk_rope_head_dim, rope_style, bits),
                    "BC_MLAttention: fused cache append not supported for this configuration");
    }
    kvq_mode = mode;
}

void BC_MLAttention::run_gr
(""")

# fork point: right after the fused q_a + kv_a projection
sub(c, """    const bool qkv_dual = mla_dual_ok(q_a_proj, kv_a_proj, x2, s.q_a, s.ckv_kpe);   // hecate
""", """    const bool qkv_dual = mla_dual_ok(q_a_proj, kv_a_proj, x2, s.q_a, s.ckv_kpe);   // hecate

    // hecate (set_kvq): the fused cache append replaces the staging, rope, quant and scatter stages. With the fused
    // q_a + kv_a projection it forks onto a side stream right after it (so it runs under the q_a norm and the q_b GEMV)
    // and joins before the absorb kernel. The side stream and events exist once created by an uncaptured run; a
    // capture that finds none runs the append in line
    const bool kvq = (kvq_mode & 1) && quant_cache && cache_scales.has_value() && rope_active;
    cudaStream_t kvq_side = nullptr;
    cudaEvent_t kvq_ev[3] = {};
    bool kvq_fork = false;
    if (kvq && qkv_dual && q_a_proj)
    {
        cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &cap_st));
        kvq_fork = mla_kvq_side(x.get_device(), &kvq_side, kvq_ev, cap_st == cudaStreamCaptureStatusNone);
        static bool kvq_told = false;
        if (!kvq_fork && !kvq_told)
        {
            kvq_told = true;
            printf("BC_MLAttention: fused cache append runs in line (captured before any uncaptured run made the side stream)\\n");
            fflush(stdout);
        }
    }
    // bit 1: the full-indexer key stages follow the append on the side stream (kpool planes stay in line)
    const bool kvq_idx = kvq_fork && (kvq_mode & 2) && idx_mode == 1 && !index_kpool;
    bool kvq_idx_joined = false;
""")
sub(c, """                             kv_a_proj->w8ctr.value(), graph);
        else
            mla_gemm_gr(x_q, q_a_proj, s.q_a, xh_x, graph);
""", """                             kv_a_proj->w8ctr.value(), graph);
        else
            mla_gemm_gr(x_q, q_a_proj, s.q_a, xh_x, graph);
        if (kvq_fork) cuda_check(cudaEventRecord(kvq_ev[0], stream));   // hecate: the side branch forks here
""")

# staging + rope only on the unfused path
sub(c, """    // statics, so the captured node needs no patching
    {
        std::vector<void*> args =
        {
            (void*) s.q_full.data_ptr(),""", """    // statics, so the captured node needs no patching
    if (!kvq)
    {
        std::vector<void*> args =
        {
            (void*) s.q_full.data_ptr(),""")
sub(c, """    if (rope_active)
    {
        c10::optional<at::Tensor> out_k = s.kpe4;""", """    if (rope_active && !kvq)
    {
        c10::optional<at::Tensor> out_k = s.kpe4;""")

# the fused append (side branch or in line), then the absorb kernel with the query rope
sub(c, """    // Absorb W_UK into the queries, per head, straight from the flat layout
    {
        std::vector<void*> args =
        {
            (void*) s.q_full.data_ptr(),
            (void*) w_uk_flat.data_ptr(),
            (void*) s.q_lat.data_ptr(),
            (void*) (intptr_t) R,
        };
""", """    // hecate (set_kvq): fused cache append, launched after the q_b GEMV so its patch sites follow the projections' in
    // node order (the side branch depends on the projection fork point only); it writes the rows' rope sin/cos table
    // into the ckv static, which the unfused path uses for the normalized latent
    if (kvq)
    {
        const int kvq_pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
        cudaStream_t ks = stream;
        if (kvq_fork)
        {
            cuda_check(cudaStreamWaitEvent(kvq_side, kvq_ev[0], 0));
            ks = kvq_side;
        }
        at::Tensor cs = cache_scales.value();
        mla_kvq_gr(s.ckv_kpe, kv_norm_w, inv_freq, (uint32_t) position, positions, position_ids, kvq_pid_stride, rope_style,
                   attn_factor, norm_eps, block_table, cache_seqlens, cache_ckv, cs, cache_kpe, s.ckv, R, q_len,
                   page_size, ks, graph);
        if (kvq_fork)
        {
            cuda_check(cudaEventRecord(kvq_ev[1], kvq_side));
            cuda_check(cudaStreamWaitEvent(stream, kvq_ev[1], 0));
        }
        dbg("kvq");
    }

    // Absorb W_UK into the queries, per head, straight from the flat layout
    {
        std::vector<void*> args =
        {
            (void*) s.q_full.data_ptr(),
            (void*) w_uk_flat.data_ptr(),
            (void*) s.q_lat.data_ptr(),
            (void*) (intptr_t) R,
        };
        if (kvq)   // hecate: rope variant (q_pe out, sin/cos table)
        {
            args.push_back((void*) s.q_pe.data_ptr());
            args.push_back((void*) s.ckv.data_ptr());
        }
""")
sub(c, """    if (quant_cache)
    {
        quant_cache_cont_gr(s.ckv, s.qtmp, s.stmp, 0.0f, graph);""", """    if (kvq) {}   // hecate: appended by the fused kernel above
    else if (quant_cache)
    {
        quant_cache_cont_gr(s.ckv, s.qtmp, s.stmp, 0.0f, graph);""")

# bit 1: indexer-key stages on the side stream (eager: current-stream guard; module capture: the capture stream swapped)
sub(c, """    // DSA indexer keys (full-indexer layers, both regimes): stage x for the unquantized
    // GEMMs, project + biased-LayerNorm + partial rope the chunk's keys, and append them to
    // the paged side plane so the selection has complete history once it activates
    if (idx_mode == 1)
    {""", """    // hecate (set_kvq bit 1): the stages below run on the side stream, after the fused append in stream order. The _gr
    // helpers launch on the current stream (eager / outer capture) or the module graph's capture stream, so both are
    // pointed at the side stream for the block. The wk GEMM is the only cuBLAS call in flight meanwhile (the main
    // stream runs int8 GEMVs and Triton kernels until the join), so the shared cuBLAS workspace is not contended
    cudaStream_t kvq_main_stream = stream;
    cudaStream_t kvq_saved_cs = nullptr;
    c10::optional<c10::cuda::CUDAStreamGuard> kvq_sg;
    if (kvq_idx)
    {
        if (graph) { kvq_saved_cs = graph->capture_stream; graph->capture_stream = kvq_side; }
        else kvq_sg.emplace(c10::cuda::getStreamFromExternal(kvq_side, x.get_device()));
        stream = kvq_side;
    }

    // DSA indexer keys (full-indexer layers, both regimes): stage x for the unquantized
    // GEMMs, project + biased-LayerNorm + partial rope the chunk's keys, and append them to
    // the paged side plane so the selection has complete history once it activates
    if (idx_mode == 1)
    {""")
sub(c, """    // hecate: the tcgen05 kernel covers one sequence over the Q6 cache with GLM-5.3's MLA shapes
    const bool hmla_on = hmla_scale > 0.f && bsz == 1 && quant_cache && cache_scales && num_q_heads == 64 &&""",
"""    if (kvq_idx)
    {
        cuda_check(cudaEventRecord(kvq_ev[2], kvq_side));
        if (graph) graph->capture_stream = kvq_saved_cs;
        else kvq_sg.reset();
        stream = kvq_main_stream;
    }
    auto kvq_idx_join = [&]()
    {
        if (kvq_idx && !kvq_idx_joined)
        {
            cuda_check(cudaStreamWaitEvent(stream, kvq_ev[2], 0));
            kvq_idx_joined = true;
        }
    };

    // hecate: the tcgen05 kernel covers one sequence over the Q6 cache with GLM-5.3's MLA shapes
    const bool hmla_on = hmla_scale > 0.f && bsz == 1 && quant_cache && cache_scales && num_q_heads == 64 &&""")
sub(c, """            at::Tensor wts_rows = s.wts;   // full padded height (cuBLASLt M >= 8)
            hgemm_gr(s.x_st, idx_weights_w.value(), wts_rows, graph);""",
"""            at::Tensor wts_rows = s.wts;   // full padded height (cuBLASLt M >= 8)
            kvq_idx_join();   // hecate: staged x and the new plane rows (side branch)
            hgemm_gr(s.x_st, idx_weights_w.value(), wts_rows, graph);""")
sub(c, """    dbg("o_proj");
}
""", """    dbg("o_proj");
    kvq_idx_join();   // hecate: the side branch's indexer-key stages (dense regime)
}
""")
print("apply_kvq: ok")
