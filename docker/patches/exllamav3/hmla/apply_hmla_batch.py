"""hecate: the tcgen05 sparse attention (hmla.cu) for decode batches of several sequences (EXL3_HMLA_BATCH=1). Until now
the kernel ran for one sequence only and a batch in the sparse (DSA) regime fell back to the Triton gather + combine.
Each CTA works on one query row r of one split; with B > 1 sequences x q_len rows each, row r belongs to sequence
r / q_len, so the kernel reads its block table row at (r / q_len) * width (one sequence: stride 0, as before). The
stride is the block-table width, a graph parameter of the BC module graphs (pushed with the gather's GP_attn_num_pages).
The selection already holds -1 for unused entries of short rows, which the kernel masks.
   python3 apply_hmla_batch.py EXT_DIR      (after apply_hmla.py)"""
import pathlib, sys
ext = pathlib.Path(sys.argv[1])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


cu = ext / "hmla.cu"
# kernel: rows per sequence as the last argument, block-table row of the row's sequence
sub(cu, "    int rot_q, float sl2, int ns, half* __restrict__ out, float* __restrict__ part_o, float* __restrict__ part_ml, int dbg)\n{",
    "    int rot_q, float sl2, int ns, half* __restrict__ out, float* __restrict__ part_o, float* __restrict__ part_ml, int dbg,\n"
    "    int rq)\n{")
sub(cu, "        return e < 0 ? -1 : __ldg(bt + (size_t) r * bt_stride + e / PAGE) * PAGE + e % PAGE;",
    "        return e < 0 ? -1 : __ldg(bt + (size_t) (r / rq) * bt_stride + e / PAGE) * PAGE + e % PAGE;")
sub(cu, """static void hmla_launch(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& ckv, const at::Tensor& ckv_s,
                        const at::Tensor& kpe, const at::Tensor& block_table, const int* idx, int k_pad, float scale,
                        at::Tensor& out, int ns, cudaStream_t stream, Graph* graph)
{
    const int R = (int) q_lat.size(1);""", """static void hmla_launch(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& ckv, const at::Tensor& ckv_s,
                        const at::Tensor& kpe, const at::Tensor& block_table, const int* idx, int k_pad, float scale,
                        at::Tensor& out, int ns, cudaStream_t stream, Graph* graph, int bsz = 1)
{
    const int R = (int) q_lat.size(1);
    TORCH_CHECK(bsz >= 1 && R % bsz == 0 && (bsz == 1 || block_table.size(0) == bsz), "hmla: batch shape");
    const int bt_stride = bsz > 1 ? (int) block_table.size(1) : 0;""")
sub(cu, "        idx, nullptr, 0, k_pad, k_pad, 0, R, 1, scale * 1.4426950408889634f, ns, (half*) out.data_ptr(), po, pml, 0);",
    "        idx, nullptr, bt_stride, k_pad, k_pad, 0, R, 1, scale * 1.4426950408889634f, ns, (half*) out.data_ptr(), po, pml, 0,\n"
    "        R / bsz);")
sub(cu, """        graph->record_param((void*) hmla_kernel, GP_dsa_indices, 6);
        graph->record_param((void*) hmla_kernel, GP_end, 0);""", """        graph->record_param((void*) hmla_kernel, GP_dsa_indices, 6);
        if (bsz > 1) graph->record_param((void*) hmla_kernel, GP_attn_num_pages, 8, 4);   // block-table row stride
        graph->record_param((void*) hmla_kernel, GP_end, 0);""")
sub(cu, """                    const at::Tensor& kpe, const at::Tensor& block_table, const at::Tensor& indices, float scale, at::Tensor& out,
                    int ns, cudaStream_t stream, Graph* graph)
{
    hmla_launch(q_lat, q_pe, ckv, ckv_s, kpe, block_table, (const int*) indices.data_ptr(), (int) indices.size(1), scale, out, ns,
                stream, graph);""", """                    const at::Tensor& kpe, const at::Tensor& block_table, const at::Tensor& indices, float scale, at::Tensor& out,
                    int ns, cudaStream_t stream, Graph* graph, int bsz)
{
    hmla_launch(q_lat, q_pe, ckv, ckv_s, kpe, block_table, (const int*) indices.data_ptr(), (int) indices.size(1), scale, out, ns,
                stream, graph, bsz);""")

# test entry: the decode launch for B sequences of q_len rows each (tools/dev/test_hmla_batch.py compares it with B
# one-sequence launches; the rows must be bit-identical)
sub(cu, "#include <torch/extension.h>", """// hecate (test): hmla_sparse_gr for a batch, outside any graph
void hmla_attn_batch(at::Tensor q_lat, at::Tensor q_pe, at::Tensor ckv, at::Tensor ckv_s, at::Tensor kpe, at::Tensor block_table,
                     at::Tensor indices, double scale, int64_t ns, at::Tensor out, int64_t bsz)
{
    const at::cuda::OptionalCUDAGuard guard(q_lat.device());
    hmla_launch(q_lat, q_pe, ckv, ckv_s, kpe, block_table, (const int*) indices.data_ptr(), (int) indices.size(1), (float) scale, out,
                (int) ns, at::cuda::getCurrentCUDAStream().stream(), nullptr, (int) bsz);
}

#include <torch/extension.h>""")

h = ext / "hmla.cuh"
sub(h, "                    int ns, cudaStream_t stream, Graph* graph);",
    "                    int ns, cudaStream_t stream, Graph* graph, int bsz = 1);")
s_ = h.read_text(); h.write_text(s_ + """void hmla_attn_batch(at::Tensor q_lat, at::Tensor q_pe, at::Tensor ckv, at::Tensor ckv_s, at::Tensor kpe, at::Tensor block_table,
                     at::Tensor indices, double scale, int64_t ns, at::Tensor out, int64_t bsz);
""")
sub(ext / "bindings.cpp", '    m.def("hmla_attn", &hmla_attn, "hmla_attn");\n',
    '    m.def("hmla_attn", &hmla_attn, "hmla_attn");\n    m.def("hmla_attn_batch", &hmla_attn_batch, "hmla_attn_batch");\n')

c = ext / "libtorch/mla_attention.cpp"
gate = "hmla_scale > 0.f && bsz == 1 && quant_cache && cache_scales && num_q_heads == 64 &&"
new_gate = "hmla_scale > 0.f && (bsz == 1 || (hmla_batch_on() && bsz * q_len <= 8)) && quant_cache && cache_scales && num_q_heads == 64 &&"
sub(c, gate, new_gate, count=2)
sub(c, '#include "../hmla.cuh"\n', '''#include "../hmla.cuh"
#include <cstdlib>
// hecate (EXL3_HMLA_BATCH=1): the tcgen05 sparse attention for batches of up to 8 rows too
static bool hmla_batch_on() { static const bool on = [] { const char* e = std::getenv("EXL3_HMLA_BATCH"); return e && std::atoi(e) == 1; }(); return on; }
''')
sub(c, """            hmla_sparse_gr(s.q_lat, s.q_pe, cache_ckv, cache_scales.value(), cache_kpe, block_table, idx_h, hmla_scale, s.o_lat,
                           hmla_ns, stream, graph);""", """            hmla_sparse_gr(s.q_lat, s.q_pe, cache_ckv, cache_scales.value(), cache_kpe, block_table, idx_h, hmla_scale, s.o_lat,
                           hmla_ns, stream, graph, bsz);""")
sub(c, """        if (multirow && !hmla_on)
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);   // gather""", """        if (multirow && !hmla_on)
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);   // gather
        if (hmla_on && bsz > 1)
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);   // hmla block-table row stride""")
print("apply_hmla_batch: ok")
