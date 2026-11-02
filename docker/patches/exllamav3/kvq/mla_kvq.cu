// hecate: fused MLA decode cache append for quantized latent caches (EXL3_MLA_KVQ). One CTA per appended row does the
// kv_a RMSNorm of the latent, the rope of the shared key, the H32-rotated group quantization and the paged scatter, in
// place of the graph's staging kernel (kv half), the rope kernel (key head), quant_cache_cont and the quant scatter.
// It also writes the row's rope sin/cos table, which the absorb kernel then applies to the query rope columns. The BC
// MLA graph launches it on a side branch forked after the latent projection, so it runs under the q_a norm and q_b
// GEMV instead of on the attention block's critical path.
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"
#include "rope.cuh"
#include "cache/q_cache_kernels.cuh"
#include "mla_kvq.cuh"

namespace {

constexpr int KVQ_D_C = 512;   // latent width: 16 groups of 32, 4 warps x 4 groups
constexpr int KVQ_D_R = 64;    // rope key width: 32 rotation pairs

template <int rope_mode, int bits>
__global__ __launch_bounds__(128)
void mla_kvq_kernel
(
    const half* __restrict__ ckv_kpe,       // (R, ckv_stride): latent (D_c) | rope key (D_r), kv_a output
    const half* __restrict__ kv_norm_w,     // (D_c,)
    const float* __restrict__ inv_freq,     // (D_r / 2,)
    const int position,
    const uint32_t* __restrict__ positions,
    const uint32_t* __restrict__ position_ids,
    const int pid_stride,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    const int num_pages_per_seq,
    uint32_t* __restrict__ cache_ckv,       // (pages * page_size, D_c / 32 * bits)
    half* __restrict__ cache_scales,        // (pages * page_size, D_c / 32)
    half* __restrict__ cache_kpe,           // (pages * page_size, D_r)
    float* __restrict__ sincos,             // out (R, D_r): sin of the 32 pairs, then cos
    const int ckv_stride,
    const int q_len,
    const int page_size,
    const float attn_factor,
    const float eps
)
{
    constexpr int NG = KVQ_D_C / 32;
    __shared__ half sh_y[KVQ_D_C];
    __shared__ uint32_t sh_pack[4][32];
    __shared__ float sh_ss[4];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int batch = row / q_len;
    const int tpos = row - batch * q_len;
    const half* src = ckv_kpe + (size_t) row * ckv_stride;

    // Cache slot of the row (as _mla_kv_quant_scatter_kernel)
    const int abs_pos = cache_seqlens[batch] + tpos;
    const int page = abs_pos / page_size;
    const int page_off = abs_pos - page * page_size;
    const int phys = block_table[batch * num_pages_per_seq + page];
    const size_t tok = (size_t) phys * page_size + page_off;

    // Latent: 4 values per thread, kv_a RMSNorm in fp32 (as _mla_stage_kernel)
    const uint2 raw = ((const uint2*) src)[t];
    const half2 x01 = *(const half2*) &raw.x;
    const half2 x23 = *(const half2*) &raw.y;
    float x[4] = { __low2float(x01), __high2float(x01), __low2float(x23), __high2float(x23) };
    float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, i);
    if (lane == 0) sh_ss[warp] = ss;

    // Rope key (warp 0, one rotation pair per lane), published sin/cos table for the query side (as rope_kernel)
    if (warp == 0)
    {
        int pos = tpos + position;
        if (positions) pos = tpos + (int) positions[batch];
        else if (position_ids) pos = (int) position_ids[(batch * q_len + tpos) * pid_stride];
        float sin, cos;
        sincos_accurate(inv_freq[lane] * __int2float_rn(pos), &sin, &cos);
        sin *= attn_factor;
        cos *= attn_factor;
        sincos[row * KVQ_D_R + lane] = sin;
        sincos[row * KVQ_D_R + KVQ_D_R / 2 + lane] = cos;
        const half* kpe = src + KVQ_D_C;
        half* dst = cache_kpe + tok * KVQ_D_R;
        if constexpr (rope_mode == ROPESTYLE_GPTJ)
        {
            half2 v = ((const half2*) kpe)[lane];
            float v1 = __low2float(v);
            float v2 = __high2float(v);
            float r1 = v1 * cos - v2 * sin;
            float r2 = v2 * cos + v1 * sin;
            ((half2*) dst)[lane] = __floats2half2_rn(r1, r2);
        }
        else
        {
            float v1 = __half2float(kpe[lane]);
            float v2 = __half2float(kpe[lane + KVQ_D_R / 2]);
            float r1 = v1 * cos - v2 * sin;
            float r2 = v2 * cos + v1 * sin;
            dst[lane] = __float2half_rn(r1);
            dst[lane + KVQ_D_R / 2] = __float2half_rn(r2);
        }
    }
    __syncthreads();

    const float rmf = 1.0f / sqrtf((sh_ss[0] + sh_ss[1] + sh_ss[2] + sh_ss[3]) / (float) KVQ_D_C + eps);
    const uint2 wraw = ((const uint2*) kv_norm_w)[t];
    const half2 w01 = *(const half2*) &wraw.x;
    const half2 w23 = *(const half2*) &wraw.y;
    const float w[4] = { __low2float(w01), __high2float(w01), __low2float(w23), __high2float(w23) };
    half2 y01 = __floats2half2_rn(x[0] * w[0] * rmf, x[1] * w[1] * rmf);
    half2 y23 = __floats2half2_rn(x[2] * w[2] * rmf, x[3] * w[3] * rmf);
    ((half2*) sh_y)[t * 2] = y01;
    ((half2*) sh_y)[t * 2 + 1] = y23;
    __syncthreads();

    // Quantize the normalized latent straight into the row's cache slot: warp w covers groups 4w .. 4w+3
    quant_block_x4<bits>(sh_y + warp * 128, cache_ckv + tok * (NG * bits) + warp * 4 * bits,
                         cache_scales + tok * NG + warp * 4, sh_pack[warp], 4, 0.0f);
}

template <int rope_mode>
void* kvq_kernel_for_bits(int bits)
{
    switch (bits)
    {
        case 4: return (void*) mla_kvq_kernel<rope_mode, 4>;
        case 5: return (void*) mla_kvq_kernel<rope_mode, 5>;
        case 6: return (void*) mla_kvq_kernel<rope_mode, 6>;
        case 8: return (void*) mla_kvq_kernel<rope_mode, 8>;
        default: return nullptr;
    }
}

}  // namespace

bool mla_kvq_supported(int kv_lora_rank, int qk_rope_head_dim, int rope_style, int bits)
{
    return kv_lora_rank == KVQ_D_C && qk_rope_head_dim == KVQ_D_R &&
           (rope_style == ROPESTYLE_GPTJ || rope_style == ROPESTYLE_NEOX) &&
           (bits == 4 || bits == 5 || bits == 6 || bits == 8);
}

void mla_kvq_gr
(
    const at::Tensor& ckv_kpe,
    const at::Tensor& kv_norm_w,
    const at::Tensor& inv_freq,
    uint32_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int pid_stride,
    int rope_style,
    float attn_factor,
    float eps,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    at::Tensor& cache_ckv,
    at::Tensor& cache_scales,
    at::Tensor& cache_kpe,
    at::Tensor& sincos,
    int R,
    int q_len,
    int page_size,
    cudaStream_t stream,
    Graph* graph
)
{
    const int bits = (int) (cache_ckv.size(-1) / (KVQ_D_C / 32));
    void* kernel_ptr = rope_style == ROPESTYLE_GPTJ ? kvq_kernel_for_bits<ROPESTYLE_GPTJ>(bits)
                                                    : kvq_kernel_for_bits<ROPESTYLE_NEOX>(bits);
    TORCH_CHECK(kernel_ptr, "mla_kvq: unsupported cache bitrate ", bits);
    TORCH_CHECK(ckv_kpe.stride(0) % 4 == 0 && ckv_kpe.size(1) >= KVQ_D_C + KVQ_D_R, "mla_kvq: ckv_kpe layout");
    TORCH_CHECK(sincos.numel() * sincos.element_size() >= (int64_t) R * KVQ_D_R * 4, "mla_kvq: sincos too small");

    const half* p_ckv_kpe = (const half*) ckv_kpe.data_ptr();
    const half* p_norm = (const half*) kv_norm_w.data_ptr();
    const float* p_inv_freq = (const float*) inv_freq.data_ptr();
    int p_position = (int) position;
    const uint32_t* p_positions = positions ? (const uint32_t*) positions.value().data_ptr() : nullptr;
    const uint32_t* p_position_ids = position_ids ? (const uint32_t*) position_ids.value().data_ptr() : nullptr;
    const int* p_block_table = (const int*) block_table.data_ptr();
    const int* p_seqlens = (const int*) cache_seqlens.data_ptr();
    int p_num_pages = (int) block_table.size(1);
    uint32_t* p_cache_ckv = (uint32_t*) cache_ckv.data_ptr();
    half* p_cache_scales = (half*) cache_scales.data_ptr();
    half* p_cache_kpe = (half*) cache_kpe.data_ptr();
    float* p_sincos = (float*) sincos.data_ptr();
    int p_ckv_stride = (int) ckv_kpe.stride(0);
    int p_q_len = q_len;
    int p_page_size = page_size;
    float p_attn_factor = attn_factor;
    float p_eps = eps;

    // Argument indices for the graph patch sites: inv_freq 2, position 3, positions 4, position_ids 5, pid_stride 6,
    // block_table 7, cache_seqlens 8, num_pages 9
    void* args[] = { &p_ckv_kpe, &p_norm, &p_inv_freq, &p_position, &p_positions, &p_position_ids, &pid_stride,
                     &p_block_table, &p_seqlens, &p_num_pages, &p_cache_ckv, &p_cache_scales, &p_cache_kpe,
                     &p_sincos, &p_ckv_stride, &p_q_len, &p_page_size, &p_attn_factor, &p_eps };
    cuda_check(cudaLaunchKernel(kernel_ptr, dim3(R), dim3(128), args, 0, stream));

    if (graph)
    {
        // Same order as the rope stage + cache append sites this kernel replaces (BC_MLAttention::run's param list)
        graph->record_param(kernel_ptr, GP_rope_inv_freq, 2);
        graph->record_param(kernel_ptr, GP_rope_position, 3, 4);
        graph->record_param(kernel_ptr, GP_rope_positions, 4);
        graph->record_param(kernel_ptr, GP_rope_position_ids, 5);
        graph->record_param(kernel_ptr, GP_rope_pid_stride, 6, 4);
        graph->record_param(kernel_ptr, GP_attn_block_table, 7);
        graph->record_param(kernel_ptr, GP_attn_seqlens, 8);
        graph->record_param(kernel_ptr, GP_attn_num_pages, 9, 4);
        graph->record_param(kernel_ptr, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}

// Side branch for the graph: one non-blocking stream and three events (fork, append join, indexer join) per device,
// created by an uncaptured run
bool mla_kvq_side(int device, cudaStream_t* side, cudaEvent_t* ev, bool may_create)
{
    static cudaStream_t s_side[16] = {};
    static cudaEvent_t s_ev[16][3] = {};
    if (device < 0 || device >= 16) return false;
    if (!s_side[device])
    {
        if (!may_create) return false;
        cuda_check(cudaStreamCreateWithFlags(&s_side[device], cudaStreamNonBlocking));
        for (int i = 0; i < 3; ++i) cuda_check(cudaEventCreateWithFlags(&s_ev[device][i], cudaEventDisableTiming));
    }
    *side = s_side[device];
    for (int i = 0; i < 3; ++i) ev[i] = s_ev[device][i];
    return true;
}

// Python entry for tests: the eager kernel on the current stream (scalar position or a positions tensor)
void mla_kvq
(
    at::Tensor ckv_kpe,
    at::Tensor kv_norm_w,
    at::Tensor inv_freq,
    int64_t position,
    c10::optional<at::Tensor> positions,
    int64_t rope_style,
    double attn_factor,
    double eps,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    at::Tensor cache_ckv,
    at::Tensor cache_scales,
    at::Tensor cache_kpe,
    at::Tensor sincos,
    int64_t q_len
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ckv_kpe.device());
    c10::optional<at::Tensor> no_pid;
    mla_kvq_gr(ckv_kpe, kv_norm_w, inv_freq, (uint32_t) position, positions, no_pid, 1, (int) rope_style, (float) attn_factor,
               (float) eps, block_table, cache_seqlens, cache_ckv, cache_scales, cache_kpe, sincos, (int) ckv_kpe.size(0),
               (int) q_len, (int) cache_ckv.size(1), at::cuda::getCurrentCUDAStream().stream(), nullptr);
}
