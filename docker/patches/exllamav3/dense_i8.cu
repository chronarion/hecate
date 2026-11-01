// hecate: int8 decode GEMV for EXL3 linears with an int8 copy (EXL3_I8_LINEARS), 1..8 rows.
// y[R][N] = x[R][K] (fp16, row-major) * W^T; W int8 (code - 128) with a per-output-row fp32 scale, pre-swizzled per lane
// in mma.sync m16n8k16 A-fragment order: [group of SPW strips][k-block of 32][strip][lane] x 16 bytes (see
// apply_i8_linear.py, pack_i8). Decode: LOP3 magic number into f16 + HADD2 (exact). Stream-K over contiguous units;
// partial groups accumulate with fp32 atomics into a per-linear scratch (y32 [8][N]) and the last warp per group
// (counter in ctr[N / (16 SPW)]) writes y, then zeroes the scratch and the counter: one kernel, graph-safe.
// Graph params: x (GP_gemm_A) and y (GP_gemm_C), patched per call like the trellis GEMM.
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "dense_i8.cuh"
#include "../util.h"

namespace {

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 4 bytes -> (b0, b2) and (b1, b3) as f16x2, minus 128
__device__ __forceinline__ void dec_i8(uint32_t w, uint32_t* a)
{
    uint32_t h0, h1;
    asm("lop3.b32 %0, %1, 0x00ff00ff, 0x64006400, 0xea;" : "=r"(h0) : "r"(w));
    asm("lop3.b32 %0, %1, 0x00ff00ff, 0x64006400, 0xea;" : "=r"(h1) : "r"(w >> 8));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(a[0]) : "r"(h0), "r"(0x64806480u));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(a[1]) : "r"(h1), "r"(0x64806480u));
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ half cvt_out<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }

constexpr int WPB = 8;

template <int SPW, int PD, typename OUT>
__global__ void __launch_bounds__(WPB * 32) dense_i8_kernel
(
    const uint4* __restrict__ w,
    const float* __restrict__ scale,
    const half* __restrict__ x,
    OUT* __restrict__ y,
    float* y32,
    int* ctr,
    int M, int N, int K, int upw
)
{
    const int kb_n = K / 32, groups = N / (16 * SPW), lane = threadIdx.x % 32;
    const int total = groups * kb_n, gw = blockIdx.x * WPB + threadIdx.x / 32;
    const int u0 = gw * upw, n = min(total - u0, upw);
    if (n <= 0) return;
    int kb = u0 % kb_n, gr = u0 / kb_n;
    const uint4* pf = w + (size_t) u0 * 32 * SPW + lane;
    const uint4* const pend = w + (size_t) (u0 + n) * 32 * SPW + lane;
    uint4 ring[PD][SPW];
    #pragma unroll
    for (int i = 0; i < PD; ++i)
    {
        #pragma unroll
        for (int s = 0; s < SPW; ++s) ring[i][s] = pf < pend ? __ldg(pf + s * 32) : make_uint4(0, 0, 0, 0);
        pf += 32 * SPW;
    }
    const int xn = lane / 4, kq = (lane % 4) * 2;
    const bool live = xn < M;
    const uint32_t* xr = (const uint32_t*) (x + (size_t) (live ? xn : 0) * K + kq);
    bool whole = kb == 0;
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;

    for (int done = 0; done < n; done += PD)
    {
        #pragma unroll
        for (int j = 0; j < PD; ++j)
        {
            if (done + j >= n) break;
            const bool more = pf < pend;
            uint32_t bb[4];
            #pragma unroll
            for (int t = 0; t < 2; ++t)
            {
                const int k0 = kb * 32 + t * 16;
                bb[2 * t] = live ? __ldg(xr + k0 / 2) : 0u;
                bb[2 * t + 1] = live ? __ldg(xr + k0 / 2 + 4) : 0u;
            }
            #pragma unroll
            for (int s = 0; s < SPW; ++s)
            {
                const uint32_t ww[4] = {ring[j][s].x, ring[j][s].y, ring[j][s].z, ring[j][s].w};
                #pragma unroll
                for (int t = 0; t < 2; ++t)
                {
                    uint32_t a[4];
                    dec_i8(ww[2 * t], a); dec_i8(ww[2 * t + 1], a + 2);
                    mma16816(c[s], a, bb[2 * t], bb[2 * t + 1]);
                }
                if (more) ring[j][s] = __ldg(pf + s * 32);
            }
            pf += 32 * SPW;
            ++kb;
            if (kb == kb_n || done + j == n - 1)
            {
                const int col = (lane % 4) * 2, rb = gr * SPW * 16 + lane / 4;
                if (whole && kb == kb_n)
                {
                    #pragma unroll
                    for (int s = 0; s < SPW; ++s)
                    {
                        const int r0 = rb + s * 16;
                        const float s0 = scale[r0], s1 = scale[r0 + 8];
                        if (col < M)     { y[(size_t) col * N + r0] = cvt_out<OUT>(c[s][0] * s0);       y[(size_t) col * N + r0 + 8] = cvt_out<OUT>(c[s][2] * s1); }
                        if (col + 1 < M) { y[(size_t) (col + 1) * N + r0] = cvt_out<OUT>(c[s][1] * s0); y[(size_t) (col + 1) * N + r0 + 8] = cvt_out<OUT>(c[s][3] * s1); }
                    }
                }
                else
                {
                    #pragma unroll
                    for (int s = 0; s < SPW; ++s)
                    {
                        const int r0 = rb + s * 16;
                        if (col < M)     { atomicAdd(y32 + (size_t) col * N + r0, c[s][0]);       atomicAdd(y32 + (size_t) col * N + r0 + 8, c[s][2]); }
                        if (col + 1 < M) { atomicAdd(y32 + (size_t) (col + 1) * N + r0, c[s][1]); atomicAdd(y32 + (size_t) (col + 1) * N + r0 + 8, c[s][3]); }
                    }
                    __threadfence();
                    __syncwarp();
                    const int nsplit = ((gr + 1) * kb_n - 1) / upw - (gr * kb_n) / upw + 1;
                    int last = 0;
                    if (lane == 0) last = atomicAdd(ctr + gr, 1) == nsplit - 1;
                    last = __shfl_sync(0xffffffffu, last, 0);
                    if (last)
                    {
                        __threadfence();
                        for (int i = lane; i < SPW * 16 * M; i += 32)
                        {
                            const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                            float* p = y32 + (size_t) m * N + r;
                            y[(size_t) m * N + r] = cvt_out<OUT>(__ldcg(p) * scale[r]);
                            *p = 0.f;
                        }
                        if (lane == 0) ctr[gr] = 0;
                    }
                }
                #pragma unroll
                for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
                whole = true;
                if (kb == kb_n) { kb = 0; ++gr; }
            }
        }
    }
}

template <int SPW, int PD, typename OUT>
void launch(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
            at::Tensor& ctr, int M, int N, int K, cudaStream_t stream, Graph* graph)
{
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int total = N / (16 * SPW) * (K / 32), warps = sms * 2 * WPB;
    const int upw = (total + warps - 1) / warps;
    const int blocks = (total + upw * WPB - 1) / (upw * WPB);
    auto kernel = dense_i8_kernel<SPW, PD, OUT>;
    kernel<<<blocks, WPB * 32, 0, stream>>>
    (
        (const uint4*) w8.data_ptr(), (const float*) scale.data_ptr(), (const half*) x.data_ptr(),
        (OUT*) y.data_ptr(), (float*) y32.data_ptr(), (int*) ctr.data_ptr(), M, N, K, upw
    );
    if (graph)
    {
        graph->record_param((void*) kernel, GP_gemm_A, 2);
        graph->record_param((void*) kernel, GP_gemm_C, 3);
        graph->record_param((void*) kernel, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}

}  // namespace

void dense_i8_gr
(
    const at::Tensor& x,
    const at::Tensor& w8,
    const at::Tensor& scale,
    at::Tensor& y,
    at::Tensor& y32,
    at::Tensor& ctr,
    int spw,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    const int K = (int) x.size(-1), M = (int) (x.numel() / K), N = (int) y.size(-1);
    TORCH_CHECK(M >= 1 && M <= 8, "dense_i8: 1..8 rows, got ", M);
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous(), "dense_i8: x must be contiguous fp16");
    TORCH_CHECK(K % 32 == 0 && N % (16 * spw) == 0, "dense_i8: shape ", N, "x", K, " vs SPW ", spw);
    TORCH_CHECK(y32.numel() >= (int64_t) 8 * N && ctr.numel() >= N / (16 * spw), "dense_i8: scratch too small");
    const bool f32 = y.dtype() == at::kFloat;
    TORCH_CHECK(f32 || y.dtype() == at::kHalf, "dense_i8: y must be fp16 or fp32");
    if (spw == 4) { if (f32) launch<4, 2, float>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph); else launch<4, 2, half>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph); }
    else if (spw == 2) { if (f32) launch<2, 4, float>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph); else launch<2, 4, half>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph); }
    else { if (f32) launch<1, 8, float>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph); else launch<1, 8, half>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph); }
}

void dense_i8
(
    const at::Tensor& x,
    const at::Tensor& w8,
    const at::Tensor& scale,
    at::Tensor& y,
    at::Tensor& y32,
    at::Tensor& ctr,
    int64_t spw
)
{
    dense_i8_gr(x, w8, scale, y, y32, ctr, (int) spw, nullptr);
}
