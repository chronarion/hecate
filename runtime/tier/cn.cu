// hecate (EXL3_TIER_COMBINE_NORM): the MoE combine of layer L fused into the input norm of layer L+1. One CTA per row,
// 1024 threads (rms_norm_kernel's block for H = 6144): wait for the RTX band's DONE (as moe4_combine_kernel), then per
// float4 column a = f + y (+ o) (+ o2) (the combine's order), y zeroed for the next layer, r += a (fp32 residual),
// xn = r * rsqrt(mean(r^2) + eps) * (w + bias) * scale. Column ownership (t, t + 1024) and the summation order are those of
// rms_norm_kernel, so r and xn are bit-identical to moe4_combine_kernel followed by rms_norm_res_in; f is not written
// (the deferred pending sum is consumed here).
// combine_norm_pdl (EXL3_CN_PDL): the same kernel as a programmatic dependent of moe4 B (which triggers at its start): its
// CTAs become resident as B's retire, wait for the RTX DONE and the Grace CPU_DONE (wait_reset_k folded in: the last CTA
// past the wait resets CPU_DONE), load f (or none), both C2C partials, r and w before griddepcontrol.wait, and only then
// read y. Same per-column order ((f + y) + o) + o2 and reduction, so r and xn are bit-identical; f = nullptr reads as +0
// (the zero-filled f it replaces).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
constexpr int H = 6144, NT = 1024, COLS = H / 4;

__device__ __forceinline__ float reduce_1024(float sum, int warp, int lane)
{
    // reduce_dyn of norm.cu for 32 warps: butterfly in the warp, the warp sums through shared memory, butterfly again
    __shared__ float sums[32];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    sum = sums[lane];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

__device__ __forceinline__ float4 ldw(const half* w, int c)
{
    const uint2 u = *(const uint2*) (w + 4 * c);
    const float2 a = __half22float2(*(const half2*) &u.x), b = __half22float2(*(const half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 ldw(const __nv_bfloat16* w, int c)
{
    const uint2 u = *(const uint2*) (w + 4 * c);
    const __nv_bfloat162 p = *(const __nv_bfloat162*) &u.x, q = *(const __nv_bfloat162*) &u.y;
    return make_float4(__bfloat162float(__low2bfloat16(p)), __bfloat162float(__high2bfloat16(p)),
                       __bfloat162float(__low2bfloat16(q)), __bfloat162float(__high2bfloat16(q)));
}

template <typename weight_t>
__global__ void __launch_bounds__(NT) combine_norm_kernel(const float* __restrict__ f, const float* __restrict__ o,
    float* __restrict__ y, const float* __restrict__ o2, const unsigned* done, float* __restrict__ r,
    const weight_t* __restrict__ w, half* __restrict__ xn, float eps, float cbias, float cscale, int dim)
{
    // hecate (EXL3_I8_PDL): the next layer's q_a / kv_a GEMV may launch now and fill its weight ring while this kernel
    // waits for the bands (it reads xn only after griddepcontrol.wait, i.e. after this kernel completes)
    asm volatile("griddepcontrol.launch_dependents;");
    if (done)
    {
        if (threadIdx.x == 0) while (*(volatile const unsigned*) done < 1u) __nanosleep(100);
        __syncthreads();
    }
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, row = blockIdx.x;
    const size_t ro = (size_t) row * H;
    const int ca = t, cb = t + NT;
    const bool hb = cb < COLS;
    float4 a[2], rr[2], ww[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        const int c = k ? cb : ca;
        if (k && !hb) continue;
        const size_t i = ro + 4 * (size_t) c;
        float4 v = *(const float4*) (f + i); const float4 yc = *(const float4*) (y + i);
        v.x += yc.x; v.y += yc.y; v.z += yc.z; v.w += yc.w;
        if (o) { const float4 b = __ldcv((const float4*) (o + i)); v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w; }
        if (o2) { const float4 b = __ldcv((const float4*) (o2 + i)); v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w; }
        a[k] = v;
        rr[k] = *(const float4*) (r + i);
        ww[k] = ldw(w, c);
        *(float4*) (y + i) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        if (k && !hb) continue;
        float4& x4 = a[k];
        x4.x += rr[k].x; x4.y += rr[k].y; x4.z += rr[k].z; x4.w += rr[k].w;      // pending + residual (RES_IN)
        *(float4*) (r + ro + 4 * (size_t) (k ? cb : ca)) = x4;
        sum = fmaf(x4.x, x4.x, sum); sum = fmaf(x4.y, x4.y, sum); sum = fmaf(x4.z, x4.z, sum); sum = fmaf(x4.w, x4.w, sum);
    }
    sum = reduce_1024(sum, warp, lane);
    const float rmf = rsqrtf(sum / (float) dim + eps) * cscale;   // runtime dim: the same fast-math division as rms_norm_kernel
    #pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        if (k && !hb) continue;
        float4 x4 = a[k], w4 = ww[k];
        if (cbias != 0.f) { w4.x += cbias; w4.y += cbias; w4.z += cbias; w4.w += cbias; }
        x4.x = x4.x * w4.x * rmf; x4.y = x4.y * w4.y * rmf; x4.z = x4.z * w4.z * rmf; x4.w = x4.w * w4.w * rmf;
        const half2 h0 = __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y));
        const half2 h1 = __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w));
        uint2 u; u.x = *(const uint32_t*) &h0; u.y = *(const uint32_t*) &h1;
        *(uint2*) (xn + ro + 4 * (size_t) (k ? cb : ca)) = u;
    }
}
__device__ __forceinline__ void add4(float4& v, const float4& b) { v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w; }

template <typename weight_t>
__global__ void __launch_bounds__(NT) combine_norm_pdl_kernel(const float* __restrict__ f, const float* __restrict__ o,
    float* __restrict__ y, const float* __restrict__ o2, const unsigned* done, unsigned* cdone, unsigned cwant, int* cctr,
    float* __restrict__ r, const weight_t* __restrict__ w, half* __restrict__ xn, float eps, float cbias, float cscale, int dim)
{
    asm volatile("griddepcontrol.launch_dependents;");          // the next layer's q_a / kv_a GEMV (EXL3_I8_PDL)
    if (done || cdone)
    {
        if (threadIdx.x == 0)
        {
            if (done) while (*(volatile const unsigned*) done < 1u) __nanosleep(100);
            if (cdone) while (*(volatile const unsigned*) cdone < cwant) __nanosleep(100);
        }
        __syncthreads();
    }
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, row = blockIdx.x;
    const size_t ro = (size_t) row * H;
    const int ca = t, cb = t + NT;
    const bool hb = cb < COLS;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 fv[2] = {z, z}, ov[2] = {z, z}, o2v[2] = {z, z}, rr[2] = {z, z}, ww[2] = {z, z}, a[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)          // everything but y: independent of the hot band
    {
        const int c = k ? cb : ca;
        if (k && !hb) continue;
        const size_t i = ro + 4 * (size_t) c;
        if (f) fv[k] = *(const float4*) (f + i);
        if (o) ov[k] = __ldcv((const float4*) (o + i));
        if (o2) o2v[k] = __ldcv((const float4*) (o2 + i));
        rr[k] = *(const float4*) (r + i);
        ww[k] = ldw(w, c);
    }
    // the folded wait_reset_k: once every CTA is past the Grace wait, reset CPU_DONE so a stale (larger) generation can
    // never satisfy a later wait
    if (cdone && t == 0 && atomicAdd(cctr, 1) == (int) gridDim.x - 1)
    {
        *cctr = 0; *(volatile unsigned*) cdone = 0u; __threadfence_system();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");          // moe4 B complete: y holds the hot band
    #pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        if (k && !hb) continue;
        const size_t i = ro + 4 * (size_t) (k ? cb : ca);
        float4 v = fv[k]; const float4 yc = *(const float4*) (y + i);
        add4(v, yc);
        if (o) add4(v, ov[k]);
        if (o2) add4(v, o2v[k]);
        a[k] = v;
        *(float4*) (y + i) = z;
    }
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        if (k && !hb) continue;
        float4& x4 = a[k];
        x4.x += rr[k].x; x4.y += rr[k].y; x4.z += rr[k].z; x4.w += rr[k].w;
        *(float4*) (r + ro + 4 * (size_t) (k ? cb : ca)) = x4;
        sum = fmaf(x4.x, x4.x, sum); sum = fmaf(x4.y, x4.y, sum); sum = fmaf(x4.z, x4.z, sum); sum = fmaf(x4.w, x4.w, sum);
    }
    sum = reduce_1024(sum, warp, lane);
    const float rmf = rsqrtf(sum / (float) dim + eps) * cscale;
    #pragma unroll
    for (int k = 0; k < 2; ++k)
    {
        if (k && !hb) continue;
        float4 x4 = a[k], w4 = ww[k];
        if (cbias != 0.f) { w4.x += cbias; w4.y += cbias; w4.z += cbias; w4.w += cbias; }
        x4.x = x4.x * w4.x * rmf; x4.y = x4.y * w4.y * rmf; x4.z = x4.z * w4.z * rmf; x4.w = x4.w * w4.w * rmf;
        const half2 h0 = __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y));
        const half2 h1 = __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w));
        uint2 u; u.x = *(const uint32_t*) &h0; u.y = *(const uint32_t*) &h1;
        *(uint2*) (xn + ro + 4 * (size_t) (k ? cb : ca)) = u;
    }
}
}  // namespace

// f [rows][H] fp32 (the layer's pending MoE sum without the hot / RTX / CPU partials), y = the hot accumulator (zeroed
// here), o / o2 = RTX / CPU partials (host pointers, 0 = none), done = the RTX DONE flag (0 = none); r [rows][H] fp32
// residual (updated in place), w [H] fp16 / bf16 norm weight, xn [rows][H] fp16 out
void combine_norm(torch::Tensor f, int64_t o_ptr, torch::Tensor y, int64_t rows, int64_t o2_ptr, int64_t done_ptr,
                  torch::Tensor r, torch::Tensor w, torch::Tensor xn, double eps, double cbias, double cscale)
{
    const at::cuda::OptionalCUDAGuard guard(f.device());
    TORCH_CHECK(f.dtype() == torch::kFloat && r.dtype() == torch::kFloat && xn.dtype() == torch::kHalf && f.size(-1) == H &&
                r.size(-1) == H && xn.size(-1) == H && w.numel() == H && f.is_contiguous() && r.is_contiguous() &&
                xn.is_contiguous() && w.is_contiguous() && rows >= 1 && rows <= 64, "combine_norm: shapes / dtypes");
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    if (w.dtype() == torch::kBFloat16)
        combine_norm_kernel<__nv_bfloat16><<<(int) rows, NT, 0, st>>>(f.data_ptr<float>(), (const float*) o_ptr,
            y.data_ptr<float>(), (const float*) o2_ptr, (const unsigned*) done_ptr, r.data_ptr<float>(),
            (const __nv_bfloat16*) w.data_ptr(), (half*) xn.data_ptr(), (float) eps, (float) cbias, (float) cscale, H);
    else
    {
        TORCH_CHECK(w.dtype() == torch::kHalf, "combine_norm: weight dtype");
        combine_norm_kernel<half><<<(int) rows, NT, 0, st>>>(f.data_ptr<float>(), (const float*) o_ptr,
            y.data_ptr<float>(), (const float*) o2_ptr, (const unsigned*) done_ptr, r.data_ptr<float>(),
            (const half*) w.data_ptr(), (half*) xn.data_ptr(), (float) eps, (float) cbias, (float) cscale, H);
    }
}

// f_ptr: the pending MoE sum (0 = none, read as +0); o / o2 = RTX / CPU partials (0 = none); done = RTX DONE (0 = none);
// cdone / cwant = the Grace CPU_DONE flag and generation (0 = none; reset to 0 here), cctr = one zeroed device int;
// pdl = launch as a programmatic dependent of the previous kernel
void combine_norm_pdl(int64_t f_ptr, int64_t o_ptr, torch::Tensor y, int64_t rows, int64_t o2_ptr, int64_t done_ptr,
                      int64_t cdone_ptr, int64_t cwant, torch::Tensor cctr, torch::Tensor r, torch::Tensor w, torch::Tensor xn,
                      double eps, double cbias, double cscale, int64_t pdl)
{
    const at::cuda::OptionalCUDAGuard guard(r.device());
    TORCH_CHECK(r.dtype() == torch::kFloat && xn.dtype() == torch::kHalf && r.size(-1) == H && xn.size(-1) == H &&
                w.numel() == H && r.is_contiguous() && xn.is_contiguous() && w.is_contiguous() && rows >= 1 && rows <= 64 &&
                cctr.dtype() == torch::kInt, "combine_norm_pdl: shapes / dtypes");
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    cudaLaunchConfig_t c = {};
    c.gridDim = dim3((unsigned) rows); c.blockDim = dim3(NT); c.dynamicSmemBytes = 0; c.stream = st;
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; la[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = la; c.numAttrs = pdl ? 1 : 0;
    auto go = [&](auto* wp) {
        cudaLaunchKernelEx(&c, combine_norm_pdl_kernel<std::remove_cv_t<std::remove_pointer_t<decltype(wp)>>>,
            (const float*) f_ptr, (const float*) o_ptr, y.data_ptr<float>(), (const float*) o2_ptr, (const unsigned*) done_ptr,
            (unsigned*) cdone_ptr, (unsigned) cwant, cctr.data_ptr<int>(), r.data_ptr<float>(), wp, (half*) xn.data_ptr(),
            (float) eps, (float) cbias, (float) cscale, (int) H);
    };
    if (w.dtype() == torch::kBFloat16) go((const __nv_bfloat16*) w.data_ptr());
    else { TORCH_CHECK(w.dtype() == torch::kHalf, "combine_norm_pdl: weight dtype"); go((const half*) w.data_ptr()); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("combine_norm", &combine_norm); m.def("combine_norm_pdl", &combine_norm_pdl); }
