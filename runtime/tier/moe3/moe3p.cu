// Hecate trellis MoE prefill (moe3p): the RTX band's routed experts (3-bit EXL3 trellis, mul1 codebook) for prompt chunks,
// sm_120. Grouped GEMMs that decode the trellis inside the MMA loop, so no expert is materialized in FP16 (the previous
// path reconstructed every picked expert per chunk: ~6.4 GB of FP16 written and read per layer):
//   rot  : per band pick (picks sorted by expert) and projection p (gate, up): xr[p][pos] = H128(x[tok] * suh) (fp16), its sum
//   gemm : block = (row block of <= 64 picks of one expert, projection, 256-column block). A cp.async ring holds the trellis
//          words (4 k-tiles x 16 n-tiles per stage) and the picks' inputs (64 x 64 fp16, 16-byte chunks swizzled by row).
//          Each warp decodes its two 16-column tiles per k-tile straight into m16n8k16 A fragments (the moe3 register
//          decode) and multiplies them by every 8-row group of the block (B fragments by ldmatrix): the weights stream once
//          per row block; raw sums sum(h x) in fp32, stored without atomics (each block owns its outputs)
//   act  : g = k_inv * sum(h x) + k_bias * sum(x) (likewise u), H128, * svh, silu(g) * u, * down suh, H128 -> ar, its sum
//   gemm : the down projection over ar (same kernel, one matrix)
//   fin  : out[tok] += w * svh * H128(k_inv * d + k_bias * sum(ar))
// Same math as the decode kernels (moe3.cu), only the k order of the fp32 sums differs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

constexpr int H = 6144, I = 2048, TW = 24;              // TW: uint32 words per 16x16 3-bit tile
constexpr int WARPS = 8, NT = WARPS * 32;
constexpr int BM = 64;                                  // picks per row block
constexpr int BN = WARPS * 32;                          // output columns per block: two 16-column tiles per warp
constexpr int BK = 64;                                  // k per stage (4 k-tiles)
#ifndef STAGES
#define STAGES 3
#endif
constexpr int XS_BYTES = BM * BK * 2;                   // inputs per stage
constexpr int WS_WORDS = 4 * (BN / 16) * TW;            // trellis words per stage
constexpr int SMEM = STAGES * (XS_BYTES + WS_WORDS * 4);
constexpr float R_SCALE = 0.088388347648f;              // 1 / sqrt(128)

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int shift)
{
    const uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}
// two mul1 codebook entries without the affine: fp16 patterns 0x6400 + byte sum of (idx * 0x83DCD12D)
__device__ __forceinline__ uint32_t dec2(uint32_t i0, uint32_t i1)
{
    const uint32_t s0 = __dp4a(i0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(i1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return __byte_perm(s0, s1, 0x5410);
}
// the lane's 8 weights of a tile as the A fragment of the transposed tile (see moe3.cu)
__device__ __forceinline__ void dq8_a(uint32_t a, uint32_t b, int s2, uint32_t* af)
{
    const uint32_t w7 = fshift(b, a, s2), w6 = w7 >> 3, w5 = w6 >> 3, w4 = w5 >> 3;
    const uint32_t w3 = fshift(b, a, s2 + 12), w2 = w3 >> 3, w1 = w2 >> 3, w0 = w1 >> 3;
    af[0] = dec2(w0 & 0xffff, w1 & 0xffff);
    af[2] = dec2(w2 & 0xffff, w3 & 0xffff);
    af[1] = dec2(w4 & 0xffff, w5 & 0xffff);
    af[3] = dec2(w6 & 0xffff, w7 & 0xffff);
}
__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
    float h0 = (v0 + v1) + (v2 + v3), h1 = (v0 - v1) + (v2 - v3), h2 = (v0 + v1) - (v2 + v3), h3 = (v0 - v1) - (v2 - v3);
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, i), p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, i), p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        const bool hi = lane & i;
        h0 = (hi ? -h0 : h0) + p0; h1 = (hi ? -h1 : h1) + p1; h2 = (hi ? -h2 : h2) + p2; h3 = (hi ? -h3 : h3) + p3;
    }
    v0 = h0 * R_SCALE; v1 = h1 * R_SCALE; v2 = h2 * R_SCALE; v3 = h3 * R_SCALE;
}
__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int i = 16; i; i >>= 1) v += __shfl_xor_sync(0xffffffffu, v, i);
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red)        // all threads call; result valid in thread 0
{
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    if (threadIdx.x == 0)
        #pragma unroll
        for (int i = 0; i < WARPS; ++i) t += red[i];
    return t;
}
__device__ __forceinline__ float k_inv() { return __half2float(__ushort_as_half((unsigned short) 0x1eee)); }
__device__ __forceinline__ float k_bias() { return __half2float(__ushort_as_half((unsigned short) 0xc931)); }
__device__ __forceinline__ float4 ld_h4(const half* p)
{
    const uint2 v = *(const uint2*) p;
    const float2 a = __half22float2(*(const half2*) &v.x), b = __half22float2(*(const half2*) &v.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
// fp16-round four values, store them, return the sum of the rounded values
__device__ __forceinline__ float st_h4(half* p, float v0, float v1, float v2, float v3)
{
    const half2 o01 = __floats2half2_rn(v0, v1), o23 = __floats2half2_rn(v2, v3);
    uint2 ov; ov.x = *(const uint32_t*) &o01; ov.y = *(const uint32_t*) &o23;
    *(uint2*) p = ov;
    const float2 f01 = __half22float2(o01), f23 = __half22float2(o23);
    return (f01.x + f01.y) + (f23.x + f23.y);
}

// ---------------------------------------------------------------------------------------------------------------------
// rot: block (pick pos, projection p); warp w takes the 128-chunks w, w + 8, ... of H
__global__ void __launch_bounds__(NT) p3_rot_kernel(const half* __restrict__ x, const int* __restrict__ tok,
    const int* __restrict__ eid, const int64_t* __restrict__ su0, const int64_t* __restrict__ su1, half* __restrict__ xr,
    float* __restrict__ xs, int ps)
{
    __shared__ float red[WARPS];
    const int pos = blockIdx.x, p = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const half* suh = (const half*) (p ? su1 : su0)[eid[pos]];
    const half* xrow = x + (size_t) tok[pos] * H;
    half* orow = xr + ((size_t) p * ps + pos) * H;
    float s = 0.f;
    for (int c = w; c < H / 128; c += WARPS)
    {
        const int col = c * 128 + lane * 4;
        const float4 xv = ld_h4(xrow + col), sv = ld_h4(suh + col);
        float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
        had128(v0, v1, v2, v3, lane);
        s += st_h4(orow + col, v0, v1, v2, v3);
    }
    s = block_sum(s, red);
    if (threadIdx.x == 0) xs[p * ps + pos] = s;
}

// ---------------------------------------------------------------------------------------------------------------------
// gemm: block (row block rb, projection p and 256-column block cb). rbt[rb] = {expert, first pick, picks}.
// x [MATS][ps][K] fp16 inputs per pick, out [MATS][ps][N] fp32 raw sums sum_k h[k][n] x[k]
template <int N, int K>
__global__ void __launch_bounds__(NT, 2) p3_gemm_kernel(const int4* __restrict__ rbt, const int64_t* __restrict__ tr0,
    const int64_t* __restrict__ tr1, const half* __restrict__ x, float* __restrict__ out, int ps)
{
    extern __shared__ __align__(128) uint8_t smem[];
    constexpr int NTN = N / 16, KB = K / BK, CBN = N / BN;
    const int4 rb = rbt[blockIdx.x];
    const int e = rb.x, pos0 = rb.y, rows = rb.z;
    const int p = blockIdx.y / CBN, cb = blockIdx.y % CBN;
    const uint32_t* wb = (const uint32_t*) (p ? tr1 : tr0)[e] + (size_t) cb * (BN / 16) * TW;
    const half* xb = x + ((size_t) p * ps + pos0) * K;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    uint8_t* xsm = smem;                                     // [STAGES][BM][BK] fp16, 16-byte chunk c of row r at c ^ (r & 7)
    uint32_t* wsm = (uint32_t*) (smem + STAGES * XS_BYTES);  // [STAGES][4 k-tiles][16 tiles][TW]
    const int nnt = (rows + 7) >> 3;                         // live 8-row groups

    auto issue = [&](int kb)
    {
        if (kb < KB)
        {
            const int st = kb % STAGES;
            #pragma unroll
            for (int i = 0; i < BM * 8 / NT; ++i)
            {
                const int c = t + i * NT, r = c >> 3, ch = c & 7;
                if (r < rows)
                    cp16(su32(xsm + st * XS_BYTES + r * (BK * 2) + ((ch ^ (r & 7)) << 4)), xb + (size_t) r * K + kb * BK + ch * 8);
            }
            #pragma unroll
            for (int i = 0; i < (4 * (BN / 16) * TW / 4 + NT - 1) / NT; ++i)     // 4 k-tiles x 16 tiles x 96 B
            {
                const int c = t + i * NT, q = c / ((BN / 16) * TW / 4), off = c % ((BN / 16) * TW / 4);
                if (q < 4)
                    cp16(su32(wsm + st * WS_WORDS + q * (BN / 16) * TW + off * 4), wb + (size_t) (kb * 4 + q) * NTN * TW + off * 4);
            }
        }
        cp_commit();
    };

    // the lane's word pair in a tile (3-bit register decode of exl3_gemv: t_offset = lane * 8)
    const int b1 = (lane * 8 + 257) * 3, b2 = b1 + 21;
    const int src_a = ((b1 - 16) / 32) % TW, src_b = ((b2 - 1) / 32) % TW, s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    // ldmatrix.x4 row address of this lane: matrices (rows +0, k +0), (rows +0, k +8), (rows +8, k +0), (rows +8, k +8)
    const int lm_r = ((lane >> 4) << 3) + (lane & 7), lm_c = (lane >> 3) & 1;

    float c[2][8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) c[mt][nt][0] = c[mt][nt][1] = c[mt][nt][2] = c[mt][nt][3] = 0.f;

    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) issue(i);
    for (int kb = 0; kb < KB; ++kb)
    {
        cp_wait<STAGES - 2>();
        __syncthreads();                                     // stage kb visible; stage kb - 1 free for the refill
        issue(kb + STAGES - 1);
        const int st = kb % STAGES;
        const uint32_t xsb = su32(xsm + st * XS_BYTES);
        const uint32_t* wst = wsm + st * WS_WORDS;
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            uint32_t af[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
            {
                const uint32_t* tw = wst + (q * (BN / 16) + w * 2 + mt) * TW;
                dq8_a(tw[src_a], tw[src_b], s2, af[mt]);
            }
            #pragma unroll
            for (int np = 0; np < 4; ++np)
            {
                if (np * 2 < nnt)
                {
                    const int r = np * 16 + lm_r, ch = q * 2 + lm_c;
                    uint32_t b00, b01, b10, b11;
                    ldsm4(xsb + r * (BK * 2) + ((ch ^ (r & 7)) << 4), b00, b01, b10, b11);
                    #pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                    {
                        mma16816(c[mt][np * 2], af[mt], b00, b01);
                        if (np * 2 + 1 < nnt) mma16816(c[mt][np * 2 + 1], af[mt], b10, b11);
                    }
                }
            }
        }
    }
    cp_wait<0>();

    // lane holds (out column w*32 + mt*16 + g (+8), picks nt*8 + 2*(lane%4) (+1))
    float* ob = out + ((size_t) p * ps + pos0) * N + cb * BN + w * 32;
    const int g = lane >> 2, r0 = (lane & 3) * 2;
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
        {
            const int r = nt * 8 + r0, col = mt * 16 + g;
            if (r < rows) { ob[(size_t) r * N + col] = c[mt][nt][0]; ob[(size_t) r * N + col + 8] = c[mt][nt][2]; }
            if (r + 1 < rows) { ob[(size_t) (r + 1) * N + col] = c[mt][nt][1]; ob[(size_t) (r + 1) * N + col + 8] = c[mt][nt][3]; }
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// act: block (pick pos); warp w takes the 128-chunks w, w + 8 of I
__global__ void __launch_bounds__(NT) p3_act_kernel(const int* __restrict__ eid, const int64_t* __restrict__ svg,
    const int64_t* __restrict__ svu, const int64_t* __restrict__ sud, const float* __restrict__ gu, const float* __restrict__ xs,
    half* __restrict__ ar, float* __restrict__ as, int ps)
{
    __shared__ float red[WARPS];
    const int pos = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, e = eid[pos];
    const float ki = k_inv(), kbias = k_bias();
    const float bg = kbias * xs[pos], bu = kbias * xs[ps + pos];
    float s = 0.f;
    for (int cc = w; cc < I / 128; cc += WARPS)
    {
        const int col = cc * 128 + lane * 4;
        float o[2][4];
        #pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            const float4 v = *(const float4*) (gu + ((size_t) p * ps + pos) * I + col);
            const float bs = p ? bu : bg;
            o[p][0] = fmaf(ki, v.x, bs); o[p][1] = fmaf(ki, v.y, bs); o[p][2] = fmaf(ki, v.z, bs); o[p][3] = fmaf(ki, v.w, bs);
            had128(o[p][0], o[p][1], o[p][2], o[p][3], lane);
            const float4 sv = ld_h4((const half*) (p ? svu : svg)[e] + col);
            o[p][0] *= sv.x; o[p][1] *= sv.y; o[p][2] *= sv.z; o[p][3] *= sv.w;
        }
        const float4 dsu = ld_h4((const half*) sud[e] + col);
        float a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = o[0][i] / (1.f + __expf(-o[0][i])) * o[1][i];
        a[0] *= dsu.x; a[1] *= dsu.y; a[2] *= dsu.z; a[3] *= dsu.w;
        had128(a[0], a[1], a[2], a[3], lane);
        s += st_h4(ar + (size_t) pos * I + col, a[0], a[1], a[2], a[3]);
    }
    s = block_sum(s, red);
    if (threadIdx.x == 0) as[pos] = s;
}

// fin: block (pick pos); warp w takes the 128-chunks w, w + 8, ... of H
__global__ void __launch_bounds__(NT) p3_fin_kernel(const int* __restrict__ tok, const int* __restrict__ eid,
    const float* __restrict__ wt, const int64_t* __restrict__ svd, const float* __restrict__ d, const float* __restrict__ as,
    float* __restrict__ out)
{
    const int pos = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float ki = k_inv(), bs = k_bias() * as[pos], wj = wt[pos];
    const half* sv = (const half*) svd[eid[pos]];
    float* orow = out + (size_t) tok[pos] * H;
    for (int cc = w; cc < H / 128; cc += WARPS)
    {
        const int col = cc * 128 + lane * 4;
        const float4 v = *(const float4*) (d + (size_t) pos * H + col);
        float d0 = fmaf(ki, v.x, bs), d1 = fmaf(ki, v.y, bs), d2 = fmaf(ki, v.z, bs), d3 = fmaf(ki, v.w, bs);
        had128(d0, d1, d2, d3, lane);
        const float4 s = ld_h4(sv + col);
        atomicAdd((float4*) (orow + col), make_float4(wj * d0 * s.x, wj * d1 * s.y, wj * d2 * s.z, wj * d3 * s.w));
    }
}
}  // namespace

// scratch for up to npos band picks per call: xr [2][npos][H] fp16 (the down sums d [npos][H] fp32 reuse it), xs [2][npos],
// gu [2][npos][I] fp32, ar [npos][I] fp16, as [npos]
std::vector<at::Tensor> moe3p_scratch(at::Device dev, int64_t npos)
{
    auto f = at::TensorOptions().device(dev);
    return {at::empty({2, npos, H}, f.dtype(at::kHalf)), at::empty({2, npos}, f.dtype(at::kFloat)),
            at::empty({2, npos, I}, f.dtype(at::kFloat)), at::empty({npos, I}, f.dtype(at::kHalf)),
            at::empty({npos}, f.dtype(at::kFloat))};
}

// x [rows][H] fp16; tok / eid [npos] int32: band picks sorted by expert (token row, local expert id); wt [npos] fp32 routing
// weights; rbt [nrb][4] int32 row blocks {local expert, first pick, picks (<= 64), 0}; tables int64 [nb] device pointers
// (trellis / suh / svh for gate, up, down); out [rows][H] fp32, accumulated into
void moe3p_forward(at::Tensor x, at::Tensor tok, at::Tensor eid, at::Tensor wt, at::Tensor rbt, at::Tensor gtr, at::Tensor gsu,
                   at::Tensor gsv, at::Tensor utr, at::Tensor usu, at::Tensor usv, at::Tensor dtr, at::Tensor dsu, at::Tensor dsv,
                   at::Tensor xr, at::Tensor xs, at::Tensor gu, at::Tensor ar, at::Tensor as, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int npos = (int) tok.numel(), nrb = (int) rbt.size(0), ps = (int) xr.size(1);
    TORCH_CHECK(x.size(1) == H && x.dtype() == at::kHalf && out.dtype() == at::kFloat && npos <= ps && rbt.size(1) == 4,
                "moe3p: shapes");
    if (npos == 0) return;
    auto P = [](const at::Tensor& t) { return (const int64_t*) t.data_ptr(); };
    static bool attr = [] {
        cudaFuncSetAttribute(p3_gemm_kernel<I, H>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(p3_gemm_kernel<H, I>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        return true;
    }();
    (void) attr;
    const int* tk = (const int*) tok.data_ptr(); const int* ei = (const int*) eid.data_ptr();
    half* xrp = (half*) xr.data_ptr(); float* xsp = (float*) xs.data_ptr();
    p3_rot_kernel<<<dim3(npos, 2), NT, 0, st>>>((const half*) x.data_ptr(), tk, ei, P(gsu), P(usu), xrp, xsp, ps);
    p3_gemm_kernel<I, H><<<dim3(nrb, 2 * (I / BN)), NT, SMEM, st>>>((const int4*) rbt.data_ptr(), P(gtr), P(utr), xrp,
                                                                    (float*) gu.data_ptr(), ps);
    p3_act_kernel<<<npos, NT, 0, st>>>(ei, P(gsv), P(usv), P(dsu), (const float*) gu.data_ptr(), xsp, (half*) ar.data_ptr(),
                                       (float*) as.data_ptr(), ps);
    float* dp = (float*) xr.data_ptr();                      // the down sums reuse the (consumed) rotated inputs
    p3_gemm_kernel<H, I><<<dim3(nrb, H / BN), NT, SMEM, st>>>((const int4*) rbt.data_ptr(), P(dtr), P(dtr),
                                                              (const half*) ar.data_ptr(), dp, ps);
    p3_fin_kernel<<<npos, NT, 0, st>>>(tk, ei, (const float*) wt.data_ptr(), P(dsv), dp, (const float*) as.data_ptr(),
                                       (float*) out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scratch", &moe3p_scratch);
    m.def("forward", &moe3p_forward);
}
