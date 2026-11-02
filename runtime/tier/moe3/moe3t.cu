// Hecate trellis MoE prefill on Blackwell tensor cores (moe3t v3: inputs by TMA tiles, trellis words through a raw
// ring), sm_103a: the RTX band's 3-bit EXL3 trellis experts (mul1 codebook) computed by the GB300 for long prompt
// chunks, from a copy of the band's trellis in GB300-readable memory.
// moe3p's math (rot / act / fin kernels are moe3p's) with the grouped GEMM on tcgen05:
//   gemm : CTA = (row block of <= 256 picks of one expert, matrix, 128 output channels). Per 64-k stage:
//          warp 6     one thread streams the tile's trellis words (4 k-tiles x 768 B per stage, bulk async copies) through
//                     a RAW-deep shared-memory ring ahead of the decode (latency-tolerant over C2C)
//          warps 0-3  read their word pairs from the ring, decode (the moe3 register decode: one m16n8k16 A fragment
//                     of the transposed tile) and stores it into a K-major SW128 shared-memory tile (4-byte stores,
//                     bank-conflict free), then arrive
//          warp 4     one thread loads the picks' 256 x 64 rotated-input tile by TMA (SW128, out-of-range rows zero-filled)
//          warp 5     one thread issues tcgen05.mma M128 x N(picks, rounded to 16) x K16 into a TMEM accumulator
//          after the last stage warps 0-3 store the raw sums sum(h x) (fp32; lane = output channel)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace {

constexpr int H = 6144, I = 2048, TM = 128, TN = 256, TK = 64, NT = 224, WARPS = 8;   // gemm: warps 0-3 decode, 4 TMA, 5 MMA, 6 raw
#ifndef RAW
#define RAW 4
#endif
constexpr int RAW_BYTES = 4 * 8 * 96;                    // trellis words of one stage: 4 k-tiles x 8 tiles x 96 B
constexpr int NA = 256;                                 // rot / act / fin threads
constexpr int TW = 24;                                  // uint32 words per 16x16 3-bit trellis tile
constexpr float R_SCALE = 0.088388347648f;              // 1 / sqrt(128)
#ifndef STAGES
#define STAGES 2
#endif
constexpr int A_BYTES = TM * TK * 2;                    // 16 KB decoded weights
constexpr int B_BYTES = TN * TK * 2;                    // 32 KB inputs
constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
constexpr int SM_RAW = STAGES * STAGE_BYTES;            // raw trellis ring
constexpr int SM_BAR = SM_RAW + RAW * RAW_BYTES;        // mbarriers: full_a[S], full_b[S], empty[S], done, rfull[RAW], rempty[RAW]
constexpr int SMEM = SM_BAR + 8 * (3 * STAGES + 1 + 2 * RAW) + 16 + 1024;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(uint32_t b, uint32_t parity)
{
    asm volatile("{\n .reg .pred P1;\n W:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @!P1 bra W;\n}\n"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t b)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout)
{
    return (uint64_t) ((addr >> 4) & 0x3FFF) | ((uint64_t) ((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t) ((sbo >> 4) & 0x3FFF) << 32)
         | (1ull << 46) | ((uint64_t) layout << 61);
}
__device__ __forceinline__ void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, int acc)
{
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
#define TLD32(r, addr) asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), \
      "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), \
      "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), \
      "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(addr))
__device__ __forceinline__ void tld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void bar_expect_tx(uint32_t b, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_copy(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void st32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(v) : "memory"); }
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

// byte offset of the 16-byte chunk c (k 8c .. 8c + 7) of row r in a K-major SW128 tile (64 k = 128 B per row)
__device__ __forceinline__ uint32_t sw128(int r, int c) { return (r >> 3) * 1024 + (r & 7) * 128 + ((c ^ (r & 7)) << 4); }

// ---------------------------------------------------------------------------------------------------------------------
// gemm: CTA (row block rb, matrix p and 128-channel tile ct). rbt[rb] = {expert, first pick, picks (<= 256)}.
// Inputs: tmb = TMA map of the picks' rotated inputs, row (p * plane + pick) for matrix p. w0 / w1: int64 [nb] pointers to the
// experts' trellis of matrix p = 0 / 1 ([K / 16][N / 16] tiles of 24 words); out [MATS][ps][N] fp32 raw sums sum_k h[k][n] x[k]
template <int N, int K>
__global__ void __launch_bounds__(NT, 2) p3t_gemm_kernel(const __grid_constant__ CUtensorMap tmb, const int4* __restrict__ rbt,
    const int64_t* __restrict__ w0, const int64_t* __restrict__ w1, float* __restrict__ out, int ps, int plane)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    constexpr int KB = K / TK, CT = N / TM;
    const int4 rb = rbt[blockIdx.x];
    const int e = rb.x, pos0 = rb.y, rows = rb.z;
    const int p = blockIdx.y / CT, ct = blockIdx.y % CT;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t bar0 = su32(smem + SM_BAR), done = bar0 + 8 * 3 * STAGES;
    uint32_t* tm_base = (uint32_t*) (smem + SM_BAR + 8 * (3 * STAGES + 1 + 2 * RAW));
    const int nrow16 = (rows + 15) & ~15;
    const uint32_t rbar = done + 8;                         // rfull[r] = rbar + 8 r, rempty[r] = rbar + 8 (RAW + r)

    if (tid == 0)
    {
        for (int s = 0; s < STAGES; ++s) { bar_init(bar0 + 8 * s, 128); bar_init(bar0 + 8 * (STAGES + s), 1); bar_init(bar0 + 8 * (2 * STAGES + s), 1); }
        bar_init(done, 1);
        for (int r = 0; r < RAW; ++r) { bar_init(rbar + 8 * r, 1); bar_init(rbar + 8 * (RAW + r), 128); }
    }
    if (warp == 5)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(su32(tm_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = *tm_base;

    if (warp < 4)
    {
        // -------------------------------------------------------------------- decode: n-tiles ct*8 + 2w + j, k-tiles kb*4 + t
        constexpr int NTN = N / 16;
        const int g = lane >> 2, c = lane & 3;
        // the lane's word pair in a tile (3-bit register decode of exl3_gemv: t_offset = lane * 8), as moe3
        const int b1 = (lane * 8 + 257) * 3, b2 = b1 + 21;
        const int src_a = ((b1 - 16) / 32) % TW, src_b = ((b2 - 1) / 32) % TW, s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
        for (int kb = 0; kb < KB; ++kb)
        {
            const int st = kb % STAGES, r = kb % RAW;
            bar_wait(rbar + 8 * r, (kb / RAW) & 1);              // this stage's trellis words are in the ring
            if (kb >= STAGES) bar_wait(bar0 + 8 * (2 * STAGES + st), ((kb / STAGES) - 1) & 1);
            const uint32_t A = su32(smem + st * STAGE_BYTES);
            const uint32_t* rw = (const uint32_t*) (smem + SM_RAW + r * RAW_BYTES);
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                {
                    const uint32_t* tw = rw + (t * 8 + 2 * warp + j) * TW;
                    uint32_t a[4];
                    dq8_a(tw[src_a], tw[src_b], s2, a);
                    const int rlo = (2 * warp + j) * 16 + g, rhi = rlo + 8;
                    st32(A + sw128(rlo, 2 * t) + 4 * c, a[0]);
                    st32(A + sw128(rhi, 2 * t) + 4 * c, a[1]);
                    st32(A + sw128(rlo, 2 * t + 1) + 4 * c, a[2]);
                    st32(A + sw128(rhi, 2 * t + 1) + 4 * c, a[3]);
                }
            bar_arrive(rbar + 8 * (RAW + r));                   // ring slot read
            fence_async_smem();
            bar_arrive(bar0 + 8 * st);
        }
        // -------------------------------------------------------------------- epilogue: lane = output channel
        bar_wait(done, 0);
        tc_fence_after();
        const uint32_t lanes = (uint32_t) (warp * 32) << 16;
        float* ob = out + ((size_t) p * ps + pos0) * N + ct * TM + warp * 32 + lane;
        for (int c0 = 0; c0 < nrow16; c0 += 32)
        {
            uint32_t v[32];
            TLD32(v, tmem + lanes + c0);
            tld_wait();
            #pragma unroll
            for (int j = 0; j < 32; ++j)
                if (c0 + j < rows) ob[(size_t) (c0 + j) * N] = __uint_as_float(v[j]);
        }
        tc_fence_before();
    }
    else if (warp == 4)
    {
        // -------------------------------------------------------------------- inputs: one TMA tile per stage
        if (lane == 0)
            for (int kb = 0; kb < KB; ++kb)
            {
                const int st = kb % STAGES;
                if (kb >= STAGES) bar_wait(bar0 + 8 * (2 * STAGES + st), ((kb / STAGES) - 1) & 1);
                bar_expect_tx(bar0 + 8 * (STAGES + st), B_BYTES);
                tma_load_2d(su32(smem + st * STAGE_BYTES + A_BYTES), &tmb, kb * TK, p * plane + pos0, bar0 + 8 * (STAGES + st));
            }
        __syncwarp();
    }
    else if (warp == 6)
    {
        // -------------------------------------------------------------------- trellis words: 4 bulk copies per stage, RAW ahead
        if (lane == 0)
        {
            constexpr int NTN = N / 16;
            const uint8_t* wb = (const uint8_t*) (p ? w1 : w0)[e] + (size_t) ct * 8 * TW * 4;
            for (int kb = 0; kb < KB; ++kb)
            {
                const int r = kb % RAW;
                if (kb >= RAW) bar_wait(rbar + 8 * (RAW + r), ((kb / RAW) - 1) & 1);
                bar_expect_tx(rbar + 8 * r, RAW_BYTES);
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    bulk_copy(su32(smem + SM_RAW + r * RAW_BYTES + t * 768), wb + (size_t) (kb * 4 + t) * NTN * TW * 4, 768, rbar + 8 * r);
            }
        }
        __syncwarp();
    }
    else if (warp == 5)
    {
        // -------------------------------------------------------------------- MMA issue
        if (lane == 0)
        {
            const uint32_t idesc = (1u << 4) | ((uint32_t) (nrow16 >> 3) << 17) | ((uint32_t) (TM >> 4) << 24);
            for (int kb = 0; kb < KB; ++kb)
            {
                const int st = kb % STAGES;
                bar_wait(bar0 + 8 * st, (kb / STAGES) & 1);
                bar_wait(bar0 + 8 * (STAGES + st), (kb / STAGES) & 1);
                tc_fence_after();
                const uint32_t a = su32(smem + st * STAGE_BYTES), b = a + A_BYTES;
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    mma_ss(tmem, sdesc(a + s * 32, 16, 1024, 2), sdesc(b + s * 32, 16, 1024, 2), idesc, kb > 0 || s > 0);
                tc_commit(bar0 + 8 * (2 * STAGES + st));
            }
            tc_commit(done);
        }
        __syncwarp();
    }
    __syncthreads();
    if (warp == 5)
    {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem) : "memory");
    }
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// rot: block (pick pos, projection p); warp w takes the 128-chunks w, w + 8, ... of H
__global__ void __launch_bounds__(NA) p3_rot_kernel(const half* __restrict__ x, const int* __restrict__ tok,
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

// act: block (pick pos); warp w takes the 128-chunks w, w + 8 of I
__global__ void __launch_bounds__(NA) p3_act_kernel(const int* __restrict__ eid, const int64_t* __restrict__ svg,
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
__global__ void __launch_bounds__(NA) p3_fin_kernel(const int* __restrict__ tok, const int* __restrict__ eid,
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

PFN_cuTensorMapEncodeTiled_v12000 encode_fn()
{
    static PFN_cuTensorMapEncodeTiled_v12000 f = [] {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        TORCH_CHECK(p && q == cudaDriverEntryPointSuccess, "moe3t: cuTensorMapEncodeTiled not found");
        return (PFN_cuTensorMapEncodeTiled_v12000) p;
    }();
    return f;
}
// TMA map of a row-major fp16 [rows][K] matrix in 64 x 256 tiles, 128-byte swizzle (the UMMA K-major SW128 layout)
CUtensorMap make_map(const void* ptr, int K, int rows)
{
    CUtensorMap m;
    cuuint64_t dims[2] = {(cuuint64_t) K, (cuuint64_t) rows};
    cuuint64_t strides[1] = {(cuuint64_t) K * 2};
    cuuint32_t box[2] = {TK, TN}, es[2] = {1, 1};
    const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*) ptr, dims, strides, box, es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "moe3t: cuTensorMapEncodeTiled failed ", (int) r);
    return m;
}

template <int N, int K>
void launch_gemm(const CUtensorMap& tmb, const at::Tensor& rbt, const int64_t* w0, const int64_t* w1, float* out, int ps, int mats,
                 int plane, cudaStream_t st)
{
    static bool attr = [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(p3t_gemm_kernel<N, K>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        return true;
    }();
    (void) attr;
    p3t_gemm_kernel<N, K><<<dim3((unsigned) rbt.size(0), mats * (N / TM)), NT, SMEM, st>>>(tmb, (const int4*) rbt.data_ptr(),
                                                                                         w0, w1, out, ps, plane);
}
}  // namespace

// one grouped GEMM (tests): gate/up shape (N = I, K = H, x [2][ps][H]) or down shape (N = H, K = I, x [ps][I])
void moe3t_gemm(at::Tensor x, at::Tensor rbt, at::Tensor w0, at::Tensor w1, at::Tensor out, int64_t down)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int ps = (int) out.size(1);
    if (down)
        launch_gemm<H, I>(make_map(x.data_ptr(), I, (int) x.size(0)), rbt, (const int64_t*) w0.data_ptr(),
                          (const int64_t*) w0.data_ptr(), (float*) out.data_ptr(), ps, 1, 0, st);
    else
        launch_gemm<I, H>(make_map(x.data_ptr(), H, 2 * ps), rbt, (const int64_t*) w0.data_ptr(), (const int64_t*) w1.data_ptr(),
                          (float*) out.data_ptr(), ps, 2, ps, st);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// scratch for up to npos band picks per call: xr [2][npos][H] fp16 (the down sums d [npos][H] fp32 reuse it), xs [2][npos],
// gu [2][npos][I] fp32, ar [npos][I] fp16, as [npos]
std::vector<at::Tensor> moe3t_scratch(at::Device dev, int64_t npos)
{
    auto f = at::TensorOptions().device(dev);
    return {at::empty({2, npos, H}, f.dtype(at::kHalf)), at::empty({2, npos}, f.dtype(at::kFloat)),
            at::empty({2, npos, I}, f.dtype(at::kFloat)), at::empty({npos, I}, f.dtype(at::kHalf)),
            at::empty({npos}, f.dtype(at::kFloat))};
}

// as moe3p_forward (row blocks of <= 256 picks); the trellis tables may point to any GB300-readable memory
void moe3t_forward(at::Tensor x, at::Tensor tok, at::Tensor eid, at::Tensor wt, at::Tensor rbt, at::Tensor gtr, at::Tensor gsu,
                   at::Tensor gsv, at::Tensor utr, at::Tensor usu, at::Tensor usv, at::Tensor dtr, at::Tensor dsu, at::Tensor dsv,
                   at::Tensor xr, at::Tensor xs, at::Tensor gu, at::Tensor ar, at::Tensor as, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int npos = (int) tok.numel(), ps = (int) xr.size(1);
    TORCH_CHECK(x.size(1) == H && x.dtype() == at::kHalf && out.dtype() == at::kFloat && npos <= ps && rbt.size(1) == 4,
                "moe3t: shapes");
    if (npos == 0) return;
    auto P = [](const at::Tensor& t) { return (const int64_t*) t.data_ptr(); };
    const int* tk = (const int*) tok.data_ptr(); const int* ei = (const int*) eid.data_ptr();
    half* xrp = (half*) xr.data_ptr(); float* xsp = (float*) xs.data_ptr();
    p3_rot_kernel<<<dim3(npos, 2), NA, 0, st>>>((const half*) x.data_ptr(), tk, ei, P(gsu), P(usu), xrp, xsp, ps);
    launch_gemm<I, H>(make_map(xrp, H, 2 * ps), rbt, P(gtr), P(utr), (float*) gu.data_ptr(), ps, 2, ps, st);
    p3_act_kernel<<<npos, NA, 0, st>>>(ei, P(gsv), P(usv), P(dsu), (const float*) gu.data_ptr(), xsp, (half*) ar.data_ptr(),
                                       (float*) as.data_ptr(), ps);
    float* dp = (float*) xr.data_ptr();                      // the down sums reuse the (consumed) rotated inputs
    launch_gemm<H, I>(make_map(ar.data_ptr(), I, ps), rbt, P(dtr), P(dtr), dp, ps, 1, 0, st);
    p3_fin_kernel<<<npos, NA, 0, st>>>(tk, ei, (const float*) wt.data_ptr(), P(dsv), dp, (const float*) as.data_ptr(),
                                       (float*) out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &moe3t_gemm);
    m.def("scratch", &moe3t_scratch);
    m.def("forward", &moe3t_forward);
}
