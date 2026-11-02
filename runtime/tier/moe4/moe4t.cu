// Hecate int4 MoE prefill on Blackwell tensor cores (moe4t v3), sm_103a: grouped GEMMs over the moe4 int4 weights as
// stored (int4 code - 8, per-row fp32 scale, mma.sync A-fragment order), for long prompt chunks.
//   gather: the picks' input rows x[tok] into a contiguous buffer (gate / up), so the GEMM reads B by TMA tiles
//   gemm : CTA = (row block of <= 256 picks of one expert, matrix, 128 output channels). Per 64-k stage:
//          warps 0-3  decode the tile's 8 int4 strips (one 16-byte fragment word per lane and k-tile, LOP3 into f16) into
//                     a K-major SW128 shared-memory tile (4-byte stores, bank-conflict free), then arrive
//          warp 4     one thread loads the picks' 256 x 64 input tile by TMA (SW128, out-of-range rows zero-filled)
//          warp 5     one thread issues tcgen05.mma M128 x N(picks, rounded to 16) x K16 into a TMEM accumulator and
//                     commits the stage back to the producers
//          after the last stage warps 0-3 read the accumulator (lane = output channel) and store the raw sums (fp32)
//   gate + up: one CTA per (row block, 128 channels) holds both accumulators (TMEM 512 columns) over one input tile per
//          stage (8 decode warps: 4 for gate, 4 for up) and writes a = silu(s_g g) * (s_u u) * w (fp16) from TMEM
//   down : the gemm above on a; its epilogue adds s_d * d straight into out[tok] (coalesced fp32 RED, lane = channel)
//   CTA order (both GEMMs): expert-major on a 1-D grid, channel tile slow inside an expert's run of row blocks, so the
//          run's input slabs stay in L2 across its channel tiles and each weight tile is read once for all row blocks
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace {

constexpr int H = 6144, I = 2048, TM = 128, TN = 256, TK = 64, NT = 192;   // warps 0-3 decode, 4 TMA, 5 MMA
constexpr int NA = 256;                                 // act / fin threads
#ifndef STAGES
#define STAGES 2
#endif
constexpr int A_BYTES = TM * TK * 2;                    // 16 KB decoded weights
constexpr int B_BYTES = TN * TK * 2;                    // 32 KB inputs
constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
constexpr int SM_BAR = STAGES * STAGE_BYTES;            // mbarriers: full_a[S], full_b[S], empty[S], done
constexpr int SMEM = SM_BAR + 8 * (3 * STAGES + 1) + 16 + 1024;

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
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void st32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(v) : "memory"); }
// one 32-bit fragment word = A fragment of one k-tile: {(row g, k 2c..), (row g+8, k 2c..), (row g, k 2c+8..), (row g+8,
// k 2c+8..)}, LOP3 magic number into f16 minus 8 (exact; as moe4)
__device__ __forceinline__ void dec8(uint32_t w, uint32_t* a)
{
    const uint32_t top = w >> 8; uint32_t h0, h1, h2, h3;
    asm("lop3.b32 %0, %1, 0x000f000f, 0x64006400, 0xea;" : "=r"(h0) : "r"(w));
    asm("lop3.b32 %0, %1, 0x00f000f0, 0x64006400, 0xea;" : "=r"(h1) : "r"(w));
    asm("lop3.b32 %0, %1, 0x000f000f, 0x64006400, 0xea;" : "=r"(h2) : "r"(top));
    asm("lop3.b32 %0, %1, 0x00f000f0, 0x64006400, 0xea;" : "=r"(h3) : "r"(top));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(a[0]) : "r"(h0), "r"(0x64086408u));
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a[1]) : "r"(h1), "r"(0x2c002c00u), "r"(0xd480d480u));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(a[2]) : "r"(h2), "r"(0x64086408u));
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a[3]) : "r"(h3), "r"(0x2c002c00u), "r"(0xd480d480u));
}
// byte offset of the 16-byte chunk c (k 8c .. 8c + 7) of row r in a K-major SW128 tile (64 k = 128 B per row)
__device__ __forceinline__ uint32_t sw128(int r, int c) { return (r >> 3) * 1024 + (r & 7) * 128 + ((c ^ (r & 7)) << 4); }

// hecate (e): expert-major CTA order on a 1-D grid. Every row block contributes Y CTAs (Y = matrices x channel tiles), so
// an expert's run of row blocks [r0, r1) owns CTAs [Y r0, Y r1); inside the run the channel tile is the slow index. The
// run's input slabs stay in L2 across its channel tiles and each weight tile is read once for all its row blocks.
__device__ __forceinline__ void cta_map(const int4* __restrict__ rbt, int nrb, int Y, int b, int& rb, int& y)
{
    const int g = b / Y, e = rbt[g].x;
    int r0 = g; while (r0 > 0 && rbt[r0 - 1].x == e) --r0;
    int r1 = g + 1; while (r1 < nrb && rbt[r1].x == e) ++r1;
    const int n = r1 - r0, i = b - Y * r0;
    y = i / n; rb = r0 + i % n;
}

// ---------------------------------------------------------------------------------------------------------------------
// gemm: CTA (row block rb, matrix p and 128-channel tile ct). rbt[rb] = {expert, first pick, picks (<= 256)}.
// Inputs: tmb = TMA map of the picks' inputs, row (p * plane + pick) holds pick's K inputs for matrix p.
// w0 / w1: int64 [nb] pointers to the experts' fragment-order int4 of matrix p = 0 / 1; out [MATS][ps][N] fp32 raw sums
// hecate (f): FIN = the down GEMM adds s_d[ch] * sum straight into the output row of its pick's token (coalesced fp32 RED,
// lane = channel), replacing the fp32 [picks][H] round trip and the fin kernel (same products; the adds were atomic before)
template <int N, int K, bool FIN>
__global__ void __launch_bounds__(NT, 2) p4t_gemm_kernel(const __grid_constant__ CUtensorMap tmb, const int4* __restrict__ rbt,
    const int64_t* __restrict__ w0, const int64_t* __restrict__ w1, float* __restrict__ out, int ps, int plane, int nrb, int mats,
    const int* __restrict__ tok, const int64_t* __restrict__ sd)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    constexpr int KB = K / TK, CT = N / TM;
    int rbi, yi; cta_map(rbt, nrb, mats * CT, blockIdx.x, rbi, yi);
    const int4 rb = rbt[rbi];
    const int e = rb.x, pos0 = rb.y, rows = rb.z;
    const int p = yi / CT, ct = yi % CT;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t bar0 = su32(smem + SM_BAR), done = bar0 + 8 * 3 * STAGES;
    uint32_t* tm_base = (uint32_t*) (smem + SM_BAR + 8 * (3 * STAGES + 1));
    const int nrow16 = (rows + 15) & ~15;

    if (tid == 0)
    {
        for (int s = 0; s < STAGES; ++s) { bar_init(bar0 + 8 * s, 128); bar_init(bar0 + 8 * (STAGES + s), 1); bar_init(bar0 + 8 * (2 * STAGES + s), 1); }
        bar_init(done, 1);
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
        // -------------------------------------------------------------------- decode: strips 2w, 2w + 1 of the tile
        const uint4* wsrc = (const uint4*) (p ? w1 : w0)[e];
        const int g = lane >> 2, c = lane & 3;
        const uint4* src[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const int S = ct * 8 + 2 * warp + j;                    // moe4 strip: group S / 4, strip S % 4
            src[j] = wsrc + ((size_t) (S >> 2) * KB * 4 + (S & 3)) * 32 + lane;
        }
        uint4 nxt[2] = {__ldg(src[0]), __ldg(src[1])};
        for (int kb = 0; kb < KB; ++kb)
        {
            const int st = kb % STAGES;
            const uint4 cur[2] = {nxt[0], nxt[1]};
            if (kb + 1 < KB) { nxt[0] = __ldg(src[0] + (size_t) (kb + 1) * 128); nxt[1] = __ldg(src[1] + (size_t) (kb + 1) * 128); }
            if (kb >= STAGES) bar_wait(bar0 + 8 * (2 * STAGES + st), ((kb / STAGES) - 1) & 1);
            const uint32_t A = su32(smem + st * STAGE_BYTES);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                const int rlo = (2 * warp + j) * 16 + g, rhi = rlo + 8;
                const uint32_t wv[4] = {cur[j].x, cur[j].y, cur[j].z, cur[j].w};
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                {
                    uint32_t a[4];
                    dec8(wv[t], a);
                    st32(A + sw128(rlo, 2 * t) + 4 * c, a[0]);
                    st32(A + sw128(rhi, 2 * t) + 4 * c, a[1]);
                    st32(A + sw128(rlo, 2 * t + 1) + 4 * c, a[2]);
                    st32(A + sw128(rhi, 2 * t + 1) + 4 * c, a[3]);
                }
            }
            fence_async_smem();
            bar_arrive(bar0 + 8 * st);
        }
        // -------------------------------------------------------------------- epilogue: lane = output channel
        bar_wait(done, 0);
        tc_fence_after();
        const uint32_t lanes = (uint32_t) (warp * 32) << 16;
        float* ob = out + ((size_t) p * ps + pos0) * N + ct * TM + warp * 32 + lane;
        const int ch = ct * TM + warp * 32 + lane;
        const float sc = FIN ? ((const float*) sd[e])[ch] : 0.f;
        for (int c0 = 0; c0 < nrow16; c0 += 32)
        {
            uint32_t v[32];
            TLD32(v, tmem + lanes + c0);
            tld_wait();
            #pragma unroll
            for (int j = 0; j < 32; ++j)
                if (c0 + j < rows)
                {
                    if (FIN) atomicAdd(out + (size_t) __ldg(tok + pos0 + c0 + j) * N + ch, __uint_as_float(v[j]) * sc);
                    else ob[(size_t) (c0 + j) * N] = __uint_as_float(v[j]);
                }
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

// gate + up: CTA (row block rb, 128-channel tile ct of I); 10 warps: 0-3 decode gate strips (and the epilogue), 4-7 decode
// up strips, 8 TMA, 9 MMA. ar [ps][I] fp16 = silu(s_g * g) * (s_u * u) * w
constexpr int NT_GU = 320, STAGES_GU = 3, STAGE_GU = 2 * A_BYTES + B_BYTES, SM_BAR_GU = STAGES_GU * STAGE_GU;
constexpr int SMEM_GU = SM_BAR_GU + 8 * (3 * STAGES_GU + 1) + 16 + 1024;
__global__ void __launch_bounds__(NT_GU, 1) p4t_gu_kernel(const __grid_constant__ CUtensorMap tmb, const int4* __restrict__ rbt,
    const int64_t* __restrict__ gw, const int64_t* __restrict__ uw, const int64_t* __restrict__ gs, const int64_t* __restrict__ us,
    const float* __restrict__ wt, half* __restrict__ ar, int nrb)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    constexpr int KB = H / TK;
    int rbi, ct; cta_map(rbt, nrb, I / TM, blockIdx.x, rbi, ct);  // hecate (e)
    const int4 rb = rbt[rbi];
    const int e = rb.x, pos0 = rb.y, rows = rb.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t bar0 = su32(smem + SM_BAR_GU), done = bar0 + 8 * 3 * STAGES_GU;
    uint32_t* tm_base = (uint32_t*) (smem + SM_BAR_GU + 8 * (3 * STAGES_GU + 1));
    const int nrow16 = (rows + 15) & ~15;
    if (tid == 0)
    {
        for (int s = 0; s < STAGES_GU; ++s)
        {
            bar_init(bar0 + 8 * s, 256); bar_init(bar0 + 8 * (STAGES_GU + s), 1); bar_init(bar0 + 8 * (2 * STAGES_GU + s), 1);
        }
        bar_init(done, 1);
    }
    if (warp == 9)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(su32(tm_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = *tm_base;

    if (warp < 8)
    {
        // -------------------------------------------------------------------- decode: matrix m = warp / 4, strips 2w', 2w' + 1
        const int m = warp >> 2, wq = warp & 3;
        const uint4* wsrc = (const uint4*) (m ? uw : gw)[e];
        const int g = lane >> 2, c = lane & 3;
        const uint4* src[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const int S = ct * 8 + 2 * wq + j;
            src[j] = wsrc + ((size_t) (S >> 2) * KB * 4 + (S & 3)) * 32 + lane;
        }
        uint4 nxt[2] = {__ldg(src[0]), __ldg(src[1])};
        for (int kb = 0; kb < KB; ++kb)
        {
            const int st = kb % STAGES_GU;
            const uint4 cur[2] = {nxt[0], nxt[1]};
            if (kb + 1 < KB) { nxt[0] = __ldg(src[0] + (size_t) (kb + 1) * 128); nxt[1] = __ldg(src[1] + (size_t) (kb + 1) * 128); }
            if (kb >= STAGES_GU) bar_wait(bar0 + 8 * (2 * STAGES_GU + st), ((kb / STAGES_GU) - 1) & 1);
            const uint32_t A = su32(smem + st * STAGE_GU + m * A_BYTES);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                const int rlo = (2 * wq + j) * 16 + g, rhi = rlo + 8;
                const uint32_t wv[4] = {cur[j].x, cur[j].y, cur[j].z, cur[j].w};
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                {
                    uint32_t a[4];
                    dec8(wv[t], a);
                    st32(A + sw128(rlo, 2 * t) + 4 * c, a[0]);
                    st32(A + sw128(rhi, 2 * t) + 4 * c, a[1]);
                    st32(A + sw128(rlo, 2 * t + 1) + 4 * c, a[2]);
                    st32(A + sw128(rhi, 2 * t + 1) + 4 * c, a[3]);
                }
            }
            fence_async_smem();
            bar_arrive(bar0 + 8 * st);
        }
        if (warp < 4)
        {
            // ---------------------------------------------------------------- epilogue: lane = channel, both accumulators
            bar_wait(done, 0);
            tc_fence_after();
            const uint32_t lanes = (uint32_t) (warp * 32) << 16;
            const int ch = ct * TM + warp * 32 + lane;
            const float sg = ((const float*) gs[e])[ch], su = ((const float*) us[e])[ch];
            half* ob = ar + (size_t) pos0 * I + ch;
            for (int c0 = 0; c0 < nrow16; c0 += 32)
            {
                uint32_t vg[32], vu[32];
                TLD32(vg, tmem + lanes + c0);
                TLD32(vu, tmem + lanes + 256 + c0);
                tld_wait();
                #pragma unroll
                for (int j = 0; j < 32; ++j)
                    if (c0 + j < rows)
                    {
                        const float gg = __uint_as_float(vg[j]) * sg, uu = __uint_as_float(vu[j]) * su;
                        ob[(size_t) (c0 + j) * I] = __float2half_rn(gg / (1.f + __expf(-gg)) * uu * __ldg(wt + pos0 + c0 + j));
                    }
            }
            tc_fence_before();
        }
    }
    else if (warp == 8)
    {
        // -------------------------------------------------------------------- inputs: one TMA tile per stage
        if (lane == 0)
            for (int kb = 0; kb < KB; ++kb)
            {
                const int st = kb % STAGES_GU;
                if (kb >= STAGES_GU) bar_wait(bar0 + 8 * (2 * STAGES_GU + st), ((kb / STAGES_GU) - 1) & 1);
                bar_expect_tx(bar0 + 8 * (STAGES_GU + st), B_BYTES);
                tma_load_2d(su32(smem + st * STAGE_GU + 2 * A_BYTES), &tmb, kb * TK, pos0, bar0 + 8 * (STAGES_GU + st));
            }
        __syncwarp();
    }
    else
    {
        // -------------------------------------------------------------------- MMA issue: gate into cols 0.., up into 256..
        if (lane == 0)
        {
            const uint32_t idesc = (1u << 4) | ((uint32_t) (nrow16 >> 3) << 17) | ((uint32_t) (TM >> 4) << 24);
            for (int kb = 0; kb < KB; ++kb)
            {
                const int st = kb % STAGES_GU;
                bar_wait(bar0 + 8 * st, (kb / STAGES_GU) & 1);
                bar_wait(bar0 + 8 * (STAGES_GU + st), (kb / STAGES_GU) & 1);
                tc_fence_after();
                const uint32_t ag = su32(smem + st * STAGE_GU), au = ag + A_BYTES, b = ag + 2 * A_BYTES;
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                {
                    const uint64_t bd = sdesc(b + s * 32, 16, 1024, 2);
                    mma_ss(tmem, sdesc(ag + s * 32, 16, 1024, 2), bd, idesc, kb > 0 || s > 0);
                    mma_ss(tmem + 256, sdesc(au + s * 32, 16, 1024, 2), bd, idesc, kb > 0 || s > 0);
                }
                tc_commit(bar0 + 8 * (2 * STAGES_GU + st));
            }
            tc_commit(done);
        }
        __syncwarp();
    }
    __syncthreads();
    if (warp == 9)
    {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
    }
#endif
}

// act: block (pick pos), 8 columns per thread: a = silu(s_g * g) * (s_u * u) * w
__global__ void __launch_bounds__(NA) p4_act_kernel(const int* __restrict__ eid, const float* __restrict__ wt,
    const int64_t* __restrict__ sg, const int64_t* __restrict__ su, const float* __restrict__ gu, half* __restrict__ ar, int ps)
{
    const int pos = blockIdx.x, e = eid[pos], col = threadIdx.x * 8;
    const float wj = wt[pos];
    const float* g = gu + (size_t) pos * I + col; const float* u = gu + ((size_t) ps + pos) * I + col;
    const float* sgp = (const float*) sg[e] + col; const float* sup = (const float*) su[e] + col;
    __align__(16) half o[8];
    #pragma unroll
    for (int i = 0; i < 8; i += 4)
    {
        const float4 gv = *(const float4*) (g + i), uv = *(const float4*) (u + i);
        const float4 a = *(const float4*) (sgp + i), b = *(const float4*) (sup + i);
        const float g0 = gv.x * a.x, g1 = gv.y * a.y, g2 = gv.z * a.z, g3 = gv.w * a.w;
        o[i + 0] = __float2half_rn(g0 / (1.f + __expf(-g0)) * uv.x * b.x * wj);
        o[i + 1] = __float2half_rn(g1 / (1.f + __expf(-g1)) * uv.y * b.y * wj);
        o[i + 2] = __float2half_rn(g2 / (1.f + __expf(-g2)) * uv.z * b.z * wj);
        o[i + 3] = __float2half_rn(g3 / (1.f + __expf(-g3)) * uv.w * b.w * wj);
    }
    *(uint4*) (ar + (size_t) pos * I + col) = *(const uint4*) o;
}

// rows of x[tok[pos]] into xg [npos][H] (gate / up inputs, contiguous for the TMA tiles)
__global__ void __launch_bounds__(256) gather_kernel(const half* __restrict__ x, const int* __restrict__ tok, half* __restrict__ xg)
{
    const uint4* src = (const uint4*) (x + (size_t) tok[blockIdx.x] * H);
    uint4* dst = (uint4*) (xg + (size_t) blockIdx.x * H);
    for (int i = threadIdx.x; i < H / 8; i += 256) dst[i] = src[i];
}

PFN_cuTensorMapEncodeTiled_v12000 encode_fn()
{
    static PFN_cuTensorMapEncodeTiled_v12000 f = [] {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        TORCH_CHECK(p && q == cudaDriverEntryPointSuccess, "moe4t: cuTensorMapEncodeTiled not found");
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
    TORCH_CHECK(r == CUDA_SUCCESS, "moe4t: cuTensorMapEncodeTiled failed ", (int) r);
    return m;
}

template <int N, int K, bool FIN = false>
void launch_gemm(const CUtensorMap& tmb, const at::Tensor& rbt, const int64_t* w0, const int64_t* w1, float* out, int ps, int mats,
                 int plane, cudaStream_t st, const int* tok = nullptr, const int64_t* sd = nullptr)
{
    static bool attr = [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(p4t_gemm_kernel<N, K, FIN>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        return true;
    }();
    (void) attr;
    const int nrb = (int) rbt.size(0);
    p4t_gemm_kernel<N, K, FIN><<<(unsigned) (nrb * mats * (N / TM)), NT, SMEM, st>>>(tmb, (const int4*) rbt.data_ptr(),
                                                                                    w0, w1, out, ps, plane, nrb, mats, tok, sd);
}
}  // namespace

// one grouped GEMM (tests): gate/up shape (N = I, K = H; xs [rows][H] are the picks' inputs, both matrices read them) or
// down shape (N = H, K = I, xs [rows][I])
void moe4t_gemm(at::Tensor xs, at::Tensor rbt, at::Tensor w0, at::Tensor w1, at::Tensor out, int64_t down)
{
    const at::cuda::OptionalCUDAGuard guard(xs.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int ps = (int) out.size(1);
    const CUtensorMap m = make_map(xs.data_ptr(), (int) xs.size(1), (int) xs.size(0));
    if (down)
        launch_gemm<H, I>(m, rbt, (const int64_t*) w0.data_ptr(), (const int64_t*) w0.data_ptr(), (float*) out.data_ptr(), ps, 1, 0, st);
    else
        launch_gemm<I, H>(m, rbt, (const int64_t*) w0.data_ptr(), (const int64_t*) w1.data_ptr(), (float*) out.data_ptr(), ps, 2, 0, st);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// scratch for up to npos picks per call: gu [2][npos][I] fp32, ar [npos][I] fp16, d [1][npos][H] fp32
std::vector<at::Tensor> moe4t_scratch(at::Device dev, int64_t npos)
{
    auto f = at::TensorOptions().device(dev);
    return {at::empty({2, npos, I}, f.dtype(at::kFloat)), at::empty({npos, I}, f.dtype(at::kHalf)),
            at::empty({1, npos, H}, f.dtype(at::kFloat))};
}

// as moe4p_forward, row blocks of <= 256 picks
void moe4t_forward(at::Tensor x, at::Tensor tok, at::Tensor eid, at::Tensor wt, at::Tensor rbt, at::Tensor gw, at::Tensor uw,
                   at::Tensor dw, at::Tensor gs, at::Tensor us, at::Tensor ds, at::Tensor gu, at::Tensor ar, at::Tensor d,
                   at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int npos = (int) tok.numel(), ps = (int) gu.size(1);
    TORCH_CHECK(x.size(1) == H && x.dtype() == at::kHalf && out.dtype() == at::kFloat && npos <= ps && rbt.size(1) == 4,
                "moe4t: shapes");
    if (npos == 0) return;
    auto P = [](const at::Tensor& t) { return (const int64_t*) t.data_ptr(); };
    const int* tk = (const int*) tok.data_ptr(); const int* ei = (const int*) eid.data_ptr();
    float* gup = (float*) gu.data_ptr(); half* arp = (half*) ar.data_ptr(); float* dp = (float*) d.data_ptr();
    half* xg = (half*) d.data_ptr();                         // the gathered inputs use the down buffer (consumed first)
    gather_kernel<<<npos, 256, 0, st>>>((const half*) x.data_ptr(), tk, xg);
    static bool attr_gu = [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(p4t_gu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_GU));
        return true;
    }();
    (void) attr_gu; (void) gup;
    p4t_gu_kernel<<<(unsigned) (rbt.size(0) * (I / TM)), NT_GU, SMEM_GU, st>>>(make_map(xg, H, npos), (const int4*) rbt.data_ptr(),
        P(gw), P(uw), P(gs), P(us), (const float*) wt.data_ptr(), arp, (int) rbt.size(0));
    // hecate (f): the down GEMM adds into out itself (no d round trip, no fin kernel)
    launch_gemm<H, I, true>(make_map(arp, I, npos), rbt, P(dw), P(dw), (float*) out.data_ptr(), ps, 1, 0, st, tk, P(ds));
    (void) dp; (void) ei;                                    // the fin kernel's inputs, now unused
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &moe4t_gemm);
    m.def("scratch", &moe4t_scratch);
    m.def("forward", &moe4t_forward);
}
