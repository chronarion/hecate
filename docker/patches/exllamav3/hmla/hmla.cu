// hecate: tcgen05 (UMMA / TMEM) sparse attention for GLM-5.3's MLA prefill over ExLlamaV3's Q6 latent cache, sm_103a.
// Replaces the DSA gather kernel of long prompt chunks (EXL3_HMLA_PREFILL): out = softmax(q K^T) K over each query row's
// top-k selected cache rows (V is the latent part of K), for all 64 heads of the row at once.
//
// CTA = one query row x 64 heads, 416 threads:
//   warps 0-3  softmax (TMEM lane quadrants): the two S halves summed through shared memory, online softmax (rescale only
//              when the running maximum grows by more than 2^8), P (fp16) into shared memory, O rescaled in TMEM, epilogue
//              (normalize, inverse per-32 Hadamard rotation, store)
//   warps 4-11 loads: Q (rotated) staged once; then two halves of 4 warps alternate 64-token tiles (half h fills K tile
//              h), gathering the packed Q6 rows straight from global memory (issued before the buffer wait, so their
//              latency overlaps it) and dequantizing them into a K-major SW128 tile; cache slots resolved four tiles ahead
//   warp 12    TMEM allocation; lane 0 copies Q into TMEM (tcgen05.cp) and issues every tcgen05.mma from an event loop
//              (QK of the next tile and PV of the previous one as soon as their inputs are ready)
// Dual-half layout (as in FlashMLA's SM100 kernels): a .ws M64 N128 MMA reads A from TMEM lanes 0-63 for n < 64 and lanes
// 64-127 for n >= 64, and B rows 0-63 / 64-127 respectively. Q's even 64-dim atoms sit in lanes 0-63 and its odd atoms in
// lanes 64-127; the K tile (64 tokens x 8 atoms, K-major SW128, 8 KB atom columns) read as 128 rows x 4 atom-column pairs
// splits the same way, so one MMA yields two partial dot products of 64 tokens; the 64 rope dims split 32 / 32 (K-major
// SW64). S = the sum of the two halves. V = the K tile's latent part read MN-major (SW128, 64-row view).
// TMEM: O cols 0..255 (.ws M64 N256 x 2), Q cols 256..399, S cols 400..463.
// Q6 cache: per token 16 groups of 32 values, 6 int32 words each (4-bit plane: word v/8, shift (v%8)*4; 2-bit plane:
// word 4 + v/16, shift (v%16)*2), value = (raw - 31.5) * scale / 32, in the per-32 rotated domain.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "hmla.cuh"
#include "util.h"
#include "util.cuh"

namespace {
constexpr int DC = 512, DR = 64, H = 64, TT = 64, PAGE = 256, NT = 416, NL = 256;
constexpr int KTILE = 65536 + 8192;              // bytes: latent 8 atom columns x 8 KB (SW128) + rope 128 rows x 64 B (SW64)
constexpr int SM_K0 = 0, SM_K1 = KTILE;          // K tiles (tile 1 also stages Q before the first copy)
constexpr int SM_P = 2 * KTILE;                  // P: 64 heads x 64 tokens fp16, K-major interleaved
constexpr int SM_XB = SM_P + 8192;               // float [2][64][33] S-half exchange (padded rows)
constexpr int SM_XS = SM_XB + 2 * 64 * 33 * 4;   // float [2][64]
constexpr int SM_RING = SM_XS + 2 * 64 * 4;      // int [8][64] cache slots (tile t's written at load iteration t - 4)
constexpr int SM_BAR = SM_RING + 8 * 64 * 4;     // mbarriers: q, qcp, k[2], s, sfree, p, o[2]
constexpr int SM_TM = SM_BAR + 10 * 8;
constexpr int SM_END = SM_TM + 16;
constexpr uint32_t TM_O = 0, TM_Q = 256, TM_S = 400;
constexpr uint32_t ID_QK = (1u << 4) | ((128u >> 3) << 17) | ((64u >> 4) << 24);                 // F16 -> F32, M64 N128 (dual), K/K
constexpr uint32_t ID_PV = (1u << 4) | (1u << 16) | ((256u >> 3) << 17) | ((64u >> 4) << 24);    // M64 N256, A K-major, B MN-major

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(uint32_t b, uint32_t parity)
{
    asm volatile("{\n .reg .pred P1;\n W:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @!P1 bra W;\n}\n"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ bool bar_test(uint32_t b, uint32_t parity)
{
    uint32_t ok;
    asm volatile("{\n .reg .pred P1;\n mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n selp.u32 %0, 1, 0, P1;\n}\n"
                 : "=r"(ok) : "r"(b), "r"(parity) : "memory");
    return ok != 0;
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
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
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\n}\n"
                 :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void mma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc, int acc)
{
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\n}\n"
                 :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void tc_cp(uint32_t taddr, uint64_t sd)
{
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(taddr), "l"(sd) : "memory");
}
#define TLD32(r, addr) asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), \
      "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), \
      "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), \
      "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(addr))
#define TST32(addr, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" \
    :: "r"(addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), \
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), \
      "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), \
      "r"(r[30]), "r"(r[31]) : "memory")
__device__ __forceinline__ void tld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tst_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t pack_h2(float x, float y) { __half2 h = __floats2half2_rn(x, y); return *(uint32_t*) &h; }

// latent element (row, dim) of a K-major SW128 tile in the 64-row view: 16-byte chunk address
__device__ __forceinline__ int lat_off(int row, int dim) { const int c = (dim & 63) >> 3; return (dim >> 6) * 8192 + (row >> 3) * 1024 + (row & 7) * 128 + ((c ^ (row & 7)) << 4); }
// rope element (row, e) e < 64: dual row r' = row + 64 * (e / 32), K-major SW64 (64 B rows, 512 B atoms)
__device__ __forceinline__ int rope_off(int row, int e) { const int r2 = row + 64 * (e >> 5), c = (e & 31) >> 3; return 65536 + (r2 >> 3) * 512 + (r2 & 7) * 64 + ((c ^ ((r2 >> 1) & 3)) << 4); }
__device__ __forceinline__ void wht32r(float* x)
{
    #pragma unroll
    for (int b = 1; b < 32; b <<= 1)
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            if (!(i & b)) { const float u = x[i], v = x[i + b]; x[i] = u + v; x[i + b] = u - v; }
    #pragma unroll
    for (int i = 0; i < 32; ++i) x[i] *= 0.17677669529663687f;
}
// 32 Q6 values of one group (6 words + scale) -> 4 x uint4 fp16, byte-parallel: per 8 values, the 4-bit plane word's
// even / odd nibbles and the matching 2-bit fields (spread nibble -> byte) form 6-bit codes in bytes, PRMT places each
// (even, odd) pair into the two halves of 0x6400 (1024 + code exactly, sign-replicated zero bytes), then
// (h - 1055) * s - 0.5 s = (code - 31.5) * s in fp16
__device__ __forceinline__ uint32_t lop_or_and(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xA8;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;   // (a | b) & c
}
__device__ __forceinline__ uint32_t spread4(uint32_t x)          // 16-bit x: nibble j -> low nibble of byte j
{
    const uint32_t t = lop_or_and(x, x << 8, 0x00FF00FFu);
    return lop_or_and(t, t << 4, 0x0F0F0F0Fu);
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel)
{
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ void deq_group(const uint32_t* wd, float scale, uint4* o)
{
    const __half2 s2 = __float2half2_rn(scale * (1.f / 32.f));
    const __half2 b2 = __float2half2_rn(-0.5f * scale * (1.f / 32.f));
    const __half2 k1055 = __float2half2_rn(1055.f);
    #pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const uint32_t w = wd[k], c = (wd[4 + (k >> 1)] >> (16 * (k & 1))) & 0xFFFFu;
        const uint32_t lo = ((w & 0x0F0F0F0Fu) << 2) | spread4(c & 0x3333u);
        const uint32_t hi = (((w >> 4) & 0x0F0F0F0Fu) << 2) | spread4((c >> 2) & 0x3333u);
        uint32_t h[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint32_t v = prmt(lo, hi, (8u << 12) | ((4u + j) << 8) | (8u << 4) | (uint32_t) j) | 0x64006400u;
            __half2 x = __hsub2(*(__half2*) &v, k1055);
            x = __hfma2(x, s2, b2);
            h[j] = *(uint32_t*) &x;
        }
        o[k] = make_uint4(h[0], h[1], h[2], h[3]);
    }
}
__global__ void __launch_bounds__(NT, 1) hmla_kernel(const half* __restrict__ q_lat, const half* __restrict__ q_pe,
    const int* __restrict__ ckv, const half* __restrict__ ckv_s, const half* __restrict__ kpe, const int* __restrict__ bt,
    const int* __restrict__ idx, int k_pad, int R, float sl2, half* __restrict__ out)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200   // tcgen05: sm_100 family (the RTX build gets an empty body)
    extern __shared__ __align__(1024) unsigned char smem[];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned char* Kt[2] = {smem + SM_K0, smem + SM_K1};
    unsigned char* Ps = smem + SM_P;
    float* XB = (float*) (smem + SM_XB); float* XS = (float*) (smem + SM_XS); int* RING = (int*) (smem + SM_RING);
    const uint32_t bq = su32(smem + SM_BAR), bqcp = bq + 8, bk0 = bq + 16, bs = bq + 32, bsf = bq + 40, bp = bq + 48, bo0 = bq + 56;
    uint32_t* tm_base = (uint32_t*) (smem + SM_TM);

    const int ntiles = (k_pad + TT - 1) / TT;
    auto slot_of = [&](int j) -> int                      // cache slot of list entry j (block table row 0: one sequence)
    {
        if (j >= k_pad) return -1;
        const int e = __ldg(idx + (size_t) r * k_pad + j);
        return e < 0 ? -1 : __ldg(bt + e / PAGE) * PAGE + e % PAGE;
    };

    if (tid == 0)
    {
        bar_init(bq, NL); bar_init(bqcp, 1); bar_init(bk0, 128); bar_init(bk0 + 8, 128); bar_init(bs, 1); bar_init(bsf, 128);
        bar_init(bp, 128); bar_init(bo0, 1); bar_init(bo0 + 8, 1);
    }
    if (warp == 12)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(su32(tm_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    if (tid < 256) { const int t = tid >> 6, i = tid & 63; if (t < ntiles) RING[t * 64 + i] = slot_of(t * TT + i); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = *tm_base;

    if (warp >= 4 && warp < 12)
    {
        // ------------------------------------------------------------------------------------------- load WG (256 thr)
        const int lt = tid - 128;
        // Q into K tile 1 in the dual-half layout: (head, 32-dim block) items
        for (int it = lt; it < H * 18; it += NL)
        {
            const int h = it / 18, b = it % 18;
            uint4 v[4];
            if (b < 16)
            {
                const uint4* src = (const uint4*) (q_lat + ((size_t) h * R + r) * DC + b * 32);
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = __ldg(src + i);
                {
                    float x[32];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        const half2* hp = (const half2*) &v[i];
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) { const float2 f = __half22float2(hp[k]); x[i * 8 + 2 * k] = f.x; x[i * 8 + 2 * k + 1] = f.y; }
                    }
                    wht32r(x);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        v[i] = make_uint4(pack_h2(x[i * 8], x[i * 8 + 1]), pack_h2(x[i * 8 + 2], x[i * 8 + 3]),
                                          pack_h2(x[i * 8 + 4], x[i * 8 + 5]), pack_h2(x[i * 8 + 6], x[i * 8 + 7]));
                }
                // dual view: atom kj -> row h + 64 * (kj % 2), atom column pair kj / 2 (pair stride 16 KB = 2 x 8 KB columns)
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    const int d = b * 32 + i * 8, kj = d >> 6, r2 = h + 64 * (kj & 1), c = (d & 63) >> 3;
                    *(uint4*) (Kt[1] + (kj >> 1) * 16384 + (r2 >> 3) * 1024 + (r2 & 7) * 128 + ((c ^ (r2 & 7)) << 4)) = v[i];
                }
            }
            else
            {
                const uint4* src = (const uint4*) (q_pe + ((size_t) r * H + h) * DR + (b - 16) * 32);
                #pragma unroll
                for (int i = 0; i < 4; ++i) *(uint4*) (Kt[1] + rope_off(h, (b - 16) * 32 + i * 8)) = __ldg(src + i);
            }
        }
        fence_async_smem();
        bar_arrive(bq);
        const int hh = lt >> 7, ht = lt & 127;            // half h fills K tile h for tiles h, h + 2, ...
        for (int t = hh; t < ntiles; t += 2)
        {
            const int buf = hh;
            const int* sl = RING + (t & 7) * 64;
            uint32_t wd[8][6]; float scv[8]; uint4 rp[4];
            // gathers first: their latency overlaps the buffer wait
            {
                #pragma unroll
                for (int u = 0; u < 8; ++u)
                {
                    const int c = ht + u * 128, ti = c >> 4, gi = c & 15;
                    const int sv = sl[ti], s0 = sv < 0 ? 0 : sv;
                    const int2* src = (const int2*) (ckv + (size_t) s0 * 96 + gi * 6);
                    const int2 a = __ldg(src), b2 = __ldg(src + 1), c2 = __ldg(src + 2);
                    wd[u][0] = a.x; wd[u][1] = a.y; wd[u][2] = b2.x; wd[u][3] = b2.y; wd[u][4] = c2.x; wd[u][5] = c2.y;
                    scv[u] = sv < 0 ? 0.f : __half2float(__ldg(ckv_s + (size_t) s0 * 16 + gi));
                }
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    const int c = ht + u * 128, ti = c >> 3, q = c & 7, sv = sl[ti];
                    rp[u] = sv < 0 ? make_uint4(0, 0, 0, 0) : __ldg((const uint4*) (kpe + (size_t) sv * DR + q * 8));
                }
            }
            if (t >= 2) bar_wait(bo0 + 8 * buf, ((t - 2) >> 1) & 1);      // PV(t-2) done: tile buffer and ring slot free
            if (t == 1) bar_wait(bqcp, 0);                                 // Q copied out of tile buffer 1
            if (ht < 64 && t + 4 < ntiles) RING[((t + 4) & 7) * 64 + ht] = slot_of((t + 4) * TT + ht);
            {
                #pragma unroll
                for (int u = 0; u < 8; ++u)
                {
                    const int c = ht + u * 128, ti = c >> 4, gi = c & 15;
                    uint4 o[4];
                    deq_group(wd[u], scv[u], o);
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) *(uint4*) (Kt[buf] + lat_off(ti, gi * 32 + q * 8)) = o[q];
                }
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    const int c = ht + u * 128, ti = c >> 3, q = c & 7;
                    *(uint4*) (Kt[buf] + rope_off(ti, q * 8)) = rp[u];
                }
            }
            fence_async_smem();
            bar_arrive(bk0 + 8 * buf);
        }
    }
    else if (warp == 12)
    {
        // ------------------------------------------------------------------------------------------- MMA warp
        if (lane == 0)
        {
            const uint32_t pa = su32(Ps);
            bar_wait(bq, 0);
            tc_fence_after();
            const uint32_t qa = su32(Kt[1]);
            #pragma unroll
            for (int s = 0; s < 16; ++s) tc_cp(tmem + TM_Q + s * 8, sdesc(qa + (s >> 2) * 16384 + (s & 3) * 32, 16, 1024, 2));
            #pragma unroll
            for (int s = 0; s < 2; ++s) tc_cp(tmem + TM_Q + 128 + s * 8, sdesc(qa + 65536 + s * 32, 16, 512, 4));
            tc_commit(bqcp);
            int nq = 0, np_ = 0;                           // next QK tile, next PV tile
            while (np_ < ntiles)
            {
                // QK(nq): K tile ready, and softmax has read S(nq - 1)
                if (nq < ntiles && bar_test(bk0 + 8 * (nq & 1), (nq >> 1) & 1) && (nq == 0 || bar_test(bsf, (nq - 1) & 1)))
                {
                    tc_fence_after();
                    const uint32_t ka = su32(Kt[nq & 1]);
                    {
                        #pragma unroll
                        for (int s = 0; s < 16; ++s)
                            mma_ts(tmem + TM_S, tmem + TM_Q + s * 8, sdesc(ka + (s >> 2) * 16384 + (s & 3) * 32, 16, 1024, 2), ID_QK, s > 0);
                        #pragma unroll
                        for (int s = 0; s < 2; ++s)
                            mma_ts(tmem + TM_S, tmem + TM_Q + 128 + s * 8, sdesc(ka + 65536 + s * 32, 16, 512, 4), ID_QK, 1);
                    }
                    tc_commit(bs);
                    ++nq;
                }
                // PV(np_): P ready
                if (np_ < nq && bar_test(bp, np_ & 1))
                {
                    tc_fence_after();
                    const uint32_t ka = su32(Kt[np_ & 1]);
                    {
                        #pragma unroll
                        for (int s = 0; s < 4; ++s)
                            #pragma unroll
                            for (int nh = 0; nh < 2; ++nh)
                                mma_ss(tmem + TM_O + nh * 128, sdesc(pa + s * 2048, 1024, 128, 0),
                                       sdesc(ka + nh * 4 * 8192 + s * 2048, 8192, 1024, 2), ID_PV, np_ > 0 || s > 0);
                    }
                    tc_commit(bo0 + 8 * (np_ & 1));
                    ++np_;
                }
            }
        }
        __syncwarp();
    }
    else
    {
        // ------------------------------------------------------------------------------------------- softmax WG (warps 0-3)
        const int row = (warp & 1) * 32 + lane, part = warp >> 1;      // TMEM lane 32w + lane = row + 64 * part
        const uint32_t lanes = (uint32_t) (warp * 32) << 16;
        float m = -1e30f, l = 0.f;
        for (int t = 0; t < ntiles; ++t)
        {
            bar_wait(bs, t & 1);
            tc_fence_after();
            uint32_t sa[32], sb[32];
            TLD32(sa, tmem + lanes + TM_S);
            TLD32(sb, tmem + lanes + TM_S + 32);
            tld_wait();
            tc_fence_before();
            bar_arrive(bsf);                               // S may be overwritten by QK(t+1)
            // this thread owns tokens part*32 .. +31 of its row: send the other 32 partial sums to the partner warp
            const uint32_t* mine = part ? sb : sa; const uint32_t* other = part ? sa : sb;
            #pragma unroll
            for (int j = 0; j < 32; ++j) XB[(part * 64 + row) * 33 + j] = __uint_as_float(other[j]);
            named_bar(2, 128);
            float s[32], mx = -1e30f;
            uint32_t okm = 0;
            const int* sl = (const int*) (smem + SM_RING) + (t & 7) * 64 + part * 32;
            #pragma unroll
            for (int j = 0; j < 32; ++j)
            {
                const bool ok = sl[j] >= 0;
                okm |= (uint32_t) ok << j;
                s[j] = ok ? (__uint_as_float(mine[j]) + XB[((part ^ 1) * 64 + row) * 33 + j]) * sl2 : -1e30f;
                mx = fmaxf(mx, s[j]);
            }
            XS[part * 64 + row] = mx;
            named_bar(2, 128);
            mx = fmaxf(mx, XS[(part ^ 1) * 64 + row]);
            const bool grow = mx > m + 8.f || t == 0;
            const float mn = grow ? fmaxf(m, mx) : m;
            const float alpha = exp2f(m - mn);
            float sum = 0.f;
            uint32_t ph[16];
            #pragma unroll
            for (int j = 0; j < 32; j += 2)
            {
                const float p0 = (okm >> j) & 1 ? exp2f(s[j] - mn) : 0.f, p1 = (okm >> (j + 1)) & 1 ? exp2f(s[j + 1] - mn) : 0.f;
                sum += p0 + p1; ph[j >> 1] = pack_h2(p0, p1);
            }
            l = l * alpha + sum; m = mn;
            if (t > 0)
            {
                bar_wait(bo0 + 8 * ((t - 1) & 1), ((t - 1) >> 1) & 1);     // PV(t-1) done: O stable, P free
                tc_fence_after();
                if (__any_sync(0xffffffffu, alpha != 1.f))
                {
                    #pragma unroll
                    for (int c = 0; c < 256; c += 32)
                    {
                        uint32_t o[32];
                        TLD32(o, tmem + lanes + TM_O + c);
                        tld_wait();
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) o[j] = __float_as_uint(__uint_as_float(o[j]) * alpha);
                        TST32(tmem + lanes + TM_O + c, o);
                    }
                    tst_wait();
                }
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u)
            {
                const int k = part * 32 + u * 8;
                *(uint4*) (Ps + (k >> 3) * 1024 + (row >> 3) * 128 + (row & 7) * 16) = make_uint4(ph[4 * u], ph[4 * u + 1], ph[4 * u + 2], ph[4 * u + 3]);
            }
            fence_async_smem();
            tc_fence_before();
            bar_arrive(bp);
            named_bar(2, 128);                             // XB / XS reads done before the next tile's writes
        }
        XS[part * 64 + row] = l;
        named_bar(2, 128);
        const float lt_ = l + XS[(part ^ 1) * 64 + row];
        if (ntiles > 0) { bar_wait(bo0 + 8 * ((ntiles - 1) & 1), ((ntiles - 1) >> 1) & 1); tc_fence_after(); }
        const float f = lt_ > 0.f ? 1.f / lt_ : 0.f;
        #pragma unroll 1
        for (int c = 0; c < 256; c += 32)
        {
            uint32_t o[32];
            float x[32];
            if (ntiles > 0) { TLD32(o, tmem + lanes + TM_O + c); tld_wait(); }
            #pragma unroll
            for (int j = 0; j < 32; ++j) x[j] = ntiles > 0 ? __uint_as_float(o[j]) * f : 0.f;
            const int d0 = (c >> 7) * 256 + part * 128 + (c & 127);       // 32 contiguous latent dims of this row
            wht32r(x);                                     // back from the cache's rotated domain
            uint4* dst = (uint4*) (out + ((size_t) row * R + r) * DC + d0);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                dst[i] = make_uint4(pack_h2(x[i * 8], x[i * 8 + 1]), pack_h2(x[i * 8 + 2], x[i * 8 + 3]), pack_h2(x[i * 8 + 4], x[i * 8 + 5]), pack_h2(x[i * 8 + 6], x[i * 8 + 7]));
        }
        tc_fence_before();
    }
    __syncthreads();
    if (warp == 12)
    {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
    }
#endif
}
}  // namespace

// q_lat (H, R, 512) fp16 (unrotated absorbed queries), q_pe (R, H, 64); ckv (pages, 256, 96) int32, ckv_s (pages, 256, 16)
// fp16, kpe (pages, 256, [1,] 64) fp16; block_table (1, pages) int32; indices (R, k_pad) int32, -1 = padding; out (H, R, 512)
void hmla_attn(at::Tensor q_lat, at::Tensor q_pe, at::Tensor ckv, at::Tensor ckv_s, at::Tensor kpe, at::Tensor block_table,
               at::Tensor indices, double scale, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(q_lat.device());
    const int R = (int) q_lat.size(1);
    const int64_t slots = kpe.numel() / DR;
    TORCH_CHECK(q_lat.size(0) == H && q_lat.size(2) == DC && q_pe.numel() == (int64_t) R * H * DR && ckv.numel() == slots * 96 &&
                ckv_s.numel() == slots * 16 && indices.dim() == 2 && indices.size(0) == R && block_table.size(0) == 1 &&
                q_lat.is_contiguous() && q_pe.is_contiguous() && indices.is_contiguous() && block_table.is_contiguous(), "hmla_attn: shapes");
    static bool attr = [] { cudaFuncSetAttribute(hmla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_END); return true; }();
    (void) attr;
    hmla_kernel<<<R, NT, SM_END, at::cuda::getCurrentCUDAStream().stream()>>>((const half*) q_lat.data_ptr(), (const half*) q_pe.data_ptr(),
        (const int*) ckv.data_ptr(), (const half*) ckv_s.data_ptr(), (const half*) kpe.data_ptr(), (const int*) block_table.data_ptr(),
        (const int*) indices.data_ptr(), (int) indices.size(1), R, (float) scale * 1.4426950408889634f, (half*) out.data_ptr());
    cuda_check(cudaPeekAtLastError());
}
