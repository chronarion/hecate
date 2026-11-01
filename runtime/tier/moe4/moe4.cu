// RED-only flushes (A: gate / up partials into the fp32 scratch, act = silu(g) * u * w by a separate one-pass kernel that
// also re-zeroes the scratch; B: partials straight into y, no y32 / counters / fences), gate+up on the register ring (no
// dynamic shared memory), both main kernels capped at 2 blocks / SM (A launched on B's grid), and programmatic dependent
// launch (act / B run their routing prologue while the previous kernel drains), selected by the V7_* switches below.
// Layer-40 routing, 4 rows, 144 hot experts: 77.1 us per MoE call (94.9 without these switches).
#define V7_A_RED
#define V7_B_RED
#define V7_REGRING
#define V7_ACT_WIDE
#define V7_PDL
// Routed int4 MoE for the verify shape (route B hot band): experts stored as uniform int4 (code - 8) with per-row fp32
// scales, pre-swizzled per lane in mma.sync m16n8k16 A-fragment order ([group of 4 strips][k-block of 64][strip][lane]
// x 16 bytes, see moe4_pack). Two kernels per MoE layer, both stream-K over the work of the step's distinct hot experts
// (computed in-kernel from the selection, so the launch shape is static and graph-safe):
//   A: gate and up GEMVs (K 6144) for every distinct expert; partial sums -> fp32 scratch with atomics; the last warp
//      to finish a (expert, 64-row group) across gate and up writes act = silu(g) * u * routing weight (fp16, zero for
//      rows that did not pick the expert) and cleans the scratch
//   B: down GEMVs (K 2048) of every distinct expert on its act, scaled per row, accumulated across experts into an
//      fp32 scratch; the last warp per output group adds it into y (fp32, the caller's MoE output) and cleans up.
// Decode: LOP3 magic number into f16 + HADD2/HFMA2 (exact); x and act read row-major through L1; rows >= R are zero.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

#ifndef PDV
#define PDV 2
#endif
constexpr int H = 6144, I = 2048, SPW = 4, WPB = 8, PD = PDV, TOPK = 8, MAXR = 8, MAXE = 64;
constexpr int GA = I / (16 * SPW), KA = H / 64;        // phase A: 32 groups x 96 k-blocks per matrix
constexpr int GB = H / (16 * SPW), KBB = I / 64;       // phase B: 96 groups x 32 k-blocks
constexpr int UA = 2 * GA * KA, UB = GB * KBB;          // units per expert: 6144 (gate + up), 3072 (down)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
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

// distinct hot experts of this step (sorted), routing weight per (slot, row); shared by both kernels. Parallel: picks
// set bits of a 256-bit mask in shared memory, thread 0 turns the mask into the sorted slot list (popc/ffs over 8 words),
// then every pick adds its weight into its (slot, row) cell. No serialized global loads.
struct Route { int m; int e[MAXE]; float w[MAXE][MAXR]; unsigned mask[8]; short slot[256]; };
__device__ void route(const int64_t* sel, const half* rw, int R, int lo, int n_hot, Route& rt)   // ids [lo, lo + n_hot) -> slots 0..
{
    const int t = threadIdx.x, np = R * TOPK;
    if (t < 8) rt.mask[t] = 0u;
    for (int i = t; i < MAXE * MAXR; i += blockDim.x) (&rt.w[0][0])[i] = 0.f;
    int e = -1; float wv = 0.f;
    if (t < np) { e = (int) (sel[t] - lo); wv = __half2float(rw[t]); if (e < 0 || e >= n_hot) e = -1; }
    __syncthreads();
    if (e >= 0) atomicOr(&rt.mask[e >> 5], 1u << (e & 31));
    __syncthreads();
    if (t == 0)
    {
        int m = 0;
        for (int wd = 0; wd < 8; ++wd)
        {
            unsigned b = rt.mask[wd];
            while (b) { const int bit = __ffs(b) - 1; b &= b - 1; const int ee = wd * 32 + bit; rt.slot[ee] = (short) m; rt.e[m++] = ee; }
        }
        rt.m = m;
    }
    __syncthreads();
    if (e >= 0) atomicAdd(&rt.w[rt.slot[e]][t / TOPK], wv);
    __syncthreads();
}

// one stream-K pass over `n` units starting at global unit u0; unit -> (slot j, matrix mat, group gr, k-block kb).
// MATS matrices per expert, G groups x KBN k-blocks per matrix; K = KBN * 64; xsrc(j) gives the B rows.
template <int MATS, int G, int KBN, typename WPtr, typename XPtr, typename Flush>
__device__ __forceinline__ void stream(int u0, int n, int R, WPtr wptr, XPtr xptr, Flush flush)
{
    constexpr int UPM = G * KBN, UPE = MATS * UPM, K = KBN * 64;
    const int lane = threadIdx.x % 32;
    // compute cursor and prefetch cursor
    int j = u0 / UPE, mat = (u0 % UPE) / UPM, gr = (u0 % UPM) / KBN, kb = u0 % KBN;
    int pj = j, pmat = mat, pu = u0 % UPM;                       // prefetch: (slot, matrix, unit within matrix)
    const uint4* pp = wptr(pj, pmat) + (size_t) pu * 32 * SPW + lane;
    int left = n;                                                 // units not yet prefetched
    auto advance_p = [&]() {
        --left; pp += 32 * SPW;
        if (++pu == UPM) { pu = 0; if (++pmat == MATS) { pmat = 0; ++pj; } if (left > 0) pp = wptr(pj, pmat) + lane; }
    };
    uint4 ring[PD][SPW];
    const int xn = lane / 4, kq = (lane % 4) * 2; const bool live = xn < R;
#ifdef V7_XPF
    uint32_t xring[PD][8];
    auto xload = [&](uint32_t* d) {
        const uint32_t* xb = (const uint32_t*) (xptr(pj) + (size_t) (live ? xn : 0) * K + kq) + (pu % KBN) * 32;
        #pragma unroll
        for (int t = 0; t < 4; ++t) { d[2 * t] = live ? __ldg(xb + t * 8) : 0u; d[2 * t + 1] = live ? __ldg(xb + t * 8 + 4) : 0u; }
    };
#endif
    #pragma unroll
    for (int i = 0; i < PD; ++i)
    {
        #pragma unroll
        for (int s = 0; s < SPW; ++s) ring[i][s] = left > 0 ? __ldg(pp + s * 32) : make_uint4(0, 0, 0, 0);
#ifdef V7_XPF
        if (left > 0) xload(xring[i]);
#endif
        if (left > 0) advance_p();
    }
    const uint32_t* xr = (const uint32_t*) (xptr(j) + (size_t) (live ? xn : 0) * K + kq);
    bool whole = kb == 0;
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
    for (int done = 0; done < n; done += PD)
    {
        #pragma unroll
        for (int jj = 0; jj < PD; ++jj)
        {
            if (done + jj >= n) break;
            const bool more = left > 0;
            uint32_t bb[8];
#ifdef V7_XPF
            #pragma unroll
            for (int t = 0; t < 8; ++t) bb[t] = xring[jj][t];
            if (more) xload(xring[jj]);
#else
            #pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const int k0 = kb * 64 + t * 16;
                bb[2 * t] = live ? __ldg(xr + k0 / 2) : 0u; bb[2 * t + 1] = live ? __ldg(xr + k0 / 2 + 4) : 0u;
            }
#endif
            #pragma unroll
            for (int s = 0; s < SPW; ++s)
            {
                const uint32_t ww[4] = {ring[jj][s].x, ring[jj][s].y, ring[jj][s].z, ring[jj][s].w};
                #pragma unroll
                for (int t = 0; t < 4; ++t) { uint32_t a[4]; dec8(ww[t], a); mma16816(c[s], a, bb[2 * t], bb[2 * t + 1]); }
                if (more) ring[jj][s] = __ldg(pp + s * 32);
            }
            if (more) advance_p();
            ++kb;
            if (kb == KBN || done + jj == n - 1)
            {
                flush(j, mat, gr, c, whole && kb == KBN);
                #pragma unroll
                for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
                whole = true;
                if (kb == KBN)
                {
                    kb = 0;
                    if (++gr == G) { gr = 0; if (++mat == MATS) { mat = 0; ++j; } }
                    if (done + jj < n - 1) xr = (const uint32_t*) (xptr(j) + (size_t) (live ? xn : 0) * K + kq);
                }
            }
        }
    }
}


// stream() with a cp.async shared-memory ring of STAGES units per warp instead of the register ring (gate+up phase:
// 4.98 vs 3.97 TB/s in the stagebench sweep); same cursors and flush protocol
constexpr int STAGES = 3;
template <int MATS, int G, int KBN, typename WPtr, typename XPtr, typename Flush>
__device__ __forceinline__ void stream_cp(int u0, int n, int R, WPtr wptr, XPtr xptr, Flush flush, uint4* ring)
{
    constexpr int UPM = G * KBN, UPE = MATS * UPM, K = KBN * 64;
    const int lane = threadIdx.x % 32;
    int j = u0 / UPE, mat = (u0 % UPE) / UPM, gr = (u0 % UPM) / KBN, kb = u0 % KBN;
    int pj = j, pmat = mat, pu = u0 % UPM;
    const uint4* pp = wptr(pj, pmat) + (size_t) pu * 32 * SPW + lane;
    int left = n;
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(ring + lane);
    auto issue = [&](int slot) {
        #pragma unroll
        for (int s = 0; s < SPW; ++s)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sbase + (uint32_t) ((slot * SPW + s) * 32 * 16)), "l"(pp + s * 32));
        --left; pp += 32 * SPW;
        if (++pu == UPM) { pu = 0; if (++pmat == MATS) { pmat = 0; ++pj; } if (left > 0) pp = wptr(pj, pmat) + lane; }
    };
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) { if (left > 0) issue(i); asm volatile("cp.async.commit_group;"); }
    const int xn = lane / 4, kq = (lane % 4) * 2; const bool live = xn < R;
    const uint32_t* xr = (const uint32_t*) (xptr(j) + (size_t) (live ? xn : 0) * K + kq);
    bool whole = kb == 0;
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
    int slot = 0;
    for (int done = 0; done < n; ++done)
    {
        if (left > 0) issue((slot + STAGES - 1) % STAGES);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1));
        uint32_t bb[8];
        #pragma unroll
        for (int t = 0; t < 4; ++t)
        {
            const int k0 = kb * 64 + t * 16;
            bb[2 * t] = live ? __ldg(xr + k0 / 2) : 0u; bb[2 * t + 1] = live ? __ldg(xr + k0 / 2 + 4) : 0u;
        }
        #pragma unroll
        for (int s = 0; s < SPW; ++s)
        {
            const uint4 wv = ring[(slot * SPW + s) * 32 + lane];
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
            #pragma unroll
            for (int t = 0; t < 4; ++t) { uint32_t a[4]; dec8(ww[t], a); mma16816(c[s], a, bb[2 * t], bb[2 * t + 1]); }
        }
        slot = (slot + 1) % STAGES;
        ++kb;
        if (kb == KBN || done == n - 1)
        {
            flush(j, mat, gr, c, whole && kb == KBN);
            #pragma unroll
            for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
            whole = true;
            if (kb == KBN)
            {
                kb = 0;
                if (++gr == G) { gr = 0; if (++mat == MATS) { mat = 0; ++j; } }
                if (done < n - 1) xr = (const uint32_t*) (xptr(j) + (size_t) (live ? xn : 0) * K + kq);
            }
        }
    }
    asm volatile("cp.async.wait_all;");
}

__device__ __forceinline__ int splits(int ua, int ub, int upw) { return ub / upw - ua / upw + 1; }

__global__ void __launch_bounds__(WPB * 32, 2) moe4_a_kernel
(
    const half* __restrict__ x, const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo, int n_hot,
    const int64_t* __restrict__ gate_w, const int64_t* __restrict__ up_w,
    const int64_t* __restrict__ gate_s, const int64_t* __restrict__ up_s,
    half* __restrict__ act,          // [MAXE][MAXR][I]
    float* gu32,                     // [MAXE][2][MAXR][I], zero
    int* ctr                         // [MAXE][GA], zero
)
{
    __shared__ Route rt;
    extern __shared__ __align__(16) uint4 ring_s[];          // [WPB][STAGES][SPW][32]
#ifdef V7_PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    route(sel, rw, R, lo, n_hot, rt);
#if defined(V7_ROUTE_ONLY) || defined(V7_SKIP_A)
    return;
#endif
    const int m = rt.m; if (m == 0) return;
    const int total = m * UA, nwarps = gridDim.x * WPB;
    const int upw = (total + nwarps - 1) / nwarps, gw = blockIdx.x * WPB + threadIdx.x / 32;
    const int u0 = gw * upw, n = min(total - u0, upw);
    if (n <= 0) return;
    const int lane = threadIdx.x % 32;
    auto wptr = [&](int j, int mat) { return (const uint4*) (mat ? up_w : gate_w)[rt.e[min(j, m - 1)]]; };
    auto xptr = [&](int) { return x; };
    auto flush = [&](int j, int mat, int gr, float (*c)[4], bool) {
#ifdef V7_NOFLUSH
        if (c[0][0] != 1234.5f) return;
#endif
        const int col = (lane % 4) * 2, rb = gr * SPW * 16 + lane / 4;
        float* base = gu32 + ((size_t) (j * 2 + mat) * MAXR) * I;
        #pragma unroll
        for (int s = 0; s < SPW; ++s)
        {
            const int r0 = rb + s * 16;
            if (col < R)     { atomicAdd(base + (size_t) col * I + r0, c[s][0]);       atomicAdd(base + (size_t) col * I + r0 + 8, c[s][2]); }
            if (col + 1 < R) { atomicAdd(base + (size_t) (col + 1) * I + r0, c[s][1]); atomicAdd(base + (size_t) (col + 1) * I + r0 + 8, c[s][3]); }
        }
#ifdef V7_A_RED
        return;
#endif
        __threadfence(); __syncwarp();
        const int ua = j * UA + gr * KA, ub = ua + KA - 1;                     // gate group; up group is + GA * KA
        const int need = splits(ua, ub, upw) + splits(ua + GA * KA, ub + GA * KA, upw);
        int last = 0;
        if (lane == 0) last = atomicAdd(ctr + j * GA + gr, 1) == need - 1;
        last = __shfl_sync(0xffffffffu, last, 0);
        if (!last) return;
        __threadfence();
        const float* gs = (const float*) gate_s[rt.e[j]]; const float* us = (const float*) up_s[rt.e[j]];
        float* gp = gu32 + ((size_t) (j * 2) * MAXR) * I; float* upp = gu32 + ((size_t) (j * 2 + 1) * MAXR) * I;
        for (int i = lane; i < SPW * 16 * R; i += 32)
        {
            const int r = gr * SPW * 16 + i % (SPW * 16), row = i / (SPW * 16);
            const size_t o = (size_t) row * I + r;
            const float g = __ldcg(gp + o) * gs[r], u = __ldcg(upp + o) * us[r];
            act[((size_t) j * MAXR) * I + o] = __float2half(g / (1.f + __expf(-g)) * u * rt.w[j][row]);
            gp[o] = 0.f; upp[o] = 0.f;
        }
        if (lane == 0) ctr[j * GA + gr] = 0;
    };
#ifdef V7_REGRING
    stream<2, GA, KA>(u0, n, R, wptr, xptr, flush);
#else
    stream_cp<2, GA, KA>(u0, n, R, wptr, xptr, flush, ring_s + (size_t) (threadIdx.x / 32) * STAGES * SPW * 32);
#endif
}

__global__ void __launch_bounds__(WPB * 32, 2) moe4_b_kernel
(
    const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo, int n_hot,
    const int64_t* __restrict__ down_w, const int64_t* __restrict__ down_s,
    const half* __restrict__ act,    // [MAXE][MAXR][I]
    float* __restrict__ y,           // [R][H], accumulated into
    float* y32,                      // [MAXR][H], zero
    int* ctr                         // [GB], zero
)
{
    // group-major units: (output group gr, expert slot j, k-block kb); a warp sums its experts in registers (each
    // expert's partial scaled by its row scale at the expert boundary) and flushes once per group it touches
    __shared__ Route rt;
#ifdef V7_SKIP_B
    return;
#endif
    route(sel, rw, R, lo, n_hot, rt);
#ifdef V7_PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const int m = rt.m; if (m == 0) return;
    const int UG = m * KBB, total = GB * UG, nwarps = gridDim.x * WPB;
    const int upw = (total + nwarps - 1) / nwarps, gw = blockIdx.x * WPB + threadIdx.x / 32;
    const int u0 = gw * upw, n = min(total - u0, upw);
    if (n <= 0) return;
    const int lane = threadIdx.x % 32;
    int gr = u0 / UG, j = (u0 % UG) / KBB, kb = u0 % KBB;
    // prefetch cursor
    int pgr = gr, pj = j, pkb = kb, left = n;
    auto pbase = [&](int g, int jj) { return (const uint4*) down_w[rt.e[jj]] + (size_t) g * KBB * 32 * SPW + lane; };
    const uint4* pp = pbase(pgr, pj) + (size_t) pkb * 32 * SPW;
    auto advance_p = [&]() {
        --left; pp += 32 * SPW;
        if (++pkb == KBB) { pkb = 0; if (++pj == m) { pj = 0; ++pgr; } if (left > 0) pp = pbase(pgr, pj); }
    };
    uint4 ring[PD][SPW];
    const int xn = lane / 4, kq = (lane % 4) * 2; const bool live = xn < R;
    auto xrow = [&](int jj) { return (const uint32_t*) (act + ((size_t) jj * MAXR + (live ? xn : 0)) * I + kq); };
#ifdef V7_XPF
    uint32_t xring[PD][8];
    auto xload = [&](uint32_t* d) {
        const uint32_t* xb = xrow(pj) + pkb * 32;
        #pragma unroll
        for (int t = 0; t < 4; ++t) { d[2 * t] = live ? __ldcg(xb + t * 8) : 0u; d[2 * t + 1] = live ? __ldcg(xb + t * 8 + 4) : 0u; }
    };
#endif
    #pragma unroll
    for (int i = 0; i < PD; ++i)
    {
        #pragma unroll
        for (int s = 0; s < SPW; ++s) ring[i][s] = left > 0 ? __ldg(pp + s * 32) : make_uint4(0, 0, 0, 0);
#ifdef V7_XPF
        if (left > 0) xload(xring[i]);
#endif
        if (left > 0) advance_p();
    }
    const uint32_t* xr = xrow(j);
#ifdef V7_DSPF
    float dsv[SPW][2];
    auto dsload = [&](int jj, int g) {
        const float* ds = (const float*) down_s[rt.e[jj]];
        #pragma unroll
        for (int s = 0; s < SPW; ++s) { const int r0 = g * SPW * 16 + s * 16 + lane / 4; dsv[s][0] = __ldg(ds + r0); dsv[s][1] = __ldg(ds + r0 + 8); }
    };
    dsload(j, gr);
#endif
    float c[SPW][4], tot[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) { c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f; tot[s][0] = tot[s][1] = tot[s][2] = tot[s][3] = 0.f; }
    const int col = (lane % 4) * 2;
    for (int done = 0; done < n; done += PD)
    {
        #pragma unroll
        for (int jj = 0; jj < PD; ++jj)
        {
            if (done + jj >= n) break;
            const bool more = left > 0;
            uint32_t bb[8];
#ifdef V7_XPF
            #pragma unroll
            for (int t = 0; t < 8; ++t) bb[t] = xring[jj][t];
            if (more) xload(xring[jj]);
#else
            #pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const int k0 = kb * 64 + t * 16;
                bb[2 * t] = live ? __ldg(xr + k0 / 2) : 0u; bb[2 * t + 1] = live ? __ldg(xr + k0 / 2 + 4) : 0u;
            }
#endif
            #pragma unroll
            for (int s = 0; s < SPW; ++s)
            {
                const uint32_t ww[4] = {ring[jj][s].x, ring[jj][s].y, ring[jj][s].z, ring[jj][s].w};
                #pragma unroll
                for (int t = 0; t < 4; ++t) { uint32_t a[4]; dec8(ww[t], a); mma16816(c[s], a, bb[2 * t], bb[2 * t + 1]); }
                if (more) ring[jj][s] = __ldg(pp + s * 32);
            }
            if (more) advance_p();
            ++kb;
            const bool last_unit = done + jj == n - 1;
            if (kb == KBB || last_unit)
            {
                // expert boundary: fold this expert's partial into the group total with its row scales
                const float* ds = (const float*) down_s[rt.e[j]];
                #pragma unroll
                for (int s = 0; s < SPW; ++s)
                {
#ifdef V7_DSPF
                    const float s0 = dsv[s][0], s1 = dsv[s][1]; (void) ds;
#else
                    const int r0 = gr * SPW * 16 + s * 16 + lane / 4; const float s0 = ds[r0], s1 = ds[r0 + 8];
#endif
                    tot[s][0] += c[s][0] * s0; tot[s][1] += c[s][1] * s0; tot[s][2] += c[s][2] * s1; tot[s][3] += c[s][3] * s1;
                    c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
                }
                const bool group_end = kb == KBB && j == m - 1;
                if (group_end || last_unit)
                {
#ifdef V7_B_RED
                    float* const yd = y;
#else
                    float* const yd = y32;
#endif
                    #pragma unroll
                    for (int s = 0; s < SPW; ++s)
                    {
                        const int r0 = gr * SPW * 16 + s * 16 + lane / 4;
                        if (col < R)     { atomicAdd(yd + (size_t) col * H + r0, tot[s][0]);       atomicAdd(yd + (size_t) col * H + r0 + 8, tot[s][2]); }
                        if (col + 1 < R) { atomicAdd(yd + (size_t) (col + 1) * H + r0, tot[s][1]); atomicAdd(yd + (size_t) (col + 1) * H + r0 + 8, tot[s][3]); }
                        tot[s][0] = tot[s][1] = tot[s][2] = tot[s][3] = 0.f;
                    }
#ifndef V7_B_RED
                    __threadfence(); __syncwarp();
                    const int need = splits(gr * UG, (gr + 1) * UG - 1, upw);
                    int lastw = 0;
                    if (lane == 0) lastw = atomicAdd(ctr + gr, 1) == need - 1;
                    lastw = __shfl_sync(0xffffffffu, lastw, 0);
                    if (lastw)
                    {
                        __threadfence();
                        for (int i = lane; i < SPW * 16 * R; i += 32)
                        {
                            const int r = gr * SPW * 16 + i % (SPW * 16), row = i / (SPW * 16);
                            float* p = y32 + (size_t) row * H + r;
                            y[(size_t) row * H + r] += __ldcg(p); *p = 0.f;
                        }
                        if (lane == 0) ctr[gr] = 0;
                    }
#endif
                }
                if (kb == KBB)
                {
                    kb = 0;
                    if (++j == m) { j = 0; ++gr; }
                    if (!last_unit) xr = xrow(j);
#ifdef V7_DSPF
                    if (!last_unit) dsload(j, gr);
#endif
                }
            }
        }
    }
}

// V7_A_RED: act[j][r][i] = silu(g * gs) * (u * us) * w[j][r] from the complete gate / up sums in gu32 (zeroed again)
__global__ void __launch_bounds__(256) moe4_act_kernel(const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo,
    int n_hot, const int64_t* __restrict__ gate_s, const int64_t* __restrict__ up_s, half* __restrict__ act, float* gu32)
{
    __shared__ Route rt;
    route(sel, rw, R, lo, n_hot, rt);
#ifdef V7_PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    const int m = rt.m, n = m * R * (I / 4);
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n; q += gridDim.x * blockDim.x)
    {
        const int j = q / (R * (I / 4)), r = (q / (I / 4)) % R, i = (q % (I / 4)) * 4;
        float* gp = gu32 + ((size_t) (j * 2) * MAXR + r) * I + i; float* up = gp + (size_t) MAXR * I;
        const float4 g4 = __ldcg((const float4*) gp), u4 = __ldcg((const float4*) up);
        const float4 gs = *(const float4*) ((const float*) gate_s[rt.e[j]] + i), us = *(const float4*) ((const float*) up_s[rt.e[j]] + i);
        const float w = rt.w[j][r];
        auto f = [&](float g, float s1, float u, float s2) { g *= s1; return __float2half(g / (1.f + __expf(-g)) * u * s2 * w); };
        half2* ap = (half2*) (act + ((size_t) j * MAXR + r) * I + i);
        ap[0] = __halves2half2(f(g4.x, gs.x, u4.x, us.x), f(g4.y, gs.y, u4.y, us.y));
        ap[1] = __halves2half2(f(g4.z, gs.z, u4.z, us.z), f(g4.w, gs.w, u4.w, us.w));
        *(float4*) gp = make_float4(0.f, 0.f, 0.f, 0.f); *(float4*) up = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

// prefill path: packed int4 (fragment order) -> fp16 [N][K] row-major, times the row scale. One thread per lane-uint4.
__global__ void moe4_dequant_kernel(const uint4* __restrict__ q, const float* __restrict__ sc, half* __restrict__ out, int N, int K)
{
    const int kb_n = K / 64;
    const size_t total = (size_t) N / 16 * kb_n * 32;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x)
    {
        const int l = (int) (i % 32), s = (int) ((i / 32) % SPW), kb = (int) ((i / (32 * SPW)) % kb_n), gr = (int) (i / ((size_t) 32 * SPW * kb_n));
        const int g = l / 4, qq = l % 4, row0 = (gr * SPW + s) * 16 + g;
        const uint4 v = q[i]; const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib)
            {
                const int c = nib >> 2, h = (nib >> 1) & 1, rh = nib & 1;
                const int row = row0 + 8 * rh, k = kb * 64 + t * 16 + h * 8 + qq * 2 + c;
                out[(size_t) row * K + k] = __float2half((float) (int) (((ww[t] >> (4 * nib)) & 15) - 8) * sc[row]);
            }
    }
}

// combine: f[r][h] += o[r][h] (RTX partials, pinned host memory read in place, may be null) + y[r][h] (hot band, then
// zeroed for the next layer: replaces the separate zero_ before moe4 and the two adds)
__global__ void moe4_combine_kernel(float* __restrict__ f, const float* __restrict__ o, float* __restrict__ y, int n,
                                    const float* __restrict__ o2, const unsigned* done)
{
    if (done)       // RTX graph: wait for this layer's DONE (reset by the next handoff of the layer, before its ISSUE)
    {
        if (threadIdx.x == 0) while (*(volatile const unsigned*) done < 1u) __nanosleep(100);
        __syncthreads();
    }
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    float4 a = *(float4*) (f + i); const float4 c = *(const float4*) (y + i);
    a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
    if (o) { const float4 b = __ldcv((const float4*) (o + i)); a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
    if (o2) { const float4 b = __ldcv((const float4*) (o2 + i)); a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
    *(float4*) (f + i) = a; *(float4*) (y + i) = make_float4(0.f, 0.f, 0.f, 0.f);
}

// handoff to the RTX band: one block copies the MoE input, selection and routing weights into the pinned slots (host
// memory, written over C2C), then releases the ISSUE flag (system scope) that the RTX stream waits on. Replaces three
// copy-engine copies + cuStreamWriteValue32 on the GB300 stream.
__global__ void __launch_bounds__(1024) moe4_handoff_kernel(const uint4* __restrict__ y, const uint4* __restrict__ sel,
    const uint32_t* __restrict__ rw, uint4* dy, uint4* ds, uint32_t* dw, unsigned* flag, unsigned gen, int ny, int ns, int nw,
    unsigned* meta, unsigned m0, unsigned m1, unsigned* flag2, unsigned gen2, unsigned* zero)
{
    for (int i = threadIdx.x; i < ny; i += blockDim.x) dy[i] = y[i];
    for (int i = threadIdx.x; i < ns; i += blockDim.x) ds[i] = sel[i];
    for (int i = threadIdx.x; i < nw; i += blockDim.x) dw[i] = rw[i];
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (meta) { meta[0] = m0; meta[1] = m1; }      // job header for the CPU band (layer, rows)
        if (zero) *(volatile unsigned*) zero = 0u;      // this layer's previous DONE (RTX graph), before the new ISSUE
        __threadfence_system();
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag), "r"(gen) : "memory");
        if (flag2) asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag2), "r"(gen2) : "memory");
    }
}

// stage: copy the step's distinct picks in the Grace-pinned trellis band [lo, hi) (first S in pick order) into HBM staging
// slots and point the band's gate / up / down trellis tables at them; every other band entry is reset to its pinned
// address (orig). Full-grid copy (16 B per thread per iteration) for memory-level parallelism over C2C; block 0 writes
// the tables. tab_* / orig_* are int64 [E] device tables (ExLlamaV3 MultiLinear.ptrs_trellis).
__global__ void __launch_bounds__(512) moe4_stage_kernel(const int64_t* __restrict__ sel, int nsel, int lo, int hi,
    const int64_t* __restrict__ orig_g, const int64_t* __restrict__ orig_u, const int64_t* __restrict__ orig_d,
    int64_t* tab_g, int64_t* tab_u, int64_t* tab_d, char* staging, int64_t gu_bytes, int64_t d_bytes, int S)
{
    __shared__ int pick[8]; __shared__ int np;
    if (threadIdx.x == 0)
    {
        int n = 0;
        for (int i = 0; i < nsel && n < S; ++i)
        {
            const int64_t e = sel[i];
            if (e < lo || e >= hi) continue;
            const int l = (int) (e - lo); bool dup = false;
            for (int k = 0; k < n; ++k) dup |= pick[k] == l;
            if (!dup) pick[n++] = l;
        }
        np = n;
    }
    __syncthreads();
    const int n = np, E = hi - lo;
    const int64_t slot = 2 * gu_bytes + d_bytes;
    if (blockIdx.x == 0)
    {
        for (int e = threadIdx.x; e < E; e += blockDim.x) { tab_g[e] = orig_g[e]; tab_u[e] = orig_u[e]; tab_d[e] = orig_d[e]; }
        __syncthreads();
        if (threadIdx.x < n)
        {
            const int l = pick[threadIdx.x]; char* b = staging + threadIdx.x * slot;
            tab_g[l] = (int64_t) b; tab_u[l] = (int64_t) (b + gu_bytes); tab_d[l] = (int64_t) (b + 2 * gu_bytes);
        }
    }
    // copy: chunk index over n slots x (gate | up | down), 16 B each
    const int64_t per = slot / 16, total = per * n;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x)
    {
        const int k = (int) (i / per); const int64_t o = (i % per) * 16; const int l = pick[k];
        const char* src = o < gu_bytes ? (const char*) orig_g[l] + o
                        : o < 2 * gu_bytes ? (const char*) orig_u[l] + (o - gu_bytes) : (const char*) orig_d[l] + (o - 2 * gu_bytes);
        *(uint4*) (staging + k * slot + o) = __ldcs((const uint4*) src);
    }
}

}  // namespace

void moe4_stage(torch::Tensor sel, int64_t lo, int64_t hi, torch::Tensor orig, torch::Tensor tab_g, torch::Tensor tab_u,
                torch::Tensor tab_d, torch::Tensor staging, int64_t gu_bytes, int64_t d_bytes, int64_t S, int64_t blocks,
                int64_t threads)
{
    const at::cuda::OptionalCUDAGuard guard(sel.device());
    TORCH_CHECK(sel.is_contiguous() && sel.scalar_type() == torch::kLong && S <= 8, "moe4_stage: sel / S");
    TORCH_CHECK(gu_bytes % 16 == 0 && d_bytes % 16 == 0, "moe4_stage: sizes");
    const int E = (int) (hi - lo);
    const int64_t* o = orig.data_ptr<int64_t>();
    TORCH_CHECK(threads >= 64 && threads <= 512 && threads % 32 == 0, "moe4_stage: threads");
    moe4_stage_kernel<<<(int) blocks, (int) threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(sel.data_ptr<int64_t>(), (int) sel.numel(),
        (int) lo, (int) hi, o, o + E, o + 2 * E, tab_g.data_ptr<int64_t>(), tab_u.data_ptr<int64_t>(), tab_d.data_ptr<int64_t>(),
        (char*) staging.data_ptr(), gu_bytes, d_bytes, (int) S);
}

void moe4_handoff(torch::Tensor y, torch::Tensor sel, torch::Tensor rw, int64_t dy, int64_t ds, int64_t dw, int64_t flag,
                  int64_t gen, int64_t rows, int64_t meta, int64_t m0, int64_t flag2, int64_t gen2, int64_t zero)
{
    const at::cuda::OptionalCUDAGuard guard(y.device());
    TORCH_CHECK(y.is_contiguous() && sel.is_contiguous() && rw.is_contiguous(), "moe4_handoff: contiguous");
    const int ny = (int) (rows * y.size(-1) * 2 / 16), ns = (int) (rows * sel.size(-1) * 8 / 16), nw = (int) (rows * rw.size(-1) * 2 / 4);
    moe4_handoff_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>((const uint4*) y.data_ptr(),
        (const uint4*) sel.data_ptr(), (const uint32_t*) rw.data_ptr(), (uint4*) dy, (uint4*) ds, (uint32_t*) dw,
        (unsigned*) flag, (unsigned) gen, ny, ns, nw, (unsigned*) meta, (unsigned) m0, (unsigned) rows, (unsigned*) flag2, (unsigned) gen2, (unsigned*) zero);
}

void moe4_combine(torch::Tensor f, int64_t o_ptr, torch::Tensor y, int64_t rows, int64_t o2_ptr, int64_t done_ptr)
{
    const at::cuda::OptionalCUDAGuard guard(f.device());
    const int n = (int) rows * H;
    TORCH_CHECK(f.is_contiguous() && f.size(-1) == H && f.dtype() == torch::kFloat, "moe4_combine: f");
    moe4_combine_kernel<<<(n / 4 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(f.data_ptr<float>(),
        (const float*) o_ptr, y.data_ptr<float>(), n, (const float*) o2_ptr, (const unsigned*) done_ptr);
}

// x [R][H] fp16, sel [R][8] int32 (hot-band local ids, others ignored), rw [R][8] fp16; tables int64 [n_hot] device
// pointers; y [R][H] fp32 accumulated into. Scratch from moe4_scratch().
void moe4_forward(torch::Tensor x, torch::Tensor sel, torch::Tensor rw, int64_t n_hot,
                  torch::Tensor gate_w, torch::Tensor up_w, torch::Tensor down_w,
                  torch::Tensor gate_s, torch::Tensor up_s, torch::Tensor down_s,
                  torch::Tensor act, torch::Tensor gu32, torch::Tensor ctr_a, torch::Tensor y32, torch::Tensor ctr_b,
                  torch::Tensor y, int64_t blocks, int64_t blocks_a, int64_t lo)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int R = (int) x.size(0);
    TORCH_CHECK(R >= 1 && R <= MAXR && x.size(1) == H && sel.size(1) == TOPK, "moe4: shape");
    TORCH_CHECK(sel.scalar_type() == torch::kLong && sel.is_contiguous(), "moe4: selection must be contiguous int64");
    TORCH_CHECK(R * TOPK <= MAXE, "moe4: too many picks");
#ifdef V7_REGRING
    constexpr int SMEM_A = 0;
#else
    constexpr int SMEM_A = WPB * STAGES * SPW * 32 * 16;
#endif
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(moe4_a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_A); attr = true; }
#ifdef V7_PDL
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; la[0].val.programmaticStreamSerializationAllowed = 1;
    auto cfg = [&](int g, int smem) { cudaLaunchConfig_t c = {}; c.gridDim = dim3(g); c.blockDim = dim3(WPB * 32); c.dynamicSmemBytes = smem;
                                      c.stream = st; c.attrs = la; c.numAttrs = 1; return c; };
    {
        cudaLaunchConfig_t c = cfg((int) blocks, SMEM_A);   // 2 blocks / SM (register ring); blocks_a is the v4 cp.async grid
        cudaLaunchKernelEx(&c, moe4_a_kernel, (const half*) x.data_ptr(), (const int64_t*) sel.data_ptr<int64_t>(), (const half*) rw.data_ptr(),
            R, (int) lo, (int) n_hot, (const int64_t*) gate_w.data_ptr<int64_t>(), (const int64_t*) up_w.data_ptr<int64_t>(),
            (const int64_t*) gate_s.data_ptr<int64_t>(), (const int64_t*) up_s.data_ptr<int64_t>(), (half*) act.data_ptr(), gu32.data_ptr<float>(),
            ctr_a.data_ptr<int>());
        cudaLaunchConfig_t c2 = cfg((int) blocks, 0);
        cudaLaunchKernelEx(&c2, moe4_act_kernel, (const int64_t*) sel.data_ptr<int64_t>(), (const half*) rw.data_ptr(), R, (int) lo, (int) n_hot,
            (const int64_t*) gate_s.data_ptr<int64_t>(), (const int64_t*) up_s.data_ptr<int64_t>(), (half*) act.data_ptr(), gu32.data_ptr<float>());
        cudaLaunchConfig_t c3 = cfg((int) blocks, 0);
        cudaLaunchKernelEx(&c3, moe4_b_kernel, (const int64_t*) sel.data_ptr<int64_t>(), (const half*) rw.data_ptr(), R, (int) lo, (int) n_hot,
            (const int64_t*) down_w.data_ptr<int64_t>(), (const int64_t*) down_s.data_ptr<int64_t>(), (const half*) act.data_ptr(), y.data_ptr<float>(),
            y32.data_ptr<float>(), ctr_b.data_ptr<int>());
        return;
    }
#endif
    moe4_a_kernel<<<(int) blocks_a, WPB * 32, SMEM_A, st>>>((const half*) x.data_ptr(), sel.data_ptr<int64_t>(), (const half*) rw.data_ptr(),
        R, (int) lo, (int) n_hot, gate_w.data_ptr<int64_t>(), up_w.data_ptr<int64_t>(), gate_s.data_ptr<int64_t>(), up_s.data_ptr<int64_t>(),
        (half*) act.data_ptr(), gu32.data_ptr<float>(), ctr_a.data_ptr<int>());
#if defined(V7_A_RED) && !defined(V7_NO_ACT)
#ifdef V7_ACT_WIDE
    moe4_act_kernel<<<(int) blocks, 256, 0, st>>>
#else
    moe4_act_kernel<<<64, 256, 0, st>>>
#endif
        (sel.data_ptr<int64_t>(), (const half*) rw.data_ptr(), R, (int) lo, (int) n_hot,
        gate_s.data_ptr<int64_t>(), up_s.data_ptr<int64_t>(), (half*) act.data_ptr(), gu32.data_ptr<float>());
#endif
    moe4_b_kernel<<<(int) blocks, WPB * 32, 0, st>>>(sel.data_ptr<int64_t>(), (const half*) rw.data_ptr(), R, (int) lo, (int) n_hot,
        down_w.data_ptr<int64_t>(), down_s.data_ptr<int64_t>(), (const half*) act.data_ptr(), y.data_ptr<float>(),
        y32.data_ptr<float>(), ctr_b.data_ptr<int>());
}

std::vector<torch::Tensor> moe4_scratch(torch::Device dev)
{
    auto f = torch::TensorOptions().device(dev).dtype(torch::kFloat), i = torch::TensorOptions().device(dev).dtype(torch::kInt);
    return {torch::zeros({MAXE, MAXR, I}, torch::TensorOptions().device(dev).dtype(torch::kHalf)),
            torch::zeros({MAXE, 2, MAXR, I}, f), torch::zeros({MAXE, GA}, i), torch::zeros({MAXR, H}, f), torch::zeros({GB}, i)};
}

torch::Tensor moe4_dequant(torch::Tensor q, torch::Tensor sc, int64_t N, int64_t K)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto out = torch::empty({N, K}, torch::TensorOptions().device(q.device()).dtype(torch::kHalf));
    moe4_dequant_kernel<<<1024, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>((const uint4*) q.data_ptr(), sc.data_ptr<float>(),
        (half*) out.data_ptr(), (int) N, (int) K);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dequant", &moe4_dequant);
    m.def("forward", &moe4_forward);
    m.def("scratch", &moe4_scratch);
    m.def("combine", &moe4_combine);
    m.def("handoff", &moe4_handoff);
    m.def("stage", &moe4_stage);
}
