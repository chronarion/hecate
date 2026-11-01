// hecate: int8 decode GEMV for EXL3 linears with an int8 copy (EXL3_I8_LINEARS), 1..8 rows.
// y[R][N] = x[R][K] (fp16, row-major) * W^T; W int8 (code - 128) with a per-output-row fp32 scale, pre-swizzled per lane
// in mma.sync m16n8k16 A-fragment order: [group of SPW strips][k-block of 32][strip][lane] x 16 bytes (see
// apply_i8_linear.py, pack_i8). Decode: LOP3 magic number into f16 + HADD2 (exact). Stream-K over contiguous units;
// partial groups accumulate with fp32 atomics into a per-linear scratch (y32 [8][N]) and the last warp per group
// (counter in ctr[N / (16 SPW)]) writes y, then zeroes the scratch and the counter: one kernel, graph-safe.
// Graph params: x (GP_gemm_A) and y (GP_gemm_C), patched per call like the trellis GEMM.
// Optional rank-4 correction (DenseI8LR, unset in the release): y[m] += f(m) * us (v^T x[m]), f(m) the
// row's mode flag (modes[item], item = m / (M / modes[8])). The same grid computes t = x v [M][4]: its first NT blocks
// each reduce a K slice into a partial, the last of them sums the partials and raises a ready flag; the GEMV warps only
// wait for it (spin, lane 0) right before they write y, and only when some row is flagged. All counters are monotonic
// (never reset, so no tail work): each t block bumps cnt after its partial; the last group bumps launches
// (done % groups); a writer waits for cnt >= (launches + 1) * NT (launches read while its own group is still open)
// and sums the NT partials of the rows it writes.
// t scratch (fp32, >= 548): [0, 512) partials [NT][32], [512, 544) t, then int counters cnt / ready / done / launches.
#include <cuda_fp16.h>
#include <cooperative_groups.h>
// Cluster-reduced int8 GEMVs (dense_i8_cl / dual_cl / gu_cl kernels); EXL3_I8_CL=0 selects the
// stream-K kernels again
#define I8_CLUSTER
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "dense_i8.cuh"
#include "../util.h"
#include "../util.cuh"

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

// t blocks: partial t[m][j] = sum_{k in slice} x[m][k] v[k][j] (m < M <= 8, j < 4), v [K][4] fp16
constexpr int LR_MAX_NT = 16;
constexpr int LR_T_OFF = LR_MAX_NT * 32, LR_CNT = LR_T_OFF + 32, LR_READY = LR_CNT + 1, LR_DONE = LR_CNT + 2, LR_LAUNCH = LR_CNT + 3;

__device__ __forceinline__ void lr_t_block(const half* __restrict__ x, const half* __restrict__ v, float* tb,
                                           int M, int K, int nt, int b)
{
    __shared__ float red[WPB][33];
    const int kc = (((K + nt - 1) / nt + 1) / 2) * 2, k0 = b * kc, k1 = min(K, k0 + kc);   // covers K for any nt
    float acc[8][4];
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
    #pragma unroll 2
    for (int k = k0 + threadIdx.x * 2; k < k1; k += blockDim.x * 2)
    {
        const uint4 vv = __ldg((const uint4*) (v + (size_t) k * 4));        // v[k][0..3], v[k + 1][0..3]
        const half2* vh = (const half2*) &vv;
        const float2 v0a = __half22float2(vh[0]), v0b = __half22float2(vh[1]);
        const float2 v1a = __half22float2(vh[2]), v1b = __half22float2(vh[3]);
        #pragma unroll
        for (int m = 0; m < 8; ++m)
        {
            if (m >= M) break;
            const float2 xx = __half22float2(__ldg((const half2*) (x + (size_t) m * K + k)));
            acc[m][0] += xx.x * v0a.x + xx.y * v1a.x;
            acc[m][1] += xx.x * v0a.y + xx.y * v1a.y;
            acc[m][2] += xx.x * v0b.x + xx.y * v1b.x;
            acc[m][3] += xx.x * v0b.y + xx.y * v1b.y;
        }
    }
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int m = 0; m < 8; ++m)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float a = acc[m][j];
            #pragma unroll
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) red[wid][m * 4 + j] = a;
        }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float a = 0.f;
        #pragma unroll
        for (int w = 0; w < WPB; ++w) a += red[w][threadIdx.x];
        tb[b * 32 + threadIdx.x] = a;
        __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd((unsigned*) (tb + LR_CNT), 1u);
}

__device__ __forceinline__ void lr_wait_ready(float* tb, int nt)
{
    // this launch's partials are in once cnt reaches (launches before it + 1) * nt (launches is stable while our group
    // is open)
    const unsigned target = (*(volatile unsigned*) (tb + LR_LAUNCH) + 1u) * (unsigned) nt;
    while ((int) (*(volatile unsigned*) (tb + LR_CNT) - target) < 0) __nanosleep(20);
    __threadfence();
}

__device__ __forceinline__ void lr_group_done(float* tb, int lane, int groups)
{
    // the last group of this launch counts the launch (the next launch starts after this kernel, t blocks included)
    if (lane == 0 && (atomicAdd((unsigned*) (tb + LR_DONE), 1u) + 1u) % (unsigned) groups == 0u)
        atomicAdd((unsigned*) (tb + LR_LAUNCH), 1u);
    __syncwarp();
}

struct LRArgs { const half* us; const half* v; float* t; const float* modes; int nt; };

// flag(m) * t[m] (after the ready flag)
__device__ __forceinline__ float4 lr_trow(const LRArgs& lr, int m, int M)
{
    const int ni = (int) lr.modes[8];
    const int item = min(m / max(1, M / max(1, ni)), max(1, ni) - 1);
    const float f = ni > 0 ? lr.modes[item] : 0.f;
    float4 tt = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int b = 0; b < lr.nt; ++b)
    {
        const float4 p = __ldcg((const float4*) (lr.t + b * 32 + m * 4));
        tt.x += p.x; tt.y += p.y; tt.z += p.z; tt.w += p.w;
    }
    tt.x *= f; tt.y *= f; tt.z *= f; tt.w *= f;
    return tt;
}

// first use per warp: wait for this launch's partials, keep the lane's two output rows (col, col + 1) in registers and
// all M rows in the warp's shared slots
__device__ __forceinline__ void lr_fetch(const LRArgs& lr, int lane, int col, int M, float4 (*tsm)[8], float4& t0, float4& t1)
{
    if (lane == 0) lr_wait_ready(lr.t, lr.nt);
    __syncwarp();
    if (lane < M) tsm[threadIdx.x / 32][lane] = lr_trow(lr, lane, M);
    __syncwarp();
    t0 = col < M ? tsm[threadIdx.x / 32][col] : make_float4(0.f, 0.f, 0.f, 0.f);
    t1 = col + 1 < M ? tsm[threadIdx.x / 32][col + 1] : make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ __forceinline__ uint2 lr_us(const LRArgs& lr, int n) { return __ldg((const uint2*) (lr.us + (size_t) n * 4)); }

__device__ __forceinline__ float lr_dot(uint2 uu, float4 tt)
{
    const float2 u01 = __half22float2(*(const half2*) &uu.x), u23 = __half22float2(*(const half2*) &uu.y);
    return u01.x * tt.x + u01.y * tt.y + u23.x * tt.z + u23.y * tt.w;
}


static bool i8_cl_on() { static const bool on = [] { const char* e = getenv("EXL3_I8_CL"); return !e || atoi(e) != 0; }(); return on; }

// hecate (I8_CLUSTER): units [a, a + n) of one 16*SPW-row group's weights (wg = the group's unit 0) into c
template <int SPW, int PD>
__device__ __forceinline__ void cl_units(const uint4* __restrict__ wg, int a, int n, const half* __restrict__ x, int M, int K,
                                         float (*c)[4])
{
    const int lane = threadIdx.x % 32;
    if (n <= 0) return;
    const uint4* pf = wg + (size_t) a * 32 * SPW + lane;
    const uint4* const pend = pf + (size_t) n * 32 * SPW;
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
    int kb = a;
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
                    uint32_t aa[4];
                    dec_i8(ww[2 * t], aa); dec_i8(ww[2 * t + 1], aa + 2);
                    mma16816(c[s], aa, bb[2 * t], bb[2 * t + 1]);
                }
                if (more) ring[j][s] = __ldg(pf + s * 32);
            }
            pf += 32 * SPW;
            ++kb;
        }
    }
}

// per-warp accumulators -> block partial part[SPW * 4 * 32] (red: [WPB][SPW * 4][33] scratch); ends synchronized
template <int SPW>
__device__ __forceinline__ void cl_block_sum(float (*c)[4], float (*red)[SPW * 4][33], float* part)
{
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int s = 0; s < SPW; ++s)
        #pragma unroll
        for (int e = 0; e < 4; ++e) red[wid][s * 4 + e][lane] = c[s][e];
    __syncthreads();
    for (int i = threadIdx.x; i < SPW * 4 * 32; i += blockDim.x)
    {
        float v_ = 0.f;
        #pragma unroll
        for (int w_ = 0; w_ < WPB; ++w_) v_ += red[w_][i / 32][i % 32];
        part[i] = v_;
    }
    __syncthreads();
}

// part index -> (output row within the group, input row m), m16n8k16 accumulator layout
template <int SPW>
__device__ __forceinline__ void cl_index(int i, int& row, int& m)
{
    const int q = i / 32, l = i % 32, s = q / 4, e = q % 4;
    row = s * 16 + l / 4 + (e >= 2 ? 8 : 0); m = (l % 4) * 2 + (e & 1);
}

// two GEMVs on one x (q_a + kv_a): clusters [0, GA) own A's groups, the rest B's
template <int SPW, int PD>
__global__ void __launch_bounds__(WPB * 32) dense_i8_dual_cl_kernel
(
    const uint4* __restrict__ wa, const float* __restrict__ sa, half* __restrict__ ya, int Na,
    const uint4* __restrict__ wb, const float* __restrict__ sb, half* __restrict__ yb, int Nb,
    const half* __restrict__ x, int M, int K
)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int CS = (int) cl.num_blocks(), rank = (int) cl.block_rank(), wid = threadIdx.x / 32;
    const int GA = Na / (16 * SPW), ci = blockIdx.x / CS, kb_n = K / 32;
    const bool B = ci >= GA;
    const int g = B ? ci - GA : ci, N = B ? Nb : Na;
    const uint4* w = (B ? wb : wa) + (size_t) g * kb_n * 32 * SPW;
    const float* scale = B ? sb : sa; half* y = B ? yb : ya;
    const int s0 = (int) ((int64_t) kb_n * rank / CS), s1 = (int) ((int64_t) kb_n * (rank + 1) / CS);
    const int a = s0 + (s1 - s0) * wid / WPB, b = s0 + (s1 - s0) * (wid + 1) / WPB;
    __shared__ float red[WPB][SPW * 4][33];
    __shared__ float part[SPW * 4 * 32];
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
    cl_units<SPW, PD>(w, a, b - a, x, M, K, c);
    cl_block_sum<SPW>(c, red, part);
    cl.sync();
    if (rank == 0)
        for (int i = threadIdx.x; i < SPW * 4 * 32; i += blockDim.x)
        {
            int row, m; cl_index<SPW>(i, row, m);
            if (m >= M) continue;
            row += g * SPW * 16;
            float v_ = 0.f;
            for (int r = 0; r < CS; ++r) v_ += *cl.map_shared_rank(&part[i], r);
            y[(size_t) m * N + row] = __float2half_rn(v_ * scale[row]);
        }
    cl.sync();
}

// gated MLP gate + up and act = silu(g * sg) * (u * su): each block streams both matrices over its K slice
template <int SPW, int PD>
__global__ void __launch_bounds__(WPB * 32) dense_i8_gu_cl_kernel
(
    const uint4* __restrict__ wg, const float* __restrict__ sg, const uint4* __restrict__ wu, const float* __restrict__ su,
    const half* __restrict__ x, half* __restrict__ act, int M, int N, int K
)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int CS = (int) cl.num_blocks(), rank = (int) cl.block_rank(), wid = threadIdx.x / 32;
    const int g = blockIdx.x / CS, kb_n = K / 32;
    const int s0 = (int) ((int64_t) kb_n * rank / CS), s1 = (int) ((int64_t) kb_n * (rank + 1) / CS);
    // warps 0..3 stream gate, 4..7 up, each over a quarter of the slice
    constexpr int HW = WPB / 2;
    const bool up = wid >= HW; const int hw = wid % HW;
    const int a = s0 + (s1 - s0) * hw / HW, b = s0 + (s1 - s0) * (hw + 1) / HW;
    __shared__ float red[WPB][SPW * 4][33];
    __shared__ float pg[SPW * 4 * 32], pu[SPW * 4 * 32];
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
    cl_units<SPW, PD>((up ? wu : wg) + (size_t) g * kb_n * 32 * SPW, a, b - a, x, M, K, c);
    {
        const int lane = threadIdx.x % 32;
        #pragma unroll
        for (int s = 0; s < SPW; ++s)
            #pragma unroll
            for (int e = 0; e < 4; ++e) red[wid][s * 4 + e][lane] = c[s][e];
        __syncthreads();
        for (int i = threadIdx.x; i < SPW * 4 * 32; i += blockDim.x)
        {
            float vg = 0.f, vu = 0.f;
            #pragma unroll
            for (int w_ = 0; w_ < HW; ++w_) { vg += red[w_][i / 32][i % 32]; vu += red[w_ + HW][i / 32][i % 32]; }
            pg[i] = vg; pu[i] = vu;
        }
        __syncthreads();
    }
    cl.sync();
    if (rank == 0)
        for (int i = threadIdx.x; i < SPW * 4 * 32; i += blockDim.x)
        {
            int row, m; cl_index<SPW>(i, row, m);
            if (m >= M) continue;
            row += g * SPW * 16;
            float gv = 0.f, uv = 0.f;
            for (int r = 0; r < CS; ++r) { gv += *cl.map_shared_rank(&pg[i], r); uv += *cl.map_shared_rank(&pu[i], r); }
            gv *= sg[row]; uv *= su[row];
            act[(size_t) m * N + row] = __float2half_rn(gv / (1.f + __expf(-gv)) * uv);
        }
    cl.sync();
}

// hecate (I8_CLUSTER): cluster-reduced int8 GEMV. A cluster of CS blocks owns one 16*SPW-row output group; block r of
// the cluster streams the r-th K slice (its 8 warps take contiguous unit ranges of the slice), reduces across its warps
// in shared memory, and with the rank-4 correction also reduces its slice of t = x v. The leader (rank 0) sums the
// cluster's partials through distributed shared memory and writes y = scale * sum + flag(m) * us t[m]. No scratch,
// atomics, fences or counters: graph-safe, one pass.
template <int SPW, int PD, typename OUT, bool LR>
__global__ void __launch_bounds__(WPB * 32) dense_i8_cl_kernel
(
    const uint4* __restrict__ w,
    const float* __restrict__ scale,
    const half* __restrict__ x,
    OUT* __restrict__ y,
    int M, int N, int K,
    LRArgs lr,
    int lead_lr, int lead_plain            // the leader's K share in eighths of a slice (it stages the epilogue meanwhile)
)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int CS = (int) cl.num_blocks(), rank = (int) cl.block_rank();
    int g = blockIdx.x / CS;
    const int kb_n = K / 32, lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    const int groups = N / (16 * SPW);
    if constexpr (LR)
    {
        const int tcl = lr.nt / CS;              // the first tcl clusters compute t slices (scheduled first)
        if (g < tcl) { lr_t_block(x, lr.v, lr.t, M, K, lr.nt, blockIdx.x); return; }
        g -= tcl;
    }
    bool lr_need = false;
    if constexpr (LR)
    {
        const int ni = (int) lr.modes[8];
        for (int i = 0; i < ni && i < 8; ++i) lr_need |= lr.modes[i] != 0.f;
    }
    const int lead = CS > 1 ? (lr_need ? lead_lr : lead_plain) : 8;
    auto cum = [&](int r) { return r == 0 ? 0 : lead + 8 * (r - 1); };      // in eighths
    const int W8 = cum(CS);
    const int s0 = (int) ((int64_t) kb_n * cum(rank) / W8), s1 = (int) ((int64_t) kb_n * cum(rank + 1) / W8);
    const int a = s0 + (s1 - s0) * wid / WPB, b = s0 + (s1 - s0) * (wid + 1) / WPB, n = b - a;
    __shared__ float red[WPB][SPW * 4][33];
    __shared__ float part[SPW * 4 * 32];
    __shared__ float4 ft[8];
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
    if (n > 0)
    {
        const uint4* pf = w + ((size_t) g * kb_n + a) * 32 * SPW + lane;
        const uint4* const pend = pf + (size_t) n * 32 * SPW;
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
        int kb = a;
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
                        uint32_t aa[4];
                        dec_i8(ww[2 * t], aa); dec_i8(ww[2 * t + 1], aa + 2);
                        mma16816(c[s], aa, bb[2 * t], bb[2 * t + 1]);
                    }
                    if (more) ring[j][s] = __ldg(pf + s * 32);
                }
                pf += 32 * SPW;
                ++kb;
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < SPW; ++s)
        #pragma unroll
        for (int e = 0; e < 4; ++e) red[wid][s * 4 + e][lane] = c[s][e];
    __syncthreads();
    for (int i = threadIdx.x; i < SPW * 4 * 32; i += blockDim.x)
    {
        float v_ = 0.f;
        #pragma unroll
        for (int w_ = 0; w_ < WPB; ++w_) v_ += red[w_][i / 32][i % 32];
        part[i] = v_;
    }
    // the leader stages the epilogue's operands (row scales, correction us rows, flag * t) before the cluster barrier, so
    // their latency overlaps the wait for the other slices
    constexpr int PER = (SPW * 4 * 32 + WPB * 32 - 1) / (WPB * 32);
    float sc[PER]; float ex[PER];
    if (rank == 0)
    {
        if constexpr (LR) if (lr_need)
        {
            if (threadIdx.x == 0) lr_wait_ready(lr.t, lr.nt);
            __syncthreads();
            if (threadIdx.x < M) ft[threadIdx.x] = lr_trow(lr, threadIdx.x, M);
            __syncthreads();
        }
        #pragma unroll
        for (int u = 0; u < PER; ++u)
        {
            const int i = threadIdx.x + u * WPB * 32;
            sc[u] = 0.f; ex[u] = 0.f;
            if (i < SPW * 4 * 32)
            {
                const int q = i / 32, l = i % 32, s = q / 4, e = q % 4;
                const int row = g * SPW * 16 + s * 16 + l / 4 + (e >= 2 ? 8 : 0), m = (l % 4) * 2 + (e & 1);
                if (m < M)
                {
                    sc[u] = scale[row];
                    if constexpr (LR) if (lr_need) ex[u] = lr_dot(lr_us(lr, row), ft[m]);
                }
            }
        }
    }
    cl.sync();
    if (rank == 0)
    {
        #pragma unroll
        for (int u = 0; u < PER; ++u)
        {
            const int i = threadIdx.x + u * WPB * 32;
            if (i >= SPW * 4 * 32) continue;
            const int q = i / 32, l = i % 32, s = q / 4, e = q % 4;
            const int row = g * SPW * 16 + s * 16 + l / 4 + (e >= 2 ? 8 : 0), m = (l % 4) * 2 + (e & 1);
            if (m >= M) continue;
            float v_ = 0.f;
            for (int r = 0; r < CS; ++r) v_ += *cl.map_shared_rank(&part[i], r);
            y[(size_t) m * N + row] = cvt_out<OUT>(v_ * sc[u] + ex[u]);
        }
        if constexpr (LR) { __syncthreads(); if (threadIdx.x < 32) lr_group_done(lr.t, threadIdx.x, groups); }
    }
    cl.sync();
}

template <int SPW, int PD, typename OUT, bool LR>
__global__ void __launch_bounds__(WPB * 32) dense_i8_kernel
(
    const uint4* __restrict__ w,
    const float* __restrict__ scale,
    const half* __restrict__ x,
    OUT* __restrict__ y,
    float* y32,
    int* ctr,
    int M, int N, int K, int upw,
    LRArgs lr
)
{
    int bid = blockIdx.x;
    bool lr_need = false;
    __shared__ float4 lr_tsm[LR ? WPB : 1][8];
    bool lr_have = false;                       // warp-uniform: t fetched
    float4 lr_t0 = make_float4(0.f, 0.f, 0.f, 0.f), lr_t1 = lr_t0;
    if constexpr (LR)
    {
        if (bid < lr.nt) { lr_t_block(x, lr.v, lr.t, M, K, lr.nt, bid); return; }
        bid -= lr.nt;
        const int ni = (int) lr.modes[8];
        for (int i = 0; i < ni && i < 8; ++i) lr_need |= lr.modes[i] != 0.f;
    }
    const int kb_n = K / 32, groups = N / (16 * SPW), lane = threadIdx.x % 32;
    const int total = groups * kb_n, gw = bid * WPB + threadIdx.x / 32;
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
                    float e[SPW][4];
                    #pragma unroll
                    for (int s = 0; s < SPW; ++s) e[s][0] = e[s][1] = e[s][2] = e[s][3] = 0.f;
                    if constexpr (LR) if (lr_need)
                    {
                        // all correction loads issued before any store (stores could alias them otherwise)
                        uint2 ua[SPW], ub[SPW];
                        #pragma unroll
                        for (int s = 0; s < SPW; ++s) { ua[s] = lr_us(lr, rb + s * 16); ub[s] = lr_us(lr, rb + s * 16 + 8); }
                        if (!lr_have) { lr_fetch(lr, lane, col, M, lr_tsm, lr_t0, lr_t1); lr_have = true; }
                        #pragma unroll
                        for (int s = 0; s < SPW; ++s)
                        {
                            e[s][0] = lr_dot(ua[s], lr_t0); e[s][2] = lr_dot(ub[s], lr_t0);
                            e[s][1] = lr_dot(ua[s], lr_t1); e[s][3] = lr_dot(ub[s], lr_t1);
                        }
                    }
                    #pragma unroll
                    for (int s = 0; s < SPW; ++s)
                    {
                        const int r0 = rb + s * 16;
                        const float s0 = scale[r0], s1 = scale[r0 + 8];
                        if (col < M)     { y[(size_t) col * N + r0] = cvt_out<OUT>(c[s][0] * s0 + e[s][0]);       y[(size_t) col * N + r0 + 8] = cvt_out<OUT>(c[s][2] * s1 + e[s][2]); }
                        if (col + 1 < M) { y[(size_t) (col + 1) * N + r0] = cvt_out<OUT>(c[s][1] * s0 + e[s][1]); y[(size_t) (col + 1) * N + r0 + 8] = cvt_out<OUT>(c[s][3] * s1 + e[s][3]); }
                    }
                    if constexpr (LR) lr_group_done(lr.t, lane, groups);
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
                        if constexpr (LR) if (lr_need && !lr_have)
                        {
                            // the M flagged t rows, staged per warp in shared memory (indexed by row below)
                            lr_fetch(lr, lane, col, M, lr_tsm, lr_t0, lr_t1); lr_have = true;
                        }
                        // loads of 4 elements first, then their stores (stores could alias the loads otherwise)
                        const int tot = SPW * 16 * M;
                        for (int i0 = lane; i0 < tot; i0 += 128)
                        {
                            float yv[4], sv[4], ev[4];
                            #pragma unroll
                            for (int u = 0; u < 4; ++u)
                            {
                                const int i = i0 + 32 * u;
                                yv[u] = 0.f; sv[u] = 0.f; ev[u] = 0.f;
                                if (i < tot)
                                {
                                    const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                                    yv[u] = __ldcg(y32 + (size_t) m * N + r); sv[u] = scale[r];
                                    if constexpr (LR) if (lr_need) ev[u] = lr_dot(lr_us(lr, r), lr_tsm[threadIdx.x / 32][m]);
                                }
                            }
                            #pragma unroll
                            for (int u = 0; u < 4; ++u)
                            {
                                const int i = i0 + 32 * u;
                                if (i < tot)
                                {
                                    const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                                    y[(size_t) m * N + r] = cvt_out<OUT>(yv[u] * sv[u] + ev[u]);
                                    y32[(size_t) m * N + r] = 0.f;
                                }
                            }
                        }
                        if (lane == 0) ctr[gr] = 0;
                        if constexpr (LR) lr_group_done(lr.t, lane, groups);
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

// hecate: block-split variant for shapes with few output groups and long K (o_proj, shared down, q_b): block b takes
// group b / S and the b % S-th slice of its k-blocks; the 8 warps interleave the slice's k-blocks, reduce in shared
// memory, and warp 0 writes y (S == 1) or adds the block's partial into the fp32 scratch once (S > 1; the last block of
// the group writes y and cleans up): ~8x fewer scratch atomics than the per-warp stream-K split. Same optional rank-4
// correction protocol as dense_i8_kernel (t blocks first; writers wait once).
template <int SPW, int PD, typename OUT, bool LR>
__global__ void __launch_bounds__(WPB * 32) dense_i8_blk_kernel
(
    const uint4* __restrict__ w,
    const float* __restrict__ scale,
    const half* __restrict__ x,
    OUT* __restrict__ y,
    float* y32,
    int* ctr,
    int M, int N, int K, int S,
    LRArgs lr
)
{
    __shared__ float red[WPB][SPW * 4][33];
    __shared__ int last_s;
    __shared__ float4 lr_tsm[LR ? WPB : 1][8];
    int bid = blockIdx.x;
    bool lr_need = false;
    if constexpr (LR)
    {
        if (bid < lr.nt) { lr_t_block(x, lr.v, lr.t, M, K, lr.nt, bid); return; }
        bid -= lr.nt;
        const int ni = (int) lr.modes[8];
        for (int i = 0; i < ni && i < 8; ++i) lr_need |= lr.modes[i] != 0.f;
    }
    const int kb_n = K / 32, groups = N / (16 * SPW), lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    const int gr = bid / S, part = bid % S;
    const int kc = (kb_n + S - 1) / S, kb0 = part * kc, kb1 = min(kb_n, kb0 + kc);
    const int nk = kb0 + wid < kb1 ? (kb1 - kb0 - wid + WPB - 1) / WPB : 0;
    const uint4* gbase = w + ((size_t) gr * kb_n) * 32 * SPW + lane;
    uint4 ring[PD][SPW];
    #pragma unroll
    for (int i = 0; i < PD; ++i)
    {
        const uint4* q = gbase + (size_t) (kb0 + wid + i * WPB) * 32 * SPW;
        #pragma unroll
        for (int s = 0; s < SPW; ++s) ring[i][s] = i < nk ? __ldg(q + s * 32) : make_uint4(0, 0, 0, 0);
    }
    const int xn = lane / 4, kq = (lane % 4) * 2;
    const bool live = xn < M;
    const uint32_t* xr = (const uint32_t*) (x + (size_t) (live ? xn : 0) * K + kq);
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
    for (int i0 = 0; i0 < nk; i0 += PD)
    {
        #pragma unroll
        for (int j = 0; j < PD; ++j)
        {
            const int i = i0 + j;
            if (i >= nk) break;
            const int kb = kb0 + wid + i * WPB;
            const bool more = i + PD < nk;
            const uint4* q = gbase + (size_t) (kb + PD * WPB) * 32 * SPW;
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
                if (more) ring[j][s] = __ldg(q + s * 32);
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < SPW; ++s)
        #pragma unroll
        for (int j = 0; j < 4; ++j) red[wid][s * 4 + j][lane] = c[s][j];
    __syncthreads();
    if (wid != 0) return;
    #pragma unroll
    for (int s = 0; s < SPW; ++s)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float a = 0.f;
            #pragma unroll
            for (int v = 0; v < WPB; ++v) a += red[v][s * 4 + j][lane];
            c[s][j] = a;
        }
    const int col = (lane % 4) * 2, rb = gr * SPW * 16 + lane / 4;
    float4 t0 = make_float4(0.f, 0.f, 0.f, 0.f), t1 = t0;
    if (S == 1)
    {
        float e[SPW][4];
        #pragma unroll
        for (int s = 0; s < SPW; ++s) e[s][0] = e[s][1] = e[s][2] = e[s][3] = 0.f;
        if constexpr (LR) if (lr_need)
        {
            uint2 ua[SPW], ub[SPW];
            #pragma unroll
            for (int s = 0; s < SPW; ++s) { ua[s] = lr_us(lr, rb + s * 16); ub[s] = lr_us(lr, rb + s * 16 + 8); }
            lr_fetch(lr, lane, col, M, lr_tsm, t0, t1);
            #pragma unroll
            for (int s = 0; s < SPW; ++s)
            {
                e[s][0] = lr_dot(ua[s], t0); e[s][2] = lr_dot(ub[s], t0);
                e[s][1] = lr_dot(ua[s], t1); e[s][3] = lr_dot(ub[s], t1);
            }
        }
        #pragma unroll
        for (int s = 0; s < SPW; ++s)
        {
            const int r0 = rb + s * 16;
            const float s0 = scale[r0], s1 = scale[r0 + 8];
            if (col < M)     { y[(size_t) col * N + r0] = cvt_out<OUT>(c[s][0] * s0 + e[s][0]);       y[(size_t) col * N + r0 + 8] = cvt_out<OUT>(c[s][2] * s1 + e[s][2]); }
            if (col + 1 < M) { y[(size_t) (col + 1) * N + r0] = cvt_out<OUT>(c[s][1] * s0 + e[s][1]); y[(size_t) (col + 1) * N + r0 + 8] = cvt_out<OUT>(c[s][3] * s1 + e[s][3]); }
        }
        if constexpr (LR) lr_group_done(lr.t, lane, groups);
        return;
    }
    #pragma unroll
    for (int s = 0; s < SPW; ++s)
    {
        const int r0 = rb + s * 16;
        if (col < M)     { atomicAdd(y32 + (size_t) col * N + r0, c[s][0]);       atomicAdd(y32 + (size_t) col * N + r0 + 8, c[s][2]); }
        if (col + 1 < M) { atomicAdd(y32 + (size_t) (col + 1) * N + r0, c[s][1]); atomicAdd(y32 + (size_t) (col + 1) * N + r0 + 8, c[s][3]); }
    }
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) last = atomicAdd(ctr + gr, 1) == S - 1;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) return;
    __threadfence();
    if constexpr (LR) if (lr_need) lr_fetch(lr, lane, col, M, lr_tsm, t0, t1);
    const int tot = SPW * 16 * M;
    for (int i0 = lane; i0 < tot; i0 += 128)
    {
        float yv[4], sv[4], ev[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            const int i = i0 + 32 * u;
            yv[u] = 0.f; sv[u] = 0.f; ev[u] = 0.f;
            if (i < tot)
            {
                const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                yv[u] = __ldcg(y32 + (size_t) m * N + r); sv[u] = scale[r];
                if constexpr (LR) if (lr_need) ev[u] = lr_dot(lr_us(lr, r), lr_tsm[0][m]);
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            const int i = i0 + 32 * u;
            if (i < tot)
            {
                const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                y[(size_t) m * N + r] = cvt_out<OUT>(yv[u] * sv[u] + ev[u]);
                y32[(size_t) m * N + r] = 0.f;
            }
        }
    }
    if (lane == 0) ctr[gr] = 0;
    if constexpr (LR) lr_group_done(lr.t, lane, groups);
}

// hecate: gate + up GEMVs of a gated MLP in one stream-K pass over both matrices (shared x fragments), partials into
// the two linears' own fp32 scratch (atomics), and the last warp to finish a 16*SPW-row group across BOTH matrices
// writes act = silu(g * sg) * (u * su) (fp16) and cleans the scratch / counter (gate's counter): replaces two GEMV
// launches and the activation kernel. Units: [gate: G groups x KB k-blocks][up: same]; act [M][N] row-major.
template <int SPW, int PD>
__global__ void __launch_bounds__(WPB * 32) dense_i8_gu_act_kernel
(
    const uint4* __restrict__ wg, const float* __restrict__ sg,
    const uint4* __restrict__ wu, const float* __restrict__ su,
    const half* __restrict__ x, half* __restrict__ act,
    float* y32g, float* y32u, int* ctr,
    int M, int N, int K, int upw
)
{
    const int kb_n = K / 32, groups = N / (16 * SPW), lane = threadIdx.x % 32;
    const int per = groups * kb_n, total = 2 * per, gw = blockIdx.x * WPB + threadIdx.x / 32;
    const int u0 = gw * upw, n = min(total - u0, upw);
    if (n <= 0) return;
    auto wp = [&](int u) { return (u < per ? wg + (size_t) u * 32 * SPW : wu + (size_t) (u - per) * 32 * SPW) + lane; };
    int mat = u0 / per, kb = (u0 % per) % kb_n, gr = (u0 % per) / kb_n;
    int pu = u0; const int pend = u0 + n;
    uint4 ring[PD][SPW];
    #pragma unroll
    for (int i = 0; i < PD; ++i)
    {
        const uint4* q = wp(pu);
        #pragma unroll
        for (int s = 0; s < SPW; ++s) ring[i][s] = pu < pend ? __ldg(q + s * 32) : make_uint4(0, 0, 0, 0);
        ++pu;
    }
    const int xn = lane / 4, kq = (lane % 4) * 2;
    const bool live = xn < M;
    const uint32_t* xr = (const uint32_t*) (x + (size_t) (live ? xn : 0) * K + kq);
    float c[SPW][4];
    #pragma unroll
    for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;

    for (int done = 0; done < n; done += PD)
    {
        #pragma unroll
        for (int j = 0; j < PD; ++j)
        {
            if (done + j >= n) break;
            const bool more = pu < pend;
            const uint4* q = wp(pu);
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
                if (more) ring[j][s] = __ldg(q + s * 32);
            }
            ++pu;
            ++kb;
            if (kb == kb_n || done + j == n - 1)
            {
                const int col = (lane % 4) * 2, rb = gr * SPW * 16 + lane / 4;
                float* y32 = mat ? y32u : y32g;
                #pragma unroll
                for (int s = 0; s < SPW; ++s)
                {
                    const int r0 = rb + s * 16;
                    if (col < M)     { atomicAdd(y32 + (size_t) col * N + r0, c[s][0]);       atomicAdd(y32 + (size_t) col * N + r0 + 8, c[s][2]); }
                    if (col + 1 < M) { atomicAdd(y32 + (size_t) (col + 1) * N + r0, c[s][1]); atomicAdd(y32 + (size_t) (col + 1) * N + r0 + 8, c[s][3]); }
                }
                __threadfence();
                __syncwarp();
                // warps touching this group's units in the gate and in the up matrix
                const int ga = gr * kb_n, gb = ga + kb_n - 1;
                const int need = (gb / upw - ga / upw + 1) + ((gb + per) / upw - (ga + per) / upw + 1);
                int last = 0;
                if (lane == 0) last = atomicAdd(ctr + gr, 1) == need - 1;
                last = __shfl_sync(0xffffffffu, last, 0);
                if (last)
                {
                    __threadfence();
                    const int tot = SPW * 16 * M;
                    for (int i0 = lane; i0 < tot; i0 += 128)
                    {
                        float gv[4], uv[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u)
                        {
                            const int i = i0 + 32 * u;
                            gv[u] = 0.f; uv[u] = 0.f;
                            if (i < tot)
                            {
                                const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                                gv[u] = __ldcg(y32g + (size_t) m * N + r) * sg[r];
                                uv[u] = __ldcg(y32u + (size_t) m * N + r) * su[r];
                            }
                        }
                        #pragma unroll
                        for (int u = 0; u < 4; ++u)
                        {
                            const int i = i0 + 32 * u;
                            if (i < tot)
                            {
                                const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                                act[(size_t) m * N + r] = __float2half(gv[u] / (1.f + __expf(-gv[u])) * uv[u]);
                                y32g[(size_t) m * N + r] = 0.f; y32u[(size_t) m * N + r] = 0.f;
                            }
                        }
                    }
                    if (lane == 0) ctr[gr] = 0;
                }
                #pragma unroll
                for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
                if (kb == kb_n)
                {
                    kb = 0;
                    if (++gr == groups) { gr = 0; ++mat; }
                }
            }
        }
    }
}

// hecate: two int8 GEMVs on the same x (e.g. MLA q_a and kv_a) in one stream-K pass: units [A: GA groups x KB][B: GB x
// KB], each segment with its own weights, scales, fp16 output (row stride = its N), scratch and counters; same epilogue as
// dense_i8_kernel (whole group -> direct write, else fp32 atomics + last warp). Saves a launch and the second tail.
template <int SPW, int PD>
__global__ void __launch_bounds__(WPB * 32) dense_i8_dual_kernel
(
    const uint4* __restrict__ wa, const float* __restrict__ sa, half* __restrict__ ya, float* y32a, int* ctra, int Na,
    const uint4* __restrict__ wb, const float* __restrict__ sb, half* __restrict__ yb, float* y32b, int* ctrb, int Nb,
    const half* __restrict__ x, int M, int K, int upw
)
{
    const int kb_n = K / 32, ga = Na / (16 * SPW), gbn = Nb / (16 * SPW), lane = threadIdx.x % 32;
    const int per = ga * kb_n, total = per + gbn * kb_n, gw = blockIdx.x * WPB + threadIdx.x / 32;
    const int u0 = gw * upw, n = min(total - u0, upw);
    if (n <= 0) return;
    auto wp = [&](int u) { return (u < per ? wa + (size_t) u * 32 * SPW : wb + (size_t) (u - per) * 32 * SPW) + lane; };
    int seg = u0 >= per, uu = seg ? u0 - per : u0, kb = uu % kb_n, gr = uu / kb_n;
    int pu = u0; const int pend = u0 + n;
    uint4 ring[PD][SPW];
    #pragma unroll
    for (int i = 0; i < PD; ++i)
    {
        const uint4* q = wp(pu);
        #pragma unroll
        for (int s = 0; s < SPW; ++s) ring[i][s] = pu < pend ? __ldg(q + s * 32) : make_uint4(0, 0, 0, 0);
        ++pu;
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
            const bool more = pu < pend;
            const uint4* q = wp(pu);
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
                if (more) ring[j][s] = __ldg(q + s * 32);
            }
            ++pu;
            ++kb;
            if (kb == kb_n || done + j == n - 1)
            {
                const int N = seg ? Nb : Na;
                const float* scale = seg ? sb : sa;
                half* y = seg ? yb : ya;
                float* y32 = seg ? y32b : y32a;
                int* ctr = seg ? ctrb : ctra;
                const int col = (lane % 4) * 2, rb = gr * SPW * 16 + lane / 4;
                if (whole && kb == kb_n)
                {
                    #pragma unroll
                    for (int s = 0; s < SPW; ++s)
                    {
                        const int r0 = rb + s * 16;
                        const float s0 = scale[r0], s1 = scale[r0 + 8];
                        if (col < M)     { y[(size_t) col * N + r0] = __float2half_rn(c[s][0] * s0);       y[(size_t) col * N + r0 + 8] = __float2half_rn(c[s][2] * s1); }
                        if (col + 1 < M) { y[(size_t) (col + 1) * N + r0] = __float2half_rn(c[s][1] * s0); y[(size_t) (col + 1) * N + r0 + 8] = __float2half_rn(c[s][3] * s1); }
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
                    const int base = seg ? per : 0;
                    const int ua = base + gr * kb_n, ub = ua + kb_n - 1;
                    const int nsplit = ub / upw - ua / upw + 1;
                    int last = 0;
                    if (lane == 0) last = atomicAdd(ctr + gr, 1) == nsplit - 1;
                    last = __shfl_sync(0xffffffffu, last, 0);
                    if (last)
                    {
                        __threadfence();
                        for (int i = lane; i < SPW * 16 * M; i += 32)
                        {
                            const int r = gr * SPW * 16 + i % (SPW * 16), m = i / (SPW * 16);
                            float* pp = y32 + (size_t) m * N + r;
                            y[(size_t) m * N + r] = __float2half_rn(__ldcg(pp) * scale[r]);
                            *pp = 0.f;
                        }
                        if (lane == 0) ctr[gr] = 0;
                    }
                }
                #pragma unroll
                for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
                whole = true;
                if (kb == kb_n)
                {
                    kb = 0;
                    if (++gr == (seg ? gbn : ga)) { gr = 0; seg = 1; }
                }
            }
        }
    }
}

template <int SPW, int PD, typename OUT>
void launch(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
            at::Tensor& ctr, int M, int N, int K, cudaStream_t stream, Graph* graph, const DenseI8LR* lrp)
{
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int total = N / (16 * SPW) * (K / 32), warps = sms * 2 * WPB;
    const int upw = (total + warps - 1) / warps;
    const int blocks = (total + upw * WPB - 1) / (upw * WPB);
    const uint4* wp = (const uint4*) w8.data_ptr();
    const float* sp = (const float*) scale.data_ptr();
    const half* xp = (const half*) x.data_ptr();
    OUT* yp = (OUT*) y.data_ptr();
    float* y32p = (float*) y32.data_ptr();
    int* ctrp = (int*) ctr.data_ptr();
    void* kfn;
    static const int blk_mode = [] { const char* e = getenv("EXL3_I8_BLK"); return e ? atoi(e) : 1; }();
    static const int blk_mink = [] { const char* e = getenv("EXL3_I8_BLK_MINK"); return e ? atoi(e) : 8192; }();   // long K only (o_proj)
    static const int blk_bps = [] { const char* e = getenv("EXL3_I8_BLK_BPS"); return e ? atoi(e) : 2; }();
    const int groups = N / (16 * SPW), kb_n = K / 32;
#ifdef I8_CLUSTER
    if (i8_cl_on())
    {
        static const int cl_bps = [] { const char* e = getenv("EXL3_I8_CL_BPS"); return e ? atoi(e) : 2; }();
        int CS = (sms * cl_bps + groups / 2) / groups;
        CS = std::max(1, std::min(8, std::min(CS, kb_n / WPB)));
        cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.gridDim = dim3(groups * CS); cfg.blockDim = dim3(WPB * 32); cfg.dynamicSmemBytes = 0; cfg.stream = stream; cfg.attrs = at; cfg.numAttrs = 1;
        LRArgs lra{nullptr, nullptr, nullptr, nullptr, 0};
        if (lrp)
        {
            int tcl = (std::max(1, std::min(LR_MAX_NT, K / 1024)) + CS - 1) / CS;        // t clusters
            while (tcl * CS > LR_MAX_NT) --tcl;
            tcl = std::max(1, tcl);
            lra = LRArgs{(const half*) lrp->us.data_ptr(), (const half*) lrp->v.data_ptr(), (float*) lrp->t.data_ptr(), (const float*) lrp->modes.data_ptr(), tcl * CS};
            cfg.gridDim = dim3((groups + tcl) * CS);
        }
        static const int lead_lr = [] { const char* e = getenv("EXL3_I8_CL_LEAD_LR"); return e ? atoi(e) : 6; }();
        static const int lead_pl = [] { const char* e = getenv("EXL3_I8_CL_LEAD"); return e ? atoi(e) : 8; }();
        if (lrp) { auto kernel = dense_i8_cl_kernel<SPW, PD, OUT, true>; cuda_check(cudaLaunchKernelEx(&cfg, kernel, wp, sp, xp, yp, M, N, K, lra, lead_lr, lead_pl)); kfn = (void*) kernel; }
        else { auto kernel = dense_i8_cl_kernel<SPW, PD, OUT, false>; cuda_check(cudaLaunchKernelEx(&cfg, kernel, wp, sp, xp, yp, M, N, K, lra, lead_lr, lead_pl)); kfn = (void*) kernel; }
        if (graph)
        {
            graph->record_param(kfn, GP_gemm_A, 2);
            graph->record_param(kfn, GP_gemm_C, 3);
            graph->record_param(kfn, GP_end, 0);
        }
        return;
    }
#endif
    if (blk_mode && groups < sms && kb_n >= 2 * WPB && K >= blk_mink)
    {
        int S = std::max(1, (sms * blk_bps + groups - 1) / groups);
        S = std::min(S, kb_n / WPB);
        if (lrp)
        {
            const int nt = std::max(1, std::min(LR_MAX_NT, K / 1024));
            LRArgs lr{(const half*) lrp->us.data_ptr(), (const half*) lrp->v.data_ptr(), (float*) lrp->t.data_ptr(),
                      (const float*) lrp->modes.data_ptr(), nt};
            auto kernel = dense_i8_blk_kernel<SPW, PD, OUT, true>;
            kernel<<<groups * S + nt, WPB * 32, 0, stream>>>(wp, sp, xp, yp, y32p, ctrp, M, N, K, S, lr);
            kfn = (void*) kernel;
        }
        else
        {
            auto kernel = dense_i8_blk_kernel<SPW, PD, OUT, false>;
            kernel<<<groups * S, WPB * 32, 0, stream>>>(wp, sp, xp, yp, y32p, ctrp, M, N, K, S, LRArgs{nullptr, nullptr, nullptr, nullptr, 0});
            kfn = (void*) kernel;
        }
    }
    else if (lrp)
    {
        const int nt = std::max(1, std::min(LR_MAX_NT, K / 1024));
        LRArgs lr{(const half*) lrp->us.data_ptr(), (const half*) lrp->v.data_ptr(), (float*) lrp->t.data_ptr(),
                  (const float*) lrp->modes.data_ptr(), nt};
        auto kernel = dense_i8_kernel<SPW, PD, OUT, true>;
        kernel<<<blocks + nt, WPB * 32, 0, stream>>>(wp, sp, xp, yp, y32p, ctrp, M, N, K, upw, lr);
        kfn = (void*) kernel;
    }
    else
    {
        auto kernel = dense_i8_kernel<SPW, PD, OUT, false>;
        kernel<<<blocks, WPB * 32, 0, stream>>>(wp, sp, xp, yp, y32p, ctrp, M, N, K, upw, LRArgs{nullptr, nullptr, nullptr, nullptr, 0});
        kfn = (void*) kernel;
    }
    if (graph)
    {
        graph->record_param(kfn, GP_gemm_A, 2);
        graph->record_param(kfn, GP_gemm_C, 3);
        graph->record_param(kfn, GP_end, 0);
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
    Graph* graph,
    const DenseI8LR* lr
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
    if (lr)
    {
        TORCH_CHECK(lr->us.size(0) == N && lr->us.size(1) == 4 && lr->v.size(0) == K && lr->v.size(1) == 4 &&
                    lr->us.dtype() == at::kHalf && lr->v.dtype() == at::kHalf && lr->t.numel() >= LR_LAUNCH + 1 &&
                    lr->t.dtype() == at::kFloat && lr->modes.numel() >= 9 && lr->modes.dtype() == at::kFloat &&
                    K % 2 == 0, "dense_i8: low-rank correction shape/dtype mismatch");
    }
    TORCH_CHECK(f32 || y.dtype() == at::kHalf, "dense_i8: y must be fp16 or fp32");
    if (spw == 4) { if (f32) launch<4, 2, float>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph, lr); else launch<4, 2, half>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph, lr); }
    else if (spw == 2) { if (f32) launch<2, 4, float>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph, lr); else launch<2, 4, half>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph, lr); }
    else { if (f32) launch<1, 8, float>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph, lr); else launch<1, 8, half>(x, w8, scale, y, y32, ctr, M, N, K, stream, graph, lr); }
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

void dense_i8_gu_act_gr
(
    const at::Tensor& x,
    const at::Tensor& wg, const at::Tensor& sg, at::Tensor& y32g, at::Tensor& ctr,
    const at::Tensor& wu, const at::Tensor& su, at::Tensor& y32u,
    at::Tensor& act,
    int spw,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    const int K = (int) x.size(-1), M = (int) (x.numel() / K), N = (int) act.size(-1);
    TORCH_CHECK(M >= 1 && M <= 8 && spw == 4 && N % 64 == 0 && K % 32 == 0, "dense_i8_gu_act: shape");
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && act.dtype() == at::kHalf, "dense_i8_gu_act: dtypes");
    TORCH_CHECK(y32g.numel() >= (int64_t) 8 * N && y32u.numel() >= (int64_t) 8 * N && ctr.numel() >= N / 64, "dense_i8_gu_act: scratch");
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int total = 2 * (N / 64) * (K / 32), warps = sms * 2 * WPB;
    const int upw = (total + warps - 1) / warps;
    const int blocks = (total + upw * WPB - 1) / (upw * WPB);
#ifdef I8_CLUSTER
    if (i8_cl_on())
    {
        static const int cl_bps = [] { const char* e = getenv("EXL3_I8_CL_BPS"); return e ? atoi(e) : 2; }();
        const int groups = N / 64, kb_n = K / 32;
        int CS = std::max(1, std::min(8, std::min((sms * cl_bps + groups / 2) / groups, kb_n / WPB)));
        cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.gridDim = dim3(groups * CS); cfg.blockDim = dim3(WPB * 32); cfg.stream = stream; cfg.attrs = at; cfg.numAttrs = 1;
        auto kc = dense_i8_gu_cl_kernel<4, 2>;
        cuda_check(cudaLaunchKernelEx(&cfg, kc, (const uint4*) wg.data_ptr(), (const float*) sg.data_ptr(), (const uint4*) wu.data_ptr(),
            (const float*) su.data_ptr(), (const half*) x.data_ptr(), (half*) act.data_ptr(), M, N, K));
        if (graph)
        {
            graph->record_param((void*) kc, GP_gemm_A, 4);
            graph->record_param((void*) kc, GP_end, 0);
        }
        cuda_check(cudaPeekAtLastError());
        return;
    }
#endif
    auto kernel = dense_i8_gu_act_kernel<4, 2>;
    kernel<<<blocks, WPB * 32, 0, stream>>>
    (
        (const uint4*) wg.data_ptr(), (const float*) sg.data_ptr(), (const uint4*) wu.data_ptr(), (const float*) su.data_ptr(),
        (const half*) x.data_ptr(), (half*) act.data_ptr(), (float*) y32g.data_ptr(), (float*) y32u.data_ptr(),
        (int*) ctr.data_ptr(), M, N, K, upw
    );
    if (graph)
    {
        graph->record_param((void*) kernel, GP_gemm_A, 4);
        graph->record_param((void*) kernel, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}

void dense_i8_dual_gr
(
    const at::Tensor& x,
    const at::Tensor& wa, const at::Tensor& sa, at::Tensor& ya, at::Tensor& y32a, at::Tensor& ctra,
    const at::Tensor& wb, const at::Tensor& sb, at::Tensor& yb, at::Tensor& y32b, at::Tensor& ctrb,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    const int K = (int) x.size(-1), M = (int) (x.numel() / K), Na = (int) ya.size(-1), Nb = (int) yb.size(-1);
    TORCH_CHECK(M >= 1 && M <= 8 && K % 32 == 0 && Na % 64 == 0 && Nb % 64 == 0, "dense_i8_dual: shape");
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && ya.dtype() == at::kHalf && yb.dtype() == at::kHalf &&
                ya.is_contiguous() && yb.is_contiguous(), "dense_i8_dual: dtypes / layout");
    TORCH_CHECK(y32a.numel() >= (int64_t) 8 * Na && y32b.numel() >= (int64_t) 8 * Nb && ctra.numel() >= Na / 64 &&
                ctrb.numel() >= Nb / 64, "dense_i8_dual: scratch");
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int total = (Na / 64 + Nb / 64) * (K / 32), warps = sms * 2 * WPB;
    const int upw = (total + warps - 1) / warps;
    const int blocks = (total + upw * WPB - 1) / (upw * WPB);
#ifdef I8_CLUSTER
    if (i8_cl_on())
    {
        static const int cl_bps = [] { const char* e = getenv("EXL3_I8_CL_BPS"); return e ? atoi(e) : 2; }();
        const int groups = Na / 64 + Nb / 64, kb_n = K / 32;
        int CS = std::max(1, std::min(8, std::min((sms * cl_bps + groups / 2) / groups, kb_n / WPB)));
        cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.gridDim = dim3(groups * CS); cfg.blockDim = dim3(WPB * 32); cfg.stream = stream; cfg.attrs = at; cfg.numAttrs = 1;
        auto kc = dense_i8_dual_cl_kernel<4, 2>;
        cuda_check(cudaLaunchKernelEx(&cfg, kc, (const uint4*) wa.data_ptr(), (const float*) sa.data_ptr(), (half*) ya.data_ptr(), Na,
            (const uint4*) wb.data_ptr(), (const float*) sb.data_ptr(), (half*) yb.data_ptr(), Nb, (const half*) x.data_ptr(), M, K));
        if (graph)
        {
            graph->record_param((void*) kc, GP_gemm_A, 8);
            graph->record_param((void*) kc, GP_end, 0);
        }
        cuda_check(cudaPeekAtLastError());
        return;
    }
#endif
    auto kernel = dense_i8_dual_kernel<4, 2>;
    kernel<<<blocks, WPB * 32, 0, stream>>>
    (
        (const uint4*) wa.data_ptr(), (const float*) sa.data_ptr(), (half*) ya.data_ptr(), (float*) y32a.data_ptr(), (int*) ctra.data_ptr(), Na,
        (const uint4*) wb.data_ptr(), (const float*) sb.data_ptr(), (half*) yb.data_ptr(), (float*) y32b.data_ptr(), (int*) ctrb.data_ptr(), Nb,
        (const half*) x.data_ptr(), M, K, upw
    );
    if (graph)
    {
        graph->record_param((void*) kernel, GP_gemm_A, 12);
        graph->record_param((void*) kernel, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}

void dense_i8_dual
(
    const at::Tensor& x,
    const at::Tensor& wa, const at::Tensor& sa, at::Tensor& ya, at::Tensor& y32a, at::Tensor& ctra,
    const at::Tensor& wb, const at::Tensor& sb, at::Tensor& yb, at::Tensor& y32b, at::Tensor& ctrb
)
{
    dense_i8_dual_gr(x, wa, sa, ya, y32a, ctra, wb, sb, yb, y32b, ctrb, nullptr);
}

void dense_i8_gu_act
(
    const at::Tensor& x,
    const at::Tensor& wg, const at::Tensor& sg, at::Tensor& y32g, at::Tensor& ctr,
    const at::Tensor& wu, const at::Tensor& su, at::Tensor& y32u,
    at::Tensor& act
)
{
    dense_i8_gu_act_gr(x, wg, sg, y32g, ctr, wu, su, y32u, act, 4, nullptr);
}

void dense_i8_lr
(
    const at::Tensor& x,
    const at::Tensor& w8,
    const at::Tensor& scale,
    at::Tensor& y,
    at::Tensor& y32,
    at::Tensor& ctr,
    int64_t spw,
    const at::Tensor& us,
    const at::Tensor& v,
    const at::Tensor& t,
    const at::Tensor& modes
)
{
    DenseI8LR lr{us, v, t, modes};
    dense_i8_gr(x, w8, scale, y, y32, ctr, (int) spw, nullptr, &lr);
}
