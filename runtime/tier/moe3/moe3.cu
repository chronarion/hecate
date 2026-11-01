// Hecate trellis MoE (moe3): the RTX band's routed experts (3-bit EXL3 trellis, mul1 codebook) for the decode shape
// (R <= 8 rows), sm_120. Replaces ExLlamaV3's coop kernel pair with a bandwidth-first design:
//   rot  : per distinct band expert j and projection p (gate, up): xr[j][p][r] = H128(x[r] * suh) (fp16) and sum_k xr
//   A    : stream-K over (expert, projection, 64-column group, 64-k block) units of gate + up. A unit's trellis words
//          (4 k-tiles x 4 n-tiles x 96 B) arrive by cp.async in a per-warp shared-memory ring. The 8 weights a lane
//          decodes from a 16x16 tile are exactly an m16n8k16 A fragment of the transposed tile, so the weights are the
//          A operand (16 output columns) and the <= 8 token rows fill the n8 side: one HMMA per tile. mul1 decode
//          without the per-weight affine (h = 1024 + byte sum, an exact fp16), fp32 accumulation, flushed with RED
//          into an fp32 scratch at every (expert, projection, group) boundary.
//   act  : g = k_inv * sum(h x) + k_bias * sum(x) (likewise u), H128, * svh, silu(g) * u, * down suh, H128 -> the
//          rotated down input (fp16) and its row sums; clears the gate / up scratch
//   B    : stream-K down projection on the rotated activations, RED into a per-expert fp32 scratch
//   fin  : per output 128-chunk: sum over experts of w[j][r] * svh * H128(k_inv * acc + k_bias * sum(a)) -> out;
//          clears the down scratch
// A and B prefetch their first weight units before griddepcontrol.wait (programmatic dependent launch), so the weight
// stream starts while the previous kernel drains.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

constexpr int H = 6144, I = 2048, TOPK = 8, MAXR = 8, MAXJ = 64, WPB = 8, NT = WPB * 32;
constexpr int TW = 24;                                 // uint32 words per 16x16 3-bit tile
constexpr int UNIT_W = 4 * 4 * TW;                     // words per unit: 4 k-tiles x 4 n-tiles
#ifndef STAGES_A
#define STAGES_A 3
#endif
#ifndef STAGES_B
#define STAGES_B 3
#endif
constexpr float R_SCALE = 0.088388347648f;             // 1 / sqrt(128)

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
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
// the lane's 8 weights of a tile (a = word src_a, b = word src_b of the tile, s2 its shift) as the A fragment of the
// transposed tile: {(n g, k 2c..), (n g+8, k 2c..), (n g, k 2c+8..), (n g+8, k 2c+8..)}
__device__ __forceinline__ void dq8_a(uint32_t a, uint32_t b, int s2, uint32_t* af)
{
    const uint32_t w7 = fshift(b, a, s2), w6 = w7 >> 3, w5 = w6 >> 3, w4 = w5 >> 3;
    const uint32_t w3 = fshift(b, a, s2 + 12), w2 = w3 >> 3, w1 = w2 >> 3, w0 = w1 >> 3;
    af[0] = dec2(w0 & 0xffff, w1 & 0xffff);            // f0[0]: n g,     k 2c, 2c+1
    af[2] = dec2(w2 & 0xffff, w3 & 0xffff);            // f0[1]: n g,     k 2c+8, 2c+9
    af[1] = dec2(w4 & 0xffff, w5 & 0xffff);            // f1[0]: n g + 8, k 2c, 2c+1
    af[3] = dec2(w6 & 0xffff, w7 & 0xffff);            // f1[1]: n g + 8, k 2c+8, 2c+9
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
__device__ __forceinline__ float k_inv() { return __half2float(__ushort_as_half((unsigned short) 0x1eee)); }
__device__ __forceinline__ float k_bias() { return __half2float(__ushort_as_half((unsigned short) 0xc931)); }

struct Tabs { const int64_t* tr[3]; const int64_t* su[3]; const int64_t* sv[3]; };
struct Scr
{
    half* xr;        // [MAXJ][2][MAXR][H] rotated gate / up inputs
    float* xs;       // [MAXJ][2][MAXR] their sums
    float* gu;       // [MAXJ][2][MAXR][I] raw gate / up sums
    half* ar;        // [MAXJ][MAXR][I] rotated down input
    float* as;       // [MAXJ][MAXR] its sums
    float* ds;       // [MAXJ][MAXR][H] raw down sums
};

// distinct band experts of this step (sorted by local id) and the routing weight per (slot, row)
struct Route { int m; int e[MAXJ]; float w[MAXJ][MAXR]; unsigned mask[4]; short slot[128]; };
__device__ void route(const int64_t* sel, const half* rw, int R, int lo, int nb, Route& rt)
{
    const int t = threadIdx.x, np = R * TOPK;
    if (t < 4) rt.mask[t] = 0u;
    for (int i = t; i < MAXJ * MAXR; i += blockDim.x) (&rt.w[0][0])[i] = 0.f;
    int e = -1; float wv = 0.f;
    if (t < np) { e = (int) (sel[t] - lo); wv = __half2float(rw[t]); if (e < 0 || e >= nb) e = -1; }
    __syncthreads();
    if (e >= 0) atomicOr(&rt.mask[e >> 5], 1u << (e & 31));
    __syncthreads();
    if (t == 0)
    {
        int m = 0;
        for (int wd = 0; wd < 4; ++wd)
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

// ---------------------------------------------------------------------------------------------------------------------
// rot: grid (R * TOPK slots, 8 parts); warp job = (projection, row, 128-chunk)
__global__ void __launch_bounds__(NT) moe3_rot_kernel(const half* __restrict__ x, const int64_t* __restrict__ sel,
    const half* __restrict__ rw, int R, int lo, int nb, Tabs tb, Scr sc)
{
    __shared__ Route rt;
    pdl_wait();                                            // selection written / the previous call's fin done
    pdl_launch();
    route(sel, rw, R, lo, nb, rt);
    const int j = blockIdx.x;
    if (j == 0 && blockIdx.y == 0)
        for (int i = threadIdx.x; i < MAXJ * MAXR; i += NT) sc.as[i] = 0.f;
    if (j >= rt.m) return;
    const int le = rt.e[j], lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int jobs = 2 * R * (H / 128);
    for (int q = blockIdx.y * WPB + w; q < jobs; q += gridDim.y * WPB)
    {
        const int p = q / (R * (H / 128)), r = (q / (H / 128)) % R, c = q % (H / 128);
        const half* suh = (const half*) tb.su[p][le];
        const int col = c * 128 + lane * 4;
        const uint2 xv = *(const uint2*) (x + (size_t) r * H + col), sv = *(const uint2*) (suh + col);
        const float2 x01 = __half22float2(*(const half2*) &xv.x), x23 = __half22float2(*(const half2*) &xv.y);
        const float2 s01 = __half22float2(*(const half2*) &sv.x), s23 = __half22float2(*(const half2*) &sv.y);
        float v0 = x01.x * s01.x, v1 = x01.y * s01.y, v2 = x23.x * s23.x, v3 = x23.y * s23.y;
        had128(v0, v1, v2, v3, lane);
        const half2 o01 = __floats2half2_rn(v0, v1), o23 = __floats2half2_rn(v2, v3);
        uint2 ov; ov.x = *(const uint32_t*) &o01; ov.y = *(const uint32_t*) &o23;
        *(uint2*) (sc.xr + (((size_t) j * 2 + p) * MAXR + r) * H + col) = ov;
        const float2 f01 = __half22float2(o01), f23 = __half22float2(o23);
        const float s = warp_sum((f01.x + f01.y) + (f23.x + f23.y));
        if (lane == 0) atomicAdd(sc.xs + (j * 2 + p) * MAXR + r, s);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// stream-K GEMV phase over units (expert slot j, matrix p, group gr of 4 n-tiles, k-block kb of 4 k-tiles)
// MATS matrices per expert (A: gate, up; B: down), N output columns, K inputs
template <int MATS, int N, int K>
struct Phase
{
    static constexpr int G = N / 64, KBN = K / 64, UPM = G * KBN, UPE = MATS * UPM, NTN = N / 16;
};

template <int MATS, int N, int K, int STAGES, typename XPtr, typename Flush>
__device__ __forceinline__ void stream3(const Route& rt, const int64_t* const* trp, int R, XPtr xptr, Flush flush,
                                        uint32_t* ring, bool before_wait)
{
    using P = Phase<MATS, N, K>;
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * WPB + (threadIdx.x >> 5), nw = gridDim.x * WPB;
    const int U = rt.m * P::UPE;
    const int per = (U + nw - 1) / nw, u0 = min(U, gw * per), n = min(U, u0 + per) - u0;
    // lane's word pair in a tile (3-bit register decode of exl3_gemv: t_offset = lane * 8)
    const int b1 = (lane * 8 + 257) * 3, b2 = b1 + 21;
    const int src_a = ((b1 - 16) / 32) % TW, src_b = ((b2 - 1) / 32) % TW, s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    auto issue = [&](int i)                                // unit u0 + i into ring slot i % STAGES
    {
        if (i < n)
        {
            const int u = u0 + i, j = u / P::UPE, p = (u % P::UPE) / P::UPM, gr = (u % P::UPM) / P::KBN, kb = u % P::KBN;
            const uint32_t* base = (const uint32_t*) trp[p][rt.e[j]];
            uint32_t* dst = ring + (i % STAGES) * UNIT_W;
            #pragma unroll
            for (int c3 = 0; c3 < 3; ++c3)
            {
                const int c = lane + 32 * c3, q = c / 24, off = c % 24;      // 96 x 16 B: 4 k-tiles x 384 B
                cp16(su32(dst + q * 96 + off * 4), base + ((size_t) (kb * 4 + q) * P::NTN + gr * 4) * TW + off * 4);
            }
        }
        cp_commit();
    };
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) issue(i);
    if (before_wait) { pdl_wait(); pdl_launch(); }
    if (n <= 0) { cp_wait<0>(); return; }
    int u = u0, j = u / P::UPE, p = (u % P::UPE) / P::UPM, gr = (u % P::UPM) / P::KBN, kb = u % P::KBN;
    const int xn = lane >> 2, kq = (lane & 3) * 2;
    const bool live = xn < R;
    float c[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.f;
    for (int i = 0; i < n; ++i)
    {
        issue(i + STAGES - 1);
        cp_wait<STAGES - 1>();
        __syncwarp();
        const uint32_t* st = ring + (i % STAGES) * UNIT_W;
        const uint32_t* xr = (const uint32_t*) (xptr(j, p) + (size_t) (live ? xn : 0) * K + kq);
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const int k0 = (kb * 4 + q) * 16;
            const uint32_t xb0 = live ? __ldg(xr + k0 / 2) : 0u, xb1 = live ? __ldg(xr + k0 / 2 + 4) : 0u;
            #pragma unroll
            for (int t = 0; t < 4; ++t)
            {
                const uint32_t* tw = st + (q * 4 + t) * TW;
                uint32_t af[4];
                dq8_a(tw[src_a], tw[src_b], s2, af);
                mma16816(c[t], af, xb0, xb1);
            }
        }
        __syncwarp();                                      // ring slot free before the next issue overwrites it
        ++kb;
        if (kb == P::KBN || i == n - 1)
        {
            flush(j, p, gr, c);
            #pragma unroll
            for (int t = 0; t < 4; ++t) c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.f;
            if (kb == P::KBN) { kb = 0; if (++gr == P::G) { gr = 0; if (++p == MATS) { p = 0; ++j; } } }
        }
    }
    cp_wait<0>();
}

// RED flush of a warp's raw sums: lane holds (out col gr*64 + t*16 + g (+8), rows 2c, 2c+1)
template <int N>
__device__ __forceinline__ void red_flush(float* dst_rows, int R, int gr, float (*c)[4])   // dst_rows = row 0 of [MAXR][N]
{
    const int lane = threadIdx.x & 31, g = lane >> 2, r0 = (lane & 3) * 2;
    #pragma unroll
    for (int t = 0; t < 4; ++t)
    {
        const int col = gr * 64 + t * 16 + g;
        if (r0 < R) { atomicAdd(dst_rows + (size_t) r0 * N + col, c[t][0]); atomicAdd(dst_rows + (size_t) r0 * N + col + 8, c[t][2]); }
        if (r0 + 1 < R) { atomicAdd(dst_rows + (size_t) (r0 + 1) * N + col, c[t][1]); atomicAdd(dst_rows + (size_t) (r0 + 1) * N + col + 8, c[t][3]); }
    }
}

__global__ void __launch_bounds__(NT, 2) moe3_a_kernel(const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo,
    int nb, Tabs tb, Scr sc)
{
    __shared__ Route rt;
    extern __shared__ __align__(16) uint32_t ring_dyn[];
    route(sel, rw, R, lo, nb, rt);
    const int64_t* trp[2] = {tb.tr[0], tb.tr[1]};
    stream3<2, I, H, STAGES_A>(rt, trp, R,
        [&](int j, int p) { return sc.xr + ((size_t) j * 2 + p) * MAXR * H; },
        [&](int j, int p, int gr, float (*c)[4]) { red_flush<I>(sc.gu + ((size_t) j * 2 + p) * MAXR * I, R, gr, c); },
        ring_dyn + (threadIdx.x >> 5) * STAGES_A * UNIT_W, true);
}

// act: warp job (slot j, row r, 128-chunk c of I)
__global__ void __launch_bounds__(NT) moe3_act_kernel(const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo,
    int nb, Tabs tb, Scr sc)
{
    __shared__ Route rt;
    route(sel, rw, R, lo, nb, rt);
    pdl_wait();
    pdl_launch();
    const int lane = threadIdx.x & 31, jobs = rt.m * R * (I / 128);
    const float ki = k_inv(), kbias = k_bias();
    for (int q = blockIdx.x * WPB + (threadIdx.x >> 5); q < jobs; q += gridDim.x * WPB)
    {
        const int j = q / (R * (I / 128)), r = (q / (I / 128)) % R, cc = q % (I / 128), le = rt.e[j];
        const int col = cc * 128 + lane * 4;
        float g[4], u[4];
        #pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            float* src = sc.gu + (((size_t) j * 2 + p) * MAXR + r) * I + col;
            const float4 v = __ldcg((const float4*) src);
            *(float4*) src = make_float4(0.f, 0.f, 0.f, 0.f);
            const float bs = kbias * sc.xs[(j * 2 + p) * MAXR + r];
            float* o = p ? u : g;
            o[0] = fmaf(ki, v.x, bs); o[1] = fmaf(ki, v.y, bs); o[2] = fmaf(ki, v.z, bs); o[3] = fmaf(ki, v.w, bs);
            had128(o[0], o[1], o[2], o[3], lane);
            const uint2 svv = *(const uint2*) ((const half*) tb.sv[p][le] + col);
            const float2 s01 = __half22float2(*(const half2*) &svv.x), s23 = __half22float2(*(const half2*) &svv.y);
            o[0] *= s01.x; o[1] *= s01.y; o[2] *= s23.x; o[3] *= s23.y;
        }
        const uint2 dsv = *(const uint2*) ((const half*) tb.su[2][le] + col);
        const float2 d01 = __half22float2(*(const half2*) &dsv.x), d23 = __half22float2(*(const half2*) &dsv.y);
        float a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = g[i] / (1.f + __expf(-g[i])) * u[i];
        a[0] *= d01.x; a[1] *= d01.y; a[2] *= d23.x; a[3] *= d23.y;
        had128(a[0], a[1], a[2], a[3], lane);
        const half2 o01 = __floats2half2_rn(a[0], a[1]), o23 = __floats2half2_rn(a[2], a[3]);
        uint2 ov; ov.x = *(const uint32_t*) &o01; ov.y = *(const uint32_t*) &o23;
        *(uint2*) (sc.ar + ((size_t) j * MAXR + r) * I + col) = ov;
        const float2 f01 = __half22float2(o01), f23 = __half22float2(o23);
        const float s = warp_sum((f01.x + f01.y) + (f23.x + f23.y));
        if (lane == 0) atomicAdd(sc.as + j * MAXR + r, s);
    }
}

__global__ void __launch_bounds__(NT, 2) moe3_b_kernel(const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo,
    int nb, Tabs tb, Scr sc)
{
    __shared__ Route rt;
    extern __shared__ __align__(16) uint32_t ring_dyn[];
    route(sel, rw, R, lo, nb, rt);
    const int64_t* trp[1] = {tb.tr[2]};
    stream3<1, H, I, STAGES_B>(rt, trp, R,
        [&](int j, int) { return sc.ar + (size_t) j * MAXR * I; },
        [&](int j, int, int gr, float (*c)[4]) { red_flush<H>(sc.ds + (size_t) j * MAXR * H, R, gr, c); },
        ring_dyn + (threadIdx.x >> 5) * STAGES_B * UNIT_W, true);
}

// fin: block (row r, 128-chunk c of H); warp w adds experts w, w + 8, ...:
// out[r] = sum_j w[j][r] * svh_j * H128(k_inv * ds + k_bias * as[j][r])
__global__ void __launch_bounds__(NT) moe3_fin_kernel(const int64_t* __restrict__ sel, const half* __restrict__ rw, int R, int lo,
    int nb, Tabs tb, Scr sc, float* __restrict__ out)
{
    __shared__ Route rt;
    __shared__ float4 part[WPB][32];
    route(sel, rw, R, lo, nb, rt);
    pdl_wait();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r = blockIdx.x / (H / 128), cc = blockIdx.x % (H / 128), col = cc * 128 + lane * 4;
    const float ki = k_inv(), kbias = k_bias();
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = w; j < rt.m; j += WPB)
    {
        float* src = sc.ds + ((size_t) j * MAXR + r) * H + col;
        const float4 v = __ldcg((const float4*) src);
        *(float4*) src = make_float4(0.f, 0.f, 0.f, 0.f);
        const float wj = rt.w[j][r];
        if (wj == 0.f) continue;
        const float bs = kbias * sc.as[j * MAXR + r];
        float d0 = fmaf(ki, v.x, bs), d1 = fmaf(ki, v.y, bs), d2 = fmaf(ki, v.z, bs), d3 = fmaf(ki, v.w, bs);
        had128(d0, d1, d2, d3, lane);
        const uint2 svv = *(const uint2*) ((const half*) tb.sv[2][rt.e[j]] + col);
        const float2 s01 = __half22float2(*(const half2*) &svv.x), s23 = __half22float2(*(const half2*) &svv.y);
        o[0] += wj * d0 * s01.x; o[1] += wj * d1 * s01.y; o[2] += wj * d2 * s23.x; o[3] += wj * d3 * s23.y;
    }
    part[w][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (w == 0)
    {
        float4 t = part[0][lane];
        #pragma unroll
        for (int q = 1; q < WPB; ++q) { const float4 u = part[q][lane]; t.x += u.x; t.y += u.y; t.z += u.z; t.w += u.w; }
        *(float4*) (out + (size_t) r * H + col) = t;
    }
    if (blockIdx.x == 0)                                   // the gate / up input sums are consumed (act ran before)
        for (int i = threadIdx.x; i < MAXJ * 2 * MAXR; i += NT) sc.xs[i] = 0.f;
}

template <typename Kern, typename... Args>
void launch_pdl(Kern k, dim3 grid, cudaStream_t st, int dsmem, bool pdl, Args... args)
{
    cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid; cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = dsmem; cfg.stream = st; cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}
}  // namespace

std::vector<at::Tensor> moe3_scratch(at::Device dev, int64_t h, int64_t i)
{
    TORCH_CHECK(h == H && i == I, "moe3: shapes");
    auto f = at::TensorOptions().device(dev);
    return {at::zeros({MAXJ, 2, MAXR, H}, f.dtype(at::kHalf)), at::zeros({MAXJ, 2, MAXR}, f.dtype(at::kFloat)),
            at::zeros({MAXJ, 2, MAXR, I}, f.dtype(at::kFloat)), at::zeros({MAXJ, MAXR, I}, f.dtype(at::kHalf)),
            at::zeros({MAXJ, MAXR}, f.dtype(at::kFloat)), at::zeros({MAXJ, MAXR, H}, f.dtype(at::kFloat))};
}

// x [R][H] fp16, sel [R][8] int64 (permuted expert ids; ids in [lo, lo + nb) are the band), rw [R][8] fp16; tables
// int64 [nb] device pointers (trellis / suh / svh for gate, up, down); out [R][H] fp32 (overwritten)
void moe3_forward(at::Tensor x, at::Tensor sel, at::Tensor rw, int64_t lo, int64_t nb, at::Tensor gtr, at::Tensor gsu,
                  at::Tensor gsv, at::Tensor utr, at::Tensor usu, at::Tensor usv, at::Tensor dtr, at::Tensor dsu, at::Tensor dsv,
                  at::Tensor xr, at::Tensor xs, at::Tensor gu, at::Tensor ar, at::Tensor as, at::Tensor ds, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int R = (int) x.size(0);
    TORCH_CHECK(R >= 1 && R <= MAXR && x.size(1) == H && sel.size(1) == TOPK && nb <= 128, "moe3: shapes");
    Tabs tb{{(const int64_t*) gtr.data_ptr(), (const int64_t*) utr.data_ptr(), (const int64_t*) dtr.data_ptr()},
            {(const int64_t*) gsu.data_ptr(), (const int64_t*) usu.data_ptr(), (const int64_t*) dsu.data_ptr()},
            {(const int64_t*) gsv.data_ptr(), (const int64_t*) usv.data_ptr(), (const int64_t*) dsv.data_ptr()}};
    Scr sc{(half*) xr.data_ptr(), (float*) xs.data_ptr(), (float*) gu.data_ptr(), (half*) ar.data_ptr(), (float*) as.data_ptr(),
           (float*) ds.data_ptr()};
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t* s = (const int64_t*) sel.data_ptr(); const half* w = (const half*) rw.data_ptr();
    static bool attr = [] {
        cudaFuncSetAttribute(moe3_a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, WPB * STAGES_A * UNIT_W * 4);
        cudaFuncSetAttribute(moe3_b_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, WPB * STAGES_B * UNIT_W * 4);
        return true;
    }();
    (void) attr;
    launch_pdl(moe3_rot_kernel, dim3(R * TOPK, 8), st, 0, false, (const half*) x.data_ptr(), s, w, R, (int) lo, (int) nb, tb, sc);
    launch_pdl(moe3_a_kernel, dim3(sms * 2), st, WPB * STAGES_A * UNIT_W * 4, true, s, w, R, (int) lo, (int) nb, tb, sc);
    launch_pdl(moe3_act_kernel, dim3(sms), st, 0, true, s, w, R, (int) lo, (int) nb, tb, sc);
    launch_pdl(moe3_b_kernel, dim3(sms * 2), st, WPB * STAGES_B * UNIT_W * 4, true, s, w, R, (int) lo, (int) nb, tb, sc);
    launch_pdl(moe3_fin_kernel, dim3(R * (H / 128)), st, 0, true, s, w, R, (int) lo, (int) nb, tb, sc, (float*) out.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scratch", &moe3_scratch);
    m.def("forward", &moe3_forward);
}
