// hecate (nr4): nr3 + the MoE handoff to the RTX and Grace bands (see the end of this comment). The post-attention RMSNorm (fused residual add) and GLM's sigmoid / bias top-8 router of a decode step
// (2..8 rows of 6144, 256 experts) as one kernel of 4 thread-block clusters x 16 CTAs.
//
// Cluster g routes experts [64 g, +64). CTA c of every cluster owns the columns of rms_norm_res_in's warps c and 16 + c
// (float4 columns [32c, +32) and [1024 + 32c, +32), and [512 + 32c, +32)), computes those warps' partial sums with the
// same lane mapping, fma order and xor tree, and exchanges them through distributed shared memory, so each CTA
// finishes the norm's second-level reduction over the same 32 values: the residual update and the normed rows are
// bit-identical to rms_norm_res_in. Each CTA multiplies its 384 normed columns by its cluster's 64 x 384 slice of the
// transposed gate (staged with cp.async while the norm runs) on mma.sync with fp32 accumulation; the cluster leader
// sums the 16 partial logit sets through distributed shared memory and takes its 64 experts' top 8 of
// sigmoid(logit) + bias (the fused router's key order: key desc, then lowest index). The leaders post these
// candidates to global memory; the last to arrive at the call counter merges the 4 x 8 candidates per row with the same
// order, which selects exactly the global top 8, and writes the indices and the normalized, scaled weights.
// Cluster 0 writes the normed rows; each residual column set is written by the last of its 4 readers (a per-set
// counter), so no CTA overwrites a residual another cluster still reads. Counters reset themselves (graph-safe).
// The logits are summed in a different fp32 order than the fused router's (whose k slices meet in fp32 atomics), so
// exact near-ties can resolve differently, as they already can between its runs. Build with --use_fast_math (as
// ExLlamaV3's norm.cu: the norm's division and flush-to-zero must match); the router math uses explicit
// round-to-nearest division like the fused router's non-fast-math build.
// Handoff (hy != nullptr): cluster 0 also writes the normed rows into the layer's pinned slot (host memory over C2C) and
// fences them at system scope before its cluster barrier; the last leader writes the selection and weights into the
// slot, the Grace job header, resets the layer's RTX DONE, fences, and releases the RTX and Grace ISSUE flags, as
// moe4_handoff does one kernel later.
// x2f (EXL3_MOE4_XF): cluster 0 also writes the normed rows in moe4's B-fragment order (a lane's 8 words of each 64-k
// block contiguous), which moe4 phase A reads with two 16-byte loads per unit.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace hecate_nr4 {
namespace cg = cooperative_groups;

constexpr int CS = 16, NT = 512, DIM = 6144, NE = 256, KC = 384, MAXM = 8, TOPK = 8, WST = KC + 8;
constexpr int G = 4, EPC = NE / G, MT = EPC / 16, WPM = 16 / MT, KPW = (KC / 16) / WPM;
constexpr int SM_W = EPC * WST * 2;               // gate slice [64 e][384 k] (+8 pad: ldmatrix conflict-free)
constexpr int SM_X = MAXM * WST * 2;              // normed slice [m][k]
constexpr int SM_PW = 16 * 16 * MAXM * 4;         // per-warp partial logits [warp][16 e][m]
constexpr int SM_P = EPC * MAXM * 4;              // this CTA's partial logits [e][m]
constexpr int SM_L = MAXM * EPC * 4;              // leader: summed logits [m][e]
constexpr int SM_S = MAXM * 32 * 4;               // the norm's 32 warp sums per row
constexpr int SMEM = SM_W + SM_X + SM_PW + SM_P + SM_L + SM_S + 16;
static_assert(KPW % 2 == 0 && G * TOPK <= 32, "nr3 shape");

struct Params
{
    const float* y; float* r; const void* w; half* x2; const half* gt; const float* bias;
    int64_t* sel; half* rw;
    int4* cand;              // [G][MAXM][TOPK] (key, activation bits, expert, 0)
    int* ctr;                // [CS] residual set counters, [CS] call counter (zero, self-resetting)
    int M;
    float eps, cbias, cscale, scale;
    half* hy; int64_t* hs; half* hw;                  // handoff slot (nullptr: no handoff)
    unsigned* hflag; unsigned hgen; unsigned* hmeta; unsigned hm0; unsigned* hflag2; unsigned hgen2; unsigned* hzero;
    half* x2f;               // the normed rows again in moe4's B-fragment order (nullptr: none)
#ifdef NR_STAMPS
    unsigned long long* st;
#endif
};

__device__ __forceinline__ int chunk_base(int c, int j) { return j == 0 ? 128 * c : j == 1 ? 4096 + 128 * c : 2048 + 128 * c; }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ unsigned okey(float v) { const unsigned u = __float_as_uint(v); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

__device__ __forceinline__ float sum_sq4(float s, const float4& f)
{
    s = fmaf(f.x, f.x, s); s = fmaf(f.y, f.y, s); s = fmaf(f.z, f.z, s); s = fmaf(f.w, f.w, s);
    return s;
}

template <bool WBF16>
__device__ __forceinline__ float4 load_w4(const void* w, int e)
{
    float4 f;
    if constexpr (WBF16)
    {
        const __nv_bfloat162* p = (const __nv_bfloat162*) ((const __nv_bfloat16*) w + e);
        const __nv_bfloat162 a = p[0], b = p[1];
        f.x = __bfloat162float(__low2bfloat16(a)); f.y = __bfloat162float(__high2bfloat16(a));
        f.z = __bfloat162float(__low2bfloat16(b)); f.w = __bfloat162float(__high2bfloat16(b));
    }
    else
    {
        const half2* p = (const half2*) ((const half*) w + e);
        const half2 a = p[0], b = p[1];
        f.x = __low2float(a); f.y = __high2float(a); f.z = __low2float(b); f.w = __high2float(b);
    }
    return f;
}

__device__ __forceinline__ float4 add4(const float4& a, const float4& b)
{
    float4 s; s.x = a.x + b.x; s.y = a.y + b.y; s.z = a.z + b.z; s.w = a.w + b.w;
    return s;
}

// x * (w + cbias) * rmf -> fp16, into the shared slice and (cluster 0) the output rows, the handoff slot and the
// fragment-order copy (gf = that row, e = the first of the 4 columns: words e / 2, e / 2 + 1 -> moe4's frag_word)
__device__ __forceinline__ void out4(float4 s, float4 w, float cbias, float rmf, half* g, half* sm, half* h, half* gf = nullptr, int e = 0)
{
    if (cbias != 0.0f) { w.x += cbias; w.y += cbias; w.z += cbias; w.w += cbias; }
    s.x = s.x * w.x * rmf; s.y = s.y * w.y * rmf; s.z = s.z * w.z * rmf; s.w = s.w * w.w * rmf;
    const half2 h0 = __halves2half2(__float2half_rn(s.x), __float2half_rn(s.y));
    const half2 h1 = __halves2half2(__float2half_rn(s.z), __float2half_rn(s.w));
    if (g) { ((half2*) g)[0] = h0; ((half2*) g)[1] = h1; }
    if (h) { ((half2*) h)[0] = h0; ((half2*) h)[1] = h1; }
    if (gf)
    {
        uint32_t* gw = (uint32_t*) gf;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int p = e / 2 + i, wq = p & 31;
            gw[(p & ~31) + (wq & 3) * 8 + (wq >> 3) * 2 + ((wq & 7) >> 2)] = i ? *(const uint32_t*) &h1 : *(const uint32_t*) &h0;
        }
    }
    ((half2*) sm)[0] = h0; ((half2*) sm)[1] = h1;
}

template <bool WBF16>
__global__ void __launch_bounds__(NT, 1) nr4_kernel(const Params p)
{
    cg::cluster_group cl = cg::this_cluster();
    const int c = (int) cl.block_rank(), gi = (int) blockIdx.x / CS;
    extern __shared__ __align__(16) unsigned char smem[];
    half* Ws = (half*) smem;
    half* Xs = (half*) (smem + SM_W);
    float* Pw = (float*) (smem + SM_W + SM_X);
    float* Ps = (float*) (smem + SM_W + SM_X + SM_PW);
    float* Ls = (float*) (smem + SM_W + SM_X + SM_PW + SM_P);
    float* Ss = (float*) (smem + SM_W + SM_X + SM_PW + SM_P + SM_L);
    int* flag = (int*) (smem + SM_W + SM_X + SM_PW + SM_P + SM_L + SM_S);
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, M = p.M;
#ifdef NR4_PDL
    // hecate (EXL3_I8_PDL): let moe4 phase A (a programmatic dependent that waits before reading the routing) become
    // resident on the 88 SMs this kernel leaves idle
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#ifdef NR_STAMPS
    auto stamp = [&](int i) { if (tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); p.st[blockIdx.x * 8 + i] = t; } };
#else
    auto stamp = [](int) {};
#endif
    stamp(0);

    // the norm's loads first: warp 2m = rms_norm_res_in's warp c of row m (chunks 0, 1), warp 2m + 1 its warp 16 + c
    const int m = wid >> 1, vb = wid & 1;
    const bool nw = m < M;
    const int ja = vb ? 2 : 0;
    const int ea = chunk_base(c, ja) + 4 * lane, eb = chunk_base(c, 1) + 4 * lane;
    const size_t ro = (size_t) m * DIM;
    float4 ya = {}, ra = {}, yb = {}, rb = {}, wa = {}, wb = {};
#ifdef NR4_PDLW
    // hecate (EXL3_NR4_PDLW): a programmatic dependent of o_proj: the constants (norm weight, score bias, the gate slice
    // below) are requested before griddepcontrol.wait, the attention output and the residual after it
    if (nw) { wa = load_w4<WBF16>(p.w, ea); if (!vb) wb = load_w4<WBF16>(p.w, eb); }
#else
    if (nw)
    {
        ya = *(const float4*) (p.y + ro + ea); ra = *(const float4*) (p.r + ro + ea); wa = load_w4<WBF16>(p.w, ea);
        if (!vb) { yb = *(const float4*) (p.y + ro + eb); rb = *(const float4*) (p.r + ro + eb); wb = load_w4<WBF16>(p.w, eb); }
    }
#endif
    float bz0 = 0.f, bz1 = 0.f;                                   // the leader's rows: score bias of its 2 experts per lane
    if (c == 0 && wid < M && p.bias) { bz0 = __ldg(p.bias + gi * EPC + lane); bz1 = __ldg(p.bias + gi * EPC + 32 + lane); }

    // this cluster's gate slice -> shared memory (lands while the norm runs)
    {
        const uint32_t wsb = (uint32_t) __cvta_generic_to_shared(Ws);
        const half* g0 = p.gt + (size_t) gi * EPC * DIM;
        #pragma unroll
        for (int u = 0; u < EPC * 48 / NT; ++u)
        {
            const int i = tid + u * NT, e = i / 48, j = (i % 48) / 16, pc = i % 16;
            const half* src = g0 + (size_t) e * DIM + chunk_base(c, j) + pc * 8;
            const uint32_t dst = wsb + (uint32_t) ((e * WST + j * 128 + pc * 8) * 2);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#ifdef NR4_PDLW
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (nw)
    {
        ya = *(const float4*) (p.y + ro + ea); ra = *(const float4*) (p.r + ro + ea);
        if (!vb) { yb = *(const float4*) (p.y + ro + eb); rb = *(const float4*) (p.r + ro + eb); }
    }
#endif

    float4 sa = {}, sb = {};
    if (nw)
    {
        sa = add4(ya, ra);
        if (!vb) sb = add4(yb, rb);
        float s = sum_sq4(0.f, sa);
        if (!vb) s = sum_sq4(s, sb);
        #pragma unroll
        for (int o = 16; o > 0; o /= 2) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane < CS) *cl.map_shared_rank(Ss + m * 32 + (vb ? 16 + c : c), lane) = s;
    }
    stamp(1);
    __syncthreads();                                              // this CTA's residual reads are done (consumed above)
    cl_arrive();
    if (tid == NT - 1)                                            // count them while the cluster barrier completes
    {
        int old_;
        asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], 1;" : "=r"(old_) : "l"(p.ctr + c) : "memory");
        *flag = old_ == G - 1;
    }
    cl_wait();
    stamp(2);
    if (nw)
    {
        float s = Ss[m * 32 + lane];
        #pragma unroll
        for (int o = 16; o > 0; o /= 2) s += __shfl_xor_sync(0xffffffffu, s, o);
        const float rmf = rsqrtf(s / (float) DIM + p.eps) * p.cscale;
        const bool ho = gi == 0 && p.hy;
        half* const xf = gi == 0 && p.x2f ? p.x2f + ro : nullptr;
        out4(sa, wa, p.cbias, rmf, gi == 0 ? p.x2 + ro + ea : nullptr, Xs + m * WST + ja * 128 + 4 * lane, ho ? p.hy + ro + ea : nullptr, xf, ea);
        if (!vb) out4(sb, wb, p.cbias, rmf, gi == 0 ? p.x2 + ro + eb : nullptr, Xs + m * WST + 128 + 4 * lane, ho ? p.hy + ro + eb : nullptr, xf, eb);
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (gi == 0 && p.hy && tid == 0) __threadfence_system();     // this CTA's slot rows, before its cluster barrier
    stamp(3);
    if (*flag)                                                    // the last reader of this column set writes it back
    {
        if (nw)
        {
            *(float4*) (p.r + ro + ea) = sa;
            if (!vb) *(float4*) (p.r + ro + eb) = sb;
        }
        if (tid == NT - 1) p.ctr[c] = 0;
    }

    // warp w: experts [16 (w / WPM), +16), k-tiles [(w % WPM) KPW, +KPW) as two accumulator chains
    {
        float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
        const int g = lane >> 2, q = lane & 3, mt = wid / WPM, kg = wid % WPM;
        const uint32_t arow = (uint32_t) __cvta_generic_to_shared(Ws) + (uint32_t) (((16 * mt + (lane & 15)) * WST + (lane >> 4) * 8) * 2);
        const half* xr = Xs + g * WST + 2 * q;
        const bool live = g < M;
        #pragma unroll
        for (int t = 0; t < KPW; t += 2)
        {
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int kt = kg * KPW + t + h;
                uint32_t f0, f1, f2, f3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(f0), "=r"(f1), "=r"(f2), "=r"(f3) : "r"(arow + kt * 32));
                const uint32_t b0 = live ? *(const uint32_t*) (xr + kt * 16) : 0u;
                const uint32_t b1 = live ? *(const uint32_t*) (xr + kt * 16 + 8) : 0u;
                if (h == 0)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(a0[0]), "+f"(a0[1]), "+f"(a0[2]), "+f"(a0[3])
                                 : "r"(f0), "r"(f1), "r"(f2), "r"(f3), "r"(b0), "r"(b1));
                else
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(a1[0]), "+f"(a1[1]), "+f"(a1[2]), "+f"(a1[3])
                                 : "r"(f0), "r"(f1), "r"(f2), "r"(f3), "r"(b0), "r"(b1));
            }
        }
        float* pw = Pw + wid * 16 * MAXM;
        *(float2*) (pw + g * MAXM + 2 * q) = make_float2(a0[0] + a1[0], a0[1] + a1[1]);
        *(float2*) (pw + (g + 8) * MAXM + 2 * q) = make_float2(a0[2] + a1[2], a0[3] + a1[3]);
    }
    __syncthreads();
    {                                                             // sum the k groups of each m-tile, fixed order
        const int e = tid / MAXM, mm = tid % MAXM, mt = e / 16;
        float v = 0.f;
        #pragma unroll
        for (int kg = 0; kg < WPM; ++kg) v += Pw[(mt * WPM + kg) * 16 * MAXM + (e % 16) * MAXM + mm];
        Ps[tid] = v;
    }
    stamp(4);
    cl_arrive();                                                  // partial logits visible to the leader
    cl_wait();
    stamp(5);
    if (c != 0)
    {
        cl_arrive();                                              // stay resident until the leader has read them
        cl_wait();
        return;
    }
    {                                                             // leader: sum the 16 partial sets for its 64 experts
        const int e = tid / MAXM, mm = tid % MAXM;
        if (mm < M)
        {
            float v[CS];
            #pragma unroll
            for (int rk = 0; rk < CS; ++rk) v[rk] = *cl.map_shared_rank(Ps + tid, rk);
            float t = 0.f;
            #pragma unroll
            for (int rk = 0; rk < CS; ++rk) t += v[rk];
            Ls[mm * EPC + e] = t;
        }
    }
    __syncthreads();
    cl_arrive();
    cl_wait();
    stamp(6);

    // leader, warp r: this cluster's top 8 of row r (2 experts per lane) -> candidates
    if (wid < M)
    {
        const int r = wid;
        const float act0 = __frcp_rn(1.f + __expf(-Ls[r * EPC + lane])), act1 = __frcp_rn(1.f + __expf(-Ls[r * EPC + 32 + lane]));
        unsigned k0 = okey(act0 + bz0), k1 = okey(act1 + bz1);
        int4* cd = p.cand + ((size_t) gi * MAXM + r) * TOPK;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k)
        {
            const unsigned best = k0 >= k1 ? k0 : k1;
            const unsigned mx = __reduce_max_sync(0xffffffffu, best);
            const unsigned mine = best == mx ? (unsigned) (gi * EPC + (k0 == mx ? lane : 32 + lane)) : 0xffffffffu;
            const unsigned bi = __reduce_min_sync(0xffffffffu, mine);
            if (mine == bi)
            {
                const bool first = k0 == mx;
                cd[k] = make_int4((int) mx, __float_as_int(first ? act0 : act1), (int) bi, 0);
                if (first) k0 = 0u; else k1 = 0u;
            }
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        int old_;
        asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], 1;" : "=r"(old_) : "l"(p.ctr + CS) : "memory");
        *flag = old_ == G - 1;
        if (old_ == G - 1) p.ctr[CS] = 0;
    }
    __syncthreads();
    if (!*flag) return;

    // the last leader: merge the 4 x 8 candidates of row r (one per lane) in the same order
    if (wid < M)
    {
        const int r = wid;
        const int4 cv = __ldcg(p.cand + ((size_t) (lane / TOPK) * MAXM + r) * TOPK + lane % TOPK);
        unsigned lk = (unsigned) cv.x; const float la = __int_as_float(cv.y); const unsigned lid = (unsigned) cv.z;
        float wsel[TOPK]; unsigned isel[TOPK]; float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k)
        {
            const unsigned mx = __reduce_max_sync(0xffffffffu, lk);
            const unsigned bi = __reduce_min_sync(0xffffffffu, lk == mx ? lid : 0xffffffffu);
            const int src = __ffs(__ballot_sync(0xffffffffu, lk == mx && lid == bi)) - 1;
            const float a = __shfl_sync(0xffffffffu, la, src);
            if (lane == src) lk = 0u;
            wsel[k] = a; isel[k] = bi; sum += a;
        }
        float a = 0.f; unsigned id = 0;
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) if (k == lane) { a = wsel[k]; id = isel[k]; }
        if (lane < TOPK)
        {
            const half h = __float2half_rn(__fdiv_rn(a * p.scale, sum + 1e-20f));
            p.sel[r * TOPK + lane] = (int64_t) id;
            p.rw[r * TOPK + lane] = h;
            if (p.hy) { p.hs[r * TOPK + lane] = (int64_t) id; p.hw[r * TOPK + lane] = h; }
        }
    }
    if (!p.hy) { stamp(7); return; }
    __syncthreads();
    if (tid == 0)                                                 // the job header, RTX DONE reset, then ISSUE (as moe4_handoff)
    {
        if (p.hmeta) { p.hmeta[0] = p.hm0; p.hmeta[1] = (unsigned) M; }
        if (p.hzero) *(volatile unsigned*) p.hzero = 0u;
        // one system fence orders every prior write; both ISSUE stores are then relaxed and leave back to back (two
        // st.release.sys serialized a host round trip before the second, the Grace's; -1.3 us per layer), Grace first
        __threadfence_system();
        if (p.hflag2) asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p.hflag2), "r"(p.hgen2) : "memory");
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p.hflag), "r"(p.hgen) : "memory");
    }
    stamp(7);
}

}  // namespace hecate_nr4
