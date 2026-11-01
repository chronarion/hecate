// Hecate int4 MoE prefill (moe4p): the GB300's int4 tiers (hot experts in HBM, cold experts read in place from pinned
// Grace memory over C2C) for prompt chunks, sm_103. Grouped GEMMs on the moe4 weights as stored (int4 code - 8, per-row fp32
// scale, mma.sync m16n8k16 A-fragment order), so no expert is dequantized to bf16 (the previous path wrote and read back
// ~12.9 GB of bf16 per layer and chunk):
//   gemm : block = (row block of <= 64 picks of one expert, matrix, 256 output rows). A cp.async ring holds the int4
//          fragments (16 strips x 32 lanes x 16 B per 64-k stage) and the picks' inputs (64 x 64 fp16, 16-byte chunks
//          swizzled by row; gate / up gather the token rows straight from the MoE input). Each warp owns two 16-row strips:
//          LOP3 decode of a 16-byte lane word = A fragments of 4 k-tiles, multiplied by every 8-row group of the block
//          (B fragments by ldmatrix). Raw sums sum((code - 8) x) in fp32, stored without atomics
//   act  : a = silu(s_g * g) * (s_u * u) * routing weight (fp16, as the decode kernel's act)
//   gemm : the down projection over a (same kernel)
//   fin  : out[tok] += s_d * d
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

constexpr int H = 6144, I = 2048;
constexpr int WARPS = 8, NT = WARPS * 32;
constexpr int BM = 64;                                  // picks per row block
constexpr int BN = WARPS * 32;                          // output rows per block: two 16-row strips per warp
constexpr int BK = 64;                                  // k per stage (4 k-tiles = one moe4 k-block)
#ifndef STAGES
#define STAGES 4
#endif
constexpr int XS_BYTES = BM * BK * 2;                   // inputs per stage (8 KB)
constexpr int WS_BYTES = (BN / 16) * 32 * 16;           // int4 fragments per stage: 16 strips x 32 lanes x 16 B (8 KB)
constexpr int SMEM = STAGES * (XS_BYTES + WS_BYTES);

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
// one 32-bit word of a lane's fragment = A fragment of one k-tile: LOP3 magic number into f16, minus 8 (exact; as moe4)
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

// ---------------------------------------------------------------------------------------------------------------------
// gemm: block (row block rb, matrix p and 256-output-row block cb). rbt[rb] = {expert, first pick, picks}.
// Inputs: tok != nullptr: row r of the block is x[tok[pos0 + r]] (x [rows][K]); else x [ps][K] by pick.
// w0 / w1: int64 [nb] pointers to the experts' fragment-order int4 of matrix p = 0 / 1; out [MATS][ps][N] fp32 raw sums
template <int N, int K>
__global__ void __launch_bounds__(NT, 2) p4_gemm_kernel(const int4* __restrict__ rbt, const int64_t* __restrict__ w0,
    const int64_t* __restrict__ w1, const half* __restrict__ x, const int* __restrict__ tok, float* __restrict__ out, int ps)
{
    extern __shared__ __align__(128) uint8_t smem[];
    constexpr int KB = K / BK, CBN = N / BN;
    const int4 rb = rbt[blockIdx.x];
    const int e = rb.x, pos0 = rb.y, rows = rb.z;
    const int p = blockIdx.y / CBN, cb = blockIdx.y % CBN;
    // fragment-order int4: uint4 ((gr * KB + kb) * 4 + s) * 32 + lane; this block's groups gr = cb * 4 .. cb * 4 + 3
    const uint4* wb = (const uint4*) (p ? w1 : w0)[e] + (size_t) cb * 4 * KB * 4 * 32;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    uint8_t* xsm = smem;                                     // [STAGES][BM][BK] fp16, 16-byte chunk c of row r at c ^ (r & 7)
    uint8_t* wsm = smem + STAGES * XS_BYTES;                 // [STAGES][16 strips][32 lanes] x 16 B
    const int nnt = (rows + 7) >> 3;

    // the two input rows this thread copies (chunks t and t + NT: rows t / 8 and t / 8 + 32, 16-byte chunk t % 8)
    const half* xr[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const int r = (t >> 3) + 32 * i;
        xr[i] = r < rows ? x + (size_t) (tok ? tok[pos0 + r] : (p * ps + pos0 + r)) * K + (t & 7) * 8 : nullptr;
    }
    auto issue = [&](int kb)
    {
        if (kb < KB)
        {
            const int st = kb % STAGES;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                const int r = (t >> 3) + 32 * i, ch = t & 7;
                if (xr[i]) cp16(su32(xsm + st * XS_BYTES + r * (BK * 2) + ((ch ^ (r & 7)) << 4)), xr[i] + kb * BK);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)                      // 512 chunks: 4 groups x (4 strips x 32 lanes), 2 KB each
            {
                const int c = t + i * NT, g = c >> 7, o = c & 127;
                cp16(su32(wsm + st * WS_BYTES + c * 16), wb + ((size_t) g * KB + kb) * 4 * 32 + o);
            }
        }
        cp_commit();
    };

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
        __syncthreads();
        issue(kb + STAGES - 1);
        const int st = kb % STAGES;
        const uint32_t xsb = su32(xsm + st * XS_BYTES);
        uint4 wf[2];
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) wf[mt] = *(const uint4*) (wsm + st * WS_BYTES + ((w * 2 + mt) * 32 + lane) * 16);
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            uint32_t af[2][4];
            dec8(q == 0 ? wf[0].x : q == 1 ? wf[0].y : q == 2 ? wf[0].z : wf[0].w, af[0]);
            dec8(q == 0 ? wf[1].x : q == 1 ? wf[1].y : q == 2 ? wf[1].z : wf[1].w, af[1]);
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

    // lane holds (output row w*32 + mt*16 + g (+8), picks nt*8 + 2*(lane%4) (+1))
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

// act: block (pick pos), 8 columns per thread: a = silu(s_g * g) * (s_u * u) * w
__global__ void __launch_bounds__(NT) p4_act_kernel(const int* __restrict__ eid, const float* __restrict__ wt,
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

// fin: block (pick pos), 4 columns per thread per step: out[tok] += s_d * d
__global__ void __launch_bounds__(NT) p4_fin_kernel(const int* __restrict__ tok, const int* __restrict__ eid,
    const int64_t* __restrict__ sd, const float* __restrict__ d, float* __restrict__ out)
{
    const int pos = blockIdx.x;
    const float* s = (const float*) sd[eid[pos]];
    float* orow = out + (size_t) tok[pos] * H;
    for (int col = threadIdx.x * 4; col < H; col += NT * 4)
    {
        const float4 v = *(const float4*) (d + (size_t) pos * H + col), sv = *(const float4*) (s + col);
        atomicAdd((float4*) (orow + col), make_float4(v.x * sv.x, v.y * sv.y, v.z * sv.z, v.w * sv.w));
    }
}
}  // namespace

// scratch for up to npos picks per call: gu [2][npos][I] fp32, ar [npos][I] fp16, d [npos][H] fp32
std::vector<at::Tensor> moe4p_scratch(at::Device dev, int64_t npos)
{
    auto f = at::TensorOptions().device(dev);
    return {at::empty({2, npos, I}, f.dtype(at::kFloat)), at::empty({npos, I}, f.dtype(at::kHalf)),
            at::empty({npos, H}, f.dtype(at::kFloat))};
}

// x [rows][H] fp16 (the MoE input); tok / eid [npos] int32: picks of the int4 tiers sorted by expert (token row, expert id);
// wt [npos] fp32 routing weights; rbt [nrb][4] int32 row blocks {expert, first pick, picks (<= 64), 0}; tables int64 [nb]
// (fragment-order int4 and fp32 row scales of gate, up, down); out [rows][H] fp32, accumulated into
void moe4p_forward(at::Tensor x, at::Tensor tok, at::Tensor eid, at::Tensor wt, at::Tensor rbt, at::Tensor gw, at::Tensor uw,
                   at::Tensor dw, at::Tensor gs, at::Tensor us, at::Tensor ds, at::Tensor gu, at::Tensor ar, at::Tensor d,
                   at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int npos = (int) tok.numel(), nrb = (int) rbt.size(0), ps = (int) gu.size(1);
    TORCH_CHECK(x.size(1) == H && x.dtype() == at::kHalf && out.dtype() == at::kFloat && npos <= ps && rbt.size(1) == 4,
                "moe4p: shapes");
    if (npos == 0) return;
    auto P = [](const at::Tensor& t) { return (const int64_t*) t.data_ptr(); };
    static bool attr = [] {
        cudaFuncSetAttribute(p4_gemm_kernel<I, H>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(p4_gemm_kernel<H, I>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        return true;
    }();
    (void) attr;
    const int* tk = (const int*) tok.data_ptr(); const int* ei = (const int*) eid.data_ptr();
    float* gup = (float*) gu.data_ptr(); half* arp = (half*) ar.data_ptr(); float* dp = (float*) d.data_ptr();
    p4_gemm_kernel<I, H><<<dim3(nrb, 2 * (I / BN)), NT, SMEM, st>>>((const int4*) rbt.data_ptr(), P(gw), P(uw),
                                                                    (const half*) x.data_ptr(), tk, gup, ps);
    p4_act_kernel<<<npos, NT, 0, st>>>(ei, (const float*) wt.data_ptr(), P(gs), P(us), gup, arp, ps);
    p4_gemm_kernel<H, I><<<dim3(nrb, H / BN), NT, SMEM, st>>>((const int4*) rbt.data_ptr(), P(dw), P(dw), arp, nullptr, dp, ps);
    p4_fin_kernel<<<npos, NT, 0, st>>>(tk, ei, P(ds), dp, (float*) out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scratch", &moe4p_scratch);
    m.def("forward", &moe4p_forward);
}
