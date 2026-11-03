// hecate: tcgen05 lightning-indexer scoring for prefill chunks (GLM-5.3 DSA indexer: 32 heads x 128 dims, fp16, raw-token
// keys). scores[r][s] = scale * sum_h w[r][h] * relu(q[r][h] . k[s]) for s < min(q_pos0 + r + 1, bound_max), -inf for the
// rest of [0, T) (as dsa_triton's _dsa_indexer_kernel, compress_rate 1); keys contiguous or in a paged pool (bt, epp:
// key s at row bt[s / epp] * epp + s % epp, epp a multiple of 128).
// CTA = 8 query rows x a range of 128-key tiles. Per tile one UMMA chain D[128 keys][256 = 8 queries x 32 heads] =
// K_tile . Q_blk^T (M128 N256, 8 x K16, fp32 in TMEM, two accumulators so the epilogue of tile i overlaps the MMAs of
// tile i + 1). Q block resident in shared memory (64 KB), key tiles through a 4-stage TMA ring. Warps 0-7 epilogue (TMEM
// lane quadrant = warp % 4 -> 32 keys; query half = warp / 4 -> 4 queries): each query's 32 head columns are reduced with
// the ReLU and the head weights in head order (fmaf, as the Triton kernel's loop), times scale, masked, fp16. Warp 8 TMA,
// warp 9 TMEM allocation + MMA issue.
#include <cuda_fp16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
constexpr int HI = 32, DI = 128, KT = 128, QB = 8, NSTG = 4, NT = 320;
constexpr int Q_BYTES = QB * HI * DI * 2;          // 64 KB: 256 rows x 128 dims, two SW128 atoms of 64 dims (32 KB each)
constexpr int K_BYTES = KT * DI * 2;               // 32 KB: 128 rows x 128 dims, two atoms (16 KB each)
constexpr int SM_Q = 0, SM_K = Q_BYTES, SM_W = SM_K + NSTG * K_BYTES;          // W: float [8][32] head weights
constexpr int SM_BAR = SM_W + QB * HI * 4;         // qfull, kfull[4], kempty[4], dfull[2], dempty[2]
constexpr int SM_TM = SM_BAR + 13 * 8;
constexpr int SMEM = SM_TM + 16 + 1024;            // + alignment slack

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(uint32_t b, uint32_t parity)
{
    asm volatile("{\n .reg .pred P1;\n W:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @!P1 bra W;\n}\n"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint32_t b, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
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

__global__ void __launch_bounds__(NT, 1) hidx_kernel(const __grid_constant__ CUtensorMap tmq, const __grid_constant__ CUtensorMap tmk,
    const half* __restrict__ w, half* __restrict__ scores, int R, int T, int q_pos0, int bound_max, int s_stride, float scale, int tpc,
    const int* __restrict__ bt, int epp)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = (unsigned char*) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = blockIdx.x * QB;
    const uint32_t bq = su32(smem + SM_BAR), bkf = bq + 8, bke = bq + 40, bdf = bq + 72, bde = bq + 88;
    uint32_t* tm_base = (uint32_t*) (smem + SM_TM);
    float* W = (float*) (smem + SM_W);
    // tiles of this CTA: [t0, t1) of the row block's [0, ceil(T / KT)); MMA only below the block's largest causal bound
    int maxb = 0;
    #pragma unroll
    for (int q = 0; q < QB; ++q) if (r0 + q < R) maxb = max(maxb, min(q_pos0 + r0 + q + 1, bound_max));
    const int nt_all = (T + KT - 1) / KT, t0 = blockIdx.y * tpc, t1 = min(t0 + tpc, nt_all);
    const int tc1 = max(t0, min(t1, (maxb + KT - 1) / KT)), nct = tc1 - t0;   // computed tiles [t0, tc1)
    if (tid == 0)
    {
        bar_init(bq, 1);
        for (int s = 0; s < NSTG; ++s) { bar_init(bkf + 8 * s, 1); bar_init(bke + 8 * s, 1); }
        for (int b = 0; b < 2; ++b) { bar_init(bdf + 8 * b, 1); bar_init(bde + 8 * b, 8); }
    }
    if (warp == 9)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(su32(tm_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    for (int i = tid; i < QB * HI; i += NT)
        W[i] = r0 + i / HI < R ? __half2float(w[(size_t) (r0 + i / HI) * HI + i % HI]) : 0.f;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = *tm_base;

    if (warp == 8)
    {
        // ------------------------------------------------------------------ TMA: the Q block once, then the key tiles
        if (lane == 0 && nct > 0)
        {
            bar_expect_tx(bq, Q_BYTES);
            for (int a = 0; a < 2; ++a) tma_load_2d(su32(smem + SM_Q + a * (Q_BYTES / 2)), &tmq, a * 64, r0 * HI, bq);
            for (int i = 0; i < nct; ++i)
            {
                const int st = i % NSTG;
                if (i >= NSTG) bar_wait(bke + 8 * st, ((i / NSTG) - 1) & 1);
                bar_expect_tx(bkf + 8 * st, K_BYTES);
                const int kr = (t0 + i) * KT, prow = bt ? __ldg(bt + kr / epp) * epp + kr % epp : kr;
                for (int a = 0; a < 2; ++a)
                    tma_load_2d(su32(smem + SM_K + st * K_BYTES + a * (K_BYTES / 2)), &tmk, a * 64, prow, bkf + 8 * st);
            }
        }
        __syncwarp();
    }
    else if (warp == 9)
    {
        // ------------------------------------------------------------------ MMA: D[b] = K_tile . Q_blk^T
        if (lane == 0 && nct > 0)
        {
            const uint32_t idesc = (1u << 4) | ((uint32_t) (256 >> 3) << 17) | ((uint32_t) (KT >> 4) << 24);
            const uint32_t qa = su32(smem + SM_Q);
            bar_wait(bq, 0);
            for (int i = 0; i < nct; ++i)
            {
                const int st = i % NSTG, b = i & 1;
                bar_wait(bkf + 8 * st, (i / NSTG) & 1);
                if (i >= 2) bar_wait(bde + 8 * b, ((i >> 1) - 1) & 1);
                tc_fence_after();
                const uint32_t ka = su32(smem + SM_K + st * K_BYTES);
                #pragma unroll
                for (int s = 0; s < 8; ++s)
                    mma_ss(tmem + b * 256, sdesc(ka + (s >> 2) * (K_BYTES / 2) + (s & 3) * 32, 16, 1024, 2),
                           sdesc(qa + (s >> 2) * (Q_BYTES / 2) + (s & 3) * 32, 16, 1024, 2), idesc, s > 0);
                tc_commit(bke + 8 * st);
                tc_commit(bdf + 8 * b);
            }
        }
        __syncwarp();
    }
    else
    {
        // ------------------------------------------------------------------ epilogue (warps 0-7)
        const int qd = warp & 3, qh = warp >> 2, key_l = qd * 32 + lane;
        const uint32_t lanes = (uint32_t) (qd * 32) << 16;
        int bnd[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { const int r = r0 + qh * 4 + j; bnd[j] = r < R ? min(q_pos0 + r + 1, bound_max) : 0; }
        for (int i = 0; i < nct; ++i)
        {
            const int b = i & 1, key = (t0 + i) * KT + key_l;
            bar_wait(bdf + 8 * b, (i >> 1) & 1);
            tc_fence_after();
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int q = qh * 4 + j;
                uint32_t v[32];
                TLD32(v, tmem + lanes + b * 256 + q * 32);
                tld_wait();
                float acc = 0.f;
                #pragma unroll
                for (int h = 0; h < HI; ++h) acc = fmaf(fmaxf(__uint_as_float(v[h]), 0.f), W[q * HI + h], acc);
                acc *= scale;
                if (r0 + q < R && key < T)
                    scores[(size_t) (r0 + q) * s_stride + key] = __float2half(key < bnd[j] ? acc : -INFINITY);
            }
            tc_fence_before();
            __syncwarp();
            if (lane == 0) bar_arrive(bde + 8 * b);
        }
        // tiles past the block's largest bound: -inf only
        for (int i = tc1; i < t1; ++i)
        {
            const int key = i * KT + key_l;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int q = qh * 4 + j;
                if (r0 + q < R && key < T) scores[(size_t) (r0 + q) * s_stride + key] = __float2half(-INFINITY);
            }
        }
    }
    __syncthreads();
    if (warp == 9)
    {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
    }
#endif
}

PFN_cuTensorMapEncodeTiled_v12000 encode_fn()
{
    static PFN_cuTensorMapEncodeTiled_v12000 f = [] {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        TORCH_CHECK(p && q == cudaDriverEntryPointSuccess, "hidx: cuTensorMapEncodeTiled not found");
        return (PFN_cuTensorMapEncodeTiled_v12000) p;
    }();
    return f;
}
// fp16 [rows][128] row-major in boxes of 64 dims x box_rows rows, 128-byte swizzle (UMMA K-major SW128); OOB rows read 0
CUtensorMap make_map(const void* ptr, int rows, int box_rows)
{
    CUtensorMap m;
    cuuint64_t dims[2] = {(cuuint64_t) DI, (cuuint64_t) rows};
    cuuint64_t strides[1] = {(cuuint64_t) DI * 2};
    cuuint32_t box[2] = {64, (cuuint32_t) box_rows}, es[2] = {1, 1};
    const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*) ptr, dims, strides, box, es,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "hidx: cuTensorMapEncodeTiled failed ", (int) r);
    return m;
}
}  // namespace

// q (R, 32, 128) fp16 contiguous, w (R, 32) fp16; k (rows, 128) fp16 contiguous: the keys themselves (T = rows) or, with
// a block table bt (int32), the flat paged pool (key s at row bt[s / epp] * epp + s % epp); T = the key count (-1: rows);
// scores (>= R, s_stride) fp16, columns [0, T) written; tpc = 128-key tiles per CTA
void hidx_scores(at::Tensor q, at::Tensor w, at::Tensor k, int64_t q_pos0, int64_t bound_max, double scale, at::Tensor scores, int64_t tpc,
                 c10::optional<at::Tensor> bt, int64_t epp, int64_t T_)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    const int R = (int) q.size(0), T = T_ >= 0 ? (int) T_ : (int) k.size(0);
    TORCH_CHECK(q.size(1) == HI && q.size(2) == DI && k.size(1) == DI && q.is_contiguous() && k.is_contiguous() && w.is_contiguous() &&
                scores.stride(1) == 1 && scores.size(1) >= T && (!bt || (epp > 0 && epp % KT == 0 && bt->is_contiguous())), "hidx: shapes");
    static bool attr = [] { cudaFuncSetAttribute(hidx_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); return true; }();
    (void) attr;
    const CUtensorMap tmq = make_map(q.data_ptr(), R * HI, QB * HI), tmk = make_map(k.data_ptr(), (int) k.size(0), KT);
    const int nt = (T + KT - 1) / KT;
    const dim3 grid((unsigned) ((R + QB - 1) / QB), (unsigned) ((nt + tpc - 1) / tpc));
    hidx_kernel<<<grid, NT, SMEM, at::cuda::getCurrentCUDAStream().stream()>>>(tmq, tmk, (const half*) w.data_ptr(),
        (half*) scores.data_ptr(), R, T, (int) q_pos0, (int) bound_max, (int) scores.stride(0), (float) scale, (int) tpc,
        bt ? (const int*) bt->data_ptr() : nullptr, (int) epp);
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hidx_scores", &hidx_scores); }
