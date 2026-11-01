"""EXL3_ROUTER_FUSED=1: GLM's sigmoid / bias top-k router ("dots") for 2..8 rows as one kernel instead of a cuBLAS
GEMM + split-K reduce + the top-k kernel. 32 blocks of 8 warps: each block stages the rows' x in shared memory, each
warp computes one expert's logits for every row (fp16 weights from the transposed gate, fp32 accumulation); the last
block to finish (counter, self-resetting) picks per row the top-k of sigmoid(logit) + bias and writes the selected ids
and the normalized, scaled weights into the module's usual routing buffers. Selection keys in fp32 (the stock path
rounds logits and the mean-centered bias to fp16), so near-ties may resolve differently (KLD-neutral or better)."""
import os
import torch

_K_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
namespace {
constexpr int WPB = 8, MAXR = 8, MAXE = 256, TOPK_MAX = 8, KS = 4;
// block b: experts [(b % EB) * WPB, +WPB), k slice b / EB of KS; partial logits added into logits[r][e] (fp32 atomics);
// the last block selects per row and zeroes logits + counter for the next call (self-cleaning, graph-safe)
__global__ void __launch_bounds__(WPB * 32) router_k(const half* __restrict__ x, const half* __restrict__ wt,
    const float* __restrict__ bias, float* logits, int* ctr, int64_t* sel, half* rw, int R, int K, int E, int topk,
    float scale)
{
    extern __shared__ __align__(16) half xs[];        // [R][KC] (this block's k slice)
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    const int EB = (E + WPB - 1) / WPB, eb = blockIdx.x % EB, ks = blockIdx.x / EB, KC = K / KS, k0 = ks * KC;
    for (int i = threadIdx.x * 8; i < R * KC; i += blockDim.x * 8)
    {
        const int r = i / KC, kk = i % KC;
        *(uint4*) (xs + i) = *(const uint4*) (x + (size_t) r * K + k0 + kk);
    }
    __syncthreads();
    const int e = eb * WPB + wid;
    if (e < E)
    {
        float acc[MAXR];
        #pragma unroll
        for (int r = 0; r < MAXR; ++r) acc[r] = 0.f;
        const half* w = wt + (size_t) e * K + k0;
        constexpr int LA = 6;                          // KC / 256 = 6 for K 6144: every load of the slice in flight
        for (int kb = lane * 8; kb < KC; kb += 256 * LA)
        {
            uint4 wv[LA];
            #pragma unroll
            for (int j = 0; j < LA; ++j) { const int k = kb + j * 256; wv[j] = k < KC ? __ldg((const uint4*) (w + k)) : make_uint4(0, 0, 0, 0); }
            #pragma unroll
            for (int j = 0; j < LA; ++j)
            {
                const int k = kb + j * 256;
                if (k >= KC) break;
                const half2* wh = (const half2*) &wv[j];
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    if (r >= R) break;
                    const uint4 xv = *(const uint4*) (xs + r * KC + k);
                    const half2* xh = (const half2*) &xv;
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) { const float2 a = __half22float2(wh[q]), b = __half22float2(xh[q]); acc[r] += a.x * b.x + a.y * b.y; }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            if (r >= R) break;
            float a = acc[r];
            #pragma unroll
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) atomicAdd(logits + r * MAXE + e, a);
        }
    }
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) last = atomicAdd(ctr, 1) == (int) gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (wid < R)
    {
        const int r = wid;
        float key[MAXE / 32], act[MAXE / 32];
        #pragma unroll
        for (int i = 0; i < MAXE / 32; ++i)
        {
            const int ee = lane + 32 * i;
            if (ee < E)
            {
                const float l = __ldcg(logits + r * MAXE + ee);
                logits[r * MAXE + ee] = 0.f;
                act[i] = 1.f / (1.f + __expf(-l));
                key[i] = act[i] + (bias ? bias[ee] : 0.f);
            }
            else { act[i] = 0.f; key[i] = -1e30f; }
        }
        float wsel[TOPK_MAX]; int isel[TOPK_MAX]; float sum = 0.f;
        for (int k = 0; k < topk; ++k)
        {
            float bk = -1e30f; int bi = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < MAXE / 32; ++i)
                if (key[i] > bk || (key[i] == bk && lane + 32 * i < bi)) { bk = key[i]; bi = lane + 32 * i; }
            #pragma unroll
            for (int o = 16; o; o >>= 1)
            {
                const float ok = __shfl_xor_sync(0xffffffffu, bk, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
            }
            float a = 0.f;
            if ((bi & 31) == lane) { const int i = bi >> 5;
                #pragma unroll
                for (int j = 0; j < MAXE / 32; ++j) if (j == i) { a = act[j]; key[j] = -1e30f; } }
            a = __shfl_sync(0xffffffffu, a, bi & 31);
            wsel[k] = a; isel[k] = bi; sum += a;
        }
        if (lane < topk)
        {
            float a = 0.f; int id = 0;
            for (int k = 0; k < TOPK_MAX; ++k) if (k == lane) { a = wsel[k]; id = isel[k]; }
            sel[r * topk + lane] = id;
            rw[r * topk + lane] = __float2half_rn(a * scale / (sum + 1e-20f));
        }
    }
    if (threadIdx.x == 0) *ctr = 0;
}
}  // namespace

void router_fused(at::Tensor x, at::Tensor wt, c10::optional<at::Tensor> bias, at::Tensor logits, at::Tensor ctr,
                  at::Tensor sel, at::Tensor rw, int64_t topk, double scale)
{
    const int K = (int) x.size(-1), R = (int) (x.numel() / K), E = (int) wt.size(0);
    TORCH_CHECK(R >= 1 && R <= MAXR && E <= MAXE && K % (256 * KS) == 0 && topk <= TOPK_MAX && x.dtype() == at::kHalf &&
                wt.dtype() == at::kHalf && x.is_contiguous() && wt.is_contiguous(), "router_fused: shape / dtype");
    const float* bp = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    const int blocks = (E + WPB - 1) / WPB * KS;
    const size_t smem = (size_t) R * (K / KS) * sizeof(half);
    router_k<<<blocks, WPB * 32, smem, at::cuda::getCurrentCUDAStream().stream()>>>((const half*) x.data_ptr(),
        (const half*) wt.data_ptr(), bp, logits.data_ptr<float>(), ctr.data_ptr<int>(), sel.data_ptr<int64_t>(),
        (half*) rw.data_ptr(), R, K, E, (int) topk, (float) scale);
}
"""

_ext = None
_BUF = {}


def _ext_mod():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        bd = os.path.join(os.environ.get("EXL3_TIER_BUILD", "/tmp"), "build_router_fused"); os.makedirs(bd, exist_ok=True)
        _ext = load_inline(name="exl3_router_fused3", cpp_sources="void router_fused(at::Tensor x, at::Tensor wt, c10::optional<at::Tensor> bias, at::Tensor logits, at::Tensor ctr, at::Tensor sel, at::Tensor rw, int64_t topk, double scale);",
                           cuda_sources=_K_SRC, functions=["router_fused"], build_directory=bd,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a", "-gencode=arch=compute_120a,code=sm_120a"])
    return _ext


def install():
    import exllamav3.modules.block_sparse_mlp as bsm
    import exllamav3.modules.block_sparse_mlp_routing as R
    orig = bsm.routing_dots

    def routing_dots(bsz, cfg, y, params, _o=orig):
        if not (2 <= bsz <= 8) or params.get("activate_all_experts") or y.dtype != torch.half or \
                getattr(cfg, "router_bias", None) is not None or cfg.num_experts > 256:
            return _o(bsz, cfg, y, params)
        st = getattr(cfg, "_fused", None)
        if st is None:
            wt = R._gate_t(cfg).contiguous()
            b = cfg.e_score_correction_bias
            st = cfg._fused = {"wt": wt, "bias": b.float().contiguous().view(-1) if b is not None else None}
        dev = y.device
        buf = _BUF.get(dev)
        if buf is None:
            buf = _BUF[dev] = (torch.zeros(8, 256, dtype=torch.float, device=dev), torch.zeros(1, dtype=torch.int, device=dev))
        _, sel, rw = R._routing_buffers(cfg, bsz, dev)
        _ext_mod().router_fused(y.view(bsz, -1).contiguous(), st["wt"], st["bias"], buf[0], buf[1], sel, rw,
                                cfg.num_experts_per_tok, float(cfg.routed_scaling_factor))
        return sel, rw
    bsm.routing_dots = routing_dots
    print(" -- hecate: fused router for 2..8 rows (EXL3_ROUTER_FUSED)", flush=True)
