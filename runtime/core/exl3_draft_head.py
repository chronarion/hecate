"""EXL3_DRAFT_HEAD_I4=1: an int4 copy of the LM head for the MTP draft passes only (the verify forward keeps the int8
head). One kernel computes the logits of up to 8 rows and their argmax: every warp owns whole 64-row groups (no split-K,
no scratch), scales its rows, reduces the per-row maximum across its lanes and posts it with a 64-bit atomicMax
(ordered float bits << 32 | ~row, so ties pick the lower id); a second tiny kernel decodes the ids and clears the keys.
The copy is built from the stock int8 head at the first draft call: per-row MSE-optimal uniform int4 (code - 8, fp32
scale), packed in the moe4 fragment order ([64-row group][64-wide k-block][strip][lane] x 16 bytes). ~0.48 GB of HBM."""
import os
import torch

_K_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
namespace {
constexpr int SPW = 4, WPB = 8, PD = 2;
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
__device__ __forceinline__ unsigned long long key_of(float v, int row)
{
    unsigned u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) u << 32) | (unsigned long long) (0xFFFFFFFFu - (unsigned) row);
}
__global__ void __launch_bounds__(WPB * 32) head_i4_argmax_kernel(const uint4* __restrict__ w, const float* __restrict__ scale,
    const half* __restrict__ x, unsigned long long* keys, int M, int N, int K, int V)
{
    const int lane = threadIdx.x % 32, KB = K / 64, groups = N / (16 * SPW);
    const int nw = gridDim.x * WPB;
    const int xn = lane / 4, kq = (lane % 4) * 2; const bool live = xn < M;
    const uint32_t* xr = (const uint32_t*) (x + (size_t) (live ? xn : 0) * K + kq);
    const int col = (lane % 4) * 2;
    for (int g = blockIdx.x * WPB + threadIdx.x / 32; g < groups; g += nw)
    {
        const uint4* base = w + (size_t) g * KB * 32 * SPW + lane;
        uint4 ring[PD][SPW];
        #pragma unroll
        for (int i = 0; i < PD; ++i)
            #pragma unroll
            for (int s = 0; s < SPW; ++s) ring[i][s] = __ldg(base + (size_t) (i * SPW + s) * 32);
        float c[SPW][4];
        #pragma unroll
        for (int s = 0; s < SPW; ++s) c[s][0] = c[s][1] = c[s][2] = c[s][3] = 0.f;
        for (int kb0 = 0; kb0 < KB; kb0 += PD)
        {
            #pragma unroll
            for (int j = 0; j < PD; ++j)
            {
                const int kb = kb0 + j;
                uint32_t bb[8];
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                {
                    const int k0 = kb * 64 + t * 16;
                    bb[2 * t] = live ? __ldg(xr + k0 / 2) : 0u; bb[2 * t + 1] = live ? __ldg(xr + k0 / 2 + 4) : 0u;
                }
                const bool more = kb + PD < KB;
                #pragma unroll
                for (int s = 0; s < SPW; ++s)
                {
                    const uint32_t ww[4] = {ring[j][s].x, ring[j][s].y, ring[j][s].z, ring[j][s].w};
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) { uint32_t a[4]; dec8(ww[t], a); mma16816(c[s], a, bb[2 * t], bb[2 * t + 1]); }
                    if (more) ring[j][s] = __ldg(base + (size_t) ((kb + PD) * SPW + s) * 32);
                }
            }
        }
        // per lane: best of its 8 values for rows col and col + 1, then across the 8 lanes sharing lane % 4
        unsigned long long b0 = 0ull, b1 = 0ull;
        #pragma unroll
        for (int s = 0; s < SPW; ++s)
        {
            const int r0 = g * SPW * 16 + s * 16 + lane / 4, r1 = r0 + 8;
            const float s0 = scale[r0], s1 = scale[r1];
            if (r0 < V) { b0 = max(b0, key_of(c[s][0] * s0, r0)); b1 = max(b1, key_of(c[s][1] * s0, r0)); }
            if (r1 < V) { b0 = max(b0, key_of(c[s][2] * s1, r1)); b1 = max(b1, key_of(c[s][3] * s1, r1)); }
        }
        #pragma unroll
        for (int o = 4; o < 32; o <<= 1)
        {
            b0 = max(b0, __shfl_xor_sync(0xffffffffu, b0, o));
            b1 = max(b1, __shfl_xor_sync(0xffffffffu, b1, o));
        }
        if (lane < 4)
        {
            if (col < M && b0) atomicMax(keys + col, b0);
            if (col + 1 < M && b1) atomicMax(keys + col + 1, b1);
        }
    }
}
__global__ void head_i4_decode_kernel(unsigned long long* keys, int64_t* ids, int M)
{
    const int m = threadIdx.x;
    if (m < M) { ids[m] = (int64_t) (0xFFFFFFFFu - (unsigned) (keys[m] & 0xFFFFFFFFull)); keys[m] = 0ull; }
}
}  // namespace

void head_i4_argmax(at::Tensor w, at::Tensor scale, at::Tensor x, at::Tensor keys, at::Tensor ids, int64_t N, int64_t V)
{
    const int K = (int) x.size(-1), M = (int) (x.numel() / K);
    TORCH_CHECK(M >= 1 && M <= 8 && K % 64 == 0 && N % 64 == 0 && x.dtype() == at::kHalf && x.is_contiguous(), "head_i4: shape");
    auto st = at::cuda::getCurrentCUDAStream().stream();
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int groups = (int) (N / 64), blocks = std::min((groups + WPB - 1) / WPB, sms * 2);
    head_i4_argmax_kernel<<<blocks, WPB * 32, 0, st>>>((const uint4*) w.data_ptr(), (const float*) scale.data_ptr(),
        (const half*) x.data_ptr(), (unsigned long long*) keys.data_ptr(), M, (int) N, K, (int) V);
    head_i4_decode_kernel<<<1, 32, 0, st>>>((unsigned long long*) keys.data_ptr(), ids.data_ptr<int64_t>(), M);
}
"""

_ext = None
_HEAD = {}
_U4F = (2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.7, 4.0)


def _ext_mod():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        bd = os.path.join(os.environ.get("EXL3_TIER_BUILD", "/tmp"), "build_draft_head"); os.makedirs(bd, exist_ok=True)
        _ext = load_inline(name="exl3_draft_head_i4", cpp_sources="void head_i4_argmax(at::Tensor w, at::Tensor scale, at::Tensor x, at::Tensor keys, at::Tensor ids, int64_t N, int64_t V);",
                           cuda_sources=_K_SRC, functions=["head_i4_argmax"], build_directory=bd,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"])
    return _ext


def _quant_u4(W):
    """uniform int4 (code - 8) per row, MSE-best scale over rms / f -> codes uint8 [N, K], scale fp32 [N]"""
    s0 = W.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12); best = bq = bs = None
    for f in _U4F:
        s = s0 / f; q = torch.round(W / s).clamp(-8, 7); e = (W - q * s).pow(2).sum(1, keepdim=True)
        if best is None: best, bq, bs = e, q, s
        else: m = e < best; best = torch.where(m, e, best); bq = torch.where(m, q, bq); bs = torch.where(m, s, bs)
    return (bq + 8).to(torch.uint8), bs.view(-1).float()


def _pack_u4(codes):
    N, K = codes.shape
    c = codes.view(N // 64, 4, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 8, 6, 2).contiguous().view(-1, 2)
    return (c[:, 0] | (c[:, 1] << 4)).contiguous()


def build(lm):
    """int4 copy of the LM head's stock int8 weights (inner.w8, dense_i8 fragment order), chunked by 64-row groups"""
    import hecate_runtime as O
    inner = lm.inner
    q8, s8, _, _, spw = inner.w8
    N, K = s8.numel(), lm.in_features
    dev = q8.device
    per_group = 64 * K                      # int8 codes per 64 output rows (whole dense_i8 groups when spw * 16 divides 64)
    packs, scales = [], []
    CH = 8192
    for r0 in range(0, N, CH):
        r1 = min(N, r0 + CH)
        W = O.unpack_i8(q8[r0 * K:r1 * K], s8[r0:r1], r1 - r0, K, spw).float()
        c, s = _quant_u4(W)
        packs.append(_pack_u4(c)); scales.append(s)
        del W, c
    w = torch.cat(packs).view(torch.uint8).contiguous()
    scale = torch.cat(scales).contiguous()
    keys = torch.zeros(8, dtype=torch.int64, device=dev)
    ids = torch.zeros(8, dtype=torch.int64, device=dev)
    return {"w": w, "scale": scale, "keys": keys, "ids": ids, "N": N, "dev": dev}


def argmax(lm, x2d, V):
    h = _HEAD.get(id(lm))
    if h is None:
        h = _HEAD[id(lm)] = build(lm)
        print(f" -- draft head: int4 copy of the LM head, {h['w'].numel() / 1e9:.2f} GB", flush=True)
    M = x2d.shape[0]
    # a fresh id tensor per call: the generator keeps every draft column and concatenates them after the last pass
    ids = torch.empty(M, dtype=torch.int64, device=x2d.device)
    _ext_mod().head_i4_argmax(h["w"], h["scale"], x2d, h["keys"], ids, h["N"], V)
    return ids


def install():
    from exllamav3.architecture.glm_moe_dsa_mtp import GlmMoeDsaMTPModel
    orig = GlmMoeDsaMTPModel.sample_from_state

    def sample_from_state(self, state, params, _o=orig):
        if params.get("export_draft_conf") or params.get("sparse_head") is True:
            return _o(self, state, params)
        tgt = self.attached_model()
        if tgt is None or tgt.loaded_tp:
            return _o(self, state, params)
        lm = tgt.modules[tgt.logit_layer_idx]
        x = lm.prepare_for_device(state, params)
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] > 8 or getattr(lm.inner, "w8", None) is None:
            return _o(self, state, params)
        if x2.dtype != torch.half or not x2.is_contiguous():
            x2 = x2.half().contiguous()
        ids = argmax(lm, x2, tgt.config.vocab_size)
        return ids.view(*state.shape[:-1])
    GlmMoeDsaMTPModel.sample_from_state = sample_from_state
    print(" -- hecate: int4 draft LM head (EXL3_DRAFT_HEAD_I4)", flush=True)
