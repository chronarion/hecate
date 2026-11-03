"""EXL3_NORM_ROUTE=1: the post-attention RMSNorm (fused residual add) and the GLM "dots" router of a MoE block as one
kernel of 4 thread-block clusters x 16 CTAs for decode rows (2..8 rows), instead of rms_norm_res_in followed by the
fused router (EXL3_ROUTER_FUSED). The residual update and the normed input are bit-identical to rms_norm_res_in; the
router logits are summed in another fp32 order (near-ties only). Kernel: nr4_kernel.cuh. The norm hands
(cfg, selected, weights) to the MoE's routing call through params. With the tier split (EXL3_NORM_ROUTE_HANDOFF=1), the
kernel also writes the layer's band slot and releases the RTX / Grace ISSUE flags (tier_submit then skips moe4_handoff)."""
import os
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "nr4_kernel.cuh"

void norm_route(at::Tensor y, at::Tensor r, at::Tensor w, at::Tensor x2, at::Tensor gt, c10::optional<at::Tensor> bias,
                at::Tensor sel, at::Tensor rw, at::Tensor cand, at::Tensor ctr, double scale, double eps, double cbias, double cscale,
                std::vector<int64_t> ho)
{
    using namespace hecate_nr4;
    const int M = (int) (y.numel() / DIM);
    TORCH_CHECK(y.size(-1) == DIM && M >= 2 && M <= MAXM && y.dtype() == at::kFloat && r.dtype() == at::kFloat &&
                r.numel() == y.numel() && x2.dtype() == at::kHalf && x2.numel() == y.numel() && gt.dtype() == at::kHalf &&
                gt.size(0) == NE && gt.size(1) == DIM && w.numel() == DIM && y.is_contiguous() && r.is_contiguous() &&
                x2.is_contiguous() && gt.is_contiguous() && w.is_contiguous() && sel.dtype() == at::kLong && rw.dtype() == at::kHalf &&
                sel.numel() >= M * TOPK && rw.numel() >= M * TOPK && cand.numel() >= G * MAXM * TOPK * 4 && ctr.numel() >= 2 * CS &&
                cand.dtype() == at::kInt && ctr.dtype() == at::kInt, "norm_route: shape / dtype");
    const bool wbf = w.dtype() == at::kBFloat16;
    TORCH_CHECK(wbf || w.dtype() == at::kHalf, "norm_route: weight must be fp16 or bf16");
    static bool init = false;
    if (!init)
    {
        for (auto k : {(const void*) nr4_kernel<false>, (const void*) nr4_kernel<true>})
        {
            C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }
        init = true;
    }
    Params p{(const float*) y.data_ptr(), (float*) r.data_ptr(), w.data_ptr(), (half*) x2.data_ptr(), (const half*) gt.data_ptr(),
             bias.has_value() ? (const float*) bias->data_ptr() : nullptr, (int64_t*) sel.data_ptr(), (half*) rw.data_ptr(),
             (int4*) cand.data_ptr(), (int*) ctr.data_ptr(), M, (float) eps, (float) cbias, (float) cscale, (float) scale,
             nullptr, nullptr, nullptr, nullptr, 0u, nullptr, 0u, nullptr, 0u, nullptr, nullptr};
    if (ho.size() == 10 || ho.size() == 11)   // handoff slot (y, sel, rw), ISSUE flag + gen, Grace header + layer, Grace
    {                                         // flag + gen, RTX DONE (+ moe4's fragment-order x, EXL3_MOE4_XF)
        p.hy = (half*) ho[0]; p.hs = (int64_t*) ho[1]; p.hw = (half*) ho[2]; p.hflag = (unsigned*) ho[3]; p.hgen = (unsigned) ho[4];
        p.hmeta = (unsigned*) ho[5]; p.hm0 = (unsigned) ho[6]; p.hflag2 = (unsigned*) ho[7]; p.hgen2 = (unsigned) ho[8];
        p.hzero = (unsigned*) ho[9];
        if (!ho[5]) p.hmeta = nullptr;
        if (!ho[7]) p.hflag2 = nullptr;
        if (!ho[9]) p.hzero = nullptr;
        if (ho.size() == 11) p.x2f = (half*) ho[10];
    }
    cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = dim3(CS * G); cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = SMEM;
#ifdef NR4_PDLW
    cfg.stream = at::cuda::getCurrentCUDAStream().stream(); cfg.attrs = at; cfg.numAttrs = 2;   // EXL3_NR4_PDLW
#else
    cfg.stream = at::cuda::getCurrentCUDAStream().stream(); cfg.attrs = at; cfg.numAttrs = 1;
#endif
    if (wbf) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, nr4_kernel<true>, p));
    else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, nr4_kernel<false>, p));
}
"""
_DECL = ("void norm_route(at::Tensor y, at::Tensor r, at::Tensor w, at::Tensor x2, at::Tensor gt, c10::optional<at::Tensor> bias, "
         "at::Tensor sel, at::Tensor rw, at::Tensor cand, at::Tensor ctr, double scale, double eps, double cbias, double cscale, "
         "std::vector<int64_t> ho);")

_HANDOFF = os.environ.get("EXL3_NORM_ROUTE_HANDOFF", "1") == "1"   # with the tier split: write the band slot, release ISSUE
_ext = None
_SCR = {}       # device -> (candidates, counters): persistent (baked into graphs), counters zero between calls


def _ext_mod():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        bd = os.path.join(os.environ.get("EXL3_TIER_BUILD", "/tmp"), "build_normroute"); os.makedirs(bd, exist_ok=True)
        _pw = os.environ.get("EXL3_NR4_PDLW", "0") == "1"
        _ext = load_inline(name="exl3_normroute4f" + ("w" if _pw else ""), cpp_sources=_DECL, cuda_sources=_HOST,
                           functions=["norm_route"], build_directory=bd,
                           extra_include_paths=[_DIR],
                           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_103a,code=sm_103a"]
                           + (["-DNR4_PDL"] if os.environ.get("EXL3_I8_PDL", "0") == "1" else [])
                           + (["-DNR4_PDLW"] if _pw else []))
    return _ext


def install():
    import exllamav3.modules.block_sparse_mlp as bsm
    import exllamav3.modules.block_sparse_mlp_routing as R
    from exllamav3.modules.rmsnorm import RMSNorm
    from exllamav3.modules.transformer import TransformerBlock

    orig_route = bsm.routing_dots

    def routing_dots(bsz, cfg, y, params, _o=orig_route):
        pre = params.get("_nr_route") if params is not None else None
        if pre is not None and pre[0] is cfg and pre[3] == bsz:
            del params["_nr_route"]
            return pre[1], pre[2]
        return _o(bsz, cfg, y, params)
    bsm.routing_dots = routing_dots

    ti = TransformerBlock.__init__

    def __init__(self, *a, _ti=ti, **k):
        _ti(self, *a, **k)
        mlp, nrm = getattr(self, "mlp", None), getattr(self, "mlp_norm", None)
        if isinstance(mlp, bsm.BlockSparseMLP) and isinstance(nrm, RMSNorm) and getattr(mlp, "routing_fn", None) is routing_dots:
            nrm._nr_mlp = mlp
    TransformerBlock.__init__ = __init__

    of = RMSNorm.forward

    def forward(self, x, params, out_dtype=None, residual=None, residual_in=None, _of=of):
        mlp = getattr(self, "_nr_mlp", None)
        if mlp is not None and residual_in is not None and residual is None and params is not None:
            rows = x.numel() // x.shape[-1]
            cfg = mlp.routing_cfg
            if (2 <= rows <= 8 and cfg is not None and x.shape[-1] == 6144 and x.dtype == torch.float and
                    residual_in.dtype == torch.float and (out_dtype or self.out_dtype) == torch.half and
                    x.is_contiguous() and residual_in.is_contiguous() and self.weight is not None and
                    self.weight.dtype in (torch.half, torch.bfloat16) and cfg.num_experts == 256 and
                    cfg.num_experts_per_tok == 8 and getattr(cfg, "router_bias", None) is None and
                    mlp.router_pre_norm is None and not mlp.alt_residual_channel and not params.get("activate_all_experts")):
                gt = R._gate_t(cfg)
                # the scratch buffers are allocated on the first eager call (never inside a graph capture's pool)
                if gt.dtype == torch.half and gt.is_contiguous() and (x.device in _SCR or not torch.cuda.is_current_stream_capturing()):
                    b = getattr(cfg, "_nr_bias", None)
                    if b is None and cfg.e_score_correction_bias is not None:
                        b = cfg._nr_bias = cfg.e_score_correction_bias.float().contiguous().view(-1)
                    _, sel, rw = R._routing_buffers(cfg, rows, x.device)
                    scr = _SCR.get(x.device)
                    if scr is None:
                        scr = _SCR[x.device] = (torch.zeros(4 * 8 * 8 * 4, dtype=torch.int, device=x.device),
                                                torch.zeros(64, dtype=torch.int, device=x.device))
                    y = torch.empty_like(x, dtype=torch.half)
                    ext_ = _ext_mod()
                    hf = getattr(mlp, "nr_handoff", None) if _HANDOFF else None
                    ho = hf(rows) if hf is not None else None        # tier split: this kernel also does the band handoff
                    ext_.norm_route(x.view(rows, -1), residual_in.view(rows, -1), self.weight, y.view(rows, -1), gt, b,
                                    sel, rw, scr[0], scr[1], float(cfg.routed_scaling_factor),
                                    float(self.rms_norm_eps), float(self.constant_bias), float(self.constant_scale),
                                    list(ho) if ho is not None else [])
                    params["_nr_route"] = (cfg, sel, rw, rows)
                    return y
        return _of(self, x, params, out_dtype=out_dtype, residual=residual, residual_in=residual_in)
    RMSNorm.forward = forward
    print(" -- hecate: post-attention norm + router as one 4 x 16-CTA cluster kernel for 2..8 rows (EXL3_NORM_ROUTE)", flush=True)
