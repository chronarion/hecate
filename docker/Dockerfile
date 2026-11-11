# Hecate: Hybrid Inference and Quantization for DGX Station (ExLlamaV3)
# GLM-5.3 (753B MoE) at batch 1 on NVIDIA DGX Station GB300 (GB300 + RTX PRO 6000 Blackwell + Grace), arm64 / CUDA 13.
#
# Stages:
#   base     ExLlamaV3 v1.6.5 + TabbyAPI (pinned) built for sm_103a (GB300) and sm_120a (RTX PRO 6000)
#   patched  C++ / CUDA / Python patches to the ExLlamaV3 extension, applied in order (docker/patches/exllamav3)
#   runtime  Hecate runtime modules (tier split, int8 base / CUDA graphs / router / draft head, MLA kernels), defaults
#
#   docker build -t hecate:latest -f docker/Dockerfile .
# RUNTIME_BASE=<image> builds only the runtime stage on top of an existing patched image (development)
ARG RUNTIME_BASE=patched
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS base

ARG EXLLAMAV3_REF=v1.6.5
ARG TABBY_REF=564f8fbb231a2d7e3b339574c85ae7d54ffd9618
ARG TORCH_VERSION=2.14.1
ARG TORCH_CUDA_ARCH_LIST="10.3a;12.0a"
ARG MAX_JOBS=32

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_HTTP_TIMEOUT=600 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev git build-essential ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.12 /opt/venv
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu130
# TabbyAPI gates uvloop to x86_64, but its main.py imports it unconditionally
RUN git clone https://github.com/theroyallab/tabbyAPI /opt/tabbyAPI && cd /opt/tabbyAPI && git checkout ${TABBY_REF} \
    && uv pip install . uvloop
RUN git clone --depth 1 --branch ${EXLLAMAV3_REF} https://github.com/turboderp-org/exllamav3 /opt/exllamav3 \
    && cd /opt/exllamav3 && uv pip install -r requirements.txt && uv pip install --no-build-isolation .

# ---------------------------------------------------------------------------------------------------------------------
FROM base AS patched
ARG MAX_JOBS=32
ENV SP=/opt/venv/lib/python3.12/site-packages/exllamav3 EXT=/opt/exllamav3/exllamav3/exllamav3_ext
COPY docker/patches/exllamav3/ /tmp/p/
# 1. Python-side base versions the later patch scripts edit (MLA kernels, routing helper, generator); the runtime stage
#    replaces the MLA kernels and the generator with their final versions
RUN cp /tmp/p/mla_triton.py /tmp/p/bc_mla.py ${SP}/modules/attention_fn/ \
 && cp /tmp/p/block_sparse_mlp_routing.py ${SP}/modules/ && cp /tmp/p/generator.py ${SP}/generator/
# 2. MoE coop kernels: 2 blocks/SM, guard-free ring-2 main loop, 3-bit mul1 affine fold (fp32 MMA); FP16 / pinned-expert
#    hooks used by the int8 path below
RUN MOE_COOP_MINB=2 MOE_COOP_LOOP2=1 MOE_COOP_AFFINE=1 python3 /tmp/p/apply_cpp.py ${EXT} \
 && python3 /tmp/p/apply_fp16_linear.py ${EXT} ${SP} && python3 /tmp/p/apply_pin_experts.py ${SP}
# 3. int8 decode GEMVs for attention / shared expert / LM head / dense MLPs (EXL3_I8_LINEARS)
RUN mkdir -p /tmp/i8 && cp /tmp/p/apply_i8_linear.py /tmp/p/dense_i8.cu /tmp/i8/ \
 && python3 /tmp/i8/apply_i8_linear.py ${EXT} ${SP} /tmp/i8
# 4. thread-block-cluster int8 GEMVs, fused q_a+kv_a and shared
#    gate+up+act kernels, MLA split floor knob (EXL3_MLA_SPLIT_TILES)
RUN python3 /tmp/p/fold/apply_fold.py ${EXT} /tmp/p/fold
# 5. module graphs run eagerly while an outer (whole-forward) CUDA graph is being captured
RUN python3 /tmp/p/vg_eager.py ${EXT}
# 6. tcgen05 sparse attention for the MLA prefill and the long-context decode (EXL3_HMLA_PREFILL, EXL3_HMLA_DECODE)
RUN python3 /tmp/p/hmla/apply_hmla.py ${EXT} ${SP} /tmp/p/hmla
# 7. single-sequence sparse (DSA) decode on the batched kernels, so long-context steps replay as CUDA graphs
RUN python3 /tmp/p/sparse/apply_dsa_multirow.py ${EXT}
# 8. fused MLA decode cache append (norm + rope + Q6 quant + scatter) on a side branch of the graph (EXL3_MLA_KVQ)
RUN python3 /tmp/p/kvq/apply_kvq.py ${EXT} /tmp/p/kvq
# 9. RMSNorm rows of <= 2 x blockDim float4 columns in registers, loads first (EXL3_NORM2, bit-identical)
RUN python3 /tmp/p/norm/apply_norm2.py ${EXT}
# hecate (EXL3_NORM_PDL): rms_norm_kernel as a programmatic dependent (waits for x, lets the next GEMV launch early)
RUN python3 /tmp/p/norm/apply_normpdl.py ${EXT}
# 10. the decode MoE combine fused into the next block's input norm (EXL3_TIER_COMBINE_NORM, runtime/tier/cn.cu); with
#     EXL3_CN_PDL the tier's MoE layers skip the zero-filled routed sum that combine_norm_pdl never reads
RUN python3 /tmp/p/cn/apply_cn.py ${SP} && python3 /tmp/p/cn/apply_cnfill.py ${SP}
# 11. multi-block radix select for the DSA indexer top-k of few-row selections (decode) and prefill slabs
#     (EXL3_DSA_TOPK_RADIX)
RUN python3 /tmp/p/topk/apply_topk_radix.py ${EXT} /tmp/p/topk
# 12. tcgen05 prefill DSA indexer scoring (runtime/core/hidx.cu, JIT-built; EXL3_HIDX)
RUN python3 /tmp/p/hidx/apply_hidx.py ${SP}/modules/mla_attn.py /opt/hecate/core/hidx.cu
# 13. the tcgen05 sparse attention (6) for decode batches of several sequences (EXL3_HMLA_BATCH); after 7, which adds
#     the second call site
RUN python3 /tmp/p/hmla/apply_hmla_batch.py ${EXT}
RUN cd /opt/exllamav3 && (MAX_JOBS=${MAX_JOBS} python setup.py build_ext > /tmp/build_ext.log 2>&1 \
      || (grep -B2 -A8 -E " error|Error" /tmp/build_ext.log | head -80; exit 1)) \
 && cp build/lib.linux-aarch64-cpython-312/exllamav3_ext.cpython-312-aarch64-linux-gnu.so /opt/venv/lib/python3.12/site-packages/ \
 && find ${SP} -name __pycache__ -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime
ENV SP=/opt/venv/lib/python3.12/site-packages/exllamav3
# MLA / DSA decode kernels (alignment-hinted AOT compile, one-load combine, DSA top-k selection)
COPY runtime/mla/bc_mla.py runtime/mla/mla_triton.py runtime/mla/dsa_triton.py ${SP}/modules/attention_fn/
# generator patches: MTP gap merge, draft-graph hook, decode share during prefill, event-loop yield (tools/dev/make_patches.py)
COPY runtime/patched/generator.py runtime/patched/async_generator.py ${SP}/generator/
COPY runtime/tier/ /opt/hecate/tier/
COPY runtime/core/ /opt/hecate/core/
COPY runtime/warmup.py runtime/entrypoint.sh runtime/render_config.py /opt/hecate/
COPY config/ /opt/hecate/config/
# Validated defaults (every switch is documented in docs/configuration.md; the report measures this exact set)
ENV EXL3_MOE4_SRC=/opt/hecate/tier/moe4 EXL3_TIER_BUILD=/root/.cache/hecate-tier TRITON_CACHE_DIR=/root/.cache/triton \
    EXL3_PIN_EXPERTS=/opt/hecate/tier/pin-none.json \
    EXL3_RTX_GRAPH=1 EXL3_RTX_STAGE=1 EXL3_RTX_MOE3=1 EXL3_HMLA_PREFILL=1 EXL3_HMLA_DECODE=1 EXL3_HMLA_NS=16 EXL3_PRELAUNCH=1 \
    EXL3_TIER_CPU=1 EXL3_TIER_CPU_THREADS=60 EXL3_TIER_CPU_CORE0=12 EXL3_TIER_PREFILL_MOE4=1 EXL3_TIER_SH_FAST=1 \
    EXL3_EMBED_GPU=1 EXL3_DRAFT_HEAD_I4=1 EXL3_DRAFT_COUPLE=1 EXL3_COPY_DRAFT=1 EXL3_DRAFT_MAX_ROWS=8 EXL3_MTP_MERGE_GAP=1 \
    EXL3_UNFOLD_BV=32 EXL3_UNFOLD_CFG=2,4 EXL3_UNFOLD_BK=128 EXL3_ROUTER_DET=0 EXL3_BT_FINE=1 EXL3_INT8_SQ_MAX_M=4 \
    EXL3_MOE_COOP_KSPLIT=2 EXL3_MOE_COOP_WIDE=-1 EXL3_BC_MLA=1 EXL3_MOE_SHARED_COOP=0 \
    EXL3_I8_LINEARS='self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj|indexer\.wq_b)|mlp\.shared_experts\.|^lm_head$|^model\.layers\.[0-2]\.mlp\.(gate|up|down)_proj$' \
    EXL3_FP16_LINEARS='self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj|indexer\.wq_b)' \
    EXL3_SKIP_TRELLIS=1 \
    EXL3_RTX_CAP_ROWS=8192 EXL3_PREFILL_GROUP=24 EXL3_RTX_PREFILL_ROWS=64 EXL3_PREFILL_DECODE_SHARE=0.3 \
    EXL3_RTX_PREFILL_MOE3P=1 EXL3_TIER_PREFILL_MOE4P=1 EXL3_PREFILL_GB300_ROWS=2048 \
    EXL3_GRAPHS=1 EXL3_VERIFY_GRAPH=1 EXL3_VERIFY_GRAPH_EVERY=2000 EXL3_DRAFT_GRAPH=1 EXL3_VERIFY_GRAPH_BATCH=1 EXL3_DRAFT_GRAPH_BATCH=1 EXL3_BATCH_ONESYNC=1 EXL3_HMLA_BATCH=1 EXL3_ROUTER_FUSED=1 \
    EXL3_DSA_MULTIROW=1 EXL3_VERIFY_GRAPH_SPARSE=1 EXL3_MLA_KVQ=3 EXL3_TIER_HSIDE=1 EXL3_NORM2=1 EXL3_TIER_COMBINE_NORM=1 \
    EXL3_DSA_TOPK_SPLIT=0 EXL3_MLA_SPLIT_TPS=2 EXL3_MLA_SPLIT_TILES=1 EXL3_MLA_COMBINE_DSPLIT=16 EXL3_COMBINE_CFG=8,1 \
    EXL3_MLA_BLOCK_H_Q=4 EXL3_DSA_SCORE_TILE=262144 EXL3_HIDX=1 EXL3_DSA_TOPK_RADIX=16384 \
    EXL3_NORM_ROUTE=1 EXL3_NORM_ROUTE_HANDOFF=1 EXL3_WARMUP=1 \
    EXL3_I8_CL_CS_LK=4 EXL3_I8_CL_CS_DUAL=8 EXL3_I8_PDL=1 EXL3_NORM_PDL=1 EXL3_TIER_SH_MOE4=1 \
    EXL3_CN_PDL=1 EXL3_MOE4_XF=1 EXL3_RTX_STAGE1=1
RUN find ${SP} -name __pycache__ -prune -exec rm -rf {} +
WORKDIR /opt/tabbyAPI
EXPOSE 8103
ENTRYPOINT ["sh", "/opt/hecate/entrypoint.sh"]
