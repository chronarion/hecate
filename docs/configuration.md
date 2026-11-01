# Configuration

## Context and cache (`HECATE_*`, rendered into TabbyAPI's config at start)

| Variable | Default | Meaning |
|---|---|---|
| `HECATE_MAX_SEQ_LEN` | 262144 | longest single conversation in tokens (the model supports 1,048,576) |
| `HECATE_CACHE_TOKENS` | 524288 | KV page pool shared by all conversations (multiple of 256) |
| `HECATE_CACHE_MODE` | Q6 | KV cache format (ExLlamaV3 modes; Q6 measured equal to FP16 on KLD) |
| `HECATE_DRAFTS` | 3 | MTP draft tokens per step (3 measured fastest) |
| `HECATE_PLACEMENT` | `tier-144-85-far27-mtp256` | expert placement file in the weights' `placement/` folder |
| `HECATE_RTX_I4` | 0 | far experts per layer run as int4 on the RTX (`EXL3_TIER_RTX_I4`; the larger-KV profiles below use 4) |
| `HECATE_CPU_THREADS` / `HECATE_CPU_CORE0` | 60 / 12 | Grace cold-tier worker threads and their first core |
| `HECATE_CONFIG` | — | path to a complete TabbyAPI config.yml inside the container (overrides the above) |
| `HECATE_PORT` | 8103 | host port compose publishes the API on (`127.0.0.1:8103` keeps it local; no API key by default) |

The cache is one pool of pages. `HECATE_CACHE_TOKENS=1048576` with `HECATE_MAX_SEQ_LEN=1048576` serves one 1M-token
conversation, two 512K ones, or any mix that fits.

### Memory budget: KV against speed

A Q6 cache costs ~48 KB per token: 544 B per layer on 79 layers, plus the DSA indexer keys on 21 layers. 1 GB of GB300
HBM therefore holds ~20K tokens. The default profile fills HBM with 144 hot experts per layer and a 512K-token pool.
Larger pools come from moving experts out of HBM:

| Profile | `HECATE_PLACEMENT` | `HECATE_RTX_I4` | KV pool fits | Decode, c=1 (stock) |
|---|---|---|---|---|
| default | `tier-144-85-far27-mtp256` | 0 | 512K | ~140 tok/s |
| +5.4 GB | `tier-140-85-far31-mtp256` | 4 | ~620K (622,592 tested) | ~131 tok/s (-5%) |
| 1M | `tier-120-85-far51-mtp256` | 4 | 1M (16 GB HBM left) | ~100 tok/s (-27%); 375K-token prompt: 1,507 tok/s prefill, 67 tok/s decode |

Placements come from `tools/placement/kv_placement.py`. It moves the coldest hot experts to the head of the cold tier.
With `HECATE_RTX_I4=k`, the first k of those run as int4 on the RTX instead of on the Grace cores (same weights); the RTX
band then uses ExLlamaV3's cooperative MoE kernels instead of `moe3`. Every placement here is lossless: they change where
experts run, not their weights.

## Runtime switches (validated defaults are baked into the image)

| Variable | Default | |
|---|---|---|
| `EXL3_VERIFY_GRAPH` / `EXL3_DRAFT_GRAPH` | 1 / 1 | decode-verify forward and MTP draft passes as CUDA graphs (contexts ≤ 2048 tokens) |
| `EXL3_ROUTER_FUSED` | 1 | router GEMV + sigmoid/bias top-8 as one kernel |
| `EXL3_DRAFT_HEAD_I4` | 1 | int4 copy of the LM head for MTP drafts (+0.48 GB) |
| `EXL3_TIER_SH_FAST` | 1 | shared expert straight into the hot-tier accumulator |
| `EXL3_EMBED_GPU` | 1 | token embedding in HBM (+1.9 GB; avoids a host sync per draft pass) |
| `EXL3_RTX_STAGE` | 1 | copy each layer's mailbox into RTX VRAM before the RTX kernels |
| `EXL3_RTX_MOE3` | 1 | run the RTX band on the moe3 trellis MoE kernels; 0 = ExLlamaV3's cooperative MoE kernels |
| `EXL3_HMLA_PREFILL` | 1 | sparse (DSA) attention of prompt chunks on the tcgen05 kernel (`docker/patches/exllamav3/hmla`); 0 = the Triton gather kernel |
| `EXL3_HMLA_DECODE` / `EXL3_HMLA_NS` | 1 / 16 | sparse (DSA) attention of decode steps (contexts above 2,048 tokens) on the tcgen05 kernel, each query row's key list split across NS CTAs; 0 = the Triton split + combine. Shorter contexts keep the Triton kernels |
| `EXL3_MLA_BLOCK_H_Q`, `EXL3_MLA_SPLIT_TILES`, `EXL3_MLA_COMBINE_DSPLIT`, `EXL3_COMBINE_CFG` | 4, 1, 16, 8,1 | MLA decode launch shapes |
| `EXL3_DSA_TOPK_SPLIT` | 0 | DSA indexer top-k on one block per row over the live scan width |

## Hardware assumptions

- **GPUs:** the GB300 is `cuda:0` and the RTX is `cuda:1` inside the container. compose sets `CUDA_VISIBLE_DEVICES` from
  the two UUIDs.
- **Grace cores:** 60 of 72 are reserved for the cold tier (they spin while decoding and back off when idle). The server's
  own threads stay on cores 0-11.
- **Driver:** 615 or newer (CDMM). On 595, host memory is NUMA-mapped HBM and the GPUs cannot share a CUDA process.
