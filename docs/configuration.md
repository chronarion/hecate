# Configuration

## Context and cache (`HECATE_*`, rendered into TabbyAPI's config at start)

| Variable | Default | Meaning |
|---|---|---|
| `HECATE_MAX_SEQ_LEN` | 262144 | longest single conversation in tokens (the model supports 1,048,576) |
| `HECATE_CACHE_TOKENS` | 524288 | KV page pool shared by all conversations (multiple of 256) |
| `HECATE_CACHE_MODE` | Q6 | KV cache format (ExLlamaV3 modes; Q6 measured equal to FP16 on KLD) |
| `HECATE_DRAFTS` | 3 | MTP draft tokens per step (3 measured fastest) |
| `HECATE_CHUNK` | 8192 | prefill chunk in tokens; 16384 cuts time to first token by another ~8% from 50K tokens up, for 2.6 GiB more GB300 memory while serving (about 55K tokens of KV cache) |
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
| default | `tier-144-85-far27-mtp256` | 0 | 512K | ~148 tok/s |
| +5.4 GB | `tier-140-85-far31-mtp256` | 4 | ~620K (622,592 tested) | ~131 tok/s (-5% vs the default then) |
| 1M | `tier-120-85-far51-mtp256` | 4 | 1M (16 GB HBM left) | ~100 tok/s (-27%); 375K-token prompt: 1,507 tok/s prefill, 67 tok/s decode |

The two larger profiles were measured on release 2, before the pipelined decode step (the default ran ~140 tok/s
then). Placements come from `tools/placement/kv_placement.py`. It moves the coldest hot experts to the head of the cold tier.
With `HECATE_RTX_I4=k`, the first k of those run as int4 on the RTX instead of on the Grace cores (same weights); the RTX
band then uses ExLlamaV3's cooperative MoE kernels instead of `moe3`. Every placement here is lossless: they change where
experts run, not their weights.

## Runtime switches (validated defaults are baked into the image)

| Variable | Default | |
|---|---|---|
| `EXL3_VERIFY_GRAPH` / `EXL3_DRAFT_GRAPH` | 1 / 1 | decode-verify forward and MTP draft passes as CUDA graphs |
| `EXL3_DSA_MULTIROW` / `EXL3_VERIFY_GRAPH_SPARSE` | 1 / 1 | contexts above 2,048 tokens (the DSA regime): a single conversation's sparse attention reads its scan widths and causal bounds from device state instead of host scalars, so its decode steps replay as CUDA graphs too (keyed by regime); 0 / 0 = those steps run eagerly |
| `EXL3_MLA_KVQ` | 3 | decode attention with the quantized latent cache: one kernel does the kv_a norm, the rope key, the Q6 group quantization and the paged cache write, on a side branch of the graph that runs while the q_a norm and q_b GEMV run (the absorb kernel applies the query rope); bit-identical cache contents. 3 (default) = also the indexer-key stages of the 21 full-indexer layers on that branch, 1 = the cache append only; 0 = the separate staging, rope, quant and scatter kernels |
| `EXL3_PRELAUNCH` | 1 | pipelined decode step for a single conversation: the next MTP drafts + verify are launched as soon as a step's accepted tokens are known, before its bookkeeping (needs the two graphs above) |
| `EXL3_ROUTER_FUSED` | 1 | router GEMV + sigmoid/bias top-8 as one kernel (each top-k round is two warp reductions over order-preserving keys; the same selections as a sequential scan) |
| `EXL3_NORM_ROUTE` | 1 | decode MoE layers (2..8 rows): the post-attention RMSNorm and the router as one kernel of 4 thread-block clusters x 16 CTAs (`runtime/core/nr4_kernel.cuh`). Each CTA owns the column sets of two of the norm kernel's warps and repeats their partial sums, so the residual and the MoE input are bit-identical to the separate norm; it stages its cluster's 64-expert slice of the gate while the norm runs, the partial logits run on tensor cores and are summed through distributed shared memory, and the clusters' top-8 candidates are merged by the last cluster leader in the fused router's key order (the same selections up to exact near-ties). 0 = the norm kernel + `EXL3_ROUTER_FUSED` |
| `EXL3_NORM_ROUTE_HANDOFF` | 1 | with `EXL3_NORM_ROUTE`: the same kernel writes the layer's band slot (normed rows, selection, weights) and releases the RTX / Grace ISSUE flags, so the bands start without the handoff kernel; 0 = the handoff kernel (`EXL3_TIER_HSIDE`) |
| `EXL3_DRAFT_HEAD_I4` | 1 | int4 copy of the LM head for MTP drafts (+0.48 GB) |
| `EXL3_TIER_SH_FAST` | 1 | shared expert straight into the hot-tier accumulator |
| `EXL3_TIER_HSIDE` | 1 | decode MoE layers: the handoff kernel (MoE input and routing copied into the RTX / Grace mailboxes, then the release flag) runs on a side stream, so the GB300's shared expert and hot experts start without waiting for it; joined before the combine. `EXL3_TIER_HANDOFF_THREADS` (256) sizes its single block |
| `EXL3_TIER_COMBINE_NORM` | 1 | decode MoE layers whose output goes straight to the next block's input norm leave their combine (wait for the RTX / Grace bands, add the partials) to that norm: one kernel adds the partials, updates the residual and normalizes (runtime/tier/cn.cu); bit-identical |
| `EXL3_NORM2` | 1 | RMSNorm rows of up to two float4 columns per thread (the 6144-wide norms) keep their values in registers between the reduction and the scaling, with every load issued first; bit-identical; 0 = the original paths |
| `EXL3_EMBED_GPU` | 1 | token embedding in HBM (+1.9 GB; avoids a host sync per draft pass) |
| `EXL3_RTX_STAGE` | 1 | copy each layer's mailbox into RTX VRAM before the RTX kernels |
| `EXL3_RTX_MOE3` | 1 | run the RTX band on the moe3 trellis MoE kernels; 0 = ExLlamaV3's cooperative MoE kernels |
| `EXL3_RTX_PREFILL_MOE3P` | 1 | prefill of the RTX band (chunks over 64 rows) on the moe3p kernels: grouped GEMMs that decode the 3-bit trellis inside the MMA loop; 0 = each picked expert reconstructed to FP16 per chunk + grouped GEMMs |
| `EXL3_TIER_PREFILL_MOE4P` | 1 | prefill of the int4 tiers (chunks over 64 rows) on the moe4p kernels: grouped GEMMs on the stored int4 fragments, the Grace tier read in place over C2C; 0 = bf16 dequant per chunk + grouped GEMMs |
| `EXL3_I8_UNPACK_FUSED` | 1 | prefill unpacks the int8 linears (attention, shared experts, LM head) with one fused kernel; 0 = torch ops (same values) |
| `EXL3_I8_GEMV_ROWS` | 64 | int8 linears run the decode GEMV in 8-row chunks up to this many rows (decode batches at concurrency 16-32); more rows unpack the weights |
| `EXL3_RTX_P3_NPOS` / `EXL3_TIER_P4_NPOS` | 16384 / 16384 | expert picks per moe3p / moe4p call (scratch ~45 KB per pick on the RTX / GB300) |
| `EXL3_PREFILL_GB300_ROWS` | 2048 | prompt chunks above this many rows run every routed expert on the GB300's tensor cores: the int4 tiers on the tcgen05 grouped GEMM `moe4t` (gate and up in one CTA, SiLU epilogue), the RTX band's 3-bit trellis on `moe3t`, read in place over C2C from a pinned copy in Grace memory (~85 GB host RAM; +0.7 GB HBM scratch), on a second stream; the RTX idles. Shorter chunks keep the split path above; 0 = off |
| `EXL3_HMLA_PREFILL` | 1 | sparse (DSA) attention of prompt chunks on the tcgen05 kernel (`docker/patches/exllamav3/hmla`); 0 = the Triton gather kernel |
| `EXL3_HMLA_DECODE` / `EXL3_HMLA_NS` | 1 / 16 | sparse (DSA) attention of decode steps (contexts above 2,048 tokens) on the tcgen05 kernel, each query row's key list split across NS CTAs; 0 = the Triton split + combine. Shorter contexts keep the Triton kernels |
| `EXL3_MLA_BLOCK_H_Q`, `EXL3_MLA_SPLIT_TILES`, `EXL3_MLA_COMBINE_DSPLIT`, `EXL3_COMBINE_CFG` | 4, 1, 16, 8,1 | MLA decode launch shapes |
| `EXL3_DSA_TOPK_SPLIT` | 0 | DSA indexer top-k on one block per row over the live scan width |
| `EXL3_DSA_TOPK_RADIX` | 16384 | DSA indexer top-k over rows of at least this many keys as a multi-block radix select (4,096-key slices across CTAs: histogram passes, then ordered compaction; rows of up to 3 live slices in one CTA), for up to 16 rows in the decode graphs and up to 256 rows per eager prefill slab; the same index lists as the one-block kernel; 0 = off |
| `EXL3_DSA_SCORE_TILE` | 262144 | prompt slabs (256 rows) score up to this many keys per indexer call and select with one top-k per row; longer contexts fall back to tiles merged by torch.topk (the merge is exact). ExLlamaV3's default was 32768 |
| `EXL3_HIDX` | 1 | prefill DSA indexer scoring on the tcgen05 kernel (`runtime/core/hidx.cu`, JIT-built at the first long prompt): 8 queries x 128-key tiles per UMMA chain with the head reduction in the epilogue; bit-identical scores to the Triton kernel; 0 = Triton |

## Hardware assumptions

- **GPUs:** the GB300 is `cuda:0` and the RTX is `cuda:1` inside the container. compose sets `CUDA_VISIBLE_DEVICES` from
  the two UUIDs.
- **Grace cores:** 60 of 72 are reserved for the cold tier (they spin while decoding and back off when idle). The server's
  own threads stay on cores 0-11.
- **Driver:** 615 or newer (CDMM). On 595, host memory is NUMA-mapped HBM and the GPUs cannot share a CUDA process.
