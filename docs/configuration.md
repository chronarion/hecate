# Configuration

## Context and cache (`HECATE_*`, rendered into TabbyAPI's config at start)

| Variable | Default | Meaning |
|---|---|---|
| `HECATE_MAX_SEQ_LEN` | 262144 | longest single conversation in tokens (the model supports 1,048,576) |
| `HECATE_CACHE_TOKENS` | 524288 | KV page pool shared by all conversations (multiple of 256) |
| `HECATE_CACHE_MODE` | Q6 | KV cache format (ExLlamaV3 modes; Q6 measured equal to FP16 on KLD) |
| `HECATE_DRAFTS` | 3 | MTP draft tokens per step (3 measured fastest) |
| `HECATE_SYSMEM_KV_GB` | 200 | system-RAM tier behind the KV pool, in GB (0 = off; see "System-RAM KV tier" below) |
| `HECATE_MAX_BATCH` | 8 | requests that decode at the same time; more wait in the queue (see "Concurrency" below; 0 = ExLlamaV3's 128) |
| `HECATE_INTERACTIVE_RESERVE` | 2 | of those, slots that non-streamed requests leave free for streamed ones |
| `HECATE_PRIORITY` | `stream` | streamed requests start ahead of non-streamed ones; `off` = first come, first served |
| `HECATE_CHUNK` | 8192 | prefill chunk in tokens; 16384 cuts time to first token by another ~8% from 50K tokens up, for 2.6 GiB more GB300 memory while serving (about 55K tokens of KV cache) |
| `HECATE_PLACEMENT` | `tier-144-85-far27-mtp256-r2` | expert placement file in the weights' `placement/` folder |
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
| default | `tier-144-85-far27-mtp256-r2` | 0 | 512K | ~176 tok/s (short greedy bench) |
| +5.4 GB | `tier-140-85-far31-mtp256` | 4 | ~620K (622,592 tested) | ~131 tok/s (-5% vs the default then) |
| 1M | `tier-120-85-far51-mtp256` | 4 | 1M (16 GB HBM left) | ~100 tok/s (-27%); 375K-token prompt: 1,507 tok/s prefill, 67 tok/s decode |

The two larger profiles were measured on release 2, before the pipelined decode step (the default ran ~140 tok/s
then). Placements come from `tools/placement/kv_placement.py`. It moves the coldest hot experts to the head of the cold tier.
The default (`-r2`, release 1.9) ranks each layer's int4 experts by how often decode steps picked them over ~30K verify
steps of reasoning, code, long-context, tool-use, editing and chat traffic (`EXL3_ROUTE_LOG`,
`tools/placement/rerank_placement.py`); the 1.8 ranking (`tier-144-85-far27-mtp256`, 24 short prompts) put experts that
code and tool calls use often on the Grace band. The larger profiles still use the 1.8 ranking.
With `HECATE_RTX_I4=k`, the first k of those run as int4 on the RTX instead of on the Grace cores (same weights); the RTX
band then uses ExLlamaV3's cooperative MoE kernels instead of `moe3`. Placements change where experts run, not their
weights; the Grace band computes with 8-bit activations, so an expert that moves between HBM and the Grace computes slightly
differently (the default's KLD is measured: 0.0734 with `-r2` vs 0.0739 with the 1.8 ranking, same code).

### Concurrency

Hecate is built for a few conversations at a time. A decode step processes every running request's rows together; up to
8 rows it stays on a fast path (CUDA graphs, the fused MoE kernels, the RTX and Grace bands), and above 8 rows each step
falls back to a much slower path. So by default at most 8 requests decode at once (`HECATE_MAX_BATCH`) and the rest wait
in the queue: a waiting request starts late but then runs at full speed, instead of every request crawling at a few
tokens per second and running into client timeouts. MTP drafts take rows too: one conversation drafts 3 tokens per
step, two conversations 3 each, four conversations 1 each, eight none.

Streamed requests (chat front ends, coding agents) are treated as interactive and start before non-streamed ones (batch
jobs, evaluations), and batch requests may only fill `HECATE_MAX_BATCH - HECATE_INTERACTIVE_RESERVE` slots, so an
interactive turn starts within a step even when a batch client keeps the queue full. With only batch traffic the reserve
leaves those slots idle; set `HECATE_INTERACTIVE_RESERVE=0` for a batch-only server. A gateway in front of Hecate
should also limit each client's parallel requests and must not retry timed-out generations (the retry runs the whole
generation again). A request whose client disconnects is cancelled, also while it waits in the queue.

Prompts are processed one 8K chunk per scheduler pass, so a request that arrives while long prompts are being read is
admitted within one chunk. Interactive prompts go first; batch prompts take turns, so a batch finishes its prompts together
and starts generating as one wide step. While requests are generating, prompt processing is paced so that the generating
requests keep a share of the GPU: `EXL3_PREFILL_DECODE_SHARE` (0.3) in general, and `HECATE_DECODE_SHARE_INTERACTIVE`
(0.8) while an interactive request is generating and the waiting prompt belongs to a batch request, so an agent keeps its
speed while a batch client sends long prompts. An interactive prompt starts without waiting when no other interactive
request is generating. Measured with `tools/bench/overload_check.py --prompt-tokens 30000` (24 batch requests with
unique 30K-token prompts, a streamed request every 45 s): agents got
their first token within 2.0 s (1.9: up to 38 s) and decoded at 30-93 tok/s (1.9: 17-35), and the batch finished sooner
(median 229 s, 1.9: 258 s).

Streamed responses carry usage on every chunk when the request asks for `stream_options.continuous_usage_stats` (vLLM's
extension; TabbyAPI otherwise reports usage only at the end of a request). Benchmarks that count tokens from usage, such
as llm-inference-bench, need it to measure requests that run longer than their window.

### System-RAM KV tier

`HECATE_SYSMEM_KV_GB` (200 GB by default since 1.9; 0 turns it off) sizes ExLlamaV3's second-tier page cache (TabbyAPI `memory.sysmem_kv_cache`). Finished
256-token pages that leave the GPU pool are copied into page-locked system memory, and a later prompt that starts with
them gets them copied back instead of prefilled again. The GPU pool still bounds what the conversations in flight can
hold; the tier keeps the ones that come back, such as agent sessions and chats pushed out by other traffic. The whole
tier is allocated and locked at start, on top of Hecate's own host memory, so the host must have it free.

Measured on the DGX Station GB300: an 82K-token prompt pushed out of the GPU pool by ~1.3M tokens of other prompts came
back in 1.0 s instead of 16.8 s of prefill, with the same answers (`tools/bench/kvtier.py`). A 200 GB tier holds ~4M
tokens, 8x the default pool.

## Runtime switches (validated defaults are baked into the image)

| Variable | Default | |
|---|---|---|
| `EXL3_VERIFY_GRAPH` / `EXL3_DRAFT_GRAPH` | 1 / 1 | decode-verify forward and MTP draft passes as CUDA graphs |
| `EXL3_DSA_MULTIROW` / `EXL3_VERIFY_GRAPH_SPARSE` | 1 / 1 | contexts above 2,048 tokens (the DSA regime): a single conversation's sparse attention reads its scan widths and causal bounds from device state instead of host scalars, so its decode steps replay as CUDA graphs too (keyed by regime); 0 / 0 = those steps run eagerly |
| `EXL3_MLA_KVQ` | 3 | decode attention with the quantized latent cache: one kernel does the kv_a norm, the rope key, the Q6 group quantization and the paged cache write, on a side branch of the graph that runs while the q_a norm and q_b GEMV run (the absorb kernel applies the query rope); bit-identical cache contents. 3 (default) = also the indexer-key stages of the 21 full-indexer layers on that branch, 1 = the cache append only; 0 = the separate staging, rope, quant and scatter kernels |
| `EXL3_MLA_BLOCK_ROWS_B` | 16 | decode attention with several requests (dense regime, contexts under 2,048 tokens): each program takes 16 (row, head) pairs, as a single request's verify step does, instead of 4 heads of one row; fewer programs re-read the same cached keys. 6-8 requests at 1-1.5K tokens of context: 15-20% shorter decode steps. 0 = the 4-head block |
| `EXL3_VERIFY_GRAPH_BATCH` / `EXL3_DRAFT_GRAPH_BATCH` / `EXL3_BATCH_ONESYNC` | 1 / 1 / 1 | decode steps of 2..8 requests replay from CUDA graphs too (one per batch size, rows per request, block-table width and attention regime; the regime is the batch's longest request), and so do the MTP draft passes of 2..4 requests; the samplers of all requests launch before one readback. Graph and eager logits agree within the eager-vs-eager noise floor (`EXL3_VERIFY_GRAPH_CHECK`). 2-4 requests: 3-5% shorter steps; 6-8 requests: 2-4% |
| `EXL3_HMLA_BATCH` | 1 | contexts above 2,048 tokens with several requests: the tcgen05 sparse attention kernel (until now one request only) runs the whole batch, each row reading its own request's block table; bit-identical per row to one-request launches (`tools/dev/test_hmla_batch.py`). 2-8 requests at ~16K tokens of context: steps 2.7-3.0 ms (10%) shorter |
| `EXL3_PRELAUNCH` | 1 | pipelined decode step for a single conversation: the next MTP drafts + verify are launched as soon as a step's accepted tokens are known, before its bookkeeping (needs the two graphs above) |
| `EXL3_ROUTER_FUSED` | 1 | router GEMV + sigmoid/bias top-8 as one kernel (each top-k round is two warp reductions over order-preserving keys; the same selections as a sequential scan) |
| `EXL3_NORM_ROUTE` | 1 | decode MoE layers (2..8 rows): the post-attention RMSNorm and the router as one kernel of 4 thread-block clusters x 16 CTAs (`runtime/core/nr4_kernel.cuh`). Each CTA owns the column sets of two of the norm kernel's warps and repeats their partial sums, so the residual and the MoE input are bit-identical to the separate norm; it stages its cluster's 64-expert slice of the gate while the norm runs, the partial logits run on tensor cores and are summed through distributed shared memory, and the clusters' top-8 candidates are merged by the last cluster leader in the fused router's key order (the same selections up to exact near-ties). 0 = the norm kernel + `EXL3_ROUTER_FUSED` |
| `EXL3_NORM_ROUTE_HANDOFF` | 1 | with `EXL3_NORM_ROUTE`: the same kernel writes the layer's band slot (normed rows, selection, weights) and releases the RTX / Grace ISSUE flags, so the bands start without the handoff kernel; 0 = the handoff kernel (`EXL3_TIER_HSIDE`) |
| `EXL3_DRAFT_HEAD_I4` | 1 | int4 copy of the LM head for MTP drafts (+0.48 GB) |
| `EXL3_DRAFT_COUPLE` | 1 | sampled requests (temperature > 0): each MTP draft is picked with the same random noise the sampler will use at that position (ExLlamaV3 samples by adding Gumbel noise to the logits and taking the largest), so a draft matches the sampled token far more often when the draft and the model agree on the likely candidates. The sampler itself is untouched, so the output is exactly what it was; only the speed changes. Greedy requests are unaffected. Measured with `tools/bench/specmix.py` at temperature 1: +10% tokens per step, +6% decode speed |
| `EXL3_COPY_DRAFT` | 1 | one conversation whose newest `EXL3_COPY_MIN` (16) tokens repeat earlier context (code edits, tool arguments that quote a file, quoted documents) verifies the up to `EXL3_COPY_MAX` (7) tokens that followed that earlier occurrence instead of the MTP drafts (the MTP draft passes still run, so their cache stays complete); the output is unchanged. Measured with `tools/bench/specmix.py` (agent turns that write files, edits): 3.8 -> 5.5 tokens per step, +22% decode speed; no change on reasoning, code and long-context questions |
| `EXL3_TIER_SH_FAST` | 1 | shared expert straight into the hot-tier accumulator |
| `EXL3_TIER_HSIDE` | 1 | decode MoE layers: the handoff kernel (MoE input and routing copied into the RTX / Grace mailboxes, then the release flag) runs on a side stream, so the GB300's shared expert and hot experts start without waiting for it; joined before the combine. `EXL3_TIER_HANDOFF_THREADS` (256) sizes its single block |
| `EXL3_TIER_COMBINE_NORM` | 1 | decode MoE layers whose output goes straight to the next block's input norm leave their combine (wait for the RTX / Grace bands, add the partials) to that norm: one kernel adds the partials, updates the residual and normalizes (runtime/tier/cn.cu); bit-identical |
| `EXL3_NORM2` | 1 | RMSNorm rows of up to two float4 columns per thread (the 6144-wide norms) keep their values in registers between the reduction and the scaling, with every load issued first; bit-identical; 0 = the original paths |
| `EXL3_EMBED_GPU` | 1 | token embedding in HBM (+1.9 GB; avoids a host sync per draft pass) |
| `EXL3_RTX_STAGE` | 1 | copy each layer's mailbox into RTX VRAM before the RTX kernels |
| `EXL3_RTX_MOE3` | 1 | run the RTX band on the moe3 trellis MoE kernels; 0 = ExLlamaV3's cooperative MoE kernels |
| `EXL3_RTX_PREFILL_MOE3P` | 1 | prefill of the RTX band (chunks over 64 rows) on the moe3p kernels: grouped GEMMs that decode the 3-bit trellis inside the MMA loop; 0 = each picked expert reconstructed to FP16 per chunk + grouped GEMMs |
| `EXL3_TIER_PREFILL_MOE4P` | 1 | prefill of the int4 tiers (chunks over 64 rows) on the moe4p kernels: grouped GEMMs on the stored int4 fragments, the Grace tier read in place over C2C; 0 = bf16 dequant per chunk + grouped GEMMs |
| `EXL3_TIER_SH_MOE4` | 1 | decode MoE layers run the shared expert inside moe4: its int8 gate / up units join phase A's stream-K list (the dense GEMV's decode), act with weight 1, its down units phase B, instead of a separate gate+up and down GEMV before the hot experts; 0 = the separate GEMVs |
| `EXL3_I8_CL_CS_LK` | 4 | cluster size of the decode int8 GEMVs with K >= 8192 (o_proj); 0 = the formula (~2 CTAs per SM: 3) |
| `EXL3_I8_CL_CS_DUAL` | 8 | cluster size of the fused q_a + kv_a GEMV; 0 = the formula (7) |
| `EXL3_I8_PDL` | 1 | the decode int8 GEMVs launch as programmatic dependents: after the combine-norm (4 CTAs, waiting on the bands), the q_a norm (4 CTAs) and nr4 (64 CTAs) they become resident early, fill their weight rings and wait (griddepcontrol.wait) for their input; each GEMV lets the next kernel launch after its main loop |
| `EXL3_NORM_PDL` | 1 | RMSNorm launches as a programmatic dependent and lets the next kernel launch at its entry |
| `EXL3_CN_PDL` | 1 | with `EXL3_TIER_COMBINE_NORM`: the combine-norm launches as a programmatic dependent of `moe4` phase B (which lets it launch at B's start): its CTAs become resident as B's retire, wait for the RTX DONE and the Grace CPU_DONE themselves (the separate Grace wait kernel is folded in), load both partials, the residual and the norm weight while B drains, and read the hot band only after `griddepcontrol.wait`; the MoE layer no longer zero-fills a routed sum it never reads. Bit-identical |
| `EXL3_MOE4_XF` | 1 | `moe4` reads its activations in MMA-fragment order: nr4 (`EXL3_NORM_ROUTE_HANDOFF`) also writes the MoE input that way and the act kernel writes act that way, so each 2 KB weight unit's activation operand is two 16-byte loads instead of eight 4-byte ones (phase B always uses the fragment-order act). Bit-identical |
| `EXL3_RTX_STAGE1` | 1 | the RTX graph waits for each layer's ISSUE and copies the layer's slot (rows, selection, weights) into VRAM in one kernel, instead of a wait kernel and three copy-engine nodes (~14 µs from ISSUE to the first RTX kernel, mostly node transitions; ~5 µs now) |
| `EXL3_UNFOLD_PDL` / `EXL3_NR4_PDLW` | 0 / 0 | the MLA unfold kernel lets o_proj launch at its start, and nr4 launches as a programmatic dependent of o_proj (constants loaded before `griddepcontrol.wait`); off: in the 1.7 A/B the pair cost 2.6% on short prompts and gained ~2% at 23K |
| `EXL3_I8_UNPACK_FUSED` | 1 | prefill unpacks the int8 linears (attention, shared experts, LM head) with one fused kernel; 0 = torch ops (same values) |
| `EXL3_I8_GEMV_ROWS` | 64 | int8 linears run the decode GEMV in 8-row chunks up to this many rows (decode batches at concurrency 16-32); more rows unpack the weights |
| `EXL3_RTX_P3_NPOS` / `EXL3_TIER_P4_NPOS` | 16384 / 16384 | expert picks per moe3p / moe4p call (scratch ~45 KB per pick on the RTX / GB300) |
| `EXL3_PREFILL_GB300_ROWS` | 2048 | prompt chunks above this many rows run every routed expert on the GB300's tensor cores: the int4 tiers on the tcgen05 grouped GEMM `moe4t` (gate and up in one CTA, SiLU epilogue), the RTX band's 3-bit trellis on `moe3t`, read in place over C2C from a pinned copy in Grace memory (~85 GB host RAM; +0.7 GB HBM scratch), on a second stream; the RTX idles. Shorter chunks keep the split path above; 0 = off |
| `EXL3_HMLA_PREFILL` | 1 | sparse (DSA) attention of prompt chunks on the tcgen05 kernel (`docker/patches/exllamav3/hmla`); 0 = the Triton gather kernel |
| `EXL3_HMLA_DECODE` / `EXL3_HMLA_NS` | 1 / 16 | sparse (DSA) attention of decode steps (contexts above 2,048 tokens) on the tcgen05 kernel, each query row's key list split across NS CTAs; 0 = the Triton split + combine. Shorter contexts keep the Triton kernels |
| `EXL3_ROUTE_LOG` | unset | file to append every decode step's expert selection to (`tools/placement/route_stats.py`, `rerank_placement.py`) |
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
