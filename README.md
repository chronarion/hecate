# Hecate: Hybrid Inference and Quantization for DGX Station (ExLlamaV3)

Hecate serves **GLM-5.3 (753B MoE)** on a single **NVIDIA DGX Station GB300** at concurrency-1 decode speeds of
**~140 tok/s** (stock model, MTP speculative decoding, 2 × 256K context). Its KL divergence to the BF16 model is lower
than the published 3-bit EXL3 quant's.

The model does not fit in any one memory of the machine. Hecate splits every MoE layer across all three processors at
once:
- the **GB300** (252 GB HBM3e);
- the **RTX PRO 6000 Blackwell** (96 GB GDDR7, PCIe, no P2P);
- the **72-core Grace CPU** (LPDDR5X, coherent over NVLink-C2C).

Each tier uses the weight format and kernel that suit its hardware. Hecate is built on
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [TabbyAPI](https://github.com/theroyallab/tabbyAPI): it is
a set of kernels, runtime modules and patches, and the result is an OpenAI-compatible server in one container.

Weights: [chronarion/GLM-5.3-Hecate](https://huggingface.co/chronarion/GLM-5.3-Hecate), derived only from the stock BF16
release, [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16).

## Results (DGX Station GB300, driver 615, CUDA 13)

| | Value |
|---|---|
| Decode, concurrency 1, 10 prompts × 3 passes, greedy, 400 tokens | **~140 tok/s** (stock; 138-143 across the release test runs) |
| Decode vs context (llm-inference-bench, C=1, mean of 3 runs) | 16K: 119 · 32K: 119 · 64K: 116 · 128K: 118 tok/s |
| One long prompt, then 300 tokens of decode | 27.6K: 1,930 tok/s prefill, 133 tok/s decode · 109K: 1,880 / 125 |
| Prefill (llm-inference-bench standalone, cold) | ~2,080 tok/s at 8K-32K, 2,010 at 64K, 1,900 at 128K |
| KL divergence vs BF16 (rtx6kpro protocol, 36,848 positions) | **0.0740** (published EXL3 3.0 bpw: 0.1013) |
| Context | 2 × 256K Q6 KV by default; a single 1M-token conversation with the long-context profile |

## Benchmarks

local-inference-lab `llm-inference-bench` v0.8.0 on the release image: 30 s decode windows, ignore_eos, 2048 max tokens,
the default 512K-token KV pool (cells that would not fit in it are skipped). Raw JSON and logs:
[`benchmarks/`](benchmarks/), invocations in [`tools/bench/llm_inference_bench.sh`](tools/bench/llm_inference_bench.sh).

**Decode, aggregate tok/s by context and concurrency** (mean of the runs that measured the cell; C=1 cells from 2-3 runs):

| Context | C=1 | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 135 | 186 | 113 | 240 | 68 | 122 |
| 4K | 121 | 162 | 113 | 239 | - | - |
| 16K | 119 | 175 | 111 | 240 | - | - |
| 32K | 119 | 168 | 109 | 237 | - | - |
| 64K | 116 | 158 | 107 | - | - | - |
| 128K | 118 | 158 | - | - | - | - |

Each request gets the aggregate divided by C. Hecate is tuned for one or two conversations: from 4 concurrent requests
it trims MTP drafts to stay on the 8-row decode fast path (4: one draft, 8: none), and from 16 it leaves that path.
Single C=1 cells vary by up to ±10% between runs, because MTP acceptance depends on the generated text.

**Prefill, standalone and cold** (one request, mean of 2 rounds; time to first token includes the first decode step):

| Prompt | 1K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill tok/s | 570 | 1,479 | 2,073 | 2,088 | 2,088 | 2,014 | 1,904 |
| Time to first token | 2.0 s | 2.8 s | 4.0 s | 7.8 s | 15.5 s | 32.0 s | 67.7 s |

Short prompts pay a fixed ~1.7 s, so time to first token is 2.0 s even at 1K tokens.

## What made it fast

Each gain was measured when the change was added: concurrency-1 decode, same session as the build before it. "Idea from"
credits public work the change builds on; "—" means it came out of profiling this stack.

| Change | What it does | Gain | Idea from |
|---|---|---|---|
| RTX PRO 6000 as an expert coprocessor | 85 mid-popularity experts per MoE layer run on the RTX at the same time as the GB300; inputs and partial sums cross through pinned host-memory mailboxes and flags (no P2P needed) | +22% | expert parallelism; activation-aware expert placement (MoE-Infinity) |
| Cold experts on the Grace CPU | the 27 least-used experts per layer are computed by 60 Grace cores (NEON int8 dot products, only the rows that picked them); the GPU never reads them over C2C | +11% (with the next row) | CPU-side expert compute: [Fiddler](https://arxiv.org/abs/2402.07033), [kTransformers](https://github.com/kvcache-ai/ktransformers) |
| RTX band as one CUDA graph per step | the RTX runs all 75 layers' expert work from one graph that waits on per-layer flags, so no host thread is on the critical path | (included above) | — |
| Fused router + MLA decode launch tuning + RTX mailbox staging + coop K-split | router GEMV + sigmoid/bias top-8 in one kernel (-2 ms of host time per step); MLA block_h 4 with 1-tile splits and a latent-split combine; one bulk copy of each layer's mailbox into RTX VRAM; K-split 2 for the RTX MoE kernel | +9.3% | MLA split re-tune and MoE K-split: [b12x](https://github.com/local-inference-lab/b12x) (RTX 6000 community) |
| int4 hot tier with a custom MoE kernel | the 144 most-used experts per layer as int4 in HBM, run by `moe4` (LOP3 int4 → fp16 MMA, stream-K, ~4.6 TB/s) | +7% | LOP3 dequant: FasterTransformer / [Marlin](https://github.com/IST-DASLab/marlin); [Stream-K](https://arxiv.org/abs/2301.03598); weights by [GPTQ](https://arxiv.org/abs/2210.17323) |
| int8 attention / shared expert / LM head | int8 copies straight from BF16, run by a LOP3-decoded int8 GEMV instead of the 3-bit trellis kernels | +6% | — |
| Whole decode-verify forward as one CUDA graph | ~1,900 kernel launches per step replayed from one graph (host enqueue 19 ms → 0.6 ms) | +4.1% | full-forward graphs in vLLM / SGLang / TensorRT-LLM; megakernel work ([Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [Mirage MPK](https://github.com/mirage-project/mirage)) |
| Thread-block-cluster int8 GEMVs + moe4 flush rework | int8 GEMVs reduce through distributed shared memory (no scratch, atomics or counters); moe4 drops its flush round trips | +3.7% | Hopper/Blackwell thread-block clusters (NVIDIA CUDA programming guide) |
| MTP draft passes as one CUDA graph | the 3 draft passes replay as one graph, with per-pass sequence lengths computed on the device | +3.6% | graphed draft loops in vLLM / TensorRT-LLM speculative decoding |
| MLA W_UV unfold split | more programs per head for the value up-projection | +3.6% | — |
| Alignment hints on the AOT-compiled MLA kernels | 16-byte divisibility hints the JIT would have added (MLA 73.5 → 52.5 us per layer) | +2.5% | Triton's divisibility specialization |
| MLA combine in one tile load | each split's (m, l) loaded as one masked tile | +2.0% | split-KV combine of [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) / [FlashMLA](https://github.com/deepseek-ai/FlashMLA) |
| MoE cooperative-kernel tweaks | 2 blocks per SM (+1.1%), guard-free ring-2 main loop (+4.8%), 3-bit affine fold with fp32 MMA (+2.1%) | +8% | — |
| DSA indexer top-k on one block over the live width | replaces a split + merge path chosen by the static width | +5.6% at 23K context | — |
| Trellis MoE kernel for the RTX (`moe3`) | the RTX band's 3-bit experts run stream-K with the decoded trellis tile as the MMA A operand, so the token rows fill the n8 side (one MMA per tile, no padded rows); weights stream through a cp.async ring; kernels chain with programmatic dependent launch. 80 → 55 µs per MoE layer | +1.5% | [Stream-K](https://arxiv.org/abs/2301.03598); EXL3 trellis codebook ([QTIP](https://arxiv.org/abs/2406.11235), ExLlamaV3) |
| tcgen05 sparse attention for prefill and long-context decode (`hmla`) | the DSA attention (contexts past 2,048 tokens) as one Blackwell tensor-core (UMMA / TMEM) kernel: all 64 heads per CTA, Q held in TMEM, the gathered Q6 rows dequantized once into swizzled shared memory, double-buffered K tiles, warp-specialized load / MMA / softmax roles. Prefill: one CTA per query row, 2.2x the Triton kernel; decode: each row's 2,048 selected keys split across 16 CTAs plus a combine. Shorter contexts keep the Triton kernels, which are faster there | +8-11% prefill at 27K-109K; decode +3-5% past 16K (attention 35 → 24 µs per layer) | [FlashMLA](https://github.com/deepseek-ai/FlashMLA) SM100 kernels (mechanics) |
| RTX prefill: in-place reconstruct | the RTX's prefill dequantizes each expert group straight into one reused stacked buffer instead of stacking copies | +15% prefill (8K chunks) | — |
| Smaller wins | token embedding in HBM (removes a host sync per draft), int4 draft LM head, MTP layer in int4, fused shared-expert gate + up + SiLU, shared expert straight into the hot-tier accumulator | +1-3% each | int4 draft-only LM head: [b12x](https://github.com/local-inference-lab/b12x) |

**Fidelity.** These changes made the model closer to BF16 than the published 3-bit quant, not just faster:
- the int4 tiers are GPTQ-requantized from BF16, calibrated on the states of the quantized model, layer by layer;
- the warm RTX tier is requantized from BF16 the same way;
- the calibration follows [brandonmusic](https://huggingface.co/brandonmusic)'s published TR3 recipe for GLM EXL3
  quants: the layers are rebuilt causally (each one calibrated on the output of the already-quantized layers before
  it), and each expert's down projection is calibrated on its reconstructed, quantized gate/up output;
- the attention projections, shared experts, dense MLPs and LM head are int8 straight from BF16; the router, DSA
  indexer, kv_b_proj and the MTP eh_proj stay FP16.

KLD is 0.0740, against 0.1013 for the published EXL3 3.0 bpw on the same windows.

### KLD compared with other GLM-5.3 quants

> **Note:** this KLD comparison is preliminary. The methodology is still early: the evaluation sets differ, and
> most numbers come from each author's own setup. Future work will rescore Hecate and the other quants (including
> NVFP4 / MXFP4 / int4) on one shared protocol and window set.

All values are mean KL(BF16 ‖ quant) over the full vocabulary, as published by each quant's author. The evaluation sets
differ, and that alone moves the number a lot: brandonmusic's TR3 3.0 bpw scores 0.098 on 16 windows but 0.038 on the
4-window "panel-v1". Compare rows within a group; across groups, read differences under ~30% as noise. Only the first
group shares one exact window set.

| Quant | bpw | KLD | Evaluation set |
|---|---|---|---|
| **Hecate** (this release) | 3.8² | **0.0740**¹ | ours: 18 windows, 36,848 positions (wiki, code, chat, math, zh, two 8K windows), verify-shape decode path |
| Infatoshi EXL3 3.0 bpw (scored by us) | 3.04 | 0.1013 | ours, same windows (against the BF16 of its own lineage) |
| brandonmusic EXL3 TR3 | 3.0 | 0.0979 | 16 sealed windows, 32,752 positions |
| vcruz305 EXL3 | 3.38 | 0.0948 | own set, 10,240 positions, 1,024-token sequences |
| bertholomus EXL3 | 3.04 | 0.109 (0.119 at 4 × 16K) | wikitext-2, 32 × 2,048 (`model_diff`) |
| drowzeys EXL3 | 2.75 | 0.124 | 32 × 2,048 (`model_diff`) |
| Mia-AiLab EXL3 TensorFold, as served | 2.75 | 0.124 | math, code, chat, wiki, tool calls |
| davidsyoung EXL3 TR3 | 3.42 / 3.25 | 0.024 / 0.026 | panel-v1: 4 windows, 8,188 positions (on the same panel brandonmusic's TR3 3.0 scores 0.038) |
| AesSedai GGUF Q5_K_M / Q4_K_M / IQ4_XS / IQ3_S | 5.96 / 4.98 / 3.90 / 3.03 | 0.020 / 0.037 / 0.102 / 0.275 | llama.cpp KLD |
| NVFP4 (nvidia, Inferact, RedHatAI, local-inference-lab), MXFP4 (AMD Quark, RedHatAI), AWQ int4 | ~4.5 | not published | the cards report task accuracy only |

¹ The release weights on the served runtime (prompt in 2048-token prefill chunks, then 5-row verify-shape decode, FP16
KV cache); full report in [`benchmarks/kld/report.md`](benchmarks/kld/report.md).

² Every stored weight counted: 359 GB over 753B parameters, including the int8 attention / shared experts / LM head
and the unquantized embeddings.

Credits: [docs/credits.md](docs/credits.md).

## Running it

Requirements:
- DGX Station GB300 (GB300 + RTX PRO 6000 Blackwell) on DGX OS 7 / Ubuntu 24.04 arm64;
- NVIDIA driver 615+ (CDMM, both GPUs in one CUDA process);
- Docker with the NVIDIA Container Toolkit's CDI spec (`/var/run/cdi/nvidia.yaml`; DGX OS 7 generates it, otherwise
  `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`);
- the `hf` CLI (`huggingface_hub`) and ~360 GB of disk for the weights;
- ~120 GB of free host RAM (the cold tier and pinned buffers).

**1. Get the code and the weights**

```bash
git clone https://github.com/chronarion/hecate && cd hecate
hf download chronarion/GLM-5.3-Hecate --local-dir /data/models/GLM-5.3-Hecate
```

**2. Configure.** Copy `.env.example` to `.env` and set the two GPU UUIDs (`nvidia-smi -L`) and the weights folder:

```bash
cp .env.example .env
# HECATE_GB300_UUID=GPU-...   HECATE_RTX_UUID=GPU-...   HECATE_WEIGHTS=/data/models/GLM-5.3-Hecate
```

**3. Build and start**

```bash
docker compose build                 # ~10 min on the Grace CPU: ExLlamaV3 + the patched extension (sm_103a / sm_120a)
docker compose up -d
docker compose logs -f hecate        # ready at the line starting with "warmup:"
```

A start takes ~4-5 min with the weights in the page cache, longer when they come from disk. The first start also
compiles the runtime kernels (moe4, moe3, Grace band, Triton) into `./cache`, ~9 min in all, once. The warmup ends with
a ~10K-token prompt, so the first long request already runs at full speed.

**4. Use it.** The server speaks the OpenAI API on port 8103 (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`); the model is called `base`:

```bash
curl localhost:8103/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "base", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1000}'
```

GLM-5.3 reasons before it answers. The reasoning comes back in `reasoning_content`, the answer in `content`.

The API has no key by default, and compose publishes the port on all interfaces. To keep it on this machine, set
`HECATE_PORT=127.0.0.1:8103` in `.env`. Stop the server with `docker compose down`.

**Long context.** By default the KV pool holds 512K tokens: two 256K conversations. For a single conversation of up to
1M tokens, use the 1M profile in `.env`; it decodes ~27% slower:

```bash
HECATE_PLACEMENT=tier-120-85-far51-mtp256
HECATE_RTX_I4=4
HECATE_MAX_SEQ_LEN=1048576
HECATE_CACHE_TOKENS=1048576
```

[docs/configuration.md](docs/configuration.md) lists every setting and the profiles in between.

## Repository layout

| Path | Contents |
|---|---|
| `docker/` | Dockerfile (pinned ExLlamaV3 v1.6.5 + unmodified TabbyAPI) and the ordered patches to ExLlamaV3's C++/CUDA extension, including the tcgen05 MLA prefill kernel (`hmla/`) |
| `runtime/tier/` | tier split: int4 MoE kernels (`moe4/`), the RTX band's trellis MoE kernels (`moe3/`) and step graph, Grace band engine (`cpuband.cpp`) |
| `runtime/core/` | int8 base, whole-forward and draft CUDA graphs, fused router, int4 draft head, loader for the release folder |
| `runtime/patched/` | ExLlamaV3 generator patches (MTP gap merge, draft-graph hook, decode share, event-loop yield) |
| `runtime/mla/` | MLA / DSA decode kernels (Triton, alignment-hinted AOT compile) |
| `tools/` | weight building (GPTQ int4, int8, warm-band trellis requant, base folder), placement generator, KLD and benchmark scripts |
| `benchmarks/` | raw results of the release benchmarks |
| `docs/` | configuration, credits, model card |

## License

- Hecate's own code is MIT.
- The ExLlamaV3 parts are MIT (turboderp).
- TabbyAPI is used unmodified (AGPL-3.0).
- The weights are derived from zai-org/GLM-5.3 and follow the GLM-5.3 License.
