---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3-BF16/blob/main/LICENSE
base_model: zai-org/GLM-5.3-BF16
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - glm
  - moe
  - exl3
  - int4
  - gptq
  - dgx-station
  - gb300
  - blackwell
  - speculative-decoding
---

# GLM-5.3-Hecate

Hybrid-quantized GLM-5.3 (753B MoE) for one **NVIDIA DGX Station GB300**: the GB300, the RTX PRO 6000 Blackwell and
the Grace CPU each serve a tier of experts. Served by [Hecate](https://github.com/chronarion/hecate)
(ExLlamaV3 + TabbyAPI + custom kernels). The files are only usable with that runtime: they are not a standard EXL3,
GGUF or safetensors checkpoint.

**Every tensor derives from the stock BF16 release, [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16).**

| | |
|---|---|
| Decode, concurrency 1, realistic workloads (temperature 1) | 144-212 tok/s by workload |
| Decode, concurrency 1 (10 prompts, greedy, MTP 3 drafts) | ~176 tok/s |
| Decode, several conversations (llm-inference-bench, total) | 2: 179 · 4: 244 · 8: 292 tok/s at short context |
| KL divergence vs BF16 (rtx6kpro protocol, 36,848 positions) | 0.0734 (see the benchmarks below) |
| Context | 2 × 256K (Q6 KV) by default; one 1M-token conversation with the long-context profile |

## What is in this repository

| Folder | Contents | Size |
|---|---|---|
| `base/` | config, tokenizer, chat template; unquantized tensors (embeddings, norms, router, DSA indexer, kv_b_proj, MTP norms and eh_proj); `virtual.json` (shapes of EXL3 tensors the runtime never loads) | ~4.7 GB |
| `int4/` | routed experts of the hot tier (144 per layer, GB300 HBM) and the cold tier (27 per layer, Grace), plus all 256 experts of the MTP layer: uniform int4 with a per-row scale, GPTQ with causal state propagation, packed in the moe4 kernel's fragment order | ~247 GB |
| `rtx-band/` | the warm tier (85 experts per layer) as 3-bit EXL3 trellis (mul1 codebook), requantized from BF16 with the same calibration | ~91 GB |
| `int8/` | attention projections, shared experts, dense MLPs, LM head: int8 with a per-row MSE-clipped scale, in the cluster GEMV's fragment order | ~17 GB |
| `placement/` | per-layer expert permutations by routing popularity (default and larger-KV profiles) | small |

## How it was made

- **Placement.** Each MoE layer's experts are ranked by how often they are routed. Since 1.9 the data are decode
  routing logs of ~30K verify steps over reasoning, code, long-context, tool-use, editing and chat requests
  (`placement/tier-144-85-far27-mtp256-r2.json`); the 1.8 ranking came from 2,599 steps over 24 short prompts.
- **int4 tiers.** GPTQ on 852K calibration tokens:
  - act-order;
  - per-expert Hessians shrunk toward the layer-wide input covariance;
  - down projections calibrated on the quantized gate/up output;
  - every layer calibrated on the states the quantized model produces up to it.

  The last two follow [brandonmusic](https://huggingface.co/brandonmusic)'s published TR3 recipe for GLM EXL3 quants
  (causal layer rebuild, down calibration on the reconstructed gate/up output).
- **Warm tier.** ExLlamaV3's EXL3 quantizer (LDLQ trellis, K = 3) with the same per-expert Hessians, in the same pass.
- **int8.** Per-row scale, with the clip chosen by MSE on each weight row.

## Benchmarks

Format of the RTX PRO 6000 community releases: local-inference-lab `llm-inference-bench` v0.8.0 and the
`rtx6kpro` KLD protocol. Hardware: DGX Station GB300 (GB300 252 GB + RTX PRO 6000 Blackwell Max-Q 96 GB + Grace 72-core),
driver 615.71, CUDA 13.0.

Raw JSON and logs: [`benchmarks/`](https://github.com/chronarion/hecate/tree/main/benchmarks).

#### Decode, total tok/s by context and concurrency (llm-inference-bench: 30 s windows, ignore_eos, 2048 max tokens, the server's default temperature 1.0; release 1.10)

| Context | C=1 | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 140 | 179 | 244 | 292 | 293 (8 running, 8 queued) | 297 (8 running, 24 queued) |
| 4K | 132 | 187 | 249 | 293 | - | - |
| 16K | 136 | 188 | 243 | 283 | - | - |
| 32K | 131 | 187 | 245 | 284 | - | - |
| 64K | 131 | 186 | 237 | - | - | - |
| 128K | 127 | 178 | - | - | - | - |

Each request gets the total divided by C. At most 8 requests decode at once (`HECATE_MAX_BATCH`); later ones wait in
the queue and then run at full speed. MTP drafts take rows too: 4 concurrent requests get one draft each, 8 none.
Streamed requests start ahead of non-streamed ones (see the GitHub docs, configuration, "Concurrency"). Single C=1 cells vary by up
to ±10% between runs, because MTP acceptance depends on the generated text.

The benchmark counts tokens from the usage numbers a server attaches to every streamed chunk (vLLM's
`stream_options.continuous_usage_stats`). TabbyAPI sent usage only at the end of a request, so until 1.9 the benchmark
credited a finished request's whole output when it ended (C=1 and C=2 read ~15-20% high) and fell back to counting
chunks when no request finished inside its 30 s window (C=4 read half its real value: one chunk holds a whole MTP step).
Hecate sends per-chunk usage since 1.10, so these numbers are comparable with other servers' but not with the tables of
earlier releases. Compare releases with the same-session numbers below.

#### Several conversations, 1.9 vs 1.10 in one session (each request's own decode speed, tok/s, streamed requests)

| Conversations | short prompt, 1.9 | 1.10 | ~16K-token context, 1.9 | 1.10 |
|---|---:|---:|---:|---:|
| 2 | 90 | 97 | 81 | 98 |
| 4 | 57 | 65 | 54 | 64 |
| 8 | 30 | 36 | 32 | 37 |

Under overload (`tools/bench/overload_check.py` in the GitHub repo: 48 batch requests and a streamed request every 45 s) the running batch
requests decode at 43 tok/s (1.9: 35) and streamed requests get their first token within 0.3 s; with 24 batch requests
carrying unique 30K-token prompts, streamed requests get their first token within 2.3 s (1.9: up to 38 s).

On our short greedy test 1.9 was ~3% slower than 1.8 (~176 vs ~181 tok/s, means over 6 and 3 server restarts; a single
restart moves it by ±3%). Switching the new drafts or the request scheduling off does not change it. About 1% comes from
the new expert placement, which is ranked for real traffic rather than for this test (1.9 with the old placement: ~178);
the rest is likely the Grace band fix, which now computes experts that were skipped before.

#### Realistic workloads

Measured with `tools/bench/specmix.py` (GitHub repo) at temperature 1, release 1.8 vs 1.9 in one session, decode tok/s:

| Workload | 1.8 | 1.9 |
|---|---:|---:|
| Reasoning | 136 | 146 |
| Code | 139 | 146 |
| Questions about a 46K-token document | 136 | 144 |
| Everyday chat (reasoning effort low) | 156 | 154 |
| Agent turns with tool calls that write files, file edits | 146 | 212 |

#### Prefill (llm-inference-bench standalone cold prefill, mean of 2 rounds)

| Prompt | 1K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill tok/s | 3,380 | 3,870 | 4,768 | 4,836 | 4,810 | 4,750 | 4,602 |
| Time to first token | 0.3 s | 1.1 s | 1.7 s | 3.4 s | 6.7 s | 13.6 s | 28.0 s |

Once the 512K-token GPU pool is full, the pages a long prompt pushes out are copied into the system-RAM conversation
cache while it prefills, which makes prompts from 16K up ~5% slower (4,330-4,590 tok/s); with `HECATE_SYSMEM_KV_GB=0`
they are dropped instead and prefill stays at the numbers above.

#### Our speed test

10 chat prompts, greedy, 400 tokens, concurrency 1: **~176 tok/s** (172-179 over 6 server restarts; 1.8: ~181 in the same
session, see above). Release 1.8, a 27K-token prompt: prefill 4,648 tok/s, then 163 tok/s decode; a 109K-token
prompt: prefill 4,712 tok/s, then 158 tok/s (means of 2 runs; single runs vary ±10%).

#### KLD compared with other GLM-5.3 quants

> **Note:** this KLD comparison is preliminary. The methodology is still early: the evaluation sets differ, and
> most numbers come from each author's own setup. Future work will rescore Hecate and the other quants (including
> NVFP4 / MXFP4 / int4) on one shared protocol and window set.

All values are mean KL(BF16 ‖ quant) over the full vocabulary, as published by each quant's author. The evaluation sets
differ, and that alone moves the number a lot: brandonmusic's TR3 3.0 bpw scores 0.098 on 16 windows but 0.038 on the
4-window "panel-v1". Compare rows within a group; across groups, read differences under ~30% as noise. Only the first
group shares one exact window set.

| Quant | bpw | KLD | Evaluation set |
|---|---|---|---|
| **Hecate** (this release) | 3.8² | **0.0734**¹ | ours: 18 windows, 36,848 positions (wiki, code, chat, math, zh, two 8K windows), verify-shape decode path |
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
KV cache); full report in [`benchmarks/kld/report.md`](https://github.com/chronarion/hecate/blob/main/benchmarks/kld/report.md).

² Every stored weight counted: 359 GB over 753B parameters, including the int8 attention / shared experts / LM head
and the unquantized embeddings.

## Usage

```bash
git clone https://github.com/chronarion/hecate && cd hecate
hf download chronarion/GLM-5.3-Hecate --local-dir /data/models/GLM-5.3-Hecate
cp .env.example .env    # GPU UUIDs, weights path
docker compose up -d    # OpenAI-compatible API on :8103
```

Hardware requirements:
- a DGX Station GB300 or an equivalent GB300 + Blackwell PCIe GPU + Grace system;
- driver 615 or newer;
- ~150 GB of free host RAM (default profile).

## License and credits

The weights are derived from GLM-5.3 by Z.ai and distributed under the GLM-5.3 License (see `base/LICENSE`). Quantization
and runtime build on ExLlamaV3 (turboderp, MIT); see the
[credits](https://github.com/chronarion/hecate/blob/main/docs/credits.md) for the methods and projects this work draws
on.
