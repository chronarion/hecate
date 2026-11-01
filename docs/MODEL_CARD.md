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
| Decode, concurrency 1 (10 prompts × 3 passes, greedy, MTP 3 drafts) | ~148 tok/s |
| KL divergence vs BF16 (rtx6kpro protocol, 36,848 positions) | 0.0738 (see the benchmarks below) |
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

- **Placement.** Each MoE layer's experts are ranked by how often they are routed. The data is a routing log of 2,599
  verify steps over 24 prompts.
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

#### Decode, aggregate tok/s by context and concurrency (llm-inference-bench: 30 s windows, ignore_eos, 2048 max tokens, the server's default temperature 1.0; mean of the runs that measured the cell)

| Context | C=1 | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 156 | 201 | 114 | 245 | 153 | 250 |
| 4K | 118 | 171 | 111 | 244 | - | - |
| 16K | 126 | 165 | 111 | 240 | - | - |
| 32K | 119 | 163 | 110 | 240 | - | - |
| 64K | 116 | 160 | 107 | - | - | - |
| 128K | 114 | 147 | - | - | - | - |

Each request gets the aggregate divided by C. Hecate is tuned for one or two conversations: from 4 concurrent requests
the server trims MTP drafts to stay on the 8-row decode fast path (4: one draft, 8: none); from 16 it leaves that path.
Past 2,048 tokens of context the model's sparse attention (DSA) takes over. Single concurrency-1 cells vary by up to
±10% between runs, because MTP acceptance depends on the generated text.

#### Prefill (llm-inference-bench standalone cold prefill, mean of 2 rounds)

| Prompt | 1K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill tok/s | 3,316 | 3,591 | 3,711 | 3,752 | 3,710 | 3,506 | 3,210 |
| Time to first token | 0.3 s | 1.2 s | 2.2 s | 4.3 s | 8.7 s | 18.4 s | 40.1 s |

#### Our speed test

10 chat prompts × 3 passes, greedy, 400 tokens, concurrency 1: **~148 tok/s** (151.1 / 148.2 / 148.2 on release 4;
149.0 / 148.5 / 145.4 on release 3; ~140 on release 2, whose decode step waited ~1.9 ms per step on the host). A 27K-token prompt: prefill 3,790 tok/s,
then 129 tok/s decode; a 109K-token prompt: prefill 3,370 tok/s, then 122 tok/s (means of 3 runs; single runs vary ±5%).

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
| **Hecate** (this release) | 3.8² | **0.0738**¹ | ours: 18 windows, 36,848 positions (wiki, code, chat, math, zh, two 8K windows), verify-shape decode path |
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
- ~120 GB of free host RAM.

## License and credits

The weights are derived from GLM-5.3 by Z.ai and distributed under the GLM-5.3 License (see `base/LICENSE`). Quantization
and runtime build on ExLlamaV3 (turboderp, MIT); see the
[credits](https://github.com/chronarion/hecate/blob/main/docs/credits.md) for the methods and projects this work draws
on.
