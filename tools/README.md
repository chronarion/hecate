# Tools

## Building the weights from stock BF16

All Hecate weights derive from `zai-org/GLM-5.3-BF16`. The 1.5 TB checkpoint is streamed module by module.
`kld/stage.py` fetches each module's byte ranges over ssh into sparse local files and punches them back out when the
module is done, so ~60 GB of local disk is enough.

| Step | Tool | Output | GB300 time |
|---|---|---|---|
| 1. routing log + popularity ranks | (served-model routing log, 2,599 verify steps) | `rank.json` | — |
| 2. placement | `placement/tier_placement.py rank.json 144 85 …` (`FAR=1`), `placement/kv_placement.py` for larger KV pools | `placement/*.json` | — |
| 3. int4 hot + cold tiers (GPTQ, causal state propagation and down calibration on the quantized gate/up output, after brandonmusic's TR3 recipe) | `quant/gptq_int4.py` | `int4/layer{L}.safetensors`, `layer{L}.r229.safetensors` | ~2 h |
| 4. warm tier (3-bit EXL3 trellis, same calibration) | `quant/requant_rtx_band.py` | `rtx-band/layer{L}.safetensors` | several hours |
| 5. int8 non-expert linears | `quant/build_int8.py` | `int8/{layer}.safetensors` | CPU, minutes |
| 6. base folder (unquantized tensors, virtual tensor list, config, tokenizer) | `quant/build_skeleton.py` | `base/` | CPU, minutes |

`quant/test_moe4_rtx.py` checks the int4 MoE kernel built for the RTX (sm_120a) against the GB300 build and an fp32
reference.

## Fidelity: KLD against BF16

This follows the local-inference-lab rtx6kpro protocol.
- `kld/build_corpus.py`: the frozen 18-window corpus.
- `kld/kld_ref.py`: the BF16 reference logits, one streaming pass.
- `kld/kld_cand.py`: a candidate built from the served stack's modules, at the verify shape.
- `kld/kld_report.py`: mean KL, a window-bootstrap CI and top-1 agreement.

## Speed

- `bench/speed.py`: 10 fixed prompts, greedy, 400 tokens.
- `bench/qa.py`: 6 known-answer questions.
- `bench/longctx.py`: one long prompt, reporting prefill and decode speed at that depth.
- `bench/kvtier.py`: the system-RAM KV tier: a long document with planted facts is pushed out of the GPU pool and
  asked about again (needs `HECATE_SYSMEM_KV_GB` > 0 to come back fast).
- The community benchmark (`llm-inference-bench`) invocations are in `bench/llm_inference_bench.sh`.
- `bench/specmix.py`: decode speed and draft acceptance on everyday workloads at the sampling clients send (default
  temperature 1.0): reasoning, code, long-context questions, multi-turn tool use over a small repository, file edits,
  and general chat; reports tokens per decode step from the usage block's draft counts.
- `bench/incprefill.py`: time to first token when a cached conversation grows by 500-8,000 new tokens.

## Expert placement from routing statistics

- `EXL3_ROUTE_LOG=FILE` (server environment): appends every decode step's expert selection per layer (read from the
  band mailboxes, no extra GPU work).
- `placement/route_stats.py`: per-band load from such a log, the band-race model for RTX copies of cold experts, and
  `--write-copies` (each layer's cold experts ranked by use).
- `placement/rerank_placement.py`: re-ranks which int4 experts sit in HBM and which on the Grace from one or more logs
  (weighted), keeping the RTX band as it is; the result needs a KLD check (the Grace computes with 8-bit activations).

## Kernel tests

- `dev/test_draft_couple.py`: the draft head's sampling noise against ExLlamaV3's sampler, bit for bit.
- `dev/test_grace_picks.py`: Grace band jobs with up to 20 distinct cold picks against their one-pick jobs summed in
  pick order, bit for bit (until 1.9 the band skipped picks past the 8th)..
