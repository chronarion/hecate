# Credits and prior work

Hecate is a set of patches, kernels and runtime modules on top of other people's work. These are the projects and papers
it builds on or borrows ideas from.

## Software it builds on
- **GLM-5.3** by Z.ai (zai-org/GLM-5.3): the model, its MLA + DSA attention, and its MTP draft layer. All weights here are
  derived from the stock BF16 release under the GLM-5.3 License.
- **ExLlamaV3** by turboderp (MIT): the inference engine, the EXL3 trellis format and quantizer (used for the warm tier),
  the MLA / DSA kernels we tuned, the paged generator and MTP drafting.
- **TabbyAPI** by theroyallab (AGPL-3.0): the OpenAI-compatible server, used unmodified.
- **PyTorch, Triton, CUDA 13**, and NVIDIA's DGX Station GB300 development guide (coherency, ATS, CDMM, Grace tuning).

## Ideas and techniques
- **QTIP / trellis quantization** (Tseng et al., 2024), as implemented in EXL3: the warm tier's format.
- **GPTQ** (Frantar et al., 2022): the int4 requant, with act-order, a block-wise error update, and per-expert Hessians
  shrunk toward the layer covariance.
- **brandonmusic's TR3 quantization recipe** (the model cards of
  [GLM-5.2-EXL3-TR3v4](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78) and
  [GLM-5.3-EXL3-TR3](https://huggingface.co/brandonmusic/GLM-5.3-EXL3-TR3-3bpw)): rebuilding the layers causally, each
  calibrated on the output of the already-quantized layers before it, and calibrating each expert's down projection on
  its reconstructed (quantized) gate/up output. Hecate's int4 GPTQ and warm-tier trellis requants follow both. Its
  published KLD numbers are in our comparison table.
- **LOP3 / magic-number int4 to fp16 decode**, from FasterTransformer and Marlin (Frantar et al., 2024): the decode trick in
  the int4 MoE kernel and the draft head.
- **Stream-K** (Osama et al., 2023): work decomposition of the MoE kernels (int4 on the GB300, 3-bit trellis on the RTX)
  and the int8 GEMVs.
- **Thread-block clusters with distributed shared memory** (Hopper/Blackwell programming guide): the cluster-reduced int8
  GEMVs.
- **CPU-side expert computation for MoE offloading**: Fiddler (Kamahori et al., 2024) and kTransformers. These motivated
  computing the cold tier on the Grace cores instead of moving weights.
- **Activation-aware expert placement** (MoE-Infinity and related MoE offloading work): ranking each layer's experts by
  routing popularity to pick their tier.
- **Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) and **multi-token prediction** (DeepSeek-V3 /
  GLM): the draft / verify loop.
- **DeepSeek MLA / DSA, FlashMLA and FlashDecoding (split-KV decode)**: reference designs for the MLA decode split /
  combine work. The tcgen05 sparse attention kernel (prefill and long-context decode) follows the mechanics of FlashMLA's
  SM100 sparse kernels (`.ws` MMAs with TMEM accumulators, the dual-half 64-head layout, P through shared memory,
  commit-driven mbarrier pipeline); the code is our own and reads ExLlamaV3's Q6 latent cache directly.
- **Whole-forward CUDA graphs and megakernels**: the full-graph decode paths of vLLM, SGLang and TensorRT-LLM (also their
  graphed speculative-decoding draft loops), Mirage MPK and Hazy Research's megakernel posts. These motivated graphing the
  whole verify forward and the draft loop.
- **[b12x](https://github.com/local-inference-lab/b12x)** (Luke Alonso, Martin Vit, Jason Cook, MadeBy561 and Brandon
  Music; local-inference-lab, RTX PRO 6000 community; also FlashInfer `experimental/b12x`): K-split of the cooperative MoE
  kernel, MLA split-count re-tuning, and the int4 draft-only LM head.
- **local-inference-lab / rtx6kpro community**: the KLD benchmark protocol (`kld/README.md`) and `llm-inference-bench`,
  which we use for every fidelity and speed number in the README.
