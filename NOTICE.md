# Third-party code and licenses

| Component | Where | License |
|---|---|---|
| ExLlamaV3 (turboderp) v1.6.5 | built in the image; modified files in `runtime/patched/`, `runtime/mla/`, `docker/patches/exllamav3/` (patch scripts and replacement sources for its extension) | MIT |
| TabbyAPI (theroyallab) @ 564f8fbb | built into the image unmodified | AGPL-3.0 |
| PyTorch, Triton, CUDA | image base | BSD-3 / MIT / NVIDIA EULA |
| GLM-5.3 weights (Z.ai) | the weights repo is derived from zai-org/GLM-5.3-BF16 | GLM-5.3 License |

Everything else in this repository is MIT (see LICENSE).
