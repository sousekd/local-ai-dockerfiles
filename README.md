# Local AI Dockerfiles

Dockerfiles and deployment recipes for local AI inference and supporting services.

Most model-serving images target:

- NVIDIA RTX PRO 6000 Blackwell GPUs (TP: 1 or TP: 2)
- AMD EPYC 9355 Turin
- Up to 768 GB RAM using 12 × 64 GB DDR5-6400

The configurations are hardware-oriented working recipes rather than universal images. Check the documentation accompanying each Dockerfile for compatibility, build instructions, launch parameters and benchmarks.

## Models

- [GLM 5 — FreeToken v2](models/glm-5/freetoken-v2.md) (for GLM 5.2 NVFP4)
- [GLM 5 MoE DSA — SGLang + KTransformers v1](models/glm-5/sglang-ktransformers-moe-dsa-v1.md) (for GLM 5, 5.1, 5.2 and 5.3)
- [GLM 5 Next — SGLang + KTransformers v1](models/glm-5/sglang-ktransformers-next-v1.md) (for GLM 5.3 Flash)
- [Kimi K2 — SGLang + KTransformers v3](models/kimi-k2/sglang-ktransformers-v3.md) (for Kimi K2.5, 2.6 and 2.7 Code)
