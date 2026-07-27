# Local AI Dockerfiles

Dockerfiles and deployment recipes for local AI inference and supporting services.

Most model-serving images target:

- NVIDIA RTX PRO 6000 Blackwell GPUs (TP: 1 or TP: 2)
- AMD EPYC 9355 Turin
- Up to 768 GB RAM using 12 × 64 GB DDR5-6400

The configurations are hardware-oriented working recipes rather than universal images. Check the documentation accompanying each Dockerfile for compatibility, build instructions, launch parameters and benchmarks.

## Models

- [Kimi K2 — SGLang + KTransformers v3](models/kimi-k2/sglang-ktransformers-v3.md)
