# DeepSeek V3.2 — SGLang + KTransformers v2

SGLang and KTransformers image for hybrid CPU/GPU inference of DeepSeek V3.2 FP8 on one NVIDIA Blackwell GPU.

- [Dockerfile](Dockerfile.sglang-ktransformers-v2)
- [DeepSeek-V3.2 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [KTransformers revision](https://github.com/kvcache-ai/ktransformers/tree/a694e718e8b3ec2e221277d674e369f3be38d2a4)
- [SGLang SM120 backport](https://github.com/trilog-inc/sglang/tree/922c076bbd1258e542f7786c72be1b08fd1f7930)

This recipe has been tested and benchmarked with the official DeepSeek-V3.2 FP8 checkpoint. It uses the AVX-512 BF16 CPU expert backend, layerwise GPU prefill, FlashInfer sparse MLA and Triton FP8 GEMM/MoE kernels. Model paths and launch options are supplied at runtime.

## Tested configuration

- Model: `deepseek-ai/DeepSeek-V3.2`
- Model revision: `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`
- Architecture: DeepSeek V3.2 MoE with DeepSeek Sparse Attention (DSA)
- Checkpoint and CPU expert method: FP8
- Activations: BF16
- Resident GPU routed experts: 0 per MoE layer
- CPU expert backend: AVX-512 BF16, one 24-thread pool on one NUMA node
- KV cache: FP8 E4M3, 262,144 aggregate tokens
- GPU: 1 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB
- CPU: AMD EPYC 9355 Turin
- Host memory: 768 GB DDR5-6400
- NVIDIA driver: 615.71.10
- GPU power limit: 350 W
- Container CUDA toolkit: 13.0.2 (nvcc 13.0.88)
- CUDA architecture: SM120; sgl-kernel targets SM120a
- Tensor parallelism: TP: 1
- Maximum concurrent requests: 2
- Maximum context per request: 131,072 tokens
- Prefill chunk size: 16,384 tokens
- CUDA graph batch sizes: 1 and 2

## Component versions

| Component | Version or revision |
|---|---|
| Base image | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01` |
| KT-Kernel | `0.7.2` at `a694e718e8b3ec2e221277d674e369f3be38d2a4` |
| SGLang-KT | `0.6.3.post1+ds32` at `922c076bbd1258e542f7786c72be1b08fd1f7930` |
| SGL kernel | `0.3.21+ds32.sm120` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| FlashInfer | `0.6.13+ds32` at `5f2bdc41f9ffecef9d8ed590e688e7c0f108504f` |
| Python | 3.12 |
| PyTorch | `2.9.1+cu130` |
| Triton | `3.5.1` |
| Transformers-KT | `5.6.0.post6` |
| Accelerate-KT | `1.14.0.post4` |
| NVIDIA Cutlass DSL | `4.5.2` |
| Quack Kernels | `0.3.3` |
| CUDA Python / bindings | `13.2.0` |

## Design decisions

### Pinned source builds and compatibility patches

The self-contained Dockerfile builds KT-Kernel, SGL kernel, DeepGEMM and FlashInfer from fixed revisions. KTransformers CPU kernels are paired with the SGLang fork's SM120 sparse-MLA and NSA top-k support.

The inline compatibility patches:

- Load the checkpoint's DeepSeek V3.2 configuration through the typed V3 configuration class, preserving DSA/indexer, YaRN and quantization fields. The mounted checkpoint remains read-only.
- Allow DeepSeek V3.2 through the existing FlashInfer sparse-MLA architecture check while retaining its packed-cache layout and scale checks.
- Align Python dependencies and redirect `compressed-tensors` and `xgrammar` wheel dependency metadata to Transformers-KT, avoiding overlapping ownership of the `transformers` namespace.
- Add CUDA 13's CCCL include directory for host C++ compilation.
- Limit SGL kernel compilation to SM120a, retaining its common kernels and upstream math flags while excluding unused Hopper, FA3, FlashMLA and bundled DeepGEMM build targets.
- Import DeepGEMM in the NSA SM-count context manager even when its optional GEMM dispatcher is disabled.

KT-Kernel is compiled with AVX-512 BF16/VNNI/VBMI and AMX disabled. FlashInfer's SM120 sparse-MLA module is precompiled without requiring a GPU during the build and installed outside the runtime cache mount.

The build runs dependency and API checks and records the resolved Python environment in `/opt/installed-packages.txt`. Ubuntu packages and ordinary transitive dependencies are not fully locked.

### Native FP8 experts and sparse attention

The launch uses the official FP8 weights without converting CPU experts to INT4 or INT8. `--kt-max-deferred-experts-per-token 0` keeps all routed expert contributions.

The packed NSA cache stores the 512-element non-RoPE component in FP8 with four dynamic FP32 scales, and the 64-element RoPE component in BF16. FlashInfer receives `kv_scale_format="arbitrary_fp32"`. The generic startup warning about absent FP8 KV calibration factors does not describe these dynamically scaled cache blocks. FP8 KV is not mathematically identical to BF16; no reference quality-equivalence evaluation is included.

Triton handles FP8 GEMM and GPU MoE execution. DeepGEMM remains installed for the NSA indexer, while `SGLANG_ENABLE_JIT_DEEPGEMM=0` disables its general GEMM dispatcher.

### CPU experts and layerwise prefill

Routed experts reside on CPU during decode. Prefill batches of at least 1,024 tokens use the full-GPU layerwise path, which stages expert weights onto the GPU one layer at a time.

The 16,384-token chunk size amortizes these transfers across more prompt tokens. Smaller 4,096-token chunks were substantially slower for prefill on the tested system. Changing chunk size did not improve single-request decode, which remains limited by CPU expert execution and host-memory access.

The tested command retains `--kt-enable-dynamic-expert-update`; with zero resident GPU experts there are no persistent GPU expert slots to populate.

### Chat encoding

The official checkpoint provides Python message encoding rather than a Jinja chat template. With `--tool-call-parser deepseekv32`, this SGLang revision selects its DeepSeek V3.2 Python chat encoder. No external chat-template file is required. `--reasoning-parser deepseek-v3` separates reasoning output, and `--sampling-defaults model` uses the checkpoint's defaults (`temperature=1.0`, `top_p=0.95`).

## Build

Run from the repository root:

```bash
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg BUILD_JOBS=24 \
  --build-arg NVCC_THREADS=1 \
  -t kt-ds32:v2 \
  - < models/deepseek-v3.2/Dockerfile.sglang-ktransformers-v2
```

Only the Dockerfile is needed as local build input. Source downloads require network access. The native builds can take substantial time and temporary disk space; reduce `BUILD_JOBS` if necessary. Reuse the same build arguments and existing BuildKit caches for incremental builds.

## Launch DeepSeek V3.2 FP8

Download the pinned checkpoint into a Hugging Face cache before launching in offline mode. Set the cache paths, exact model snapshot and GPU:

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
RUNTIME_CACHE=/path/to/kt-ds32-v2-cache
MODEL_SNAPSHOT=models--deepseek-ai--DeepSeek-V3.2/snapshots/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6
GPU_DEVICE=0
CONTAINER_MODEL="/root/.cache/huggingface/hub/${MODEL_SNAPSHOT}"

test -f "${HF_HUB_CACHE}/${MODEL_SNAPSHOT}/config.json"
mkdir -p "${RUNTIME_CACHE}"
```

Mount the entire Hugging Face hub directory so snapshot symlinks can resolve their model blobs. `GPU_DEVICE` accepts a GPU index or UUID.

```bash
docker run --rm \
  --name deepseek-v3.2-fp8 \
  --init --ipc=host --cap-add=SYS_NICE \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  --runtime=nvidia --gpus "device=${GPU_DEVICE}" \
  -v "${HF_HUB_CACHE}:/root/.cache/huggingface/hub:ro" \
  -v "${RUNTIME_CACHE}:/cache:rw" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e SGLANG_APPLY_CONFIG_BACKUP=none \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_NSA_FORCE_MLA=1 \
  -e SGLANG_NSA_FUSE_TOPK=0 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e SGLANG_DG_CACHE_DIR=/cache/deep_gemm \
  -e DG_JIT_CACHE_DIR=/cache/deep_gemm \
  -p 8000:8000 \
  kt-ds32:v2 \
  python -m sglang.launch_server \
  --model-path "${CONTAINER_MODEL}" \
  --kt-weight-path "${CONTAINER_MODEL}" \
  --served-model-name deepseek-v3.2-fp8 \
  --host 0.0.0.0 --port 8000 \
  --mem-fraction-static 0.94 \
  --trust-remote-code \
  --sleep-on-idle \
  --enable-metrics --enable-cache-report \
  --kt-cpuinfer 24 --kt-threadpool-count 1 \
  --context-length 131072 \
  --max-running-requests 2 --prefill-max-requests 2 \
  --max-total-tokens 262144 \
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --page-size 64 \
  --kt-num-gpu-experts 0 --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-max-deferred-experts-per-token 0 \
  --kt-enable-dynamic-expert-update \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 16384 \
  --attention-backend nsa \
  --nsa-prefill-backend flashinfer_sparse_mla \
  --nsa-decode-backend flashinfer_sparse_mla \
  --fp8-gemm-backend triton --moe-runner-backend triton \
  --cuda-graph-bs 1 2 \
  --reasoning-parser deepseek-v3 --tool-call-parser deepseekv32 \
  --sampling-defaults model
```

The CPU-inference, context and memory settings may need adjustment for other CPU topologies or memory capacities. The cache mount preserves runtime compilation work between disposable containers.

## Verify the server

The server is ready when this returns HTTP 200:

```bash
curl -f http://localhost:8000/health
```

Send an OpenAI-compatible chat request:

```bash
curl -f http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "deepseek-v3.2-fp8",
    "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
    "max_tokens": 1024,
    "stream": false
  }'
```

## Runtime memory

The server reported 21.24 GB for GPU weights and model state before KV allocation, 11.74 GB for the 262,144-token KV pool, and 0.11 GB for CUDA graphs. `nvidia-smi` showed 35,346 MiB on the serving GPU after startup and three short warm-up requests. Layerwise prefill requires additional temporary memory; startup usage is not peak serving usage.

## Benchmarks

Measured on 24 October 2026 with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) 0.8.0 through the OpenAI-compatible endpoint.

The benchmark used 30-second sustained decode cells, concurrency 1 and 2, no additional decode warm-up, integrated prefix-cache scout requests and 16,384-token prefill chunks. Prefill throughput is client prompt tokens divided by time to first token. Decode values are aggregate sustained throughput across the requested concurrency.

### Prefill

| Context | Prompt tokens | TTFT | Throughput | Requests |
|---:|---:|---:|---:|---:|
| 8k | ~8,192 | 16.84 s | 486 tok/s | 1 |
| 16k | 16,250 | 21.97 s | 740 tok/s | 1 |
| 32k | 32,342 | 45.05 s | 718 tok/s | 1 |
| 64k | 64,559 | 94.05 s | 686 tok/s | 1 |

### Aggregate sustained decode

| Context | Concurrency 1 | Concurrency 2 |
|---:|---:|---:|
| 0 | 7.7 tok/s | 9.3 tok/s |
| 8k | 7.7 tok/s | 9.3 tok/s |
| 16k | 7.7 tok/s | 9.2 tok/s |
| 32k | 7.6 tok/s | 9.3 tok/s |
| 64k | 7.6 tok/s | 9.1 tok/s |

At concurrency 2, per-request sustained throughput was approximately 4.5–4.6 tok/s.

The integrated scout primes the radix prefix used by each sustained decode cell. The concurrency-2 results describe cached-context scheduling rather than simultaneous uncached prefill of two unrelated prompts. The 131,072-token per-request limit and 262,144-token pool were configured and initialized; benchmark coverage extends to approximately 64k, not 128k.

These results describe this specific hybrid CPU/GPU placement and are not intended as general DeepSeek V3.2, FP8 or KTransformers performance claims.

## Known limitations

- The image targets Linux/x86-64, CUDA 13 and SM120. Only one RTX PRO 6000, TP: 1 and the pinned official FP8 checkpoint are benchmarked here.
- Model weights are not included. Serving this checkpoint requires hundreds of gigabytes of host memory; the tested system provided 768 GB.
- Weight loading took approximately 9 minutes; CUDA graph capture took another 42 seconds.
- Decode remains limited by CPU expert execution and host-memory access. Additional free GPU memory does not automatically place more experts on GPU.
- SGLang uses default Triton W8A8 kernel configurations for several RTX PRO 6000 matrix shapes and warns that tuning files are absent.
- The pinned stack emits generic Transformers 5.x RoPE, image-processor and Cutlass package-discovery warnings. They did not prevent startup, graph capture or the reported benchmark.
- The build excludes optional FA3 modules; model-registry import warnings for unrelated architectures are expected.
- The benchmark verifies serving and performance, not reference numerical equivalence, long-running stability or comprehensive tool-calling correctness.
