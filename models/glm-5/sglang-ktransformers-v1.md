# GLM 5 — SGLang + KTransformers v1

SGLang and KTransformers image for hybrid CPU/GPU inference of the GLM 5 model family on one NVIDIA Blackwell GPU.

- [Dockerfile](Dockerfile.sglang-ktransformers-v1)
- [SGLang SM120 backport](https://github.com/trilog-inc/sglang/tree/922c076bbd1258e542f7786c72be1b08fd1f7930)
- [KTransformers fork](https://github.com/trilog-inc/ktransformers/tree/340d6b118560731f15dc4cf1c8d20be625f64dee)
- [GLM-5.2-FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.2-FP8)

This family-level recipe has been tested and benchmarked with the GLM-5.2 FP8 checkpoint. It uses the native KT-Kernel CPU backend, layerwise GPU prefill, FlashInfer sparse MLA attention and DeepGEMM on SM120.

## Tested configuration

- Model: `zai-org/GLM-5.2-FP8`
- Model revision: `ba978f7d347eaf65d22f1a86833408afdb953541`
- Architecture: GLM-5.2, 753B total parameters and 40B activated
- Checkpoint and CPU expert method: FP8
- Resident GPU routed experts: 0 per MoE layer
- CPU expert backend: AVX-512 BF16
- KV cache: FP8 E4M3, 327,680 aggregate tokens
- GPU: 1 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB
- CPU: AMD EPYC 9355 Turin
- Host memory: 768 GB DDR5-6400
- NVIDIA driver: 610.57.04
- Container CUDA toolkit: 12.9.1
- CUDA architecture: SM120
- Tensor parallelism: TP: 1
- Maximum concurrent requests: 2
- Maximum context per request: 163,840 tokens

## Component versions

| Component | Version or revision |
|---|---|
| Base image | `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04@sha256:a2e1e2360c85298ac47ec2543b406ab1e8cec42e31ee47e4d32140ebc82e1067` |
| SGLang SM120 backport | `922c076bbd1258e542f7786c72be1b08fd1f7930` |
| KTransformers | `0.6.3.post1` at `340d6b118560731f15dc4cf1c8d20be625f64dee` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| FlashInfer | `0.6.13` at `5f2bdc41f9ffecef9d8ed590e688e7c0f108504f` |
| NVIDIA Cutlass DSL | `4.5.2` |
| Python | 3.12 |
| PyTorch | `2.9.1+cu129` |
| Transformers | `5.14.1` |
| CUDA architecture | SM120 |

## Design decisions

### Pinned SM120 stack

The image builds the complete accelerated path from fixed source revisions:

- SGLang with the GLM-5.2 SM120 NSA top-k backport;
- KT-Kernel with native AMD Turin and CUDA SM120 support;
- the DeepGEMM revision used by the backport;
- FlashInfer sparse MLA for SM120.

The KTransformers SGLang submodule is deliberately not installed. The separate pinned SGLang checkout contains the required SM120 changes.

### Native Turin and SM120 build

KT-Kernel is compiled with AVX-512, AVX-512 VNNI, AVX-512 BF16, AVX-512 VBMI and CUDA SM120 enabled. The resulting image is hardware-oriented and should be built on the same CPU class on which it will run.

### FlashInfer AOT cache

The Docker build compiles the matching `flashinfer-jit-cache` package with `FLASHINFER_CUDA_ARCH_LIST=12.0f`. This packages the SM120 sparse-MLA module used by GLM-5.2.

The optional generic `flashinfer-cubin` package is intentionally omitted because it downloads more than 16,000 kernels for multiple architectures. Cutlass DSL is pinned to 4.5.2 because FlashInfer 0.6.13 uses the pre-4.6 `OperandMajorMode` API.

### CPU experts and layerwise prefill

The tested configuration keeps no routed experts permanently on the GPU:

```text
--kt-num-gpu-experts 0
```

Small uniform GPU-expert allocations did not materially improve decode throughput on the tested system. Keeping all persistent routed experts on CPU avoids the per-layer GPU expert launch and merge path and leaves more VRAM for long-context KV cache.

Prompts with at least 1,024 tokens can use the full-GPU layerwise-prefill fallback:

```text
--kt-gpu-prefill-token-threshold 1024
```

Decode remains CPU-memory-bandwidth-bound because routed expert weights are read from host memory for every generated token.

### Persistent DeepGEMM cache

DeepGEMM kernels are selected and JIT-compiled for the model and launch shape. Mounting a persistent directory at `/root/.cache/deep_gemm` preserves this work across disposable containers.

`SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` compiles the sampled decode and prefill shapes used by the tested 16,384-token chunk size. Missing shapes can still be added to the same cache later.

## Build

Run from the repository root:

```bash
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg BUILD_JOBS=16 \
  --build-arg NVCC_THREADS=4 \
  -f models/glm-5/Dockerfile.sglang-ktransformers-v1 \
  -t sglang-kt-glm:v1 \
  .
```

The SGL kernel and FlashInfer AOT builds can take a long time. Repeat the same command without `--no-cache` to reuse the BuildKit layers and compiler cache.

## Launch GLM 5.2 FP8

Specifying a chat-template file is optional. To use the template from the tested model revision explicitly:

```bash
curl -fL \
  https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/ba978f7d347eaf65d22f1a86833408afdb953541/chat_template.jinja \
  -o GLM-5.2-Default.jinja
```

Set the persistent DeepGEMM cache, Hugging Face cache and exact model snapshot:

```bash
DEEPGEMM_CACHE=/path/to/sglang-kt-glm-v1-cache
HF_HUB_CACHE=/path/to/huggingface/hub
MODEL_SNAPSHOT=models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541
CHAT_TEMPLATE=/path/to/GLM-5.2-Default.jinja

mkdir -p "${DEEPGEMM_CACHE}"
```

Launch the server on GPU 0:

```bash
docker run --rm \
  --name glm-5.2-fp8 \
  --ipc=host \
  --cap-add=SYS_NICE \
  --runtime=nvidia \
  --gpus device=0 \
  -v "${HF_HUB_CACHE}:/root/.cache/huggingface/hub:ro" \
  -v "${DEEPGEMM_CACHE}:/root/.cache/deep_gemm:rw" \
  -v "${CHAT_TEMPLATE}:/opt/templates/GLM-5.2-Default.jinja:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e SGLANG_DG_CACHE_DIR=/root/.cache/deep_gemm \
  -e SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
  -p 8000:8000 \
  sglang-kt-glm:v1 \
  python -m sglang.launch_server \
  --model-path "/root/.cache/huggingface/hub/${MODEL_SNAPSHOT}" \
  --kt-weight-path "/root/.cache/huggingface/hub/${MODEL_SNAPSHOT}" \
  --served-model-name glm-5.2-fp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.94 \
  --trust-remote-code \
  --sleep-on-idle \
  --enable-metrics \
  --kt-cpuinfer 24 \
  --kt-threadpool-count 1 \
  --context-length 163840 \
  --max-running-requests 2 \
  --prefill-max-requests 2 \
  --max-total-tokens 327680 \
  --kv-cache-dtype fp8_e4m3 \
  --kt-num-gpu-experts 0 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-max-deferred-experts-per-token 1 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 16384 \
  --attention-backend nsa \
  --nsa-prefill-backend flashinfer_sparse_mla \
  --nsa-decode-backend flashinfer_sparse_mla \
  --fp8-gemm-backend deep_gemm \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --chat-template /opt/templates/GLM-5.2-Default.jinja \
  --sampling-defaults model
```

The explicit chat template is optional. Remove its mount and `--chat-template` argument to use the template included with the checkpoint.

## Optional DeepGEMM precompilation

SGLang can populate the same persistent cache before the normal server launch. Use the complete launch command above with the same model, batching, chunking and backend arguments, but replace:

```text
python -m sglang.launch_server
```

with:

```text
python -m sglang.compile_deep_gemm --timeout 3600
```

The precompile command starts the model, captures the DeepGEMM call shapes, writes the kernels to `${DEEPGEMM_CACHE}` and exits. It moves the one-time JIT work out of the normal startup path; it does not reduce steady-state inference time. Use a separate cache after changing the image revision, GPU architecture or kernel-shaping launch arguments.

## Runtime memory

The server reported:

- 22.37 GB for loaded GPU weights and runtime state before KV allocation;
- 18.76 GB for 327,680 FP8 KV tokens;
- 52.72 GB available after CUDA graph capture.

`nvidia-smi` showed 58,140 MiB allocated during benchmark warm-up and 79,998 MiB after the long-context benchmark had populated the KV cache. The post-benchmark value is not a permanent minimum and does not imply that all memory must be committed before serving begins.

## Benchmarks

Measured in August 2026 with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) 0.4.29 through an OpenAI-compatible llama-swap endpoint.

The benchmark used 30-second sustained decode cells, concurrency 1 and 2, no additional decode warm-up, and integrated prefix-cache scout requests. Prefill throughput is client prompt tokens divided by time to first token. Decode values are aggregate sustained throughput across the requested concurrency.

### Prefill

| Context | Prompt tokens | TTFT | Throughput | Requests |
|---:|---:|---:|---:|---:|
| 8k | 8,198 | 17.62 s | 465 tok/s | 1 |
| 16k | 16,228 | 20.98 s | 773 tok/s | 1 |
| 32k | 32,319 | 42.02 s | 769 tok/s | 1 |
| 64k | 64,509 | 84.82 s | 761 tok/s | 1 |
| 128k | 128,878 | 170.59 s | 755 tok/s | 1 |

The 8k result appears to include first-use overhead and is not representative of the stable longer-prompt prefill rate.

### Aggregate sustained decode

| Context | Concurrency 1 | Concurrency 2 |
|---:|---:|---:|
| 0 | 8.0 tok/s | 9.4 tok/s |
| 8k | 7.9 tok/s | 9.4 tok/s |
| 16k | 8.0 tok/s | 9.3 tok/s |
| 32k | 7.8 tok/s | 9.3 tok/s |
| 64k | 7.9 tok/s | 9.1 tok/s |
| 128k | 7.9 tok/s | 9.1 tok/s |

The mean across the six contexts was 7.9 tok/s at concurrency 1 and 9.3 tok/s aggregate at concurrency 2. At concurrency 2, per-request sustained throughput was approximately 4.5–4.7 tok/s.

The integrated scout primes the radix prefix used by each sustained decode cell, and the concurrent workers use the same generated context. The concurrency-2 table therefore measures aggregate cached-context decode scheduling rather than simultaneous uncached prefill of two unrelated prompts.

These results describe this specific CPU/GPU placement and are not intended as general GLM-5.2, FP8 or KTransformers performance claims.

## Known limitations

- The image targets Linux/x86-64, CUDA 12.9 and SM120, and KT-Kernel is compiled natively for AMD Turin.
- Only `zai-org/GLM-5.2-FP8`, TP: 1 and zero resident GPU experts are benchmarked here.
- Model weights and chat templates are not included.
- Loading the checkpoint took approximately 8 minutes 44 seconds on the tested system.
- Decode is limited by CPU expert execution and host-memory bandwidth; the GPU remains lightly utilized during steady decode.
- The checkpoint uses block-wise FP8 scales rather than UE8M0. SGLang warns that DeepGEMM may degrade accuracy on Blackwell with this scale format.
- No model-provided FP8 KV scale factors were found, so SGLang defaults them to 1.0 and warns about possible accuracy loss.
- The pinned Transformers 5.14.1 stack warns about possible GLM RoPE incompatibilities with Transformers 5.x. No RoPE failure was observed in this benchmark.
