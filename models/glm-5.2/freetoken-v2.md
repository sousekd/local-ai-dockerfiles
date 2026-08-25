# GLM 5.2 — FreeToken v2

FreeToken image and deployment recipe for hybrid CPU/GPU inference of GLM 5.2 on one NVIDIA Blackwell GPU.

- [Dockerfile](Dockerfile.freetoken-v2)
- [FreeToken](https://github.com/FlashML-org/FreeToken)
- [NVIDIA GLM-5.2-NVFP4 checkpoint](https://huggingface.co/nvidia/GLM-5.2-NVFP4)

This version has been tested with the NVIDIA NVFP4 checkpoint, two-request scheduling, radix prefix caching, reasoning output and OpenAI-compatible tool calling.

## Tested configuration

- Model: `nvidia/GLM-5.2-NVFP4`
- Model revision: `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`
- Architecture: GLM-5.2, 753B total parameters and 40B activated
- Expert quantization: NVFP4
- Resident attention, dense MLP and LM-head weights: BF16
- KV cache: BF16, 327,680 aggregate tokens
- GPU: 1 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB
- CPU: AMD EPYC 9355 Turin
- Host memory: 768 GB DDR5-6400
- NVIDIA driver: 610.57.04
- Container CUDA toolkit: 13.0.1
- CUDA architecture: SM120
- Tensor parallelism: TP: 1
- Maximum concurrent requests: 2
- Maximum context per request: 163,840 tokens

## Component versions

| Component | Version or revision |
|---|---|
| Base image | `nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:84e5f33efdccadc21599978ea7e6ed80cbe19bb432f9c339b15a66c4025ac983` |
| FreeToken | `0.1.2` at `2757bb5f91156fc8a44d88ec4b302a81f10c9e81` |
| Python | 3.12 |
| PyTorch | `2.11.0+cu130` |
| Transformers | `5.15.1` |
| Apache TVM FFI | `0.1.13.post3` |
| Triton | `3.6.0` |
| flashlib | `0.3.0` |
| `sglang-kernel` | `0.4.5+cu130` |
| `flashinfer-python` | `0.6.17` |
| `flashinfer-cubin` | `0.6.17` |
| `flashinfer-jit-cache` | `0.6.17+cu130` |
| CUDA architecture | SM120 |

## Design decisions and patches

### Model-agnostic runtime

The image contains FreeToken and its acceleration stack, but no model weights or model-specific launch command. The `ft` CLI is the image entrypoint, so the same image can run `serve`, `bench bw`, `checkpoint` and other FreeToken commands with configuration supplied at runtime.

The GLM-specific choices live only in the launch recipe.

### Pinned CUDA 13 and FreeToken source

The CUDA development base is pinned by digest and FreeToken is checked out at an exact commit. The Docker build verifies the checked-out SHA before installing it.

The development image is intentional. FreeToken, TVM FFI and FlashInfer can JIT a kernel variant that is not already present in their caches, so the runtime keeps a CUDA 13 `nvcc` matching Torch cu130.

### SM120 kernel cache

The companion `freetoken-kernel-cache` package is compiled for compute capability 12.0 during the image build. This moves the known TVM-FFI compilation work into the image while preserving a writable runtime cache for uncovered shapes.

Set this build argument to skip the prebuild:

```bash
--build-arg BUILD_FREETOKEN_KERNEL_CACHE=0
```

The missing kernels will then compile on first use and be stored in the mounted FreeToken cache.

### Turin CPU dispatch

FreeToken's CPU MoE extension performs runtime dispatch rather than inheriting a global `-march=native` from the Docker build host.

On the tested EPYC 9355, the server selected:

```text
avx512bf16+avx512vnni(nvfp4-w4a8)
```

The tested launch reserves 24 physical cores for the pinned MoE worker pool. FreeToken leaves the remaining cores for Torch, tokenization and server work.

### Strict readiness endpoint

FreeToken starts Uvicorn before the model backend has finished loading. Its `/health` endpoint intentionally returns HTTP 200 during that loading phase so FreeToken clients can display progress, which is not suitable for orchestrators that treat any 200 response as ready.

The Dockerfile adds a downstream `/ready` route:

- HTTP 503 while the model is loading or after a fatal initialization error;
- HTTP 200 only when the backend is serving.

For llama-swap, use:

```yaml
checkEndpoint: /ready
```

The patch validates its exact source anchor and fails the Docker build if the pinned FreeToken route layout changes.

### Persistent runtime cache and bandwidth profile

Mounting a persistent directory at `/root/.cache` preserves:

- the per-GPU `ft bench bw` profile;
- TVM-FFI, Torch extension and Triton caches;
- any kernels compiled on first use.

`ft bench bw` measures the real CPU and PCIe MoE paths. The tested profile caused `--moe-backend auto` to select `hybrid`, and `--moe-hybrid-max-fetch -1` assigned 19.5% of each decode step's cache misses to PCIe transfers while the CPU computed the rest.

The profile is GPU-specific. Re-run the bandwidth benchmark after changing the GPU, CPU, NUMA placement or PCIe topology.

### Runtime memory allocation

The launch recipe reserves 327,680 aggregate KV tokens for two requests. The tested initialization resolved to:

| Resource | Observed value |
|---|---:|
| KV capacity | 327,680 tokens |
| KV allocation | 29.06 GiB |
| GPU MoE cache | 1,231 expert slots |
| Free VRAM after initialization | 4.92 GiB |
| Free VRAM after CUDA graphs | 4.86 GiB |

The final configuration leaves limited workspace headroom. Prefix caching is enabled with `--cache-type radix`, and `--enable-cache-report` exposes cache-hit counts in compatible API responses.

## Build

Run from the repository root:

```bash
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg BUILD_JOBS=16 \
  -f models/glm-5.2/Dockerfile.freetoken-v2 \
  -t freetoken-bw:v2 \
  .
```

The image is hardware-oriented but not model-specific. It can serve other FreeToken-supported models on Linux/x86-64 and SM120 with their corresponding runtime arguments.

## Calibrate the hybrid MoE backend

Create a persistent cache directory and benchmark the GPU/CPU path once before launching the model:

```bash
FREETOKEN_CACHE=/path/to/freetoken-cache
mkdir -p "${FREETOKEN_CACHE}"

docker run --rm \
  --init \
  --ipc=host \
  --cap-add=SYS_NICE \
  --ulimit memlock=-1 \
  --runtime=nvidia \
  --gpus device=0 \
  -v "${FREETOKEN_CACHE}:/root/.cache:rw" \
  freetoken-bw:v2 \
  bench bw \
  --gpu 0 \
  --dtype nvfp4 \
  --cpu-threads 24
```

## Launch GLM 5.2 NVFP4

Set the persistent cache, Hugging Face cache and exact local model snapshot:

```bash
FREETOKEN_CACHE=/path/to/freetoken-cache
HF_HUB_CACHE=/path/to/huggingface/hub
MODEL_SNAPSHOT=models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
```

Launch the server on one GPU:

```bash
docker run --rm \
  --name glm-5.2-nvfp4 \
  --init \
  --ipc=host \
  --cap-add=SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  --runtime=nvidia \
  --gpus device=0 \
  -v "${FREETOKEN_CACHE}:/root/.cache:rw" \
  -v "${HF_HUB_CACHE}:/root/.cache/huggingface/hub:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e FREETOKEN_GLM_ATTN_FP8=0 \
  -e FREETOKEN_GLM_MLP_FP8=0 \
  -p 8000:8000 \
  freetoken-bw:v2 \
  serve \
  --model "/root/.cache/huggingface/hub/${MODEL_SNAPSHOT}" \
  --served-model-name glm-5.2-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --memory-ratio 0.94 \
  --dtype bfloat16 \
  --max-seq-len-override 163840 \
  --num-tokens 327680 \
  --kv-reserve-tokens 327680 \
  --max-running-requests 2 \
  --cuda-graph-max-bs 2 \
  --max-prefill-length 8192 \
  --cache-type radix \
  --moe-backend auto \
  --moe-cache-auto \
  --moe-cpu-threads 24 \
  --moe-hybrid-max-fetch -1 \
  --nvfp4-backend flashinfer \
  --tool-call-parser glm47 \
  --reasoning-parser glm \
  --sampling-defaults model \
  --enable-cache-report
```

The server is ready when this returns HTTP 200:

```bash
curl -f http://localhost:8000/ready
```

## GitHub Copilot BYOK

This configuration has been tested with GitHub Copilot BYOK through the OpenAI Chat Completions API. GLM 5.2 completed a full planning session with two parallel subagents and many tool calls without parser or protocol failures. The hybrid inference speed remains the practical limitation.

```json
{
  "name": "Local",
  "vendor": "customendpoint",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "glm-5.2-nvfp4",
      "name": "GLM 5.2 NVFP4",
      "url": "http://<server>:8000/v1/chat/completions",
      "toolCalling": true,
      "vision": false,
      "thinking": true,
      "streaming": true,
      "maxInputTokens": 131072,
      "maxOutputTokens": 32768
    }
  ]
}
```

## Benchmarks

Measured in August 2026 with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) 0.4.29 through an OpenAI-compatible llama-swap endpoint.

The benchmark used 30-second sustained decode cells, concurrency 1 and 2, no additional decode warmup, and integrated prefix-cache scout requests. Prefill throughput is client prompt tokens divided by time to first token. Decode values are aggregate sustained throughput across the requested concurrency.

### Prefill

| Context | Prompt tokens | TTFT | Throughput | Requests |
|---:|---:|---:|---:|---:|
| 8k | 8,198 | 18.41 s | 445 tok/s | 1 |
| 16k | 16,228 | 37.35 s | 434 tok/s | 1 |
| 32k | 32,319 | 75.93 s | 426 tok/s | 1 |
| 64k | 64,509 | 155.43 s | 415 tok/s | 1 |
| 128k | 128,878 | 321.16 s | 401 tok/s | 1 |

### Aggregate sustained decode

| Context | Concurrency 1 | Concurrency 2 |
|---:|---:|---:|
| 0 | 15.7 tok/s | 20.2 tok/s |
| 8k | 14.7 tok/s | 19.2 tok/s |
| 16k | 14.8 tok/s | 19.8 tok/s |
| 32k | 15.2 tok/s | 20.2 tok/s |
| 64k | 14.9 tok/s | 19.2 tok/s |
| 128k | 15.7 tok/s | 19.7 tok/s |

The mean across the six decode contexts was 15.2 tok/s at concurrency 1 and 19.7 tok/s aggregate at concurrency 2. At concurrency 2, per-request sustained throughput was approximately 9.6–10.1 tok/s.

The integrated scout primes the radix prefix used by the sustained decode cell, and the concurrent workers use the same generated context. The concurrency-2 table therefore measures aggregate cached-context decode scheduling; it does not measure the simultaneous uncached prefill of two unrelated 128k prompts. The 327,680-token capacity itself is verified by the server's initialization log.

These results describe this specific hybrid CPU/GPU placement and are not intended as general GLM 5.2, NVFP4 or FreeToken performance claims.

## Known limitations

- The image targets Linux/x86-64, CUDA 13 and SM120. Its prebuilt FreeToken kernel cache is not portable to older NVIDIA architectures.
- The tested recipe uses one 96 GB GPU and TP: 1.
- The model weights are not included. The NVIDIA checkpoint is approximately 465 GB and must be downloaded separately.
- The full two-request KV allocation leaves approximately 4.86 GiB after CUDA graph capture.
- Loading the raw Hugging Face checkpoint builds expert banks and took approximately four minutes on the tested machine. FreeToken's optional FTW conversion can improve subsequent loading, but it was not used for these benchmark numbers.
- Several Python dependencies are resolved from constrained ranges by pip rather than a lockfile. The component table records the exact versions in the tested image.
- `/ready` is a downstream patch and should be removed if an equivalent strict readiness route is added upstream.
- The concurrency-2 benchmark uses a shared, prefix-cached context and should not be interpreted as a two-independent-prompt prefill result.
