# GLM 5 Next — SGLang + KTransformers v1

SGLang and KTransformers image for hybrid CPU/GPU inference of the GLM 5 Next model family on one NVIDIA Blackwell GPU.

- [Dockerfile](Dockerfile.sglang-ktransformers-next-v1)
- [KTransformers GLM-5.3 Flash tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [GLM-5.3-Flash checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash)

This GLM 5 Next recipe has been tested and benchmarked with the official GLM-5.3-Flash FP8 checkpoint. It uses the released KTransformers wheel stack, the native AVX-512 CPU expert backend, layerwise GPU prefill and the model-specific SM120 DSA kernels.

## Tested configuration

- Model: `zai-org/GLM-5.3-Flash`
- Model revision: `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`
- Architecture: GLM-5-Next, approximately 321B parameters, 34 KDA layers and 11 DSA layers
- Checkpoint and CPU expert method: FP8
- Resident GPU routed experts: 0 per MoE layer
- CPU expert backend: AVX-512 BF16
- KV and KPool caches: FP8 E4M3, 1,048,576 aggregate tokens
- GPU: 1 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB
- CPU: AMD EPYC 9355 Turin, 26 worker threads on one NUMA node
- Host memory: 768 GB DDR5-6400
- NVIDIA driver: 610.57.04
- Container CUDA toolkit: 12.9.1
- CUDA architecture: SM120
- Tensor parallelism: TP: 1
- Maximum concurrent requests: 2
- Maximum context per request: 524,288 tokens
- Prefill chunk size: 16,384 tokens

## Component versions

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04@sha256:a2e1e2360c85298ac47ec2543b406ab1e8cec42e31ee47e4d32140ebc82e1067` |
| KTransformers | `0.7.0.post2` |
| KT-Kernel | `0.7.0.post2` |
| SGLang-KT | `0.7.0.post2` |
| Transformers-KT | `5.6.0.post4` |
| SGL kernel payload | `0.3.21.post2` |
| Python | 3.12 |
| PyTorch | `2.9.1+cu129` |
| FlashInfer | `0.6.3` |
| Quack Kernels | `0.2.4` |
| NVIDIA Cutlass DSL | `4.3.5` |
| CUDA Python | `13.2.0` |
| CUDA architecture | SM120 |

## Design decisions

### Coordinated release wheels

The image follows the official installation path:

```text
ktransformers[sglang]==0.7.0.post2
```

KTransformers, KT-Kernel and SGLang-KT are kept on the same release. Transformers-KT is pinned to the exact version required by SGLang-KT. PyTorch cu129, FlashInfer, Quack Kernels, Cutlass DSL and CUDA Python are pinned to the tested compatible versions.

The large SGL kernel payload is split across the four coordinated wheels. The Docker build verifies every payload part, the exact manifest version and archive checksum, the AVX-512 BF16 KT extension, Torch cu129 and the CUDA 12.9 toolkit.

### Model configuration overlay

The tested model revision sets:

```json
"index_share_for_mtp_iteration": true
```

SGLang-KT 0.7.0.post2 does not run MTP for GLM 5 Next and rejects this metadata setting. The launch recipe mounts a copy with the field set to `false`. The Hugging Face cache and model weights remain unchanged.

### SM120 execution profile

On SM120, SGLang-KT selects the `blackwell_fp8` profile automatically:

- FP8 E4M3 KV cache;
- FP8 E4M3 KPool cache;
- NSA attention;
- the model-local TRT-LLM dispatcher for graph-safe SM120 kernels;
- Triton MoE execution around the native KTransformers CPU experts.

### Layerwise prefill and CPU threads

The tested configuration exposes one 32-core NUMA node to the VM and uses one 26-thread KTransformers pool. Layerwise prefill starts at 2,048 tokens, keeps zero resident GPU experts and uses 16,384-token chunks.

### Runtime memory

The tested runtime allocated:

- 15.79 GB for GPU weights and model state before cache allocation;
- approximately 0.41 GB for KDA/Mamba state;
- 6.92 GB for 1,048,576 FP8 KV tokens;
- 0.35 GB for CUDA graphs at batch sizes 1 and 2.

### Prefix caching disabled

SGLang-KT 0.7.0.post2 forcibly disables radix prefix caching for GLM 5 Next. The request-local KPool live tail is not restored on a cache hit, so force-enabling the ordinary radix cache could produce incorrect results. The same compatibility gate rejects HiCache, LMCache and HiSparse.

Every new request therefore recomputes its complete prompt. This is a major limitation for multi-turn agentic workflows even though cold prefill is relatively fast.

## Build

Run from the repository root:

```bash
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -f models/glm-5/Dockerfile.sglang-ktransformers-next-v1 \
  -t sglang-kt-glm-next:v1 \
  .
```

## Prepare the model configuration overlay

Set the Hugging Face cache, exact model snapshot and a persistent path for the patched configuration:

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
MODEL_SNAPSHOT=models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a
PATCHED_CONFIG=/path/to/GLM-5.3-Flash-KTransformers.config.json

HOST_CONFIG="${HF_HUB_CACHE}/${MODEL_SNAPSHOT}/config.json"

sed \
  's/"index_share_for_mtp_iteration": true/"index_share_for_mtp_iteration": false/' \
  "${HOST_CONFIG}" > "${PATCHED_CONFIG}"

test "$(jq -r '.text_config.index_share_for_mtp_iteration' "${PATCHED_CONFIG}")" = false
```

The overlay is tied to the pinned model revision.

## Launch GLM 5.3 Flash FP8

Resolve the checkpoint's configuration blob so the read-only overlay follows the Hugging Face snapshot symlink:

```bash
CONTAINER_HF_HUB=/root/.cache/huggingface/hub
CONTAINER_MODEL="${CONTAINER_HF_HUB}/${MODEL_SNAPSHOT}"
CONFIG_BLOB_REL="$(realpath --relative-to="${HF_HUB_CACHE}" "${HOST_CONFIG}")"
CONTAINER_CONFIG="${CONTAINER_HF_HUB}/${CONFIG_BLOB_REL}"
```

Launch the server on GPU 0:

```bash
docker run --rm \
  --name glm-5.3-flash-fp8 \
  --init \
  --ipc=host \
  --cap-add=SYS_NICE \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --ulimit nofile=1048576:1048576 \
  --runtime=nvidia \
  --gpus device=0 \
  -v "${HF_HUB_CACHE}:${CONTAINER_HF_HUB}:ro" \
  -v "${PATCHED_CONFIG}:${CONTAINER_CONFIG}:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p 8000:8000 \
  sglang-kt-glm-next:v1 \
  python -m sglang.launch_server \
  --model-path "${CONTAINER_MODEL}" \
  --kt-weight-path "${CONTAINER_MODEL}" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --context-length 524288 \
  --max-running-requests 2 \
  --prefill-max-requests 2 \
  --max-total-tokens 1048576 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --kt-method FP8 \
  --kt-cpuinfer 26 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 0 \
  --kt-gpu-prefill-token-threshold 2048 \
  --cuda-graph-bs 1 2 \
  --limit-mm-data-per-request '{"image":8,"video":1}' \
  --mm-process-config '{"image":{"max_pixels":1254400}}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --sampling-defaults model \
  --enable-metrics
```

The server is ready when this returns HTTP 200:

```bash
curl -f http://localhost:8000/health
```

## Benchmarks

Measured in August 2026 with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) 0.4.29 through an OpenAI-compatible llama-swap endpoint.

The benchmark used 30-second sustained decode cells, concurrency 1 and 2, no additional decode warm-up, 16,384-token prefill chunks and contexts up to 128k. Prefill throughput is client prompt tokens divided by time to first token. Decode values are aggregate sustained throughput after the request contexts have been admitted.

### Prefill

| Context | Prompt tokens | TTFT | Throughput | Requests |
|---:|---:|---:|---:|---:|
| 8k | 8,199 | 9.66 s | 849 tok/s | 1 |
| 16k | 16,229 | 14.29 s | 1,136 tok/s | 1 |
| 32k | 32,320 | 29.91 s | 1,081 tok/s | 1 |
| 64k | 64,510 | 60.85 s | 1,060 tok/s | 1 |
| 128k | 128,879 | 132.83 s | 970 tok/s | 1 |

### Aggregate sustained decode

| Context | Concurrency 1 | Concurrency 2 |
|---:|---:|---:|
| 0 | 13.3 tok/s | 15.8 tok/s |
| 8k | 13.4 tok/s | 15.5 tok/s |
| 16k | 13.3 tok/s | 15.8 tok/s |
| 32k | 13.3 tok/s | 15.7 tok/s |
| 64k | 13.3 tok/s | 15.3 tok/s |
| 128k | 13.3 tok/s | 15.8 tok/s |

At concurrency 2, per-request sustained throughput was approximately 7.7–7.9 tok/s.

Because prefix caching is disabled, the prefill table represents cold-context processing. The sustained decode cells begin after the requested contexts have been admitted and therefore do not include the repeated full-prompt prefill cost that a multi-turn agent experiences.

These results describe this specific CPU/GPU placement and are not intended as general GLM-5.3-Flash, FP8 or KTransformers performance claims.

## Known limitations

- The image targets Linux/x86-64, CUDA 12.9 and SM120. Only one RTX PRO 6000 and TP: 1 are benchmarked here.
- The official FP8 checkpoint is approximately 306 GiB and requires at least 350 GB of available system memory.
- Model weights are not included.
- Loading the checkpoint, initializing the CPU experts and capturing CUDA graphs took approximately nine minutes on the tested system.
- Radix prefix caching, HiCache, LMCache and HiSparse are disabled by the GLM 5 Next compatibility gate. Multi-turn requests must prefill their full prompt again.
- MTP/speculative decoding is not supported by this SGLang-KT release.
- The model configuration overlay is required by `0.7.0.post2`. Later compatible releases can use the checkpoint metadata directly.
- Layerwise prefill forces the persistent GPU expert count to zero.
- Decode remains limited by CPU expert execution and host-memory bandwidth; the GPU is lightly utilized during steady decode.
- No model-provided FP8 KV scale factors were found, so SGLang defaults them to 1.0 and warns about possible accuracy loss.
- Several transitive Python dependencies are resolved from constrained ranges rather than a lockfile. The component table records the performance-critical versions in the tested image.
- The pinned Transformers-KT release emits generic Transformers 5.x RoPE and image-processor deprecation warnings. No RoPE failure was observed; multimodal serving was not benchmarked.
