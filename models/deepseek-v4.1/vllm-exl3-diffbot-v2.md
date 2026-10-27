# DeepSeek V4.1 — vLLM + EXL3 v2

Convenience Dockerfile for [Diffbot's DeepSeek V4.1 Flash EXL3 recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) on two NVIDIA Blackwell GPUs. The quantization, patches, kernels and tuning come from Diffbot and the upstream projects credited there.

- [Dockerfile](Dockerfile.vllm-exl3-diffbot-v2)
- [Diffbot deployment documentation](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/main/recipe/README.md)
- [Upstream notices and licenses](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/main/recipe/THIRD_PARTY_NOTICES.md)

The image packages the pinned recipe and provides a foreground OpenAI-compatible server entrypoint. Model weights are downloaded separately and mounted read-only.

## Tested configuration

- Model: `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000`, Flash 1017 release
- Benchmark model and recipe revision: `4e691b368c9cb06855c058e80f244b251fed050b`
- GPU: 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each, 350 W per GPU
- CPU: AMD EPYC 9355 Turin
- VM memory: approximately 740 GiB
- Tensor parallelism: TP: 2
- Expert execution: GPU (`CPU_EXPERTS=0`), including experts accessed from host RAM
- KV cache: FP8 sparse MLA, 4 GiB per GPU; reported capacity 813,859 tokens
- CPU KV tier: 256 GiB through vLLM's native `OffloadingConnector`
- Maximum concurrent requests: 8, subject to available GPU KV capacity
- Maximum context per request: 524,288 tokens
- Prefill chunk: 16,384 tokens; DSpark speculative decoding: 5 draft tokens

## Component versions

| Component | Version |
|---|---|
| vLLM | `0.31.1rc1.dev376+g5555a2d8f` |
| vllm-exl3 | `0.4.2` |
| ExLlamaV3 | `1.4.9` |
| Python / CUDA toolkit | 3.12 / 13.0 |
| GPU architecture | SM120a |

The base image is pinned by digest, and recipe and component sources are pinned by commit in the Dockerfile. The build applies Diffbot's patches, compiles the TP2 kernels and records Python packages in `/opt/installed-packages.txt`.

The Dockerfile uses recipe revision [`451f7dc`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/commit/451f7dcb8ecb0502a963f3314e16b34c5a8ce5ae), including all 12 upstream vLLM patches. The 25 October update preserves tool namespaces and makes explicit thinking settings take precedence over `reasoning_effort` in prompt rendering and response parsing. Weights and kernels are unchanged; the cached model revision below remains usable. Rebuild the image and restart to apply the fixes.

## Build

Run from the repository root:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg BUILD_JOBS=24 \
  --build-arg BUILD_CPU_EXPERTS=1 \
  -t diffbot-ds41f:v2 \
  - < models/deepseek-v4.1/Dockerfile.vllm-exl3-diffbot-v2
```

Only the Dockerfile is required as local build input. The build downloads source code, including the recipe from Hugging Face, but no model weights. A CUDA 13-compatible driver and NVIDIA Container Toolkit are required for serving.

CPU kernels use `-march=native`; build on the serving host. Set `BUILD_CPU_EXPERTS=0` to omit the optional CPU runtime and its CPU feature requirements.

## Launch DeepSeek V4.1 Flash 1017 EXL3

Set the local Hugging Face cache and a writable runtime-cache directory:

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
RUNTIME_CACHE=/path/to/diffbot-ds41f-v2-cache
MODEL_REPO=diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
MODEL_REVISION=4e691b368c9cb06855c058e80f244b251fed050b

mkdir -p "${RUNTIME_CACHE}"
hf download "${MODEL_REPO}" \
  --revision "${MODEL_REVISION}" --cache-dir "${HF_HUB_CACHE}"
```

Mount the whole Hub cache so snapshot symlinks can resolve their blobs. The following command uses GPU expert execution and a 256 GiB RAM KV tier:

```bash
docker run --rm \
  --name deepseek-v4.1-flash \
  --init --ipc=private --shm-size=384g \
  --cap-add=SYS_NICE --cap-add=IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  --runtime=nvidia --gpus all \
  -v "${HF_HUB_CACHE}:/models:ro" \
  -v "${RUNTIME_CACHE}:/root/.cache:rw" \
  -e HF_HUB_CACHE=/models \
  -e HF_HUB_OFFLINE=1 \
  -e MODEL_REVISION="${MODEL_REVISION}" \
  -e SERVED_MODEL_NAME=deepseek-v4.1-flash \
  -e CPU_EXPERTS=0 \
  -e MAX_MODEL_LEN=524288 \
  -e KV_MEM=4294967296 \
  -e KV_OFFLOAD_GB=256 \
  -e ALLOC_CONF=expandable_segments:False \
  -e MAX_NUM_SEQS=8 \
  -e MAX_BATCHED=16384 \
  -e SPEC_K=5 \
  -p 8000:8000 \
  diffbot-ds41f:v2 "${MODEL_REPO}" \
  --safetensors-load-strategy prefetch \
  --enable-request-id-headers \
  --enable-force-include-usage \
  --enable-per-request-metrics \
  --enable-prompt-tokens-details
```

The entrypoint resolves cached model IDs offline; a local snapshot directory is also accepted. It supplies the served model name, remote-code trust, DeepSeek encoder, reasoning parser and tool-call parser. No external chat template is required.

`prefetch` was useful with checkpoints mounted through VirtioFS. Initialization still took approximately 13 minutes with the RAM tier enabled; allow sufficient readiness time.

## Runtime memory and CPU options

`CPU_EXPERTS=0` keeps expert computation on the GPUs, while host RAM still stores some experts and Engram tables. Allow approximately 260 GiB of free host RAM before adding a KV-offload tier.

`KV_OFFLOAD_GB` is GiB total across workers. The native tier uses pinned host memory backed by `/dev/shm`; `--shm-size` is its capacity limit, not a separate RAM reservation. The tier is temporary and disappears when the container exits. It retains reusable prefixes beyond the GPU cache; the active-request GPU KV capacity and per-request context limit still apply.

For a smaller-memory launch, set `KV_OFFLOAD_GB=0` and use `--shm-size=64g`. With offload enabled, retain `ALLOC_CONF=expandable_segments:False`. The same runtime-cache directory can be reused across context sizes and expert-execution modes. An existing v1 runtime-cache directory can also be reused with v2; no cleanup is required.

**Optional CPU-assisted expert execution has not been tested or benchmarked for this recipe.** With `BUILD_CPU_EXPERTS=1`, set `CPU_EXPERTS=1` at runtime. `CPU_EXPERTS_PREFILL=0` limits CPU assistance to decode; its default of `1` also enables CPU prefill. CPU expert execution and CPU KV caching are independent settings. See [Diffbot's CPU runtime documentation](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/main/recipe/cpuexp/README.md).

## Verify the server

```bash
curl -f http://localhost:8000/health

curl -f http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
    "temperature": 1.0,
    "top_p": 1.0,
    "reasoning_effort": "none",
    "chat_template_kwargs": {"thinking": false},
    "max_tokens": 128
  }'
```

For maximum reasoning, use `reasoning_effort: "max"` and `chat_template_kwargs: {"thinking": true}`.

## Benchmarks

Measured on 25 October 2026 with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) 0.8.0, GPU expert execution and the launch settings above. Decode cells ran for 30 seconds without additional decode warm-up. Prefill values are client prompt tokens divided by time to first token, with one scout request per context. Decode values are aggregate throughput in tok/s.

### Prefill

| Context | Prompt tokens | TTFT | Throughput |
|---:|---:|---:|---:|
| 8k | 8,190 | 0.80 s | 10,260 tok/s |
| 16k | 16,211 | 1.43 s | 11,346 tok/s |
| 32k | 32,253 | 2.85 s | 11,327 tok/s |
| 64k | 64,348 | 5.62 s | 11,451 tok/s |

### Aggregate sustained decode

| Context | Concurrency 1 | Concurrency 2 | Concurrency 4 | Concurrency 8 |
|---:|---:|---:|---:|---:|
| 0 | 82.3 | 121.9 | 163.7 | 203.7 |
| 8k | 104.1 | 131.3 | 165.1 | 211.6 |
| 16k | 90.2 | 129.0 | 163.0 | 220.8 |
| 32k | 94.3 | 128.9 | 155.1 | 209.5 |
| 64k | 91.0 | 111.6 | 162.7 | 208.8 |

Diffbot's bundled code benchmark separately reported 190.1 tok/s for one 46K-context stream and 363.7 tok/s aggregate for four. These are different workloads; their results are not directly comparable with the sustained-decode matrix.

## Known limitations

- This image builds the two-GPU TP2 profile only. Other checkpoint revisions and hardware configurations are not validated here.
- Eight request slots do not imply eight simultaneous 512K contexts; the reported GPU KV capacity permits approximately 1.55 full-length requests.
- RAM-tier initialization is verified, but its hit rate and performance benefit are workload-dependent; the benchmark does not isolate that benefit.
- Diffbot's default int8 prefill kernel and 8-bit prefill all-reduce are approximate. Set `DO_INT8=0` and `DS41_FP8AR=0` for their exact alternatives.
- The `/v1/messages` encoder merges inline system messages into the leading prompt, which can invalidate prefix reuse.
- Upstream licensing applies, including AGPL-3.0-only for the patched vllm-exl3 component. Notices are retained under `/opt/recipe`.
