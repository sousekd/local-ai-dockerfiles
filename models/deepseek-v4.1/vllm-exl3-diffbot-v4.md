# DeepSeek V4.1 Flash 1017 — vLLM + EXL3 v4

Dockerfile for Diffbot’s 3-bit EXL3 checkpoint on two 96 GB NVIDIA RTX PRO 6000 Blackwell GPUs (SM120, tensor parallel size 2). Requires a CUDA 13 compatible driver, the NVIDIA Container Toolkit, and about 260 GiB of available host RAM for the default profile.

- [Dockerfile](Dockerfile.vllm-exl3-diffbot-v4)
- [Diffbot recipe at `2205b25`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/tree/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/recipe)
- [Upstream notices and licenses](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/recipe/THIRD_PARTY_NOTICES.md)
- [v2 recipe and benchmarks](vllm-exl3-diffbot-v2.md#benchmarks)

## Upstream defaults

The image uses Diffbot’s pinned vLLM nightly (`5555a2d8`, `0.31.1rc1.dev376+g5555a2d8f`), the same patch series, vLLM EXL3 `0.4.2` (`d3cfd394`), and ExLlamaV3 `1.4.9` (`5be88657`). It builds the five upstream TP2 kernels and packages the recipe, including `sitecustomize.py`, inside the image. The entrypoint derives its serving settings and seven warmup requests from the pinned `serve.sh`.

| Setting | Default |
|---|---|
| Context and GPU KV | 1,048,576 tokens, FP8 KV, 4.25 GiB per GPU |
| Scheduling | 8 sequence slots, 16,384 batched tokens |
| Expert execution | GPU execution with pinned host weights and 512 GPU cache slots per rank |
| Precision | INT8 expert prefill, 8-bit prefill all-reduce, FP8 LM head, INT4 Engram in host RAM |
| Speculation | Released DSpark drafter, 5 speculative tokens |
| Chat and vision | DeepSeek V4.1 tokenizer, reasoning parser and tool parser, vision enabled |
| Host KV tier | Disabled (`KV_OFFLOAD_GB=0`) |
| CUDA allocator | `expandable_segments:True` |

CPU expert overlays are selected only when the CPU runtime is active and `DS41_CPU_SPEEDUPS=1`. The final-window replay optimization is enabled on the default GPU path. See [Diffbot’s launcher](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/recipe/serve/serve.sh) for the full set of environment overrides.

## Build

Run from the repository root. The build fetches recipe and kernel sources. The checkpoint is supplied at launch.

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg BUILD_JOBS=24 \
  -t diffbot-ds41f:v4 \
  - < models/deepseek-v4.1/Dockerfile.vllm-exl3-diffbot-v4
```

For CPU expert execution, add `--build-arg BUILD_CPU_EXPERTS=1` on the serving host, then pass `-e CPU_EXPERTS=1` at launch. The optional CPU kernels use `-march=native`. See the [CPU runtime requirements](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/recipe/cpuexp/README.md).

## Run

Set the cache paths and populate the Hugging Face Hub cache with the pinned checkpoint if needed.

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
RUNTIME_CACHE=/path/to/diffbot-runtime-cache
MODEL_REPO=diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
MODEL_REVISION=4e691b368c9cb06855c058e80f244b251fed050b

mkdir -p "$RUNTIME_CACHE"
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --cache-dir "$HF_HUB_CACHE" \
  --include 'model-*.safetensors' --include '*.json'
```

Mount the whole Hub cache so snapshot symlinks resolve. The entrypoint accepts a cached Hub ID or a local model directory mounted inside the container. Expose the two intended GPUs. Extra arguments after the model go to `vllm serve`.

```bash
docker run --rm --name deepseek-v4.1-flash \
  --init --ipc=private --shm-size=32g \
  --cap-add=SYS_NICE --cap-add=IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --runtime=nvidia --gpus all \
  -v "$HF_HUB_CACHE:/models:ro" \
  -v "$RUNTIME_CACHE:/root/.cache:rw" \
  -e HF_HUB_CACHE=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e MODEL_REVISION="$MODEL_REVISION" \
  -p 127.0.0.1:8000:8000 \
  diffbot-ds41f:v4 "$MODEL_REPO"
```

The default served model name is `deepseek-v4.1-flash`. Set `SERVED_MODEL_NAME` to override it. To require bearer authentication, export `API_KEY` and pass `-e API_KEY` before the image name. Keep the writable runtime cache between launches to reuse compiled and tuned kernels.

## Tested 512K configuration

Apply these Docker overrides before the image name. Replace the shared memory limit in the example with `--shm-size=384g`.

| Override | Value |
|---|---|
| `MAX_MODEL_LEN` | `524288` |
| `KV_MEM` | `4294967296` bytes per GPU |
| `KV_OFFLOAD_GB` | `256` GiB shared host KV tier |
| `ALLOC_CONF` | `expandable_segments:False` |

The shared host KV tier needs additional host RAM and `/dev/shm` space. This setup reported **813,859 GPU KV tokens** and an estimated **1.55× concurrency** at the full 524,288-token limit. The cold launch reached API readiness in about **21 minutes**, including **17m 28s** of model loading. All seven warmup requests returned HTTP 200 in **36s**. The run confirmed INT8 expert prefill, 8-bit prefill all-reduce, FlashInfer PCIe IPC during CUDA graph capture, and writes to the host KV tier.

The tested launch also appended `--enable-request-id-headers`, `--enable-force-include-usage`, `--enable-per-request-metrics`, and `--enable-prompt-tokens-details` after the model argument.

For startup comparisons on VirtioFS, test `--safetensors-load-strategy prefetch` after the model argument. The cold run used the default loader, which skipped automatic prefetch on VirtioFS.

## Verify

Wait for `[warmup] done` before collecting timings. If `API_KEY` is configured, include `-H "Authorization: Bearer $API_KEY"` on the completion request.

```bash
curl -fsS http://127.0.0.1:8000/health

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
    "chat_template_kwargs": {"thinking": false},
    "temperature": 0,
    "max_tokens": 128
  }'
```

For benchmark reports, record the recipe and checkpoint revisions, context, KV settings, and any request or engine overrides. With the pinned checkpoint, omitted chat sampling parameters use `temperature=1.0` and `top_p=1.0`.
