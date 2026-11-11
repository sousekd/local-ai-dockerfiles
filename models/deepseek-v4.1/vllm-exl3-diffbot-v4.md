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

Build from the repository root.

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
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

Mount the full Hub cache and a writable runtime cache. The model argument accepts a cached Hub ID or a local directory inside the container. Subsequent arguments go to `vllm serve`.

```bash
docker run --rm --name deepseek-v4.1-flash \
  --init --ipc=private --shm-size=32g \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --gpus all \
  -v "$HF_HUB_CACHE:/models:ro" \
  -v "$RUNTIME_CACHE:/root/.cache:rw" \
  -e HF_HUB_CACHE=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e MODEL_REVISION="$MODEL_REVISION" \
  -p 127.0.0.1:8000:8000 \
  diffbot-ds41f:v4 "$MODEL_REPO"
```

Set `SERVED_MODEL_NAME` to change the API model ID from `deepseek-v4.1-flash`. For bearer authentication, export `API_KEY` and pass `-e API_KEY` before the image name.

## Verify

If `API_KEY` is configured, include `-H "Authorization: Bearer $API_KEY"` on the completion request.

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
