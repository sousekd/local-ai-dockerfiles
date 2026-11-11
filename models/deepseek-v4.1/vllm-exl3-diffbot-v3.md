# DeepSeek V4.1 — vLLM + EXL3 v3

Dockerfile for serving DeepSeek V4.1 Flash 1017 EXL3 on two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120) with tensor parallelism. The model quantization, kernels and base patches come from Diffbot and the projects credited in its recipe. [Upstream recipe and credits](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)

- [Dockerfile](Dockerfile.vllm-exl3-diffbot-v3)
- [Previous v2 recipe and benchmarks](vllm-exl3-diffbot-v2.md)
- [Upstream notices and licenses](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/recipe/THIRD_PARTY_NOTICES.md)

V3 pins Diffbot recipe `2205b25`. It retains the vLLM patch series from v2 and adds optional decoder replay skipping and tokenizer memoization for GPU expert execution. The memo retains references to cached token IDs across concurrent eviction. CPU expert execution also supports upstream's fixed MXFP8 prefill tactic and thread isolation. [Pinned upstream revision](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/commit/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea)

## Build

From the repository root, run:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg BUILD_JOBS=24 \
  --build-arg BUILD_CPU_EXPERTS=1 \
  -t diffbot-ds41f:v3 \
  - < models/deepseek-v4.1/Dockerfile.vllm-exl3-diffbot-v3
```

The build downloads source and compiles the TP2 kernels. Supply the checkpoint separately at launch. Serving requires a CUDA 13-compatible driver and the NVIDIA Container Toolkit. Build on the serving host when compiling CPU expert support because those kernels use `-march=native`. Set `BUILD_CPU_EXPERTS=0` to omit that runtime.

## Launch

Set the Hugging Face cache and a writable runtime cache. When upgrading from v2, reuse the existing runtime cache and checkpoint at `4e691b3`. Download the checkpoint only if that revision is missing from the cache.

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
RUNTIME_CACHE=/path/to/diffbot-runtime-cache
MODEL_REPO=diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
MODEL_REVISION=4e691b368c9cb06855c058e80f244b251fed050b

mkdir -p "$RUNTIME_CACHE"
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --cache-dir "$HF_HUB_CACHE"
```

Mount the whole Hub cache so its snapshot symlinks can resolve. The following configuration uses GPU expert execution, a 512K context limit, DSpark with five draft tokens and the native 256 GiB RAM tier for KV cache. It keeps Diffbot's default precision settings as the performance baseline.

```bash
docker run --rm \
  --name deepseek-v4.1-flash \
  --init --ipc=private --shm-size=384g \
  --cap-add=SYS_NICE --cap-add=IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --runtime=nvidia --gpus all \
  -v "$HF_HUB_CACHE:/models:ro" \
  -v "$RUNTIME_CACHE:/root/.cache:rw" \
  -e HF_HUB_CACHE=/models \
  -e MODEL_REVISION="$MODEL_REVISION" \
  -e SERVED_MODEL_NAME=deepseek-v4.1-flash \
  -e CPU_EXPERTS=0 \
  -e MAX_MODEL_LEN=524288 \
  -e KV_MEM=4294967296 \
  -e KV_OFFLOAD_GB=256 \
  -e ALLOC_CONF=expandable_segments:False \
  -e MAX_NUM_SEQS=8 \
  -e MAX_BATCHED=16384 \
  -e SPEC_K=5 \
  -e DO_INT8=1 \
  -e DS41_FP8AR=1 \
  -e DS41_FP8_LMHEAD=1 \
  -e VLLM_DS41_REPLAY_SKIP=1 \
  -e DS41_TOKENIZER_MEMO=1 \
  -p 8000:8000 \
  diffbot-ds41f:v3 "$MODEL_REPO" \
  --safetensors-load-strategy prefetch \
  --enable-request-id-headers \
  --enable-force-include-usage \
  --enable-per-request-metrics \
  --enable-prompt-tokens-details
```

The entrypoint serves offline from a cached Hub ID or a local snapshot path. It configures the DeepSeek encoder, reasoning parser and tool-call parser. `DSV41_ENGRAM_DISK=0` is set in the image, so Engram tables use pinned host RAM.

With `CPU_EXPERTS=0`, set `VLLM_DS41_REPLAY_SKIP` and `DS41_TOKENIZER_MEMO` individually to enable either optimization. Both default to `0` on this path. Replay skipping avoids work on eligible placeholder-only prefill chunks. The tokenizer memo reuses encoded text segments across requests. Both default to `1` when `CPU_EXPERTS=1` and `DS41_CPU_SPEEDUPS=1`. CPU expert execution requires the optional build and a supported host CPU. [Upstream CPU runtime documentation](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/recipe/cpuexp/README.md)

## Precision options

| Set at launch | Effect |
|---|---|
| `DO_INT8=0` | Uses FP16 expert prefill arithmetic. |
| `DS41_FP8AR=0` | Uses uncompressed NCCL all-reduce for prefill. |
| `DS41_FP8_LMHEAD=0` | Retains the BF16 output heads, using more VRAM. |

Try these settings against the baseline with the same prompts and sampling parameters. Diffbot measured changes within run-to-run noise for the two prefill approximations. Higher-precision prefill can increase time to first token. [Upstream fidelity measurements](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/2205b2520c1aefd92d4fbd46f58ff1bffdc589ea/docs/FIDELITY.md)

## Memory and startup

`KV_MEM` is bytes per GPU. The launch above reserves 4 GiB per GPU for FP8 KV, yielding a reported GPU pool of 813,859 tokens. `MAX_NUM_SEQS=8` allows eight sequence slots, subject to that pool. Two independent full 512K contexts exceed the GPU pool.

`KV_OFFLOAD_GB` sets the shared RAM tier in GiB. Keep `ALLOC_CONF=expandable_segments:False` and enough private `/dev/shm` capacity for the tier and process communication. Allow roughly 260 GiB of host RAM for weights and runtime in addition to the RAM tier. Set `KV_OFFLOAD_GB=0` to disable the tier.

The 512K configuration reached API readiness in about 13 minutes from model loading on two RTX PRO 6000 Blackwell Workstation Edition cards. Its startup log confirmed PCIe IPC during CUDA graph capture, the shared RAM tier and tokenizer memo activation. The replay skip path logs `placeholder steps skipped` when exercised by an eligible request. For measured throughput, see the [v2 benchmark results](vllm-exl3-diffbot-v2.md#benchmarks).

Inline system messages in `/v1/messages` can reduce prefix reuse when the renderer moves them to the start of the prompt.

## Verify the server

```bash
curl -fsS http://localhost:8000/health

curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
    "reasoning_effort": "none",
    "chat_template_kwargs": {"thinking": false},
    "max_tokens": 128
  }'
```
