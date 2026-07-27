# Kimi K2 — SGLang + KTransformers v3

SGLang and KTransformers image for hybrid CPU/GPU inference of the Kimi K2 model family.

- [Dockerfile](Dockerfile.sglang-ktransformers-v3)

This version has been tested in production with Kimi K2.7 Code RAWINT4, including agentic tool calling through GitHub Copilot BYOK.

## Tested configuration

- Model: `moonshotai/Kimi-K2.7-Code`
- Quantization: RAWINT4
- GPU: 1 × NVIDIA RTX PRO 6000 Blackwell, 96 GB
- CPU: AMD EPYC 9355 Turin
- Host memory: 768 GB DDR5-6400
- CUDA: 13
- Tensor parallelism: TP: 1
- Maximum concurrent requests: 2
- Context length: 163,840 tokens
- Frontend: GitHub Copilot BYOK using the OpenAI Chat Completions API

The same SGLang and KTransformers configuration should work with Kimi K2.6 and Kimi K2.5 by selecting the corresponding checkpoint and chat template. Only Kimi K2.7 Code RAWINT4 is benchmarked here.

GitHub Copilot currently provides model-specific compatibility handling for Kimi K2.6 and Kimi K2.7 Code, but not Kimi K2.5.

## Component versions

| Component | Version or revision |
|---|---|
| Base image | `lmsysorg/sglang:dev-cu13` |
| KTransformers | `d1a3ed8a308cf45a2bdf8dc0ec18ea0cf782486c` |
| KTransformers SGLang submodule | `1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee` |
| SGL kernel | `0.3.21+cu130` |
| Kimi parser backport | `af2a2ac618394a3c2d36c194f6b4ff346758ea7c` |
| cuDNN | `9.16.0.29` |
| CUDA architecture | SM120 |

## Changes from v2

Version 3 retains the working SGLang-KT and RAWINT4 stack from v2 and concentrates on Kimi tool-calling reliability:

- Backports a newer upstream Kimi K2 tool-call parser.
- Adds response-local tool-call indexing.
- Buffers incomplete tool markers split between streamed chunks.
- Keeps the RAWINT4 layerwise-prefill restoration fix.
- Keeps the native AMD Turin and Blackwell SM120 kernel build.

## Design decisions and patches

### Pinned SGLang-KT stack

The base image follows the rolling SGLang CUDA 13 development image, but its SGLang, KTransformers, kernel, Transformers-KT and FlashInfer packages are not assumed to be mutually compatible.

The Dockerfile removes that stack and installs:

- KTransformers at a known working commit;
- the matching SGLang submodule selected by that commit;
- the corresponding CUDA 13 SGL kernel wheel;
- the Python dependencies required by the pinned SGLang-KT package.

This avoids mixing a current rolling SGLang installation with an older KTransformers integration.

### Native Turin and SM120 build

`kt-kernel` is compiled on the target host with:

- AVX-512 VNNI;
- AVX-512 BF16;
- AVX-512 VBMI;
- CUDA SM120 support.

The resulting image is hardware-oriented. Build it on the same class of machine on which it will run.

### CUDA 13 PTX assembler

Triton's bundled `ptxas` may not support SM120. The Dockerfile makes Triton use the assembler provided by the installed CUDA 13 toolkit.

### Flash Attention cleanup

Uninstalling the base image's standalone `flash-attn-4` package can leave an unregistered `flash_attn` directory behind. It is removed to prevent stale files from interfering with the pinned package stack.

### cuDNN 9.16

Torch 2.9.1 normally brings cuDNN 9.13. The image installs cuDNN 9.16 because Kimi multimodal processing can reach a BF16 Conv3d path that requires cuDNN 9.15 or newer.

### RAWINT4 weight restoration

The pinned KTransformers SGLang fork can recreate layerwise-prefill weights using a `torch.empty()` call that inherits gradient tracking.

The downstream patch creates these temporary weights with:

```python
requires_grad=False
```

This stabilizes RAWINT4 layerwise-prefill weight restoration.

### Kimi parser backport

The Kimi parser bundled with the pinned SGLang-KT fork predates several relevant upstream fixes. Version 3 backports a compatible parser that supports:

- standard Kimi function IDs;
- bare numeric tool-call counters;
- hyphenated tool names;
- tool-name inference from argument schemas.

The parser is taken from a fixed upstream revision rather than from the moving `main` branch.

### Response-local tool indexes

Kimi may emit tool-call counters that continue across the conversation. SGLang's serving layer already adds the number of historical tool calls when constructing response IDs.

Passing the model's conversation-level counter through the parser causes that history offset to be applied twice. The patch instead returns zero-based indexes local to the current response.

### Streaming marker buffering

A streamed special token such as:

```text
<|tool_call_begin|>
```

may be split across multiple network chunks. The original parser could treat the incomplete first fragment as ordinary assistant text and clear its buffer.

The patch retains any trailing fragment that could be the beginning of a Kimi tool marker until the remaining characters arrive.

### DeepGEMM disabled at launch

The launch recipe uses:

```bash
-e SGLANG_ENABLE_JIT_DEEPGEMM=0
```

and:

```bash
--fp8-gemm-backend triton
```

This prevents JIT DeepGEMM initialization and selects the more stable Triton path for the pinned stack on SM120.

## Build

Run from the repository root:

```bash
docker build \
  --progress=plain \
  -f models/kimi-k2/Dockerfile.sglang-ktransformers-v3 \
  -t sglang-kt-kimi:v3 \
  .
```

## Launch Kimi K2.7 Code

Specifying a chat-template file is optional. Without `--chat-template`, SGLang uses the template included with the model checkpoint.

To explicitly use the latest official Kimi K2.7 Code template, download it from [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/main/chat_template.jinja):

```bash
curl -fL \
  https://huggingface.co/moonshotai/Kimi-K2.7-Code/resolve/main/chat_template.jinja \
  -o Kimi-K2.7-Default.jinja
```

Set the locations of the Hugging Face cache and model snapshot:

```bash
HF_HUB_CACHE=/path/to/huggingface/hub
MODEL_SNAPSHOT=models--moonshotai--Kimi-K2.7-Code/snapshots/<snapshot-id>
CHAT_TEMPLATE=/path/to/Kimi-K2.7-Default.jinja
```

Launch the server on GPU 0:

```bash
docker run --rm \
  --name kimi-k2.7-code \
  --ipc=host \
  --cap-add=SYS_NICE \
  --runtime=nvidia \
  --gpus device=0 \
  -v "${HF_HUB_CACHE}:/root/.cache/huggingface/hub:ro" \
  -v "${CHAT_TEMPLATE}:/opt/templates/Kimi-K2.7-Default.jinja:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -p 8000:8000 \
  sglang-kt-kimi:v3 \
  python -m sglang.launch_server \
  --model-path "/root/.cache/huggingface/hub/${MODEL_SNAPSHOT}" \
  --kt-weight-path "/root/.cache/huggingface/hub/${MODEL_SNAPSHOT}" \
  --served-model-name kimi-k2.7-code \
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
  --kt-num-gpu-experts 8 \
  --kt-method RAWINT4 \
  --kt-gpu-prefill-token-threshold 400 \
  --kt-max-deferred-experts-per-token 1 \
  --kt-enable-dynamic-expert-update \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 16384 \
  --attention-backend flashinfer \
  --fp8-gemm-backend triton \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --chat-template /opt/templates/Kimi-K2.7-Default.jinja \
  --sampling-defaults model
```

The template mount and `--chat-template` argument can both be removed to use the template included with the checkpoint.

The tested configuration keeps eight routed experts per MoE layer on the GPU:

```bash
--kt-num-gpu-experts 8
```

You can experiment with this value to change the balance between GPU memory use and CPU/GPU expert execution. On the tested hardware, however, changing the number of GPU experts had only a minimal performance effect.

The CPU-inference, context and memory settings are likewise tuned for the tested hardware and may need adjustment for systems with a different CPU topology, GPU-memory capacity or available RAM.

## GitHub Copilot BYOK

This configuration has been tested in production with GitHub Copilot BYOK:

```json
{
  "name": "Local",
  "vendor": "customendpoint",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "kimi-k2.7-code",
      "name": "Kimi K2.7 Code",
      "url": "http://<server>:8000/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "thinking": true,
      "streaming": true,
      "maxInputTokens": 131072,
      "maxOutputTokens": 32768
    }
  ]
}
```

The model ID must contain one of these lowercase strings:

```text
kimi-k2.6
kimi-k2.7-code
```

GitHub Copilot uses the model ID to enable its corresponding Kimi-specific compatibility handling. A custom ID without the appropriate string may bypass those fixes.

The Anthropic-compatible `/v1/messages` route did not work reliably with the combination of SGLang, Kimi and GitHub Copilot BYOK at the time this recipe was published. Use `apiType: "chat-completions"` with `/v1/chat/completions`.

## Benchmarks

Measured in July 2026 with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) using the tested Kimi K2.7 Code RAWINT4 configuration.

The benchmark reports prefill throughput as prompt tokens divided by time to first token. Decode values are aggregate sustained throughput across the requested concurrency.

### Prefill

| Context | Prompt tokens | TTFT | Throughput | Requests |
|---:|---:|---:|---:|---:|
| 8k | 8,189 | 16.33 s | 501 tok/s | 1 |
| 16k | 16,230 | 24.18 s | 671 tok/s | 1 |
| 32k | 32,313 | 44.83 s | 721 tok/s | 1 |
| 64k | 64,482 | 92.70 s | 696 tok/s | 1 |
| 128k | 128,807 | 220.35 s | 585 tok/s | 1 |

### Aggregate sustained decode

| Context | Concurrency 1 | Concurrency 2 |
|---:|---:|---:|
| 0 | 18.1 tok/s | 23.5 tok/s |
| 8k | 17.8 tok/s | 23.1 tok/s |
| 16k | 16.7 tok/s | 23.1 tok/s |
| 32k | 16.4 tok/s | 22.6 tok/s |
| 64k | 15.7 tok/s | 21.9 tok/s |
| 128k | 15.4 tok/s | 20.4 tok/s |

These results describe this specific hybrid CPU/GPU placement and are not intended as general Kimi K2.7 performance claims.

## Known limitations

- The `lmsysorg/sglang:dev-cu13` base image is rolling. Rebuilding later can inherit upstream changes even though the KTransformers stack is pinned.
- The image is compiled natively for AMD Turin and Blackwell SM120.
- Model weights and chat templates are not included.
- Different Kimi checkpoints require the appropriate model-specific chat template.
- The parser and RAWINT4 changes are downstream compatibility patches that should be removed when equivalent fixes reach the pinned KTransformers SGLang fork.
- The launch recipe targets one GPU and two concurrent requests.
