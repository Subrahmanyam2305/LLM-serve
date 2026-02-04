# Issues Faced During Triton TensorRT-LLM Setup

## Overview

This document describes the issues encountered while setting up Triton Inference Server with TensorRT-LLM backend for accurate concurrency benchmarking.

## 1. Concurrency Measurement Issues

### Problem: Batch Processing vs True Concurrency

The original `benchmarks/benchmark.py` file used batch processing rather than true concurrent request handling. This led to misleading concurrency metrics because:

- **Batch processing**: Groups multiple prompts together and processes them in a single inference call
- **True concurrency**: Multiple independent requests arrive asynchronously and are handled by the server's scheduler

**Why this matters:**
- Batch processing measures GPU utilization and throughput for batched workloads
- True concurrency measures how the server handles independent requests arriving at different times
- Production workloads typically involve independent requests, not pre-batched data

### Solution

Use Triton Inference Server with TensorRT-LLM backend, which provides:
- Inflight batching: Dynamically batches requests as they arrive
- Proper request scheduling with `max_utilization` policy
- Native support for concurrent gRPC/HTTP requests

## 2. TensorRT-LLM Version Compatibility

### Problem

The TensorRT-LLM engine was built with version 1.1.0, which requires a specific Triton Docker image version.

### Solution

Use the compatible Triton Docker image:
```bash
nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3
```

This image includes TensorRT-LLM 1.1.0 backend support.

## 3. Triton Model Configuration

### Problem

The Triton TensorRT-LLM backend requires many configuration parameters that must be set correctly.

### Solution

Use the `fill_template.py` script from TensorRT-LLM to generate the config:

```bash
python /path/to/TensorRT-LLM/triton_backend/tools/fill_template.py \
    -i /tmp/triton_model_repo/tensorrt_llm/config.pbtxt \
    "triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:True,..."
```

Key parameters:
- `batching_strategy:inflight_fused_batching` - Enable inflight batching
- `batch_scheduler_policy:max_utilization` - Maximize GPU utilization
- `decoupled_mode:True` - Required for streaming responses
- `tokenizer_dir:/path/to/tokenizer` - Required for tokenization

## 4. Streaming Response Handling

### Problem

When using decoupled mode (required for streaming), the Triton server:
1. Sends tokens one at a time via gRPC streaming
2. Sends an empty token `[]` when generation completes naturally (EOS)
3. Does NOT send an empty token when `max_tokens` is reached

This causes issues with detecting end-of-generation in the OpenAI wrapper.

### Current Status

- Non-streaming requests work correctly and are fast (~0.15s for short responses)
- Streaming requests work but may hang when `max_tokens` is reached before EOS
- The wrapper includes logic to detect both EOS and max_tokens conditions

### Workaround

For benchmarking, use non-streaming mode or ensure prompts generate responses that hit EOS naturally.

## 5. genai-bench Integration

### Problem

genai-bench expects streaming responses by default for TTFT (Time To First Token) measurement.

### Current Status

- genai-bench can connect to the OpenAI-compatible wrapper
- TTFT measurement requires streaming mode
- Non-streaming mode doesn't provide TTFT metrics

### Alternative Approaches

1. **Direct Triton gRPC benchmarking**: Use the `inflight_batcher_llm_client.py` for direct benchmarking
2. **Custom benchmark script**: Create a script that measures TTFT using the streaming endpoint

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   genai-bench   │────▶│  OpenAI Wrapper      │────▶│  Triton Server  │
│   (HTTP/REST)   │     │  (HTTP → gRPC)       │     │  (gRPC)         │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  TensorRT-LLM   │
                                                     │  Engine         │
                                                     └─────────────────┘
```

## Files Created

- `servers/launch_triton_server.py` - Script to launch Triton Docker container
- `servers/triton_openai_wrapper.py` - OpenAI-compatible HTTP wrapper for Triton gRPC
- `genai_bench/run_genai_bench.py` - Wrapper for genai-bench tool

## Running the Setup

1. **Start Triton Server:**
```bash
python servers/launch_triton_server.py \
    --engine-dir /path/to/trt_engines \
    --tokenizer-dir /path/to/tokenizer \
    --detach
```

2. **Start OpenAI Wrapper:**
```bash
python servers/triton_openai_wrapper.py \
    --triton-url localhost:8001 \
    --tokenizer-dir /path/to/tokenizer \
    --port 8080
```

3. **Test with curl:**
```bash
# Non-streaming (works reliably)
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "model-name", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Streaming (may have issues with max_tokens)
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "model-name", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50, "stream": true}'
```

## Recommendations

1. **For accurate concurrency testing**: Use Triton with inflight batching
2. **For TTFT measurement**: Use streaming mode with prompts that generate short responses (hit EOS naturally)
3. **For throughput testing**: Non-streaming mode works reliably
4. **For production**: Consider using Triton's native HTTP endpoint with the ensemble model for full preprocessing/postprocessing pipeline
