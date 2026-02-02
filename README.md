# LLM-serve: Inference Engine Benchmarking

A benchmarking suite to compare **TensorRT-LLM**, **vLLM**, and **SGLang** for LLM inference performance across different batch sizes, measuring throughput and latency.

## Overview

This project provides a unified framework to:
1. Deploy the same model (Qwen2.5-3B-Instruct) across three inference engines
2. Run standardized benchmarks with varying batch sizes
3. Compare throughput (tokens/sec) and latency (time-to-first-token, inter-token latency)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hugging Face Model                          │
│                   (Qwen2.5-3B-Instruct)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  TensorRT-LLM │     │     vLLM      │     │    SGLang     │
│               │     │               │     │               │
│ • Convert to  │     │ • Direct HF   │     │ • Direct HF   │
│   checkpoint  │     │   loading     │     │   loading     │
│ • Build TRT   │     │ • PagedAttn   │     │ • RadixAttn   │
│   engine      │     │ • Continuous  │     │ • Continuous  │
│ • Optimized   │     │   batching    │     │   batching    │
│   CUDA kernels│     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Benchmark     │
                    │   Framework     │
                    │                 │
                    │ • Throughput    │
                    │ • Latency       │
                    │ • Memory usage  │
                    └─────────────────┘
```

## Quick Start

### 1. Download the Model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
```

### 2. Setup Each Engine

#### TensorRT-LLM (Requires conversion + compilation)

```bash
# Step 1: Convert HF model to TensorRT-LLM checkpoint format
cd tensorrt
python convert_checkpoint.py \
    --model_dir ../models/Qwen2.5-3B-Instruct \
    --output_dir ../checkpoints/trt_qwen2.5_3b_fp16 \
    --dtype float16

# Step 2: Build TensorRT engine (optimized for your GPU)
trtllm-build \
    --checkpoint_dir ../checkpoints/trt_qwen2.5_3b_fp16 \
    --output_dir ../engines/trt_qwen2.5_3b_fp16 \
    --gemm_plugin float16

# Step 3: Run inference
python run_trt.py \
    --engine_dir ../engines/trt_qwen2.5_3b_fp16 \
    --tokenizer_dir ../models/Qwen2.5-3B-Instruct \
    --input_text "Hello, how are you?"
```

#### vLLM (Direct loading)

```bash
cd vllm
pip install vllm

# Run inference (loads model directly)
python run_vllm.py \
    --model ../models/Qwen2.5-3B-Instruct \
    --input_text "Hello, how are you?"
```

#### SGLang (Direct loading)

```bash
cd sglang
pip install sglang[all]

# Run inference
python run_sglang.py \
    --model ../models/Qwen2.5-3B-Instruct \
    --input_text "Hello, how are you?"
```

### 3. Run Benchmarks

```bash
cd benchmarks
python benchmark.py \
    --model_path ../models/Qwen2.5-3B-Instruct \
    --trt_engine_dir ../engines/trt_qwen2.5_3b_fp16 \
    --batch_sizes 1 4 8 16 32 \
    --output_tokens 128 \
    --output results.json
```

## How Each Engine Works

### TensorRT-LLM

TensorRT-LLM requires a **two-step conversion process**:

1. **Checkpoint Conversion** (`convert_checkpoint.py`):
   - Loads PyTorch/safetensors weights from Hugging Face format
   - Reorganizes weight layouts for optimal GPU memory access
   - Applies optional quantization (INT8/INT4)
   - Outputs TensorRT-LLM checkpoint format

2. **Engine Building** (`trtllm-build`):
   - Compiles the model into optimized CUDA kernels
   - Fuses operations (attention, MLP layers)
   - Generates GPU-specific code for your hardware
   - Creates a serialized engine file

**Pros**: Fastest inference, lowest latency
**Cons**: Longer setup time, GPU-specific engines

### vLLM

vLLM loads models directly from Hugging Face format with **PagedAttention**:

- **PagedAttention**: Manages KV cache like virtual memory pages
- **Continuous Batching**: Dynamically adds/removes requests mid-generation
- **No pre-compilation**: Works immediately with any HF model

**Pros**: Easy setup, good throughput, flexible
**Cons**: Slightly higher latency than TensorRT

### SGLang

SGLang uses **RadixAttention** for efficient prefix caching:

- **RadixAttention**: Shares KV cache across requests with common prefixes
- **Optimized for multi-turn**: Great for chat applications
- **Continuous Batching**: Similar to vLLM

**Pros**: Excellent for chat/multi-turn, prefix caching
**Cons**: Newer, smaller community

## Benchmark Metrics

| Metric | Description |
|--------|-------------|
| **Throughput** | Tokens generated per second (tokens/s) |
| **TTFT** | Time to First Token - latency before generation starts |
| **ITL** | Inter-Token Latency - time between consecutive tokens |
| **Memory** | Peak GPU memory usage during inference |

## Project Structure

```
LLM-serve/
├── README.md
├── requirements.txt
├── models/                    # Downloaded HF models
├── checkpoints/               # TensorRT-LLM checkpoints
├── engines/                   # Compiled TensorRT engines
├── tensorrt/
│   ├── convert_checkpoint.py  # HF → TRT-LLM checkpoint
│   ├── run_trt.py             # TensorRT inference
│   └── requirements.txt
├── vllm/
│   ├── run_vllm.py            # vLLM inference
│   └── requirements.txt
├── sglang/
│   ├── run_sglang.py          # SGLang inference
│   └── requirements.txt
└── benchmarks/
    ├── benchmark.py           # Unified benchmark runner
    ├── prompts.json           # Test prompts
    └── analyze_results.py     # Generate comparison charts
```

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- ~16GB GPU memory for Qwen2.5-3B in FP16

## Status

| Engine | Status | Notes |
|--------|--------|-------|
| TensorRT-LLM | ✅ Done | Checkpoint conversion + engine build working |
| vLLM | ✅ Done | Tested with batch inference |
| SGLang | ✅ Installed | Ready to test |
| Benchmarks | ✅ Done | Framework implemented |

## Initial Benchmark Results (Qwen2.5-3B-Instruct, FP16)

| Engine | Batch Size | Avg Latency | Throughput |
|--------|------------|-------------|------------|
| vLLM | 4 | 802 ms | **249 tokens/sec** |

*Note: Full comparative benchmarks pending. Run `benchmark.py` for complete results.*

## References

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
