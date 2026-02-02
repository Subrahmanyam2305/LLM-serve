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
                    │ • TTFT          │
                    │ • ITL           │
                    │ • Memory usage  │
                    └─────────────────┘
```

## Fair Comparison Guidelines

To ensure fair benchmarking across all three engines, we standardize:

### Sampling Parameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| `temperature` | 0.0 | Greedy decoding for reproducibility |
| `top_k` | 1 | Greedy (TRT uses 1, vLLM uses -1 to disable) |
| `top_p` | 1.0 | No nucleus sampling |
| `max_tokens` | 128 | Same output length limit |

### Token Counting
- All engines use the **actual tokenizer** to count generated tokens
- Previous SGLang implementation used word count estimation (unfair!)
- Now all use `tokenizer.encode()` for accurate counts

### Metrics Measured
| Metric | Description |
|--------|-------------|
| **Throughput** | Tokens generated per second (tokens/s) |
| **TTFT** | Time to First Token - latency before first token arrives |
| **ITL** | Inter-Token Latency - average time between consecutive tokens |
| **Total Latency** | End-to-end time for complete generation |
| **Memory** | Peak GPU memory usage during inference |

## Installation

### Option 1: Single Environment (Quick Testing)

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install all frameworks (may have dependency conflicts)
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
pip install vllm
pip install "sglang[all]"
```

> ⚠️ **Warning**: These frameworks have conflicting dependencies. Use separate environments for production benchmarking.

### Option 2: Separate Environments (Recommended for Accurate Benchmarking)

```bash
# TensorRT-LLM environment
python3.12 -m venv .venv_trt
source .venv_trt/bin/activate
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm transformers sentencepiece tiktoken

# vLLM environment
python3.12 -m venv .venv_vllm
source .venv_vllm/bin/activate
pip install vllm

# SGLang environment
python3.12 -m venv .venv_sgl
source .venv_sgl/bin/activate
pip install "sglang[all]"
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
    --input_text "Hello, how are you?" \
    --temperature 0.0
```

#### vLLM (Direct loading)

```bash
cd vllm

# Run inference (loads model directly)
python run_vllm.py \
    --model ../models/Qwen2.5-3B-Instruct \
    --input_text "Hello, how are you?" \
    --temperature 0.0
```

#### SGLang (Direct loading)

```bash
cd sglang

# Run inference
python run_sglang.py \
    --model ../models/Qwen2.5-3B-Instruct \
    --input_text "Hello, how are you?" \
    --temperature 0.0
```

### 3. Run Benchmarks

```bash
cd benchmarks

# Benchmark single engine
python benchmark.py \
    --model_path ../models/Qwen2.5-3B-Instruct \
    --trt_engine_dir ../engines/trt_qwen2.5_3b_fp16 \
    --batch_sizes 1 4 8 16 \
    --max_output_tokens 128 \
    --engines vllm \
    --output results_vllm.json

# Analyze results
python analyze_results.py --input results.json --output_dir ./analysis
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

**Pros**: Fastest inference, lowest latency, best for production
**Cons**: Longer setup time, GPU-specific engines, less flexible

### vLLM

vLLM loads models directly from Hugging Face format with **PagedAttention**:

- **PagedAttention**: Manages KV cache like virtual memory pages
- **Continuous Batching**: Dynamically adds/removes requests mid-generation
- **No pre-compilation**: Works immediately with any HF model

**Pros**: Easy setup, good throughput, flexible, great for serving
**Cons**: Slightly higher latency than TensorRT

### SGLang

SGLang uses **RadixAttention** for efficient prefix caching:

- **RadixAttention**: Shares KV cache across requests with common prefixes
- **Optimized for multi-turn**: Great for chat applications
- **Continuous Batching**: Similar to vLLM

**Pros**: Excellent for chat/multi-turn, prefix caching, structured generation
**Cons**: Newer, smaller community

## Project Structure

```
LLM-serve/
├── README.md
├── requirements.txt
├── models/                    # Downloaded HF models (gitignored)
├── checkpoints/               # TensorRT-LLM checkpoints (gitignored)
├── engines/                   # Compiled TensorRT engines (gitignored)
├── tensorrt/
│   ├── convert_checkpoint.py  # HF → TRT-LLM checkpoint
│   ├── run_trt.py             # TensorRT inference with TTFT
│   └── requirements.txt
├── vllm/
│   ├── run_vllm.py            # vLLM inference with TTFT
│   └── requirements.txt
├── sglang/
│   ├── run_sglang.py          # SGLang inference with TTFT
│   └── requirements.txt
└── benchmarks/
    ├── benchmark.py           # Unified benchmark runner
    ├── prompts.json           # Test prompts
    └── analyze_results.py     # Generate comparison charts
```

## Requirements

- NVIDIA GPU with CUDA support (tested on A10G, A100)
- Python 3.10+ (3.12 recommended)
- ~16GB GPU memory for Qwen2.5-3B in FP16
- ~60GB disk space for all three frameworks

## Status

| Engine | Status | Notes |
|--------|--------|-------|
| TensorRT-LLM | ✅ Done | Checkpoint conversion + engine build working |
| vLLM | ✅ Done | Tested with batch inference |
| SGLang | ✅ Done | Fixed token counting |
| Benchmarks | ✅ Done | Framework with fair comparison |

## Known Issues & Limitations

1. **Dependency Conflicts**: TensorRT-LLM, vLLM, and SGLang have conflicting dependencies. Use separate virtual environments for accurate benchmarking.

2. **TTFT Measurement**: Accurate TTFT requires streaming mode. Current implementation measures end-to-end latency; streaming support is available but may have overhead.

3. **Token Counting**: All engines now use the actual tokenizer for accurate token counting.

## References

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## License

MIT License - see [LICENSE](LICENSE) for details.
