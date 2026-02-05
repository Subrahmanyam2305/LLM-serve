# LLM-serve: Inference Engine Benchmarking

A benchmarking suite to compare **TensorRT-LLM**, **vLLM**, and **SGLang** for LLM inference performance across different batch sizes, measuring throughput and latency.

## Overview

This project provides a unified framework to:
1. Deploy models (Llama, Qwen, Gemma, Phi) across three inference engines
2. Run standardized benchmarks with varying batch sizes using ShareGPT dataset
3. Compare throughput (tokens/sec) and latency (time-to-first-token, inter-token latency)
4. Orchestrate the full TensorRT-LLM pipeline (download → convert → build → run)

## Supported Models

| Model | HuggingFace ID | Gated | Notes |
|-------|----------------|-------|-------|
| Llama | meta-llama/Llama-3.2-3B-Instruct | Yes | Requires HF_TOKEN |
| Qwen | Qwen/Qwen2.5-3B-Instruct | No | Default model |
| Gemma | google/gemma-3-4b-it | Yes | Requires HF_TOKEN |
| Phi | microsoft/Phi-4-mini-instruct | No | Open access |

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

### Separate Environments (Recommended)

These frameworks have conflicting dependencies. Use separate virtual environments for accurate benchmarking.

**Important**: Set a custom temp directory to avoid running out of space during installation. For **gated models (Llama, Gemma)** on Hugging Face, set your token in `~/.bashrc`:

```bash
# Add to ~/.bashrc
export TMPDIR=/path/to/large/disk/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR

# Required for gated models (Llama, Gemma); scripts use this for downloads
export HF_TOKEN=your_huggingface_token_here

source ~/.bashrc
```

```bash
# TensorRT-LLM environment (~15GB)
python3.12 -m venv .venv_trt
source .venv_trt/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm transformers sentencepiece tiktoken

# vLLM environment (~10GB)
python3.12 -m venv .venv_vllm
source .venv_vllm/bin/activate
pip install --upgrade pip
pip install vllm

# SGLang environment (~11GB)
python3.12 -m venv .venv_sgl
source .venv_sgl/bin/activate
pip install --upgrade pip
pip install "sglang[all]"
```

**Repo layout:** This repo is **LLM-serve** inside **llm_host**. The backend virtual environments (`.venv_trt`, `.venv_vllm`, `.venv_sgl`) live at the **llm_host** root. From the llm_host directory, activate the matching env then run scripts under `LLM-serve/`.

### Running with Specific Environment

```bash
# From llm_host root:
# TensorRT-LLM
source .venv_trt/bin/activate
python LLM-serve/tensorrt/run_trt.py --engine_dir ./engines/trt_qwen2.5_3b_fp16 ...

# vLLM
source .venv_vllm/bin/activate
python LLM-serve/vllm/run_vllm.py --model ./models/Qwen2.5-3B-Instruct ...

# SGLang
source .venv_sgl/bin/activate
python LLM-serve/sglang/run_sglang.py --model ./models/Qwen2.5-3B-Instruct ...
```

## Quick Start

### 1. Download the Model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
```

For **gated models (Llama, Gemma)**, ensure `HF_TOKEN` is set in your environment (see Installation). Then use `huggingface-cli login` or pass the token: `huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/Llama-3.2-3B-Instruct --token $HF_TOKEN`.

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

# Benchmark with default prompts
python benchmark.py \
    --model_path ../models/Qwen2.5-3B-Instruct \
    --trt_engine_dir ../engines/trt_qwen2.5_3b_fp16 \
    --batch_sizes 1 4 8 16 \
    --max_output_tokens 128 \
    --engines vllm \
    --output results_vllm.json

# Benchmark with ShareGPT dataset (recommended)
python benchmark.py \
    --model_path ../models/Qwen2.5-3B-Instruct \
    --trt_engine_dir ../engines/trt_qwen2.5_3b_fp16 \
    --sharegpt --max_prompts 200 \
    --batch_sizes 1 4 8 16 \
    --engines tensorrt vllm sglang \
    --output results_sharegpt.json

# Analyze results
python analyze_results.py --input results.json --output_dir ./analysis
```

### 4. TensorRT Orchestrator (Full Pipeline)

The orchestrator automates the entire TensorRT-LLM workflow:

```bash
cd tensorrt

# Full pipeline for Qwen (download → convert → build → run)
python orchestrate_trt.py \
    --model qwen \
    --models_dir ../models \
    --checkpoints_dir ../checkpoints \
    --engines_dir ../engines \
    --dtype float16

# Full pipeline for Llama (requires HF_TOKEN)
python orchestrate_trt.py \
    --model llama \
    --dtype float16 \
    --run_benchmark --sharegpt --max_prompts 100

# Skip steps if already done
python orchestrate_trt.py \
    --model qwen \
    --skip_download --skip_convert \
    --run_benchmark --sharegpt --max_prompts 200

# Run only (assumes engine exists)
python orchestrate_trt.py \
    --model qwen \
    --run_only \
    --input_text "Explain quantum computing"
```

### 5. GenAI-Bench UI (Optional)

For advanced benchmarking with live UI:

```bash
# Install genai-bench
pip install genai-bench

# Start a local server (e.g., vLLM)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 8000

# Run genai-bench
cd genai_bench
python run_genai_bench.py \
    --model qwen \
    --backend openai \
    --api_base http://localhost:8000 \
    --generate_report --generate_plots
```

See [genai_bench/README.md](genai_bench/README.md) for more details.

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

The repo **LLM-serve** lives inside **llm_host**. Backend venvs (`.venv_trt`, `.venv_vllm`, `.venv_sgl`) are at llm_host root.

```
llm_host/
├── .venv_trt/      # TensorRT-LLM
├── .venv_vllm/     # vLLM
├── .venv_sgl/      # SGLang
└── LLM-serve/
├── README.md
├── requirements.txt
├── docs/
│   └── BENCHMARK_VARIABLES.md  # Variables and metrics reference
├── models/                    # Downloaded HF models (gitignored)
├── checkpoints/               # TensorRT-LLM checkpoints (gitignored)
├── engines/                   # Compiled TensorRT engines (gitignored)
├── tensorrt/
│   ├── orchestrate_trt.py     # Full pipeline: download → convert → build → run
│   ├── convert_checkpoint.py  # Qwen HF → TRT-LLM checkpoint
│   ├── convert_llama.py       # Llama HF → TRT-LLM checkpoint
│   ├── convert_phi.py         # Phi HF → TRT-LLM checkpoint
│   ├── convert_gemma.py       # Gemma HF → TRT-LLM checkpoint
│   ├── run_trt.py             # TensorRT inference with TTFT
│   └── requirements.txt
├── vllm/
│   ├── run_vllm.py            # vLLM inference with TTFT
│   └── requirements.txt
├── sglang/
│   ├── run_sglang.py          # SGLang inference with TTFT
│   └── requirements.txt
├── benchmarks/
│   ├── benchmark.py           # Unified benchmark runner
│   ├── load_sharegpt.py       # ShareGPT dataset loader
│   ├── prompts.json           # Fallback test prompts
│   └── analyze_results.py     # Generate comparison charts
└── genai_bench/
    ├── run_genai_bench.py     # GenAI-Bench integration
    └── README.md              # GenAI-Bench usage guide
```

## Requirements

- **GPU**: NVIDIA A10G (g5.xlarge) or similar with CUDA support
  - Compute Capability: 8.6
  - TensorRT engines are GPU-specific and optimized for A10G
- Python 3.10+ (3.12 recommended)
- ~16GB GPU memory for Qwen2.5-3B in FP16
- ~40GB disk space for all three frameworks

## Hardware Tested

| Instance | GPU | VRAM | Compute Capability |
|----------|-----|------|-------------------|
| g5.xlarge | NVIDIA A10G | 24GB | 8.6 |

> **Note**: TensorRT engines built on A10G are optimized for Ampere architecture (SM86). They may not work on different GPU architectures.

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
