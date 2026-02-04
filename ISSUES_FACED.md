# Issues Faced & Solutions

This document captures all the issues encountered during the setup and benchmarking of TensorRT-LLM, vLLM, and SGLang inference engines. Use this as a troubleshooting guide when you encounter similar problems.

## Table of Contents

1. [Installation Issues](#installation-issues)
   - [OSError: No Space Left on Device](#1-oserror-no-space-left-on-device)
   - [TensorRT-LLM Extra Index URL](#2-tensorrt-llm-extra-index-url)
   - [Dependency Conflicts Between Frameworks](#3-dependency-conflicts-between-frameworks)
2. [SGLang Issues](#sglang-issues)
   - [Flashinfer nvcc Not Found](#4-flashinfer-nvcc-not-found)
   - [CUDA_HOME Not Set](#5-cuda_home-not-set)
   - [CUDA Graph Capture Failed](#6-cuda-graph-capture-failed)
   - [nvcc vs Triton Backend Conflict](#7-nvcc-vs-triton-backend-conflict)
   - [Cached nvcc Path from Wrong Virtual Environment](#8-cached-nvcc-path-from-wrong-virtual-environment)
3. [TensorRT-LLM Issues](#tensorrt-llm-issues)
   - [Engine Build Requires Specific GPU](#9-engine-build-requires-specific-gpu)
   - [Checkpoint Conversion Memory Issues](#10-checkpoint-conversion-memory-issues)
4. [Benchmarking Issues](#benchmarking-issues)
   - [TTFT and ITL Metrics Showing None](#11-ttft-and-itl-metrics-showing-none)
   - [Unfair Token Counting](#12-unfair-token-counting)
   - [GenAI-Bench CLI Argument Error](#13-genai-bench-cli-argument-error)
5. [Environment & Path Issues](#environment--path-issues)
   - [Virtual Environment Activation in Scripts](#14-virtual-environment-activation-in-scripts)
   - [HuggingFace Token for Gated Models](#15-huggingface-token-for-gated-models)

---

## Installation Issues

### 1. OSError: No Space Left on Device

**Symptom:**
```
OSError: [Errno 28] No space left on device
```

This occurs during pip installation when the default `/tmp` directory runs out of space. Large packages like TensorRT-LLM, vLLM, and SGLang require significant temporary storage during installation.

**Solution:**

Set a custom temp directory with more space before installing:

```bash
# Create a temp directory on a larger disk
mkdir -p /home/ec2-user/llm_host/tmp

# Set environment variables (add to ~/.bashrc for persistence)
export TMPDIR=/home/ec2-user/llm_host/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PIP_CACHE_DIR=/home/ec2-user/llm_host/tmp/pip_cache
mkdir -p $PIP_CACHE_DIR

# Source bashrc or set in current session
source ~/.bashrc

# Now run pip install
pip install <package>
```

**Prevention:**

Add these lines to your `~/.bashrc`:
```bash
export TMPDIR=/home/ec2-user/llm_host/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PIP_CACHE_DIR=/home/ec2-user/llm_host/tmp/pip_cache
```

---

### 2. TensorRT-LLM Extra Index URL

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement tensorrt-llm
ERROR: No matching distribution found for tensorrt-llm
```

**Cause:**
TensorRT-LLM is hosted on NVIDIA's PyPI server, not the default PyPI.

**Solution:**

Always use the `--extra-index-url` flag when installing TensorRT-LLM:

```bash
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
```

Full installation command with all dependencies:
```bash
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm transformers sentencepiece tiktoken
```

---

### 3. Dependency Conflicts Between Frameworks

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

Or runtime errors like:
```
ImportError: cannot import name 'X' from 'torch'
```

**Cause:**
TensorRT-LLM, vLLM, and SGLang have conflicting dependencies (different versions of torch, transformers, etc.).

**Solution:**

**Always use separate virtual environments for each framework:**

```bash
# TensorRT-LLM environment
python3.12 -m venv .venv_trt
source .venv_trt/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm transformers sentencepiece tiktoken
deactivate

# vLLM environment
python3.12 -m venv .venv_vllm
source .venv_vllm/bin/activate
pip install --upgrade pip
pip install vllm
deactivate

# SGLang environment
python3.12 -m venv .venv_sgl
source .venv_sgl/bin/activate
pip install --upgrade pip
pip install "sglang[all]"
deactivate
```

**Directory Structure:**
```
llm_host/
├── .venv_trt/      # TensorRT-LLM virtual environment
├── .venv_vllm/     # vLLM virtual environment
├── .venv_sgl/      # SGLang virtual environment
└── LLM-serve/      # This repository
```

---

## SGLang Issues

### 4. Flashinfer nvcc Not Found

**Symptom:**
```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Or:
```
/bin/sh: line 1: /path/to/nvcc: No such file or directory
```

**Cause:**
SGLang uses `flashinfer` for attention computation, which requires JIT (Just-In-Time) compilation of CUDA kernels using `nvcc`. If `nvcc` is not installed or not in PATH, this fails.

**Solution:**

Use alternative backends that don't require JIT compilation:

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-cuda-graph
```

Or in the `sglang_backend_server.py`:
```bash
python servers/sglang_backend_server.py \
    --model /path/to/model \
    --attention-backend triton \
    --sampling-backend pytorch
```

---

### 5. CUDA_HOME Not Set

**Symptom:**
```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Or:
```
Could not find CUDA installation. Please set CUDA_HOME environment variable.
```

**Cause:**
Flashinfer and other CUDA-dependent libraries need to know where CUDA is installed to find `nvcc` and CUDA headers. If `CUDA_HOME` is not set and CUDA is not in the default location (`/usr/local/cuda`), the build fails.

**Solution:**

1. **Find your CUDA installation path:**

```bash
# Method 1: Check nvidia-smi for CUDA version, then look for it
nvidia-smi | grep "CUDA Version"

# Method 2: Find nvcc location
which nvcc
# Or search for it
find /usr -name "nvcc" 2>/dev/null
find /opt -name "nvcc" 2>/dev/null

# Method 3: Check common locations
ls -la /usr/local/cuda*
ls -la /opt/cuda*

# Method 4: Check if CUDA is in a Python package (common in venvs)
python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME)"

# Method 5: For AWS Deep Learning AMIs
ls -la /usr/local/cuda
# Usually symlinked to /usr/local/cuda-XX.X
```

2. **Set CUDA_HOME environment variable:**

```bash
# If CUDA is at /usr/local/cuda-12.1
export CUDA_HOME=/usr/local/cuda-12.1

# Or if it's a symlink at /usr/local/cuda
export CUDA_HOME=/usr/local/cuda

# Add nvcc to PATH as well
export PATH=$CUDA_HOME/bin:$PATH

# Add to ~/.bashrc for persistence
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

3. **Verify the setup:**

```bash
# Check CUDA_HOME is set
echo $CUDA_HOME

# Check nvcc is accessible
nvcc --version

# Should output something like:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.1, V12.1.105
```

**Common CUDA Paths by Platform:**

| Platform | Typical CUDA Path |
|----------|------------------|
| Ubuntu/Debian | `/usr/local/cuda` or `/usr/local/cuda-XX.X` |
| AWS Deep Learning AMI | `/usr/local/cuda` |
| NVIDIA NGC Containers | `/usr/local/cuda` |
| Conda environment | `$CONDA_PREFIX/lib` (may not have nvcc) |
| pip nvidia-cuda-toolkit | Check with `python -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__)"` |

**Alternative: Use Triton Backend (No nvcc Required)**

If you can't set up CUDA_HOME or don't have nvcc installed, use the triton backend:

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-cuda-graph
```

---

### 6. CUDA Graph Capture Failed

**Symptom:**
```
Exception: Capture cuda graph failed: Ninja build failed.
```

With suggestions like:
```
Possible solutions:
1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)
2. set --cuda-graph-max-bs to a smaller value (e.g., 16)
3. disable torch compile by not using --enable-torch-compile
4. disable CUDA graph by --disable-cuda-graph
```

**Cause:**
CUDA graph capture requires flashinfer JIT compilation, which needs `nvcc`.

**Solution:**

Disable CUDA graph capture:

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --disable-cuda-graph \
    --attention-backend triton \
    --sampling-backend pytorch
```

**Note:** Disabling CUDA graph may result in some performance loss, but the server will work.

---

### 7. nvcc vs Triton Backend Conflict

**Symptom:**
When trying to use FP8 quantization or flashinfer with nvcc, you get errors like:
```
AssertionError: cuda_home is not None
```

Or when using triton backend, nvcc-dependent features fail.

**Cause:**
There's a fundamental conflict between two approaches:
1. **nvcc-based compilation**: Requires full CUDA toolkit with `nvcc` compiler. Used by flashinfer, FP8 quantization, and deep_gemm.
2. **Triton-based compilation**: Uses Triton's JIT compiler, doesn't require nvcc. Used as fallback for attention/sampling.

Installing one approach's dependencies can break the other. The pip-installed CUDA packages (`nvidia-cuda-runtime`, `cuda-toolkit`) don't include `nvcc` - they only provide runtime libraries.

**Solution:**

**Option A: Use Triton Backend (No nvcc required)**
```bash
# For SGLang - use triton backend
python -m sglang.launch_server \
    --model-path /path/to/model \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-cuda-graph

# Limitation: Cannot use FP8 quantization or flashinfer optimizations
```

**Option B: Install Full CUDA Toolkit (Enables nvcc)**
```bash
# Install CUDA toolkit (requires sudo)
sudo dnf install cuda-toolkit-12-4  # Amazon Linux 2023
# or
sudo apt install nvidia-cuda-toolkit  # Ubuntu

# Set environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Verify
nvcc --version

# Now you can use:
# - FP8 quantization
# - flashinfer attention backend
# - deep_gemm optimizations
```

**Recommendation:**
- For **production benchmarking**: Install full CUDA toolkit for maximum performance
- For **quick testing**: Use triton backend to avoid nvcc dependency
- **Never mix**: Don't try to use nvcc-dependent features without proper CUDA toolkit installation

---

### 8. Cached nvcc Path from Wrong Virtual Environment

**Symptom:**
```
/bin/sh: line 1: /home/ec2-user/llm_host/.venv_trt/lib64/python3.12/site-packages/nvidia/cuda_runtime/bin/nvcc: No such file or directory
```

Notice the path points to `.venv_trt` even though you're running SGLang from `.venv_sgl`.

**Cause:**
Flashinfer caches the `nvcc` path in `~/.cache/flashinfer/`. If you previously ran something in the TensorRT environment that set this path, it gets cached and reused incorrectly.

**Solution:**

1. **Clear the flashinfer cache:**
```bash
rm -rf ~/.cache/flashinfer
```

2. **Use alternative backends:**
```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-cuda-graph
```

3. **Ensure clean environment activation:**
```bash
# Deactivate any active environment first
deactivate 2>/dev/null

# Clear flashinfer cache
rm -rf ~/.cache/flashinfer

# Activate SGLang environment
source .venv_sgl/bin/activate

# Run SGLang
python -m sglang.launch_server ...
```

---

## TensorRT-LLM Issues

### 9. Engine Build Requires Specific GPU

**Symptom:**
TensorRT engine built on one GPU doesn't work on another GPU architecture.

**Cause:**
TensorRT engines are optimized for specific GPU compute capabilities. An engine built for A10G (SM86) won't work on V100 (SM70) or H100 (SM90).

**Solution:**

Always build engines on the target GPU:

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build engine on the target machine
trtllm-build \
    --checkpoint_dir ./checkpoints/model \
    --output_dir ./engines/model \
    --gemm_plugin float16
```

**GPU Compute Capabilities:**
| GPU | Compute Capability |
|-----|-------------------|
| V100 | 7.0 |
| T4 | 7.5 |
| A10G | 8.6 |
| A100 | 8.0 |
| H100 | 9.0 |

---

### 10. Checkpoint Conversion Memory Issues

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

During checkpoint conversion.

**Cause:**
Converting large models requires significant GPU memory.

**Solution:**

1. **Use CPU for conversion:**
```bash
CUDA_VISIBLE_DEVICES="" python convert_checkpoint.py \
    --model_dir ./models/Model \
    --output_dir ./checkpoints/model \
    --dtype float16
```

2. **Reduce batch size or use quantization:**
```bash
python convert_checkpoint.py \
    --model_dir ./models/Model \
    --output_dir ./checkpoints/model \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8
```

---

## Benchmarking Issues

### 11. TTFT and ITL Metrics Showing None

**Symptom:**
```
| Engine | Throughput | Latency | TTFT | ITL |
|--------|------------|---------|------|-----|
| vLLM   | 1234.5     | 100.2   | None | None|
```

**Cause:**
The benchmark script was calling non-streaming inference methods, which don't capture per-token timing.

**Solution:**

Use the `--streaming` flag when running benchmarks:

```bash
python benchmark.py \
    --model_path ./models/Model \
    --engines vllm tensorrt sglang \
    --streaming \
    --batch_sizes 1 4 8
```

The streaming mode uses the `run_streaming()` methods which capture:
- **TTFT**: Time from request start to first token
- **ITL**: Average time between consecutive tokens

---

### 12. Unfair Token Counting

**Symptom:**
SGLang showing much higher throughput than other engines.

**Cause:**
Original SGLang implementation used word count estimation (`len(output.split())`) instead of actual token counting.

**Solution:**

All engines now use the tokenizer for accurate counting:

```python
# Correct way (all engines)
tokens_generated = len(tokenizer.encode(output_text))

# Wrong way (was used in SGLang)
tokens_generated = len(output_text.split()) * 1.3  # Estimation - unfair!
```

This fix is already applied in `benchmark.py`.

---

### 12. GenAI-Bench CLI Argument Error

**Symptom:**
```
Error: No such option: --concurrency Did you mean --num-concurrency?
```

**Cause:**
The `genai-bench` CLI uses `--num-concurrency` instead of `--concurrency`.

**Solution:**

Use the correct argument name:

```bash
# Wrong
genai-bench benchmark --concurrency 10

# Correct
genai-bench benchmark --num-concurrency 10
```

This fix is applied in `genai_bench/run_genai_bench.py`.

---

## Environment & Path Issues

### 14. Virtual Environment Activation in Scripts

**Symptom:**
Scripts fail to find the correct packages even though they're installed.

**Cause:**
Running scripts without activating the correct virtual environment.

**Solution:**

Always activate the correct environment before running scripts:

```bash
# For TensorRT-LLM scripts
source /home/ec2-user/llm_host/.venv_trt/bin/activate
python LLM-serve/tensorrt/run_trt.py ...

# For vLLM scripts
source /home/ec2-user/llm_host/.venv_vllm/bin/activate
python LLM-serve/vllm/run_vllm.py ...

# For SGLang scripts
source /home/ec2-user/llm_host/.venv_sgl/bin/activate
python LLM-serve/sglang/run_sglang.py ...
```

Or use the full path to the Python interpreter:

```bash
/home/ec2-user/llm_host/.venv_vllm/bin/python LLM-serve/vllm/run_vllm.py ...
```

---

### 15. HuggingFace Token for Gated Models

**Symptom:**
```
401 Client Error: Unauthorized for url: https://huggingface.co/meta-llama/...
```

Or:
```
Access to model meta-llama/Llama-3.2-3B-Instruct is restricted
```

**Cause:**
Some models (Llama, Gemma) are gated and require authentication.

**Solution:**

1. **Get a HuggingFace token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read access

2. **Set the token in your environment:**
```bash
# Add to ~/.bashrc
export HF_TOKEN=your_token_here
source ~/.bashrc
```

3. **Accept the model license:**
   - Visit the model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   - Click "Agree and access repository"

4. **Use the token when downloading:**
```bash
huggingface-cli login
# Or
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --token $HF_TOKEN
```

---

## Quick Reference: Common Commands

### Clear All Caches
```bash
# Clear flashinfer cache (fixes nvcc path issues)
rm -rf ~/.cache/flashinfer

# Clear pip cache
rm -rf ~/.cache/pip
rm -rf /home/ec2-user/llm_host/tmp/pip_cache

# Clear HuggingFace cache (warning: re-downloads models)
rm -rf ~/.cache/huggingface
```

### Check GPU Status
```bash
# GPU info
nvidia-smi

# CUDA version
nvcc --version  # If installed
nvidia-smi | grep "CUDA Version"

# GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Find CUDA_HOME path
which nvcc                           # If nvcc is in PATH
find /usr/local -name "cuda*" -type d 2>/dev/null  # Common location
ls -la /usr/local/cuda               # Check symlink
echo $CUDA_HOME                      # Check if already set
```

### Verify Virtual Environment
```bash
# Check which Python is active
which python

# Check installed packages
pip list | grep -E "tensorrt|vllm|sglang|torch"

# Check torch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Summary Table

| Issue | Quick Fix |
|-------|-----------|
| No space left on device | `export TMPDIR=/path/to/large/disk/tmp` |
| TensorRT-LLM not found | `pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm` |
| Dependency conflicts | Use separate virtual environments |
| Flashinfer nvcc error | `--attention-backend triton --sampling-backend pytorch` |
| CUDA_HOME not set | `export CUDA_HOME=/usr/local/cuda` (find path with `which nvcc`) |
| CUDA graph failed | `--disable-cuda-graph` |
| nvcc vs Triton conflict | Choose one: install CUDA toolkit for nvcc OR use triton backend |
| Cached wrong nvcc path | `rm -rf ~/.cache/flashinfer` |
| TTFT/ITL showing None | Use `--streaming` flag |
| Gated model access | Set `HF_TOKEN` and accept license |

---

## Contributing

If you encounter new issues, please:
1. Document the symptom (error message)
2. Identify the cause
3. Provide the solution
4. Add to this document via PR
