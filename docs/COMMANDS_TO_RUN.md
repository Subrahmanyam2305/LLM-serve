# Run Triton server
python servers/launch_triton_server.py         --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B         --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct         --http-port 8000         --grpc-port 8001

# Benchmark Triton server
python benchmarks/benchmark_triton.py --engine triton --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --concurrency 1 2 4 8 16 32 64 --sharegpt --url localhost:8001

# Run vLLM server
python servers/vllm_backend_server.py --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --dtype float16

# Benchmark vLLM server (model name is auto-detected)
python benchmarks/benchmark_triton.py --engine vllm --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --concurrency 1 2 4 8 16 32 64 --sharegpt --url http://localhost:8000

# Commands to Run

Quick reference for starting servers and running benchmarks.

---

## Environment Setup

```bash
# Set temp directory (prevents "No space left" errors)
export TMPDIR=/home/ec2-user/llm_host/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR

# Set CUDA_HOME (if needed for flashinfer)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

---

## Start Backend Servers

### vLLM Server (Port 8000)

```bash
cd /home/ec2-user/llm_host/LLM-serve
source /home/ec2-user/llm_host/.venv_vllm/bin/activate

python servers/vllm_backend_server.py \
    --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --port 8000
```

### SGLang Server (Port 8001)

```bash
cd /home/ec2-user/llm_host/LLM-serve
source /home/ec2-user/llm_host/.venv_sgl/bin/activate

python servers/sglang_backend_server.py \
    --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --port 8001
```

### TensorRT-LLM Server (Port 8002)

```bash
cd /home/ec2-user/llm_host/LLM-serve
source /home/ec2-user/llm_host/.venv_trt/bin/activate

python servers/tensorrt_backend_server.py \
    --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B \
    --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --port 8002
```

---

## Test Servers

### Check Server Health

```bash
# vLLM
curl http://localhost:8000/v1/models

# SGLang
curl http://localhost:8001/v1/models

# TensorRT-LLM
curl http://localhost:8002/v1/models
```

### Streaming Client

```bash
cd /home/ec2-user/llm_host/LLM-serve

# Test vLLM
python servers/streaming_client.py --port 8000 --prompt "Hello, how are you?"

# Test SGLang
python servers/streaming_client.py --port 8001 --prompt "Hello, how are you?"

# Test TensorRT-LLM
python servers/streaming_client.py --port 8002 --prompt "Hello, how are you?"
```

### cURL Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

---

## Run Benchmarks

```bash
cd /home/ec2-user/llm_host/LLM-serve/benchmarks
source /home/ec2-user/llm_host/.venv_vllm/bin/activate

# Single engine
python benchmark.py \
    --model_path /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --engines vllm \
    --batch_sizes 1 4 8 16 \
    --streaming

# With ShareGPT dataset
python benchmark.py \
    --model_path /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --engines vllm \
    --sharegpt --max_prompts 100 \
    --batch_sizes 1 4 8 \
    --streaming
```

---

## TensorRT-LLM Engine Build

```bash
cd /home/ec2-user/llm_host/LLM-serve/tensorrt
source /home/ec2-user/llm_host/.venv_trt/bin/activate

# Full pipeline: download → convert → build → run
python orchestrate_trt.py \
    --model qwen \
    --models_dir /home/ec2-user/llm_host \
    --checkpoints_dir /home/ec2-user/llm_host/trt_checkpoints \
    --engines_dir /home/ec2-user/llm_host/trt_engines \
    --dtype float16
```

---

## Kill Servers

```bash
# Find and kill by port
lsof -ti:8000 | xargs kill -9  # vLLM
lsof -ti:8001 | xargs kill -9  # SGLang
lsof -ti:8002 | xargs kill -9  # TensorRT-LLM

# Or kill all Python processes (careful!)
pkill -f "vllm_backend_server"
pkill -f "sglang_backend_server"
pkill -f "tensorrt_backend_server"
```

---

## Clear Caches

```bash
rm -rf ~/.cache/flashinfer    # Fix nvcc path issues
rm -rf ~/.cache/huggingface   # Re-download models
rm -rf $TMPDIR/pip_cache      # Clear pip cache
```
