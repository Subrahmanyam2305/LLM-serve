# Backend Servers

This directory contains OpenAI-compatible backend servers for each inference engine.

## Server Files

| File | Engine | Default Port | Description |
|------|--------|--------------|-------------|
| `vllm_backend_server.py` | vLLM | 8000 | vLLM OpenAI-compatible server |
| `sglang_backend_server.py` | SGLang | 8001 | SGLang OpenAI-compatible server |
| `tensorrt_backend_server.py` | TensorRT-LLM | 8002 | TensorRT-LLM HTTP server |
| `streaming_client.py` | - | - | Client to test streaming responses |

## Quick Start

### 1. Start a Backend Server

**vLLM Server:**
```bash
# Activate vLLM environment
source /home/ec2-user/llm_host/.venv_vllm/bin/activate

# Start server
python servers/vllm_backend_server.py \
    --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --port 8000
```

**SGLang Server:**
```bash
# Activate SGLang environment
source /home/ec2-user/llm_host/.venv_sgl/bin/activate

# Start server
python servers/sglang_backend_server.py \
    --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --port 8001
```

**TensorRT-LLM Server:**
```bash
# Activate TensorRT environment
source /home/ec2-user/llm_host/.venv_trt/bin/activate
export CUDA_HOME=/home/ec2-user/llm_host/.venv_trt/lib64/python3.12/site-packages/nvidia/cuda_runtime

# Start server
python servers/tensorrt_backend_server.py \
    --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B \
    --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --port 8002
```

### 2. Test with Streaming Client

```bash
# Test vLLM (port 8000)
python servers/streaming_client.py \
    --api-base http://localhost:8000 \
    --prompt "Explain machine learning" \
    --chat

# Test SGLang (port 8001)
python servers/streaming_client.py \
    --api-base http://localhost:8001 \
    --prompt "Explain machine learning" \
    --chat

# Test TensorRT-LLM (port 8002)
python servers/streaming_client.py \
    --api-base http://localhost:8002 \
    --prompt "Explain machine learning" \
    --chat
```

### 3. Test with curl

```bash
# Streaming chat completion
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    }'

# Non-streaming
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    }'
```

## Server Options

### vLLM Server Options
```
--model             Model path or HuggingFace model name (required)
--host              Host to bind to (default: 0.0.0.0)
--port              Port to listen on (default: 8000)
--dtype             Model dtype: float16, bfloat16, float32, auto
--gpu-memory-utilization  GPU memory utilization 0-1 (default: 0.9)
--max-model-len     Maximum model context length
--tensor-parallel-size    Number of GPUs for TP
--api-key           API key for authentication
```

### SGLang Server Options
```
--model             Model path or HuggingFace model name (required)
--host              Host to bind to (default: 0.0.0.0)
--port              Port to listen on (default: 8001)
--dtype             Model dtype: float16, bfloat16, float32, auto
--mem-fraction-static     Memory fraction for KV cache (default: 0.88)
--max-total-tokens  Maximum total tokens in system
--tensor-parallel-size    Number of GPUs for TP
--disable-cuda-graph      Disable CUDA graph (for debugging)
```

### TensorRT-LLM Server Options
```
--engine-dir        Path to TensorRT engine directory (required)
--tokenizer-dir     Path to tokenizer/HF model (required)
--host              Host to bind to (default: 0.0.0.0)
--port              Port to listen on (default: 8002)
--max-batch-size    Maximum batch size (default: 64)
--max-input-len     Maximum input length (default: 4096)
--max-output-len    Maximum output length (default: 1024)
```

### Streaming Client Options
```
--api-base          API base URL (default: http://localhost:8000)
--api-key           API key (default: EMPTY)
--model             Model name (auto-detected if not provided)
--prompt            Prompt to send
--max-tokens        Maximum tokens to generate (default: 256)
--temperature       Sampling temperature (default: 0.7)
--no-stream         Disable streaming
--chat              Use chat completions endpoint
--measure-ttft      Measure and display TTFT
```

## API Endpoints

All servers expose OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion (streaming supported) |
| `/health` | GET | Health check |

## Running with genai-bench

Once a server is running, you can use genai-bench:

```bash
# Install genai-bench
pip install genai-bench

# Run benchmark against vLLM server
genai-bench benchmark \
    --api-backend openai \
    --api-base http://localhost:8000 \
    --api-key EMPTY \
    --api-model-name Qwen2.5-3B-Instruct \
    --task text-to-text \
    --num-concurrency 1 4 8 16
```
