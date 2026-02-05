# Backend Servers

This directory contains OpenAI-compatible backend servers for each inference engine.

## Server Files

| File | Engine | Default Port | Description |
|------|--------|--------------|-------------|
| `vllm_backend_server.py` | vLLM | 8000 | vLLM OpenAI-compatible server |
| `sglang_backend_server.py` | SGLang | 8001 | SGLang OpenAI-compatible server |
| `tensorrt_backend_server.py` | TensorRT-LLM | 8002 | TensorRT-LLM HTTP server (threaded) |
| `launch_triton_server.py` | Triton | 8000/8001 | Triton Inference Server launcher |
| `triton_openai_wrapper.py` | Triton | 8080 | OpenAI wrapper for Triton gRPC |
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

## Triton Inference Server

For production deployments with proper concurrent request handling, use Triton Inference Server.

### Launch Triton Server with Docker

```bash
# Activate TensorRT environment
source /home/ec2-user/llm_host/.venv_trt/bin/activate

# Launch server
python servers/launch_triton_server.py \
    --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B \
    --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --use-docker \
    --detach
```

### Verify Server is Running

```bash
# Check health
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models
```

### Run Benchmark with Triton

```bash
python genai_bench/run_genai_bench.py \
    --backend triton \
    --triton_url localhost:8001 \
    --tokenizer_dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --concurrency 1 2 4 8 16 32
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

### Triton Server Options
| Option | Default | Description |
|--------|---------|-------------|
| `--engine-dir` | Required | Path to TensorRT-LLM engine |
| `--tokenizer-dir` | Required | Path to tokenizer/HF model |
| `--http-port` | 8000 | HTTP API port |
| `--grpc-port` | 8001 | gRPC API port |
| `--metrics-port` | 8002 | Prometheus metrics port |
| `--max-batch-size` | 64 | Maximum batch size |
| `--use-docker` | False | Run in Docker container |
| `--detach` | False | Run container in background |

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

### Triton-specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/health/ready` | GET | Server readiness check |
| `/v2/health/live` | GET | Server liveness check |
| `/v2/models` | GET | List available models |
| `/v2/models/{model}/infer` | POST | Run inference |

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

## Docker Container Management (Triton)

```bash
# View logs
docker logs triton-trtllm

# Stop server
docker stop triton-trtllm

# Remove container
docker rm triton-trtllm

# View running containers
docker ps
```

## Troubleshooting

### Server Not Starting

1. Check Docker is running: `docker info`
2. Check GPU is available: `nvidia-smi`
3. Check engine path is correct
4. View container logs: `docker logs triton-trtllm`

### Connection Refused

1. Wait for server to fully start (can take 1-2 minutes)
2. Check ports are not in use: `netstat -tlnp | grep 800`
3. Check firewall settings

### Out of Memory

1. Reduce `--max-batch-size`
2. Use smaller model or quantization
3. Check GPU memory: `nvidia-smi`

## Performance Tuning

For optimal performance:

1. **Batch Size**: Set `--max-batch-size` based on GPU memory
2. **KV Cache**: Configure in engine build for your use case
3. **Scheduling**: Default `max_utilization` policy works well for most cases

See [TensorRT-LLM Performance Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md) for more details.
