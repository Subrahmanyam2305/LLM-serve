# Triton TensorRT-LLM Server

This directory contains scripts for launching Triton Inference Server with TensorRT-LLM backend.

## Overview

The Triton Inference Server provides proper concurrent request handling through:

- **Inflight Batching**: Dynamically batches concurrent requests for optimal GPU utilization
- **KV Cache Management**: Efficiently manages memory for concurrent contexts
- **Request Queuing**: Handles requests that exceed GPU capacity
- **gRPC/HTTP APIs**: Standard interfaces for client communication

## Prerequisites

1. **TensorRT-LLM Engine**: Pre-built engine for your model
2. **Docker**: For running Triton container (recommended)
3. **NVIDIA GPU**: With appropriate drivers installed

## Quick Start

### 1. Launch Triton Server with Docker

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

### 2. Verify Server is Running

```bash
# Check health
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models
```

### 3. Run Benchmark

```bash
python genai_bench/run_genai_bench.py \
    --backend triton \
    --triton_url localhost:8001 \
    --tokenizer_dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --concurrency 1 2 4 8 16 32
```

## Server Options

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

## API Endpoints

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/health/ready` | GET | Server readiness check |
| `/v2/health/live` | GET | Server liveness check |
| `/v2/models` | GET | List available models |
| `/v2/models/{model}/infer` | POST | Run inference |

### gRPC Endpoint

The gRPC endpoint is available at `localhost:8001` and supports:

- Unary inference
- Streaming inference
- Async inference

Use the `triton_client/triton_grpc_client.py` for gRPC communication.

## Docker Container Management

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
