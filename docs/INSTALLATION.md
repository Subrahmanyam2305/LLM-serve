# Installation Guide

This guide covers the setup of Triton Inference Server with TensorRT-LLM backend for accurate concurrent request benchmarking.

## Prerequisites

- NVIDIA GPU (A10G, A100, H100, etc.)
- NVIDIA Driver 525+ installed
- Docker installed and configured
- Python 3.10+

## Quick Setup

### 1. Configure Docker for GPU Access

```bash
# Add user to docker group (if not already done)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker works with GPU
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### 2. Install Python Dependencies

```bash
# Activate TensorRT environment
source /home/ec2-user/llm_host/.venv_trt/bin/activate

# Install required packages
pip install tritonclient[grpc] transformers genai-bench
```

### 3. Pull Triton Docker Image

```bash
# Pull the TensorRT-LLM enabled Triton image
docker pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
```

This image is ~15GB and includes:
- Triton Inference Server
- TensorRT-LLM backend
- Python backend for pre/post processing

### 4. Verify TensorRT Engine

Ensure you have a pre-built TensorRT-LLM engine:

```bash
# Check engine exists
ls -la /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B/

# Should contain:
# - config.json
# - rank0.engine (or similar)
```

If you need to build an engine, see the TensorRT-LLM documentation.

## Running the Server

### Option 1: Using Docker (Recommended)

```bash
# Launch Triton server
python servers/launch_triton_server.py \
    --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B \
    --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --use-docker \
    --detach

# Verify server is running
curl http://localhost:8000/v2/health/ready
```

### Option 2: Manual Docker Run

```bash
# Create model repository
mkdir -p /tmp/triton_model_repo/tensorrt_llm/1

# Run Triton container
docker run --rm -it \
    --gpus all \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v /tmp/triton_model_repo:/triton_model_repo \
    -v /home/ec2-user/llm_host/trt_engines:/engines \
    -v /home/ec2-user/llm_host/Qwen2.5-3B-Instruct:/tokenizer \
    nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 \
    tritonserver --model-repository=/triton_model_repo
```

## Running Benchmarks

### Using Triton gRPC Client

```bash
# Run benchmark with multiple concurrency levels
python genai_bench/run_genai_bench.py \
    --backend triton \
    --triton_url localhost:8001 \
    --tokenizer_dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --concurrency 1 2 4 8 16 32 \
    --max_requests 100 \
    --output_dir results
```

### Using genai-bench CLI

```bash
# Run benchmark against HTTP endpoint
python genai_bench/run_genai_bench.py \
    --backend openai \
    --api_base http://localhost:8000 \
    --concurrency 1 2 4 8 16 32 \
    --max_requests 100
```

## Verifying Installation

### 1. Check Server Health

```bash
# HTTP health check
curl http://localhost:8000/v2/health/ready

# Should return: {"ready": true}
```

### 2. List Models

```bash
curl http://localhost:8000/v2/models

# Should list tensorrt_llm model
```

### 3. Test Inference

```bash
# Test with Triton client
python triton_client/triton_grpc_client.py \
    --url localhost:8001 \
    --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
    --prompt "Hello, how are you?" \
    --max-tokens 50
```

## Troubleshooting

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply changes (or logout/login)
newgrp docker
```

### GPU Not Available in Docker

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Server Not Starting

1. Check GPU memory: `nvidia-smi`
2. Check Docker logs: `docker logs triton-trtllm`
3. Verify engine path is correct
4. Ensure ports are not in use

### Connection Refused

1. Wait for server to fully initialize (1-2 minutes)
2. Check server is running: `docker ps`
3. Check ports: `netstat -tlnp | grep 800`

## Directory Structure

After setup, your directory should look like:

```
LLM-serve-triton/
├── genai_bench/
│   ├── README.md
│   └── run_genai_bench.py
├── servers/
│   ├── README.md
│   └── launch_triton_server.py
├── triton_client/
│   ├── __init__.py
│   └── triton_grpc_client.py
├── results/
│   └── (benchmark results)
├── INSTALLATION.md
├── ISSUES_FACED.md
└── README.md
```

## Next Steps

1. Run benchmarks with different concurrency levels
2. Compare results with batch processing approach
3. Analyze latency distribution and throughput
4. Tune server parameters for your use case
