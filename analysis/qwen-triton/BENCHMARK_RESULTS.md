
# Server Benchmark Results: Triton TRT-LLM vs vLLM

## Configuration
- **Model**: Qwen2.5-3B-Instruct
- **Precision**: FP16
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Dataset**: ShareGPT prompts
- **Max Output Tokens**: 128
- **Benchmark Type**: Server-based with concurrent requests

## Key Differences

| Feature | Triton TRT-LLM | vLLM Server |
|---------|----------------|-------------|
| Batching | Inflight Batching | Continuous Batching |
| Memory Management | TensorRT optimized | PagedAttention |
| Protocol | gRPC | HTTP/REST |
| Streaming | Native | SSE |

## Results

### Throughput (tokens/sec)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 66.10 | 67.86 | vLLM +2.6% |
| 2 | 131.12 | 97.58 | Triton +34.4% |
| 4 | 259.88 | 168.94 | Triton +53.8% |
| 8 | 453.30 | 276.87 | Triton +63.7% |
| 16 | 849.19 | 526.03 | Triton +61.4% |
| 32 | 1494.37 | 848.49 | Triton +76.1% |
| 64 | 2414.61 | 1487.42 | Triton +62.3% |

### Average Latency (ms) - End-to-End Generation Time

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 15491.74 | 12492.76 | vLLM +24.0% |
| 2 | 15611.09 | 9335.51 | vLLM +67.2% |
| 4 | 15755.46 | 8539.74 | vLLM +84.5% |
| 8 | 18064.59 | 7309.06 | vLLM +147.2% |
| 16 | 19276.89 | 7478.82 | vLLM +157.8% |
| 32 | 21883.20 | 8259.44 | vLLM +164.9% |
| 64 | 26959.64 | 9911.49 | vLLM +172.0% |

### Time to First Token (ms)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 15.99 | 29.49 | Triton +45.8% |
| 2 | 23.73 | 29.46 | Triton +19.4% |
| 4 | 28.89 | 31.79 | Triton +9.1% |
| 8 | 32.03 | 45.75 | Triton +30.0% |
| 16 | 33.96 | 57.18 | Triton +40.6% |
| 32 | 40.25 | 75.87 | Triton +47.0% |
| 64 | 68.49 | 96.68 | Triton +29.2% |

## Key Findings

1. **Throughput at High Concurrency**: Compare how both engines scale with concurrent requests
2. **Latency Under Load**: Observe latency behavior as concurrency increases
3. **Scaling Efficiency**: Both engines use dynamic batching for better GPU utilization

## Notes

- Triton TRT-LLM uses inflight batching via the TensorRT-LLM backend
- vLLM uses continuous batching with PagedAttention for memory efficiency
- Both servers handle concurrent requests dynamically (no static batching)
- Results may vary based on prompt length distribution and GPU memory

## How to Reproduce

```bash
# Start Triton server with TensorRT-LLM backend
# (See Triton TensorRT-LLM documentation)

# Start vLLM server
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000

# Run benchmark
python benchmark_triton.py \
    --triton-url localhost:8001 \
    --vllm-url http://localhost:8000 \
    --tokenizer-dir /path/to/Qwen2.5-3B-Instruct \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt
```
