
# Server Benchmark Results: Triton TRT-LLM vs vLLM

## Configuration
- **Model**: phi-2
- **Precision**: FP16
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Dataset**: ShareGPT prompts
- **Max Output Tokens**: 512
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
| 1 | 71.58 | 76.82 | vLLM +6.8% |
| 2 | 111.70 | 119.79 | vLLM +6.7% |
| 4 | 234.59 | 254.31 | vLLM +7.8% |
| 8 | 455.15 | 482.75 | vLLM +5.7% |
| 16 | 747.17 | 746.56 | Triton +0.1% |
| 32 | 1282.32 | 1206.56 | Triton +6.3% |
| 64 | 1933.54 | 1571.09 | Triton +23.1% |

### Average Latency (ms) - End-to-End Generation Time

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 7152.52 | 6665.17 | vLLM +7.3% |
| 2 | 5862.99 | 6712.00 | Triton +12.6% |
| 4 | 6381.09 | 7188.05 | Triton +11.2% |
| 8 | 8612.74 | 7639.51 | vLLM +12.7% |
| 16 | 9580.85 | 9510.31 | vLLM +0.7% |
| 32 | 11199.81 | 12848.12 | Triton +12.8% |
| 64 | 13753.29 | 18964.14 | Triton +27.5% |

### Time to First Token (ms)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 13.78 | 201.97 | Triton +93.2% |
| 2 | 20.46 | 502.05 | Triton +95.9% |
| 4 | 24.91 | 340.86 | Triton +92.7% |
| 8 | 28.66 | 503.11 | Triton +94.3% |
| 16 | 29.88 | 1865.89 | Triton +98.4% |
| 32 | 32.80 | 2382.19 | Triton +98.6% |
| 64 | 50.69 | 3419.46 | Triton +98.5% |

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
# Run benchmark for Triton
python benchmark_triton.py \
    --engine triton \
    --tokenizer-dir /path/to/phi-2 \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Run benchmark for vLLM (after stopping Triton)
python benchmark_triton.py \
    --engine vllm \
    --tokenizer-dir /path/to/phi-2 \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Generate plots
python generate_plots.py --output-subfolder phi
```
