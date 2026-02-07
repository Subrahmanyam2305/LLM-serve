
# Server Benchmark Results: Triton TRT-LLM vs vLLM

## Configuration
- **Model**: Llama-3.2-3B-Instruct
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
| 1 | 61.55 | 66.61 | vLLM +7.6% |
| 2 | 122.04 | 127.45 | vLLM +4.2% |
| 4 | 215.60 | 225.68 | vLLM +4.5% |
| 8 | 400.65 | 463.99 | vLLM +13.7% |
| 16 | 704.37 | 764.84 | vLLM +7.9% |
| 32 | 1349.38 | 1453.85 | vLLM +7.2% |
| 64 | 1407.15 | 2476.51 | vLLM +43.2% |

### Average Latency (ms) - End-to-End Generation Time

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 8317.44 | 7686.04 | vLLM +8.2% |
| 2 | 8382.75 | 8034.26 | vLLM +4.3% |
| 4 | 7703.20 | 8122.22 | Triton +5.2% |
| 8 | 9215.95 | 8365.68 | vLLM +10.2% |
| 16 | 9154.22 | 9524.92 | Triton +3.9% |
| 32 | 10471.32 | 10497.80 | Triton +0.3% |
| 64 | 15264.84 | 12285.83 | vLLM +24.2% |

### Time to First Token (ms)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 16.27 | 247.00 | Triton +93.4% |
| 2 | 24.04 | 416.59 | Triton +94.2% |
| 4 | 29.48 | 351.85 | Triton +91.6% |
| 8 | 33.19 | 510.00 | Triton +93.5% |
| 16 | 34.11 | 1874.53 | Triton +98.2% |
| 32 | 38.43 | 1944.18 | Triton +98.0% |
| 64 | 4821.63 | 2092.25 | vLLM +130.5% |

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
    --tokenizer-dir /path/to/Llama-3.2-3B-Instruct \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Run benchmark for vLLM (after stopping Triton)
python benchmark_triton.py \
    --engine vllm \
    --tokenizer-dir /path/to/Llama-3.2-3B-Instruct \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Generate plots
python generate_plots.py --output-subfolder llama
```
