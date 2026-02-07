
# Server Benchmark Results: Triton TRT-LLM vs vLLM

## Configuration
- **Model**: Qwen2.5-3B-Instruct
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
| 1 | 62.30 | 69.17 | vLLM +9.9% |
| 2 | 123.75 | 130.47 | vLLM +5.2% |
| 4 | 245.62 | 255.02 | vLLM +3.7% |
| 8 | 434.76 | 507.07 | vLLM +14.3% |
| 16 | 824.81 | 833.44 | vLLM +1.0% |
| 32 | 1485.66 | 1442.16 | Triton +3.0% |
| 64 | 2486.23 | 2426.95 | Triton +2.4% |

### Average Latency (ms) - End-to-End Generation Time

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 8217.62 | 7401.76 | vLLM +11.0% |
| 2 | 8266.87 | 7848.63 | vLLM +5.3% |
| 4 | 8332.51 | 8026.85 | vLLM +3.8% |
| 8 | 9414.51 | 8075.74 | vLLM +16.6% |
| 16 | 9911.83 | 8298.17 | vLLM +19.4% |
| 32 | 10984.98 | 9003.70 | vLLM +22.0% |
| 64 | 13032.08 | 11170.32 | vLLM +16.7% |

### Time to First Token (ms)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 16.08 | 224.30 | Triton +92.8% |
| 2 | 23.80 | 393.16 | Triton +93.9% |
| 4 | 29.35 | 298.27 | Triton +90.2% |
| 8 | 34.03 | 454.13 | Triton +92.5% |
| 16 | 35.30 | 2046.52 | Triton +98.3% |
| 32 | 40.19 | 2118.40 | Triton +98.1% |
| 64 | 60.78 | 2234.21 | Triton +97.3% |

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
    --tokenizer-dir /path/to/Qwen2.5-3B-Instruct \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Run benchmark for vLLM (after stopping Triton)
python benchmark_triton.py \
    --engine vllm \
    --tokenizer-dir /path/to/Qwen2.5-3B-Instruct \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Generate plots
python generate_plots.py --output-subfolder qwen
```
