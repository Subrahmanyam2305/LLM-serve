
# Server Benchmark Results: Triton TRT-LLM vs vLLM

## Configuration
- **Model**: gemma-3-1b-it
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
| 1 | 127.24 | 169.15 | vLLM +24.8% |
| 2 | 254.35 | 324.74 | vLLM +21.7% |
| 4 | 495.97 | 629.67 | vLLM +21.2% |
| 8 | 860.66 | 1230.77 | vLLM +30.1% |
| 16 | 1599.08 | 2181.64 | vLLM +26.7% |
| 32 | 2961.60 | 3492.26 | vLLM +15.2% |
| 64 | 3972.16 | 5120.85 | vLLM +22.4% |

### Average Latency (ms) - End-to-End Generation Time

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 4023.25 | 3026.81 | vLLM +32.9% |
| 2 | 4022.13 | 3153.25 | vLLM +27.6% |
| 4 | 4125.75 | 3154.05 | vLLM +30.8% |
| 8 | 4752.71 | 3244.30 | vLLM +46.5% |
| 16 | 5086.85 | 3602.70 | vLLM +41.2% |
| 32 | 5485.61 | 4496.92 | vLLM +22.0% |
| 64 | 7986.73 | 5981.58 | vLLM +33.5% |

### Time to First Token (ms)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
| 1 | 7.71 | 97.27 | Triton +92.1% |
| 2 | 11.39 | 171.78 | Triton +93.4% |
| 4 | 14.65 | 130.91 | Triton +88.8% |
| 8 | 16.80 | 196.18 | Triton +91.4% |
| 16 | 45.93 | 811.57 | Triton +94.3% |
| 32 | 24.98 | 927.84 | Triton +97.3% |
| 64 | 44.09 | 1108.54 | Triton +96.0% |

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
    --tokenizer-dir /path/to/gemma-3-1b-it \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Run benchmark for vLLM (after stopping Triton)
python benchmark_triton.py \
    --engine vllm \
    --tokenizer-dir /path/to/gemma-3-1b-it \
    --concurrency 1 4 8 16 32 64 \
    --sharegpt

# Generate plots
python generate_plots.py --output-subfolder gemma
```
