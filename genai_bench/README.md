# GenAI-Bench Integration

This directory contains the benchmark runner for testing TensorRT-LLM models served via Triton Inference Server.

## Overview

The `run_genai_bench.py` script provides two modes of operation:

1. **OpenAI Backend**: Uses the `genai-bench` CLI tool to benchmark OpenAI-compatible HTTP endpoints
2. **Triton Backend**: Uses the Triton gRPC client directly for more accurate concurrent request measurement

## Why This Approach?

The original `benchmarks/benchmark.py` used batch processing which doesn't accurately measure concurrent request handling. This implementation:

- Sends **independent concurrent requests** to the server
- Measures **per-request latency** including queue time
- Captures **Time to First Token (TTFT)** for streaming requests
- Provides **realistic throughput metrics** under concurrent load

## Usage

### Prerequisites

1. Start the Triton server with TensorRT-LLM model (see `servers/README.md`)
2. Install dependencies:
   ```bash
   pip install tritonclient[grpc] transformers genai-bench
   ```

### Running Benchmarks

**Using Triton gRPC client (recommended):**

```bash
python genai_bench/run_genai_bench.py \
    --backend triton \
    --triton_url localhost:8001 \
    --tokenizer_dir /path/to/tokenizer \
    --concurrency 1 2 4 8 16 32 \
    --max_requests 100 \
    --output_dir results
```

**Using genai-bench CLI:**

```bash
python genai_bench/run_genai_bench.py \
    --backend openai \
    --api_base http://localhost:8000 \
    --concurrency 1 2 4 8 16 32 \
    --max_requests 100
```

### Output

Results are saved to the `results/` directory in CSV format:

| Column | Description |
|--------|-------------|
| concurrency | Number of concurrent requests |
| total_requests | Total requests sent |
| successful_requests | Requests completed successfully |
| failed_requests | Requests that failed |
| total_time_s | Total benchmark time |
| requests_per_sec | Request throughput |
| throughput_tokens_per_sec | Token throughput |
| avg_latency_ms | Average request latency |
| p50_latency_ms | Median latency |
| p90_latency_ms | 90th percentile latency |
| p99_latency_ms | 99th percentile latency |
| avg_ttft_ms | Average time to first token |

## Concurrency Levels

The default concurrency levels are: 1, 2, 4, 8, 16, 32

These represent the number of concurrent client threads sending independent requests to the server. The server's scheduler handles these requests using:

- **Inflight batching**: Dynamically batches concurrent requests
- **KV cache management**: Efficiently manages memory for concurrent contexts
- **Request queuing**: Handles requests that exceed GPU capacity

## Comparison with Old Benchmark

| Aspect | Old benchmark.py | New run_genai_bench.py |
|--------|-----------------|------------------------|
| Request handling | Batch processing | True concurrent requests |
| Latency measurement | Single batch latency | Per-request latency distribution |
| TTFT | Not measurable | Accurate per-request |
| Queue time | Not included | Included |
| Scheduling overhead | Not included | Included |
| Realistic metrics | No | Yes |
