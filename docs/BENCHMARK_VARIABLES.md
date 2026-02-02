# Benchmark Variables and Metrics

This document defines the benchmark variables and metrics used for LLM inference benchmarking.

## Variables to Test

All benchmarks run on a single GPU (no tensor parallelism).

| Variable           | Values to Test                   | Notes |
| ------------------ | -------------------------------- | ----- |
| **Input tokens**   | 128, 256, 512, 1024, 2048, 4096  | Controlled via prompt selection/truncation |
| **Output tokens**  | 32, 64, 128, 256, 512, 1024      | Set via `--max_output_tokens` |
| **Batch size**     | 1, 2, 4, 8, 16, 32, 64           | Set via `--batch_sizes` |
| **Quantization**   | FP16, INT8, INT4 (AWQ/GPTQ), FP8 | Set during model conversion |
| **Model (family)** | Llama, Qwen, Gemma, Phi          | Set via `--model` in orchestrator |

## Metrics Measured

| Metric | Description | Unit | Implementation |
|--------|-------------|------|----------------|
| **Throughput** | Tokens generated per second | tokens/s | `total_tokens / total_time` |
| **Latency (avg)** | Average time per request/batch | ms | `mean(latencies)` |
| **Latency (std)** | Standard deviation of latency | ms | `stdev(latencies)` |
| **Latency (min)** | Minimum latency observed | ms | `min(latencies)` |
| **Latency (max)** | Maximum latency observed | ms | `max(latencies)` |
| **TTFT** | Time to First Token | ms | Time from request start to first token arrival |
| **ITL** | Inter-Token Latency (average) | ms | Average time between consecutive tokens |
| **GPU Memory** | Peak GPU memory during inference | MB | `torch.cuda.max_memory_allocated()` |

## Data Classes

### BenchmarkResult (benchmark.py)

```python
@dataclass
class BenchmarkResult:
    engine: str                      # Engine name (TensorRT-LLM, vLLM, SGLang)
    batch_size: int                  # Batch size used
    num_runs: int                    # Number of benchmark runs
    avg_latency_ms: float            # Average latency in ms
    std_latency_ms: float            # Standard deviation of latency
    min_latency_ms: float            # Minimum latency
    max_latency_ms: float            # Maximum latency
    throughput_tokens_per_sec: float # Tokens per second
    avg_tokens_per_request: float    # Average tokens generated per request
    total_tokens_generated: int      # Total tokens across all runs
    gpu_memory_mb: Optional[float]   # Peak GPU memory in MB
    ttft_ms: Optional[float]         # Time to first token (streaming only)
    itl_ms: Optional[float]          # Inter-token latency (streaming only)
```

### BenchmarkMetrics (run_trt.py)

```python
@dataclass
class BenchmarkMetrics:
    total_latency_ms: float          # Total inference time
    ttft_ms: float                   # Time to first token
    tokens_generated: int            # Number of tokens generated
    throughput_tokens_per_sec: float # Throughput
    itl_ms: float                    # Inter-token latency
```

## Dataset

### ShareGPT Dataset

- **Source**: [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- **Format**: Conversations with `{"from": "human"|"gpt", "value": "..."}`
- **Usage**: Extract first human turn from each conversation as prompts
- **Subset**: Use `--max_prompts N` for testing (recommended: 100-500)

### Fallback Prompts

For quick testing, use `benchmarks/prompts.json` with a small set of default prompts.

## Supported Models

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| Llama | meta-llama/Llama-3.2-3B-Instruct | Gated (requires HF_TOKEN) |
| Qwen | Qwen/Qwen2.5-3B-Instruct | Open access |
| Gemma | google/gemma-3-4b-it | Gated (requires HF_TOKEN) |
| Phi | microsoft/Phi-4-mini-instruct | Open access |

## Fair Comparison Guidelines

1. **Sampling Parameters**: Use identical settings across all engines
   - `temperature=0.0` (greedy decoding)
   - `top_k=1` (greedy)
   - `top_p=1.0` (no nucleus sampling)

2. **Token Counting**: Use actual tokenizer for all engines (not word estimation)

3. **Warmup**: Run warmup iterations before benchmarking to ensure GPU is warmed up

4. **Memory**: Reset peak memory stats before each benchmark run

5. **Synchronization**: Call `torch.cuda.synchronize()` after inference for accurate timing
