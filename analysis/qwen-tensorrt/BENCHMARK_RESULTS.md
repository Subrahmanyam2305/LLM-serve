
# Benchmark Results Summary

## Configuration
- **Model**: Qwen2.5-3B-Instruct
- **Precision**: FP16
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Dataset**: ShareGPT (100 prompts from ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json)
- **Max Output Tokens**: 128
- **Benchmark Runs**: 5

## Results

### Throughput (tokens/sec)

| Batch Size | TensorRT-LLM | vLLM | Difference |
|------------|--------------|------|------------|
| 1 | 71.26 | 68.18 | TRT +4.5% |
| 2 | 141.35 | 129.89 | TRT +8.8% |
| 4 | 282.00 | 253.91 | TRT +11.1% |
| 8 | 490.70 | 504.92 | vLLM +2.8% |
| 16 | 852.21 | 924.24 | vLLM +7.8% |
| 32 | 1458.65 | 1636.74 | vLLM +10.9% |
| 64 | 2359.84 | 2545.08 | vLLM +7.3% |

### Time to First Token (TTFT) in ms

| Batch Size | TensorRT-LLM | vLLM | Difference |
|------------|--------------|------|------------|
| 1 | 17.64 | 208.59 | TRT +91.5% |
| 2 | 21.51 | 343.32 | TRT +93.7% |
| 4 | 23.85 | 269.55 | TRT +91.2% |
| 8 | 41.20 | 390.26 | TRT +89.4% |
| 16 | 246.45 | 1137.57 | TRT +78.3% |
| 32 | 435.34 | 1176.09 | TRT +63.0% |
| 64 | 705.37 | 1415.44 | TRT +50.2% |

## Key Findings

1. **TTFT (Time to First Token)**: TensorRT-LLM shows significantly lower TTFT across all batch sizes
2. **Throughput**: Both engines show similar throughput, with vLLM slightly ahead at larger batch sizes
3. **Scaling Efficiency**: vLLM scales better from batch 1 to larger batches due to PagedAttention

## Notes

- SGLang benchmark used triton backend (slower than flashinfer) due to nvcc compilation issues
- All tests used greedy decoding (temperature=0.0) for reproducibility
- TTFT is measured using streaming inference
- Results may vary based on prompt length distribution

## Dataset

The benchmark uses prompts from the ShareGPT dataset:
- Source: `anon8231489123/ShareGPT_Vicuna_unfiltered`
- File: `ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
- This file removes instances of "I'm sorry" and "do not have the capability" to avoid refusal responses
- Only the first human turn (`from: "human"`) from each conversation is used as the prompt
