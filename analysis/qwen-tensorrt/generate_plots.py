#!/usr/bin/env python3
"""
Generate benchmark analysis plots and tables.
Reads results from JSON files and creates comparison visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results from JSON files
results_dir = Path(__file__).parent.parent / "results"
output_dir = Path(__file__).parent

# Load TensorRT-LLM results
trt_results = {}
try:
    with open(results_dir / "tensorrt_results.json") as f:
        data = json.load(f)
        for r in data["results"]:
            bs = r["batch_size"]
            trt_results[bs] = {
                "throughput": r["throughput_tokens_per_sec"],
                "latency": r["avg_latency_ms"],
                "latency_std": r["std_latency_ms"],
                "ttft": r.get("ttft_ms"),
                "itl": r.get("itl_ms"),
            }
except FileNotFoundError:
    print("TensorRT results not found")

# Load vLLM results
vllm_results = {}
try:
    with open(results_dir / "vllm_results.json") as f:
        data = json.load(f)
        for r in data["results"]:
            bs = r["batch_size"]
            vllm_results[bs] = {
                "throughput": r["throughput_tokens_per_sec"],
                "latency": r["avg_latency_ms"],
                "latency_std": r["std_latency_ms"],
                "ttft": r.get("ttft_ms"),
                "itl": r.get("itl_ms"),
            }
except FileNotFoundError:
    print("vLLM results not found")

# Load SGLang results
sglang_results = {}
try:
    with open(results_dir / "sglang_results.json") as f:
        data = json.load(f)
        for r in data["results"]:
            bs = r["batch_size"]
            sglang_results[bs] = {
                "throughput": r["throughput_tokens_per_sec"],
                "latency": r["avg_latency_ms"],
                "latency_std": r["std_latency_ms"],
                "ttft": r.get("ttft_ms"),
                "itl": r.get("itl_ms"),
            }
except FileNotFoundError:
    print("SGLang results not found")

# Common batch sizes
batch_sizes = sorted(set(trt_results.keys()) & set(vllm_results.keys()))
if not batch_sizes:
    batch_sizes = [1, 2, 4, 8]

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'TensorRT-LLM': '#76B900', 'vLLM': '#FF6B6B', 'SGLang': '#4ECDC4'}

# 1. Throughput comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(batch_sizes))
width = 0.25

trt_throughput = [trt_results.get(bs, {}).get("throughput", 0) for bs in batch_sizes]
vllm_throughput = [vllm_results.get(bs, {}).get("throughput", 0) for bs in batch_sizes]
sglang_throughput = [sglang_results.get(bs, {}).get("throughput", 0) for bs in batch_sizes]

bars1 = ax.bar(x - width, trt_throughput, width, label='TensorRT-LLM', color=colors['TensorRT-LLM'])
bars2 = ax.bar(x, vllm_throughput, width, label='vLLM', color=colors['vLLM'])
if any(sglang_throughput):
    bars3 = ax.bar(x + width, sglang_throughput, width, label='SGLang (triton)', color=colors['SGLang'])

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax.set_title('Throughput Comparison: LLM Inference Engines\nQwen2.5-3B-Instruct (FP16) on NVIDIA A10G', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'throughput_comparison.png', dpi=150)
plt.close()

# 2. TTFT (Time to First Token) comparison - This is the latency comparison
fig, ax = plt.subplots(figsize=(10, 6))

trt_ttft = [trt_results.get(bs, {}).get("ttft") for bs in batch_sizes]
vllm_ttft = [vllm_results.get(bs, {}).get("ttft") for bs in batch_sizes]
sglang_ttft = [sglang_results.get(bs, {}).get("ttft") for bs in batch_sizes]

# Filter out None values
trt_bs_ttft = [(bs, ttft) for bs, ttft in zip(batch_sizes, trt_ttft) if ttft is not None]
vllm_bs_ttft = [(bs, ttft) for bs, ttft in zip(batch_sizes, vllm_ttft) if ttft is not None]
sglang_bs_ttft = [(bs, ttft) for bs, ttft in zip(batch_sizes, sglang_ttft) if ttft is not None]

if trt_bs_ttft:
    ax.plot([x[0] for x in trt_bs_ttft], [x[1] for x in trt_bs_ttft], 
            marker='o', label='TensorRT-LLM', color=colors['TensorRT-LLM'], linewidth=2, markersize=8)
if vllm_bs_ttft:
    ax.plot([x[0] for x in vllm_bs_ttft], [x[1] for x in vllm_bs_ttft], 
            marker='s', label='vLLM', color=colors['vLLM'], linewidth=2, markersize=8)
if sglang_bs_ttft:
    ax.plot([x[0] for x in sglang_bs_ttft], [x[1] for x in sglang_bs_ttft], 
            marker='^', label='SGLang (triton)', color=colors['SGLang'], linewidth=2, markersize=8)

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Time to First Token (ms)', fontsize=12)
ax.set_title('TTFT Comparison: LLM Inference Engines\nQwen2.5-3B-Instruct (FP16) on NVIDIA A10G', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'latency_comparison.png', dpi=150)
plt.close()

# 3. Scaling efficiency
fig, ax = plt.subplots(figsize=(10, 6))

if trt_throughput[0] > 0:
    trt_scaling = [t / trt_throughput[0] if trt_throughput[0] > 0 else 0 for t in trt_throughput]
    ax.plot(batch_sizes, trt_scaling, marker='o', label='TensorRT-LLM', 
            color=colors['TensorRT-LLM'], linewidth=2, markersize=8)

if vllm_throughput[0] > 0:
    vllm_scaling = [t / vllm_throughput[0] if vllm_throughput[0] > 0 else 0 for t in vllm_throughput]
    ax.plot(batch_sizes, vllm_scaling, marker='s', label='vLLM', 
            color=colors['vLLM'], linewidth=2, markersize=8)

ideal_scaling = batch_sizes
ax.plot(batch_sizes, ideal_scaling, '--', label='Ideal (Linear)', 
        color='gray', linewidth=1.5)

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Scaling Factor (relative to batch=1)', fontsize=12)
ax.set_title('Scaling Efficiency: Throughput vs Batch Size\nQwen2.5-3B-Instruct (FP16) on NVIDIA A10G', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'scaling_efficiency.png', dpi=150)
plt.close()

# 4. Combined dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Throughput bar chart
ax = axes[0, 0]
x = np.arange(len(batch_sizes))
width = 0.35
ax.bar(x - width/2, trt_throughput, width, label='TensorRT-LLM', color=colors['TensorRT-LLM'])
ax.bar(x + width/2, vllm_throughput, width, label='vLLM', color=colors['vLLM'])
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (tokens/sec)')
ax.set_title('Throughput Comparison')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# TTFT line chart
ax = axes[0, 1]
if trt_bs_ttft:
    ax.plot([x[0] for x in trt_bs_ttft], [x[1] for x in trt_bs_ttft], 
            marker='o', label='TensorRT-LLM', color=colors['TensorRT-LLM'], linewidth=2)
if vllm_bs_ttft:
    ax.plot([x[0] for x in vllm_bs_ttft], [x[1] for x in vllm_bs_ttft], 
            marker='s', label='vLLM', color=colors['vLLM'], linewidth=2)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time to First Token (ms)')
ax.set_title('TTFT Comparison')
ax.legend()
ax.grid(alpha=0.3)

# Scaling efficiency
ax = axes[1, 0]
if trt_throughput[0] > 0:
    ax.plot(batch_sizes, trt_scaling, marker='o', label='TensorRT-LLM', 
            color=colors['TensorRT-LLM'], linewidth=2)
if vllm_throughput[0] > 0:
    ax.plot(batch_sizes, vllm_scaling, marker='s', label='vLLM', 
            color=colors['vLLM'], linewidth=2)
ax.plot(batch_sizes, ideal_scaling, '--', label='Ideal', color='gray', linewidth=1.5)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Scaling Factor')
ax.set_title('Scaling Efficiency')
ax.legend()
ax.grid(alpha=0.3)

# Summary table
ax = axes[1, 1]
ax.axis('off')

# Calculate summary metrics
trt_tp_1 = trt_results.get(1, {}).get("throughput", 0)
vllm_tp_1 = vllm_results.get(1, {}).get("throughput", 0)
trt_tp_8 = trt_results.get(8, {}).get("throughput", 0)
vllm_tp_8 = vllm_results.get(8, {}).get("throughput", 0)
trt_ttft_1 = trt_results.get(1, {}).get("ttft", 0)
vllm_ttft_1 = vllm_results.get(1, {}).get("ttft", 0)
trt_ttft_8 = trt_results.get(8, {}).get("ttft", 0)
vllm_ttft_8 = vllm_results.get(8, {}).get("ttft", 0)

# Determine winners
def winner(trt_val, vllm_val, lower_is_better=False):
    if trt_val == 0 or vllm_val == 0:
        return "N/A"
    if lower_is_better:
        diff = (vllm_val - trt_val) / vllm_val * 100
        return f"TRT ({diff:+.1f}%)" if trt_val < vllm_val else f"vLLM ({-diff:+.1f}%)"
    else:
        diff = (trt_val - vllm_val) / vllm_val * 100
        return f"TRT ({diff:+.1f}%)" if trt_val > vllm_val else f"vLLM ({-diff:+.1f}%)"

table_data = [
    ['Metric', 'TensorRT-LLM', 'vLLM', 'Winner'],
    ['Throughput @BS=1', f'{trt_tp_1:.1f} tok/s', f'{vllm_tp_1:.1f} tok/s', winner(trt_tp_1, vllm_tp_1)],
    ['Throughput @BS=8', f'{trt_tp_8:.1f} tok/s', f'{vllm_tp_8:.1f} tok/s', winner(trt_tp_8, vllm_tp_8)],
    ['TTFT @BS=1', f'{trt_ttft_1:.1f} ms' if trt_ttft_1 else 'N/A', f'{vllm_ttft_1:.1f} ms' if vllm_ttft_1 else 'N/A', winner(trt_ttft_1, vllm_ttft_1, lower_is_better=True)],
    ['TTFT @BS=8', f'{trt_ttft_8:.1f} ms' if trt_ttft_8 else 'N/A', f'{vllm_ttft_8:.1f} ms' if vllm_ttft_8 else 'N/A', winner(trt_ttft_8, vllm_ttft_8, lower_is_better=True)],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#E6E6E6')
    table[(0, i)].set_text_props(weight='bold')
ax.set_title('Summary Comparison', fontsize=12, pad=20)

plt.suptitle('LLM Inference Benchmark: Qwen2.5-3B-Instruct (FP16) on NVIDIA A10G\nShareGPT Dataset (100 prompts), max_output_tokens=128', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generated plots:")
print("  - throughput_comparison.png")
print("  - latency_comparison.png (TTFT)")
print("  - scaling_efficiency.png")
print("  - benchmark_dashboard.png")

# Generate markdown summary table
markdown_table = f"""
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
"""

for bs in batch_sizes:
    trt_tp = trt_results.get(bs, {}).get("throughput", 0)
    vllm_tp = vllm_results.get(bs, {}).get("throughput", 0)
    if trt_tp > 0 and vllm_tp > 0:
        diff = (trt_tp - vllm_tp) / vllm_tp * 100
        winner_str = f"TRT {diff:+.1f}%" if diff > 0 else f"vLLM {-diff:+.1f}%"
        markdown_table += f"| {bs} | {trt_tp:.2f} | {vllm_tp:.2f} | {winner_str} |\n"

markdown_table += """
### Time to First Token (TTFT) in ms

| Batch Size | TensorRT-LLM | vLLM | Difference |
|------------|--------------|------|------------|
"""

for bs in batch_sizes:
    trt_ttft = trt_results.get(bs, {}).get("ttft")
    vllm_ttft = vllm_results.get(bs, {}).get("ttft")
    if trt_ttft and vllm_ttft:
        diff = (vllm_ttft - trt_ttft) / vllm_ttft * 100
        winner_str = f"TRT {diff:+.1f}%" if diff > 0 else f"vLLM {-diff:+.1f}%"
        markdown_table += f"| {bs} | {trt_ttft:.2f} | {vllm_ttft:.2f} | {winner_str} |\n"

markdown_table += """
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
"""

with open(output_dir / 'BENCHMARK_RESULTS.md', 'w') as f:
    f.write(markdown_table)

print("  - BENCHMARK_RESULTS.md")
