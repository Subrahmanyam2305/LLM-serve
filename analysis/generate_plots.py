#!/usr/bin/env python3
"""
Generate benchmark analysis plots and tables.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results data
results = {
    "TensorRT-LLM": {
        "batch_sizes": [1, 2, 4, 8, 16],
        "throughput": [69.69, 138.52, 270.47, 462.76, 881.66],
        "latency": [1836.81, 1848.10, 1892.99, 2212.83, 2322.90],
        "latency_std": [1.96, 1.21, 0.49, 2.44, 9.28],
    },
    "vLLM": {
        "batch_sizes": [1, 2, 4, 8, 16],
        "throughput": [67.85, 129.23, 218.02, 466.78, 925.38],
        "latency": [1886.50, 1980.90, 2026.46, 2044.20, 2085.84],
        "latency_std": [0.38, 0.15, 2.63, 1.81, 0.50],
    },
}

# Create output directory
output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'TensorRT-LLM': '#76B900', 'vLLM': '#FF6B6B', 'SGLang': '#4ECDC4'}

# 1. Throughput comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results["TensorRT-LLM"]["batch_sizes"]))
width = 0.35

bars1 = ax.bar(x - width/2, results["TensorRT-LLM"]["throughput"], width, 
               label='TensorRT-LLM', color=colors['TensorRT-LLM'])
bars2 = ax.bar(x + width/2, results["vLLM"]["throughput"], width, 
               label='vLLM', color=colors['vLLM'])

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax.set_title('Throughput Comparison: TensorRT-LLM vs vLLM\nQwen2.5-3B-Instruct on NVIDIA A10G', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(results["TensorRT-LLM"]["batch_sizes"])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'throughput_comparison.png', dpi=150)
plt.close()

# 2. Latency comparison
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(results["TensorRT-LLM"]["batch_sizes"], results["TensorRT-LLM"]["latency"],
            yerr=results["TensorRT-LLM"]["latency_std"], label='TensorRT-LLM',
            marker='o', capsize=5, color=colors['TensorRT-LLM'], linewidth=2, markersize=8)
ax.errorbar(results["vLLM"]["batch_sizes"], results["vLLM"]["latency"],
            yerr=results["vLLM"]["latency_std"], label='vLLM',
            marker='s', capsize=5, color=colors['vLLM'], linewidth=2, markersize=8)

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Latency Comparison: TensorRT-LLM vs vLLM\nQwen2.5-3B-Instruct on NVIDIA A10G', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'latency_comparison.png', dpi=150)
plt.close()

# 3. Scaling efficiency
fig, ax = plt.subplots(figsize=(10, 6))

batch_sizes = results["TensorRT-LLM"]["batch_sizes"]
trt_scaling = [t / results["TensorRT-LLM"]["throughput"][0] for t in results["TensorRT-LLM"]["throughput"]]
vllm_scaling = [t / results["vLLM"]["throughput"][0] for t in results["vLLM"]["throughput"]]
ideal_scaling = [b for b in batch_sizes]

ax.plot(batch_sizes, trt_scaling, marker='o', label='TensorRT-LLM', 
        color=colors['TensorRT-LLM'], linewidth=2, markersize=8)
ax.plot(batch_sizes, vllm_scaling, marker='s', label='vLLM', 
        color=colors['vLLM'], linewidth=2, markersize=8)
ax.plot(batch_sizes, ideal_scaling, '--', label='Ideal (Linear)', 
        color='gray', linewidth=1.5)

ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Scaling Factor (relative to batch=1)', fontsize=12)
ax.set_title('Scaling Efficiency: TensorRT-LLM vs vLLM\nQwen2.5-3B-Instruct on NVIDIA A10G', fontsize=14)
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
ax.bar(x - width/2, results["TensorRT-LLM"]["throughput"], width, 
       label='TensorRT-LLM', color=colors['TensorRT-LLM'])
ax.bar(x + width/2, results["vLLM"]["throughput"], width, 
       label='vLLM', color=colors['vLLM'])
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (tokens/sec)')
ax.set_title('Throughput Comparison')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Latency line chart
ax = axes[0, 1]
ax.errorbar(batch_sizes, results["TensorRT-LLM"]["latency"],
            yerr=results["TensorRT-LLM"]["latency_std"], label='TensorRT-LLM',
            marker='o', capsize=5, color=colors['TensorRT-LLM'], linewidth=2)
ax.errorbar(batch_sizes, results["vLLM"]["latency"],
            yerr=results["vLLM"]["latency_std"], label='vLLM',
            marker='s', capsize=5, color=colors['vLLM'], linewidth=2)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Latency (ms)')
ax.set_title('Latency Comparison')
ax.legend()
ax.grid(alpha=0.3)

# Scaling efficiency
ax = axes[1, 0]
ax.plot(batch_sizes, trt_scaling, marker='o', label='TensorRT-LLM', 
        color=colors['TensorRT-LLM'], linewidth=2)
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
table_data = [
    ['Metric', 'TensorRT-LLM', 'vLLM', 'Winner'],
    ['Throughput @BS=1', '69.7 tok/s', '67.9 tok/s', 'TensorRT-LLM (+2.7%)'],
    ['Throughput @BS=16', '881.7 tok/s', '925.4 tok/s', 'vLLM (+5.0%)'],
    ['Latency @BS=1', '1836.8 ms', '1886.5 ms', 'TensorRT-LLM (-2.6%)'],
    ['Latency @BS=16', '2322.9 ms', '2085.8 ms', 'vLLM (-10.2%)'],
    ['Scaling (1→16)', '12.7x', '13.6x', 'vLLM'],
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

plt.suptitle('LLM Inference Benchmark: Qwen2.5-3B-Instruct on NVIDIA A10G\nShareGPT Dataset (100 prompts), max_output_tokens=128', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generated plots:")
print("  - throughput_comparison.png")
print("  - latency_comparison.png")
print("  - scaling_efficiency.png")
print("  - benchmark_dashboard.png")

# Generate markdown summary table
markdown_table = """
# Benchmark Results Summary

## Configuration
- **Model**: Qwen2.5-3B-Instruct
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Dataset**: ShareGPT (100 prompts)
- **Max Output Tokens**: 128
- **Benchmark Runs**: 5

## Results

### Throughput (tokens/sec)

| Batch Size | TensorRT-LLM | vLLM | Difference |
|------------|--------------|------|------------|
| 1 | 69.69 | 67.85 | TRT +2.7% |
| 2 | 138.52 | 129.23 | TRT +7.2% |
| 4 | 270.47 | 218.02 | TRT +24.1% |
| 8 | 462.76 | 466.78 | vLLM +0.9% |
| 16 | 881.66 | 925.38 | vLLM +5.0% |

### Latency (ms)

| Batch Size | TensorRT-LLM | vLLM | Difference |
|------------|--------------|------|------------|
| 1 | 1836.81 ± 1.96 | 1886.50 ± 0.38 | TRT -2.6% |
| 2 | 1848.10 ± 1.21 | 1980.90 ± 0.15 | TRT -6.7% |
| 4 | 1892.99 ± 0.49 | 2026.46 ± 2.63 | TRT -6.6% |
| 8 | 2212.83 ± 2.44 | 2044.20 ± 1.81 | vLLM -7.6% |
| 16 | 2322.90 ± 9.28 | 2085.84 ± 0.50 | vLLM -10.2% |

## Key Findings

1. **Small Batch Sizes (1-4)**: TensorRT-LLM shows better throughput and lower latency
2. **Large Batch Sizes (8-16)**: vLLM shows better scaling and higher throughput
3. **Latency Consistency**: vLLM has more consistent latency (lower std deviation)
4. **Scaling Efficiency**: vLLM scales better from batch 1 to 16 (13.6x vs 12.7x)

## Notes

- SGLang benchmark failed due to CUDA compilation issues with flashinfer
- All tests used greedy decoding (temperature=0.0) for reproducibility
- Results may vary based on prompt length distribution
"""

with open(output_dir / 'BENCHMARK_RESULTS.md', 'w') as f:
    f.write(markdown_table)

print("  - BENCHMARK_RESULTS.md")
