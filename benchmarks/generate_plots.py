#!/usr/bin/env python3
"""
Generate benchmark analysis plots and tables for Triton TensorRT-LLM vs vLLM.
Reads results from JSON files and creates comparison visualizations.

This script compares server-based benchmarks where both engines handle
concurrent requests with their respective batching strategies:
- Triton TensorRT-LLM: Inflight batching
- vLLM: Continuous batching with PagedAttention

Usage:
    python generate_plots.py --output-subfolder qwen-triton
    python generate_plots.py --output-subfolder qwen-triton --results-dir ../results
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing result JSON files (default: ../results)")
    parser.add_argument("--output-subfolder", type=str, required=True,
                        help="Subfolder name under analysis/ for output (e.g., qwen-triton)")
    parser.add_argument("--triton-results", type=str, default="triton_server_results.json",
                        help="Triton results filename")
    parser.add_argument("--vllm-results", type=str, default="vllm_server_results.json",
                        help="vLLM results filename")
    return parser.parse_args()


def extract_model_name(config: dict) -> str:
    """Extract a clean model name from config."""
    model_name = config.get("model_name", "")
    
    # Handle HuggingFace format like "Qwen/Qwen2.5-3B-Instruct"
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Handle tensorrt_llm default - try tokenizer_dir
    if model_name == "tensorrt_llm" or not model_name:
        tokenizer_dir = config.get("tokenizer_dir", "")
        if tokenizer_dir:
            model_name = Path(tokenizer_dir).name
    
    return model_name if model_name else "Unknown Model"


def main():
    args = parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir.parent / "results"
    output_dir = script_dir.parent / "analysis" / args.output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load Triton TensorRT-LLM results
    triton_results = {}
    triton_config = {}
    try:
        with open(results_dir / args.triton_results) as f:
            data = json.load(f)
            triton_config = data.get("config", {})
            for r in data["results"]:
                conc = r["concurrency"]
                triton_results[conc] = {
                    "throughput": r["throughput_tokens_per_sec"],
                    "latency": r["avg_latency_ms"],
                    "latency_std": r["std_latency_ms"],
                    "ttft": r.get("ttft_ms"),
                    "num_requests": r.get("num_requests", 0),
                }
        print(f"Loaded Triton results: {list(triton_results.keys())}")
    except FileNotFoundError:
        print(f"Triton results not found at {results_dir / args.triton_results}")

    # Load vLLM server results
    vllm_results = {}
    vllm_config = {}
    try:
        with open(results_dir / args.vllm_results) as f:
            data = json.load(f)
            vllm_config = data.get("config", {})
            for r in data["results"]:
                conc = r["concurrency"]
                vllm_results[conc] = {
                    "throughput": r["throughput_tokens_per_sec"],
                    "latency": r["avg_latency_ms"],
                    "latency_std": r["std_latency_ms"],
                    "ttft": r.get("ttft_ms"),
                    "num_requests": r.get("num_requests", 0),
                }
        print(f"Loaded vLLM results: {list(vllm_results.keys())}")
    except FileNotFoundError:
        print(f"vLLM results not found at {results_dir / args.vllm_results}")

    # Extract model name from config
    model_name = extract_model_name(triton_config) or extract_model_name(vllm_config) or "Unknown Model"
    print(f"Model name: {model_name}")

    # Common concurrency levels
    concurrency_levels = sorted(set(triton_results.keys()) | set(vllm_results.keys()))
    if not concurrency_levels:
        print("No results found. Please run benchmark_triton.py first.")
        return

    print(f"Concurrency levels: {concurrency_levels}")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'Triton-TRT-LLM': '#76B900',  # NVIDIA green
        'vLLM-Server': '#FF6B6B',      # Red
    }

    # Helper for titles
    def make_title(main_title: str) -> str:
        return f"{main_title}\n{model_name} (FP16) on NVIDIA A10G"

    # 1. Throughput comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(concurrency_levels))
    width = 0.35

    triton_throughput = [triton_results.get(c, {}).get("throughput", 0) for c in concurrency_levels]
    vllm_throughput = [vllm_results.get(c, {}).get("throughput", 0) for c in concurrency_levels]

    bars1 = ax.bar(x - width/2, triton_throughput, width, 
                   label='Triton TRT-LLM (Inflight Batching)', color=colors['Triton-TRT-LLM'])
    bars2 = ax.bar(x + width/2, vllm_throughput, width, 
                   label='vLLM Server (Continuous Batching)', color=colors['vLLM-Server'])

    ax.set_xlabel('Concurrency (Concurrent Requests)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_title(make_title('Server Throughput Comparison: Triton TRT-LLM vs vLLM'), fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(concurrency_levels)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=150)
    plt.close()
    print("Generated: throughput_comparison.png")

    # 2. TTFT comparison (Time to First Token)
    fig, ax = plt.subplots(figsize=(12, 7))

    triton_ttft_data = [(c, triton_results.get(c, {}).get("ttft")) for c in concurrency_levels]
    triton_ttft_valid = [(c, t) for c, t in triton_ttft_data if t is not None and t > 0]

    vllm_ttft_data = [(c, vllm_results.get(c, {}).get("ttft")) for c in concurrency_levels]
    vllm_ttft_valid = [(c, t) for c, t in vllm_ttft_data if t is not None and t > 0]

    if triton_ttft_valid:
        ax.plot([x[0] for x in triton_ttft_valid], [x[1] for x in triton_ttft_valid], 
                marker='o', label='Triton TRT-LLM', color=colors['Triton-TRT-LLM'], 
                linewidth=2, markersize=8)
    if vllm_ttft_valid:
        ax.plot([x[0] for x in vllm_ttft_valid], [x[1] for x in vllm_ttft_valid],
                marker='s', label='vLLM Server', color=colors['vLLM-Server'], 
                linewidth=2, markersize=8)

    ax.set_xlabel('Concurrency (Concurrent Requests)', fontsize=12)
    ax.set_ylabel('Time to First Token (ms)', fontsize=12)
    ax.set_title(make_title('TTFT Comparison: Triton TRT-LLM vs vLLM'), fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=150)
    plt.close()
    print("Generated: latency_comparison.png")

    # 3. Separate TTFT plot
    triton_ttft = triton_ttft_valid
    vllm_ttft = vllm_ttft_valid

    if triton_ttft or vllm_ttft:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if triton_ttft:
            ax.plot([x[0] for x in triton_ttft], [x[1] for x in triton_ttft], 
                    marker='o', label='Triton TRT-LLM', color=colors['Triton-TRT-LLM'], 
                    linewidth=2, markersize=8)
        
        if vllm_ttft:
            ax.plot([x[0] for x in vllm_ttft], [x[1] for x in vllm_ttft], 
                    marker='s', label='vLLM Server', color=colors['vLLM-Server'], 
                    linewidth=2, markersize=8)
        
        ax.set_xlabel('Concurrency (Concurrent Requests)', fontsize=12)
        ax.set_ylabel('Time to First Token (ms)', fontsize=12)
        ax.set_title(make_title('TTFT Comparison: Triton TRT-LLM vs vLLM'), fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ttft_comparison.png', dpi=150)
        plt.close()
        print("Generated: ttft_comparison.png")

    # 4. Scaling efficiency
    fig, ax = plt.subplots(figsize=(12, 7))

    triton_scaling = []
    vllm_scaling = []
    
    if triton_throughput and triton_throughput[0] > 0:
        triton_scaling = [t / triton_throughput[0] if triton_throughput[0] > 0 else 0 for t in triton_throughput]
        ax.plot(concurrency_levels, triton_scaling, marker='o', label='Triton TRT-LLM', 
                color=colors['Triton-TRT-LLM'], linewidth=2, markersize=8)

    if vllm_throughput and vllm_throughput[0] > 0:
        vllm_scaling = [t / vllm_throughput[0] if vllm_throughput[0] > 0 else 0 for t in vllm_throughput]
        ax.plot(concurrency_levels, vllm_scaling, marker='s', label='vLLM Server', 
                color=colors['vLLM-Server'], linewidth=2, markersize=8)

    # Ideal linear scaling
    ideal_scaling = concurrency_levels
    ax.plot(concurrency_levels, ideal_scaling, '--', label='Ideal (Linear)', 
            color='gray', linewidth=1.5)

    ax.set_xlabel('Concurrency (Concurrent Requests)', fontsize=12)
    ax.set_ylabel('Scaling Factor (relative to concurrency=1)', fontsize=12)
    ax.set_title(make_title('Scaling Efficiency: Throughput vs Concurrency'), fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_efficiency.png', dpi=150)
    plt.close()
    print("Generated: scaling_efficiency.png")

    # 5. Combined dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Throughput bar chart
    ax = axes[0, 0]
    x = np.arange(len(concurrency_levels))
    width = 0.35
    ax.bar(x - width/2, triton_throughput, width, label='Triton TRT-LLM', color=colors['Triton-TRT-LLM'])
    ax.bar(x + width/2, vllm_throughput, width, label='vLLM Server', color=colors['vLLM-Server'])
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(concurrency_levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # TTFT line chart
    ax = axes[0, 1]
    if triton_ttft_valid:
        ax.plot([x[0] for x in triton_ttft_valid], [x[1] for x in triton_ttft_valid], 
                marker='o', label='Triton TRT-LLM', color=colors['Triton-TRT-LLM'], linewidth=2)
    if vllm_ttft_valid:
        ax.plot([x[0] for x in vllm_ttft_valid], [x[1] for x in vllm_ttft_valid], 
                marker='s', label='vLLM Server', color=colors['vLLM-Server'], linewidth=2)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('TTFT Comparison')
    ax.legend()
    ax.grid(alpha=0.3)

    # Scaling efficiency
    ax = axes[1, 0]
    if triton_scaling:
        ax.plot(concurrency_levels, triton_scaling, marker='o', label='Triton TRT-LLM', 
                color=colors['Triton-TRT-LLM'], linewidth=2)
    if vllm_scaling:
        ax.plot(concurrency_levels, vllm_scaling, marker='s', label='vLLM Server', 
                color=colors['vLLM-Server'], linewidth=2)
    ax.plot(concurrency_levels, ideal_scaling, '--', label='Ideal', color='gray', linewidth=1.5)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Scaling Factor')
    ax.set_title('Scaling Efficiency')
    ax.legend()
    ax.grid(alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')

    def get_metric(results, conc, metric):
        return results.get(conc, {}).get(metric, 0)

    summary_conc = [c for c in [1, 8, 32, 64] if c in concurrency_levels]
    if not summary_conc:
        summary_conc = concurrency_levels[:4]

    def winner(triton_val, vllm_val, lower_is_better=False):
        if triton_val == 0 or vllm_val == 0:
            return "N/A"
        if lower_is_better:
            diff = (vllm_val - triton_val) / vllm_val * 100
            return f"Triton ({diff:+.1f}%)" if triton_val < vllm_val else f"vLLM ({-diff:+.1f}%)"
        else:
            diff = (triton_val - vllm_val) / vllm_val * 100
            return f"Triton ({diff:+.1f}%)" if triton_val > vllm_val else f"vLLM ({-diff:+.1f}%)"

    table_data = [['Metric', 'Triton TRT-LLM', 'vLLM Server', 'Winner']]

    for conc in summary_conc:
        triton_tp = get_metric(triton_results, conc, "throughput")
        vllm_tp = get_metric(vllm_results, conc, "throughput")
        table_data.append([
            f'Throughput @C={conc}',
            f'{triton_tp:.1f} tok/s' if triton_tp else 'N/A',
            f'{vllm_tp:.1f} tok/s' if vllm_tp else 'N/A',
            winner(triton_tp, vllm_tp)
        ])

    for conc in summary_conc[:2]:
        triton_t = get_metric(triton_results, conc, "ttft")
        vllm_t = get_metric(vllm_results, conc, "ttft")
        table_data.append([
            f'TTFT @C={conc}',
            f'{triton_t:.1f} ms' if triton_t else 'N/A',
            f'{vllm_t:.1f} ms' if vllm_t else 'N/A',
            winner(triton_t, vllm_t, lower_is_better=True) if triton_t and vllm_t else 'N/A'
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.28, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for i in range(4):
        table[(0, i)].set_facecolor('#E6E6E6')
        table[(0, i)].set_text_props(weight='bold')
    ax.set_title('Summary Comparison', fontsize=12, pad=20)

    plt.suptitle(f'Server Benchmark: Triton TRT-LLM vs vLLM\n{model_name} (FP16) on NVIDIA A10G | ShareGPT Dataset', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: benchmark_dashboard.png")

    # Generate markdown summary
    markdown_table = f"""
# Server Benchmark Results: Triton TRT-LLM vs vLLM

## Configuration
- **Model**: {model_name}
- **Precision**: FP16
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Dataset**: ShareGPT prompts
- **Max Output Tokens**: {triton_config.get('max_output_tokens', vllm_config.get('max_output_tokens', 128))}
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
"""

    for conc in concurrency_levels:
        triton_tp = get_metric(triton_results, conc, "throughput")
        vllm_tp = get_metric(vllm_results, conc, "throughput")
        if triton_tp > 0 and vllm_tp > 0:
            diff = (triton_tp - vllm_tp) / vllm_tp * 100
            winner_str = f"Triton {diff:+.1f}%" if diff > 0 else f"vLLM {-diff:+.1f}%"
            markdown_table += f"| {conc} | {triton_tp:.2f} | {vllm_tp:.2f} | {winner_str} |\n"
        elif triton_tp > 0:
            markdown_table += f"| {conc} | {triton_tp:.2f} | N/A | - |\n"
        elif vllm_tp > 0:
            markdown_table += f"| {conc} | N/A | {vllm_tp:.2f} | - |\n"

    markdown_table += """
### Average Latency (ms) - End-to-End Generation Time

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
"""

    for conc in concurrency_levels:
        triton_lat = get_metric(triton_results, conc, "latency")
        vllm_lat = get_metric(vllm_results, conc, "latency")
        if triton_lat > 0 and vllm_lat > 0:
            diff = (vllm_lat - triton_lat) / vllm_lat * 100
            winner_str = f"Triton {diff:+.1f}%" if diff > 0 else f"vLLM {-diff:+.1f}%"
            markdown_table += f"| {conc} | {triton_lat:.2f} | {vllm_lat:.2f} | {winner_str} |\n"
        elif triton_lat > 0:
            markdown_table += f"| {conc} | {triton_lat:.2f} | N/A | - |\n"
        elif vllm_lat > 0:
            markdown_table += f"| {conc} | N/A | {vllm_lat:.2f} | - |\n"

    if triton_ttft or vllm_ttft:
        markdown_table += """
### Time to First Token (ms)

| Concurrency | Triton TRT-LLM | vLLM Server | Difference |
|-------------|----------------|-------------|------------|
"""
        all_conc = sorted(set([c for c, _ in triton_ttft] + [c for c, _ in vllm_ttft]))
        triton_ttft_dict = dict(triton_ttft)
        vllm_ttft_dict = dict(vllm_ttft)
        
        for conc in all_conc:
            triton_t = triton_ttft_dict.get(conc)
            vllm_t = vllm_ttft_dict.get(conc)
            
            if triton_t is not None and vllm_t is not None:
                diff = (vllm_t - triton_t) / vllm_t * 100
                winner_str = f"Triton {diff:+.1f}%" if diff > 0 else f"vLLM {-diff:+.1f}%"
                markdown_table += f"| {conc} | {triton_t:.2f} | {vllm_t:.2f} | {winner_str} |\n"
            elif triton_t is not None:
                markdown_table += f"| {conc} | {triton_t:.2f} | N/A | - |\n"
            elif vllm_t is not None:
                markdown_table += f"| {conc} | N/A | {vllm_t:.2f} | - |\n"

    markdown_table += f"""
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
python benchmark_triton.py \\
    --engine triton \\
    --tokenizer-dir /path/to/{model_name} \\
    --concurrency 1 4 8 16 32 64 \\
    --sharegpt

# Run benchmark for vLLM (after stopping Triton)
python benchmark_triton.py \\
    --engine vllm \\
    --tokenizer-dir /path/to/{model_name} \\
    --concurrency 1 4 8 16 32 64 \\
    --sharegpt

# Generate plots
python generate_plots.py --output-subfolder {args.output_subfolder}
```
"""

    with open(output_dir / 'BENCHMARK_RESULTS.md', 'w') as f:
        f.write(markdown_table)

    print("Generated: BENCHMARK_RESULTS.md")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
