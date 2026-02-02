#!/usr/bin/env python3
"""
Analyze benchmark results and generate comparison charts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(filepath: str) -> pd.DataFrame:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    return pd.DataFrame(data["results"])


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot throughput comparison across engines and batch sizes."""
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=df,
        x="batch_size",
        y="throughput_tokens_per_sec",
        hue="engine",
        palette="viridis"
    )
    
    plt.title("Throughput Comparison: Tokens per Second", fontsize=14)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Throughput (tokens/sec)", fontsize=12)
    plt.legend(title="Engine")
    plt.tight_layout()
    
    plt.savefig(output_dir / "throughput_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'throughput_comparison.png'}")


def plot_latency_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot latency comparison across engines and batch sizes."""
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar plot with error bars
    engines = df["engine"].unique()
    batch_sizes = sorted(df["batch_size"].unique())
    x = range(len(batch_sizes))
    width = 0.25
    
    for i, engine in enumerate(engines):
        engine_data = df[df["engine"] == engine]
        means = [engine_data[engine_data["batch_size"] == bs]["avg_latency_ms"].values[0] 
                 if bs in engine_data["batch_size"].values else 0 
                 for bs in batch_sizes]
        stds = [engine_data[engine_data["batch_size"] == bs]["std_latency_ms"].values[0] 
                if bs in engine_data["batch_size"].values else 0 
                for bs in batch_sizes]
        
        offset = (i - len(engines)/2 + 0.5) * width
        plt.bar([xi + offset for xi in x], means, width, 
                yerr=stds, label=engine, capsize=3)
    
    plt.title("Latency Comparison", fontsize=14)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.xticks(x, batch_sizes)
    plt.legend(title="Engine")
    plt.tight_layout()
    
    plt.savefig(output_dir / "latency_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'latency_comparison.png'}")


def plot_scaling_efficiency(df: pd.DataFrame, output_dir: Path):
    """Plot how throughput scales with batch size."""
    plt.figure(figsize=(10, 6))
    
    for engine in df["engine"].unique():
        engine_data = df[df["engine"] == engine].sort_values("batch_size")
        plt.plot(
            engine_data["batch_size"],
            engine_data["throughput_tokens_per_sec"],
            marker='o',
            linewidth=2,
            markersize=8,
            label=engine
        )
    
    plt.title("Throughput Scaling with Batch Size", fontsize=14)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Throughput (tokens/sec)", fontsize=12)
    plt.legend(title="Engine")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / "scaling_efficiency.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'scaling_efficiency.png'}")


def plot_memory_usage(df: pd.DataFrame, output_dir: Path):
    """Plot GPU memory usage comparison."""
    if "gpu_memory_mb" not in df.columns or df["gpu_memory_mb"].isna().all():
        print("No GPU memory data available, skipping memory plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=df,
        x="batch_size",
        y="gpu_memory_mb",
        hue="engine",
        palette="magma"
    )
    
    plt.title("GPU Memory Usage Comparison", fontsize=14)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("GPU Memory (MB)", fontsize=12)
    plt.legend(title="Engine")
    plt.tight_layout()
    
    plt.savefig(output_dir / "memory_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'memory_comparison.png'}")


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate a summary markdown table."""
    summary = df.pivot_table(
        index="batch_size",
        columns="engine",
        values=["avg_latency_ms", "throughput_tokens_per_sec"],
        aggfunc="mean"
    )
    
    # Create markdown table
    md_content = "# Benchmark Results Summary\n\n"
    md_content += "## Throughput (tokens/sec)\n\n"
    md_content += "| Batch Size |"
    
    engines = df["engine"].unique()
    for engine in engines:
        md_content += f" {engine} |"
    md_content += "\n|" + "---|" * (len(engines) + 1) + "\n"
    
    for batch_size in sorted(df["batch_size"].unique()):
        md_content += f"| {batch_size} |"
        for engine in engines:
            val = df[(df["batch_size"] == batch_size) & (df["engine"] == engine)]["throughput_tokens_per_sec"]
            if len(val) > 0:
                md_content += f" {val.values[0]:.2f} |"
            else:
                md_content += " N/A |"
        md_content += "\n"
    
    md_content += "\n## Latency (ms)\n\n"
    md_content += "| Batch Size |"
    for engine in engines:
        md_content += f" {engine} |"
    md_content += "\n|" + "---|" * (len(engines) + 1) + "\n"
    
    for batch_size in sorted(df["batch_size"].unique()):
        md_content += f"| {batch_size} |"
        for engine in engines:
            val = df[(df["batch_size"] == batch_size) & (df["engine"] == engine)]["avg_latency_ms"]
            std = df[(df["batch_size"] == batch_size) & (df["engine"] == engine)]["std_latency_ms"]
            if len(val) > 0:
                md_content += f" {val.values[0]:.2f} ± {std.values[0]:.2f} |"
            else:
                md_content += " N/A |"
        md_content += "\n"
    
    with open(output_dir / "summary.md", "w") as f:
        f.write(md_content)
    
    print(f"Saved: {output_dir / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to benchmark results JSON file")
    parser.add_argument("--output_dir", type=str, default="./analysis",
                        help="Output directory for charts and tables")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {args.input}")
    df = load_results(args.input)
    
    print(f"Found {len(df)} benchmark results")
    print(f"Engines: {df['engine'].unique().tolist()}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique().tolist())}")
    
    print("\nGenerating charts...")
    plot_throughput_comparison(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_scaling_efficiency(df, output_dir)
    plot_memory_usage(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
