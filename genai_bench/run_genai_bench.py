#!/usr/bin/env python3
"""
GenAI-Bench integration for LLM-serve.
Provides a wrapper to run genai-bench against local TensorRT-LLM, vLLM, or SGLang servers.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Model configurations for genai-bench
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen2.5-3B-Instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
    },
    "llama": {
        "name": "Llama-3.2-3B-Instruct",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "gemma": {
        "name": "gemma-3-4b-it",
        "hf_id": "google/gemma-3-4b-it",
    },
    "phi": {
        "name": "Phi-4-mini-instruct",
        "hf_id": "microsoft/Phi-4-mini-instruct",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run genai-bench against local LLM servers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to benchmark",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "vllm", "sglang", "tensorrt"],
        help="API backend type",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key (use EMPTY for local servers)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-to-text",
        choices=["text-to-text", "text-to-image"],
        help="Task type",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=60,
        help="Maximum time per run in seconds",
    )
    parser.add_argument(
        "--max_requests",
        type=int,
        default=100,
        help="Maximum requests per run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate Excel report after benchmark",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate plots after benchmark",
    )
    
    return parser.parse_args()


def check_genai_bench_installed() -> bool:
    """Check if genai-bench is installed."""
    try:
        result = subprocess.run(
            ["genai-bench", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_benchmark(args) -> Optional[Path]:
    """Run genai-bench benchmark."""
    config = MODEL_CONFIGS[args.model]
    
    print(f"\n{'='*60}")
    print(f"GenAI-Bench: Benchmarking {config['name']}")
    print(f"{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"API Base: {args.api_base}")
    print(f"Concurrency: {args.concurrency}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", args.backend,
        "--api-base", args.api_base,
        "--api-key", args.api_key,
        "--api-model-name", config["hf_id"],
        "--task", args.task,
        "--max-time-per-run", str(args.max_time),
        "--max-requests-per-run", str(args.max_requests),
    ]
    
    # Add concurrency levels
    for c in args.concurrency:
        cmd.extend(["--concurrency", str(c)])
    
    print(f"Running: {' '.join(cmd[:15])}...")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Benchmark failed with code {result.returncode}")
        return None
    
    # Find the latest experiment folder
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        experiments = sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime)
        if experiments:
            return experiments[-1]
    
    return None


def generate_report(experiment_folder: Path):
    """Generate Excel report from benchmark results."""
    print(f"\n[Report] Generating Excel report from {experiment_folder}...")
    
    cmd = [
        "genai-bench", "excel",
        "--experiment-folder", str(experiment_folder),
        "--excel-name", "results",
        "--metric-percentile", "mean",
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: Report generation failed")


def generate_plots(experiments_folder: Path):
    """Generate plots from benchmark results."""
    print(f"\n[Plots] Generating plots from {experiments_folder}...")
    
    cmd = [
        "genai-bench", "plot",
        "--experiments-folder", str(experiments_folder),
        "--group-key", "traffic_scenario",
        "--preset", "2x4_default",
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: Plot generation failed")


def main():
    args = parse_args()
    
    # Check if genai-bench is installed
    if not check_genai_bench_installed():
        print("ERROR: genai-bench is not installed.")
        print("Install with: pip install genai-bench")
        sys.exit(1)
    
    # Run benchmark
    experiment_folder = run_benchmark(args)
    
    if experiment_folder:
        print(f"\n[Success] Results saved to: {experiment_folder}")
        
        # Generate report if requested
        if args.generate_report:
            generate_report(experiment_folder)
        
        # Generate plots if requested
        if args.generate_plots:
            generate_plots(Path(args.output_dir))
    else:
        print("\n[Warning] No experiment folder found")


if __name__ == "__main__":
    main()
