#!/usr/bin/env python3
"""
GenAI-Bench integration for LLM-serve.
Provides a wrapper to run genai-bench against local TensorRT-LLM, vLLM, or SGLang servers.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
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
        default=1,
        help="Maximum time per run in MINUTES (genai-bench uses minutes)",
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
        default=[1, 4, 8, 16, 32],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--traffic_scenario",
        type=str,
        nargs="+",
        default=["N(480,240)/(300,150)"],
        help="Traffic scenarios to test (e.g., 'N(480,240)/(300,150)' for normal distribution)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ec2-user/llm_host/LLM-serve/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_logs",
        action="store_true",
        default=True,
        help="Save terminal output to a log file",
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
    print(f"Traffic Scenarios: {args.traffic_scenario}")
    print(f"Max Time Per Run: {args.max_time} minutes")
    print(f"Max Requests Per Run: {args.max_requests}")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", args.backend,
        "--api-base", args.api_base,
        "--api-key", args.api_key,
        "--api-model-name", config["hf_id"],
        "--model-tokenizer", config["hf_id"],
        "--task", args.task,
        "--max-time-per-run", str(args.max_time),
        "--max-requests-per-run", str(args.max_requests),
    ]
    
    # Add concurrency levels (genai-bench expects --num-concurrency repeated for each value)
    for c in args.concurrency:
        cmd.extend(["--num-concurrency", str(c)])
    
    # Add traffic scenarios (genai-bench expects --traffic-scenario repeated for each value)
    for scenario in args.traffic_scenario:
        cmd.extend(["--traffic-scenario", scenario])
    
    print(f"Running: {' '.join(cmd)}\n")
    
    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"genai_bench_{config['name']}_{timestamp}.log"
    
    # Run benchmark and capture output
    if args.save_logs:
        print(f"Saving terminal output to: {log_file}\n")
        with open(log_file, 'w') as f:
            # Write header info
            f.write(f"GenAI-Bench Log - {config['name']}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'='*60}\n\n")
            f.flush()
            
            # Run with output going to both terminal and file
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both terminal and file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            result_code = process.returncode
    else:
        result = subprocess.run(cmd, capture_output=False)
        result_code = result.returncode
    
    if result_code != 0:
        print(f"ERROR: Benchmark failed with code {result_code}")
        return None
    
    # Find the latest experiment folder in current directory (genai-bench creates it there)
    # Look for folders matching the pattern
    cwd = Path.cwd()
    pattern = f"{args.backend}_{args.task}_{config['name']}_*"
    experiment_folders = sorted(cwd.glob(pattern), key=lambda x: x.stat().st_mtime)
    
    if experiment_folders:
        return experiment_folders[-1]
    
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
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    experiment_folder = run_benchmark(args)
    
    if experiment_folder:
        # Move experiment folder to output directory if not already there
        if experiment_folder.parent != output_dir:
            import shutil
            dest_folder = output_dir / experiment_folder.name
            if dest_folder.exists():
                shutil.rmtree(dest_folder)
            shutil.move(str(experiment_folder), str(dest_folder))
            experiment_folder = dest_folder
            print(f"\n[Moved] Results moved to: {experiment_folder}")
        else:
            print(f"\n[Success] Results saved to: {experiment_folder}")
        
        # Generate report if requested
        if args.generate_report:
            generate_report(experiment_folder)
        
        # Generate plots if requested
        if args.generate_plots:
            generate_plots(output_dir)
    else:
        print("\n[Warning] No experiment folder found")


if __name__ == "__main__":
    main()
