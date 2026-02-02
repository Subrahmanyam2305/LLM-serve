#!/usr/bin/env python3
"""
Unified benchmark runner for comparing TensorRT-LLM, vLLM, and SGLang.
Measures throughput and latency across different batch sizes.
"""

import argparse
import json
import time
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import statistics

import torch


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    engine: str
    batch_size: int
    num_runs: int
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_tokens_per_sec: float
    avg_tokens_per_request: float
    total_tokens_generated: int
    gpu_memory_mb: Optional[float] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_path: str
    trt_engine_dir: Optional[str]
    batch_sizes: List[int]
    max_output_tokens: int
    num_warmup: int
    num_runs: int
    prompts: List[str]


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def generate_prompts(batch_size: int, base_prompts: List[str]) -> List[str]:
    """Generate prompts for a given batch size."""
    prompts = []
    for i in range(batch_size):
        prompts.append(base_prompts[i % len(base_prompts)])
    return prompts


class TensorRTBenchmark:
    """Benchmark runner for TensorRT-LLM."""
    
    def __init__(self, engine_dir: str, tokenizer_dir: str):
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.runner = None
        self.tokenizer = None
        
    def setup(self):
        """Initialize the TensorRT-LLM runner."""
        from transformers import AutoTokenizer
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunnerCpp, PYTHON_BINDINGS
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, trust_remote_code=True
        )
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.end_id = self.tokenizer.eos_token_id
        
        runner_cls = ModelRunnerCpp if PYTHON_BINDINGS else None
        if runner_cls is None:
            from tensorrt_llm.runtime import ModelRunner
            runner_cls = ModelRunner
            
        self.runner = runner_cls.from_dir(
            engine_dir=self.engine_dir,
            rank=0,
        )
        
    def run(self, prompts: List[str], max_tokens: int) -> tuple:
        """Run inference and return (outputs, latency, tokens_generated)."""
        # Tokenize
        batch_input_ids = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            batch_input_ids.append(torch.tensor(ids, dtype=torch.int32))
        
        input_lengths = [len(ids) for ids in batch_input_ids]
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids,
                max_new_tokens=max_tokens,
                end_id=self.end_id,
                pad_id=self.pad_id,
                output_sequence_lengths=True,
                return_dict=True,
            )
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Count tokens
        total_tokens = 0
        seq_lengths = outputs['sequence_lengths']
        for i, input_len in enumerate(input_lengths):
            total_tokens += seq_lengths[i][0].item() - input_len
            
        return outputs, elapsed, total_tokens
    
    def cleanup(self):
        """Cleanup resources."""
        del self.runner
        torch.cuda.empty_cache()


class VLLMBenchmark:
    """Benchmark runner for vLLM."""
    
    def __init__(self, model_path: str, dtype: str = "float16"):
        self.model_path = model_path
        self.dtype = dtype
        self.llm = None
        
    def setup(self):
        """Initialize vLLM engine."""
        from vllm import LLM
        self.llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        
    def run(self, prompts: List[str], max_tokens: int) -> tuple:
        """Run inference and return (outputs, latency, tokens_generated)."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0,
        )
        
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return outputs, elapsed, total_tokens
    
    def cleanup(self):
        """Cleanup resources."""
        del self.llm
        torch.cuda.empty_cache()


class SGLangBenchmark:
    """Benchmark runner for SGLang."""
    
    def __init__(self, model_path: str, dtype: str = "float16"):
        self.model_path = model_path
        self.dtype = dtype
        self.runtime = None
        
    def setup(self):
        """Initialize SGLang runtime."""
        import sglang as sgl
        self.runtime = sgl.Runtime(
            model_path=self.model_path,
            dtype=self.dtype,
            trust_remote_code=True,
        )
        sgl.set_default_backend(self.runtime)
        
    def run(self, prompts: List[str], max_tokens: int) -> tuple:
        """Run inference and return (outputs, latency, tokens_generated)."""
        import sglang as sgl
        
        @sgl.function
        def gen(s, prompt, max_tokens):
            s += prompt
            s += sgl.gen("response", max_tokens=max_tokens)
        
        start = time.perf_counter()
        states = gen.run_batch(
            [{"prompt": p, "max_tokens": max_tokens} for p in prompts],
            progress_bar=False
        )
        elapsed = time.perf_counter() - start
        
        # Estimate tokens (would need tokenizer for exact count)
        total_tokens = sum(len(s["response"].split()) for s in states)
        return states, elapsed, total_tokens
    
    def cleanup(self):
        """Cleanup resources."""
        if self.runtime:
            self.runtime.shutdown()
        torch.cuda.empty_cache()


def run_benchmark(
    benchmark_cls,
    config: BenchmarkConfig,
    engine_name: str,
    **init_kwargs
) -> List[BenchmarkResult]:
    """Run benchmarks for a single engine across all batch sizes."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name}")
    print(f"{'='*60}")
    
    for batch_size in config.batch_sizes:
        print(f"\n[{engine_name}] Batch size: {batch_size}")
        
        # Initialize
        benchmark = benchmark_cls(**init_kwargs)
        try:
            benchmark.setup()
        except Exception as e:
            print(f"  ERROR: Failed to setup {engine_name}: {e}")
            continue
        
        prompts = generate_prompts(batch_size, config.prompts)
        
        # Warmup
        print(f"  Warming up ({config.num_warmup} runs)...")
        for _ in range(config.num_warmup):
            try:
                benchmark.run(prompts, config.max_output_tokens)
            except Exception as e:
                print(f"  ERROR during warmup: {e}")
                break
        
        # Benchmark runs
        print(f"  Running benchmark ({config.num_runs} runs)...")
        latencies = []
        total_tokens = 0
        
        torch.cuda.reset_peak_memory_stats()
        
        for i in range(config.num_runs):
            try:
                _, elapsed, tokens = benchmark.run(prompts, config.max_output_tokens)
                latencies.append(elapsed * 1000)  # Convert to ms
                total_tokens += tokens
            except Exception as e:
                print(f"  ERROR on run {i}: {e}")
                break
        
        if latencies:
            gpu_mem = get_gpu_memory_mb()
            
            result = BenchmarkResult(
                engine=engine_name,
                batch_size=batch_size,
                num_runs=len(latencies),
                avg_latency_ms=statistics.mean(latencies),
                std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                throughput_tokens_per_sec=total_tokens / (sum(latencies) / 1000),
                avg_tokens_per_request=total_tokens / (len(latencies) * batch_size),
                total_tokens_generated=total_tokens,
                gpu_memory_mb=gpu_mem,
            )
            results.append(result)
            
            print(f"  Results:")
            print(f"    Avg latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
            print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
            print(f"    GPU memory: {result.gpu_memory_mb:.0f} MB")
        
        benchmark.cleanup()
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference engines")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model")
    parser.add_argument("--trt_engine_dir", type=str, default=None,
                        help="Path to TensorRT engine directory")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to benchmark")
    parser.add_argument("--max_output_tokens", type=int, default=128,
                        help="Maximum output tokens per request")
    parser.add_argument("--num_warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of benchmark runs")
    parser.add_argument("--engines", type=str, nargs="+", 
                        default=["tensorrt", "vllm", "sglang"],
                        choices=["tensorrt", "vllm", "sglang"],
                        help="Engines to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="JSON file with test prompts")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Default prompts
    default_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis step by step.",
        "How does a neural network learn from data?",
    ]
    
    # Load custom prompts if provided
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = default_prompts
    
    config = BenchmarkConfig(
        model_path=args.model_path,
        trt_engine_dir=args.trt_engine_dir,
        batch_sizes=args.batch_sizes,
        max_output_tokens=args.max_output_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        prompts=prompts,
    )
    
    all_results = []
    
    # Run TensorRT-LLM benchmark
    if "tensorrt" in args.engines:
        if args.trt_engine_dir:
            results = run_benchmark(
                TensorRTBenchmark,
                config,
                "TensorRT-LLM",
                engine_dir=args.trt_engine_dir,
                tokenizer_dir=args.model_path,
            )
            all_results.extend(results)
        else:
            print("\nSkipping TensorRT-LLM: --trt_engine_dir not provided")
    
    # Run vLLM benchmark
    if "vllm" in args.engines:
        try:
            results = run_benchmark(
                VLLMBenchmark,
                config,
                "vLLM",
                model_path=args.model_path,
            )
            all_results.extend(results)
        except ImportError:
            print("\nSkipping vLLM: not installed (pip install vllm)")
    
    # Run SGLang benchmark
    if "sglang" in args.engines:
        try:
            results = run_benchmark(
                SGLangBenchmark,
                config,
                "SGLang",
                model_path=args.model_path,
            )
            all_results.extend(results)
        except ImportError:
            print("\nSkipping SGLang: not installed (pip install sglang[all])")
    
    # Save results
    output_data = {
        "config": asdict(config) if hasattr(config, '__dict__') else {
            "model_path": config.model_path,
            "batch_sizes": config.batch_sizes,
            "max_output_tokens": config.max_output_tokens,
            "num_runs": config.num_runs,
        },
        "results": [asdict(r) for r in all_results],
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Engine':<15} {'Batch':<8} {'Latency (ms)':<20} {'Throughput (tok/s)':<20}")
    print("-" * 80)
    for r in all_results:
        print(f"{r.engine:<15} {r.batch_size:<8} {r.avg_latency_ms:>8.2f} ± {r.std_latency_ms:<8.2f} {r.throughput_tokens_per_sec:>18.2f}")


if __name__ == "__main__":
    main()
