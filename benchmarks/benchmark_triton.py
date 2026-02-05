#!/usr/bin/env python3
"""
Benchmark runner for TensorRT-LLM via Triton Inference Server.
Compares Triton-served TensorRT-LLM (with inflight batching) against vLLM.

This benchmark uses the Triton gRPC client to send concurrent requests,
allowing Triton's inflight batcher to dynamically batch requests for
optimal GPU utilization - similar to how vLLM handles concurrent requests.
"""

import argparse
import asyncio
import json
import sys
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from triton_client.triton_grpc_client import TritonLLMClient, GenerationResult


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    engine: str
    concurrency: int  # Number of concurrent requests
    num_requests: int
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_tokens_per_sec: float
    avg_tokens_per_request: float
    total_tokens_generated: int
    gpu_memory_mb: Optional[float] = None
    ttft_ms: Optional[float] = None  # Time to first token (average)
    itl_ms: Optional[float] = None   # Inter-token latency


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    triton_url: str
    tokenizer_dir: str
    model_name: str
    concurrency_levels: List[int]
    max_output_tokens: int
    num_warmup: int
    num_runs: int
    prompts: List[str]


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


class TritonBenchmark:
    """Benchmark runner for TensorRT-LLM via Triton Server."""
    
    def __init__(self, url: str, tokenizer_dir: str, model_name: str = "tensorrt_llm"):
        self.url = url
        self.tokenizer_dir = tokenizer_dir
        self.model_name = model_name
        self.client = None
        
    def setup(self):
        """Initialize the Triton client."""
        self.client = TritonLLMClient(
            url=self.url,
            tokenizer_dir=self.tokenizer_dir,
            model_name=self.model_name,
            verbose=False,
        )
        
        if not self.client.is_server_ready():
            raise RuntimeError(f"Triton server at {self.url} is not ready")
        if not self.client.is_model_ready():
            raise RuntimeError(f"Model {self.model_name} is not ready on Triton server")
        
        print(f"Connected to Triton server at {self.url}")
        print(f"Model {self.model_name} is ready")
    
    def run_single(self, prompt: str, max_tokens: int) -> GenerationResult:
        """Run a single inference request."""
        return self.client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,  # Greedy for reproducibility
            streaming=True,   # Required for Triton TensorRT-LLM
        )
    
    def run_concurrent(
        self, 
        prompts: List[str], 
        max_tokens: int,
        concurrency: int
    ) -> Tuple[List[GenerationResult], float]:
        """
        Run concurrent inference requests.
        
        This simulates real-world load where multiple requests arrive simultaneously.
        Triton's inflight batcher will dynamically batch these requests.
        
        Args:
            prompts: List of prompts to process
            max_tokens: Maximum tokens to generate per request
            concurrency: Number of concurrent requests to send
            
        Returns:
            Tuple of (results list, total elapsed time in seconds)
        """
        results = [None] * len(prompts)
        errors = []
        
        # Create a pool of clients (one per concurrent request)
        clients = []
        for _ in range(concurrency):
            client = TritonLLMClient(
                url=self.url,
                tokenizer_dir=None,  # Share tokenizer
                model_name=self.model_name,
                verbose=False,
            )
            client.tokenizer = self.client.tokenizer
            client.pad_id = self.client.pad_id
            client.end_id = self.client.end_id
            clients.append(client)
        
        def process_prompt(args):
            idx, prompt, client_idx = args
            try:
                result = clients[client_idx % len(clients)].generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    streaming=True,
                )
                return idx, result
            except Exception as e:
                return idx, GenerationResult(
                    request_id=f"error_{idx}",
                    input_text=prompt,
                    output_text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    error=str(e),
                )
        
        # Process prompts with thread pool
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all prompts
            futures = []
            for i, prompt in enumerate(prompts):
                futures.append(executor.submit(process_prompt, (i, prompt, i)))
            
            # Collect results
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                if result.error:
                    errors.append(result.error)
        
        elapsed = time.perf_counter() - start_time
        
        if errors:
            print(f"  WARNING: {len(errors)} requests failed")
        
        return results, elapsed
    
    def cleanup(self):
        """Cleanup resources."""
        self.client = None


class VLLMServerBenchmark:
    """Benchmark runner for vLLM via OpenAI-compatible API."""
    
    def __init__(self, api_base: str, model_name: str, tokenizer_dir: str):
        self.api_base = api_base
        self.model_name = model_name
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = None
        
    def setup(self):
        """Initialize the client."""
        from transformers import AutoTokenizer
        import requests
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, trust_remote_code=True
        )
        
        # Check if server is ready
        try:
            resp = requests.get(f"{self.api_base}/v1/models", timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"vLLM server not ready: {resp.status_code}")
            print(f"Connected to vLLM server at {self.api_base}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to vLLM server: {e}")
    
    def run_concurrent(
        self,
        prompts: List[str],
        max_tokens: int,
        concurrency: int
    ) -> Tuple[List[GenerationResult], float]:
        """Run concurrent requests to vLLM server with TTFT measurement."""
        import requests
        
        results = [None] * len(prompts)
        
        def process_prompt(args):
            idx, prompt = args
            start = time.perf_counter()
            ttft = None
            output_text = ""
            output_tokens = 0
            input_tokens = 0
            
            try:
                resp = requests.post(
                    f"{self.api_base}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "stream": True,  # Enable streaming to measure TTFT
                    },
                    timeout=120,
                    stream=True,
                )
                
                if resp.status_code == 200:
                    first_chunk = True
                    for line in resp.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data_str = line[6:]  # Remove 'data: ' prefix
                                if data_str.strip() == '[DONE]':
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    
                                    # Measure TTFT on first chunk (even if no content yet)
                                    if first_chunk:
                                        ttft = (time.perf_counter() - start) * 1000
                                        first_chunk = False
                                    
                                    # Accumulate output
                                    if data.get("choices") and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            output_text += content
                                        
                                        # Get token counts from usage if available
                                        usage = data.get("usage")
                                        if usage:
                                            input_tokens = usage.get("prompt_tokens", 0)
                                            output_tokens = usage.get("completion_tokens", 0)
                                except json.JSONDecodeError:
                                    continue
                    
                    elapsed = (time.perf_counter() - start) * 1000
                    
                    # If token counts weren't provided, estimate them
                    if input_tokens == 0:
                        input_tokens = len(self.tokenizer.encode(prompt))
                    if output_tokens == 0 and output_text:
                        output_tokens = len(self.tokenizer.encode(output_text))
                    
                    return idx, GenerationResult(
                        request_id=f"req_{idx}",
                        input_text=prompt,
                        output_text=output_text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=elapsed,
                        ttft_ms=ttft,
                    )
                else:
                    elapsed = (time.perf_counter() - start) * 1000
                    return idx, GenerationResult(
                        request_id=f"error_{idx}",
                        input_text=prompt,
                        output_text="",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=elapsed,
                        error=f"HTTP {resp.status_code}: {resp.text}",
                    )
            except Exception as e:
                return idx, GenerationResult(
                    request_id=f"error_{idx}",
                    input_text=prompt,
                    output_text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    error=str(e),
                )
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_prompt, (i, p)) for i, p in enumerate(prompts)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        elapsed = time.perf_counter() - start_time
        return results, elapsed
    
    def cleanup(self):
        """Cleanup resources."""
        pass


def run_triton_benchmark(
    config: BenchmarkConfig,
    streaming: bool = True,
) -> List[BenchmarkResult]:
    """Run benchmarks for Triton TensorRT-LLM."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: TensorRT-LLM via Triton (Inflight Batching)")
    print(f"{'='*60}")
    
    benchmark = TritonBenchmark(
        url=config.triton_url,
        tokenizer_dir=config.tokenizer_dir,
        model_name=config.model_name,
    )
    
    try:
        benchmark.setup()
    except Exception as e:
        print(f"ERROR: Failed to setup Triton benchmark: {e}")
        return results
    
    for concurrency in config.concurrency_levels:
        print(f"\n[Triton TRT-LLM] Concurrency: {concurrency}")
        
        # Use concurrency number of prompts per run (like batch_size in original)
        prompts = config.prompts[:concurrency]
        if len(prompts) < concurrency:
            # Repeat prompts if we don't have enough
            prompts = (prompts * (concurrency // len(prompts) + 1))[:concurrency]
        
        # Warmup
        print(f"  Warming up ({config.num_warmup} runs)...")
        for _ in range(config.num_warmup):
            try:
                benchmark.run_concurrent(prompts, config.max_output_tokens, concurrency)
            except Exception as e:
                print(f"  ERROR during warmup: {e}")
                break
        
        # Benchmark runs
        print(f"  Running benchmark ({config.num_runs} runs, {concurrency} concurrent requests each)...")
        all_latencies = []
        all_ttfts = []
        total_tokens = 0
        total_time = 0
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        for run_idx in range(config.num_runs):
            try:
                run_results, elapsed = benchmark.run_concurrent(
                    prompts, config.max_output_tokens, concurrency
                )
                
                total_time += elapsed
                
                for r in run_results:
                    if r and not r.error:
                        all_latencies.append(r.latency_ms)
                        if r.ttft_ms:
                            all_ttfts.append(r.ttft_ms)
                        total_tokens += r.output_tokens
                        
            except Exception as e:
                print(f"  ERROR on run {run_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        if all_latencies:
            gpu_mem = get_gpu_memory_mb()
            
            result = BenchmarkResult(
                engine="Triton-TRT-LLM",
                concurrency=concurrency,
                num_requests=len(all_latencies),
                avg_latency_ms=statistics.mean(all_latencies),
                std_latency_ms=statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
                min_latency_ms=min(all_latencies),
                max_latency_ms=max(all_latencies),
                throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
                avg_tokens_per_request=total_tokens / len(all_latencies),
                total_tokens_generated=total_tokens,
                gpu_memory_mb=gpu_mem,
                ttft_ms=statistics.mean(all_ttfts) if all_ttfts else None,
            )
            results.append(result)
            
            print(f"  Results:")
            print(f"    Avg latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
            print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
            if result.ttft_ms:
                print(f"    Avg TTFT: {result.ttft_ms:.2f} ms")
            print(f"    GPU memory: {result.gpu_memory_mb:.0f} MB")
    
    benchmark.cleanup()
    return results


def run_vllm_server_benchmark(
    config: BenchmarkConfig,
    api_base: str,
    model_name: str,
) -> List[BenchmarkResult]:
    """Run benchmarks for vLLM server."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: vLLM Server (Continuous Batching)")
    print(f"{'='*60}")
    
    benchmark = VLLMServerBenchmark(
        api_base=api_base,
        model_name=model_name,
        tokenizer_dir=config.tokenizer_dir,
    )
    
    try:
        benchmark.setup()
    except Exception as e:
        print(f"ERROR: Failed to setup vLLM benchmark: {e}")
        return results
    
    for concurrency in config.concurrency_levels:
        print(f"\n[vLLM Server] Concurrency: {concurrency}")
        
        # Use concurrency number of prompts per run (like batch_size in original)
        prompts = config.prompts[:concurrency]
        if len(prompts) < concurrency:
            prompts = (prompts * (concurrency // len(prompts) + 1))[:concurrency]
        
        # Warmup
        print(f"  Warming up ({config.num_warmup} runs)...")
        for _ in range(config.num_warmup):
            try:
                benchmark.run_concurrent(prompts, config.max_output_tokens, concurrency)
            except Exception as e:
                print(f"  ERROR during warmup: {e}")
                break
        
        # Benchmark runs
        print(f"  Running benchmark ({config.num_runs} runs, {concurrency} concurrent requests each)...")
        all_latencies = []
        all_ttfts = []
        total_tokens = 0
        total_time = 0
        
        for run_idx in range(config.num_runs):
            try:
                run_results, elapsed = benchmark.run_concurrent(
                    prompts, config.max_output_tokens, concurrency
                )
                
                total_time += elapsed
                
                for r in run_results:
                    if r and not r.error:
                        all_latencies.append(r.latency_ms)
                        total_tokens += r.output_tokens
                        if r.ttft_ms is not None:
                            all_ttfts.append(r.ttft_ms)
                        
            except Exception as e:
                print(f"  ERROR on run {run_idx}: {e}")
        
        if all_latencies:
            gpu_mem = get_gpu_memory_mb()
            
            # Calculate average TTFT if available
            avg_ttft = statistics.mean(all_ttfts) if all_ttfts else None
            
            result = BenchmarkResult(
                engine="vLLM-Server",
                concurrency=concurrency,
                num_requests=len(all_latencies),
                avg_latency_ms=statistics.mean(all_latencies),
                std_latency_ms=statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
                min_latency_ms=min(all_latencies),
                max_latency_ms=max(all_latencies),
                throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
                avg_tokens_per_request=total_tokens / len(all_latencies),
                total_tokens_generated=total_tokens,
                gpu_memory_mb=gpu_mem,
                ttft_ms=avg_ttft,
            )
            results.append(result)
            
            print(f"  Results:")
            print(f"    Requests completed: {result.num_requests}")
            print(f"    Avg latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
            print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
            print(f"    GPU memory: {result.gpu_memory_mb:.0f} MB")
            if avg_ttft:
                print(f"    Avg TTFT: {avg_ttft:.2f} ms")
    
    benchmark.cleanup()
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT-LLM (Triton) or vLLM with concurrent requests"
    )
    
    # Engine selection (single engine at a time due to memory constraints)
    parser.add_argument("--engine", type=str, required=True,
                        choices=["triton", "vllm"],
                        help="Engine to benchmark (triton or vllm)")
    
    # Server URL (auto-detected based on engine if not provided)
    parser.add_argument("--url", type=str, default=None,
                        help="Server URL (default: localhost:8001 for triton, http://localhost:8000 for vllm)")
    
    # Model name
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name (default: tensorrt_llm for triton, auto-detect for vllm)")
    
    # Common settings
    parser.add_argument("--tokenizer-dir", type=str, required=True,
                        help="Path to tokenizer (HuggingFace model directory)")
    parser.add_argument("--concurrency", type=int, nargs="+", 
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="Concurrency levels to test")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Maximum output tokens per request")
    parser.add_argument("--num-warmup", type=int, default=5,
                        help="Number of warmup requests")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of benchmark iterations")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (default: {engine}_server_results.json)")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for results")
    
    # Prompts
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="JSON file with test prompts")
    parser.add_argument("--sharegpt", action="store_true",
                        help="Use ShareGPT dataset for prompts")
    parser.add_argument("--max-prompts", type=int, default=100,
                        help="Maximum number of prompts to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt sampling")
    
    args = parser.parse_args()
    
    # Set defaults based on engine
    if args.url is None:
        args.url = "localhost:8001" if args.engine == "triton" else "http://localhost:8000"
    
    if args.model_name is None:
        # For vLLM, try to auto-detect from server, otherwise use a reasonable default
        if args.engine == "vllm":
            try:
                import requests
                resp = requests.get(f"{args.url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    models_data = resp.json()
                    if models_data.get("data") and len(models_data["data"]) > 0:
                        args.model_name = models_data["data"][0]["id"]
                        print(f"[Benchmark] Auto-detected model name: {args.model_name}")
                    else:
                        args.model_name = "Qwen/Qwen2.5-3B-Instruct"
                else:
                    args.model_name = "Qwen/Qwen2.5-3B-Instruct"
            except:
                args.model_name = "Qwen/Qwen2.5-3B-Instruct"
        else:
            args.model_name = "tensorrt_llm"
    
    if args.output is None:
        args.output = f"{args.engine}_server_results.json"
    
    return args


def main():
    args = parse_args()
    
    # Default prompts
    default_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis step by step.",
        "How does a neural network learn from data?",
        "What is the difference between TCP and UDP?",
        "Explain quantum computing to a 10-year-old.",
        "Write a haiku about artificial intelligence.",
    ]
    
    # Load prompts
    if args.sharegpt:
        try:
            # Try to import from LLM-serve
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM-serve" / "benchmarks"))
            from load_sharegpt import load_sharegpt
            prompts = load_sharegpt(
                max_prompts=args.max_prompts,
                seed=args.seed,
            )
            if not prompts:
                print("WARNING: No prompts loaded from ShareGPT, using defaults")
                prompts = default_prompts
        except ImportError:
            print("WARNING: Could not import load_sharegpt, using defaults")
            prompts = default_prompts
    elif args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        if args.max_prompts and args.max_prompts < len(prompts):
            import random
            random.seed(args.seed)
            prompts = random.sample(prompts, args.max_prompts)
    else:
        prompts = default_prompts * (args.max_prompts // len(default_prompts) + 1)
        prompts = prompts[:args.max_prompts]
    
    print(f"[Benchmark] Using {len(prompts)} prompts")
    print(f"[Benchmark] Concurrency levels: {args.concurrency}")
    
    config = BenchmarkConfig(
        triton_url=args.url,
        tokenizer_dir=args.tokenizer_dir,
        model_name=args.model_name,
        concurrency_levels=args.concurrency,
        max_output_tokens=args.max_output_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        prompts=prompts,
    )
    
    all_results = []
    
    # Run benchmark for selected engine
    if args.engine == "triton":
        results = run_triton_benchmark(config)
        all_results.extend(results)
    elif args.engine == "vllm":
        results = run_vllm_server_benchmark(
            config,
            api_base=args.url,
            model_name=args.model_name,
        )
        all_results.extend(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": {
            "engine": args.engine,
            "url": args.url,
            "model_name": args.model_name,
            "tokenizer_dir": config.tokenizer_dir,
            "concurrency_levels": config.concurrency_levels,
            "max_output_tokens": config.max_output_tokens,
            "num_runs": config.num_runs,
            "num_prompts": len(config.prompts),
        },
        "results": [asdict(r) for r in all_results],
    }
    
    output_path = output_dir / args.output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nSummary:")
    print("-" * 100)
    print(f"{'Engine':<20} {'Concurrency':<12} {'Latency (ms)':<20} {'Throughput (tok/s)':<20} {'TTFT (ms)':<12}")
    print("-" * 100)
    for r in all_results:
        ttft_str = f"{r.ttft_ms:.2f}" if r.ttft_ms else "N/A"
        print(f"{r.engine:<20} {r.concurrency:<12} {r.avg_latency_ms:>8.2f} ± {r.std_latency_ms:<8.2f} {r.throughput_tokens_per_sec:>18.2f} {ttft_str:>12}")


if __name__ == "__main__":
    main()
