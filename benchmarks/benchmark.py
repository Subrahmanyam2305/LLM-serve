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
    concurrency: int
    num_runs: int
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_tokens_per_sec: float
    avg_tokens_per_request: float
    total_tokens_generated: int
    gpu_memory_mb: Optional[float] = None
    ttft_ms: Optional[float] = None  # Time to first token
    itl_ms: Optional[float] = None   # Inter-token latency


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
    """Get current GPU memory usage in MB using nvidia-smi."""
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
    
    # Fallback to PyTorch method
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def get_gpu_memory_allocated_mb() -> float:
    """Get GPU memory allocated by PyTorch in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def generate_prompts(batch_size: int, base_prompts: List[str]) -> List[str]:
    """Generate prompts for a given batch size."""
    prompts = []
    for i in range(batch_size):
        prompts.append(base_prompts[i % len(base_prompts)])
    return prompts


@dataclass
class StreamingMetrics:
    """Metrics from a streaming inference run."""
    total_latency_ms: float
    ttft_ms: float  # Time to first token
    itl_ms: float   # Inter-token latency (average)
    tokens_generated: int


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
    
    def run_streaming(self, prompts: List[str], max_tokens: int) -> tuple:
        """Run inference with streaming to measure TTFT and ITL."""
        # Tokenize
        batch_input_ids = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            batch_input_ids.append(torch.tensor(ids, dtype=torch.int32))
        
        input_lengths = [len(ids) for ids in batch_input_ids]
        
        start_time = time.perf_counter()
        first_token_time = None
        token_times = []
        final_outputs = None
        
        with torch.no_grad():
            outputs_generator = self.runner.generate(
                batch_input_ids=batch_input_ids,
                max_new_tokens=max_tokens,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=1.0,
                top_k=1,
                output_sequence_lengths=True,
                return_dict=True,
                streaming=True,
            )
            
            for outputs in outputs_generator:
                current_time = time.perf_counter()
                if first_token_time is None:
                    first_token_time = current_time
                token_times.append(current_time)
                final_outputs = outputs
            
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_latency = (end_time - start_time) * 1000
        ttft = (first_token_time - start_time) * 1000 if first_token_time else total_latency
        
        # Count tokens
        total_tokens = 0
        if final_outputs:
            seq_lengths = final_outputs['sequence_lengths']
            for i, input_len in enumerate(input_lengths):
                total_tokens += seq_lengths[i][0].item() - input_len
        
        # Calculate ITL
        if len(token_times) > 1:
            itl = (token_times[-1] - token_times[0]) * 1000 / max(len(token_times) - 1, 1)
        else:
            itl = 0
        
        metrics = StreamingMetrics(
            total_latency_ms=total_latency,
            ttft_ms=ttft,
            itl_ms=itl,
            tokens_generated=total_tokens,
        )
        
        return final_outputs, total_latency / 1000, total_tokens, metrics
    
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
            max_model_len=4608,  # Match TensorRT-LLM max_seq_len for fair comparison (4096 + 512)
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
    
    def run_streaming(self, prompts: List[str], max_tokens: int) -> tuple:
        """Run inference with streaming to measure TTFT and ITL.
        
        Note: vLLM's offline LLM class doesn't support true token-by-token streaming.
        For accurate TTFT/ITL, use the vLLM server with streaming API.
        This implementation measures batch-level timing as an approximation.
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # Greedy for reproducibility
        )
        
        # For single prompt, we can get approximate TTFT by measuring
        # time to first output in the batch
        start_time = time.perf_counter()
        
        # Generate all outputs
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_latency = (end_time - start_time) * 1000  # ms
        
        # Count tokens
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        
        # Approximate TTFT: For vLLM offline mode, we estimate based on
        # prefill time which is roughly proportional to input length
        # A rough estimate: TTFT ≈ total_latency * (avg_input_tokens / (avg_input_tokens + avg_output_tokens))
        avg_output_tokens = total_tokens / len(prompts) if prompts else 0
        
        # Get input token counts
        input_tokens = []
        for prompt in prompts:
            # Use tokenizer if available, otherwise estimate
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                input_tokens.append(len(tokenizer.encode(prompt)))
            except:
                input_tokens.append(len(prompt.split()) * 1.3)  # Rough estimate
        
        avg_input_tokens = sum(input_tokens) / len(input_tokens) if input_tokens else 100
        
        # Estimate TTFT as prefill portion of total time
        # This is an approximation - for accurate TTFT, use server-based streaming
        ttft_ratio = avg_input_tokens / (avg_input_tokens + avg_output_tokens) if avg_output_tokens > 0 else 0.1
        ttft = total_latency * ttft_ratio
        
        # ITL: (total_time - ttft) / num_output_tokens
        decode_time = total_latency - ttft
        itl = decode_time / avg_output_tokens if avg_output_tokens > 0 else 0
        
        metrics = StreamingMetrics(
            total_latency_ms=total_latency,
            ttft_ms=ttft,
            itl_ms=itl,
            tokens_generated=total_tokens,
        )
        
        return outputs, total_latency / 1000, total_tokens, metrics
    
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
        self.tokenizer = None
        
    def setup(self):
        """Initialize SGLang runtime with triton backend to avoid nvcc dependency."""
        import sglang as sgl
        from transformers import AutoTokenizer
        
        # Use triton backend to avoid flashinfer nvcc compilation issues
        self.runtime = sgl.Runtime(
            model_path=self.model_path,
            dtype=self.dtype,
            trust_remote_code=True,
            attention_backend="triton",
            sampling_backend="pytorch",
            disable_cuda_graph=True,
        )
        sgl.set_default_backend(self.runtime)
        
        # Load tokenizer for accurate token counting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
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
        
        # Use tokenizer for accurate token count
        total_tokens = sum(
            len(self.tokenizer.encode(s["response"])) for s in states
        )
        return states, elapsed, total_tokens
    
    def run_streaming(self, prompts: List[str], max_tokens: int) -> tuple:
        """Run inference with streaming to measure TTFT and ITL."""
        import sglang as sgl
        
        # SGLang streaming via Engine.generate with stream=True
        start_time = time.perf_counter()
        first_token_time = None
        token_times = []
        all_responses = []
        
        # Process each prompt individually for streaming
        for prompt in prompts:
            prompt_start = time.perf_counter()
            prompt_first_token = None
            response_text = ""
            
            # Use the runtime's generate method with streaming
            try:
                for chunk in self.runtime.generate(
                    prompt,
                    sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0},
                    stream=True,
                ):
                    current_time = time.perf_counter()
                    if prompt_first_token is None:
                        prompt_first_token = current_time
                        if first_token_time is None:
                            first_token_time = current_time
                    token_times.append(current_time)
                    response_text = chunk.get("text", response_text)
                
                all_responses.append(response_text)
            except Exception:
                # Fallback to non-streaming if streaming not supported
                @sgl.function
                def gen(s, prompt, max_tokens):
                    s += prompt
                    s += sgl.gen("response", max_tokens=max_tokens)
                
                state = gen.run(prompt=prompt, max_tokens=max_tokens)
                all_responses.append(state["response"])
                if first_token_time is None:
                    first_token_time = time.perf_counter()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_latency = (end_time - start_time) * 1000
        ttft = (first_token_time - start_time) * 1000 if first_token_time else total_latency
        
        # Count tokens
        total_tokens = sum(
            len(self.tokenizer.encode(resp)) for resp in all_responses
        )
        
        # Calculate ITL
        if len(token_times) > 1:
            itl = (token_times[-1] - token_times[0]) * 1000 / max(len(token_times) - 1, 1)
        else:
            itl = 0
        
        metrics = StreamingMetrics(
            total_latency_ms=total_latency,
            ttft_ms=ttft,
            itl_ms=itl,
            tokens_generated=total_tokens,
        )
        
        return all_responses, total_latency / 1000, total_tokens, metrics
    
    def cleanup(self):
        """Cleanup resources."""
        if self.runtime:
            self.runtime.shutdown()
        torch.cuda.empty_cache()


def run_benchmark(
    benchmark_cls,
    config: BenchmarkConfig,
    engine_name: str,
    streaming: bool = False,
    **init_kwargs
) -> List[BenchmarkResult]:
    """Run benchmarks for a single engine across all batch sizes."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name}" + (" (streaming)" if streaming else ""))
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
                if streaming and hasattr(benchmark, 'run_streaming'):
                    benchmark.run_streaming(prompts, config.max_output_tokens)
                else:
                    benchmark.run(prompts, config.max_output_tokens)
            except Exception as e:
                print(f"  ERROR during warmup: {e}")
                break
        
        # Benchmark runs
        print(f"  Running benchmark ({config.num_runs} runs)...")
        latencies = []
        total_tokens = 0
        ttfts = []
        itls = []
        
        torch.cuda.reset_peak_memory_stats()
        
        for i in range(config.num_runs):
            try:
                if streaming and hasattr(benchmark, 'run_streaming'):
                    _, elapsed, tokens, metrics = benchmark.run_streaming(
                        prompts, config.max_output_tokens
                    )
                    latencies.append(elapsed * 1000)  # Convert to ms
                    total_tokens += tokens
                    ttfts.append(metrics.ttft_ms)
                    itls.append(metrics.itl_ms)
                else:
                    _, elapsed, tokens = benchmark.run(prompts, config.max_output_tokens)
                    latencies.append(elapsed * 1000)  # Convert to ms
                    total_tokens += tokens
            except Exception as e:
                print(f"  ERROR on run {i}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        if latencies:
            gpu_mem = get_gpu_memory_mb()
            
            # Calculate average TTFT and ITL if streaming
            avg_ttft = statistics.mean(ttfts) if ttfts else None
            avg_itl = statistics.mean(itls) if itls else None
            
            result = BenchmarkResult(
                engine=engine_name,
                concurrency=batch_size,
                num_runs=len(latencies),
                avg_latency_ms=statistics.mean(latencies),
                std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                throughput_tokens_per_sec=total_tokens / (sum(latencies) / 1000),
                avg_tokens_per_request=total_tokens / (len(latencies) * batch_size),
                total_tokens_generated=total_tokens,
                gpu_memory_mb=gpu_mem,
                ttft_ms=avg_ttft,
                itl_ms=avg_itl,
            )
            results.append(result)
            
            print(f"  Results:")
            print(f"    Avg latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
            print(f"    Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
            if avg_ttft is not None:
                print(f"    TTFT: {avg_ttft:.2f} ms")
            if avg_itl is not None:
                print(f"    ITL: {avg_itl:.2f} ms")
            print(f"    GPU memory: {result.gpu_memory_mb:.0f} MB")
        
        benchmark.cleanup()
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference engines")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model")
    parser.add_argument("--trt_engine_dir", type=str, default=None,
                        help="Path to TensorRT engine directory")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="Batch sizes to benchmark")
    parser.add_argument("--max_output_tokens", type=int, default=512,
                        help="Maximum output tokens per request")
    parser.add_argument("--num_warmup", type=int, default=2,
                        help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of benchmark runs")
    parser.add_argument("--engines", type=str, nargs="+", 
                        default=["tensorrt", "vllm", "sglang"],
                        choices=["tensorrt", "vllm", "sglang"],
                        help="Engines to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json", 
                        help="Output file for results")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="JSON file with test prompts")
    # ShareGPT options
    parser.add_argument("--sharegpt", action="store_true",
                        help="Use ShareGPT dataset for prompts")
    parser.add_argument("--sharegpt_json", type=str, default=None,
                        help="Path to local ShareGPT JSON file")
    parser.add_argument("--sharegpt_split", type=str, default="train",
                        help="ShareGPT dataset split to use")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Maximum number of prompts to use (for subset testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt sampling")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode to measure TTFT and ITL")
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
    
    # Load prompts based on source
    if args.sharegpt:
        # Load from ShareGPT dataset
        from load_sharegpt import load_sharegpt
        prompts = load_sharegpt(
            max_prompts=args.max_prompts,
            json_path=args.sharegpt_json,
            split=args.sharegpt_split,
            seed=args.seed,
        )
        if not prompts:
            print("WARNING: No prompts loaded from ShareGPT, using defaults")
            prompts = default_prompts
    elif args.prompts_file and Path(args.prompts_file).exists():
        # Load from custom prompts file
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        if args.max_prompts and args.max_prompts < len(prompts):
            import random
            random.seed(args.seed)
            prompts = random.sample(prompts, args.max_prompts)
    else:
        prompts = default_prompts
    
    print(f"[Benchmark] Using {len(prompts)} prompts")
    if args.streaming:
        print(f"[Benchmark] Streaming mode enabled (measuring TTFT and ITL)")
    
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
                streaming=args.streaming,
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
                streaming=args.streaming,
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
                streaming=args.streaming,
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

    args.output = f"./analysis/{args.model}/{args.engines[0]}_server_results.json"
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nSummary:")
    if args.streaming:
        print("-" * 110)
        print(f"{'Engine':<15} {'Conc':<8} {'Latency (ms)':<20} {'Throughput (tok/s)':<18} {'TTFT (ms)':<12} {'ITL (ms)':<12}")
        print("-" * 110)
        for r in all_results:
            ttft_str = f"{r.ttft_ms:.2f}" if r.ttft_ms is not None else "N/A"
            itl_str = f"{r.itl_ms:.2f}" if r.itl_ms is not None else "N/A"
            print(f"{r.engine:<15} {r.concurrency:<8} {r.avg_latency_ms:>8.2f} ± {r.std_latency_ms:<8.2f} {r.throughput_tokens_per_sec:>16.2f} {ttft_str:>12} {itl_str:>12}")
    else:
        print("-" * 80)
        print(f"{'Engine':<15} {'Conc':<8} {'Latency (ms)':<20} {'Throughput (tok/s)':<20}")
        print("-" * 80)
        for r in all_results:
            print(f"{r.engine:<15} {r.concurrency:<8} {r.avg_latency_ms:>8.2f} ± {r.std_latency_ms:<8.2f} {r.throughput_tokens_per_sec:>18.2f}")


if __name__ == "__main__":
    main()
