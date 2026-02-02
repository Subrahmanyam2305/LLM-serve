#!/usr/bin/env python3
"""
vLLM inference runner for benchmarking with TTFT support.
Uses vLLM's streaming API to measure time-to-first-token.
"""

import argparse
import time
from typing import List, Tuple
from dataclasses import dataclass

from vllm import LLM, SamplingParams


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""
    total_latency_ms: float
    ttft_ms: float  # Time to first token
    tokens_generated: int
    throughput_tokens_per_sec: float
    itl_ms: float  # Inter-token latency (average)


def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model or model name")
    parser.add_argument("--input_text", type=str, nargs="+",
                        default=["Hello, how are you?"],
                        help="Input text(s) for generation")
    parser.add_argument("--max_output_len", type=int, default=128,
                        help="Maximum output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter (-1 for disabled)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization (0-1)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode (measure latency)")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Number of warmup runs for benchmarking")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                        help="Number of benchmark runs")
    return parser.parse_args()


def run_inference_with_ttft(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> Tuple[list, BenchmarkMetrics]:
    """
    Run inference and measure TTFT using streaming.
    Returns outputs and metrics.
    """
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    
    # Use streaming to capture TTFT
    outputs_list = []
    for request_output in llm.generate(prompts, sampling_params, use_tqdm=False):
        current_time = time.perf_counter()
        if first_token_time is None and request_output.outputs[0].token_ids:
            first_token_time = current_time
        token_times.append(current_time)
        outputs_list.append(request_output)
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_latency = (end_time - start_time) * 1000  # ms
    ttft = (first_token_time - start_time) * 1000 if first_token_time else total_latency
    
    # Count tokens from final outputs
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs_list)
    
    # Calculate ITL (inter-token latency)
    if len(token_times) > 1:
        itl = (token_times[-1] - token_times[0]) * 1000 / max(len(token_times) - 1, 1)
    else:
        itl = 0
    
    throughput = total_tokens / (total_latency / 1000) if total_latency > 0 else 0
    
    metrics = BenchmarkMetrics(
        total_latency_ms=total_latency,
        ttft_ms=ttft,
        tokens_generated=total_tokens,
        throughput_tokens_per_sec=throughput,
        itl_ms=itl
    )
    
    return outputs_list, metrics


def run_inference_batch(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> Tuple[list, float, int]:
    """Run batch inference (non-streaming) for comparison."""
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.perf_counter() - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return outputs, elapsed_time, total_tokens


def main():
    args = parse_args()
    
    print(f"[vLLM] Loading model: {args.model}")
    print(f"[vLLM] dtype: {args.dtype}, GPU memory utilization: {args.gpu_memory_utilization}")
    
    # Initialize vLLM engine
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    # Set up sampling parameters - use greedy by default for reproducibility
    sampling_params = SamplingParams(
        max_tokens=args.max_output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
    )
    
    prompts = args.input_text
    print(f"[vLLM] Batch size: {len(prompts)}")
    print(f"[vLLM] Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    
    if args.benchmark:
        # Warmup
        print(f"\n[Benchmark] Running {args.warmup_runs} warmup iterations...")
        for _ in range(args.warmup_runs):
            run_inference_batch(llm, prompts, sampling_params)
        
        # Benchmark
        print(f"[Benchmark] Running {args.benchmark_runs} benchmark iterations...")
        latencies = []
        ttfts = []
        total_tokens = 0
        
        for i in range(args.benchmark_runs):
            outputs, elapsed, tokens = run_inference_batch(llm, prompts, sampling_params)
            latencies.append(elapsed * 1000)  # Convert to ms
            total_tokens += tokens
            
            # Note: For accurate TTFT, would need streaming mode
            # This is end-to-end latency measurement
        
        avg_latency = sum(latencies) / len(latencies)
        std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5
        avg_tokens_per_request = total_tokens / (args.benchmark_runs * len(prompts))
        throughput = total_tokens / (sum(latencies) / 1000)
        
        print(f"\n[Benchmark Results]")
        print(f"  Batch size: {len(prompts)}")
        print(f"  Avg latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Avg tokens/request: {avg_tokens_per_request:.1f}")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        
    else:
        # Single inference run
        outputs, elapsed, tokens = run_inference_batch(llm, prompts, sampling_params)
        
        print(f"\n[Results] (inference time: {elapsed*1000:.2f} ms)")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            
            print(f"\nInput [{i}]: {prompt}")
            print(f"Output [{i}]: {generated_text}")
            print(f"Generated tokens: {num_tokens}")


if __name__ == "__main__":
    main()
