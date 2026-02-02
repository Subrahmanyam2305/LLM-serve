#!/usr/bin/env python3
"""
vLLM inference runner for benchmarking.
Uses vLLM's offline inference API for batch processing.
"""

import argparse
import time
from typing import List

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model or model name")
    parser.add_argument("--input_text", type=str, nargs="+",
                        default=["Hello, how are you?"],
                        help="Input text(s) for generation")
    parser.add_argument("--max_output_len", type=int, default=128,
                        help="Maximum output tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
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


def run_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams):
    """Run inference and return outputs with timing."""
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.perf_counter() - start_time
    return outputs, elapsed_time


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
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
    )
    
    prompts = args.input_text
    print(f"[vLLM] Batch size: {len(prompts)}")
    
    if args.benchmark:
        # Warmup
        print(f"\n[Benchmark] Running {args.warmup_runs} warmup iterations...")
        for _ in range(args.warmup_runs):
            run_inference(llm, prompts, sampling_params)
        
        # Benchmark
        print(f"[Benchmark] Running {args.benchmark_runs} benchmark iterations...")
        latencies = []
        total_tokens = 0
        
        for i in range(args.benchmark_runs):
            outputs, elapsed = run_inference(llm, prompts, sampling_params)
            latencies.append(elapsed)
            
            # Count generated tokens
            for output in outputs:
                total_tokens += len(output.outputs[0].token_ids)
        
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens_per_request = total_tokens / (args.benchmark_runs * len(prompts))
        throughput = total_tokens / sum(latencies)
        
        print(f"\n[Benchmark Results]")
        print(f"  Batch size: {len(prompts)}")
        print(f"  Avg latency: {avg_latency*1000:.2f} ms")
        print(f"  Avg tokens/request: {avg_tokens_per_request:.1f}")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        
    else:
        # Single inference run
        outputs, elapsed = run_inference(llm, prompts, sampling_params)
        
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
