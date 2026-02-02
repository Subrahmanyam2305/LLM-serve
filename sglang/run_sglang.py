#!/usr/bin/env python3
"""
SGLang inference runner for benchmarking.
Uses SGLang's offline inference API for batch processing.
"""

import argparse
import time
from typing import List

import sglang as sgl
from sglang import RuntimeEndpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run SGLang inference")
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
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k sampling parameter")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode (measure latency)")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Number of warmup runs for benchmarking")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                        help="Number of benchmark runs")
    return parser.parse_args()


@sgl.function
def generate_text(s, prompt, max_tokens, temperature, top_p, top_k):
    """SGLang generation function."""
    s += prompt
    s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature, 
                  top_p=top_p, top_k=top_k)


def run_inference_batch(runtime, prompts: List[str], args):
    """Run batch inference and return outputs with timing."""
    start_time = time.perf_counter()
    
    # Run batch inference
    states = generate_text.run_batch(
        [{"prompt": p, "max_tokens": args.max_output_len, 
          "temperature": args.temperature, "top_p": args.top_p, 
          "top_k": args.top_k} for p in prompts],
        progress_bar=False
    )
    
    elapsed_time = time.perf_counter() - start_time
    return states, elapsed_time


def main():
    args = parse_args()
    
    print(f"[SGLang] Loading model: {args.model}")
    print(f"[SGLang] dtype: {args.dtype}")
    
    # Initialize SGLang runtime
    runtime = sgl.Runtime(
        model_path=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    sgl.set_default_backend(runtime)
    
    prompts = args.input_text
    print(f"[SGLang] Batch size: {len(prompts)}")
    
    if args.benchmark:
        # Warmup
        print(f"\n[Benchmark] Running {args.warmup_runs} warmup iterations...")
        for _ in range(args.warmup_runs):
            run_inference_batch(runtime, prompts, args)
        
        # Benchmark
        print(f"[Benchmark] Running {args.benchmark_runs} benchmark iterations...")
        latencies = []
        total_tokens = 0
        
        for i in range(args.benchmark_runs):
            states, elapsed = run_inference_batch(runtime, prompts, args)
            latencies.append(elapsed)
            
            # Count generated tokens
            for state in states:
                response = state["response"]
                # Approximate token count (actual count would need tokenizer)
                total_tokens += len(response.split()) * 1.3  # rough estimate
        
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens_per_request = total_tokens / (args.benchmark_runs * len(prompts))
        throughput = total_tokens / sum(latencies)
        
        print(f"\n[Benchmark Results]")
        print(f"  Batch size: {len(prompts)}")
        print(f"  Avg latency: {avg_latency*1000:.2f} ms")
        print(f"  Avg tokens/request: {avg_tokens_per_request:.1f}")
        print(f"  Throughput: {throughput:.2f} tokens/sec (estimated)")
        
    else:
        # Single inference run
        states, elapsed = run_inference_batch(runtime, prompts, args)
        
        print(f"\n[Results] (inference time: {elapsed*1000:.2f} ms)")
        for i, (prompt, state) in enumerate(zip(prompts, states)):
            generated_text = state["response"]
            
            print(f"\nInput [{i}]: {prompt}")
            print(f"Output [{i}]: {generated_text}")
    
    # Cleanup
    runtime.shutdown()


if __name__ == "__main__":
    main()
