#!/usr/bin/env python3
"""
SGLang inference runner for benchmarking.
Uses SGLang's Engine API for direct inference without the sgl.function decorator.
"""

import argparse
import time
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""
    total_latency_ms: float
    ttft_ms: float  # Time to first token
    tokens_generated: int
    throughput_tokens_per_sec: float
    itl_ms: float  # Inter-token latency (average)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SGLang inference")
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
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k sampling parameter (1 for greedy)")
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


def run_inference_batch(engine, tokenizer, prompts: List[str], args) -> Tuple[list, float, int]:
    """Run batch inference and return outputs with timing and token count."""
    import sglang as sgl
    
    sampling_params = {
        "max_new_tokens": args.max_output_len,
        "temperature": args.temperature if args.temperature > 0 else 0,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    
    start_time = time.perf_counter()
    
    # Use engine.generate for direct inference
    outputs = engine.generate(
        prompts,
        sampling_params=sampling_params,
    )
    
    elapsed_time = time.perf_counter() - start_time
    
    # Count tokens using tokenizer for accuracy
    total_tokens = 0
    for output in outputs:
        if hasattr(output, 'text'):
            text = output.text
        elif isinstance(output, dict) and 'text' in output:
            text = output['text']
        else:
            text = str(output)
        
        # Use tokenizer for accurate token count
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
    
    return outputs, elapsed_time, total_tokens


def main():
    args = parse_args()
    
    print(f"[SGLang] Loading model: {args.model}")
    print(f"[SGLang] dtype: {args.dtype}")
    
    # Import and initialize SGLang engine
    import sglang as sgl
    from transformers import AutoTokenizer
    
    # Load tokenizer for accurate token counting
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Initialize SGLang engine
    engine = sgl.Engine(
        model_path=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    
    prompts = args.input_text
    print(f"[SGLang] Batch size: {len(prompts)}")
    print(f"[SGLang] Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    
    if args.benchmark:
        # Warmup
        print(f"\n[Benchmark] Running {args.warmup_runs} warmup iterations...")
        for _ in range(args.warmup_runs):
            run_inference_batch(engine, tokenizer, prompts, args)
        
        # Benchmark
        print(f"[Benchmark] Running {args.benchmark_runs} benchmark iterations...")
        latencies = []
        total_tokens = 0
        
        for i in range(args.benchmark_runs):
            outputs, elapsed, tokens = run_inference_batch(engine, tokenizer, prompts, args)
            latencies.append(elapsed * 1000)  # Convert to ms
            total_tokens += tokens
        
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
        outputs, elapsed, tokens = run_inference_batch(engine, tokenizer, prompts, args)
        
        print(f"\n[Results] (inference time: {elapsed*1000:.2f} ms)")
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            if hasattr(output, 'text'):
                generated_text = output.text
            elif isinstance(output, dict) and 'text' in output:
                generated_text = output['text']
            else:
                generated_text = str(output)
            
            num_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
            
            print(f"\nInput [{i}]: {prompt}")
            print(f"Output [{i}]: {generated_text}")
            print(f"Generated tokens: {num_tokens}")
    
    # Cleanup
    engine.shutdown()


if __name__ == "__main__":
    main()
