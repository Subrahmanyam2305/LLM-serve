#!/usr/bin/env python3
"""
TensorRT-LLM inference runner for benchmarking.
Loads a pre-built TensorRT engine and runs inference.
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp, PYTHON_BINDINGS


def parse_args():
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM inference")
    parser.add_argument("--engine_dir", type=str, required=True,
                        help="Path to TensorRT engine directory")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Path to tokenizer (HF model directory)")
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
    parser.add_argument("--use_py_session", action="store_true",
                        help="Use Python session instead of C++")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode (measure latency)")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Number of warmup runs for benchmarking")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                        help="Number of benchmark runs")
    return parser.parse_args()


def load_tokenizer(tokenizer_dir: str):
    """Load tokenizer from HuggingFace model directory."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    end_id = tokenizer.eos_token_id
    return tokenizer, pad_id, end_id


def tokenize_inputs(tokenizer, input_texts: List[str], max_length: int = 1024):
    """Tokenize input texts into tensor format."""
    batch_input_ids = []
    for text in input_texts:
        input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.int32))
    return batch_input_ids


def run_inference(runner, batch_input_ids, args, end_id, pad_id):
    """Run inference and return outputs with timing."""
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=args.max_output_len,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    return outputs, elapsed_time


def main():
    args = parse_args()
    
    print(f"[TensorRT-LLM] Version: {tensorrt_llm.__version__}")
    print(f"[TensorRT-LLM] Loading engine from: {args.engine_dir}")
    
    # Load tokenizer
    tokenizer, pad_id, end_id = load_tokenizer(args.tokenizer_dir)
    print(f"[TensorRT-LLM] Loaded tokenizer from: {args.tokenizer_dir}")
    
    # Tokenize inputs
    batch_input_ids = tokenize_inputs(tokenizer, args.input_text)
    input_lengths = [len(ids) for ids in batch_input_ids]
    print(f"[TensorRT-LLM] Batch size: {len(batch_input_ids)}, Input lengths: {input_lengths}")
    
    # Select runner class
    if args.use_py_session or not PYTHON_BINDINGS:
        runner_cls = ModelRunner
        print("[TensorRT-LLM] Using Python session")
    else:
        runner_cls = ModelRunnerCpp
        print("[TensorRT-LLM] Using C++ session")
    
    # Load runner
    runner = runner_cls.from_dir(
        engine_dir=args.engine_dir,
        rank=0,
        max_batch_size=len(batch_input_ids),
        max_input_len=max(input_lengths),
        max_output_len=args.max_output_len,
    )
    
    if args.benchmark:
        # Warmup
        print(f"\n[Benchmark] Running {args.warmup_runs} warmup iterations...")
        for _ in range(args.warmup_runs):
            run_inference(runner, batch_input_ids, args, end_id, pad_id)
        
        # Benchmark
        print(f"[Benchmark] Running {args.benchmark_runs} benchmark iterations...")
        latencies = []
        total_tokens = 0
        
        for i in range(args.benchmark_runs):
            outputs, elapsed = run_inference(runner, batch_input_ids, args, end_id, pad_id)
            latencies.append(elapsed)
            
            # Count generated tokens
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']
            for j, input_len in enumerate(input_lengths):
                generated = sequence_lengths[j][0].item() - input_len
                total_tokens += generated
        
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens_per_request = total_tokens / (args.benchmark_runs * len(batch_input_ids))
        throughput = total_tokens / sum(latencies)
        
        print(f"\n[Benchmark Results]")
        print(f"  Batch size: {len(batch_input_ids)}")
        print(f"  Avg latency: {avg_latency*1000:.2f} ms")
        print(f"  Avg tokens/request: {avg_tokens_per_request:.1f}")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        
    else:
        # Single inference run
        outputs, elapsed = run_inference(runner, batch_input_ids, args, end_id, pad_id)
        
        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        
        print(f"\n[Results] (inference time: {elapsed*1000:.2f} ms)")
        for i, input_text in enumerate(args.input_text):
            input_len = input_lengths[i]
            output_len = sequence_lengths[i][0].item()
            generated_ids = output_ids[i][0][input_len:output_len].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nInput [{i}]: {input_text}")
            print(f"Output [{i}]: {generated_text}")
            print(f"Generated tokens: {output_len - input_len}")


if __name__ == "__main__":
    main()
