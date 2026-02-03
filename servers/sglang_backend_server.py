#!/usr/bin/env python3
"""
SGLang Backend Server - OpenAI-compatible API server.
Starts an SGLang server that exposes an OpenAI-compatible API endpoint.
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Start SGLang OpenAI-compatible server")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model or model name")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port to listen on (default 8001 to avoid conflict with vLLM)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32", "auto"],
                        help="Model dtype")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88,
                        help="Static memory fraction for KV cache")
    parser.add_argument("--max-total-tokens", type=int, default=None,
                        help="Maximum total tokens in the system")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key for authentication")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="Disable CUDA graph for debugging")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SGLang OpenAI-Compatible Server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Dtype: {args.dtype}")
    print(f"Memory Fraction: {args.mem_fraction_static}")
    print("=" * 60)
    
    # Build command
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--trust-remote-code",
    ]
    
    if args.max_total_tokens:
        cmd.extend(["--max-total-tokens", str(args.max_total_tokens)])
    
    if args.tensor_parallel_size > 1:
        cmd.extend(["--tp-size", str(args.tensor_parallel_size)])
    
    if args.api_key != "EMPTY":
        cmd.extend(["--api-key", args.api_key])
    
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    
    print(f"\nStarting server with command:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"OpenAI API endpoint: http://{args.host}:{args.port}/v1")
    print("=" * 60 + "\n")
    
    # Run server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Server failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
