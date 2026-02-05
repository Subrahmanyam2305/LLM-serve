#!/usr/bin/env python3
"""
vLLM Backend Server - OpenAI-compatible API server.
Starts a vLLM server that exposes an OpenAI-compatible API endpoint.
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible server")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model or model name")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32", "auto"],
                        help="Model dtype")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization (0-1)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model context length")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key for authentication")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("vLLM OpenAI-Compatible Server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Dtype: {args.dtype}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print("=" * 60)
    
    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    
    if args.tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    
    if args.api_key != "EMPTY":
        cmd.extend(["--api-key", args.api_key])
    
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
