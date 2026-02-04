#!/usr/bin/env python3
"""
Triton TensorRT-LLM Server Launcher.

This script sets up and launches Triton Inference Server with TensorRT-LLM backend.
Triton provides proper concurrent request handling through inflight batching.

Usage:
    python servers/launch_triton_server.py \
        --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B \
        --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct \
        --http-port 8000 \
        --grpc-port 8001
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Triton Docker image compatible with TensorRT-LLM 1.1.0
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Triton Inference Server with TensorRT-LLM backend"
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        required=True,
        help="Path to TensorRT-LLM engine directory",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        required=True,
        help="Path to tokenizer/HuggingFace model directory",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="/tmp/triton_model_repo",
        help="Path to Triton model repository",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="HTTP port for Triton server",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=8001,
        help="gRPC port for Triton server",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8002,
        help="Metrics port for Triton server",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum batch size for inflight batching",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="triton-trtllm",
        help="Name for Docker container",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run container in detached mode",
    )
    
    return parser.parse_args()


def create_model_repository(
    model_repo: Path,
    engine_dir: str,
    tokenizer_dir: str,
    max_batch_size: int,
) -> None:
    """Create Triton model repository with TensorRT-LLM configuration."""
    
    print(f"Creating model repository at: {model_repo}")
    
    # Clean up existing repo
    if model_repo.exists():
        shutil.rmtree(model_repo)
    
    # Create directory structure
    tensorrt_llm_dir = model_repo / "tensorrt_llm" / "1"
    tensorrt_llm_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config.pbtxt for tensorrt_llm model
    # This config is compatible with TensorRT-LLM 1.1.0
    config_content = f'''name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: {max_batch_size}

model_transaction_policy {{
  decoupled: True
}}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: {{ shape: [ ] }}
  }},
  {{
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "beam_width"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "runtime_top_k"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "runtime_top_p"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "end_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "pad_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "streaming"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }}
]

output [
  {{
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  }},
  {{
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]

parameters: {{
  key: "gpt_model_type"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}

parameters: {{
  key: "gpt_model_path"
  value: {{
    string_value: "{engine_dir}"
  }}
}}

parameters: {{
  key: "max_beam_width"
  value: {{
    string_value: "1"
  }}
}}

parameters: {{
  key: "batching_strategy"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}

parameters: {{
  key: "batch_scheduler_policy"
  value: {{
    string_value: "max_utilization"
  }}
}}

parameters: {{
  key: "tokenizer_dir"
  value: {{
    string_value: "{tokenizer_dir}"
  }}
}}

parameters: {{
  key: "xgrammar_tokenizer_info_path"
  value: {{
    string_value: ""
  }}
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
'''
    
    config_path = model_repo / "tensorrt_llm" / "config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created config at: {config_path}")
    print("Model repository created successfully!")


def launch_triton_container(args, model_repo: Path) -> bool:
    """Launch Triton server in Docker container."""
    
    # Stop existing container if running
    subprocess.run(
        ["docker", "rm", "-f", args.container_name],
        capture_output=True,
    )
    
    # Build docker run command
    cmd = [
        "docker", "run",
        "--name", args.container_name,
        "--gpus", "all",
        "--shm-size=4g",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-p", f"{args.http_port}:8000",
        "-p", f"{args.grpc_port}:8001",
        "-p", f"{args.metrics_port}:8002",
        "-v", f"{model_repo}:/triton_model_repo",
        "-v", f"{args.engine_dir}:{args.engine_dir}",
        "-v", f"{args.tokenizer_dir}:{args.tokenizer_dir}",
    ]
    
    if args.detach:
        cmd.append("-d")
    else:
        cmd.extend(["--rm"])
    
    cmd.extend([
        TRITON_IMAGE,
        "tritonserver",
        "--model-repository=/triton_model_repo",
        "--disable-auto-complete-config",
        "--backend-config=python,shm-region-prefix-name=prefix0_",
    ])
    
    print(f"\nLaunching Triton server...")
    print(f"Command: {' '.join(cmd)}\n")
    
    if args.detach:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            container_id = result.stdout.strip()
            print(f"Container started: {container_id[:12]}")
            return True
        else:
            print(f"Failed to start container: {result.stderr}")
            return False
    else:
        # Run interactively
        subprocess.run(cmd)
        return True


def wait_for_server(http_port: int, timeout: int = 180) -> bool:
    """Wait for Triton server to be ready."""
    import urllib.request
    import urllib.error
    
    url = f"http://localhost:{http_port}/v2/health/ready"
    start_time = time.time()
    
    print(f"Waiting for server to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=5)
            if response.status == 200:
                print("Server is ready!")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError):
            pass
        
        time.sleep(3)
        elapsed = int(time.time() - start_time)
        print(f"  Waiting... ({elapsed}s)")
    
    print("Timeout waiting for server to be ready.")
    return False


def main():
    args = parse_args()
    
    # Validate paths
    engine_dir = Path(args.engine_dir).absolute()
    tokenizer_dir = Path(args.tokenizer_dir).absolute()
    model_repo = Path(args.model_repo).absolute()
    
    if not engine_dir.exists():
        print(f"ERROR: Engine directory not found: {engine_dir}")
        sys.exit(1)
    
    if not tokenizer_dir.exists():
        print(f"ERROR: Tokenizer directory not found: {tokenizer_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Triton TensorRT-LLM Server")
    print(f"{'='*60}")
    print(f"Docker Image: {TRITON_IMAGE}")
    print(f"Engine: {engine_dir}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Model Repo: {model_repo}")
    print(f"HTTP Port: {args.http_port}")
    print(f"gRPC Port: {args.grpc_port}")
    print(f"{'='*60}\n")
    
    # Create model repository
    create_model_repository(
        model_repo,
        str(engine_dir),
        str(tokenizer_dir),
        args.max_batch_size,
    )
    
    # Launch container
    if not launch_triton_container(args, model_repo):
        sys.exit(1)
    
    if args.detach:
        # Wait for server to be ready
        if wait_for_server(args.http_port):
            print(f"\n{'='*60}")
            print("Triton server is running!")
            print(f"HTTP endpoint: http://localhost:{args.http_port}")
            print(f"gRPC endpoint: localhost:{args.grpc_port}")
            print(f"\nTo check logs: docker logs {args.container_name}")
            print(f"To stop: docker stop {args.container_name}")
            print(f"{'='*60}")
        else:
            print("\nServer may not be ready. Check logs:")
            print(f"  docker logs {args.container_name}")
            sys.exit(1)


if __name__ == "__main__":
    main()
