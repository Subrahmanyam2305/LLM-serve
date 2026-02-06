#!/usr/bin/env python3
"""
TensorRT-LLM Orchestrator: Download → Convert → Build → Run pipeline.
Supports Llama, Qwen, Gemma, and Phi model families.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Import environment setup
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    from setup_environment import setup_temp_directories, clear_gpu_memory, cleanup_old_temps
except ImportError:
    print("Warning: Environment setup not available")
    setup_temp_directories = clear_gpu_memory = cleanup_old_temps = lambda *args: None

# Model configurations
MODEL_CONFIGS = {
    "llama": {
        "hf_model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "convert_script": "convert_llama.py",
        "gated": True,
    },
    "qwen": {
        "hf_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "convert_script": "convert_qwen.py",  # Existing Qwen script
        "gated": False,
    },
    "gemma": {
        "hf_model_id": "google/gemma-3-4b-it",
        "convert_script": "convert_gemma.py",
        "gated": True,
    },
    "phi": {
        "hf_model_id": "microsoft/Phi-4-mini-instruct",
        "convert_script": "convert_phi.py",
        "gated": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM orchestrator: download → convert → build → run"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model family to use",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default=None,
        help="Override HuggingFace model ID (default: use MODEL_CONFIGS)",
    )
    
    # Directory paths
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Directory to store downloaded HF models",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="Directory to store TensorRT-LLM checkpoints",
    )
    parser.add_argument(
        "--engines_dir",
        type=str,
        default="./engines",
        help="Directory to store TensorRT engines",
    )
    
    # Conversion options
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",  # Explicitly set float16 as default
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights (default: float16)",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1 for single GPU)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int8", "int4", "int4_awq", "fp8"],
        help="Quantization method (default: None - FP16 only)",
    )
    
    # Memory management options
    parser.add_argument(
        "--load_on_cpu",
        action="store_true",
        help="Load model on CPU during conversion (reduces GPU memory usage)",
    )
    parser.add_argument(
        "--single_worker",
        action="store_true",
        help="Use single worker for conversion (reduces memory usage)",
    )
    
    # Build options
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=64,
        help="Maximum batch size for engine",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=4096,
        help="Maximum input length",
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=512,  # Updated default to 512
        help="Maximum output length",
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip model download step",
    )
    parser.add_argument(
        "--skip_convert",
        action="store_true",
        help="Skip checkpoint conversion step",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip engine build step",
    )
    parser.add_argument(
        "--run_only",
        action="store_true",
        help="Only run inference (skip download, convert, build)",
    )
    
    # Inference/benchmark options
    parser.add_argument(
        "--run_benchmark",
        action="store_true",
        help="Run benchmark after building engine",
    )
    parser.add_argument(
        "--start_triton",
        action="store_true",
        help="Start Triton server before benchmarking (requires --run_benchmark)",
    )
    parser.add_argument(
        "--triton_url",
        type=str,
        default="localhost:8001",
        help="Triton server URL for benchmarking (default: localhost:8001)",
    )
    parser.add_argument(
        "--triton_http_port",
        type=int,
        default=8000,
        help="Triton HTTP port when starting server (default: 8000)",
    )
    parser.add_argument(
        "--triton_grpc_port", 
        type=int,
        default=8001,
        help="Triton gRPC port when starting server (default: 8001)",
    )
    parser.add_argument(
        "--sharegpt",
        action="store_true",
        help="Use ShareGPT dataset for benchmarking",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=100,
        help="Maximum prompts for benchmarking",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Custom prompts file for benchmarking",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        nargs="+",
        default=["Hello, how are you?"],
        help="Input text for single inference run",
    )
    
    return parser.parse_args()


def download_model(model_id: str, output_dir: Path, gated: bool = False) -> Path:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    
    model_name = model_id.split("/")[-1]
    local_dir = output_dir / model_name
    
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[Download] Model already exists at {local_dir}")
        return local_dir
    
    print(f"[Download] Downloading {model_id} to {local_dir}...")
    
    # Get HF token for gated models
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if gated and not token:
        print(f"WARNING: {model_id} is a gated model. Set HF_TOKEN environment variable.")
        print("  export HF_TOKEN=your_token")
        print("  Or add to ~/.bashrc")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"[Download] Successfully downloaded to {local_dir}")
    except Exception as e:
        print(f"[Download] ERROR: Failed to download {model_id}: {e}")
        if gated:
            print("  Make sure you have accepted the model license on HuggingFace")
            print("  and set HF_TOKEN environment variable.")
        raise
    
    return local_dir


def convert_checkpoint(
    model: str,
    model_dir: Path,
    output_dir: Path,
    dtype: str = "float16",
    tp_size: int = 1,
    quantization: Optional[str] = None,
    load_on_cpu: bool = True,
    single_worker: bool = True,
) -> Path:
    """Convert HuggingFace checkpoint to TensorRT-LLM format."""
    config = MODEL_CONFIGS[model]
    script_dir = Path(__file__).parent
    convert_script = script_dir / config["convert_script"]
    
    # Create output directory name
    quant_suffix = f"_{quantization}" if quantization else ""
    checkpoint_name = f"trt_{model}_{dtype}{quant_suffix}"
    checkpoint_dir = output_dir / checkpoint_name
    
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        print(f"[Convert] Checkpoint already exists at {checkpoint_dir}")
        return checkpoint_dir
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Convert] Converting {model} checkpoint...")
    print(f"  Model dir: {model_dir}")
    print(f"  Output dir: {checkpoint_dir}")
    print(f"  dtype: {dtype}, tp_size: {tp_size}")
    print(f"  load_on_cpu: {load_on_cpu}, single_worker: {single_worker}")
    
    # Build command based on model type
    if model == "gemma":
        # Gemma uses different argument names
        cmd = [
            sys.executable, str(convert_script),
            "--ckpt-type", "hf",
            "--model-dir", str(model_dir),
            "--output-model-dir", str(checkpoint_dir),
            "--world-size", str(tp_size),
            "--dtype", dtype,
        ]
    else:
        # Llama, Qwen, Phi use similar arguments
        cmd = [
            sys.executable, str(convert_script),
            "--model_dir", str(model_dir),
            "--output_dir", str(checkpoint_dir),
            "--dtype", dtype,
            "--tp_size", str(tp_size),
        ]
        
        # Add memory management options for non-Gemma models
        if load_on_cpu:
            cmd.append("--load_model_on_cpu")
        if single_worker:
            cmd.extend(["--workers", "1"])
    
    # Add quantization options
    if quantization:
        if quantization in ["int8", "int4"]:
            cmd.extend(["--use_weight_only", "--weight_only_precision", quantization])
        elif quantization == "int4_awq":
            cmd.extend(["--use_weight_only", "--weight_only_precision", "int4_awq"])
        elif quantization == "fp8":
            cmd.append("--use_fp8")
    
    print(f"[Convert] Running: {' '.join(cmd)}")
    if load_on_cpu:
        print(f"[Convert] Note: Using CPU loading to reduce memory usage")
    
    # Clear GPU cache before conversion
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Convert] GPU memory cleared")
    except ImportError:
        pass
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        if result.returncode == -9:
            print(f"[Convert] ERROR: Process was killed (likely out of memory)")
            print(f"[Convert] Try using --quantization int4 or --load_on_cpu options")
            print(f"[Convert] Or add more system RAM/swap space")
        raise RuntimeError(f"Checkpoint conversion failed with code {result.returncode}")
    
    print(f"[Convert] Successfully converted to {checkpoint_dir}")
    return checkpoint_dir


def build_engine(
    checkpoint_dir: Path,
    output_dir: Path,
    dtype: str = "float16",
    max_batch_size: int = 64,
    max_input_len: int = 4096,
    max_output_len: int = 1024,
) -> Path:
    """Build TensorRT engine from checkpoint."""
    engine_name = checkpoint_dir.name.replace("trt_", "engine_")
    engine_dir = output_dir / engine_name
    
    if engine_dir.exists() and any(engine_dir.glob("*.engine")):
        print(f"[Build] Engine already exists at {engine_dir}")
        return engine_dir
    
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Build] Building TensorRT engine...")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Output: {engine_dir}")
    print(f"  max_batch_size: {max_batch_size}")
    print(f"  max_input_len: {max_input_len}")
    print(f"  max_output_len: {max_output_len}")
    
    # Clear GPU cache before building
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Build] GPU memory cleared")
    except ImportError:
        pass
    
    # Determine gemm plugin dtype
    gemm_dtype = "float16" if dtype == "float16" else dtype
    
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(engine_dir),
        "--gemm_plugin", gemm_dtype,
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_seq_len", str(max_input_len + max_output_len),
    ]
    
    print(f"[Build] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        if result.returncode == -9:
            print(f"[Build] ERROR: Process was killed (likely out of memory)")
            print(f"[Build] Try reducing max_batch_size, max_input_len, or using quantization")
        raise RuntimeError(f"Engine build failed with code {result.returncode}")
    
    print(f"[Build] Successfully built engine at {engine_dir}")
    return engine_dir


def run_inference(
    engine_dir: Path,
    tokenizer_dir: Path,
    input_texts: list,
    max_output_len: int = 128,
    benchmark: bool = False,
):
    """Run inference using the built engine."""
    script_dir = Path(__file__).parent
    run_script = script_dir / "run_trt.py"
    
    print(f"[Run] Running inference...")
    print(f"  Engine: {engine_dir}")
    print(f"  Tokenizer: {tokenizer_dir}")
    
    cmd = [
        sys.executable, str(run_script),
        "--engine_dir", str(engine_dir),
        "--tokenizer_dir", str(tokenizer_dir),
        "--max_output_len", str(max_output_len),
        "--input_text", *input_texts,
    ]
    
    if benchmark:
        cmd.append("--benchmark")
    
    print(f"[Run] Running: {' '.join(cmd[:10])}...")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed with code {result.returncode}")



def start_triton_server(
    engine_dir: Path,
    tokenizer_dir: Path,
    http_port: int = 8000,
    grpc_port: int = 8001,
) -> subprocess.Popen:
    """Start Triton server with the built engine."""
    script_dir = Path(__file__).parent.parent
    server_script = script_dir / "servers" / "launch_triton_server.py"
    
    print(f"[Triton] Starting Triton server...")
    print(f"  Engine: {engine_dir}")
    print(f"  Tokenizer: {tokenizer_dir}")
    print(f"  HTTP port: {http_port}")
    print(f"  gRPC port: {grpc_port}")
    
    cmd = [
        sys.executable, str(server_script),
        "--engine-dir", str(engine_dir),
        "--tokenizer-dir", str(tokenizer_dir),
        "--http-port", str(http_port),
        "--grpc-port", str(grpc_port),
    ]
    
    print(f"[Triton] Running: {' '.join(cmd[:10])}...")
    
    # Start server in background
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait a bit for server to start
    import time
    import requests
    print("[Triton] Waiting for server to start...")
    time.sleep(15)  # Give server time to initialize
    
    # Check if process is still running
    if process.poll() is not None:
        stdout, _ = process.communicate()
        raise RuntimeError(f"Triton server failed to start: {stdout}")
    
    # Wait for server to be ready (up to 30 more seconds)
    for i in range(30):
        try:
            response = requests.get(f"http://localhost:{http_port}/v2/health/ready", timeout=1)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        print("[Triton] Warning: Server may not be fully ready yet")
    
    print("[Triton] Server started successfully")
    return process


def run_benchmark_suite(
    engine_dir: Path,
    tokenizer_dir: Path,
    sharegpt: bool = False,
    max_prompts: int = 100,
    prompts_file: Optional[str] = None,
    max_output_tokens: int = 512,  # Updated default
    triton_url: str = "localhost:8001",
    output_dir: str = "./analysis",  # Changed to analysis
):
    """
    Run full benchmark suite using Triton benchmark script.
    
    NOTE: This assumes a Triton server is already running with the engine loaded.
    To start the Triton server, run:
    python servers/launch_triton_server.py --engine-dir <engine_dir> --tokenizer-dir <tokenizer_dir>
    """
    script_dir = Path(__file__).parent.parent
    benchmark_script = script_dir / "benchmarks" / "benchmark_triton.py"
    
    print(f"[Benchmark] Running Triton benchmark suite...")
    print(f"[Benchmark] Engine directory: {engine_dir}")
    print(f"[Benchmark] Triton server URL: {triton_url}")
    print(f"[Benchmark] Output directory: {output_dir}")
    print(f"[Benchmark] Make sure Triton server is running with the engine loaded!")
    
    cmd = [
        sys.executable, str(benchmark_script),
        "--engine", "triton",
        "--tokenizer-dir", str(tokenizer_dir),
        "--url", triton_url,
        "--max-output-tokens", str(max_output_tokens),
        "--concurrency", "1", "2", "4", "8", "16", "32", "64",  # Updated concurrency levels
        "--num-runs", "5",  # Updated to 5 runs
        "--num-warmup", "2",  # Updated to 2 warmup runs
        "--output-dir", output_dir,
        "--output", "triton_server_results.json",
    ]
    
    if sharegpt:
        cmd.extend(["--sharegpt", "--max-prompts", str(max_prompts)])
    elif prompts_file:
        cmd.extend(["--prompts-file", prompts_file])
    
    print(f"[Benchmark] Running: {' '.join(cmd[:15])}...")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with code {result.returncode}")
    
    # Return the expected output file path
    return Path(output_dir) / "triton_server_results.json"


def main():
    # Initialize environment and clean caches
    print("🔧 Initializing environment...")
    try:
        setup_temp_directories()
        clear_gpu_memory()
        temp_dir = Path(os.environ.get('TMPDIR', '/tmp'))
        cleanup_old_temps(temp_dir)
        print("✅ Environment initialized\n")
    except Exception as e:
        print(f"⚠️  Environment setup warning: {e}\n")
    
    args = parse_args()
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    hf_model_id = args.hf_model_id or config["hf_model_id"]
    
    # Setup directories
    models_dir = Path(args.models_dir).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    engines_dir = Path(args.engines_dir).resolve()
    
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    engines_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TensorRT-LLM Orchestrator")
    print(f"{'='*60}")
    print(f"Model: {args.model} ({hf_model_id})")
    print(f"dtype: {args.dtype}")
    print(f"Quantization: {args.quantization or 'None (FP16)'}")
    print(f"Memory options: load_on_cpu={args.load_on_cpu}, single_worker={args.single_worker}")
    print(f"{'='*60}\n")
    
    # Provide memory optimization recommendations for large models
    if args.model == "llama" and not args.quantization and not args.load_on_cpu:
        print("💡 MEMORY OPTIMIZATION TIPS:")
        print("   --load_on_cpu        (uses CPU memory for conversion)")  
        print("   --single_worker      (reduces concurrent memory usage)")
        print("   Consider adding swap space: sudo fallocate -l 4G /swapfile")
        print("   Example: python orchestrate_trt.py --model llama --dtype float16 --load_on_cpu --single_worker --run_benchmark --start_triton --sharegpt")
        print()
    
    # Adjust max_batch_size for large models in FP16
    if not args.quantization and args.max_batch_size > 32:
        print(f"[INFO] Reducing max_batch_size from {args.max_batch_size} to 32 for FP16 model")
        args.max_batch_size = 32
    
    # Determine what to skip
    skip_download = args.skip_download or args.run_only
    skip_convert = args.skip_convert or args.run_only
    skip_build = args.skip_build or args.run_only
    
    model_dir = None
    checkpoint_dir = None
    engine_dir = None
    
    # Step 1: Download
    if not skip_download:
        model_dir = download_model(
            model_id=hf_model_id,
            output_dir=models_dir,
            gated=config["gated"],
        )
    else:
        model_name = hf_model_id.split("/")[-1]
        model_dir = models_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found at {model_dir}. Run without --skip_download")
    
    # Step 2: Convert
    if not skip_convert:
        checkpoint_dir = convert_checkpoint(
            model=args.model,
            model_dir=model_dir,
            output_dir=checkpoints_dir,
            dtype=args.dtype,
            tp_size=args.tp_size,
            quantization=args.quantization,
            load_on_cpu=args.load_on_cpu,
            single_worker=args.single_worker,
        )
    else:
        quant_suffix = f"_{args.quantization}" if args.quantization else ""
        checkpoint_name = f"trt_{args.model}_{args.dtype}{quant_suffix}"
        checkpoint_dir = checkpoints_dir / checkpoint_name
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}. Run without --skip_convert")
    
    # Step 3: Build
    if not skip_build:
        engine_dir = build_engine(
            checkpoint_dir=checkpoint_dir,
            output_dir=engines_dir,
            dtype=args.dtype,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
        )
    else:
        engine_name = checkpoint_dir.name.replace("trt_", "engine_")
        engine_dir = engines_dir / engine_name
        if not engine_dir.exists():
            raise FileNotFoundError(f"Engine not found at {engine_dir}. Run without --skip_build")
    
    # Step 4: Run inference or benchmark
    triton_process = None
    try:
        if args.run_benchmark:
            # Start Triton server if requested
            if args.start_triton:
                triton_process = start_triton_server(
                    engine_dir=engine_dir,
                    tokenizer_dir=model_dir,
                    http_port=args.triton_http_port,
                    grpc_port=args.triton_grpc_port,
                )
            
            output_dir = f"./analysis/{args.model}"
            results_file = run_benchmark_suite(
                engine_dir=engine_dir,
                tokenizer_dir=model_dir,
                sharegpt=args.sharegpt,
                max_prompts=args.max_prompts,
                prompts_file=args.prompts_file,
                max_output_tokens=args.max_output_len,
                triton_url=args.triton_url,
                output_dir=output_dir,
            )
            
            print(f"\n🎉 Benchmark Results Summary:")
            print(f"📁 Results saved to: {results_file}")
            if results_file.exists():
                print(f"📊 File size: {results_file.stat().st_size / 1024:.1f} KB")
            print(f"🚀 Use the results for analysis and plotting")
        else:
            run_inference(
                engine_dir=engine_dir,
                tokenizer_dir=model_dir,
                input_texts=args.input_text,
                max_output_len=args.max_output_len,
            )
    finally:
        # Clean up Triton server if we started it
        if triton_process:
            print("[Triton] Shutting down server...")
            triton_process.terminate()
            try:
                triton_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                triton_process.kill()
                triton_process.wait()
            print("[Triton] Server shut down")
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"Model: {model_dir}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Engine: {engine_dir}")


if __name__ == "__main__":
    main()
