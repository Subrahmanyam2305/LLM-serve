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

# Model configurations
MODEL_CONFIGS = {
    "llama": {
        "hf_model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "convert_script": "convert_llama.py",
        "gated": True,
    },
    "qwen": {
        "hf_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "convert_script": "convert_checkpoint.py",  # Existing Qwen script
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
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
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
        help="Quantization method (default: None for FP16)",
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
        default=1024,
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
    
    # Add quantization options
    if quantization:
        if quantization in ["int8", "int4"]:
            cmd.extend(["--use_weight_only", "--weight_only_precision", quantization])
        elif quantization == "int4_awq":
            cmd.extend(["--use_weight_only", "--weight_only_precision", "int4_awq"])
        elif quantization == "fp8":
            cmd.append("--enable_fp8")
    
    print(f"[Convert] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
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


def run_benchmark_suite(
    engine_dir: Path,
    tokenizer_dir: Path,
    sharegpt: bool = False,
    max_prompts: int = 100,
    prompts_file: Optional[str] = None,
    max_output_tokens: int = 128,
):
    """Run full benchmark suite."""
    script_dir = Path(__file__).parent.parent
    benchmark_script = script_dir / "benchmarks" / "benchmark.py"
    
    print(f"[Benchmark] Running benchmark suite...")
    
    cmd = [
        sys.executable, str(benchmark_script),
        "--model_path", str(tokenizer_dir),
        "--trt_engine_dir", str(engine_dir),
        "--engines", "tensorrt",
        "--max_output_tokens", str(max_output_tokens),
        "--batch_sizes", "1", "4", "8", "16",
    ]
    
    if sharegpt:
        cmd.extend(["--sharegpt", "--max_prompts", str(max_prompts)])
    elif prompts_file:
        cmd.extend(["--prompts_file", prompts_file])
    
    print(f"[Benchmark] Running: {' '.join(cmd[:15])}...")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with code {result.returncode}")


def main():
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
    print(f"{'='*60}\n")
    
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
    if args.run_benchmark:
        run_benchmark_suite(
            engine_dir=engine_dir,
            tokenizer_dir=model_dir,
            sharegpt=args.sharegpt,
            max_prompts=args.max_prompts,
            prompts_file=args.prompts_file,
            max_output_tokens=args.max_output_len,
        )
    else:
        run_inference(
            engine_dir=engine_dir,
            tokenizer_dir=model_dir,
            input_texts=args.input_text,
            max_output_len=args.max_output_len,
        )
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"Model: {model_dir}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Engine: {engine_dir}")


if __name__ == "__main__":
    main()
