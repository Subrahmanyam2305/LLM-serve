#!/usr/bin/env python3
"""
Environment setup and cache management for TensorRT-LLM operations.
This script sets up proper temporary directories and environment variables
to prevent memory issues and cache buildup.
"""

import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path
import gc


def setup_temp_directories():
    """Set up proper temporary directories to avoid memory issues."""
    # Create custom temp directory in the project space
    project_root = Path(__file__).parent.parent.parent.absolute()
    custom_tmp = project_root / "tmp"
    custom_tmp.mkdir(exist_ok=True)
    
    # Set environment variables for temporary directories
    env_vars = {
        'TMPDIR': str(custom_tmp),
        'TMP': str(custom_tmp),
        'TEMP': str(custom_tmp),
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',  # Prevent large CUDA memory allocations
        'CUDA_LAUNCH_BLOCKING': '1',  # Better error reporting
        'TRANSFORMERS_OFFLINE': '0',  # Allow downloading if needed
        'HF_HOME': str(project_root / ".cache" / "huggingface"),
        'TORCH_HOME': str(project_root / ".cache" / "torch"),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    return custom_tmp


def cleanup_old_temps(temp_dir: Path, max_age_hours: int = 24):
    """Clean up old temporary directories and files."""
    import time
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    cleaned_count = 0
    
    for item in temp_dir.iterdir():
        try:
            if item.is_dir() and item.name.startswith('tmp'):
                item_age = current_time - item.stat().st_mtime
                if item_age > max_age_seconds:
                    shutil.rmtree(item, ignore_errors=True)
                    cleaned_count += 1
            elif item.is_file() and item.name.startswith('tmp'):
                item_age = current_time - item.stat().st_mtime
                if item_age > max_age_seconds:
                    item.unlink(missing_ok=True)
                    cleaned_count += 1
        except (OSError, PermissionError):
            continue
    
    print(f"Cleaned {cleaned_count} old temporary items")


def clear_gpu_memory():
    """Clear GPU memory cache if PyTorch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU memory cache cleared")
    except ImportError:
        print("PyTorch not available, skipping GPU memory clear")
    
    # Force garbage collection
    gc.collect()
    print("Python garbage collection completed")


def setup_swap_if_needed():
    """Check if swap is available and recommend setup if not."""
    try:
        # Check current swap
        result = subprocess.run(['swapon', '--show'], 
                              capture_output=True, text=True, timeout=5)
        
        if not result.stdout.strip():
            print("\n⚠️  WARNING: No swap space detected!")
            print("Consider adding swap space to prevent OOM issues:")
            print("  sudo fallocate -l 2G /swapfile")
            print("  sudo chmod 600 /swapfile")
            print("  sudo mkswap /swapfile")
            print("  sudo swapon /swapfile")
            print("  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab")
        else:
            print("✓ Swap space is available")
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        print("Could not check swap status")


def clean_tensorrt_caches():
    """Clean TensorRT-specific cache files."""
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    cache_patterns = [
        "**/*.cache",
        "**/__pycache__",
        "**/model.cache",
        "**/.nv_cache",
    ]
    
    cleaned_files = 0
    for pattern in cache_patterns:
        for cache_file in project_root.glob(pattern):
            try:
                if cache_file.is_file():
                    cache_file.unlink()
                    cleaned_files += 1
                elif cache_file.is_dir() and cache_file.name == "__pycache__":
                    shutil.rmtree(cache_file, ignore_errors=True)
                    cleaned_files += 1
            except (OSError, PermissionError):
                continue
    
    print(f"Cleaned {cleaned_files} TensorRT cache files")


def get_memory_info():
    """Display current memory usage information."""
    try:
        # Get memory info
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = None
        mem_available = None
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) // 1024  # Convert to MB
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1]) // 1024  # Convert to MB
        
        if mem_total and mem_available:
            mem_used = mem_total - mem_available
            usage_pct = (mem_used / mem_total) * 100
            
            print(f"\n📊 Memory Status:")
            print(f"   Total: {mem_total:,} MB")
            print(f"   Used:  {mem_used:,} MB ({usage_pct:.1f}%)")
            print(f"   Available: {mem_available:,} MB")
            
            if usage_pct > 80:
                print("⚠️  High memory usage detected!")
            elif usage_pct > 60:
                print("⚡ Moderate memory usage")
            else:
                print("✓ Memory usage looks good")
        
    except Exception as e:
        print(f"Could not get memory info: {e}")


def main():
    """Main environment setup function."""
    print("🔧 Setting up TensorRT-LLM environment...")
    print("=" * 60)
    
    # Setup temporary directories
    temp_dir = setup_temp_directories()
    print()
    
    # Clean old temporary files
    cleanup_old_temps(temp_dir)
    print()
    
    # Clear GPU memory
    clear_gpu_memory()
    print()
    
    # Clean TensorRT caches
    clean_tensorrt_caches()
    print()
    
    # Check swap
    setup_swap_if_needed()
    print()
    
    # Display memory info
    get_memory_info()
    print()
    
    print("✅ Environment setup complete!")
    print("=" * 60)
    
    return temp_dir


if __name__ == "__main__":
    main()