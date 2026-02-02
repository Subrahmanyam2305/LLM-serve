#!/usr/bin/env python3
"""
ShareGPT dataset loader for LLM benchmarking.
Extracts human prompts from ShareGPT conversations.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_sharegpt_from_hf(
    max_prompts: Optional[int] = None,
    split: str = "train",
    seed: int = 42,
    min_length: int = 10,
    max_length: int = 4096,
) -> List[str]:
    """
    Load prompts from ShareGPT dataset via HuggingFace datasets.
    
    Args:
        max_prompts: Maximum number of prompts to return (None for all)
        split: Dataset split to use
        seed: Random seed for reproducible sampling
        min_length: Minimum prompt length in characters
        max_length: Maximum prompt length in characters
        
    Returns:
        List of human prompts from ShareGPT conversations
    """
    if not HAS_DATASETS:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )
    
    print(f"[ShareGPT] Loading dataset from HuggingFace...")
    
    # Use the original ShareGPT dataset
    dataset_source = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    
    try:
        print(f"[ShareGPT] Loading from {dataset_source}...")
        dataset = load_dataset(
            dataset_source,
            split=split,
            trust_remote_code=True,
        )
        print(f"[ShareGPT] Successfully loaded from {dataset_source}")
    except Exception as e:
        raise RuntimeError(f"Failed to load ShareGPT dataset: {e}")
    
    prompts = []
    for item in dataset:
        conversations = item.get("conversations", [])
        # Extract first human turn
        for turn in conversations:
            if turn.get("from") == "human":
                value = turn.get("value", "").strip()
                # Filter by length
                if min_length <= len(value) <= max_length:
                    prompts.append(value)
                break  # Only take first human turn per conversation
    
    print(f"[ShareGPT] Extracted {len(prompts)} prompts from dataset")
    
    # Shuffle and subset
    if max_prompts and max_prompts < len(prompts):
        random.seed(seed)
        prompts = random.sample(prompts, max_prompts)
        print(f"[ShareGPT] Sampled {max_prompts} prompts (seed={seed})")
    
    return prompts


def load_sharegpt_from_json(
    json_path: str,
    max_prompts: Optional[int] = None,
    seed: int = 42,
    min_length: int = 10,
    max_length: int = 4096,
) -> List[str]:
    """
    Load prompts from a local ShareGPT JSON file.
    
    Args:
        json_path: Path to ShareGPT JSON file
        max_prompts: Maximum number of prompts to return
        seed: Random seed for reproducible sampling
        min_length: Minimum prompt length in characters
        max_length: Maximum prompt length in characters
        
    Returns:
        List of human prompts
    """
    print(f"[ShareGPT] Loading from local file: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for item in data:
        conversations = item.get("conversations", [])
        for turn in conversations:
            if turn.get("from") == "human":
                value = turn.get("value", "").strip()
                if min_length <= len(value) <= max_length:
                    prompts.append(value)
                break
    
    print(f"[ShareGPT] Extracted {len(prompts)} prompts from file")
    
    if max_prompts and max_prompts < len(prompts):
        random.seed(seed)
        prompts = random.sample(prompts, max_prompts)
        print(f"[ShareGPT] Sampled {max_prompts} prompts (seed={seed})")
    
    return prompts


def load_sharegpt(
    max_prompts: Optional[int] = None,
    json_path: Optional[str] = None,
    split: str = "train",
    seed: int = 42,
    min_length: int = 10,
    max_length: int = 4096,
) -> List[str]:
    """
    Load ShareGPT prompts from HuggingFace or local JSON file.
    
    Args:
        max_prompts: Maximum number of prompts to return
        json_path: Optional path to local JSON file (uses HF if None)
        split: Dataset split (for HF loading)
        seed: Random seed for reproducible sampling
        min_length: Minimum prompt length in characters
        max_length: Maximum prompt length in characters
        
    Returns:
        List of human prompts
    """
    if json_path and Path(json_path).exists():
        return load_sharegpt_from_json(
            json_path=json_path,
            max_prompts=max_prompts,
            seed=seed,
            min_length=min_length,
            max_length=max_length,
        )
    else:
        return load_sharegpt_from_hf(
            max_prompts=max_prompts,
            split=split,
            seed=seed,
            min_length=min_length,
            max_length=max_length,
        )


def filter_prompts_by_tokens(
    prompts: List[str],
    tokenizer,
    min_tokens: int = 0,
    max_tokens: int = 4096,
) -> List[str]:
    """
    Filter prompts by token count using a tokenizer.
    
    Args:
        prompts: List of prompts to filter
        tokenizer: HuggingFace tokenizer
        min_tokens: Minimum token count
        max_tokens: Maximum token count
        
    Returns:
        Filtered list of prompts
    """
    filtered = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if min_tokens <= len(tokens) <= max_tokens:
            filtered.append(prompt)
    
    print(f"[ShareGPT] Filtered to {len(filtered)} prompts by token count ({min_tokens}-{max_tokens})")
    return filtered


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load ShareGPT prompts")
    parser.add_argument("--max_prompts", type=int, default=100,
                        help="Maximum number of prompts to load")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to local ShareGPT JSON file")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--min_length", type=int, default=10,
                        help="Minimum prompt length in characters")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum prompt length in characters")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for prompts")
    
    args = parser.parse_args()
    
    prompts = load_sharegpt(
        max_prompts=args.max_prompts,
        json_path=args.json_path,
        split=args.split,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    
    print(f"\nLoaded {len(prompts)} prompts")
    print(f"Sample prompts:")
    for i, p in enumerate(prompts[:3]):
        print(f"  [{i}] {p[:100]}...")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")
