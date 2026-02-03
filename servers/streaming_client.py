#!/usr/bin/env python3
"""
Streaming Client - Test streaming responses from OpenAI-compatible servers.
Works with vLLM, SGLang, and TensorRT-LLM backend servers.
"""

import argparse
import json
import time
import sys
from typing import Generator

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Test streaming from OpenAI-compatible server")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000",
                        help="API base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (auto-detected if not provided)")
    parser.add_argument("--prompt", type=str, 
                        default="Explain quantum computing in simple terms.",
                        help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (get full response)")
    parser.add_argument("--chat", action="store_true",
                        help="Use chat completions endpoint instead of completions")
    parser.add_argument("--measure-ttft", action="store_true",
                        help="Measure and display TTFT")
    return parser.parse_args()


def get_available_models(api_base: str, api_key: str) -> list:
    """Get list of available models from the server."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{api_base}/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        print(f"Warning: Could not fetch models: {e}")
    return []


def stream_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    measure_ttft: bool = False
) -> Generator[str, None, dict]:
    """Stream chat completion and yield tokens. Returns metrics at end."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    response = requests.post(
        f"{api_base}/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    )
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            yield content
                except json.JSONDecodeError:
                    continue
    
    end_time = time.perf_counter()
    
    metrics = {
        "total_time_ms": (end_time - start_time) * 1000,
        "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else None,
        "token_count": token_count,
        "tokens_per_sec": token_count / (end_time - start_time) if token_count > 0 else 0,
    }
    
    # Return metrics as final yield (hacky but works)
    return metrics


def stream_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    measure_ttft: bool = False
) -> Generator[str, None, dict]:
    """Stream completion and yield tokens. Returns metrics at end."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    response = requests.post(
        f"{api_base}/v1/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    )
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and chunk["choices"]:
                        text = chunk["choices"][0].get("text", "")
                        if text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            yield text
                except json.JSONDecodeError:
                    continue
    
    end_time = time.perf_counter()
    
    metrics = {
        "total_time_ms": (end_time - start_time) * 1000,
        "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else None,
        "token_count": token_count,
        "tokens_per_sec": token_count / (end_time - start_time) if token_count > 0 else 0,
    }
    
    return metrics


def non_streaming_request(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    chat: bool = False
) -> tuple:
    """Make non-streaming request. Returns (response_text, metrics)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    if chat:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        endpoint = f"{api_base}/v1/chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        endpoint = f"{api_base}/v1/completions"
    
    start_time = time.perf_counter()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    end_time = time.perf_counter()
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    data = response.json()
    
    if chat:
        text = data["choices"][0]["message"]["content"]
    else:
        text = data["choices"][0]["text"]
    
    usage = data.get("usage", {})
    
    metrics = {
        "total_time_ms": (end_time - start_time) * 1000,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_per_sec": usage.get("completion_tokens", 0) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
    }
    
    return text, metrics


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Streaming Client")
    print("=" * 60)
    print(f"API Base: {args.api_base}")
    print(f"Streaming: {not args.no_stream}")
    print(f"Mode: {'Chat' if args.chat else 'Completion'}")
    print("=" * 60)
    
    # Get model name
    model = args.model
    if not model:
        models = get_available_models(args.api_base, args.api_key)
        if models:
            model = models[0]
            print(f"Auto-detected model: {model}")
        else:
            model = "default"
            print(f"Using default model name: {model}")
    
    print(f"\nPrompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("\n" + "-" * 60)
    print("Response:")
    print("-" * 60 + "\n")
    
    try:
        if args.no_stream:
            # Non-streaming request
            text, metrics = non_streaming_request(
                args.api_base, args.api_key, model,
                args.prompt, args.max_tokens, args.temperature,
                chat=args.chat
            )
            print(text)
            print("\n" + "-" * 60)
            print("Metrics:")
            print(f"  Total time: {metrics['total_time_ms']:.2f} ms")
            print(f"  Tokens/sec: {metrics['tokens_per_sec']:.2f}")
        else:
            # Streaming request
            full_response = ""
            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0
            
            if args.chat:
                stream_func = stream_chat_completion
            else:
                stream_func = stream_completion
            
            # Manual streaming loop to capture metrics
            headers = {
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
            }
            
            if args.chat:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": args.prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "stream": True,
                }
                endpoint = f"{args.api_base}/v1/chat/completions"
            else:
                payload = {
                    "model": model,
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "stream": True,
                }
                endpoint = f"{args.api_base}/v1/completions"
            
            response = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=120)
            
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                sys.exit(1)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                if args.chat:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                else:
                                    content = chunk["choices"][0].get("text", "")
                                
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    token_count += 1
                                    full_response += content
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.perf_counter()
            
            print("\n\n" + "-" * 60)
            print("Metrics:")
            print(f"  Total time: {(end_time - start_time) * 1000:.2f} ms")
            if first_token_time:
                print(f"  TTFT: {(first_token_time - start_time) * 1000:.2f} ms")
            print(f"  Tokens generated: ~{token_count}")
            if token_count > 0:
                print(f"  Tokens/sec: {token_count / (end_time - start_time):.2f}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
