#!/usr/bin/env python3
"""
Triton gRPC Client for TensorRT-LLM Inference.

This module provides a high-level wrapper around the Triton Inference Server gRPC client
for sending concurrent requests to TensorRT-LLM models. It supports both streaming and
non-streaming inference modes.

Based on the inflight_batcher_llm_client.py from TensorRT-LLM triton_backend.

Usage:
    from triton_client.triton_grpc_client import TritonLLMClient
    
    client = TritonLLMClient(
        url="localhost:8001",
        tokenizer_dir="/path/to/tokenizer"
    )
    
    # Single request
    response = client.generate("Hello, how are you?", max_tokens=100)
    
    # Concurrent requests
    responses = client.generate_batch(
        prompts=["Hello", "World"],
        max_tokens=100,
        concurrent=True
    )
"""

import argparse
import queue
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Dict, Any
import threading

import numpy as np

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException, np_to_triton_dtype
except ImportError:
    print("ERROR: tritonclient not installed. Install with: pip install tritonclient[grpc]")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Install with: pip install transformers")
    sys.exit(1)


@dataclass
class GenerationResult:
    """Result from a single generation request."""
    request_id: str
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: Optional[float] = None  # Time to first token
    error: Optional[str] = None


class UserData:
    """Container for callback data."""
    def __init__(self):
        self._completed_requests = queue.Queue()
        self._start_times: Dict[str, float] = {}
        self._first_token_times: Dict[str, float] = {}
        self._results: Dict[str, Any] = {}


def prepare_tensor(name: str, data: np.ndarray) -> grpcclient.InferInput:
    """Prepare a tensor for Triton inference."""
    tensor = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    tensor.set_data_from_numpy(data)
    return tensor


class TritonLLMClient:
    """
    High-level client for TensorRT-LLM inference via Triton Server.
    
    This client handles tokenization, request preparation, and response parsing
    for TensorRT-LLM models served via Triton Inference Server.
    """
    
    def __init__(
        self,
        url: str = "localhost:8001",
        tokenizer_dir: str = None,
        model_name: str = "tensorrt_llm",
        verbose: bool = False,
        ssl: bool = False,
        stream_timeout: Optional[float] = None,
    ):
        """
        Initialize the Triton LLM client.
        
        Args:
            url: Triton server URL (host:port for gRPC)
            tokenizer_dir: Path to tokenizer/HuggingFace model directory
            model_name: Name of the TensorRT-LLM model in Triton
            verbose: Enable verbose output
            ssl: Use SSL for connection
            stream_timeout: Timeout for streaming requests
        """
        self.url = url
        self.model_name = model_name
        self.verbose = verbose
        self.ssl = ssl
        self.stream_timeout = stream_timeout
        
        # Load tokenizer
        if tokenizer_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                legacy=False,
                padding_side='left',
                trust_remote_code=True
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pad_id = self.tokenizer.encode(
                self.tokenizer.pad_token, add_special_tokens=False
            )[0]
            self.end_id = self.tokenizer.encode(
                self.tokenizer.eos_token, add_special_tokens=False
            )[0]
        else:
            self.tokenizer = None
            self.pad_id = 0
            self.end_id = 1
        
        # Create Triton client
        self.client = grpcclient.InferenceServerClient(
            url=self.url,
            verbose=self.verbose,
            ssl=self.ssl,
        )
    
    def _prepare_inputs(
        self,
        input_ids: np.ndarray,
        input_length: int,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.0,
        beam_width: int = 1,
        streaming: bool = False,
    ) -> List[grpcclient.InferInput]:
        """Prepare input tensors for inference."""
        inputs = [
            prepare_tensor("input_ids", input_ids),
            prepare_tensor("input_lengths", np.array([[input_length]], dtype=np.int32)),
            prepare_tensor("request_output_len", np.array([[max_tokens]], dtype=np.int32)),
            prepare_tensor("beam_width", np.array([[beam_width]], dtype=np.int32)),
            prepare_tensor("temperature", np.array([[temperature]], dtype=np.float32)),
            prepare_tensor("streaming", np.array([[streaming]], dtype=bool)),
            prepare_tensor("end_id", np.array([[self.end_id]], dtype=np.int32)),
            prepare_tensor("pad_id", np.array([[self.pad_id]], dtype=np.int32)),
            prepare_tensor("runtime_top_k", np.array([[top_k]], dtype=np.int32)),
            prepare_tensor("runtime_top_p", np.array([[top_p]], dtype=np.float32)),
        ]
        return inputs
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.0,
        streaming: bool = False,
    ) -> GenerationResult:
        """
        Generate text for a single prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            streaming: Use streaming mode
            
        Returns:
            GenerationResult with output text and metrics
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Provide tokenizer_dir.")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        input_ids_np = np.array([input_ids], dtype=np.int32)
        input_length = len(input_ids)
        
        # Prepare inputs
        inputs = self._prepare_inputs(
            input_ids_np,
            input_length,
            max_tokens,
            temperature,
            top_k,
            top_p,
            streaming=streaming,
        )
        
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.perf_counter()
        
        if streaming:
            return self._generate_streaming(
                prompt, inputs, input_ids, input_length, request_id, start_time
            )
        else:
            return self._generate_non_streaming(
                prompt, inputs, input_ids, input_length, request_id, start_time
            )
    
    def _generate_non_streaming(
        self,
        prompt: str,
        inputs: List[grpcclient.InferInput],
        input_ids: List[int],
        input_length: int,
        request_id: str,
        start_time: float,
    ) -> GenerationResult:
        """Non-streaming generation."""
        try:
            result = self.client.infer(
                self.model_name,
                inputs,
                request_id=request_id,
            )
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Parse output
            output_ids = result.as_numpy('output_ids')
            if output_ids is not None:
                output_ids_list = list(output_ids[0][0])
                # Remove input tokens from output
                generated_ids = output_ids_list[input_length:]
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                output_tokens = len(generated_ids)
            else:
                output_text = ""
                output_tokens = 0
            
            return GenerationResult(
                request_id=request_id,
                input_text=prompt,
                output_text=output_text,
                input_tokens=input_length,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            
        except InferenceServerException as e:
            return GenerationResult(
                request_id=request_id,
                input_text=prompt,
                output_text="",
                input_tokens=input_length,
                output_tokens=0,
                latency_ms=0,
                error=str(e),
            )
    
    def _generate_streaming(
        self,
        prompt: str,
        inputs: List[grpcclient.InferInput],
        input_ids: List[int],
        input_length: int,
        request_id: str,
        start_time: float,
    ) -> GenerationResult:
        """Streaming generation with TTFT measurement."""
        user_data = UserData()
        user_data._start_times[request_id] = start_time
        
        def callback(user_data, result, error):
            if error:
                user_data._completed_requests.put((None, error))
            else:
                # Record first token time
                if request_id not in user_data._first_token_times:
                    user_data._first_token_times[request_id] = time.perf_counter()
                user_data._completed_requests.put((result, None))
        
        try:
            self.client.start_stream(
                callback=partial(callback, user_data),
                stream_timeout=self.stream_timeout,
            )
            
            self.client.async_stream_infer(
                self.model_name,
                inputs,
                request_id=request_id,
            )
            
            self.client.stop_stream()
            
            # Collect results
            output_ids_list = list(input_ids)  # Start with input
            error_msg = None
            
            while True:
                try:
                    result, error = user_data._completed_requests.get(block=False)
                except queue.Empty:
                    break
                
                if error:
                    error_msg = str(error)
                    break
                
                if result is not None:
                    output_ids = result.as_numpy('output_ids')
                    if output_ids is not None:
                        tokens = list(output_ids[0][0])
                        output_ids_list.extend(tokens)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Calculate TTFT
            ttft_ms = None
            if request_id in user_data._first_token_times:
                ttft_ms = (user_data._first_token_times[request_id] - start_time) * 1000
            
            # Decode output
            generated_ids = output_ids_list[input_length:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return GenerationResult(
                request_id=request_id,
                input_text=prompt,
                output_text=output_text,
                input_tokens=input_length,
                output_tokens=len(generated_ids),
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                error=error_msg,
            )
            
        except Exception as e:
            return GenerationResult(
                request_id=request_id,
                input_text=prompt,
                output_text="",
                input_tokens=input_length,
                output_tokens=0,
                latency_ms=0,
                error=str(e),
            )
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.0,
        concurrent: bool = True,
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            concurrent: Send requests concurrently (True) or sequentially (False)
            
        Returns:
            List of GenerationResult objects
        """
        if not concurrent:
            # Sequential processing
            return [
                self.generate(prompt, max_tokens, temperature, top_k, top_p)
                for prompt in prompts
            ]
        
        # Concurrent processing using async inference
        user_data = UserData()
        results_dict: Dict[str, GenerationResult] = {}
        
        def callback(user_data, result, error):
            if error:
                user_data._completed_requests.put((None, error, None))
            else:
                try:
                    req_id = result.get_response().id
                except:
                    req_id = "unknown"
                user_data._completed_requests.put((result, None, req_id))
        
        # Prepare and send all requests
        request_data = []
        for i, prompt in enumerate(prompts):
            input_ids = self.tokenizer.encode(prompt)
            input_ids_np = np.array([input_ids], dtype=np.int32)
            input_length = len(input_ids)
            
            inputs = self._prepare_inputs(
                input_ids_np,
                input_length,
                max_tokens,
                temperature,
                top_k,
                top_p,
                streaming=False,
            )
            
            request_id = f"req_{i}_{int(time.time() * 1000)}"
            user_data._start_times[request_id] = time.perf_counter()
            
            request_data.append({
                'request_id': request_id,
                'prompt': prompt,
                'input_ids': input_ids,
                'input_length': input_length,
                'inputs': inputs,
            })
        
        # Send all requests asynchronously
        async_requests = []
        for req in request_data:
            async_req = self.client.async_infer(
                self.model_name,
                req['inputs'],
                callback=partial(callback, user_data),
                request_id=req['request_id'],
            )
            async_requests.append(async_req)
        
        # Wait for all responses
        for _ in range(len(prompts)):
            try:
                result, error, req_id = user_data._completed_requests.get(timeout=300)
            except queue.Empty:
                break
            
            if error:
                # Find the request that failed
                for req in request_data:
                    if req['request_id'] not in results_dict:
                        results_dict[req['request_id']] = GenerationResult(
                            request_id=req['request_id'],
                            input_text=req['prompt'],
                            output_text="",
                            input_tokens=req['input_length'],
                            output_tokens=0,
                            latency_ms=0,
                            error=str(error),
                        )
                        break
            else:
                # Find matching request
                for req in request_data:
                    if req['request_id'] == req_id:
                        end_time = time.perf_counter()
                        latency_ms = (end_time - user_data._start_times[req_id]) * 1000
                        
                        output_ids = result.as_numpy('output_ids')
                        if output_ids is not None:
                            output_ids_list = list(output_ids[0][0])
                            generated_ids = output_ids_list[req['input_length']:]
                            output_text = self.tokenizer.decode(
                                generated_ids, skip_special_tokens=True
                            )
                            output_tokens = len(generated_ids)
                        else:
                            output_text = ""
                            output_tokens = 0
                        
                        results_dict[req_id] = GenerationResult(
                            request_id=req_id,
                            input_text=req['prompt'],
                            output_text=output_text,
                            input_tokens=req['input_length'],
                            output_tokens=output_tokens,
                            latency_ms=latency_ms,
                        )
                        break
        
        # Return results in order
        results = []
        for req in request_data:
            if req['request_id'] in results_dict:
                results.append(results_dict[req['request_id']])
            else:
                results.append(GenerationResult(
                    request_id=req['request_id'],
                    input_text=req['prompt'],
                    output_text="",
                    input_tokens=req['input_length'],
                    output_tokens=0,
                    latency_ms=0,
                    error="No response received",
                ))
        
        return results
    
    def is_server_ready(self) -> bool:
        """Check if the Triton server is ready."""
        try:
            return self.client.is_server_ready()
        except Exception:
            return False
    
    def is_model_ready(self) -> bool:
        """Check if the model is ready for inference."""
        try:
            return self.client.is_model_ready(self.model_name)
        except Exception:
            return False


def main():
    """CLI interface for testing the Triton client."""
    parser = argparse.ArgumentParser(description="Triton LLM Client")
    parser.add_argument(
        "-u", "--url",
        type=str,
        default="localhost:8001",
        help="Triton server URL (default: localhost:8001)"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tensorrt_llm",
        help="Model name in Triton (default: tensorrt_llm)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    client = TritonLLMClient(
        url=args.url,
        tokenizer_dir=args.tokenizer_dir,
        model_name=args.model_name,
        verbose=args.verbose,
    )
    
    print(f"Server ready: {client.is_server_ready()}")
    print(f"Model ready: {client.is_model_ready()}")
    
    if client.is_model_ready():
        print(f"\nGenerating response for: {args.prompt}")
        result = client.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            streaming=args.streaming,
        )
        
        print(f"\nResult:")
        print(f"  Input: {result.input_text}")
        print(f"  Output: {result.output_text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        print(f"  Latency: {result.latency_ms:.2f} ms")
        if result.ttft_ms:
            print(f"  TTFT: {result.ttft_ms:.2f} ms")
        if result.error:
            print(f"  Error: {result.error}")


if __name__ == "__main__":
    main()
