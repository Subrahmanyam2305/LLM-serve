#!/usr/bin/env python3
"""
Triton gRPC Client for TensorRT-LLM Inference.

This module provides a high-level wrapper around the Triton Inference Server gRPC client
for sending concurrent requests to TensorRT-LLM models. It supports streaming inference
with real-time token output, which is required for models with decoupled transaction policy.

Based on the inflight_batcher_llm_client.py from TensorRT-LLM triton_backend.

Usage:
    from triton_client.triton_grpc_client import TritonLLMClient
    
    client = TritonLLMClient(
        url="localhost:8001",
        tokenizer_dir="/path/to/tokenizer"
    )
    
    # Single request with streaming (recommended for production)
    def token_callback(token_text, is_first, is_last):
        print(token_text, end="", flush=True)
        
    response = client.generate(
        "Hello, how are you?", 
        max_tokens=100,
        callback=token_callback
    )
    
    # Multiple concurrent requests with streaming
    responses = client.generate_batch(
        prompts=["Hello", "World"],
        max_tokens=100,
        concurrent=True,
        callback=token_callback
    )

Command-line usage:
    # Single request with real-time streaming
    python triton_grpc_client.py --url localhost:8001 --tokenizer-dir /path/to/tokenizer \
        --prompt "Hello, how are you?" --real-time
    
    # Multiple prompts with concurrent processing
    python triton_grpc_client.py --url localhost:8001 --tokenizer-dir /path/to/tokenizer \
        --prompts "Hello" "How are you?" --concurrent --real-time
    
    # Multiple prompts from file with real-time output
    python triton_grpc_client.py --url localhost:8001 --tokenizer-dir /path/to/tokenizer \
        --prompts-file prompts.txt --concurrent --real-time
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
        streaming: bool = True,  # Default to streaming for production use
        callback=None,  # Callback function for streaming tokens
    ) -> GenerationResult:
        """
        Generate text for a single prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            streaming: Use streaming mode (required for decoupled models)
            callback: Optional callback function for streaming tokens
                      Function signature: callback(token_text, is_first, is_last)
            
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
                prompt, inputs, input_ids, input_length, request_id, start_time, callback
            )
        else:
            # For production use with TensorRT-LLM, non-streaming mode may not work
            # due to decoupled transaction policy
            try:
                return self._generate_non_streaming(
                    prompt, inputs, input_ids, input_length, request_id, start_time
                )
            except Exception as e:
                if "decoupled transaction policy" in str(e):
                    print("WARNING: Non-streaming mode failed due to decoupled transaction policy.")
                    print("Falling back to streaming mode...")
                    # Set streaming flag in inputs
                    for inp in inputs:
                        if inp.name() == "streaming":
                            inp.set_data_from_numpy(np.array([[True]], dtype=bool))
                    return self._generate_streaming(
                        prompt, inputs, input_ids, input_length, request_id, start_time, callback
                    )
                else:
                    raise
    
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
        callback=None,  # Callback function for streaming tokens
    ) -> GenerationResult:
        """
        Streaming generation with TTFT measurement and real-time token callbacks.
        
        Args:
            prompt: Input text prompt
            inputs: Prepared input tensors
            input_ids: Tokenized input
            input_length: Length of input in tokens
            request_id: Unique request ID
            start_time: Start time for latency measurement
            callback: Optional callback function for streaming tokens
                      Function signature: callback(token_text, is_first, is_last)
        
        Returns:
            GenerationResult with output text and metrics
        """
        user_data = UserData()
        user_data._start_times[request_id] = start_time
        
        # Store all generated tokens
        all_tokens = []
        full_output_text = ""
        is_first_token = True
        
        def stream_callback(user_data, result, error):
            nonlocal is_first_token, full_output_text
            
            if error:
                user_data._completed_requests.put((None, error))
                return
                
            # Record first token time
            if request_id not in user_data._first_token_times:
                user_data._first_token_times[request_id] = time.perf_counter()
            
            # Process the token
            try:
                output_ids = result.as_numpy('output_ids')
                if output_ids is not None:
                    # Get the new token(s)
                    new_tokens = list(output_ids[0][0])
                    
                    # Skip input tokens in the first response
                    if is_first_token and len(new_tokens) > input_length:
                        new_tokens = new_tokens[input_length:]
                    
                    # Decode the new token(s)
                    if new_tokens:
                        token_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        all_tokens.extend(new_tokens)
                        
                        # Update the full output text
                        full_output_text += token_text
                        
                        # Call the callback if provided
                        if callback:
                            is_last = False  # We don't know if this is the last token yet
                            # Check if callback accepts prompt_index parameter
                            import inspect
                            callback_params = inspect.signature(callback).parameters
                            if 'prompt_index' in callback_params:
                                callback(token_text, is_first_token, is_last, prompt_index=0)
                            else:
                                callback(token_text, is_first_token, is_last)
                            
                        is_first_token = False
            except Exception as e:
                print(f"Error processing token: {e}")
            
            user_data._completed_requests.put((result, None))
        
        try:
            self.client.start_stream(
                callback=partial(stream_callback, user_data),
                stream_timeout=self.stream_timeout,
            )
            
            self.client.async_stream_infer(
                self.model_name,
                inputs,
                request_id=request_id,
            )
            
            # Wait for results with timeout
            timeout = 60  # seconds
            start_wait = time.perf_counter()
            got_response = False
            error_msg = None
            
            while time.perf_counter() - start_wait < timeout:
                try:
                    result, error = user_data._completed_requests.get(block=True, timeout=1.0)
                    got_response = True
                    
                    if error:
                        error_msg = str(error)
                        break
                except queue.Empty:
                    # Check if we've received any response yet
                    if got_response:
                        # We got some responses but now the queue is empty, likely done
                        # Call callback with is_last=True if provided
                        if callback:
                            # Check if callback accepts prompt_index parameter
                            import inspect
                            callback_params = inspect.signature(callback).parameters
                            if 'prompt_index' in callback_params:
                                callback("", False, True, prompt_index=0)
                            else:
                                callback("", False, True)
                        break
                    # Otherwise keep waiting
                    continue
            
            # Stop the stream after collecting results
            self.client.stop_stream()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Calculate TTFT
            ttft_ms = None
            if request_id in user_data._first_token_times:
                ttft_ms = (user_data._first_token_times[request_id] - start_time) * 1000
            
            return GenerationResult(
                request_id=request_id,
                input_text=prompt,
                output_text=full_output_text,
                input_tokens=input_length,
                output_tokens=len(all_tokens),
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                error=error_msg,
            )
            
        except Exception as e:
            # Try to stop the stream if it was started
            try:
                self.client.stop_stream()
            except:
                pass
                
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
        streaming: bool = True,  # Default to streaming for production use
        callback=None,  # Callback function for streaming tokens
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
            streaming: Use streaming mode (required for decoupled models)
            callback: Optional callback function for streaming tokens
            
        Returns:
            List of GenerationResult objects
        """
        # For production use with Triton TensorRT-LLM, streaming is required
        # due to the decoupled transaction policy
        
        if not concurrent:
            # Sequential processing - one prompt at a time
            return [
                self.generate(
                    prompt, 
                    max_tokens, 
                    temperature, 
                    top_k, 
                    top_p, 
                    streaming=streaming,
                    callback=callback
                )
                for prompt in prompts
            ]
        
        # Concurrent processing - process all prompts in parallel
        # Note: The Triton client doesn't support multiple concurrent streams
        # So we need to create separate client instances for each prompt
        results = []
        
        # Create a list of client instances
        clients = []
        for _ in range(len(prompts)):
            client = TritonLLMClient(
                url=self.url,
                tokenizer_dir=None,  # We'll share the tokenizer
                model_name=self.model_name,
                verbose=self.verbose,
                ssl=self.ssl,
                stream_timeout=self.stream_timeout,
            )
            # Share the tokenizer to avoid loading it multiple times
            client.tokenizer = self.tokenizer
            client.pad_id = self.pad_id
            client.end_id = self.end_id
            clients.append(client)
        
        # Process prompts in parallel using threads
        threads = []
        results = [None] * len(prompts)
        
        def process_prompt(client, prompt, index):
            try:
                # Create a wrapper callback that adds the prompt index
                if callback:
                    def indexed_callback(token_text, is_first, is_last):
                        callback(token_text, is_first, is_last, prompt_index=index)
                else:
                    indexed_callback = None
                    
                result = client.generate(
                    prompt, 
                    max_tokens, 
                    temperature, 
                    top_k, 
                    top_p, 
                    streaming=streaming,
                    callback=indexed_callback
                )
                results[index] = result
            except Exception as e:
                results[index] = GenerationResult(
                    request_id=f"req_{index}",
                    input_text=prompt,
                    output_text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    error=str(e),
                )
        
        # Create and start threads for each prompt
        for i, prompt in enumerate(prompts):
            thread = threading.Thread(
                target=process_prompt, 
                args=(clients[i], prompt, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
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
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Single input prompt"
    )
    input_group.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing one prompt per line"
    )
    input_group.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Multiple prompts provided as command line arguments"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling parameter (default: 1)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p sampling parameter (default: 0.0)"
    )
    
    # Request handling options
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (not recommended for production)"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Process requests concurrently (only applies when multiple prompts are provided)"
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Display tokens in real-time as they are generated"
    )
    parser.add_argument(
        "--output-files",
        action="store_true",
        help="Write outputs to separate files for each prompt"
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show token boundaries with visual indicators in real-time mode"
    )
    parser.add_argument(
        "--token-delay",
        type=float,
        default=0.0,
        help="Add a delay (in seconds) between tokens to make streaming more visible"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = TritonLLMClient(
        url=args.url,
        tokenizer_dir=args.tokenizer_dir,
        model_name=args.model_name,
        verbose=False,
    )
    
    print(f"Server ready: {client.is_server_ready()}")
    print(f"Model ready: {client.is_model_ready()}")
    
    if not client.is_model_ready():
        print(f"ERROR: Model {args.model_name} is not ready")
        sys.exit(1)
    
    # Collect prompts from the specified source
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        try:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"ERROR: Failed to read prompts file: {e}")
            sys.exit(1)
    
    # Determine streaming mode
    streaming = not args.no_streaming
    
    print(f"\nProcessing {len(prompts)} prompt(s) with {'streaming' if streaming else 'non-streaming'} mode")
    print(f"Concurrent: {args.concurrent}, Real-time: {args.real_time}")
    
    # Define callback for real-time token display
    if args.real_time:
        # Create a lock for thread-safe printing
        print_lock = threading.Lock()
        
        # Create separate output buffers and state for each prompt
        output_buffers = [""] * len(prompts)
        prompt_headers_printed = [False] * len(prompts)
        
        # Track which prompt is currently being displayed
        current_prompt_index = [0]
        
        def token_callback(token_text, is_first, is_last, prompt_index=0):
            with print_lock:
                # Add to the buffer
                if token_text:
                    output_buffers[prompt_index] += token_text
                
                # Only print if this is the first token or if we're continuing with the same prompt
                if is_first or current_prompt_index[0] == prompt_index:
                    current_prompt_index[0] = prompt_index
                    
                    # Print header if needed
                    if is_first:
                        print(f"\n\n{'='*20} PROMPT {prompt_index+1} {'='*20}")
                        print(f"Input: {prompts[prompt_index]}")
                        print(f"\nGenerating: ", end="", flush=True)
                        prompt_headers_printed[prompt_index] = True
                    
                    # Print the token
                    if token_text:
                        if args.show_tokens:
                            print(f"[{token_text}]", end="", flush=True)
                        else:
                            print(token_text, end="", flush=True)
                
                # Add a delay if specified
                if args.token_delay > 0 and token_text:
                    time.sleep(args.token_delay)
                
                # Print newline at the end
                if is_last:
                    print("\n")
    else:
        token_callback = None
    
    # Process prompts
    results = []
    
    if args.real_time and args.concurrent and len(prompts) > 1:
        # For real-time concurrent display, we need to handle each prompt separately
        # Create a thread for each prompt
        threads = []
        results = [None] * len(prompts)
        
        # Create a lock for thread-safe printing
        print_lock = threading.Lock()
        
        # Create output files if requested
        output_files = []
        if args.output_files:
            for i in range(len(prompts)):
                output_file = f"prompt_{i+1}_output.txt"
                with open(output_file, 'w') as f:
                    f.write(f"Prompt {i+1}: {prompts[i]}\n\nGenerating: ")
                output_files.append(output_file)
                
            print("\nOutputs will be written to separate files:")
            for i, file in enumerate(output_files):
                print(f"  Prompt {i+1}: {file}")
        
        # Create buffers for each prompt's output
        output_buffers = [""] * len(prompts)
        prompt_started = [False] * len(prompts)
        prompt_completed = [False] * len(prompts)
        
        # Track which prompt was last printed to avoid unnecessary line breaks
        last_printed_prompt = [-1]  # Use list to allow modification in nested function
        
        # Print initial headers for each prompt
        print("\n" + "="*70)
        print("CONCURRENT STREAMING OUTPUT")
        print("="*70)
        for i, prompt in enumerate(prompts):
            # Truncate long prompts for display
            display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
            print(f"[P{i+1}] Input: {display_prompt}")
        print("-"*70)
        print("Streaming outputs (each line prefixed with prompt number):")
        print("-"*70, flush=True)
        
        def process_prompt(prompt_index):
            prompt = prompts[prompt_index]
            
            # Create a client for this prompt
            local_client = TritonLLMClient(
                url=args.url,
                tokenizer_dir=args.tokenizer_dir,
                model_name=args.model_name,
                verbose=False,
            )
            local_client.tokenizer = client.tokenizer
            local_client.pad_id = client.pad_id
            local_client.end_id = client.end_id
            
            # Define a callback for this prompt
            def prompt_callback(token_text, is_first, is_last):
                nonlocal output_buffers, prompt_started, prompt_completed, last_printed_prompt
                
                # Write to file if requested
                if args.output_files and token_text:
                    with open(output_files[prompt_index], 'a') as f:
                        f.write(token_text)
                
                # Update the buffer
                if token_text:
                    output_buffers[prompt_index] += token_text
                
                # Print tokens with prompt identifier prefix
                # Each prompt gets its own line - when switching prompts, start a new line
                with print_lock:
                    # Check if we need to switch to a new line (different prompt)
                    if last_printed_prompt[0] != prompt_index:
                        if last_printed_prompt[0] != -1:
                            # End the previous line
                            print()
                        # Start new line with prompt prefix
                        print(f"[P{prompt_index+1}]", end="", flush=True)
                        last_printed_prompt[0] = prompt_index
                        prompt_started[prompt_index] = True
                    
                    if token_text:
                        # Print token (handle newlines by adding prompt prefix after each)
                        if '\n' in token_text:
                            parts = token_text.split('\n')
                            for j, part in enumerate(parts):
                                if j > 0:
                                    # After a newline, print the prompt prefix
                                    print()
                                    print(f"[P{prompt_index+1}]", end="", flush=True)
                                if args.show_tokens:
                                    print(f"[{part}]", end="", flush=True)
                                else:
                                    print(part, end="", flush=True)
                        else:
                            if args.show_tokens:
                                print(f"[{token_text}]", end="", flush=True)
                            else:
                                print(token_text, end="", flush=True)
                
                # Add a delay if specified
                if args.token_delay > 0 and token_text:
                    time.sleep(args.token_delay)
                
                # Mark completion
                if is_last:
                    prompt_completed[prompt_index] = True
                    with print_lock:
                        # Make sure we're on the right line
                        if last_printed_prompt[0] != prompt_index:
                            print()
                            print(f"[P{prompt_index+1}]", end="", flush=True)
                            last_printed_prompt[0] = prompt_index
                        print(f" [DONE]", end="", flush=True)
                    # Write completion message to file
                    if args.output_files:
                        with open(output_files[prompt_index], 'a') as f:
                            f.write("\n\nGeneration complete.")
            
            # Generate text for this prompt
            result = local_client.generate(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                streaming=True,  # Always use streaming for real-time display
                callback=prompt_callback,
            )
            
            results[prompt_index] = result
        
        # Create and start threads for each prompt
        for i in range(len(prompts)):
            thread = threading.Thread(target=process_prompt, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Print completion message
        print("\n" + "-"*70)
        print("All prompts completed.")
        print("-"*70)
        
        # Print full outputs for each prompt
        print("\nFull outputs:")
        for i, output in enumerate(output_buffers):
            print(f"\n[P{i+1}] Full output:")
            print(f"  {output}")
        
        if args.output_files:
            print("\nCheck the output files for full results.")
    else:
        # Use the standard approach for non-real-time or single prompt
        if len(prompts) == 1:
            # Single prompt processing
            print(f"\nGenerating response for: {prompts[0]}")
            # For single prompt, we can use the callback directly with prompt_index=0
            single_prompt_callback = token_callback
            if args.real_time:
                def single_prompt_callback(token_text, is_first, is_last):
                    token_callback(token_text, is_first, is_last, prompt_index=0)
                    
            result = client.generate(
                prompts[0],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                streaming=streaming,
                callback=single_prompt_callback,
            )
            results.append(result)
        else:
            # Multiple prompts processing
            print(f"\nGenerating responses for {len(prompts)} prompts...")
            batch_results = client.generate_batch(
                prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                concurrent=args.concurrent,
                streaming=streaming,
                callback=token_callback,
            )
            results.extend(batch_results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {len(results)} prompt(s):")
    print(f"{'='*60}")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Input: {result.input_text}")
        if not args.real_time:  # Only print output if not already displayed in real-time
            print(f"  Output: {result.output_text}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        print(f"  Latency: {result.latency_ms:.2f} ms")
        if result.ttft_ms:
            print(f"  TTFT: {result.ttft_ms:.2f} ms")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  {'-'*40}")  # Add separator between results
            
    # Print summary
    total_latency = sum(r.latency_ms for r in results)
    avg_latency = total_latency / len(results) if results else 0
    total_tokens = sum(r.output_tokens for r in results)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total prompts: {len(results)}")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Average latency: {avg_latency:.2f} ms")
    if total_latency > 0:
        print(f"  Throughput: {total_tokens / (total_latency / 1000):.2f} tokens/sec")
    else:
        print(f"  Throughput: N/A (no successful responses)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
