#!/usr/bin/env python3
"""
OpenAI-compatible API wrapper for Triton TensorRT-LLM server.

This server provides an OpenAI-compatible HTTP API that translates requests
to Triton gRPC calls, enabling use with tools like genai-bench.

Usage:
    python servers/triton_openai_wrapper.py \
        --triton-url localhost:8001 \
        --tokenizer-dir /path/to/tokenizer \
        --port 8080
"""

import argparse
import json
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, List, Dict, Any
import queue

import numpy as np

# Triton client
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype, InferenceServerException
except ImportError:
    print("ERROR: tritonclient not installed. Run: pip install tritonclient[grpc]")
    exit(1)

# Tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    exit(1)


# Global state
tokenizer = None
model_name = "tensorrt_llm"
args = None


def get_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def prepare_tensor(name, data):
    """Prepare a tensor for Triton inference."""
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


def prepare_triton_inputs(
    input_ids: List[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    streaming: bool = True,
) -> List[grpcclient.InferInput]:
    """Prepare inputs for Triton inference."""
    
    inputs = []
    
    # input_ids
    input_ids_data = np.array([input_ids], dtype=np.int32)
    inputs.append(prepare_tensor("input_ids", input_ids_data))
    
    # input_lengths
    input_lengths_data = np.array([[len(input_ids)]], dtype=np.int32)
    inputs.append(prepare_tensor("input_lengths", input_lengths_data))
    
    # request_output_len
    output_len_data = np.array([[max_tokens]], dtype=np.int32)
    inputs.append(prepare_tensor("request_output_len", output_len_data))
    
    # beam_width
    beam_width_data = np.array([[1]], dtype=np.int32)
    inputs.append(prepare_tensor("beam_width", beam_width_data))
    
    # temperature
    temp_data = np.array([[max(temperature, 0.01)]], dtype=np.float32)
    inputs.append(prepare_tensor("temperature", temp_data))
    
    # runtime_top_p
    top_p_data = np.array([[top_p]], dtype=np.float32)
    inputs.append(prepare_tensor("runtime_top_p", top_p_data))
    
    # end_id
    end_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    end_id_data = np.array([[end_id]], dtype=np.int32)
    inputs.append(prepare_tensor("end_id", end_id_data))
    
    # pad_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_id
    pad_id_data = np.array([[pad_id]], dtype=np.int32)
    inputs.append(prepare_tensor("pad_id", pad_id_data))
    
    # streaming - always True for decoupled mode
    streaming_data = np.array([[streaming]], dtype=bool)
    inputs.append(prepare_tensor("streaming", streaming_data))
    
    return inputs


class UserData:
    """Container for callback results."""
    def __init__(self):
        self._completed_requests = queue.Queue()


def streaming_callback(user_data, result, error):
    """Callback for streaming responses."""
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class OpenAIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""
    
    # Use HTTP/1.1 for proper streaming support
    protocol_version = "HTTP/1.1"
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def send_json_response(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/v1/models":
            self.send_json_response({
                "object": "list",
                "data": [{
                    "id": Path(args.tokenizer_dir).name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "triton"
                }]
            })
        elif self.path == "/health" or self.path == "/":
            self.send_json_response({"status": "ok"})
        else:
            self.send_json_response({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/v1/chat/completions":
            self.handle_chat_completions()
        else:
            self.send_json_response({"error": "Not found"}, 404)
    
    def handle_chat_completions(self):
        """Handle chat completions request."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request = json.loads(body)
            
            # Extract parameters
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 128)
            temperature = request.get("temperature", 1.0)
            top_p = request.get("top_p", 1.0)
            stream = request.get("stream", False)
            
            # Format messages for the model
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback: simple concatenation
                prompt = "\n".join([
                    f"{m['role']}: {m['content']}" for m in messages
                ])
                prompt += "\nassistant:"
            
            # Tokenize
            input_ids = tokenizer.encode(prompt)
            input_len = len(input_ids)
            
            if stream:
                self.handle_streaming_response(input_ids, max_tokens, temperature, top_p)
            else:
                self.handle_non_streaming_response(input_ids, max_tokens, temperature, top_p)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json_response({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }, 500)
    
    def handle_non_streaming_response(
        self, 
        input_ids: List[int], 
        max_tokens: int,
        temperature: float,
        top_p: float
    ):
        """Handle non-streaming response using streaming API internally."""
        request_id = get_request_id()
        input_len = len(input_ids)
        
        # Create new client for this request
        triton_client = grpcclient.InferenceServerClient(
            url=args.triton_url,
            verbose=False,
        )
        
        # Prepare inputs
        inputs = prepare_triton_inputs(
            input_ids, max_tokens, temperature, top_p, streaming=True
        )
        
        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("output_ids"),
            grpcclient.InferRequestedOutput("sequence_length"),
        ]
        
        # Create user data for callback
        user_data = UserData()
        
        # Start streaming
        triton_client.start_stream(
            callback=partial(streaming_callback, user_data),
            stream_timeout=120,
        )
        
        # Send request
        triton_client.async_stream_infer(
            model_name,
            inputs,
            outputs=outputs,
            request_id=request_id,
        )
        
        # Wait for completion and stop stream
        import time
        time.sleep(0.1)  # Give some time for initial responses
        
        # Collect all tokens
        all_tokens = []
        timeout_count = 0
        max_timeout = 60  # seconds
        generation_complete = False
        
        while timeout_count < max_timeout and not generation_complete:
            try:
                result = user_data._completed_requests.get(timeout=0.5)
                timeout_count = 0  # Reset on successful get
            except queue.Empty:
                timeout_count += 1
                # Check if we have tokens and no more coming
                if all_tokens and timeout_count > 1:
                    break
                continue
            
            if isinstance(result, InferenceServerException):
                raise Exception(str(result))
            
            # Get output tokens
            output_ids = result.as_numpy("output_ids")
            if output_ids is not None:
                tokens = list(output_ids[0][0])
                # Empty token list signals end of generation
                if not tokens:
                    generation_complete = True
                    break
                all_tokens.extend(tokens)
                
                # Check if we've hit max_tokens
                if len(all_tokens) >= max_tokens:
                    generation_complete = True
        
        # Stop stream
        triton_client.stop_stream()
        
        # Drain any remaining results
        while True:
            try:
                result = user_data._completed_requests.get(block=False)
                if isinstance(result, InferenceServerException):
                    continue
                output_ids = result.as_numpy("output_ids")
                if output_ids is not None:
                    tokens = list(output_ids[0][0])
                    all_tokens.extend(tokens)
            except queue.Empty:
                break
        
        if not all_tokens:
            raise Exception("No output received from model")
        
        # Decode tokens
        generated_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
        
        # Build response
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": Path(args.tokenizer_dir).name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(all_tokens),
                "total_tokens": input_len + len(all_tokens)
            }
        }
        
        self.send_json_response(response)
    
    def handle_streaming_response(
        self,
        input_ids: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float
    ):
        """Handle streaming response with true token-by-token streaming."""
        request_id = get_request_id()
        input_len = len(input_ids)
        created_time = int(time.time())
        
        # Send headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        
        triton_client = None
        try:
            # Create new client for this request
            triton_client = grpcclient.InferenceServerClient(
                url=args.triton_url,
                verbose=False,
            )
            
            # Prepare inputs
            inputs = prepare_triton_inputs(
                input_ids, max_tokens, temperature, top_p, streaming=True
            )
            
            # Prepare outputs
            outputs = [
                grpcclient.InferRequestedOutput("output_ids"),
                grpcclient.InferRequestedOutput("sequence_length"),
            ]
            
            # Create user data for callback
            user_data = UserData()
            
            # Start streaming
            triton_client.start_stream(
                callback=partial(streaming_callback, user_data),
                stream_timeout=120,
            )
            
            # Send request
            triton_client.async_stream_infer(
                model_name,
                inputs,
                outputs=outputs,
                request_id=request_id,
            )
            
            # Process streaming results
            # Note: Triton TensorRT-LLM sends one token per response (not including input tokens)
            all_output_tokens = []
            generation_complete = False
            start_time = time.time()
            timeout_count = 0
            
            while not generation_complete and (time.time() - start_time) < 60:
                try:
                    result = user_data._completed_requests.get(timeout=0.5)
                    timeout_count = 0  # Reset on successful get
                except queue.Empty:
                    timeout_count += 1
                    # If we have output tokens and hit timeout, assume done
                    if all_output_tokens and timeout_count > 2:
                        generation_complete = True
                        break
                    continue
                
                if isinstance(result, InferenceServerException):
                    raise Exception(str(result))
                
                # Get output tokens
                output_ids = result.as_numpy("output_ids")
                if output_ids is None:
                    continue
                
                tokens = list(output_ids[0][0])
                
                # Empty token list signals end of generation
                if not tokens:
                    generation_complete = True
                    break
                
                # Each response contains just the new token(s), not input tokens
                all_output_tokens.extend(tokens)
                
                # Decode new tokens and send
                new_text = tokenizer.decode(tokens, skip_special_tokens=True)
                
                if new_text:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": Path(args.tokenizer_dir).name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None
                        }]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                
                # Check if we've hit max_tokens - stop immediately
                if len(all_output_tokens) >= max_tokens:
                    generation_complete = True
                    break
            
            # Stop the stream immediately
            if triton_client:
                try:
                    triton_client.stop_stream(cancel_requests=True)
                except:
                    pass
            
            # Send final chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": Path(args.tokenizer_dir).name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            
        except BrokenPipeError:
            pass  # Client disconnected
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
            try:
                self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                self.wfile.flush()
            except:
                pass
        finally:
            # Ensure stream is stopped
            if triton_client:
                try:
                    triton_client.stop_stream()
                except:
                    pass


class ThreadedHTTPServer(HTTPServer):
    """HTTP server that handles requests in threads."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=64)
    
    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread, request, client_address)
    
    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    global tokenizer, args
    
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible wrapper for Triton TensorRT-LLM"
    )
    parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC server URL",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        required=True,
        help="Path to tokenizer/HuggingFace model directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for OpenAI-compatible API server",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}", flush=True)
    print("OpenAI-compatible Triton Wrapper", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Triton URL: {args.triton_url}", flush=True)
    print(f"Tokenizer: {args.tokenizer_dir}", flush=True)
    print(f"API Port: {args.port}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Test Triton connection
    print("Testing Triton server connection...", flush=True)
    test_client = grpcclient.InferenceServerClient(
        url=args.triton_url,
        verbose=False,
    )
    
    if not test_client.is_server_live():
        print("ERROR: Triton server is not live", flush=True)
        exit(1)
    
    if not test_client.is_model_ready(model_name):
        print(f"ERROR: Model '{model_name}' is not ready", flush=True)
        exit(1)
    
    print("Triton server connection: OK", flush=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    print("Tokenizer loaded: OK", flush=True)
    
    # Start server
    server = ThreadedHTTPServer(("0.0.0.0", args.port), OpenAIHandler)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Server running at http://localhost:{args.port}", flush=True)
    print(f"OpenAI API endpoint: http://localhost:{args.port}/v1/chat/completions")
    print(f"{'='*60}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
