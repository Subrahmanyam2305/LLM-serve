#!/usr/bin/env python3
"""
TensorRT-LLM Backend Server - OpenAI-compatible API server.
Starts a TensorRT-LLM server using Triton Inference Server or the built-in HTTP server.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Start TensorRT-LLM OpenAI-compatible server")
    parser.add_argument("--engine-dir", type=str, required=True,
                        help="Path to TensorRT engine directory")
    parser.add_argument("--tokenizer-dir", type=str, required=True,
                        help="Path to tokenizer (HuggingFace model directory)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002,
                        help="Port to listen on (default 8002 to avoid conflict)")
    parser.add_argument("--max-batch-size", type=int, default=64,
                        help="Maximum batch size")
    parser.add_argument("--max-input-len", type=int, default=4096,
                        help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=1024,
                        help="Maximum output length")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key for authentication")
    parser.add_argument("--use-triton", action="store_true",
                        help="Use Triton Inference Server instead of built-in server")
    return parser.parse_args()


def start_builtin_server(args):
    """Start the built-in TensorRT-LLM HTTP server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import torch
    from transformers import AutoTokenizer
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunnerCpp, PYTHON_BINDINGS, ModelRunner
    import time
    
    print("Loading TensorRT-LLM engine...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    end_id = tokenizer.eos_token_id
    
    # Load runner
    runner_cls = ModelRunnerCpp if PYTHON_BINDINGS else ModelRunner
    runner = runner_cls.from_dir(
        engine_dir=args.engine_dir,
        rank=0,
    )
    
    print(f"Engine loaded successfully!")
    
    class TRTLLMHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                self.handle_chat_completions()
            elif self.path == "/v1/completions":
                self.handle_completions()
            else:
                self.send_error(404, "Not Found")
        
        def do_GET(self):
            if self.path == "/v1/models":
                self.handle_models()
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy"}).encode())
            else:
                self.send_error(404, "Not Found")
        
        def handle_models(self):
            response = {
                "object": "list",
                "data": [{
                    "id": Path(args.tokenizer_dir).name,
                    "object": "model",
                    "owned_by": "tensorrt-llm"
                }]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        
        def handle_chat_completions(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode())
            
            # Extract messages and convert to prompt
            messages = request.get("messages", [])
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "assistant: "
            
            max_tokens = request.get("max_tokens", 128)
            temperature = request.get("temperature", 1.0)
            stream = request.get("stream", False)
            
            # Generate
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]
            
            if stream:
                self.handle_streaming_response(batch_input_ids, max_tokens, temperature, request)
            else:
                self.handle_non_streaming_response(batch_input_ids, max_tokens, temperature, request, prompt)
        
        def handle_completions(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode())
            
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 128)
            temperature = request.get("temperature", 1.0)
            stream = request.get("stream", False)
            
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]
            
            if stream:
                self.handle_streaming_response(batch_input_ids, max_tokens, temperature, request)
            else:
                self.handle_non_streaming_response(batch_input_ids, max_tokens, temperature, request, prompt)
        
        def handle_streaming_response(self, batch_input_ids, max_tokens, temperature, request):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            
            input_len = len(batch_input_ids[0])
            
            try:
                with torch.no_grad():
                    outputs_gen = runner.generate(
                        batch_input_ids=batch_input_ids,
                        max_new_tokens=max_tokens,
                        end_id=end_id,
                        pad_id=pad_id,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_k=1 if temperature == 0 else 50,
                        output_sequence_lengths=True,
                        return_dict=True,
                        streaming=True,
                    )
                    
                    prev_text = ""
                    for outputs in outputs_gen:
                        output_ids = outputs['output_ids'][0][0][input_len:].tolist()
                        text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        delta = text[len(prev_text):]
                        prev_text = text
                        
                        if delta:
                            chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": Path(args.tokenizer_dir).name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None
                                }]
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                
                # Send final chunk
                final_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
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
                
            except Exception as e:
                error_chunk = {"error": str(e)}
                self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
        
        def handle_non_streaming_response(self, batch_input_ids, max_tokens, temperature, request, prompt):
            input_len = len(batch_input_ids[0])
            
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids=batch_input_ids,
                    max_new_tokens=max_tokens,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_k=1 if temperature == 0 else 50,
                    output_sequence_lengths=True,
                    return_dict=True,
                )
            
            output_ids = outputs['output_ids'][0][0][input_len:].tolist()
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": Path(args.tokenizer_dir).name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": len(output_ids),
                    "total_tokens": input_len + len(output_ids)
                }
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        
        def log_message(self, format, *args):
            print(f"[{self.log_date_time_string()}] {format % args}")
    
    server = HTTPServer((args.host, args.port), TRTLLMHandler)
    print(f"\nServer listening on http://{args.host}:{args.port}")
    print(f"OpenAI API endpoint: http://{args.host}:{args.port}/v1")
    print("Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        server.shutdown()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TensorRT-LLM OpenAI-Compatible Server")
    print("=" * 60)
    print(f"Engine: {args.engine_dir}")
    print(f"Tokenizer: {args.tokenizer_dir}")
    print(f"Host: {args.host}:{args.port}")
    print("=" * 60)
    
    if args.use_triton:
        print("\nTriton mode not implemented yet. Using built-in server.")
    
    start_builtin_server(args)


if __name__ == "__main__":
    main()
