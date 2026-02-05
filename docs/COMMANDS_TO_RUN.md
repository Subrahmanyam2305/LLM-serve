# Run Triton server
python servers/launch_triton_server.py         --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B         --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct         --http-port 8000         --grpc-port 8001

# Benchmark Triton server
python benchmarks/benchmark_triton.py --engine triton --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --concurrency 1 2 4 8 16 32 64 --sharegpt --url localhost:8001

# Run vLLM server
python servers/vllm_backend_server.py --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --dtype float16

# Benchmark vLLM server (model name is auto-detected)
python benchmarks/benchmark_triton.py --engine vllm --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --concurrency 1 2 4 8 16 32 64 --sharegpt --url http://localhost:8000
