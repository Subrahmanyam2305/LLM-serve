# Run Triton server
python servers/launch_triton_server.py         --engine-dir /home/ec2-user/llm_host/trt_engines/fp16_qwen_2.5_3B         --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct         --http-port 8000         --grpc-port 8001

# Benchmark Triton server
python benchmarks/benchmark_triton.py --engine triton --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --concurrency 1 2 4 8 16 32 64 --sharegpt --url localhost:8001

# Run vLLM server
python servers/vllm_backend_server.py --model /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --dtype float16

# Benchmark vLLM server (model name is auto-detected)
python benchmarks/benchmark_triton.py --engine vllm --tokenizer-dir /home/ec2-user/llm_host/Qwen2.5-3B-Instruct --concurrency 1 2 4 8 16 32 64 --sharegpt --url http://localhost:8000

---

# Benchmark Analysis Results

This folder contains benchmark analysis results organized by model and configuration.

## Folder Structure

```
analysis/
├── qwen-triton/          # Qwen2.5-3B-Instruct via Triton server benchmark
│   ├── throughput_comparison.png
│   ├── latency_comparison.png
│   ├── ttft_comparison.png
│   ├── scaling_efficiency.png
│   ├── benchmark_dashboard.png
│   └── BENCHMARK_RESULTS.md
└── README.md
```

## Generating Plots

Run from the `benchmarks/` directory:

```bash
# Generate plots for Triton benchmark results
python generate_plots.py --output-subfolder qwen-triton

# Specify custom results directory
python generate_plots.py --output-subfolder qwen-triton --results-dir ../results

# Use different result filenames
python generate_plots.py \
    --output-subfolder qwen-triton \
    --triton-results triton_server_results.json \
    --vllm-results vllm_server_results.json
```

## Adding New Benchmarks

1. Run `benchmark_triton.py` with your configuration
2. Run `generate_plots.py --output-subfolder <your-subfolder-name>`
3. Results will be saved to `analysis/<your-subfolder-name>/`
