# GenAI-Bench Integration

This directory contains integration scripts for using [genai-bench](https://github.com/sgl-project/genai-bench) with LLM-serve.

## Installation

```bash
pip install genai-bench
```

## Usage

### Prerequisites

Before running genai-bench, you need to start a local server for your model:

#### vLLM Server
```bash
source ../.venv_vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000
```

#### SGLang Server
```bash
source ../.venv_sgl/bin/activate
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --port 8000
```

### Running Benchmarks

```bash
# Benchmark Qwen model
python run_genai_bench.py --model qwen --backend openai --api_base http://localhost:8000

# Benchmark Llama model
python run_genai_bench.py --model llama --backend openai --api_base http://localhost:8000

# With custom concurrency levels
python run_genai_bench.py --model qwen --concurrency 1 2 4 8 16 32

# Generate reports and plots
python run_genai_bench.py --model qwen --generate_report --generate_plots
```

### Supported Models

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| qwen | Qwen/Qwen2.5-3B-Instruct | Default model |
| llama | meta-llama/Llama-3.2-3B-Instruct | Requires HF_TOKEN |
| gemma | google/gemma-3-4b-it | Requires HF_TOKEN |
| phi | microsoft/Phi-4-mini-instruct | Open access |

## Direct CLI Usage

You can also use genai-bench directly:

```bash
# Run benchmark
genai-bench benchmark \
    --api-backend openai \
    --api-base http://localhost:8000 \
    --api-key EMPTY \
    --api-model-name Qwen/Qwen2.5-3B-Instruct \
    --task text-to-text \
    --max-time-per-run 60 \
    --max-requests-per-run 100

# Generate Excel report
genai-bench excel \
    --experiment-folder ./experiments/your_experiment \
    --excel-name results \
    --metric-percentile mean

# Generate plots
genai-bench plot \
    --experiments-folder ./experiments \
    --group-key traffic_scenario \
    --preset 2x4_default
```

## Output

Results are saved to the `./experiments` directory with:
- Raw metrics data
- Excel reports (if `--generate_report`)
- Visualization plots (if `--generate_plots`)
