# Multi-Model Performance Comparison

## Overview
Comprehensive comparison of both **TensorRT-LLM** and **vLLM** performance across multiple language models tested on the same hardware setup (NVIDIA A10G).

## Models Tested

### TensorRT-LLM (4 models):
- **Gemma 3-1B** (1B parameters, google/gemma-3-1b-it)
- **Llama 3.2-3B** (3B parameters, meta-llama/Llama-3.2-3B-Instruct) 
- **Phi-2 2.7B** (2.7B parameters, microsoft/phi-2)
- **Qwen 2.5-3B** (3B parameters, Qwen/Qwen2.5-3B-Instruct)

### vLLM (4 models):
- **Gemma 3-1B** (1B parameters, google/gemma-3-1b-it)
- **Llama 3.2-3B** (3B parameters, meta-llama/Llama-3.2-3B-Instruct) 
- **Phi-2 2.7B** (2.7B parameters, microsoft/phi-2)
- **Qwen 2.5-3B** (3B parameters, Qwen/Qwen2.5-3B-Instruct)

## Test Configuration
- **Hardware**: NVIDIA A10G (24GB VRAM)
- **Precision**: FP16 
- **Dataset**: ShareGPT prompts
- **Output Length**: 512 tokens (128 for Qwen)
- **Concurrency Levels**: 1, 2, 4, 8, 16, 32, 64
- **Visualization**: Clean charts with outlier exclusion (>10x median values)

## Key Findings

### 🏆 Performance Champions by Engine

**TensorRT-LLM Winner: Gemma 3-1B**
- Peak throughput: **3,972 tok/s** at C=64
- Best latency: **7.7 ms** TTFT at C=1
- Despite being smallest (1B params), dominates all metrics

**vLLM Winner: Gemma 3-1B** 
- Peak throughput: **5,121 tok/s** at C=64
- Higher throughput than TensorRT but worse latency (97.3 ms TTFT)

### 📊 Engine Comparison Summary

| Model | TensorRT Peak | vLLM Peak | TensorRT TTFT | vLLM TTFT | Throughput Winner | Latency Winner |
|-------|---------------|-----------|---------------|-----------|-------------------|----------------|
| **Gemma 3-1B** | 3,972 tok/s | **5,121 tok/s** | **7.7 ms** | 97.3 ms | vLLM (+29%) | TensorRT (-92%) |
| **Llama 3.2-3B** | 1,407 tok/s | **2,477 tok/s** | **16.3 ms** | 247.0 ms | vLLM (+76%) | TensorRT (-93%) |
| **Phi-2 2.7B** | **1,934 tok/s** | 1,571 tok/s | **13.8 ms** | 202.0 ms | TensorRT (+23%) | TensorRT (-93%) |
| **Qwen 2.5-3B** | **2,486 tok/s** | 2,427 tok/s | **16.1 ms** | 224.3 ms | TensorRT (+2%) | TensorRT (-93%) |

### 🔍 Key Insights

1. **vLLM Throughput Advantage**: Generally 20-80% higher throughput (except Phi & Qwen where TensorRT wins)
2. **TensorRT Latency Dominance**: Consistently 10-15x better TTFT performance across all models
3. **Model Architecture Matters**: Phi-2 and Qwen uniquely favor TensorRT for throughput
4. **Size vs Performance**: Smaller Gemma (1B) outperforms larger models (3B) in both engines

## Dashboard Features

### **Clean Visualization Style**
- **No Overlapping Labels**: Removed cluttered value labels for better readability
- **Automatic Outlier Exclusion**: Values >10x median automatically filtered
- **Professional Styling**: Clean, publication-ready charts
- **Individual Charts**: Separate PNG files for detailed metric analysis

### **TensorRT-LLM Dashboard** (`multi_model_dashboard.png`)
- **4-Panel Layout**: Throughput, TTFT, ITL (Inter-Token Latency), Performance Summary
- **Clean Scaling**: Outliers excluded automatically for readable charts
- **Outlier Notes**: Indicates when extreme values were excluded

### **vLLM Dashboard** (`multi_model_dashboard_vllm.png`)  
- **Same Layout**: Consistent 4-panel format for easy comparison
- **Engine-Specific Data**: Only shows vLLM performance metrics
- **Cross-Engine Analysis**: Use both dashboards for comprehensive comparison

## Performance Patterns

### **Throughput Scaling**:
- **vLLM**: Better continuous batching, higher peaks
- **TensorRT**: More efficient at lower concurrency, varies by model

### **Latency Characteristics**:
- **TensorRT**: Consistently sub-20ms TTFT across all models
- **vLLM**: Higher baseline latency but stable scaling

### **Model-Specific Behaviors**:
- **Gemma**: Excellent performer on both engines
- **Llama**: Strong vLLM throughput, some TensorRT latency spikes  
- **Phi**: Unique case where TensorRT wins throughput
- **Qwen**: TensorRT-only, solid middle performer

## Files Generated

### **Files Generated:**

#### **Main Dashboards:**
1. **`multi_model_dashboard.png`** - TensorRT-LLM 4-panel comparison
2. **`multi_model_dashboard_vllm.png`** - vLLM 4-panel comparison

#### **Individual Charts (TensorRT-LLM):**
3. **`throughput_comparison_tensorrt.png`** - Clean throughput comparison
4. **`ttft_comparison_tensorrt.png`** - Clean TTFT comparison  
5. **`itl_comparison_tensorrt.png`** - Clean ITL comparison

#### **Individual Charts (vLLM):**
6. **`throughput_comparison_vllm.png`** - Clean throughput comparison
7. **`ttft_comparison_vllm.png`** - Clean TTFT comparison
8. **`itl_comparison_vllm.png`** - Clean ITL comparison

#### **Summary Data:**
9. **`performance_summary_table.png`** - Cross-engine performance table
10. **`performance_summary.csv`** - Raw comparative data

#### **Documentation:**
11. **`README.md`** - This comprehensive analysis

## Use Case Recommendations

### 🚀 **High-Throughput Production APIs**
- **vLLM + Gemma 3-1B**: 5,121 tok/s peak throughput
- **Best for**: Batch processing, high-volume APIs, cost optimization

### ⚡ **Real-Time Interactive Applications**  
- **TensorRT-LLM + Gemma 3-1B**: 7.7ms TTFT
- **Best for**: Chatbots, real-time chat, gaming, live demos

### 🎯 **Balanced Performance & Capability**
- **vLLM + Llama 3.2-3B**: Good throughput (2,477 tok/s) + larger model
- **TensorRT-LLM + Qwen 2.5-3B**: Solid performance + good latency

### 💻 **Development & Experimentation** 
- **TensorRT-LLM + Phi-2 2.7B**: Unique throughput advantage
- **Best for**: Model development, custom architectures

## Technical Notes

- **Outlier Handling**: Values >10x median automatically excluded for scale clarity
- **Missing Data**: ITL not available for Qwen (markdown source)  
- **Hardware Consistency**: All tests on same NVIDIA A10G instance
- **Framework Versions**: TensorRT-LLM 1.1.0, vLLM latest available

## Reproduction

```bash
cd /home/ec2-user/llm_host/LLM-serve
source /home/ec2-user/llm_host/.venv_trt/bin/activate
python benchmarks/multi_model_comparison.py
```

Results saved to `analysis/all_models/`