---
layout: default
title: "The Complete Guide to LLM Inference"
---

# The Complete Guide to LLM Inference

**From Transformers to TensorRT: A Practitioner's Journey Through LLM Serving**

*By Subrahmanyam Arunachalam | February 2026*

---

<div class="toc">
<h3>Table of Contents</h3>
<ul>
<li><a href="#part-1-foundations">Part 1: Foundations</a>
  <ul>
    <li><a href="#the-transformer-refresher">The Transformer Refresher</a></li>
    <li><a href="#why-inference-is-hard">Why Inference is Hard</a></li>
    <li><a href="#metrics-that-matter">Metrics That Matter</a></li>
  </ul>
</li>
<li><a href="#part-2-inference-engines">Part 2: The Inference Engine Landscape</a>
  <ul>
    <li><a href="#vllm-pagedattention">vLLM: The PagedAttention Revolution</a></li>
    <li><a href="#sglang-radixattention">SGLang: RadixAttention & Prefix Caching</a></li>
    <li><a href="#tensorrt-llm">TensorRT-LLM: Kernel-Level Optimization</a></li>
  </ul>
</li>
<li><a href="#part-3-deep-dive">Part 3: Deep Dive into Optimizations</a>
  <ul>
    <li><a href="#tensorrt-pipeline">The TensorRT-LLM Pipeline</a></li>
    <li><a href="#quantization">Quantization Strategies</a></li>
    <li><a href="#batching-strategies">Batching Strategies Compared</a></li>
  </ul>
</li>
<li><a href="#part-4-benchmarking">Part 4: The Benchmarking Journey</a>
  <ul>
    <li><a href="#fair-comparison">Setting Up a Fair Comparison</a></li>
    <li><a href="#results">Results & Insights</a></li>
    <li><a href="#when-to-use-what">When to Use What</a></li>
  </ul>
</li>
<li><a href="#part-5-troubleshooting">Part 5: The Troubleshooting Bible</a></li>
<li><a href="#part-6-production">Part 6: Production Deployment</a></li>
</ul>
</div>

---

## Introduction

After spending months benchmarking LLM inference engines, building TensorRT pipelines, and debugging cryptic CUDA errors at 2 AM, I decided to write the guide I wish I had when I started.

This isn't a documentation dump. It's a practitioner's journey through the landscape of LLM serving—from understanding why inference is fundamentally different from training, to choosing between vLLM's elegant memory management and TensorRT's raw kernel optimization.

<div class="key-insight">
<h4>What You'll Learn</h4>
<ul>
<li>How attention works and why the KV cache is the bottleneck</li>
<li>The progression from model-level (vLLM) to kernel-level (TensorRT) optimizations</li>
<li>Real benchmark results comparing engines across 4 models</li>
<li>Every issue I faced and how I solved them</li>
</ul>
</div>

---

<h2 id="part-1-foundations">Part 1: Foundations</h2>

<h3 id="the-transformer-refresher">The Transformer Refresher</h3>

Before diving into inference optimization, let's establish a shared mental model of what happens inside a transformer during generation.

#### The Attention Mechanism

At its core, attention answers: "For each token I'm generating, which previous tokens should I pay attention to?"

<div class="ascii-diagram">
                    Query (Q)
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │         Attention Scores            │
    │    score = Q · K^T / √d_k           │
    │                                     │
    │  "How relevant is each past token   │
    │   to what I'm generating now?"      │
    └─────────────────────────────────────┘
                       │
                       ▼
              softmax(scores)
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │      Weighted Sum of Values         │
    │         output = attn · V           │
    └─────────────────────────────────────┘
</div>

The mathematical formulation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Where:
- **Q (Query)**: "What am I looking for?" - the current token's representation
- **K (Key)**: "What do I contain?" - representations of all previous tokens  
- **V (Value)**: "What information do I provide?" - the actual content to aggregate

#### The KV Cache: Why It Matters

Here's the critical insight for inference: **Keys and Values don't change for past tokens**.

When generating token 100, the K and V for tokens 1-99 are identical to what they were when generating token 99. Recomputing them is wasteful.

<div class="ascii-diagram">
Token Generation Timeline:

Step 1: Generate token 1
        Compute K₁, V₁ → Store in cache

Step 2: Generate token 2  
        Reuse K₁, V₁ from cache
        Compute K₂, V₂ → Add to cache

Step 3: Generate token 3
        Reuse K₁, K₂, V₁, V₂ from cache
        Compute K₃, V₃ → Add to cache

        ...and so on

The KV cache grows linearly with sequence length!
</div>

<div class="callout callout-info">
<div class="callout-title">💡 Key Insight</div>
<p>The KV cache is why LLM inference is <strong>memory-bound</strong>, not compute-bound. For a 7B parameter model with 4K context, the KV cache alone can consume 8-16GB of GPU memory.</p>
</div>

<h3 id="why-inference-is-hard">Why Inference is Hard</h3>

Training and inference are fundamentally different beasts:

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Parallelism** | Process entire sequences in parallel | Generate one token at a time (autoregressive) |
| **Bottleneck** | Compute-bound (matrix multiplications) | Memory-bound (KV cache access) |
| **Batch Size** | Large, fixed batches | Dynamic, varying request lengths |
| **Optimization Goal** | Maximize FLOPS utilization | Minimize latency, maximize throughput |

#### The Autoregressive Problem

LLMs generate text **one token at a time**. Each new token depends on all previous tokens. This creates a sequential dependency that's fundamentally at odds with GPU parallelism.

<div class="ascii-diagram">
The Autoregressive Loop:

    ┌──────────────────────────────────────────────┐
    │                                              │
    │  Input: "The capital of France is"           │
    │                    │                         │
    │                    ▼                         │
    │  ┌─────────────────────────────────┐         │
    │  │     Full Forward Pass           │         │
    │  │     (All transformer layers)    │         │
    │  └─────────────────────────────────┘         │
    │                    │                         │
    │                    ▼                         │
    │           Output: "Paris"                    │
    │                    │                         │
    │                    ▼                         │
    │  Input: "The capital of France is Paris"    │
    │                    │                         │
    │                    ▼                         │
    │  ┌─────────────────────────────────┐         │
    │  │     Full Forward Pass           │◄────────┘
    │  │     (But reuse KV cache!)       │
    │  └─────────────────────────────────┘
    │                    │
    │                    ▼
    │           Output: "."
    │                   ...
</div>

<h3 id="metrics-that-matter">Metrics That Matter</h3>

When benchmarking inference engines, these are the metrics that actually matter:

<div class="metric-grid">
<div class="metric-card">
<div class="metric-value">TTFT</div>
<div class="metric-label">Time to First Token</div>
</div>
<div class="metric-card">
<div class="metric-value">ITL</div>
<div class="metric-label">Inter-Token Latency</div>
</div>
<div class="metric-card">
<div class="metric-value">tok/s</div>
<div class="metric-label">Throughput</div>
</div>
<div class="metric-card">
<div class="metric-value">GB</div>
<div class="metric-label">Memory Usage</div>
</div>
</div>

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **TTFT** | Time from request to first token | User-perceived responsiveness |
| **ITL** | Average time between tokens | Streaming smoothness |
| **Throughput** | Total tokens/second across all requests | Cost efficiency, capacity |
| **Latency** | End-to-end generation time | SLA compliance |

<div class="callout callout-warning">
<div class="callout-title">⚠️ The TTFT vs Throughput Tradeoff</div>
<p>Optimizing for TTFT often hurts throughput and vice versa. TensorRT excels at TTFT (7-16ms), while vLLM wins on throughput (5000+ tok/s). Choose based on your use case.</p>
</div>

---

<h2 id="part-2-inference-engines">Part 2: The Inference Engine Landscape</h2>

The inference optimization stack can be visualized as layers, from high-level request handling down to raw CUDA kernels:

<div class="ascii-diagram">
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST LEVEL                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Continuous Batching, Request Scheduling            │    │
│  │  "How do we handle multiple concurrent requests?"   │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    MODEL LEVEL                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  PagedAttention, RadixAttention, KV Cache Mgmt      │    │
│  │  "How do we efficiently manage memory?"             │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    KERNEL LEVEL                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Fused Operations, Custom CUDA Kernels, TensorRT    │    │
│  │  "How do we make each operation as fast as possible?"│    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

        vLLM ──────────► Optimizes Request + Model Level
        SGLang ────────► Optimizes Model Level (Prefix Caching)
        TensorRT-LLM ──► Optimizes Kernel Level
</div>

<h3 id="vllm-pagedattention">vLLM: The PagedAttention Revolution</h3>

vLLM's key innovation is **PagedAttention**—treating the KV cache like virtual memory.

#### The Problem vLLM Solves

Traditional inference engines pre-allocate contiguous memory for the maximum possible sequence length. This leads to massive waste:

<div class="ascii-diagram">
Traditional KV Cache Allocation:

Request 1: "Hello" (5 tokens) → Allocated for 4096 tokens
           [█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
           Used: 5    Wasted: 4091 (99.9% waste!)

Request 2: "Hi" (2 tokens) → Allocated for 4096 tokens  
           [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
           Used: 2    Wasted: 4094 (99.9% waste!)
</div>

#### PagedAttention: Virtual Memory for KV Cache

vLLM borrows the concept of **paging** from operating systems:

<div class="ascii-diagram">
PagedAttention Memory Management:

Physical Memory (GPU):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ P8 │ P9 │  Fixed-size blocks
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

Request 1 (needs 3 blocks):
  Page Table: [P0, P3, P7]  ──► Points to non-contiguous blocks
  
Request 2 (needs 2 blocks):
  Page Table: [P1, P5]  ──► Shares physical memory efficiently

Request 3 (needs 4 blocks):
  Page Table: [P2, P4, P6, P8]

No wasted memory! Blocks allocated on-demand.
</div>

#### Why PagedAttention Matters

1. **Near-zero memory waste**: Only allocate what you need
2. **Efficient memory sharing**: Common prefixes share physical blocks
3. **Dynamic allocation**: Grow/shrink as sequences evolve
4. **Higher throughput**: Fit more concurrent requests in GPU memory

```python
# Starting a vLLM server is simple
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000
```

<div class="callout callout-success">
<div class="callout-title">✅ When to Use vLLM</div>
<ul>
<li>High-throughput batch processing</li>
<li>Variable-length requests</li>
<li>Quick deployment (no compilation needed)</li>
<li>OpenAI-compatible API required</li>
</ul>
</div>

<h3 id="sglang-radixattention">SGLang: RadixAttention & Prefix Caching</h3>

SGLang takes a different approach: **optimize for common prefixes**.

#### The Insight

In real applications, many requests share common prefixes:
- System prompts ("You are a helpful assistant...")
- Few-shot examples
- Document context in RAG applications

<div class="ascii-diagram">
Prefix Sharing Opportunity:

Request 1: "You are a helpful assistant. What is 2+2?"
Request 2: "You are a helpful assistant. Explain quantum physics."
Request 3: "You are a helpful assistant. Write a poem."
           └──────────────────────────┘
                  Shared Prefix!
                  
Why compute KV cache for "You are a helpful assistant" three times?
</div>

#### RadixAttention: A Radix Tree for KV Cache

SGLang organizes the KV cache as a **radix tree**, enabling efficient prefix lookup and sharing:

<div class="ascii-diagram">
RadixAttention Tree Structure:

                    [Root]
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   "You are"      "Hello"        "Explain"
        │              │              │
        ▼              ▼              ▼
   "a helpful"    "world"        "the concept"
        │                             │
        ▼                             ▼
   "assistant"                   "of attention"
        │
   ┌────┴────┐
   ▼         ▼
"What"    "Write"
   │         │
   ▼         ▼
"is 2+2"  "a poem"

Each node stores KV cache for that prefix segment.
New requests traverse the tree, reusing cached computations.
</div>

<div class="callout callout-info">
<div class="callout-title">💡 SGLang Shines For</div>
<ul>
<li>Multi-turn conversations (chat history is a shared prefix)</li>
<li>RAG applications (document context is shared)</li>
<li>Structured generation with common templates</li>
</ul>
</div>

<h3 id="tensorrt-llm">TensorRT-LLM: Kernel-Level Optimization</h3>

While vLLM and SGLang optimize at the model/request level, TensorRT-LLM goes deeper: **optimizing the actual CUDA kernels**.

#### The Compilation Approach

TensorRT-LLM doesn't just run your model—it **compiles** it into optimized GPU code:

<div class="ascii-diagram">
TensorRT-LLM Pipeline:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HuggingFace    │     │   TRT-LLM       │     │   TensorRT      │
│  Model Weights  │────►│   Checkpoint    │────►│   Engine        │
│  (PyTorch)      │     │   (Optimized    │     │   (Compiled     │
│                 │     │    Layout)      │     │    CUDA Code)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   safetensors            Weight layout           Fused kernels
   format                 reorganized             GPU-specific
                          for GPU access          optimizations
</div>

#### What TensorRT Actually Does

1. **Layer Fusion**: Combines multiple operations into single kernels
2. **Kernel Auto-tuning**: Tests thousands of kernel variants, picks the fastest
3. **Precision Calibration**: Optimizes INT8/FP8 quantization
4. **Memory Planning**: Optimizes tensor memory layout

<div class="ascii-diagram">
Layer Fusion Example:

Before (3 kernel launches):
┌──────────┐    ┌──────────┐    ┌──────────┐
│  MatMul  │───►│   Add    │───►│   ReLU   │
└──────────┘    └──────────┘    └──────────┘
   Kernel 1       Kernel 2       Kernel 3
   
After TensorRT Fusion (1 kernel launch):
┌────────────────────────────────────────────┐
│         Fused MatMul + Add + ReLU          │
└────────────────────────────────────────────┘
                  Kernel 1

Fewer kernel launches = Less overhead = Lower latency
</div>

#### The Trade-off

| Aspect | vLLM | TensorRT-LLM |
|--------|------|--------------|
| **Setup Time** | Seconds | Minutes to hours |
| **Flexibility** | Load any HF model | Requires conversion + build |
| **Portability** | Works on any GPU | Engine tied to specific GPU |
| **Latency** | Good | Excellent (10-15x better TTFT) |
| **Throughput** | Excellent | Good to Excellent |

```bash
# TensorRT-LLM requires a multi-step process
# Step 1: Convert checkpoint
python convert_checkpoint.py --model_dir ./Qwen2.5-3B --output_dir ./checkpoint

# Step 2: Build engine (GPU-specific!)
trtllm-build --checkpoint_dir ./checkpoint --output_dir ./engine --gemm_plugin float16

# Step 3: Run inference
python run_trt.py --engine_dir ./engine --tokenizer_dir ./Qwen2.5-3B
```

<div class="callout callout-warning">
<div class="callout-title">⚠️ TensorRT Engines are GPU-Specific</div>
<p>An engine built on A10G (SM86) won't work on V100 (SM70) or H100 (SM90). Always build on your target hardware.</p>
</div>

---

<h2 id="part-3-deep-dive">Part 3: Deep Dive into Optimizations</h2>

<h3 id="tensorrt-pipeline">The TensorRT-LLM Pipeline Explained</h3>

Let me walk you through the exact pipeline I built for this project. Understanding each step demystifies what "compilation" actually means.

#### Step 1: Checkpoint Conversion

The first step reorganizes HuggingFace weights for optimal GPU memory access patterns:

```python
# From tensorrt/convert_qwen.py
python convert_checkpoint.py \
    --model_dir ./models/Qwen2.5-3B-Instruct \
    --output_dir ./checkpoints/qwen_fp16 \
    --dtype float16
```

**What happens under the hood:**
- Weights are transposed for coalesced memory access
- Attention heads are reorganized for efficient parallel computation
- Layer normalization weights are fused where possible

#### Step 2: Engine Building

This is where the magic happens. TensorRT analyzes your model and generates optimized CUDA code:

```bash
trtllm-build \
    --checkpoint_dir ./checkpoints/qwen_fp16 \
    --output_dir ./engines/qwen_fp16 \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_output_len 512
```

<div class="ascii-diagram">
Engine Build Process (simplified):

┌─────────────────────────────────────────────────────────────┐
│                    TensorRT Builder                         │
├─────────────────────────────────────────────────────────────┤
│  1. Parse Network                                           │
│     └─► Read checkpoint, build computation graph            │
│                                                             │
│  2. Apply Optimizations                                     │
│     ├─► Layer fusion (combine ops)                          │
│     ├─► Precision calibration (FP16/INT8)                   │
│     └─► Memory optimization (tensor reuse)                  │
│                                                             │
│  3. Kernel Selection                                        │
│     └─► Test 1000s of kernel variants per layer             │
│     └─► Profile each on YOUR specific GPU                   │
│     └─► Select fastest combination                          │
│                                                             │
│  4. Serialize Engine                                        │
│     └─► Write optimized binary (GPU-specific!)              │
└─────────────────────────────────────────────────────────────┘
</div>

<div class="callout callout-info">
<div class="callout-title">💡 Why Engine Build Takes So Long</div>
<p>TensorRT literally benchmarks thousands of kernel implementations for each layer on your specific GPU. A 3B model can take 10-30 minutes to build. This is why engines are GPU-specific—the "fastest" kernel on A10G might not be fastest on H100.</p>
</div>

<h3 id="quantization">Quantization Strategies</h3>

Quantization reduces model precision to save memory and increase throughput. Here's what I tested:

| Precision | Memory | Speed | Quality | Use Case |
|-----------|--------|-------|---------|----------|
| **FP16** | Baseline | Baseline | Best | Default choice |
| **INT8** | ~50% less | ~1.5x faster | Good | Production serving |
| **INT4 (AWQ)** | ~75% less | ~2x faster | Acceptable | Memory-constrained |
| **FP8** | ~50% less | ~1.8x faster | Very Good | H100/Ada GPUs |

```bash
# INT8 quantization during conversion
python convert_checkpoint.py \
    --model_dir ./models/Qwen2.5-3B-Instruct \
    --output_dir ./checkpoints/qwen_int8 \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8
```

<h3 id="batching-strategies">Batching Strategies Compared</h3>

One of the biggest "aha moments" in this project was understanding the difference between batching strategies:

<div class="ascii-diagram">
Static Batching (Traditional):
─────────────────────────────
Request 1: ████████████████████████████████████████ (done)
Request 2: ████████████████████████████████████████ (done)
Request 3: ████████████████████████████████████████ (done)
           └─────────── Wait for ALL to finish ───────────┘
           Then start next batch

Problem: Short requests wait for long ones!


Continuous Batching (vLLM):
─────────────────────────────
Request 1: ████████████████████████████████████████ (done)
Request 2: ████████████ (done) → Request 4 starts immediately!
Request 3: ████████████████████████ (done) → Request 5 starts!

Requests join/leave the batch dynamically.


Inflight Batching (Triton + TensorRT):
─────────────────────────────
Similar to continuous, but at the iteration level.
New requests can join mid-generation of existing requests.
</div>

<div class="callout callout-warning">
<div class="callout-title">⚠️ Lesson Learned: Batch Size ≠ Concurrency</div>
<p>Early in this project, I confused "batch size" with "concurrent requests." They're different! Batch size is how many sequences the GPU processes in one forward pass. Concurrency is how many requests your server handles simultaneously. With continuous batching, these decouple—you can have 100 concurrent requests with dynamic batch sizes.</p>
</div>

---

<h2 id="part-4-benchmarking">Part 4: The Benchmarking Journey</h2>

<h3 id="fair-comparison">Setting Up a Fair Comparison</h3>

Benchmarking LLM inference engines fairly is harder than it sounds. Here's what I learned:

#### Pitfall 1: Dependency Conflicts

TensorRT-LLM, vLLM, and SGLang have **conflicting dependencies**. They require different versions of PyTorch, transformers, and CUDA libraries.

**Solution:** Separate virtual environments for each:

```bash
# Directory structure
llm_host/
├── .venv_trt/      # TensorRT-LLM environment
├── .venv_vllm/     # vLLM environment  
├── .venv_sgl/      # SGLang environment
└── LLM-serve/      # This repository
```

#### Pitfall 2: Unfair Token Counting

I initially found SGLang showing 2x higher throughput than vLLM. Suspicious!

The bug: SGLang was using word count estimation (`len(output.split())`) while vLLM used actual tokenizer counts.

**Solution:** Always use the tokenizer:

```python
# Wrong (unfair!)
tokens_generated = len(output_text.split()) * 1.3

# Correct (fair)
tokens_generated = len(tokenizer.encode(output_text))
```

#### Pitfall 3: Sampling Parameters

Different defaults across engines can skew results:

```python
# Standardized parameters for fair comparison
temperature = 0.0    # Greedy decoding
top_k = 1            # Greedy
top_p = 1.0          # No nucleus sampling
max_tokens = 512     # Same output length
```

<h3 id="results">Results & Insights</h3>

After weeks of benchmarking, here are the results across 4 models on NVIDIA A10G:

#### Performance Summary

| Model | TensorRT Peak | vLLM Peak | TensorRT TTFT | vLLM TTFT |
|-------|---------------|-----------|---------------|-----------|
| **Gemma 3-1B** | 3,972 tok/s | **5,121 tok/s** | **7.7 ms** | 97.3 ms |
| **Llama 3.2-3B** | 1,407 tok/s | **2,477 tok/s** | **16.3 ms** | 247.0 ms |
| **Phi-2 2.7B** | **1,934 tok/s** | 1,571 tok/s | **13.8 ms** | 202.0 ms |
| **Qwen 2.5-3B** | **2,486 tok/s** | 2,427 tok/s | **16.1 ms** | 224.3 ms |

![Multi-Model Dashboard](assets/images/multi_model_dashboard.png)
<p class="image-caption">TensorRT-LLM performance across all tested models</p>

![vLLM Dashboard](assets/images/multi_model_dashboard_vllm.png)
<p class="image-caption">vLLM performance across all tested models</p>

#### Key Findings

<div class="key-insight">
<h4>🔍 What the Data Tells Us</h4>
<ol>
<li><strong>vLLM wins throughput</strong> (generally 20-80% higher), except for Phi-2 and Qwen where TensorRT wins</li>
<li><strong>TensorRT dominates latency</strong> (10-15x better TTFT across ALL models)</li>
<li><strong>Model architecture matters</strong>: Phi-2 and Qwen uniquely favor TensorRT for throughput</li>
<li><strong>Smaller isn't always slower</strong>: Gemma (1B) outperforms larger models (3B) on both engines</li>
</ol>
</div>

#### Scaling with Concurrency

![Throughput Comparison](assets/images/throughput_comparison_vllm.png)
<p class="image-caption">How throughput scales with concurrent requests</p>

![TTFT Comparison](assets/images/ttft_comparison_tensorrt.png)
<p class="image-caption">TensorRT maintains low TTFT even under load</p>

<h3 id="when-to-use-what">When to Use What</h3>

Based on my benchmarking, here's a decision framework:

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Real-time chat** | TensorRT-LLM | 7-16ms TTFT feels instant |
| **Batch processing** | vLLM | Higher throughput = lower cost |
| **Multi-turn conversations** | SGLang | RadixAttention reuses context |
| **Quick prototyping** | vLLM | No compilation, instant start |
| **Production API** | Triton + TensorRT | Best of both worlds |

---

<h2 id="part-5-troubleshooting">Part 5: The Troubleshooting Bible</h2>

This section contains every issue I encountered and how I solved it. Bookmark this.

### Installation Issues

#### "No Space Left on Device"

```
OSError: [Errno 28] No space left on device
```

**Cause:** `/tmp` fills up during pip installation of large packages.

**Solution:**
```bash
export TMPDIR=/path/to/large/disk/tmp
export PIP_CACHE_DIR=$TMPDIR/pip_cache
mkdir -p $PIP_CACHE_DIR
```

#### TensorRT-LLM Not Found

```
ERROR: No matching distribution found for tensorrt-llm
```

**Cause:** TensorRT-LLM is on NVIDIA's PyPI, not the default.

**Solution:**
```bash
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
```

### CUDA Issues

#### CUDA_HOME Not Set

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

**Solution:**
```bash
# Find your CUDA installation
which nvcc
# or
ls /usr/local/cuda*

# Set environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

#### Cached nvcc Path from Wrong Environment

```
/bin/sh: line 1: /home/user/.venv_trt/.../nvcc: No such file or directory
```

**Cause:** Flashinfer caches nvcc path in `~/.cache/flashinfer/`.

**Solution:**
```bash
rm -rf ~/.cache/flashinfer
```

### Benchmarking Issues

#### TTFT Showing None

**Cause:** Non-streaming inference doesn't capture per-token timing.

**Solution:** Use `--streaming` flag:
```bash
python benchmark.py --streaming --engines vllm tensorrt
```

#### Gated Model Access Denied

```
401 Client Error: Unauthorized for url: https://huggingface.co/meta-llama/...
```

**Solution:**
1. Get token from https://huggingface.co/settings/tokens
2. Accept model license on HuggingFace
3. Set environment variable:
```bash
export HF_TOKEN=your_token_here
```

### Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| No space left | `export TMPDIR=/large/disk/tmp` |
| TensorRT not found | `--extra-index-url https://pypi.nvidia.com/` |
| Dependency conflicts | Use separate virtual environments |
| CUDA_HOME not set | `export CUDA_HOME=/usr/local/cuda` |
| Flashinfer nvcc error | `--attention-backend triton` |
| TTFT showing None | Use `--streaming` flag |
| Gated model denied | Set `HF_TOKEN` + accept license |

---

<h2 id="part-6-production">Part 6: Production Deployment</h2>

### Architecture Overview

For production, I recommend Triton Inference Server with TensorRT-LLM backend:

<div class="ascii-diagram">
Production Architecture:

┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Client App    │     │   Load Balancer      │     │  Triton Server  │
│   (Your API)    │────►│   (nginx/ALB)        │────►│  (Docker)       │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  TensorRT-LLM   │
                                                     │  Engine         │
                                                     │  (Compiled)     │
                                                     └─────────────────┘

Key Components:
• Triton handles request queuing, batching, health checks
• TensorRT-LLM provides optimized inference
• Docker ensures reproducible deployment
</div>

### Triton Server Setup

```bash
# Launch Triton with TensorRT-LLM backend
docker run --gpus all -d \
    -p 8000:8000 -p 8001:8001 \
    -v /path/to/model_repo:/models \
    nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3 \
    tritonserver --model-repository=/models
```

### Key Configuration Parameters

```protobuf
# config.pbtxt for TensorRT-LLM backend
parameters {
  key: "batching_strategy"
  value: { string_value: "inflight_fused_batching" }
}
parameters {
  key: "batch_scheduler_policy"  
  value: { string_value: "max_utilization" }
}
parameters {
  key: "decoupled_mode"
  value: { string_value: "True" }  # Required for streaming
}
```

### Monitoring & Health Checks

```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/tensorrt_llm

# Metrics endpoint (Prometheus format)
curl http://localhost:8002/metrics
```

---

## Conclusion

LLM inference optimization is a deep rabbit hole, but the core ideas are simple:

1. **Memory is the bottleneck** (KV cache), not compute
2. **vLLM solves memory** with PagedAttention (model-level optimization)
3. **TensorRT solves latency** with kernel fusion (kernel-level optimization)
4. **Choose based on your use case**: latency-sensitive → TensorRT, throughput-sensitive → vLLM

The code, benchmarks, and all the painful lessons learned are in the [LLM-serve repository](https://github.com/subrahmanyam-arunachalam/LLM-serve). Feel free to use it as a starting point for your own inference optimization journey.

<div class="key-insight">
<h4>Final Thoughts</h4>
<p>The best inference engine is the one that matches your constraints. Don't optimize for throughput if your users care about latency. Don't spend days building TensorRT engines if you're still iterating on your model. Start simple (vLLM), measure, then optimize where it matters.</p>
</div>

---

*Questions? Find me on [LinkedIn](https://linkedin.com/in/subrahmanyam-arunachalam) or open an issue on [GitHub](https://github.com/subrahmanyam-arunachalam/LLM-serve).*

---

## References & Credits

### Papers
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM paper
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) - SGLang paper

### Documentation
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)

### Image Credits
- Architecture diagrams: Original work
- Benchmark charts: Generated from [LLM-serve](https://github.com/subrahmanyam-arunachalam/LLM-serve) benchmarking suite
- Transformer diagram inspiration: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
