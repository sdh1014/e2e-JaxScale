# JaxScale

**A JAX/Flax-based LLM inference framework built from scratch for learning and performance analysis, inspired by [sglang-jax](https://github.com/sgl-project/sglang-jax).**

[中文版](README_CN.md)

JaxScale implements complete inference pipelines for GLM-family models on JAX/Flax NNX, with support for Tensor Parallelism (TP), Data Parallelism (DP), and Expert Parallelism (EP) on TPU/GPU clusters.

## Supported Models

| Model | Architecture | Parameters | Attention | FFN |
|-------|-------------|-----------|-----------|-----|
| **GLM-4-9B** | Dense Transformer | 9B (18 GB bf16) | GQA (32Q / 2KV) | SwiGLU |
| **GLM-4.7-Flash** | MoE Transformer | 62B total / 3B active (bf16) | MLA | 64 routed + 1 shared expert |

## Key Features

- **KV Cache** — Pre-allocated static arrays with `jax.lax.dynamic_update_slice`, achieving 100x decode speedup
- **GQA Attention** — Grouped-Query Attention with 16:1 Q/KV head ratio and RoPE
- **MLA Attention** — Multi-head Latent Attention with low-rank KV compression (kv_lora_rank=512)
- **MoE Layer** — Top-4 softmax routing with 64 routed experts and 1 shared expert
- **Tensor Parallelism** — Column/row-parallel sharding of attention and MLP weights
- **Expert Parallelism** — `shard_map`-based EP with local gather + single `psum` for efficient cross-device communication
- **Data Parallelism** — Planned (not yet implemented)
- **HuggingFace Weight Loading** — Direct loading from SafeTensors with automatic transpose and split transformations
- **Benchmarking Suite** — Throughput, latency, MFU, and MBU analysis with component-level profiling

## Performance

### GLM-4-9B

#### L4 GPU (24 GB, 300 GB/s bandwidth, 60 TFLOPS bf16)

| Phase | Config | Throughput | Efficiency |
|-------|--------|-----------|------------|
| Prefill | batch=1, seq=1024 | 2,836 tok/s | **MFU 83.0%** |
| Decode | batch=1 | 13.8 tok/s | **MBU 80.8%** |

#### TPU v6e (32 GB HBM, 1,600 GB/s bandwidth, 918 TFLOPS bf16)

**Single Chip:**

| Phase | Config | Throughput | Efficiency |
|-------|--------|-----------|------------|
| Prefill | batch=32, seq=128 | 16,915 tok/s | MFU 32.4% |
| Decode | batch=1 | 65 tok/s | MBU 76.5% |
| Decode | batch=32 | 1,915 tok/s | MBU 70.3% |

**TP=2 (2 chips):**

| Phase | Config | Throughput | Efficiency | vs TP=1 |
|-------|--------|-----------|------------|---------|
| Decode | batch=1 | 122 tok/s | MBU 71.4% | 1.9x |
| Decode | batch=8 | 961 tok/s | MBU 70.6% | 1.9x |
| Decode | batch=32 | 3,064 tok/s | MBU 56.2% | 1.6x |
| Prefill | batch=32, seq=512 | 17,806 tok/s | MFU 17.0% | 1.4x |
| Prefill | batch=32, seq=1024 | 16,220 tok/s | MFU 15.5% | unlocked (TP=1 OOM) |

### GLM-4.7-Flash

#### TPU v6e-4 (TP=2 + EP=2, 4 chips)

| Phase | Config | Throughput | Efficiency |
|-------|--------|-----------|------------|
| Decode | batch=1 | 10.2 tok/s | MBU 5.7% |

> **Bottleneck analysis:** MLA Attention accounts for 65% of per-layer latency despite only 24.5 MB weight reads — dominated by kernel launch overhead (7 serial matmuls) and sharding conversions rather than computation or bandwidth. See [detailed analysis](docs/glm4flash_performance_analysis.md) for optimization roadmap (shard_map wrapping, absorbed MLA, Pallas fusion) targeting 67 tok/s.

### Optimization History (GLM-4-9B)

| Stage | Decode (tok/s) | Bottleneck |
|-------|---------------|------------|
| No KV Cache, no JIT | 0.1 | Redundant computation |
| + KV Cache | 1.5 | Python dispatch overhead |
| + jax.jit | 13.8 | Memory bandwidth (physical limit) |
| + TP=2 (TPU) | 122 | Memory bandwidth (halved per device) |

## Quick Start

### Dependencies

```
jax
flax
transformers
safetensors
numpy
```

### Single-Device Inference

```bash
python main.py \
  --model_path /path/to/glm-4-9b-chat-hf \
  --prompt "Hello, how are you?" \
  --max_new_tokens 128 \
  --dtype bfloat16
```

### Tensor Parallel Inference (2 devices)

```bash
python main.py \
  --model_path /path/to/glm-4-9b-chat-hf \
  --prompt "Hello" \
  --tp 2 --dp 1
```

### GLM-4.7-Flash with Mixed Parallelism

```bash
python main.py \
  --model_path /path/to/glm-4.7-flash \
  --prompt "What is the capital of France?" \
  --tp 2 --ep 2
```

### Benchmarking

```bash
python benchmarks/benchmark.py \
  --model_path /path/to/glm-4-9b-chat-hf \
  --tp 1 --dp 1

# Component-level profiling (GLM-4.7-Flash)
python benchmarks/profile_glm4flash.py \
  --model_path /path/to/glm-4.7-flash \
  --tp 2 --ep 2
```

## Project Structure

```
JaxScale/
├── main.py                      # Entry point for inference
├── runner.py                    # Generation loops (greedy, sampling)
├── configs/
│   └── model_config.py          # Unified config for GLM-4-9B & GLM-4.7-Flash
├── models/
│   ├── glm4.py                  # GLM-4-9B dense model
│   └── glm4_flash.py            # GLM-4.7-Flash MoE model
├── layers/
│   ├── attention.py             # GQA Attention
│   ├── mla_attention.py         # MLA (Multi-head Latent Attention)
│   ├── moe.py                   # MoE layer, router, shared expert
│   ├── linear.py                # Linear layer with sharding support
│   ├── embedding.py             # Embedding & parallel LM head
│   ├── normalization.py         # RMSNorm
│   ├── rotary.py                # Rotary Position Embedding (RoPE)
│   └── kv_cache.py              # KV Cache
├── utils/
│   ├── weight_utils.py          # HuggingFace weight loading & sharding
│   └── mesh_utils.py            # Device mesh creation (TP, DP, EP)
├── benchmarks/
│   ├── benchmark.py             # Throughput & latency benchmarking
│   ├── verify_kvcache.py        # KV Cache correctness verification
│   └── profile_glm4flash.py     # Component-level profiling
└── docs/                        # Design docs & analysis (in Chinese)
```

## Architecture

### GLM-4-9B (Dense)

```
Input → Embedding → 40× DecoderLayer → RMSNorm → LM Head → Output
                         │
                    RMSNorm → GQA Attention (32Q/2KV + RoPE) → residual
                         │
                    RMSNorm → SwiGLU MLP (4096→13696→4096) → residual
```

### GLM-4.7-Flash (MoE + MLA)

```
Input → Embedding → 47× DecoderLayer → RMSNorm → LM Head → Output
                         │
                    RMSNorm → MLA Attention (low-rank KV compression + RoPE) → residual
                         │
                    RMSNorm → [Layer 0: Dense MLP]
                              [Layer 1-46: MoE (top-4 of 64 experts) + Shared Expert]
                         │
                    → residual
```

### Parallelism Strategy

```
                    ┌─────────────────────────────────┐
                    │         Device Mesh              │
                    │                                  │
                    │   TP: column/row-parallel split  │
                    │   DP: batch-level replication    │
                    │   EP: expert-level sharding      │
                    │                                  │
                    │   2D mesh: (dp, tp)              │
                    │   3D mesh: (dp, ep, tp)          │
                    └─────────────────────────────────┘
```

## Documentation

Detailed design documents and performance analysis are available in [docs/](docs/) (in Chinese):

- [KV Cache Design](docs/kv_cache_design.md)
- [Parallelism Implementation](docs/parallelism_implementation.md)
- [MFU/MBU Analysis](docs/mfu_mbu_analysis.md)
- [TP Benchmark Results](docs/tp_benchmark_results.md)
- [GLM-4.7-Flash Performance Analysis](docs/glm4flash_performance_analysis.md)
- [GLM-4.7-Flash Benchmark Analysis](docs/glm4flash_benchmark_analysis.md)

## License

This project is for educational purposes.
