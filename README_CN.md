# JaxScale

**从零构建的 JAX/Flax LLM 推理框架，用于学习和性能分析。实现参考了 [sglang-jax](https://github.com/sgl-project/sglang-jax)。**

[English](README.md)

JaxScale 在 JAX/Flax NNX 上实现了 GLM 系列模型的完整推理流水线，支持张量并行（TP）、数据并行（DP）和专家并行（EP），可在 TPU/GPU 集群上运行。

## 支持的模型

| 模型 | 架构 | 参数量 | 注意力机制 | FFN |
|------|------|--------|-----------|-----|
| **GLM-4-9B** | Dense Transformer | 9B（18 GB bf16） | GQA（32Q / 2KV） | SwiGLU |
| **GLM-4.7-Flash** | MoE Transformer | 总量 30B / 激活 3B（bf16） | MLA | 64 路由专家 + 1 共享专家 |

## 核心特性

- **KV Cache** — 预分配静态数组 + `jax.lax.dynamic_update_slice`，实现 100 倍 decode 加速
- **GQA 注意力** — Grouped-Query Attention，16:1 Q/KV head 比例，RoPE 位置编码
- **MLA 注意力** — Multi-head Latent Attention，低秩 KV 压缩（kv_lora_rank=512）
- **MoE 层** — Softmax top-4 路由，64 个路由专家 + 1 个共享专家
- **张量并行** — 注意力和 MLP 权重的列/行并行切分
- **专家并行** — 基于 `shard_map` 的 EP，本地 gather + 单次 `psum` 实现高效跨设备通信
- **数据并行** — 计划中（暂未实现）
- **HuggingFace 权重加载** — 直接从 SafeTensors 加载，自动处理转置和拆分变换
- **性能分析套件** — 吞吐、延迟、MFU、MBU 分析及组件级 profiling

## 性能指标

### GLM-4-9B

#### L4 GPU（24 GB 显存，300 GB/s 带宽，60 TFLOPS bf16）

| 阶段 | 配置 | 吞吐量 | 效率 |
|------|------|--------|------|
| Prefill | batch=1, seq=1024 | 2,836 tok/s | **MFU 83.0%** |
| Decode | batch=1 | 13.8 tok/s | **MBU 80.8%** |

#### TPU v6e（32 GB HBM，1,600 GB/s 带宽，918 TFLOPS bf16）

**单芯片：**

| 阶段 | 配置 | 吞吐量 | 效率 |
|------|------|--------|------|
| Prefill | batch=32, seq=128 | 16,915 tok/s | MFU 32.4% |
| Decode | batch=1 | 65 tok/s | MBU 76.5% |
| Decode | batch=32 | 1,915 tok/s | MBU 70.3% |

**TP=2（2 芯片）：**

| 阶段 | 配置 | 吞吐量 | 效率 | 对比 TP=1 |
|------|------|--------|------|----------|
| Decode | batch=1 | 122 tok/s | MBU 71.4% | 1.9x |
| Decode | batch=8 | 961 tok/s | MBU 70.6% | 1.9x |
| Decode | batch=32 | 3,064 tok/s | MBU 56.2% | 1.6x |
| Prefill | batch=32, seq=512 | 17,806 tok/s | MFU 17.0% | 1.4x |
| Prefill | batch=32, seq=1024 | 16,220 tok/s | MFU 15.5% | 解锁（TP=1 OOM） |

### GLM-4.7-Flash

#### TPU v6e-4（TP=2 + EP=2，4 芯片）

| 阶段 | 配置 | 吞吐量 | 效率 |
|------|------|--------|------|
| Decode | batch=1 | 10.2 tok/s | MBU 5.7% |

> **瓶颈分析：** MLA Attention 占每层 65% 的延迟，但仅读取 24.5 MB 权重——瓶颈在于 kernel launch 开销（7 个串行 matmul）和 sharding 转换，而非计算或带宽。详见[性能分析与优化路线](docs/glm4flash_performance_analysis.md)（shard_map 包裹、Absorbed MLA、Pallas 融合），目标 67 tok/s。

### 优化历程（GLM-4-9B）

| 阶段 | Decode (tok/s) | 瓶颈 |
|------|---------------|------|
| 无 KV Cache，无 JIT | 0.1 | 重复计算 |
| + KV Cache | 1.5 | Python dispatch 开销 |
| + jax.jit | 13.8 | 显存带宽（物理极限） |
| + TP=2（TPU） | 122 | 显存带宽（每设备减半） |

## 快速开始

### 依赖

```
jax
flax
transformers
safetensors
numpy
```

### 单设备推理

```bash
python main.py \
  --model_path /path/to/glm-4-9b-chat-hf \
  --prompt "你好，请介绍一下自己" \
  --max_new_tokens 128 \
  --dtype bfloat16
```

### 张量并行推理（2 设备）

```bash
python main.py \
  --model_path /path/to/glm-4-9b-chat-hf \
  --prompt "你好" \
  --tp 2 --dp 1
```

### GLM-4.7-Flash 混合并行

```bash
python main.py \
  --model_path /path/to/glm-4.7-flash \
  --prompt "法国的首都是哪里？" \
  --tp 2 --ep 2
```

### 性能测试

```bash
python benchmarks/benchmark.py \
  --model_path /path/to/glm-4-9b-chat-hf \
  --tp 1 --dp 1

# 组件级 profiling（GLM-4.7-Flash）
python benchmarks/profile_glm4flash.py \
  --model_path /path/to/glm-4.7-flash \
  --tp 2 --ep 2
```

## 项目结构

```
JaxScale/
├── main.py                      # 推理入口
├── runner.py                    # 生成循环（贪心、采样）
├── configs/
│   └── model_config.py          # GLM-4-9B & GLM-4.7-Flash 统一配置
├── models/
│   ├── glm4.py                  # GLM-4-9B 稠密模型
│   └── glm4_flash.py            # GLM-4.7-Flash MoE 模型
├── layers/
│   ├── attention.py             # GQA 注意力
│   ├── mla_attention.py         # MLA（Multi-head Latent Attention）
│   ├── moe.py                   # MoE 层、路由器、共享专家
│   ├── linear.py                # 支持 sharding 的线性层
│   ├── embedding.py             # 嵌入层 & 并行 LM Head
│   ├── normalization.py         # RMSNorm
│   ├── rotary.py                # 旋转位置编码（RoPE）
│   └── kv_cache.py              # KV Cache
├── utils/
│   ├── weight_utils.py          # HuggingFace 权重加载与分片
│   └── mesh_utils.py            # 设备网格创建（TP, DP, EP）
├── benchmarks/
│   ├── benchmark.py             # 吞吐与延迟测试
│   ├── verify_kvcache.py        # KV Cache 正确性验证
│   └── profile_glm4flash.py     # 组件级性能分析
└── docs/                        # 设计文档与分析
```

## 架构

### GLM-4-9B（稠密模型）

```
输入 → Embedding → 40× DecoderLayer → RMSNorm → LM Head → 输出
                        │
                   RMSNorm → GQA Attention（32Q/2KV + RoPE）→ 残差
                        │
                   RMSNorm → SwiGLU MLP（4096→13696→4096）→ 残差
```

### GLM-4.7-Flash（MoE + MLA）

```
输入 → Embedding → 47× DecoderLayer → RMSNorm → LM Head → 输出
                        │
                   RMSNorm → MLA Attention（低秩 KV 压缩 + RoPE）→ 残差
                        │
                   RMSNorm → [第 0 层: Dense MLP]
                             [第 1-46 层: MoE（top-4 / 64 专家）+ 共享专家]
                        │
                   → 残差
```

### 并行策略

```
                    ┌──────────────────────────────────┐
                    │           设备网格                │
                    │                                  │
                    │   TP: 列/行并行切分               │
                    │   DP: batch 级复制               │
                    │   EP: 专家级切分                  │
                    │                                  │
                    │   2D 网格: (dp, tp)              │
                    │   3D 网格: (dp, ep, tp)          │
                    └──────────────────────────────────┘
```

## 文档

详细的设计文档和性能分析位于 [docs/](docs/) 目录：

- [KV Cache 设计](docs/kv_cache_design.md)
- [并行实现](docs/parallelism_implementation.md)
- [MFU/MBU 分析](docs/mfu_mbu_analysis.md)
- [TP 性能测试结果](docs/tp_benchmark_results.md)
- [GLM-4.7-Flash 性能分析](docs/glm4flash_performance_analysis.md)
- [GLM-4.7-Flash Benchmark 分析](docs/glm4flash_benchmark_analysis.md)

## License

本项目用于学习目的。
