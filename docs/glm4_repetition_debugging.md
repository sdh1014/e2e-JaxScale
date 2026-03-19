# GLM-4 生成质量问题：发现、探究与解决

## 一、问题发现

JaxScale 的 GLM-4-9B greedy decode 输出存在严重文本重复，而 HuggingFace 原版输出正常。

**对比数据**（`benchmarks/compare_hf_output.py`，5 个 prompt，greedy decode）：

| Prompt | HF Trigram 重复率 | JaxScale | HF Tokens | JaxScale Tokens |
|--------|------------------|----------|-----------|----------------|
| ML 解释 | 5.19% | **33.03%** | 159（停止） | 256（不停） |
| 海洋诗 | 1.14% | **27.04%** | 125（停止） | 256（不停） |
| Python vs Java | 0.56% | **26.40%** | 256 | 256 |
| 光合作用 | 4.17% | **19.08%** | 256 | 256 |
| 猫的故事 | 0.00% | 5.33% | 256 | 256 |
| **平均** | **2.21%** | **22.18%** | | |

两个核心异常：
1. **HF 正确停止但 JaxScale 不停** — Prompt 1/2 中 HF 在 159/125 tokens 时停止，JaxScale 跑满 256 tokens
2. **JaxScale 重复率远高于 HF** — 平均 22% vs 2%，差距 10 倍

---

## 二、问题探究

### 第 1 层：EOS Token 问题

`generation_config.json` 定义了 **3 个 EOS token**：

```json
{ "eos_token_id": [151329, 151336, 151338] }
```

但 JaxScale 使用 `tokenizer.eos_token_id`，只返回 1 个（151329）。当模型输出 151336 或 151338 时未识别为停止信号，继续生成导致退化。

HF 的 `generate()` 自动加载 `generation_config.json` 中全部 EOS token，因此能正确停止。

### 第 2 层：逐层精度偏差

修复 EOS 后，部分 prompt 仍有残余重复。使用 `benchmarks/diagnose_logits.py` 逐层对比（TP=2，4 芯片 TPU）：

| 层 | cos 相似度 | 趋势 |
|----|-----------|------|
| Embedding | 1.000000 | 完美 |
| Layer 0 | 0.997977 | 开始偏移 |
| Layer 19 | 0.990 | 缓慢下降 |
| Layer 39 | 0.984354 | 累积偏移 |
| Final Logits | 0.974708 | Top-1 一致，Top-4+ 分叉 |

权重完全匹配（cos=1.0），排除了加载错误。偏差从 Layer 0 开始逐层累积。

### 第 3 层：Layer 0 子组件定位

使用 `benchmarks/diagnose_layer0.py` 在单芯片 TPU（TP=1）上逐组件对比 Layer 0：

| 组件 | cos 相似度 | 判定 |
|------|-----------|------|
| embedding | 1.00000000 | 完美 |
| input_layernorm | 0.99999675 | 完美 |
| q_proj | 0.99999847 | 完美 |
| k_proj | 0.99999928 | 完美 |
| v_proj | 0.99999482 | 完美 |
| RoPE Q (vs HF 风格) | 0.99999903 | 完美 |
| **attention 输出 (o_proj 输入)** | **0.74233** | **偏差爆发点** |
| self_attn 输出 (after O proj) | 0.96072 | O proj 部分恢复 |
| post_attention_layernorm | 0.80648 | 被放大 |
| gate_proj | 0.60924 | 因输入已偏差 |
| layer 0 最终输出 | 0.94146 | 累积 |

**偏差精确链路**：

```
Q/K/V (cos=0.99999)  ← TPU MXU vs CPU bf16 matmul 的 1 ULP 差异
    ↓ RoPE (cos=0.99999, 无放大)
    ↓ QK^T matmul (128 维求和，微小误差累积)
    ↓ softmax (非线性放大: 微小 score 差异 → 不同注意力分布)
attention output (cos=0.742)  ← 偏差爆发
    ↓ O projection (4096 维 matmul 部分平均化)
self_attn output (cos=0.960)  ← 部分恢复
```

### 排除的假设

| 假设 | 测试方法 | 结论 |
|------|---------|------|
| 权重加载错误 | 对比所有权重 cos=1.0 | 排除 |
| RoPE 实现差异 | 用 HF Q/K 输入测试 cos=0.99999 | 排除 |
| bf16 softmax 精度 | 改为 fp32 softmax | 改善微小 (0.96019→0.96072)，非主因 |
| fp32 QK^T matmul | 改为 fp32 attention scores | 无改善，排除 |
| HF SDPA vs eager | 强制 `attn_implementation="eager"` | 差异相同 (cos=0.740)，排除 |
| gate_up_proj 拆分方式 | 对比 fused vs split 输出 | cos=1.0，排除 |

### 根因结论

偏差有两个独立原因：

1. **EOS Token 检测不全（bug）** — 只检查 1 个 EOS token，遗漏 2 个，导致生成不停止
2. **bf16 跨平台精度差异（固有行为）** — TPU MXU 的 256×256 矩阵单元与 CPU SIMD 使用不同的浮点累积路径，Q/K 的 1 ULP 差异经 softmax 非线性放大后形成不同的注意力分布。这是所有推理框架在不同硬件上 greedy decode 结果不同的根本原因

---

## 三、问题解决

### 修复 1：多 EOS Token 支持

**改动文件**：`configs/model_config.py`、`runner.py`、`main.py`

- `ModelConfig` 新增 `eos_token_id` 字段，从 `generation_config.json` 加载
- `greedy_generate` / `sample_generate` 支持 `int | list[int]` 类型 EOS
- EOS 检查改为 `jnp.isin(next_token, jnp.array(eos_token_ids))`

**效果**：Prompt 1 从 256 tokens 降到 154 tokens（正确停止），trigram 重复率 33%→12%

### 修复 2：fp32 Softmax

**改动文件**：`layers/attention.py`、`layers/mla_attention.py`

```python
# Before
attn_weights = jax.nn.softmax(attn_weights, axis=-1)
# After
attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(v_full.dtype)
```

与 HF eager attention 一致，是业界标准做法。

### 修复 3：`jax.nn.dot_product_attention`

**改动文件**：`layers/attention.py`

替换手动的 GQA repeat → transpose → QK^T → softmax → @V → transpose 为 JAX 原生 API：

```python
attn_output = jax.nn.dot_product_attention(
    q, k_full, v_full, bias=attention_mask,
    scale=self.scaling, implementation="xla",
)
```

**效果**：消除了 greedy decode 的 `a,a,a,a` 崩溃，代码从 20 行简化到 4 行，XLA 内部优化更好

### 修复 4：Temperature Sampling + Repetition Penalty

**改动文件**：`runner.py`、`main.py`

新增完整的采样管线：

```
logits → repetition_penalty → temperature → top_k(200) → top_p(0.8) → categorical
```

**关键设计**（参考 sglang-jax）：
- Sampling 逻辑独立 JIT 编译（`@jax.jit _jit_sample`）
- Repetition penalty 用 device 侧 presence 向量，避免动态 shape
- Top_p 在 top_k 预过滤后的 200 个候选上排序，而非全 vocab（155K→200，排序量减少 775 倍）

**使用方式**：

```bash
python main.py --model_path /path/to/glm-4 --prompt "..." --sample \
    --temperature 0.8 --top_p 0.8 --repetition_penalty 1.1
```

### 修复效果总览

| 指标 | 修复前 | 修复后 |
|------|-------|-------|
| Greedy 重复率 (Prompt 1) | 33.03% | 12.21% (EOS 修复) |
| Greedy `a,a,a` 崩溃 | 有 | 无 (dot_product_attention) |
| Sampling 质量 | 无 sampling 功能 | 连贯、无重复、正确停止 |
| Sampling 速度 | — | 10.0 tok/s (greedy 23.9 tok/s 的 42%) |

### Sampling 速度优化历程

| 版本 | 速度 | 瓶颈 |
|------|------|------|
| v0: Python 循环内无 JIT | 0.5 tok/s | 每步 Python dispatch |
| v1: 预分配 token 数组 | 5.9 tok/s | 每步重建 presence 向量 |
| v2: JIT sampler + 全 vocab 排序 | 2.5 tok/s | `argsort` 155K elements |
| **v3: JIT sampler + top_k 预过滤** | **10.0 tok/s** | 接近 greedy 的 2.4x |

### Sampling 输出示例

```
Prompt: Please explain what machine learning is in simple terms.

Output: Machine learning, simply put, is a branch of artificial intelligence (AI)
that enables computers to learn from data, rather than being explicitly programmed
to perform tasks by following instructions given by humans. Instead, it learns from
examples and improves its performance over time as it analyzes more data.
```

---

## 四、经验总结

1. **先查 EOS 再查精度** — 生成不停止 ≠ 模型精度问题，可能只是漏了 EOS token
2. **逐组件对比定位偏差** — 不要猜测，用 hook 捕获中间结果逐步排查
3. **bf16 跨平台 greedy decode 结果一定不同** — 这是固有行为，用 sampling 解决
4. **Sampling 不需要外部库** — JAX 原语（`lax.top_k`、`random.categorical`）足够
5. **Top_p 必须先 top_k 预过滤** — 对全 vocab 排序是性能陷阱
6. **Sampling 耗时不计入 benchmark** — 业界标准只测 model forward
