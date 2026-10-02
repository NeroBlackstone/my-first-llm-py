# 架构与参数量设计：80MB 模型

目标：在 GTX 970（4GB 显存，实际高速部分 3.5GB，无 Tensor Core，只能 fp32 训练）上，训练一个总大小约 80MB 的 GPT。

## 1. 硬盘大小 -> 参数量换算

| 精度 | 每参数字节 | 80MB 对应参数量 | 适用场景 |
|---|---|---|---|
| fp32 | 4 | 20M | GTX 970 训练（本方案） |
| bf16/fp16 | 2 | 40M | 有 Tensor Core 的卡推理 |

本方案按 fp32 训练计算：**80MB ≈ 20M 参数**。

## 2. 推荐配置

```python
@dataclass
class GPTConfig:
    block_size: int = 512     # 最大序列长度
    vocab_size: int = 50257   # GPT-2 tiktoken 分词器
    n_layer: int = 8
    n_head: int = 4           # head_dim = n_embd / n_head = 64
    n_embd: int = 256
```

## 3. 参数量账本

GPT-2 结构中每层 Block 的参数量（`n_embd` 记为 C）：

| 组件 | 形状 | 参数量 |
|---|---|---|
| qkv 投影 `c_attn` | C × 3C (+3C bias) | 3C² |
| 输出投影 `c_proj` | C × C | C² |
| MLP `c_fc` | C × 4C | 4C² |
| MLP `c_proj` | 4C × C | 4C² |
| 2 × LayerNorm | 2C × 2 | 忽略不计 |
| **每 Block 合计** | | **≈ 12C²** |

模型总量（`lm_head` 与 `wte` 共享权重，`train_gpt2.py:94`，不重复计）：

```
total = vocab_size × n_embd        # wte（token embedding）
      + block_size × n_embd        # wpe（position embedding）
      + n_layer × 12 × n_embd²     # Transformer blocks
      + lm_head                    # = 0（共享）
```

代入推荐配置：

| 部分 | 计算 | 参数量 | fp32 大小 |
|---|---|---|---|
| `wte` | 50257 × 256 | 12.87M | 51.5MB |
| 8 × Block | 8 × 12 × 256² | 6.29M | 25.2MB |
| `wpe` | 512 × 256 | 0.13M | 0.5MB |
| **合计** | | **≈ 19.3M** | **≈ 77MB** |

## 4. 约束与设计取舍

1. **词表是大头**：`wte` 占总参数约 2/3，却只承担查表功能。
   - 若要求兼容 GPT-2 分词器（能加载官方预训练 embedding、用 tiktoken），保留 50257。
   - 若从零训练且不追求兼容，自训 8k 词表 BPE：`wte` 降到 8000 × n_embd，省下的参数全部加给层，同为 80MB 时模型深得多，效果明显更好。
2. **n_head 必须整除 n_embd**（`train_gpt2.py:15`）。256 维配 4 头 → 每头 64 维，是标准做法。
3. **为什么是 8 层而不是更深**：256 维时 12C² ≈ 786K/层，若把全部 20M 都给层（配小词表）可到 20+ 层，但 256 维的表达能力会成为瓶颈；8 层 256 维是与词表达能力的均衡点。
4. **与 GPT-2 124M 的对比**（同一架构，只改配置）：

   | | 本方案 | GPT-2 small |
   |---|---|---|
   | n_layer | 8 | 12 |
   | n_head | 4 | 12 |
   | n_embd | 256 | 768 |
   | block_size | 512 | 1024 |
   | 参数量 | 19.3M | 124M |
   | 大小 (fp32) | 77MB | 497MB |

## 5. 训练显存估算（GTX 970）

以 batch=8、seq=512、fp32、AdamW 计：

| 项 | 估算 |
|---|---|
| 权重 | 77MB |
| 梯度 | 77MB |
| Adam 状态（m, v） | 154MB |
| 激活（8 层，无 checkpointing） | ~200-500MB |
| **合计** | **< 1GB**，3.5GB 内很宽裕 |

每步耗时预计零点几秒，适合快速迭代。

注意：`torch.compile` 与 FlashAttention（Triton 需 sm_70+）在 Maxwell 上不可用，`F.scaled_dot_product_attention` 会退回到 math 实现，功能不受影响但较慢。

## 6. 代码是否需要改

不需要改结构代码，`n_layer`/`n_head`/`n_embd`/`block_size`/`vocab_size` 全部是 `GPTConfig` 的配置项。仅有的两个注意点：

- `from_pretrained` 加载 GPT-2 官方权重需要 shape 完全一致；本配置除 `vocab_size` 外均不同，无法直接加载（若想加载，需 `n_layer=12, n_head=12, n_embd=768`，代价是远超 80MB）。
- 初始化中的 `NANOGPT_SCALE_INIT` 依赖 `config.n_layer`，随配置自动生效。
