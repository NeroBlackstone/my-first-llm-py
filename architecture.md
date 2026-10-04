# 架构与参数量设计：40M 参数模型

目标：在 GTX 970（4GB 显存，实际高速部分 3.5GB，无 Tensor Core，只能 fp32 训练）上，训练一个 **0.4 亿（40M）参数** 的 GPT。

## 1. 参数量与模型大小换算

| 精度 | 每参数字节 | 40M 参数对应大小 | 适用场景 |
|---|---|---|---|
| fp32 | 4 | 160MB | GTX 970 训练（本方案） |
| bf16/fp16 | 2 | 80MB | 有 Tensor Core 的卡推理 |

本方案按 fp32 训练计算：**40M 参数 ≈ 160MB 权重文件**（本配置 40.7M ≈ 163MB；训练态占用见第 5 节）。

## 2. 推荐配置

```python
@dataclass
class GPTConfig:
    block_size: int = 512     # 最大序列长度
    vocab_size: int = 50257   # GPT-2 tiktoken 分词器
    n_layer: int = 12
    n_head: int = 6           # head_dim = n_embd / n_head = 64
    n_embd: int = 384
```

预算推导（词表固定 50257）：

- `wte` = 50257 × 384 = 19.3M（占 47%）
- 剩余 ~21M 给 Block：21M / (12 × 384² ≈ 1.77M/层) ≈ 12 层

备选方案（都接近 40M，取舍不同）：

| 配置 | 合计 | 特点 |
|---|---|---|
| **12 层 / 6 头 / 384 维（推荐）** | 40.7M | 深度适中，均衡 |
| 11 层 / 6 头 / 384 维 | 39.0M | 最贴近 40M 整数下取 |
| 4 层 / 8 头 / 512 维 | 38.6M | 更宽更浅（512 维），wte 占 67% |

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
| `wte` | 50257 × 384 | 19.30M | 77.2MB |
| 12 × Block | 12 × 12 × 384² | 21.23M | 84.9MB |
| `wpe` | 512 × 384 | 0.20M | 0.8MB |
| `ln_f` | 384 × 2 | 0.0008M | — |
| **合计** | | **≈ 40.7M** | **≈ 163MB** |

## 4. 约束与设计取舍

1. **词表占近一半**：`wte` 50257×384 = 19.3M，占 47%。
   - 若要求兼容 GPT-2 分词器（能加载官方预训练 embedding、用 tiktoken），保留 50257。
   - 若从零训练且不追求兼容，自训 8k 词表 BPE：`wte` 降到 8k × 384 = 3.1M，省下的 ~16M 全部加给层（可再加 9 层），同样 40M 时模型明显更强。
2. **n_head 必须整除 n_embd**（`train_gpt2.py:15`）。384 维配 6 头 → 每头 64 维，是标准做法。
3. **为什么是 12 层 / 384 维**：384 维已够表达一般语言模式；在 40M 预算下深度（12 层）比宽度（512 维 4 层）更有收益——深度带来更强的组合抽象。词表已占近一半，宽度再增加会把预算全吃进 `wte`。
4. **与 GPT-2 124M 的对比**（同一架构，只改配置）：

   | | 本方案 | GPT-2 small |
   |---|---|---|
   | n_layer | 12 | 12 |
   | n_head | 6 | 12 |
   | n_embd | 384 | 768 |
   | block_size | 512 | 1024 |
   | 参数量 | 40.7M | 124M |
   | 大小 (fp32) | 163MB | 497MB |

## 5. 训练显存（GTX 970，实测）

40.7M 全尺寸（12层/6头/384维/block512）、fp32、AdamW 实测（脚本见 `oom_test.py`，桌面占用 ~1GB，torch 可用 ~3.2GB）：

| 配置 | 结果 | 峰值显存 | 速度 |
|---|---|---|---|
| **B=4 × T=512（原目标）** | **❌ OOM** | 需 ~3.4GB > 可用 3.2GB | — |
| B=8 × T=256 | ❌ OOM | — | — |
| **B=3 × T=512（推荐）** | ✅ | 2206MB（余量 ~1GB） | 321ms/step |
| B=4 × T=384 | ✅ | 2206MB | 307ms/step |
| B=2 × T=512 | ✅ | 1644MB（reserved 1990MB） | 193ms/step |
| B=4 × T=256 | ✅ | 1644MB | 169ms/step |
| B=8 × T=128 | ✅ | 1644MB | 178ms/step |
| B=1 × T=512 | ✅ | 1076MB | 118ms/step |
| 生成（无梯度，64 token） | ✅ | 195MB | — |

显存构成（按实测外推）：

| 项 | 大小 |
|---|---|
| 权重 + 梯度 + Adam 状态（固定） | 652MB |
| 激活 + logits + 交叉熵工作区 | ~0.85-1.0MB/token（随 batch 略超线性） |
| 其中 logits/`log_softmax`（B×T×50257×4B，B4×T512 时两个 ~394MB 张量） | 单独 ~0.8GB |
| **B4×T512 合计** | **~3.4GB → OOM**（失败点：分配 394MB logits 时仅剩 302MB） |

早期手算的"~1.3-1.6GB"偏小：漏了 logits/交叉熵大张量，激活也比估算高一倍——以实测为准。

**经验规则：显存 ≈ 652MB（固定） + ~1MB/token，micro（B×T）≤ 1536 token 稳，2048 token 起 OOM。**

风险与对策：

- **推荐 micro = `B3×T512`**（实测 2206MB，余量 ~1GB）：batch 比 2 更大，且 T=512 让 `wpe` 的 256-511 号位置嵌入全部被训练（B4×T256 会让后半段位置从未更新）。
- 兜底：余量吃紧或桌面占用升高就退到 `B2×T512`（1644MB）；`B4×T512`、`B8×T256` 直接 OOM 别试。
- 桌面/浏览器占 ~1GB，训练前关掉能多出 1GB 余量；但 `B4×T512` 需 ~3.4GB，即使空卡也越过 GTX 970 的 3.5GB 高速分区（进入慢速 0.5GB 段），不建议硬上。
- **凑目标批量靠梯度累积**：micro=3 × T=512 × grad_accum=11 = 等效 16896 token/步，显存不变（build-nanogpt commit `01be6b3` 讲的就是这个）。
- `torch.compile` 与 FlashAttention（Triton 需 sm_70+）在 Maxwell 上不可用；`F.scaled_dot_product_attention` 退回 math 后端，功能正确但慢、且显存开销如上表。

## 6. 训练方案（分阶段，每阶段有验收标准）

总原则：**先小后大、先对后快**。每一阶段跑通、验收，再进下一阶段。

### 阶段 0：结构正确性（无数据，几分钟）

随机 token 输入（`torch.randint`），验收：

- `forward` 输出 shape = `(B, 512, 50257)`；初始 loss ≈ ln(50257) ≈ 10.8
- 参数量 = 40.7M（`sum(p.numel() for p in model.parameters())`）
- **单 batch 过拟合**：拿一个固定 batch 反复训练几十步，loss 能降到接近 0 → 证明"模型、loss、反向传播、优化器"整条链路无 bug（这是最经典的深度学习调试法）

### 阶段 1：tiny shakespeare 端到端（分钟级）

数据：项目自带 `build-nanogpt/input.txt`（约 1MB 莎士比亚）。

- 目的：跑通完整训练循环（dataloader → 前向 → 反向 → 优化 → 采样生成文本）
- 验收：loss 从 10.8 降到 **~1.5 以下**；能生成像样的英文句子
- 这阶段允许临时把配置缩小（如 4 层），只验证流程；也可以直接用全尺寸，1MB 数据几十个 epoch 很快

### 阶段 2：正式训练 FineWeb-Edu（小时级）

数据：用 `build-nanogpt/fineweb.py` 下采样 FineWeb-Edu，token 化后存 uint16 的 bin 文件（100M token ≈ 200MB 磁盘）。

- **token 预算：100-200M tokens**（学习目标，非效果最优）
  - 本机实测 `B3×T512` 每步 1536 token ÷ 321ms ≈ **4800 tokens/s**（含优化器；长跑以实测为准）→ 100-200M tokens ≈ **6-12 小时**
  - Chinchilla 最优比例是 20 token/参数 = 0.8B tokens（≈ 2 天），学习阶段不追
- 训练超参（nanoGPT 风格，起步值）：

  | 超参 | 值 |
  |---|---|
  | optimizer | AdamW, β=(0.9, 0.95), weight_decay=0.1（仅 2D 参数） |
  | 学习率 | max 1e-3 → 余弦退火至 1e-4，warmup 1000 步 |
  | grad clip | 1.0 |
  | batch | micro 3 × T512 × accum 11 = 等效 16896 token/步（B×T ≤ 1536；OOM 兜底 micro 2 × accum 16 = 16384） |
  | 精度 | fp32（Maxwell 无 bf16 加速） |

- 监控：每 100 步打印 train/val loss；训练中定期用少量 prompt 采样生成文本
- 验收：val loss 稳定下降且 train/val 不分叉（分叉 = 过拟合）；最终 val loss **~2.1-2.6**（数据/步数而定）
- 评测（可选）：用 `hellaswag.py` 算 HellaSwag 准确率（随机基线 ~25%，好模型 >30%）

### 后续（超出本方案）

- 数据/算力上去后重新对标 Chinchilla 比例；或换 8k 自训词表把预算全给层
- SFT 微调是另一课题（README 第 7 行：SFT = 换数据继续训）

## 7. 代码是否需要改

不需要改结构代码，`n_layer`/`n_head`/`n_embd`/`block_size`/`vocab_size` 全部是 `GPTConfig` 的配置项。仅有的两个注意点：

- `from_pretrained` 加载 GPT-2 官方权重需要 shape 完全一致；本配置 `n_embd=384 ≠ 768`，无法直接加载官方权重（若要兼容需 12 层/12 头/768 维，即 GPT-2 small，124M）。
- 初始化中的 `NANOGPT_SCALE_INIT` 依赖 `config.n_layer`，随配置自动生效。
