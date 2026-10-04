# my-first-llm-py

从零手写并训练一个 GPT 的学习项目。用 PyTorch 逐脚本搭建 Transformer 的每个零件，最后拼成完整模型，在消费级显卡（GTX 970，4GB）上训练一个 40M 参数的语言模型。

## 仓库目的

1. **把 GPT 拆开看懂**：每个脚本只讲一个概念，从 token embedding 到完整 Transformer，手写公式与 `torch` 高层 API 对照，弄清每一层在算什么。
2. **跑通端到端训练**：dataloader → 前向 → 反向 → 优化 → 采样生成，不依赖任何训练框架。
3. **在小显存上真能训**：目标 40M 参数、fp32、GTX 970（无 Tensor Core），显存与超参均按实测调优（见 `architecture.md`）。

代码注释对照 nanoGPT 的 `train_gpt2.py` 逐段讲解，适合作为"我的第一个 LPM/LLM"的第一手实践记录。

## 学习路线（按序号运行）

| 脚本 | 主题 |
|---|---|
| `01_embedding.py` | Token embedding：离散 token → 向量 `(B, T, C)` |
| `02_mlp.py` | MLP：升维 4 倍 → GELU → 降回来 |
| `03_attention.py` | 手算 3 个 token 的注意力（QKV、缩放、mask） |
| `04_multihead_attention.py` | 多头注意力与 `scaled_dot_product_attention` |
| `05_layernorm_residual.py` | LayerNorm 手算与残差连接 |
| `06_ffn.py` | GPT-2 的 FFN |
| `07_transformer_block.py` | 把以上零件拼成一个 Transformer Block |
| `08_full_gpt.py` | 完整 GPT：embedding + N × block + lm_head |
| `09_train_gpt.py` | 训练循环：dataloader、lr 调度、梯度裁剪、top-k 生成 |
| `10_torch_abstractions.py` | 把 09 手写的训练侧换成 PyTorch 高层抽象（DataLoader、SequentialLR、tqdm、梯度累积、EMA、断点续训） |
| `oom_test.py` | 40M 全尺寸配置在 GTX 970 上的显存实测 |
| `architecture.md` | 40M 模型的参数量账本、显存分析、分阶段训练方案 |

## 环境

- Python 3.14、Poetry、PyTorch 2.14（CUDA 12.6）
- 数据：`data/shakespeare_gpt2.bin`（GPT-2 tiktoken 分词后的 uint16 token，已被 git 忽略，需自行生成）

```bash
poetry install
```

## 运行

```bash
poetry run python 01_embedding.py        # 从头按序号学
poetry run python 09_train_gpt.py        # 训练
poetry run python 10_torch_abstractions.py
poetry run python oom_test.py            # 显存实测
```

## 模型配置

```python
GPTConfig(
    block_size=512,      # 最大序列长度
    vocab_size=50257,    # GPT-2 tiktoken 分词器
    n_layer=12,
    n_head=6,            # head_dim = 64
    n_embd=384,
)                        # ≈ 40.7M 参数，fp32 约 163MB
```

训练要点（详见 `architecture.md`）：

- 推荐 micro batch `B=3 × T=512`，峰值显存约 2.2GB；靠梯度累积凑大 batch
- AdamW + 余弦退火 + warmup + grad clip 1.0，fp32（Maxwell 无 Tensor Core）
- 验收标准：tiny shakespeare 上 loss 降到 ~1.5 以下，能生成通顺英文

## 参考

- [nanoGPT](https://github.com/karpathy/nanoGPT) / `train_gpt2.py`（Karpathy）
- [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
