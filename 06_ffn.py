import torch
from torch import nn

torch.manual_seed(42)

# =========================================================================
# 阶段 1: GPT-2 的 FFN（前馈网络）—— 02 的 MLP 换上 05 的外壳
# =========================================================================
# Block 里的第二个子层。结构: Linear(C→4C) → GELU → Linear(4C→C)
# 和 02 的区别: 升维系数固定 4 倍 + 走 Pre-Norm 残差结构
BATCH_SIZE, SEQ_LEN, EMBED_DIM = 2, 5, 8

x: torch.Tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

ffn = nn.Sequential(
    nn.Linear(EMBED_DIM, 4 * EMBED_DIM),   # c_fc: 8 → 32, 升维
    nn.GELU(),                             # 非线性（GPT-2 用 GELU 不用 ReLU）
    nn.Linear(4 * EMBED_DIM, EMBED_DIM),   # c_proj: 32 → 8, 降回
)

y = ffn(x)
print("FFN 输出 shape:", y.shape)     # 期望: [2, 5, 8] 维度不变

# 中间层的形状（02 练习 a 的答案）: (2, 5, 32)
print("升维后 shape:", ffn[0](x).shape)        # [2, 5, 32]
print("GELU 后 shape:", ffn[1](ffn[0](x)).shape)  # [2, 5, 32]

# =========================================================================
# 阶段 2: 逐位置独立 —— FFN 不让 token 之间交换信息
# =========================================================================
# 注意力(04)让 token 互相看, FFN 只管"每个 token 自己怎么加工"
# 验证: 对每个 token 单独过 FFN, 结果必须和整批过完全一样
y_batch: torch.Tensor = ffn(x)
# 注意: x[b:b+1] 输出 (1,5,8), 要用 cat 拼接 (stack 会多出一维变成 (2,1,5,8))
y_per_token: torch.Tensor = torch.cat([ffn(x[b:b+1]) for b in range(BATCH_SIZE)])
print("逐 token 独立:", torch.allclose(y_batch, y_per_token, atol=1e-5))  # 期望: True

# 等价说法: FFN = 一个对最后一维做变换的函数, 对第 0、1 维(位置)是"无记忆"的
# 对比 04 的注意力: 打乱 token 顺序, 注意力输出会变, FFN 不变
perm = torch.randperm(SEQ_LEN)
print("FFN 打乱顺序结果一致:", torch.allclose(ffn(x[:, perm]), y_batch[:, perm]))  # 期望: True

# =========================================================================
# 阶段 3: 套上 05 的 Pre-Norm 外壳 = 完整 FFN 子层
# =========================================================================
#   x = x + FFN(LayerNorm(x))
class FFNSublayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return x + self.ffn(self.ln(x))   # Pre-Norm: 归一化 → FFN → 残差

sublayer = FFNSublayer(EMBED_DIM)
y = sublayer(x)
print("\nPre-Norm FFN 子层输出 shape:", y.shape)   # 期望: [2, 5, 8]

# =========================================================================
# 阶段 4: 参数量对账（对照 architecture.md 第 3 节账本）
# =========================================================================
n_params = sum(p.numel() for p in ffn.parameters())
c2 = EMBED_DIM**2
print("\nFFN 参数量:", n_params, " (不含 bias 应为 8C² =", 8 * c2, ")")
print("账本: c_fc = 4C² =", 4 * c2, ", c_proj = 4C² =", 4 * c2)
print("验证:", n_params - 4 * EMBED_DIM - EMBED_DIM == 8 * c2)  # 去掉两个 bias 后恰为 8C²

# =========================================================================
# 练习题
#    a. 为什么 FFN 中间维是 4 倍而不是 2 倍或 8 倍？（提示: 参数预算 vs 表达能力,
#       看 architecture.md 账本里 12C² 的构成, FFN 占 8C² 是最大的一块）
#    b. 把 GELU 换成 ReLU 打印几个负数输入的输出, 两者的区别是什么?
#    c. 参数量对账: 注意力(c_attn + c_proj)是 3C² + C² = 4C², FFN 是 8C²,
#       一个 Block 合计 12C²。用 n_embd=512 算一层多少参数？
#       （答案: 12 × 512² = 3.15M, ×17 层 = 53.6M, 对照 architecture.md）
#    d. 阶段 2 的 perm 实验: 为什么打乱顺序后 FFN 结果只是跟着挪位置, 内容不变？
#    e. 对照 train_gpt2.py: 找到 c_fc 和 c_proj 两个 Linear + GELU,
#       确认它的 forward 是 x + c_proj(gelu(c_fc(ln_2(x)))) 这个 Pre-Norm 顺序。
