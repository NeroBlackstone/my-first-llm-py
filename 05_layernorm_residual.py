import torch
from torch import nn

torch.manual_seed(42)

# =========================================================================
# 阶段 1: LayerNorm 手算 —— 为什么需要归一化
# =========================================================================
# 04 的多头注意力/02 的 MLP 输出没有归一化: 经过几十层后数值会越飘越远
# (每层的线性变换都会放大或缩小激活值)。LayerNorm 把每个 token 的向量
# 拉回均值 0、方差 1, 再用可学习的 gamma/beta 恢复表达能力。
BATCH_SIZE, SEQ_LEN, EMBED_DIM = 2, 5, 8

x: torch.Tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 7 + 3  # 故意放大偏移
print("归一化前  mean/std:", x.mean().item(), x.std().item())  # ≈ 3 / 7

# 手算: 在最后一维上算均值方差（每个 token 自己算自己, 与其他 token 无关）
mean = x.mean(dim=-1, keepdim=True)              # Tensor[float32], shape (2, 5, 1)
var = x.var(dim=-1, keepdim=True, unbiased=False)  # shape (2, 5, 1), 除 N 而非 N-1
manual_ln = (x - mean) / torch.sqrt(var + 1e-5)   # shape (2, 5, 8), 每个 token: mean≈0 std≈1
print("手算后  mean/std:", manual_ln.mean().item(), manual_ln.std().item())  # ≈ 0 / 1

# 对照 nn.LayerNorm: 默认 eps=1e-5, elementwise_affine=True
ln = nn.LayerNorm(EMBED_DIM)
y: torch.Tensor = ln(x)
print("nn.LayerNorm == 手算:", torch.allclose(y, manual_ln, atol=1e-5))  # 期望: True(初始 gamma=1 beta=0)

# 验证因果性无关: 位置 0 的归一化结果只依赖 x[0], 不碰其他 token
print("每个 token 独立归一化:", torch.allclose(y[0, 0], ln(x[0:1])[0, 0]))

# =========================================================================
# 阶段 2: 残差连接（skip connection）—— 让梯度能走深层
# =========================================================================
# 问题: 纯堆叠 sublayer(x), 深层梯度会消失/爆炸。残差把输入"原样保留":
#   y = x + sublayer(x)
# 这样反向传播时梯度永远有一条"高速公路"直接回到浅层。
sublayer = nn.Sequential(
    nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
    nn.GELU(),
    nn.Linear(4 * EMBED_DIM, EMBED_DIM),
)

out_no_residual = sublayer(x)          # 不带残差
out_residual = x + sublayer(x)         # 带残差: 输出 = 输入 + 变化量

print("无残差输出 std:", out_no_residual.std().item())   # 每层随机线性变换后分布漂移
print("有残差输出 std:", out_residual.std().item())      # 因为有 x 加回来, 分布更稳
print("两者 shape 相同:", out_no_residual.shape == out_residual.shape)  # True, 都是 (2,5,8)

# 残差的直觉: sublayer 只需学"改动量"（delta）, 不用重新学整个映射
print("残差 = 输入 + 改动:", torch.allclose(out_residual, x + sublayer(x)))

# =========================================================================
# 阶段 3: 完整 Transformer 子层 = LayerNorm → sublayer → 残差（Pre-Norm 结构）
# =========================================================================
# GPT-2 用的是 Pre-Norm: 先归一化再进 sublayer, 残差在外层
#   x = x + sublayer(LayerNorm(x))
# （Post-Norm 是 x = LayerNorm(x + sublayer(x)), 训练更难稳定, 见练习 d）
class SublayerBlock(nn.Module):
    def __init__(self, embed_dim, sublayer):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(self.ln(x))   # Pre-Norm 残差: 归一化 → 变换 → 加回原输入

attn_like = nn.Sequential(   # 用一个假的注意力形状占位（真实版见 04 的 c_proj 前）
    nn.Linear(EMBED_DIM, EMBED_DIM),
)
block = SublayerBlock(EMBED_DIM, attn_like)
y = block(x)
print("\nPre-Norm 子层输出 shape:", y.shape)   # 期望: [2, 5, 8] 维度不变

# =========================================================================
# 练习题
#    a. 阶段 1 去掉 keepdim=True, x - mean 会怎样？（广播形状对不上会报错还是错算？）
#    b. LayerNorm 的 gamma/beta 初始值是什么？训练后会变成什么？（打印 ln.weight, ln.bias）
#    c. 梯度实验: 对 10 层堆叠的 Linear 分别算 out.mean() 对输入的梯度,
#       带残差的和不带的谁衰减快？（提示: x.requires_grad_(), torch.autograd.grad）
#    d. 把 SublayerBlock 改成 Post-Norm: forward 里 return self.ln(x + self.sublayer(x)),
#       查资料说说为什么深层网络更难训。
#    e. 对照 train_gpt2.py: 找到 ln_1/ln_2 和 residual 相加的那几行, 确认是 Pre-Norm。
