import torch
from torch import nn

EMBED_DIM = 8

x = torch.randn(2, 5, EMBED_DIM)  # 01 的输出形状: 向量 (B, T, C)

# MLP: 先升维 4 倍，过 GELU 激活，再降回来
mlp = nn.Sequential(
    nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
    nn.GELU(),
    nn.Linear(4 * EMBED_DIM, EMBED_DIM),
)

y = mlp(x)
print("输入 shape:", x.shape)   # 期望: [2, 5, 8]
print("输出 shape:", y.shape)   # 期望: [2, 5, 8]  ← 维度不变！

# 练习题
#    a. 中间那层之后的 shape 是多少？（打印 Sequential 逐层输出试试）
#    b. mlp 里一共有多少个可训练参数？（提示: sum(p.numel() for p in mlp.parameters())）
#    c. 为什么激活函数不能去掉？（提示: 两个 Linear 连写等于一个 Linear）
