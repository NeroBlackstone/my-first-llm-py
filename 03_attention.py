import torch
from torch import nn
from torch.nn.functional import softmax

torch.manual_seed(42)

# =========================================================================
# 阶段 1: 3 个 token 的注意力手算
# =========================================================================
SEQ_LEN, EMBED_DIM = 3, 4  # 3 个 token, 每个 4 维

x: torch.Tensor = torch.randn(SEQ_LEN, EMBED_DIM)   # Tensor[float32], shape (3, 4)
proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
Q: torch.Tensor = proj(x)                     # Tensor[float32], shape (3, 4) 我要找什么
K: torch.Tensor = proj(x)                     # Tensor[float32], shape (3, 4) 我有什么
V: torch.Tensor = proj(x)                     # Tensor[float32], shape (3, 4) 我的内容

# 打分: scores[i, j] = 位置 i 对位置 j 的兴趣度（点积 = 相似度）
scores: torch.Tensor = Q @ K.T / EMBED_DIM**0.5  # Tensor[float32], shape (3, 3)，除以 sqrt(EMBED_DIM) 防止分数过大
print("打分矩阵:\n", scores)

# 因果掩码: 位置 i 只能看 0..i, 看不到未来的 token
mask: torch.Tensor = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)) == 0  # Tensor[bool], shape (3, 3)，上三角为 True（要屏蔽的位置）
scores = scores.masked_fill(mask, float("-inf"))  # Tensor[float32], shape (3, 3)，被屏蔽处为 -inf
print("掩码后（-inf 会被 softmax 变成 0）:\n", scores)

# softmax: 每行变成"和为 1"的权重
weights: torch.Tensor = softmax(scores, dim=-1)  # Tensor[float32], shape (3, 3)
print("注意力权重（每行和=1）:\n", weights)

# 加权求和 V
out: torch.Tensor = weights @ V               # Tensor[float32], shape (3, 4)
print("输出 shape:", out.shape)

# 验证因果性: 位置 0 的输出只来自 V[0]（它谁也看不见）
print("位置0只看自己:", torch.allclose(out[0], weights[0, 0] * V[0]))

# =========================================================================
# 练习题
#    a. 去掉掩码，位置 0 的输出会看到什么？为什么违背因果性？
#    b. 去掉 "/ EMBED_DIM**0.5"，打印 softmax 权重看变化（更尖锐？为什么危险）
# =========================================================================
