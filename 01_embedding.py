import torch
from torch import nn

BATCH_SIZE, SEQ_LEN, EMBED_DIM = 2, 5, 8  # batch, 序列长度, 向量维度

# 1. 输入：2 句话，每句 5 个 token，词表大小 100
x = torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN))
print("输入 shape:", x.shape)          # 期望: [2, 5]

# 2. embedding 层 = 一张 100×8 的查表
emb = nn.Embedding(100, EMBED_DIM)
y = emb(x)
print("输出 shape:", y.shape)          # 期望: [2, 5, 8]

# 3. 验证"就是取表的第 id 行"
print("查表验证:", torch.equal(y[0, 0], emb.weight[x[0, 0]]))  # 期望: True

# 4. 练习题（改代码验证你的答案）
#    a. y[0, 0] 和 y[1, 0] 相同吗？为什么？（两句话的第 0 个 token）
#    b. 怎么让两次运行结果一样？（提示: torch.manual_seed）
#    c. emb.weight 的 shape 是多少？
