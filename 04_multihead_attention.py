import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention, softmax

torch.manual_seed(42)

# =========================================================================
# 阶段 2: 批量 + scaled_dot_product_attention 一步到位（公式同 03_attention.py）
# =========================================================================
BATCH_SIZE, SEQ_LEN, EMBED_DIM = 2, 5, 8     # 对齐 02 的输出形状 (2, 5, 8)
x: torch.Tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)  # Tensor[float32], shape (2, 5, 8)

qkv: nn.Linear = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)  # nn.Linear，非张量；可调用，输出 (2, 5, 24)，一次投影出 Q/K/V（同 train_gpt2.py:18）
Q: torch.Tensor
K: torch.Tensor
V: torch.Tensor
Q, K, V = qkv(x).split(EMBED_DIM, dim=-1)    # 中间结果 Tensor[float32] (2, 5, 24) → 各 (2, 5, 8)

out: torch.Tensor = scaled_dot_product_attention(Q, K, V, is_causal=True)  # Tensor[float32], shape (2, 5, 8)
print("SDPA 输出 shape:", out.shape)          # 期望: [2, 5, 8] 维度不变

# 验证: 手算应该和 SDPA 结果一致
scores: torch.Tensor = Q @ K.transpose(-2, -1) / EMBED_DIM**0.5  # Tensor[float32], shape (2, 5, 5)
scores = scores.masked_fill(
    torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)) == 0, float("-inf")
)                                             # Tensor[float32], shape (2, 5, 5)，上三角为 -inf
manual: torch.Tensor = softmax(scores, dim=-1) @ V  # Tensor[float32], shape (2, 5, 8)
print("手算 == SDPA:", torch.allclose(manual, out, atol=1e-5))

# =========================================================================
# 阶段 3: 多头注意力（完整形态，对照 train_gpt2.py:12-40）
# =========================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)   # QKV 投影
        self.c_proj = nn.Linear(embed_dim, embed_dim)       # 输出投影

    def forward(self, x):
        BATCH_SIZE, SEQ_LEN, EMBED_DIM = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(EMBED_DIM, dim=-1)

        # 拆头: (BATCH, SEQ, EMBED) -> (BATCH, n_head, SEQ, head_dim)
        # 把 8 维向量切成 n_head 份，每份独立做注意力
        head_dim = EMBED_DIM // self.n_head
        q = q.view(BATCH_SIZE, SEQ_LEN, self.n_head, head_dim).transpose(1, 2)
        k = k.view(BATCH_SIZE, SEQ_LEN, self.n_head, head_dim).transpose(1, 2)
        v = v.view(BATCH_SIZE, SEQ_LEN, self.n_head, head_dim).transpose(1, 2)

        y = scaled_dot_product_attention(q, k, v, is_causal=True)

        # 合头: (BATCH, n_head, SEQ, head_dim) -> (BATCH, SEQ, EMBED)
        y = y.transpose(1, 2).contiguous().view(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
        return self.c_proj(y)

attn = CausalSelfAttention(embed_dim=8, n_head=4)
y = attn(torch.randn(2, 5, 8))
print("\n多头输出 shape:", y.shape)           # 期望: [2, 5, 8] 维度不变

# =========================================================================
# 练习题
#    a. 阶段 2 去掉 is_causal=True，手算还会等于 SDPA 吗？为什么？
#    b. 阶段 3 把 n_head 改成 3 会发生什么？改成 8 呢？
#    c. 为什么要先 view 再 transpose，而不是直接 view 成 (B, n_head, T, head_dim)？
#       （提示: 内存布局，动手试: torch.arange(24).view(2,3,4) 与 .permute 后再 view）
#    d. c_proj 在多头之后做什么？去掉它模型会缺什么？
# =========================================================================
