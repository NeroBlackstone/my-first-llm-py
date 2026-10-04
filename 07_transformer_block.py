import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

torch.manual_seed(42)

BATCH_SIZE, SEQ_LEN, EMBED_DIM, N_HEAD = 2, 5, 8, 4  # 对齐 04/06 的形状

# =========================================================================
# 零件复用: 04 的注意力 + 06 的 FFN（原样搬来, 不改）
# =========================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FFN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        return self.c_proj(nn.functional.gelu(self.c_fc(x)))

# =========================================================================
# 阶段 1: 组装 Transformer Block = 两个 Pre-Norm 子层串联
# =========================================================================
# 对照 train_gpt2.py:60-68 逐行一致
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)          # 归一化(注意力前)
        self.attn = CausalSelfAttention(embed_dim, n_head)  # 子层1: 跨 token 交换信息
        self.ln_2 = nn.LayerNorm(embed_dim)          # 归一化(FFN 前)
        self.mlp = FFN(embed_dim)                    # 子层2: 逐 token 加工

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Pre-Norm 残差: 归一化 → 注意力 → 加回
        x = x + self.mlp(self.ln_2(x))    # Pre-Norm 残差: 归一化 → FFN → 加回
        return x

x: torch.Tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
block = TransformerBlock(EMBED_DIM, N_HEAD)
y = block(x)
print("Block 输入:", tuple(x.shape), "→ 输出:", tuple(y.shape))   # [2,5,8] → [2,5,8] 维度不变

# 验证两个子层确实各干各的: 注意力让 token 互相看, FFN 不会
# （对比: 若把 attn 换成恒等映射, 输出仍合法, 但 token 间不再交换信息）

# =========================================================================
# 阶段 2: 堆 N 层 —— 维度不变才能堆深
# =========================================================================
N_LAYER = 4
blocks = nn.Sequential(*[TransformerBlock(EMBED_DIM, N_HEAD) for _ in range(N_LAYER)])

# 打印每层输出的 mean/std, 观察 LayerNorm 如何压住数值漂移
h = x
print("\n逐层数值 (mean/std):")
print(f"  输入      {h.mean().item():+.4f} / {h.std().item():.4f}")
for i, blk in enumerate(blocks):
    h = blk(h)
    print(f"  block{i}    {h.mean().item():+.4f} / {h.std().item():.4f}")  # 量级稳定不发散
print("堆叠后 shape:", tuple(h.shape))   # [2,5,8] 还是不变

# =========================================================================
# 阶段 3: 参数量对账（对照 architecture.md 第 3 节账本）
# =========================================================================
def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

c2 = EMBED_DIM**2
print("\n参数量账本 (C = %d):" % EMBED_DIM)
print(f"  c_attn (3C²+3C): {count(block.attn.c_attn):>6}   理论 {3*c2 + 3*EMBED_DIM}")
print(f"  attn c_proj (C²+C): {count(block.attn.c_proj):>4}   理论 {c2 + EMBED_DIM}")
print(f"  c_fc   (4C²+4C): {count(block.mlp.c_fc):>6}   理论 {4*c2 + 4*EMBED_DIM}")
print(f"  mlp c_proj (4C²+C): {count(block.mlp.c_proj):>4}   理论 {4*c2 + EMBED_DIM}")
print(f"  2×LayerNorm: {count(block.ln_1) + count(block.ln_2):>14}   理论 {4*EMBED_DIM}")
print(f"  ---")
total = count(block)
ln_params = 4 * EMBED_DIM                       # 2 个 LN × (weight C + bias C)
bias = EMBED_DIM * 3 + EMBED_DIM + EMBED_DIM * 4 + EMBED_DIM   # 4 个 Linear 的 bias
pure = total - ln_params                        # 去掉 LN 后 = 12C² + Linear 的 bias
print(f"  单 Block 实测: {total}")
print(f"  去 LN 后: {pure}  vs 12C² + bias = {12*c2 + bias}  验证: {pure == 12*c2 + bias}")

# 全模型预算 (architecture.md 配置, C=512 时)
C512 = 512
block_params = 12 * C512**2
wte = 50257 * C512
wpe = 512 * C512
total_80m = wte + wpe + 17 * block_params
print(f"\n80M 预算 (17层/8头/512维):")
print(f"  wte       {wte/1e6:>7.2f}M")
print(f"  wpe       {wpe/1e6:>7.2f}M")
print(f"  17×Block  {17*block_params/1e6:>7.2f}M")
print(f"  合计      {total_80m/1e6:>7.2f}M   (architecture.md 目标 79.6M)")

# =========================================================================
# 练习题
#    a. 为什么 ln_1 和 ln_2 分开而不共用一个 LayerNorm？
#       （提示: 两个子层的输出分布不同, 归一化要贴着各自 sublayer 的输入）
#    b. 把 N_LAYER 改成 10, 在 __init__ 里去掉 ln_1 再看阶段 2 的逐层 std,
#       数值会怎么漂？（动手试）
#    c. 注意力和 mlp 的顺序对调（先 mlp 后 attn）理论上能训吗？
#       搜 "pre-norm order ablation" 或看 GPT-2/LLaMA 都是什么顺序, 有无影响。
#    d. nn.Sequential(*blocks) 和 nn.ModuleList(blocks) 装法有区别吗？
#       （提示: 查文档, 两者都注册参数; Sequential 多了按序调用和索引名）
#    e. 对照 train_gpt2.py:60-68, 逐行确认 forward 的顺序和本文件一致;
#       再找到 n_layer=17/12 层在哪一行被循环创建。
