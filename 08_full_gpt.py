import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention

torch.manual_seed(42)

# =========================================================================
# 零件复用: 07 的三个类原样搬来
# 给两个 c_proj 挂 NANOGPT_SCALE_INIT 标记 (对照 train_gpt2.py:21,49),
# 为什么需要它见阶段 2 的 _init_weights
# =========================================================================
# train_gpt2.py:21 的挂法是 self.c_proj.NANOGPT_SCALE_INIT = 1 (实例属性):
#   - Pylance: nn.Module.__setattr__ 类型签名只收 Tensor/Module, 给 int 标红
#   - 改成 setattr(...): ruff B010 又拦 ("Do not call setattr with a constant value")
# 等价替代: 子类类属性。_init_weights 里 getattr 读取的效果与 train_gpt2 完全一样,
# 不进 state_dict, 参数名 / isinstance(nn.Linear) / 初始化路径也全部不变
class ResidualProjLinear(nn.Linear):
    NANOGPT_SCALE_INIT = 1

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = ResidualProjLinear(embed_dim, embed_dim)   # = train_gpt2 的 nn.Linear + 标记

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
        self.c_proj = ResidualProjLinear(4 * embed_dim, embed_dim)   # train_gpt2.py:49 同上

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = FFN(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# =========================================================================
# 阶段 1: 输入端 —— wte + wpe 两张查表相加 (01 的知识, 接到塔前面)
# =========================================================================
# GPT 不直接吃 token id, 先查表变成向量, 再查位置表回答"在第几位", 两者相加
VOCAB_S, C_S, T_S = 100, 8, 5
idx = torch.randint(0, VOCAB_S, (2, T_S))          # [2,5] 2 句话各 5 个 token
wte_demo = nn.Embedding(VOCAB_S, C_S)               # 100×8 词表
wpe_demo = nn.Embedding(512, C_S)                   # 512×8 位置表 (block_size=512)

tok_emb = wte_demo(idx)                             # [2,5,8] 每个 token 查词表
pos_emb = wpe_demo(torch.arange(T_S))               # [5,8] 每个位置查位置表, 自动广播到 batch
x_in = tok_emb + pos_emb                            # [2,5,8] 相加 = "什么词" + "在第几位"

print("阶段 1: 输入端")
print("  tok_emb:", tuple(tok_emb.shape), " pos_emb:", tuple(pos_emb.shape), " 相加:", tuple(x_in.shape))
# 解答 01 的练习 a: 同一 token 在不同位置, 相加后向量不同 (位置信息进来了)
same_tok = tok_emb[0, 0]
diff_pos = not torch.equal(same_tok + wpe_demo(torch.tensor(0)),
                           same_tok + wpe_demo(torch.tensor(1)))
print("  同 token 换位置后向量不同:", diff_pos)   # 期望 True

# =========================================================================
# 阶段 2: GPT 模型类 —— 输入端 + Block塔 + 输出端 (对照 train_gpt2.py:79-128)
# =========================================================================
@dataclass
class GPTConfig:
    block_size: int = 512        # 最大序列长度 (wpe 的行数)
    vocab_size: int = 50257      # GPT-2 词表
    n_layer: int = 17            # 塔的层数 (architecture.md 推荐配置)
    n_head: int = 8
    n_embd: int = 512

class TransformerStack(nn.Module):
    # train_gpt2.py:85-90 用 nn.ModuleDict(dict(wte=..., wpe=..., h=..., ln_f=...)) 装这四样;
    # 这里换成等价的普通类: 参数名完全相同 (transformer.wte.weight 等, 可自行打印验证),
    # 原因是 ModuleDict 的属性是运行时按字符串挂的, 类型检查器猜不出来会全部标红
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # 输入端: 词表
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # 输入端: 位置表
        self.h = nn.ModuleList([TransformerBlock(config.n_embd, config.n_head)
                                for _ in range(config.n_layer)])     # Block塔: 07 阶段 2 堆的那座
        self.ln_f = nn.LayerNorm(config.n_embd)                     # 输出端: 最终归一化

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = TransformerStack(config)                 # = train_gpt2 的 nn.ModuleDict(...)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出端: → 词表 logits

        # 权重共享 (train_gpt2.py:94): lm_head 的 weight 就是 wte 的 weight
        # 语义: "把 token 编码进来" 和 "把向量解码回 token" 用同一张表 (编码/解码互逆)
        # 后果: 参数量记 0 次额外, 省 50257×512 = 25.7M
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 对照 train_gpt2.py:99-108。不初始化的话: nn.Embedding 默认 N(0,1),
        # logits 会过大 → 初始 loss 远超 ln(V), 深层梯度也不稳
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "NANOGPT_SCALE_INIT", False):
                # 残差支路上的输出投影要缩小: 17 层残差加起来, 增量得越来越细
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"
        # 输入端 (阶段 1 原样): token embedding + position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # [T]
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)       # [B,T,C]
        # Block塔: 07 阶段 2 的那座, 只是层数/宽度由 config 决定
        for block in self.transformer.h:
            x = block(x)                                                # [B,T,C] 穿层不变
        # 输出端: 最终 LN → 投影回词表 → 算 loss
        x = self.transformer.ln_f(x)                                    # [B,T,C]
        logits = self.lm_head(x)                                        # [B,T,V]
        loss = None
        if targets is not None:
            # 展平成 [B*T, V] vs [B*T]: 每个位置独立做分类, 类别数 = 词表
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# =========================================================================
# 阶段 3: forward 验收 —— 先小模型跑通, 再全尺寸对答案 (architecture.md 阶段 0)
# =========================================================================
print("\n阶段 3: forward 验收")
cfg_s = GPTConfig(block_size=32, vocab_size=1000, n_layer=2, n_head=4, n_embd=64)
model_s = GPT(cfg_s)
idx_s = torch.randint(0, 1000, (2, 10))            # 随机 token 当输入
tgt_s = torch.randint(0, 1000, (2, 10))            # 随机 token 当目标
logits_s, loss_s = model_s(idx_s, tgt_s)
print(f"  小模型: logits {tuple(logits_s.shape)} (期望 [2,10,1000])  loss {loss_s.item():.4f}  ln(1000)={math.log(1000):.4f}")

# 全尺寸 (79.6M): 输出 shape = (1, 512, 50257), 初始 loss ≈ ln(50257) ≈ 10.83
cfg_f = GPTConfig()
model_f = GPT(cfg_f)
idx_f = torch.randint(0, cfg_f.vocab_size, (1, cfg_f.block_size))
tgt_f = torch.randint(0, cfg_f.vocab_size, (1, cfg_f.block_size))
logits_f, loss_f = model_f(idx_f, tgt_f)
print(f"  全尺寸: logits {tuple(logits_f.shape)} (期望 [1,512,50257])")
print(f"  初始 loss {loss_f.item():.4f}  vs ln(50257)={math.log(50257):.4f}  验证: {abs(loss_f.item() - math.log(50257)) < 0.5}")
# 为什么 ≈ ln(V)? 模型啥也没学, 对每个 token 只能给均匀分布 1/V,
# 交叉熵 = -log(1/V) = ln(V)。训练就是把 loss 从这个上限往下压

# =========================================================================
# 阶段 4: 参数量对账 —— 07 的理论账本变实测 (architecture.md 第 3 节)
# =========================================================================
def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

print("\n阶段 4: 参数量账本 (C=512, 17层)")
wte_p = count(model_f.transformer.wte)
wpe_p = count(model_f.transformer.wpe)
blk_p = sum(count(b) for b in model_f.transformer.h)
lnf_p = count(model_f.transformer.ln_f)
total = count(model_f)

# 权重共享的验证: 指向同一块内存 (data_ptr 相同)
tied = model_f.lm_head.weight.data_ptr() == model_f.transformer.wte.weight.data_ptr()
print(f"  wte  {wte_p/1e6:>7.2f}M   理论 {50257*512/1e6:.2f}M")
print(f"  wpe  {wpe_p/1e6:>7.2f}M   理论 {512*512/1e6:.2f}M")
print(f"  17×Block {blk_p/1e6:>7.2f}M   理论 {17*(12*512**2 + 6656)/1e6:.2f}M (= 17 × (12C² + 每层bias 4608 + LN 2048))")
print(f"  ln_f {lnf_p/1e6:>7.2f}M   理论 {512*2/1e6:.4f}M")
print(f"  lm_head 共享 wte: {tied}   (若不共享会多算 {50257*512/1e6:.1f}M)")
print(f"  ---")
print(f"  实测合计 {total/1e6:.2f}M  vs architecture.md 目标 79.6M  验证: {total == 79_585_280}")
print(f"  fp32 大小 {total*4/1e6:.0f}MB   (预算表: 318MB)")

# =========================================================================
# 阶段 5: 单 batch 过拟合 —— 整条链路无 bug 的证明 (architecture.md 阶段 0)
# =========================================================================
# 拿一批固定数据反复训练: loss 能压到接近 0 → 说明模型/loss/反向传播/优化器
# 全部正确。若压不下去, 一定是哪里有 bug (最经典的深度学习调试法)
print("\n阶段 5: 单 batch 过拟合 (4 条固定随机序列, 反复练)")
torch.manual_seed(0)
cfg_o = GPTConfig(block_size=16, vocab_size=256, n_layer=2, n_head=4, n_embd=64)
model_o = GPT(cfg_o)
tokens = torch.randint(0, 256, (4, 17))            # 固定数据: 4 条长 17 的随机 token
# 切片不连续, 要 contiguous 才能被 forward 里的 view 用
x_o, y_o = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()   # 输入/目标错一位 = 预测下一个 token
opt = torch.optim.AdamW(model_o.parameters(), lr=1e-3)
print(f"  起点 loss ≈ ln(256) = {math.log(256):.4f}")
loss = None
for step in range(1000):
    _, loss = model_o(x_o, y_o)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0 or step == 999:
        print(f"  step {step:>4d}  loss {loss.item():.4f}")
print(f"  验收 (loss → 接近 0): {loss is not None and loss.item() < 0.5}")

# =========================================================================
# 练习题
#    a. 把 GPT.__init__ 里权重共享那行注释掉, 再看阶段 4 的合计变多少?
#       (期望 +25.7M ≈ 105.3M) 为什么共享能让 lm_head "参数量记 0"?
#       提示: 打印 id(model.lm_head.weight) 和 id(model.transformer.wte.weight)。
#    b. 阶段 3 里把 _init_weights 删掉 (Embedding 会变默认 N(0,1)),
#       初始 loss 会怎么变? 为什么? (动手跑, 对比 ln(50257))
#    c. targets=None 调 model(idx) 返回什么? 为什么推理时不需要 targets?
#    d. 对照 train_gpt2.py:110-128 逐行确认 forward 一致;
#       再找到 "17 层在哪一行被循环创建" (提示: train_gpt2.py:88)。
#    e. 阶段 5 若把 cfg_o 的 n_layer 改成 0 (没有 Block, 只有 embedding+lm_head),
#       loss 能压下去吗? 为什么? (提示: 没有 attn 时每个位置只看得到自己的输入)
#    f. wpe 只有 512 行, 输入 T=513 会发生什么? 在阶段 3 加一行试试。
#    g. FFN 用的 F.gelu (精确版), train_gpt2.py:47 用 GELU(approximate='tanh'),
#       两者数值上差多少? 对最终效果有影响吗? (提示: 查 GPT-2 论文用哪个)
