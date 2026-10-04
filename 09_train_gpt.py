import inspect
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention

torch.manual_seed(42)

# =========================================================================
# 零件复用: 08 的模型类原样搬来 (ResidualProjLinear / CausalSelfAttention /
#   FFN / TransformerBlock / GPTConfig / TransformerStack / GPT)
# 本课新增两件工程件 (对照 train_gpt2.py):
#   - GPT.configure_optimizers: weight decay 分组 (train_gpt2.py:179-202)
#   - GPT.generate:            top-k 采样生成     (train_gpt2.py:446-480)
#   - 阶段 3 的训练循环:        dataloader + lr 调度 + 梯度裁剪 (train_gpt2.py:353-518)
# =========================================================================
class ResidualProjLinear(nn.Linear):
    NANOGPT_SCALE_INIT = 1

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = ResidualProjLinear(embed_dim, embed_dim)

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
        self.c_proj = ResidualProjLinear(4 * embed_dim, embed_dim)

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

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class TransformerStack(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([TransformerBlock(config.n_embd, config.n_head)
                                for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = TransformerStack(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight   # 权重共享 (08 阶段 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "NANOGPT_SCALE_INIT", False):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 对照 train_gpt2.py:179-202。规则: 2D+ 参数做 weight decay, 1D 不做:
        #   Linear/Embedding 权重 (2D) → 衰减 (矩阵不该无限变大)
        #   bias / LayerNorm (1D)      → 不衰减 (它们是偏移量, 衰减=强行拉回 0 没道理)
        # named_parameters 默认去重: 权重共享的 wte/lm_head 只出现一次
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay = [p for p in param_dict.values() if p.dim() < 2]
        print(f"  decay 组: {len(decay)} 个张量, {sum(p.numel() for p in decay):,} 参数")
        print(f"  nodecay 组: {len(nodecay)} 个张量, {sum(p.numel() for p in nodecay):,} 参数")
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        print(f"  fused AdamW: {use_fused}")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50, temperature=1.0, generator=None):
        # 对照 train_gpt2.py:446-480。自回归: 每次只算最后一个位置, 吐一个 token 接上
        # top_k=50: 只在概率最高的 50 个 token 里采样 (砍掉长尾垃圾)
        # temperature: >1 更发散, <1 更保守 (先除再 softmax = 拉平/锐化分布)
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]        # 超长截断到 block_size
            logits, _ = self(idx_cond)                          # [B, T, V]
            logits = logits[:, -1, :] / temperature             # 只取最后一个位置
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, min(top_k, logits.size(-1)), dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=generator)   # 在 top-k 内掷骰子
            xcol = torch.gather(topk_indices, -1, ix)           # 掷中的是哪个 token id
            idx = torch.cat((idx, xcol), dim=1)                 # 接到序列尾部
        return idx

# =========================================================================
# 阶段 1: 数据管道 —— 文本 → token id → 随机窗口 batch (train_gpt2.py:204-252)
# =========================================================================
# 模型不读字符, 读 token id。分词用 GPT-2 的 tiktoken (BPE), 与 08 的 vocab=50257 对齐
# 编码一次缓存成 uint16 二进制 (词表 50257 < 65536 用 2 字节够了, train_gpt2 同款)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
BIN_PATH = DATA_DIR / "shakespeare_gpt2.bin"
INPUT_PATH = HERE.parent / "build-nanogpt" / "input.txt"

t0 = time.time()
if BIN_PATH.exists():
    tokens = torch.from_numpy(np.fromfile(BIN_PATH, dtype=np.uint16).astype(np.int64))
else:
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(INPUT_PATH.read_text(encoding="utf-8"))
    np.array(ids, dtype=np.uint16).tofile(BIN_PATH)            # 缓存, 下次秒开
    tokens = torch.from_numpy(np.array(ids, dtype=np.int64))
enc = tiktoken.get_encoding("gpt2")

n = int(0.9 * len(tokens))                                      # 90% 训练 / 10% 验证
train_data, val_data = tokens[:n], tokens[n:]
print("阶段 1: 数据管道")
print(f"  input.txt → {len(tokens):,} token, 编码耗时 {time.time()-t0:.1f}s (缓存在 data/*.bin)")
print(f"  train {len(train_data):,} / val {len(val_data):,}")

def get_batch(split, B, T):
    # 随机起点取一段连续文本, 前 B*T 当输入, 后移一位当目标 (下一 token 预测)
    data = train_data if split == "train" else val_data
    i = int(torch.randint(len(data) - B * T - 1, (1,)))   # 起点要留出 B*T+1 的余量
    buf = data[i : i + B * T + 1]      # 长度 B*T+1: 多拿 1 个才能配平 x/y
    x = buf[:-1].view(B, T)            # [B, T] 输入
    y = buf[1:].view(B, T)             # [B, T] 目标 = 输入右移一位
    return x, y

x_demo, y_demo = get_batch("train", 2, 8)
print(f"  batch: x {tuple(x_demo.shape)} y {tuple(y_demo.shape)}")
print(f"  x[0] = {x_demo[0].tolist()}")
print(f"  y[0] = {y_demo[0].tolist()}   ← 每个位置都是 x[0] 的下一个 token")

# =========================================================================
# 阶段 2: 训练配置 —— 设备 / 超参 / 优化器 / 学习率调度
# =========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"\n阶段 2: 训练配置  (device={device})")

# 小配置先跑通流程 (architecture.md 阶段 1 允许临时缩小)。
# 全尺寸 12层/6头/384维/block512 = 40.7M 实测 (architecture.md 第 5 节):
#   推荐 B3×T512 (2206MB, 余量 ~1GB, wpe 全位置可训), 兜底 B2×T512;
#   B4×T512 / B8×T256 (≥2048 token) 会 OOM, 批量用梯度累积凑
config = GPTConfig()
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  模型: {config.n_layer}层 / {config.n_head}头 / {config.n_embd}维 / block {config.block_size}")
print(f"  参数量: {n_params/1e6:.1f}M (全尺寸 12层/384维/block512 是 40.7M)")

B, T = 8, 256                       # micro batch: 8×256 = 2048 token/步
max_lr, min_lr = 1e-3, 1e-4         # 峰值/谷值学习率 (nanoGPT 量级, 小模型用大 lr)
warmup_steps, max_steps = 100, 3000 # 线性 warmup 步数 / 总步数
weight_decay, grad_clip = 0.1, 1.0

def get_lr(it):
    # 对照 train_gpt2.py:353-364。三段: 线性升温 → 余弦降温 → 触底
    # 为什么不直接满 lr: 训练初期梯度方向随机, 大 lr 会把随机初始化砸烂 (warmup)
    # 后期降 lr 是为了在 loss 曲面的谷底附近做更细的收敛 (cosine decay)
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))              # 1 → 0 余弦曲线
    return min_lr + coeff * (max_lr - min_lr)

print(f"  lr 调度: 0 → {max_lr} (warmup {warmup_steps}步) → {min_lr} (共 {max_steps} 步)")
print(f"  lr 抽样: step0={get_lr(0):.2e}  step100={get_lr(100):.2e}  "
      f"step1500={get_lr(1500):.2e}  step3000={get_lr(3000):.2e}")

optimizer = model.configure_optimizers(weight_decay=weight_decay,
                                       learning_rate=max_lr, device_type=device_type)

# =========================================================================
# 阶段 3: 训练循环 —— 前向 → 反向 → 裁剪 → 更新 (train_gpt2.py:376-518)
# =========================================================================
print("\n阶段 3: 训练循环")
model.train()
t_start = time.time()
train_loss_sum, train_loss_n = 0.0, 0
val_loss_accum = 0.0

def evaluate(val_batches):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(val_batches):
            xv, yv = get_batch("val", B, T)
            _, lv = model(xv.to(device), yv.to(device))
            total += lv.item()
    model.train()
    return total / val_batches

for step in range(max_steps):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr                       # 每步按调度表改写 lr

    xb, yb = get_batch("train", B, T)
    _, loss = model(xb.to(device), yb.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 梯度裁剪: 万一某步梯度爆炸 (norm 突增), 按 1.0 等比缩回, 防止一步把权重打飞
    optimizer.step()
    train_loss_sum += loss.item()
    train_loss_n += 1

    if step % 100 == 0 or step == max_steps - 1:
        dt = time.time() - t_start
        print(f"  step {step:>4d} | train loss {train_loss_sum / train_loss_n:.4f} "
              f"| lr {lr:.2e} | grad norm {norm:.3f} | {dt:.0f}s")
        train_loss_sum, train_loss_n = 0.0, 0
        t_start = time.time()

    # 定期验证: train loss 降但 val loss 不降 = 过拟合 (本数据集很小, 会有点过拟合是正常的)
    if step % 250 == 0 or step == max_steps - 1:
        val_loss_accum = evaluate(20)
        print(f"  step {step:>4d} | val   loss {val_loss_accum:.4f}  ← 初始应 ≈ ln(50257)={math.log(50257):.2f}")

# =========================================================================
# 阶段 4: 生成验收 —— 让模型写莎士比亚 (train_gpt2.py:446-480)
# =========================================================================
print("\n阶段 4: 采样生成")
prompt = enc.encode("ROMEO:")
idx = torch.tensor([prompt], dtype=torch.long, device=device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)
out = model.generate(idx, max_new_tokens=64, top_k=50, temperature=1.0, generator=sample_rng)
print('  prompt: "ROMEO:"')
print(f'  生成:   "{enc.decode(out[0].tolist())}"')

print("\n验收")
print(f"  最终 val loss {val_loss_accum:.4f}  (初始 ≈ ln(50257)=10.83 → 阶段 1 目标 < 1.5)")
print(f"  验证: {val_loss_accum < 1.5}")

# =========================================================================
# 练习题
#    a. 阶段 1 里把 y 改成不移位 (y = buf[:B*T]) —— loss 会异常顺利地降到接近 0,
#       但生成全是复读。为什么"预测自己"是作弊? (提示: 残差连接能直接抄输入)
#    b. get_lr 从 warmup=0 开始训, 前 100 步 loss 会怎样? 为什么 warmup 有用?
#    c. 关掉梯度裁剪 (不调 clip_grad_norm_), 打大 max_lr=0.05, 观察 grad norm 和 loss
#       什么时候爆炸? (提示: 爆炸 = 某步 loss 突然变 nan)
#    d. generate 的 temperature 试 0.5 / 2.0, top_k 试 5 / 1000, 各生成一次:
#       哪个更保守? 哪个更胡言乱语? 为什么 top_k=1000 ≈ 贪心?
#    e. weight_decay 试 0.0 / 1.0, 对比 val loss 和生成质量 (正则化的取舍)
#    f. 对照 train_gpt2.py:482-518 逐段确认训练循环一致, 找出它有而这里没有的三件事
#       (提示: 梯度累积 / val 之外的 hellaswag / checkpoint 保存)
#    g. 把 config 改成 architecture.md 的全尺寸 12层/6头/384维/block512 (= 40.7M) 再跑,
#       B=4/T=512 会 OOM; 推荐 B3×T512 (2206MB), 等效批量怎么用梯度累积凑 16896 token/步?
#       为什么 micro 必须用 T=512 而不是 T=256? (提示: wpe 位置嵌入 + 第 5 节实测表)
