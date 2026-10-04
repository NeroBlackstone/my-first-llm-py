import inspect
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.manual_seed(42)

# =========================================================================
# 零件复用: 09 的模型 5 类原样搬来, 仅两处标注的最小改动:
#   - GPT.forward 加 self.grad_checkpoint 分支 (默认 False, 行为与 09 逐位一致)
#   - generate 装饰器 @torch.no_grad() → @torch.inference_mode()
# 本课主题: 把 09 手写的训练侧全部换成 PyTorch 高层抽象 (对照 09):
#   get_batch        → torch.utils.data.Dataset + DataLoader      (阶段 1)
#   get_lr + 手动改 lr → SequentialLR(LinearLR → CosineAnnealingLR) (阶段 2)
#   手写 no_grad 循环  → torch.inference_mode                      (阶段 3)
#   print 进度        → tqdm                                       (阶段 3)
#   09 缺的三件事     → 梯度累积 / EMA (swa_utils) / 断点 (torch.save) (阶段 3)
#   可选开关           → AMP (autocast+GradScaler) / 激活检查点, 默认关
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
        # 10 新增开关: 激活检查点 (对照 architecture.md 第 5 节 OOM 对策)。
        # 默认 False → forward 与 09 逐位一致; 开 = 每层前向不存激活、反向时重算,
        # 用算力换显存 (12 层全尺寸时把 B4×T512 从 OOM 救回来)
        self.grad_checkpoint = False
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
            if self.grad_checkpoint and self.training:
                # use_reentrant=False 是新版推荐: 不往 autograd 图里塞
                # 重入式伪节点, 与 inference_mode / 梯度累积都兼容
                x = checkpoint(block, x, use_reentrant=False)
            else:
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

    @torch.inference_mode()      # 09 是 @torch.no_grad(); inference_mode 更强 (见阶段 4)
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
# 阶段 1: 数据管道 —— 文本 → token id → Dataset → DataLoader
# 09 的 get_batch 在这里被拆成两件官方件:
#   Dataset  = "第 i 条样本长什么样" (取窗口 + 配对 x/y)
#   DataLoader = "怎么把样本攒成 batch" (堆叠/pin_memory/多进程/断批)
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

class ShakespeareDataset(Dataset):
    # 第 i 条样本 = 一段连续文本切成 (x, y): y 是 x 右移一位 (下一 token 预测, 同 09 get_batch)
    # random_windows=True  (train): 每次 __getitem__ 重新掷骰子取起点 —— 等价 09 的随机窗口
    # random_windows=False (val):   起点 = i * stride, 第 i 条永远是同一段 ——
    #   09 的 val 每次随机取窗, val loss 在不同 step 之间混入了"窗口抽样噪声"没法细比;
    #   固定窗口后 val loss 的变化才 100% 来自权重变化 (对照 09 练习 f 的 hellaswag 同理:
    #   评测集固定, 分数才可比)
    def __init__(self, data, block_size, length, random_windows=True):
        self.data = data
        self.block_size = block_size
        self.length = length
        self.random_windows = random_windows
        self.max_start = len(data) - block_size - 1             # 起点上限: 还要留 block_size 个 token
        assert self.max_start > 0, "数据比窗口还短"
        self.stride = max(1, self.max_start // length)          # 确定性起点间隔, 均匀铺满整个 split

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.random_windows:
            start = int(torch.randint(self.max_start, (1,)))    # train: 每取一次重掷
        else:
            start = min(i * self.stride, self.max_start)        # val: 固定映射, 可复现
        buf = self.data[start : start + self.block_size + 1]    # 长度 block_size+1: 多拿 1 个配平 x/y
        return buf[:-1], buf[1:]                                # x, y = y 是 x 右移一位

print("阶段 1: 数据管道 (Dataset)")
print(f"  input.txt → {len(tokens):,} token, 编码耗时 {time.time()-t0:.1f}s (缓存在 data/*.bin)")
print(f"  train {len(train_data):,} / val {len(val_data):,}")

demo_loader = DataLoader(ShakespeareDataset(train_data, block_size=8, length=4),
                         batch_size=2)                          # default_collate 自动堆叠成 [B, T]
x_demo, y_demo = next(iter(demo_loader))
print(f"  batch: x {tuple(x_demo.shape)} y {tuple(y_demo.shape)}")
print(f"  x[0] = {x_demo[0].tolist()}")
print(f"  y[0] = {y_demo[0].tolist()}   ← 每个位置都是 x[0] 的下一个 token")

# =========================================================================
# 阶段 2: 训练配置 —— 设备 / 超参 / DataLoader / 优化器 / lr 调度
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

B, T = 8, 256                       # micro batch: 8×256 = 2048 token/次前向
grad_accum = 1                      # 梯度累积: 每积累 grad_accum 次前向才更新一次
max_lr, min_lr = 1e-3, 1e-4         # 峰值/谷值学习率 (nanoGPT 量级, 小模型用大 lr)
warmup_steps, max_steps = 100, 3000 # 线性 warmup 步数 / 总更新步数
weight_decay, grad_clip = 0.1, 1.0
val_batches, ema_decay = 20, 0.999  # 每次验证取几个 batch / EMA 衰减
use_amp = False                     # AMP 总开关: GTX 970 (sm_5.2) 无 Tensor Core, fp16 只有
                                    # 1/64 算力速率 → 开了更慢, 留作对比实验 (练习 e)

# DataLoader: 09 的 get_batch(B, T) 在这里
#   batch_size=B  → 一次前向的 micro batch
#   pin_memory    → 用锁页内存, GPU H2D 拷贝更快 (配合 non_blocking=True)
#   shuffle=False → 窗口本身随机取, 打乱样本顺序没有额外信息 (与常规图像分类不同)
#   num_workers=0 → 数据已在 RAM 里, 多进程反而多一份拷贝/IPC 开销; FineWeb 级数据再开
train_loader = DataLoader(
    ShakespeareDataset(train_data, T, length=max_steps * grad_accum * B),
    batch_size=B, shuffle=False, pin_memory=device_type == "cuda", num_workers=0,
)
val_loader = DataLoader(
    ShakespeareDataset(val_data, T, length=val_batches * B, random_windows=False),
    batch_size=B, shuffle=False, pin_memory=device_type == "cuda", num_workers=0,
)
print(f"  micro batch: B{B}×T{T} = {B*T} token/次前向 × grad_accum {grad_accum} "
      f"= {B*T*grad_accum} 等效 token/步")
print(f"  train dataset {len(train_loader.dataset)} 条窗口 / val dataset {len(val_loader.dataset)} 条 (固定窗口)")

optimizer = model.configure_optimizers(weight_decay=weight_decay,
                                       learning_rate=max_lr, device_type=device_type)

# lr 调度: 09 的 get_lr 三段函数 + 手动改 g["lr"] → 官方两段调度器串联
#   LinearLR(start_factor=1/warmup, total_iters=warmup-1): 因子从 1/100 线性升到 1
#     → lr: max_lr/warmup → max_lr, 恰好逐点等于 09 的 max_lr*(it+1)/warmup
#   CosineAnnealingLR(T_max=max_steps-warmup, eta_min=min_lr): 余弦降温到谷值
#     → 等于 09 的 min_lr + 0.5*(1+cos(π·ratio))*(max_lr-min_lr)
#   SequentialLR(milestones=[warmup]): 第 warmup 步从第 1 段切到第 2 段
# 构造顺序有讲究: 先建 cosine (它的 base_lr 在构造时从 param_groups 捕获, 此时还是
# max_lr); 后建 linear (构造即执行初始 step, 把 lr 压到 max_lr/warmup)。
# 顺序反了 cosine 会把 max_lr/warmup 当成峰值, 整条降温曲线矮一个数量级。
warmup_sched = LinearLR(optimizer, start_factor=1.0 / warmup_steps,
                        end_factor=1.0, total_iters=warmup_steps - 1)
cosine_sched = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

def get_lr_09(it):
    # 09 原版手写调度, 只留作对照打印 (训练里不再调用它, 每步换 scheduler.step())
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)

print(f"  lr 调度: LinearLR(0 → {max_lr}, warmup {warmup_steps}步) → "
      f"CosineAnnealingLR(→ {min_lr}, 共 {max_steps} 步)")
# 抽样对照: 临时建一套同样的调度器走一遍, 打印与 09 get_lr 的逐点误差
with warnings.catch_warnings():          # 临时调度器没有先 optimizer.step, 忽略顺序警告
    warnings.simplefilter("ignore")
    probe_p = torch.nn.Parameter(torch.zeros(1))
    probe_opt = torch.optim.AdamW([probe_p], lr=max_lr)
    probe_cos = CosineAnnealingLR(probe_opt, T_max=max_steps - warmup_steps, eta_min=min_lr)
    probe_lin = LinearLR(probe_opt, start_factor=1.0 / warmup_steps,
                         end_factor=1.0, total_iters=warmup_steps - 1)
    probe_sched = SequentialLR(probe_opt, [probe_lin, probe_cos], milestones=[warmup_steps])
    for it in range(max_steps + 1):
        if it in (0, 1, 50, 99, 100, 1500, 2999, 3000):
            sched_lr = probe_opt.param_groups[0]["lr"]
            ref_lr = get_lr_09(it)
            print(f"    step {it:>4d}: scheduler {sched_lr:.6e} | 09 get_lr {ref_lr:.6e} "
                  f"| rel_err {abs(sched_lr - ref_lr) / ref_lr:.1e}")
        probe_sched.step()

# EMA (指数滑动平均): swa_utils 官方件, 维护一份"近期权重的滑动平均"副本,
# 通常比瞬时权重更平滑、泛化更好 (对照 09: 09 只有最后一刻的瞬时权重)
#   multi_avg_fn=get_ema_multi_avg_fn(decay): ema = decay*ema + (1-decay)*model
#   权重共享的 wte/lm_head 在 deepcopy 里仍共享, parameters() 两边都只出现一次 → 对齐没问题
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)).to(device)

# AMP 组合件: autocast 把 matmul/conv 放低精度算, GradScaler 把 loss 放大避免 fp16 下溢
# enabled=False 时两者全链路 no-op, 一行代码兼容开关 (练习 e 在本机打开会变慢)
amp_ctx = torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)
scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

CKPT_PATH = HERE / "out" / "10_ckpt.pt"
CKPT_PATH.parent.mkdir(exist_ok=True)
RESUME = False                         # 改 True 断点续训 (练习 f)

# =========================================================================
# 阶段 3: 训练循环 —— DataLoader → 前向 → 反向 → 裁剪 → 更新
# 09 缺的三件 (练习 f): 梯度累积 / EMA / checkpoint, 这里补齐
# =========================================================================
print("\n阶段 3: 训练循环 (tqdm + 梯度累积 + EMA + inference_mode 验证 + 断点)")

start_step = 0
if RESUME and CKPT_PATH.exists():
    blob = torch.load(CKPT_PATH, weights_only=True)   # 新版默认只反序列化张量类, 防任意代码执行
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])       # last_epoch 一起恢复 → lr 曲线接得上
    ema.load_state_dict(blob["ema"])                   # n_averaged 也恢复 → EMA 平均不中断
    start_step = blob["step"] + 1
    print(f"  断点续训: 从 step {start_step} 恢复 ({CKPT_PATH})")

def save_ckpt(step):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "ema": ema.state_dict(),
                "step": step}, CKPT_PATH)

@torch.inference_mode()               # 推理专用上下文, 比 no_grad 更强: 除禁梯度外还关掉
                                      # 版本计数/视图跟踪, 张量不能再被 autograd 收编
def evaluate(net, n_batches):
    net.eval()
    total, n = 0.0, 0
    for xb, yb in val_loader:         # 固定窗口 + 固定条数 → 每次 evaluate 输入完全相同
        _, lv = net(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
        total += lv.item()
        n += 1
        if n >= n_batches:
            break
    net.train()
    return total / n

model.train()
t_start = time.time()
train_loss_sum, train_loss_n = 0.0, 0
val_loss_accum = ema_val_accum = 0.0
data_iter = iter(train_loader)

pbar = tqdm(range(start_step, max_steps), desc="train", dynamic_ncols=True)
for step in pbar:
    # 梯度累积: grad_accum 次 micro 前向的梯度相加 (loss/accum 保证求和后均值不变),
    # 只在攒够时更新一次 → 等效批量 ×accum, 显存不变 (architecture.md 凑批量方案)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _ in range(grad_accum):
        try:
            xb, yb = next(data_iter)
        except StopIteration:         # 一个 epoch 的窗口取完, 重建迭代器 (随机窗口无状态)
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
        with amp_ctx:
            _, loss = model(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
        (loss / grad_accum).backward()
        loss_sum += loss.item()
    scaler.unscale_(optimizer)        # AMP: 先把梯度缩回去再裁剪/更新; disabled 时 no-op
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 梯度裁剪: 万一某步梯度爆炸 (norm 突增), 按 1.0 等比缩回, 防止一步把权重打飞
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()                  # 每次更新后走一格调度表 (09 是手写 get_lr 改 g["lr"])
    ema.update_parameters(model)      # EMA 副本跟随一步
    train_loss_sum += loss_sum / grad_accum
    train_loss_n += 1

    pbar.set_postfix(loss=f"{train_loss_sum / train_loss_n:.4f}",
                     lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    if step % 100 == 0 or step == max_steps - 1:
        dt = time.time() - t_start
        tqdm.write(f"  step {step:>4d} | train loss {train_loss_sum / train_loss_n:.4f} "
                   f"| lr {optimizer.param_groups[0]['lr']:.2e} | grad norm {norm:.3f} | {dt:.0f}s")
        train_loss_sum, train_loss_n = 0.0, 0
        t_start = time.time()

    # 定期验证: raw 与 EMA 各评一次 (固定窗口 → 跨 step 可比)
    # train loss 降但 val loss 不降 = 过拟合 (本数据集很小, 会有点过拟合是正常的)
    if step % 250 == 0 or step == max_steps - 1:
        val_loss_accum = evaluate(model, val_batches)
        ema_val_accum = evaluate(ema.module, val_batches)
        tqdm.write(f"  step {step:>4d} | val loss {val_loss_accum:.4f} | ema val {ema_val_accum:.4f}  "
                   f"← 初始应 ≈ ln(50257)={math.log(50257):.2f}")

    if step % 500 == 0 or step == max_steps - 1:
        save_ckpt(step)               # 五件套: model / optimizer / scheduler / ema / step

# =========================================================================
# 阶段 4: 生成验收 —— 让模型写莎士比亚 (EMA 权重 vs 原始权重)
# =========================================================================
print("\n阶段 4: 采样生成 (同 seed, EMA 权重 vs 原始权重)")
prompt = enc.encode("ROMEO:")
idx = torch.tensor([prompt], dtype=torch.long, device=device)

def gen(net, seed=42):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return net.generate(idx, max_new_tokens=64, top_k=50, temperature=1.0, generator=rng)

print('  prompt: "ROMEO:"')
print(f'  EMA 生成: "{enc.decode(gen(ema.module)[0].tolist())}"')
print(f'  raw 生成: "{enc.decode(gen(model)[0].tolist())}"')

print("\n验收")
print(f"  最终 val loss {val_loss_accum:.4f} / ema {ema_val_accum:.4f}  "
      f"(初始 ≈ ln(50257)=10.83 → 阶段 1 目标 < 1.5)")
print(f"  验证: {val_loss_accum < 1.5}")

# =========================================================================
# 练习题
#    a. 把 val_loader 的 random_windows 改回 True (= 09 的每次随机取窗), 对比 val loss
#       曲线的抖动。为什么评测集必须固定, 跨 step 的分数才可比? (hellaswag 同理)
#    b. probe 调度器里把 LinearLR 的 total_iters 从 warmup-1 改成 warmup, 抽样行的
#       rel_err 变成多少? (提示: warmup 结束点错一格, 切换时 lr 掉 1%)
#    c. 先建 linear 后建 cosine, 看阶段 2 抽样行的 scheduler 列 —— 整条曲线矮多少倍?
#       为什么 base_lr 是在构造时捕获的? (对照源码 LRScheduler.__init__)
#    d. scheduler.step() 挪到 scaler.step(optimizer) 之前 (或删掉), lr 曲线会怎样?
#       为什么 09 手写 get_lr 时顺序无所谓、官方调度器却必须 optimizer 先?
#    e. use_amp=True 重跑: GTX 970 上每步更快还是更慢? 为什么? (sm_5.2 无 Tensor Core,
#       fp16 算力 1/64 速率; 什么卡上才值得开? 提示 architecture.md 第 1 节)
#    f. 训练 600 步后 Ctrl-C, 把 RESUME 改 True 重跑: 从断点继续。断点存了哪五样?
#       为什么随机取窗的数据管道不需要存 RNG 状态? (提示: 每个窗口独立掷骰子)
#    g. grad_accum 改 4、B 改 2 (2×256×4 = 2048 等效 token/步不变): loss 曲线还能对齐吗?
#       为什么 12 层全尺寸必须靠它凑批量? (提示 architecture.md 第 5 节 B3×T512×accum 11)
#    h. model.grad_checkpoint = True 重跑: 峰值显存降多少、每步慢多少? 为什么是
#       "反向时重算前向"换显存, 且只在 self.training 时生效? (提示: 推理本来就不存激活)
#    i. 对照 09 全文, 列出被替换掉的手写件与对应官方件 (提示: 本文头部注释有完整映射表)
