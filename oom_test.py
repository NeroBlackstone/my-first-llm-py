"""OOM 测试: 40M 全尺寸配置 (12层/6头/384维/block512) 在 GTX 970 上的显存实测.

用法: <poetry python> oom_test.py
"""
import gc
import time
from pathlib import Path

import torch

SRC_PATH = Path(__file__).resolve().parent / "09_train_gpt.py"
SRC = SRC_PATH.read_text(encoding="utf-8")
# 只 exec 模型类部分 (到"阶段 1: 数据管道"之前), 避免触发完整训练
head = SRC.split("# 阶段 1: 数据管道")[0]
ns = {}
exec(compile(head, str(SRC_PATH), "exec"), ns)
GPTConfig, GPT = ns["GPTConfig"], ns["GPT"]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV}", torch.cuda.get_device_name(0) if DEV == "cuda" else "")


def mem_line(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  [{tag}] free {free/1e6:.0f}MB / total {total/1e6:.0f}MB "
          f"(驱动占用 {(total-free)/1e6:.0f}MB 未计入 torch 统计)")


def run_case(n_layer, n_head, n_embd, block_size, B, T, steps=3):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    mem_line(f"case 前 L{n_layer} B{B} T{T}")
    t0 = time.time()
    try:
        cfg = GPTConfig(block_size=block_size, n_layer=n_layer,
                        n_head=n_head, n_embd=n_embd)
        model = GPT(cfg).to(DEV)
        opt = model.configure_optimizers(weight_decay=0.1,
                                         learning_rate=1e-3,
                                         device_type="cuda" if DEV == "cuda" else "cpu")
        setup_dt = time.time() - t0
        torch.manual_seed(0)
        x = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)
        y = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)
        t0 = time.time()
        for _ in range(steps):
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()
        step_dt = (time.time() - t0) / steps
        n_params = sum(p.numel() for p in model.parameters())
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_res = torch.cuda.max_memory_reserved() / 1e6
        print(f"  OK  {n_params/1e6:.1f}M params | peak alloc {peak_alloc:.0f}MB | "
              f"peak reserved {peak_res:.0f}MB | loss {loss.item():.3f} | "
              f"setup {setup_dt:.1f}s | {step_dt*1000:.0f}ms/step")
        del opt, model, x, y, loss
        return True, peak_alloc
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"  OOM! {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        torch.cuda.empty_cache()
        gc.collect()
        return False, None


print("\n== 1. 对照: 09 默认小配置 (6层) + 原目标 B4×T512 (已知 OOM) ==")
print("   (09 脚本默认小配置 6层/384/block256 对照)")
run_case(6, 6, 384, 256, 8, 256)          # 09 当前默认, 对照
run_case(12, 6, 384, 512, 4, 512)         # 原训练目标 B4 T512 (已知 OOM, 复测确认)

print("\n== 2. 扫边界: 12层/384维 不同 (B, T) 组合 (规则: B×T ≤ 1536 稳, ≥ 2048 OOM) ==")
for B, T in [(4, 512), (8, 256), (3, 512), (4, 384), (2, 512), (4, 256), (8, 128), (1, 512)]:
    run_case(12, 6, 384, 512, B, T, steps=2)

print("\n== 3. 生成阶段 (无梯度, 只应占权重+激活) ==")
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
try:
    cfg = GPTConfig(block_size=512, n_layer=12, n_head=6, n_embd=384)
    model = GPT(cfg).to(DEV)
    idx = torch.tensor([[50256]], device=DEV)
    out = model.generate(idx, max_new_tokens=64, top_k=50)
    print(f"  OK  peak alloc {torch.cuda.max_memory_allocated()/1e6:.0f}MB, "
          f"生成 {out.shape[1]} token")
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" in str(e).lower():
        print(f"  OOM! {str(e).splitlines()[0][:120]}")
    else:
        raise

print("\n结论以各 case 的 OK/OOM 为准.")
