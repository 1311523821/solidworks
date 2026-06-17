"""
RL 训练入口 — Curriculum Learning + PPO
=======================================

训练流程:
  1. (可选) 行为克隆预热 — 用规则系统生成的标签训练模型
  2. Curriculum Learning — 从简单到复杂逐步训练
  3. PPO 在线训练 — 在真实环境中交互学习

用法:
    python -m RL.train              # 默认 curriculum
    python -m RL.train --folder ./1 # 单数据集训练
    python -m RL.train --eval       # 仅评估
"""

import os
import sys
import json
import math
import glob
import argparse
from datetime import datetime

import torch
import numpy as np

# 确保父目录在 path 中
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from RL.config import (
    CURRICULUM, NUM_EPOCHS, STEPS_PER_EPOCH, HIDDEN_DIM,
)
from RL.env import AssemblyEnv
from RL.models import AssemblyGNN
from RL.ppo import PPOTrainer


def train_curriculum(model, trainer, curriculum, epochs_per_stage):
    """
    Curriculum Learning: 从简单到复杂逐步训练。
    每 10 epoch 在验证集上评估，防止过拟合训练环境。
    """
    print(f"\n{'='*60}")
    print(f"  Curriculum Learning ({len(curriculum)} stages)")
    print(f"{'='*60}")

    for stage, folder in enumerate(curriculum):
        print(f"\n--- Stage {stage+1}: {folder} ---")

        try:
            env = AssemblyEnv(folder)
        except Exception as e:
            print(f"  [skip] {folder}: {e}")
            continue

        print(f"  零件数: {env.n_parts}, 特征节点数: {env.n_nodes}")

        # 验证集（自动选：训练 ./1 则用 ./2 验证，以此类推）
        val_folder = None
        other_folders = [f for f in ["./1", "./2", "./3"] if f != folder]
        for vf in other_folders:
            if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", vf.lstrip("./"))):
                val_folder = vf
                break
        val_env = None
        if val_folder:
            try:
                val_env = AssemblyEnv(val_folder)
                print(f"  验证集: {val_folder} ({val_env.n_parts} 零件)")
            except Exception:
                pass

        # 保存路径
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "best_model.pt")
        latest_path = os.path.join(save_dir, "latest_model.pt")

        from tqdm import tqdm
        stage_label = f"  {folder}" if len(curriculum) <= 1 else f"  Stage {stage+1}/{len(curriculum)}"
        pbar = tqdm(range(epochs_per_stage), desc=stage_label, unit="ep")

        # 早停追踪（三信号：reward + entropy + v_loss）
        best_reward = -999
        best_epoch = 0
        stale_count = 0
        STALE_PATIENCE = 25       # reward 不涨 + entropy 不降 → 停滞
        CONVERGE_WINDOW = 8       # 连续 N epoch 三信号同时满足 → 收敛
        WINDOW_SIZE = 10

        # 滑动窗口
        r_window = []    # reward
        e_window = []    # entropy
        v_window = []    # value loss

        for ep in pbar:
            # 收集经验
            roll_stats = trainer.collect_rollout(env, steps=STEPS_PER_EPOCH)

            # PPO 更新
            update_stats = trainer.update()

            # 清空 buffer
            trainer.buffer.reset()

            # 收集指标
            rstats = trainer.get_stats()
            cur_r = roll_stats["avg_episode_reward"]
            cur_e = rstats["entropy"]
            cur_v = rstats["value_loss"]

            r_window.append(cur_r)
            e_window.append(cur_e)
            v_window.append(cur_v)
            if len(r_window) > WINDOW_SIZE:
                r_window.pop(0); e_window.pop(0); v_window.pop(0)

            # ---- 验证集评估（每 10 epoch）----
            val_r = 0.0
            if val_env and (ep + 1) % 10 == 0:
                model_device = str(next(model.parameters()).device)
                val_result = evaluate(model, val_folder, verbose=False, device=model_device)
                val_r = val_result.get("avg_reward", 0.0)
                # 用验证集 reward 判断最优（而非训练集）
                if val_r > best_reward + 0.5:
                    best_reward = val_r
                    best_epoch = trainer.epoch
                    stale_count = 0
                    torch.save(model.state_dict(), best_path)
                    tqdm.write(f"  [best] val_reward={val_r:.1f} @epoch {trainer.epoch}")
            else:
                # 无验证集时，用训练 reward 判断
                if cur_r > best_reward + 0.5:
                    best_reward = cur_r
                    best_epoch = trainer.epoch
                    stale_count = 0
                    torch.save(model.state_dict(), best_path)
                    tqdm.write(f"  [best] train_reward={cur_r:.1f} @epoch {trainer.epoch}")
                else:
                    stale_count += 1

            # ---- 收敛检测：三信号交叉验证 ----
            if len(r_window) >= CONVERGE_WINDOW:
                recent_r = r_window[-CONVERGE_WINDOW:]
                recent_e = e_window[-CONVERGE_WINDOW:]
                recent_v = v_window[-CONVERGE_WINDOW:]
                # reward 高 + entropy 低且稳 + v_loss 低且稳
                r_ok = all(r > 80 for r in recent_r)
                e_ok = all(e < 1.5 for e in recent_e) and max(recent_e) - min(recent_e) < 0.5
                v_ok = all(v < 2.0 for v in recent_v) and max(recent_v) - min(recent_v) < 1.0
                converged = r_ok and e_ok and v_ok
            else:
                converged = False

            # ---- 策略坍塌检测：entropy 崩溃 ----
            collapsed = cur_e < 0.05 and len(r_window) >= 5

            # 更新进度条
            stop_reason = ""
            if converged:
                stop_reason = f" [CONVERGED: r>{80} e<1.5 v<2]"
            elif collapsed:
                stop_reason = " [COLLAPSED]"

            postfix = {
                "tr": f"{cur_r:.0f}",
                "ent": f"{cur_e:.2f}",
                "v": f"{cur_v:.2f}",
                "stal": f"{stale_count}/{STALE_PATIENCE}" + stop_reason,
            }
            if val_r != 0.0:
                postfix["val"] = f"{val_r:.0f}"
            pbar.set_postfix(postfix)

            # 实时保存曲线（每 10 epoch + 容错）
            if (ep + 1) % 10 == 0:
                _save_plot(trainer, save_dir)

            # 日志
            if (ep + 1) % 20 == 0:
                val_str = f" val={val_r:6.1f}" if val_r != 0.0 else ""
                tqdm.write(f"  Ep {trainer.epoch:4d} | tr={cur_r:6.1f}{val_str} "
                           f"ent={cur_e:.3f} v={cur_v:.3f} "
                           f"p={rstats['policy_loss']:.3f} "
                           f"clip={rstats['clip_fraction']:.1%}{stop_reason}")

            # 每 50 epoch 存档
            if (ep + 1) % 50 == 0:
                torch.save(model.state_dict(), latest_path)

            # 早停判断
            if collapsed:
                tqdm.write(f"  [早停] entropy={cur_e:.3f}，策略坍塌")
                torch.save(model.state_dict(), latest_path)
                break

            if converged:
                tqdm.write(f"  [早停] 三信号收敛 (r>80, ent<1.5, v<2)")
                break

            if stale_count >= STALE_PATIENCE:
                # reward 停滞 + entropy <1.5（说明不是探索中，是真不涨了）
                if cur_e < 2.0:
                    tqdm.write(f"  [早停] reward 停滞 {STALE_PATIENCE} ep (best={best_reward:.1f} @ep{best_epoch})，停止训练")
                    break
                else:
                    # entropy 还高 → 还在探索，给更多耐心
                    stale_count = STALE_PATIENCE // 2
                    tqdm.write(f"  [info] reward 停滞但 entropy={cur_e:.2f} 仍高，延长探索")

        pbar.close()


def evaluate(model, folder, verbose=True, device="cpu"):
    """
    评估模型在给定数据集上的表现。

    Returns:
        dict: 评估指标
    """
    model.eval()
    try:
        env = AssemblyEnv(folder)
    except Exception as e:
        print(f"  [eval error] {folder}: {e}")
        return {}

    total_reward = 0.0
    n_episodes = 0
    success_count = 0

    for _ in range(5):  # 运行 5 个 episode
        obs = env.reset()
        ep_reward = 0.0
        while True:
            obs_t = _obs_to_tensor(obs, device=device)
            action, _, _, _ = model.get_action(obs_t, deterministic=True)

            # 检查是否无候选动作
            if obs["candidates"] is None or len(obs["candidates"]) == 0:
                break

            if action >= len(obs["candidates"]):
                break

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break

        total_reward += ep_reward
        n_episodes += 1
        if len(env.placed_parts) == env.n_parts:
            success_count += 1

    avg_reward = total_reward / max(n_episodes, 1)
    completion_rate = success_count / max(n_episodes, 1)

    if verbose:
        print(f"  {folder}: avg_reward={avg_reward:.2f} (装配质量), "
              f"完成率={completion_rate:.0%} ({success_count}/{n_episodes} 全放完)")

    return {"avg_reward": avg_reward, "completion_rate": completion_rate, "n_episodes": n_episodes}


def _obs_to_tensor(obs, device="cpu"):
    """numpy → torch tensor，发送到指定设备"""
    obs_t = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            if k == "edge_index":
                obs_t[k] = torch.from_numpy(v).long().to(device)
            else:
                obs_t[k] = torch.from_numpy(v).float().to(device)
        elif isinstance(v, list):
            obs_t[k] = v
        else:
            obs_t[k] = v
    return obs_t


def main():
    parser = argparse.ArgumentParser(description="RL 装配训练")
    parser.add_argument("--folder", type=str, default=None,
                        help="单数据集训练（覆盖 curriculum）")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"总训练轮数（默认 {NUM_EPOCHS}）")
    parser.add_argument("--eval", action="store_true",
                        help="仅评估（不训练）")
    parser.add_argument("--load", type=str, default=None,
                        help="加载已有模型权重")
    parser.add_argument("--save", type=str, default=None,
                        help="模型保存路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    # 初始化模型
    model = AssemblyGNN().to(device)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print(f"已加载模型: {args.load}")

    # 评估模式
    if args.eval:
        print(f"\n{'='*60}")
        print(f"  评估模式")
        print(f"{'='*60}")
        folders = [args.folder] if args.folder else CURRICULUM
        for folder in folders:
            evaluate(model, folder, device=str(device))
        return

    # 训练数据
    if args.folder:
        curriculum = [args.folder]
        epochs_per_stage = args.epochs
    else:
        curriculum = CURRICULUM
        epochs_per_stage = max(1, args.epochs // len(curriculum))

    # PPO 训练
    trainer = PPOTrainer(model, device=str(device))
    train_curriculum(model, trainer, curriculum, epochs_per_stage)

    # 每阶段结束后评估
    print(f"\n{'='*60}")
    print(f"  最终评估")
    print(f"{'='*60}")
    for folder in curriculum:
        evaluate(model, folder, device=str(device))

    # 保存路径已在训练循环中自动保存
    print(f"\n模型文件:")
    print(f"  最优: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best_model.pt')}")
    print(f"  最新: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'latest_model.pt')}")

    # 训练曲线
    plot_training_curve(trainer)


def _save_plot(trainer, save_dir=None):
    """保存训练曲线（轻量版，训练中每 10 epoch 调用）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    stats = trainer.stats
    epochs = range(1, len(stats["policy_loss"]) + 1)
    if len(epochs) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (key, color, label) in zip(
        axes.flat,
        [("policy_loss", "tab:blue", "Policy Loss"),
         ("value_loss", "tab:red", "Value Loss"),
         ("entropy", "tab:green", "Entropy"),
         ("clip_fraction", "tab:orange", "Clip Fraction")],
    ):
        ax.plot(epochs, stats[key], color=color, alpha=0.7, linewidth=0.8)
        ax.set_title(label); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "training_curve.png") if save_dir else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "training_curve.png")
    plt.savefig(out, dpi=100)
    plt.close(fig)


def plot_training_curve(trainer):
    """绘制训练曲线并保存为 PNG"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [info] matplotlib 未安装，跳过绘图")
        return

    stats = trainer.stats
    epochs = range(1, len(stats["policy_loss"]) + 1)
    if len(epochs) < 2:
        print("  [info] 数据不足，跳过绘图")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training Curves", fontsize=14)

    # Policy Loss
    ax = axes[0, 0]
    ax.plot(epochs, stats["policy_loss"], color="tab:blue", alpha=0.7, linewidth=0.8)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Value Loss
    ax = axes[0, 1]
    ax.plot(epochs, stats["value_loss"], color="tab:red", alpha=0.7, linewidth=0.8)
    ax.set_title("Value Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    ax.plot(epochs, stats["entropy"], color="tab:green", alpha=0.7, linewidth=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="target ~0.5")
    ax.set_title("Entropy (exploration)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Clip Fraction
    ax = axes[1, 1]
    ax.plot(epochs, stats["clip_fraction"], color="tab:orange", alpha=0.7, linewidth=0.8)
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="max healthy ~20%")
    ax.set_title("Clip Fraction")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Fraction")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"训练曲线已保存: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
