"""
PPO 训练器 — Proximal Policy Optimization
========================================

实现 PPO-Clip + GAE (Generalized Advantage Estimation)。

核心公式:
  - PPO Clip:  L = min(r*A, clip(r, 1-ε, 1+ε)*A)
  - GAE:       A_t = δ_t + γλ·δ_{t+1} + ... ,  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  - 总损失:    L_total = L_policy - c1·L_value + c2·entropy

参考: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from RL.config import (
    CLIP_EPSILON, GAMMA, GAE_LAMBDA,
    LR_ACTOR, LR_CRITIC, ENTROPY_COEF, ENTROPY_COEF_MIN, ENTROPY_DECAY,
    VALUE_COEF, REWARD_CLIP,
    PPO_EPOCHS, BATCH_SIZE, MAX_GRAD_NORM, LR_WARMUP_EPOCHS,
)


class RolloutBuffer:
    """
    经验缓冲区 — 存储一条轨迹的 (s, a, r, log_prob, value, done)。

    每收集 STEPS_PER_EPOCH 步经验后清空。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []       # 观测 dict 列表
        self.actions = []      # int 列表
        self.rewards = []      # float 列表
        self.log_probs = []    # float 列表
        self.values = []       # float 列表
        self.dones = []        # bool 列表

    def add(self, obs, action, reward, log_prob, value, done):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def size(self):
        return len(self.actions)


class PPOTrainer:
    """
    PPO 训练器。

    用法:
        model = AssemblyGNN()
        trainer = PPOTrainer(model)
        for epoch in range(NUM_EPOCHS):
            trainer.collect_rollout(env, steps=200)
            trainer.update()
            trainer.buffer.reset()
    """

    def __init__(self, model, lr_actor=None, lr_critic=None, device="cpu"):
        """
        Args:
            model: AssemblyGNN 实例
            lr_actor: Actor 学习率（默认从 config 读取）
            lr_critic: Critic 学习率
            device: "cpu" 或 "cuda"
        """
        self.model = model
        self.device = device

        # 分离 Actor 和 Critic 参数（不同学习率）
        actor_params = []
        critic_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "scorer" in name:
                actor_params.append(param)
            elif "value_net" in name:
                critic_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = torch.optim.Adam([
            {"params": actor_params, "lr": lr_actor or LR_ACTOR},
            {"params": critic_params, "lr": lr_critic or LR_CRITIC},
            {"params": other_params, "lr": (lr_actor or LR_ACTOR) * 0.5},  # 编码器学习率减半
        ])

        self.buffer = RolloutBuffer()

        # ---- 熵衰减 ----
        self.entropy_coef = ENTROPY_COEF  # 当前值，逐步衰减

        # ---- 学习率预热 ----
        self.warmup_epochs = LR_WARMUP_EPOCHS
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]

        # ---- 奖励归一化 + 裁剪 ----
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_momentum = 0.01

        # ---- 观测归一化（running stats，仅对 node_features） ----
        self.obs_mean = None
        self.obs_std = None
        self.obs_momentum = 0.01

        # 统计
        self.total_steps = 0
        self.epoch = 0
        self.stats = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "total_reward": [], "avg_reward": [],
            "clip_fraction": [],
        }

    def collect_rollout(self, env, steps=100, deterministic=False):
        """
        收集一段轨迹经验。

        Args:
            env: AssemblyEnv 实例
            steps: 收集步数
            deterministic: 是否贪心（评估模式）

        Returns:
            dict: 统计信息
        """
        obs = env.reset()
        episode_rewards = []
        total_r = 0.0

        for _ in range(steps):
            # 转 tensor
            obs_t = self._obs_to_tensor(obs)

            # 选动作
            action, log_prob, value, entropy = self.model.get_action(
                obs_t, deterministic=deterministic
            )

            # 执行
            next_obs, reward, done, info = env.step(action)

            # 存储
            # 奖励裁剪（防止极端值冲击梯度）
            reward = max(-REWARD_CLIP, min(REWARD_CLIP, reward))
            self.buffer.add(obs_t, action, reward, log_prob, value, done)
            total_r += reward

            if done:
                episode_rewards.append(total_r)
                total_r = 0.0
                obs = env.reset()
            else:
                obs = next_obs

        self.total_steps += steps

        return {
            "episodes_completed": len(episode_rewards),
            "avg_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "steps": steps,
        }

    def update(self):
        """
        PPO 更新 — 用 buffer 中的经验做多轮优化。

        Returns:
            dict: 训练损失统计
        """
        if self.buffer.size() == 0:
            return {}

        # ---- 1. 计算 GAE advantage ----
        advantages, returns = self._compute_gae()

        # ---- 2. 转为 tensor（放到模型所在设备）----
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)

        # ---- 3. 标准化 advantage + 奖励归一化 ----
        # 更新运行统计（用于奖励 scaling）
        returns_mean = returns_t.mean().item()
        returns_std = returns_t.std().item() if returns_t.std() > 0 else 1.0
        self.reward_mean = (
            (1 - self.reward_momentum) * self.reward_mean + self.reward_momentum * returns_mean
        )
        self.reward_std = (
            (1 - self.reward_momentum) * self.reward_std + self.reward_momentum * returns_std
        )

        # 归一化 returns（稳定 value loss）
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        if advantages_t.std() > 0:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ---- 4. PPO 多轮更新 ----
        buffer_size = self.buffer.size()
        indices = np.arange(buffer_size)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
        n_updates = 0

        for _ in range(PPO_EPOCHS):
            # 随机打乱
            np.random.shuffle(indices)

            for start in range(0, buffer_size, BATCH_SIZE):
                batch_idx = indices[start:start + BATCH_SIZE]
                if len(batch_idx) == 0:
                    continue

                # 提取 batch
                batch_states = [self.buffer.states[i] for i in batch_idx]
                batch_actions = [self.buffer.actions[i] for i in batch_idx]
                batch_adv = advantages_t[batch_idx]
                batch_ret = returns_t[batch_idx]
                batch_old_lp = old_log_probs_t[batch_idx]
                batch_old_v = old_values_t[batch_idx]

                # 前向传播（对 batch 中的每个 sample 分别计算，保持计算图）
                new_lps = []
                new_ents = []
                new_vals = []

                for i, state in enumerate(batch_states):
                    log_prob, entropy, value = self.model.evaluate_action(
                        state, batch_actions[i]
                    )
                    # evaluate_action 返回 tensor（保持计算图）
                    new_lps.append(log_prob)
                    new_ents.append(entropy)
                    new_vals.append(value)

                # stack 保持梯度连接
                new_lps_t = torch.stack(new_lps)
                new_ents_t = torch.stack(new_ents)
                new_vals_t = torch.stack(new_vals)

                # ---- PPO Clip 损失 ----
                ratio = torch.exp(new_lps_t - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # clip fraction（监控用）
                clip_frac = ((ratio - 1).abs() > CLIP_EPSILON).float().mean().item()

                # ---- Value 损失 ----
                # clipped value loss (PPO paper 中的 value clipping)
                v_clipped = batch_old_v + torch.clamp(
                    new_vals_t - batch_old_v, -CLIP_EPSILON, CLIP_EPSILON
                )
                v_loss1 = (new_vals_t - batch_ret) ** 2
                v_loss2 = (v_clipped - batch_ret) ** 2
                value_loss = torch.max(v_loss1, v_loss2).mean()

                # ---- 总损失 ----
                entropy_loss = -new_ents_t.mean()
                total_loss = policy_loss + VALUE_COEF * value_loss + self.entropy_coef * entropy_loss

                # ---- 反向传播 ----
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += -entropy_loss.item()
                stats["clip_fraction"] += clip_frac
                n_updates += 1

        # 平均
        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates

        # 记录
        self.stats["policy_loss"].append(stats["policy_loss"])
        self.stats["value_loss"].append(stats["value_loss"])
        self.stats["entropy"].append(stats["entropy"])
        self.stats["clip_fraction"].append(stats["clip_fraction"])

        # ---- 熵衰减：前期多探索，后期收束 ----
        self.entropy_coef = max(ENTROPY_COEF_MIN, self.entropy_coef * ENTROPY_DECAY)

        # ---- 学习率：预热 + 余弦衰减 ----
        self._update_lr()

        self.epoch += 1
        return stats

    def _compute_gae(self):
        """
        计算 GAE (Generalized Advantage Estimation)。

        GAE 公式:
          δ_t  = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
          A_t  = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...

        Returns:
          (advantages, returns) — 同等长度的 list
        """
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        n = len(rewards)

        # 最后一步的 V(s_{t+1}) 设为 0（轨迹截断）
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0  # terminal
                next_done = dones[t]
            else:
                next_value = values[t + 1]
                next_done = dones[t]

            delta = rewards[t] + GAMMA * next_value * (1 - int(next_done)) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - int(dones[t])) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def _obs_to_tensor(self, obs):
        """将 numpy 观测转为 torch tensor，发送到模型所在设备，含观测归一化"""
        obs_t = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                if k == "node_features":
                    # 观测归一化（running z-score）
                    nf = v.copy()
                    if self.obs_mean is None:
                        self.obs_mean = np.mean(nf, axis=0)
                        self.obs_std = np.std(nf, axis=0).clip(min=1e-6)
                    else:
                        self.obs_mean = (1 - self.obs_momentum) * self.obs_mean + self.obs_momentum * np.mean(nf, axis=0)
                        self.obs_std = (1 - self.obs_momentum) * self.obs_std + self.obs_momentum * np.std(nf, axis=0).clip(min=1e-6)
                    nf = (nf - self.obs_mean) / self.obs_std.clip(min=1e-6)
                    obs_t[k] = torch.from_numpy(nf).float().to(self.device)
                elif k == "edge_index":
                    obs_t[k] = torch.from_numpy(v).long().to(self.device)
                elif k == "mask":
                    obs_t[k] = torch.from_numpy(v).float().to(self.device)
                else:
                    obs_t[k] = torch.from_numpy(v).float().to(self.device)
            elif isinstance(v, list):
                obs_t[k] = v
            else:
                obs_t[k] = v
        return obs_t

    def _update_lr(self):
        """学习率预热 + 余弦衰减"""
        if self.epoch < self.warmup_epochs:
            # 预热阶段：从 0 线性增长到 base_lr
            scale = (self.epoch + 1) / self.warmup_epochs
        else:
            # 余弦衰减：从 base_lr 平滑降到 0
            import math
            progress = (self.epoch - self.warmup_epochs) / max(500 - self.warmup_epochs, 1)
            progress = min(progress, 1.0)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            scale = max(scale, 0.05)  # 不低于 5%
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_lrs[i] * scale

    def get_stats(self):
        """获取最近的训练统计"""
        return {
            "policy_loss": np.mean(self.stats["policy_loss"][-10:]) if self.stats["policy_loss"] else 0,
            "value_loss": np.mean(self.stats["value_loss"][-10:]) if self.stats["value_loss"] else 0,
            "entropy": np.mean(self.stats["entropy"][-10:]) if self.stats["entropy"] else 0,
            "clip_fraction": np.mean(self.stats["clip_fraction"][-10:]) if self.stats["clip_fraction"] else 0,
        }
