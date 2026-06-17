"""
GNN + PPO 神经网络模型
======================

架构概览:
  NodeEncoder  → 原始特征 → d 维 embedding
  GNNLayer    → 消息传递（同零件内特征交互）
  Scorer      → 候选配对打分（Actor 的动作概率）
  ValueNet    → 状态价值估计（Critic）

设计要点:
  - 候选动作数量可变 → Scorer 逐对打分而非固定输出维度
  - 同零件内全连接图 → 卷积层数 = 零件直径
  - 与 env.py 解耦 → 纯 torch 模块，不依赖 cadquery
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from RL.config import (
    NODE_FEAT_DIM, HIDDEN_DIM, NUM_GNN_LAYERS, DROPOUT,
)


# ==================== 基础模块 ====================

class NodeEncoder(nn.Module):
    """
    将原始特征映射到隐空间。

    输入: (batch, NODE_FEAT_DIM) → 输出: (batch, HIDDEN_DIM)
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(NODE_FEAT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class EdgeEncoder(nn.Module):
    """
    边特征编码。

    输入: (batch, 3)  → 输出: (batch, HIDDEN_DIM)
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM),
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)


class GNNLayer(nn.Module):
    """
    单层消息传递 GNN。

    消息计算:  msg_{i←j} = MLP([h_i, h_j, e_ij])
    聚合:      agg_i = mean(msg_{i←j} for j in neighbors(i))
    更新:      h_i' = h_i + Dropout(ReLU(LayerNorm(MLP([h_i, agg_i]))))
    """

    def __init__(self):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, edge_emb):
        """
        Args:
            x: (N, HIDDEN_DIM) 节点特征
            edge_index: (2, E) 边索引
            edge_emb: (E, HIDDEN_DIM) 边特征

        Returns:
            (N, HIDDEN_DIM) 更新后的节点特征
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # 消息: [h_src, h_dst, e] → msg
        msg_input = torch.cat([x[src], x[dst], edge_emb], dim=-1)
        messages = self.msg_mlp(msg_input)  # (E, HIDDEN_DIM)

        # 聚合: scatter_mean
        aggregated = torch.zeros(N, HIDDEN_DIM, device=x.device)
        aggregated = aggregated.index_add(0, dst, messages)
        # 计算每个节点的入度用于平均
        degree = torch.zeros(N, device=x.device)
        degree = degree.index_add(0, dst, torch.ones(dst.shape[0], device=x.device))
        degree = degree.clamp(min=1)
        aggregated = aggregated / degree.unsqueeze(-1)

        # 更新
        update_input = torch.cat([x, aggregated], dim=-1)
        delta = self.update_mlp(update_input)
        return x + self.dropout(delta)


class GlobalEncoder(nn.Module):
    """
    全局特征编码。

    输入: (batch, 4)  → 输出: (batch, HIDDEN_DIM)
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM),
        )

    def forward(self, global_feat):
        return self.mlp(global_feat)


# ==================== 打分网络（Actor）====================

class CandidateScorer(nn.Module):
    """
    候选配对打分网络。

    对每对候选 (src_node, dst_node) 计算兼容性分数。

    输入:
      - src_emb: (num_candidates, HIDDEN_DIM) 源节点 embedding
      - dst_emb: (num_candidates, HIDDEN_DIM) 目标节点 embedding
      - global_emb: (1, HIDDEN_DIM) 全局图 embedding
      - mask_emb: (N_nodes, HIDDEN_DIM) 可选，已放置/未放置信息

    输出:
      - scores: (num_candidates,) 原始分数（未归一化）

    打分公式:
      score = MLP([src, dst, src-dst, src*dst, global])
    其中 src-dst 捕捉对称/反对称关系，src*dst 捕捉共线性。
    """

    def __init__(self):
        super().__init__()
        input_dim = HIDDEN_DIM * 4 + HIDDEN_DIM  # src, dst, diff, hadamard, global
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, src_emb, dst_emb, global_emb):
        """
        Args:
            src_emb: (K, HIDDEN_DIM)
            dst_emb: (K, HIDDEN_DIM)
            global_emb: (1, HIDDEN_DIM)

        Returns:
            (K,) 原始分数
        """
        K = src_emb.shape[0]
        g = global_emb.expand(K, -1)  # broadcast
        diff = src_emb - dst_emb
        hadamard = src_emb * dst_emb
        x = torch.cat([src_emb, dst_emb, diff, hadamard, g], dim=-1)
        return self.mlp(x).squeeze(-1)


# ==================== 价值网络（Critic）====================

class ValueNet(nn.Module):
    """
    状态价值估计 V(s)。

    从池化后的图 embedding + 全局特征 → 标量 V。

    输入:
      - pooled_emb: (1, HIDDEN_DIM) 全局平均池化
      - global_emb: (1, HIDDEN_DIM) 全局特征 embedding

    输出:
      - value: (1,) 标量
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, pooled_emb, global_emb):
        x = torch.cat([pooled_emb, global_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


# ==================== 完整模型 ====================

class AssemblyGNN(nn.Module):
    """
    完整的装配 GNN + Actor-Critic 模型。

    一次 forward 完成:
      1. 节点/边编码
      2. GNN 消息传递（NUM_GNN_LAYERS 层）
      3. 图池化（全局平均 + 已放置节点平均）
      4. 候选打分（Actor）→ action logits
      5. 价值估计（Critic）→ V(s)
    """

    def __init__(self):
        super().__init__()

        # 编码器
        self.node_encoder = NodeEncoder()
        self.edge_encoder = EdgeEncoder()
        self.global_encoder = GlobalEncoder()

        # GNN 层
        self.gnn_layers = nn.ModuleList([
            GNNLayer() for _ in range(NUM_GNN_LAYERS)
        ])

        # Actor & Critic
        self.scorer = CandidateScorer()
        self.value_net = ValueNet()

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def encode_graph(self, obs):
        """
        编码图结构 → 节点 embeddings + 全局 embedding。

        Args:
            obs: dict with keys:
              - node_features: (N, NODE_FEAT_DIM)
              - edge_index: (2, E)
              - edge_attr: (E, 3)
              - mask: (N,)
              - global_feat: (4,)

        Returns:
            node_emb: (N, HIDDEN_DIM)
            global_emb: (HIDDEN_DIM,)
            pooled: (HIDDEN_DIM,)
        """
        x = obs["node_features"]
        edge_index = obs["edge_index"]
        edge_attr = obs["edge_attr"]
        global_feat = obs["global_feat"]

        # 编码
        h = self.node_encoder(x)                    # (N, HIDDEN_DIM)
        e_emb = self.edge_encoder(edge_attr)        # (E, HIDDEN_DIM)
        g_emb = self.global_encoder(global_feat)    # (HIDDEN_DIM,)

        # GNN 消息传递
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index, e_emb)

        # 图池化: 全局平均 + 已放置节点平均
        mask = obs["mask"]
        pooled_all = h.mean(dim=0)                   # 所有节点平均

        placed_mask = mask.bool()
        if placed_mask.any():
            pooled_placed = h[placed_mask].mean(dim=0)
        else:
            pooled_placed = torch.zeros(HIDDEN_DIM, device=h.device)

        # 拼接作为图表示
        pooled = pooled_all + pooled_placed

        return h, g_emb, pooled

    def forward(self, obs):
        """
        单次前向传播。

        Args:
            obs: 环境观测（batch_size=1 时直接传 dict）

        Returns:
            action_logits: (K,) 候选动作的对数概率
            value: (1,) 状态价值
            node_emb: (N, HIDDEN_DIM) 节点 embedding（用于分析）
        """
        h, g_emb, pooled = self.encode_graph(obs)

        # 候选打分
        candidates = obs.get("candidates", [])
        if len(candidates) > 0:
            import numpy as np
            src_idx = [c[0] for c in candidates]
            dst_idx = [c[1] for c in candidates]
            src_idx_t = torch.tensor(src_idx, dtype=torch.long, device=h.device)
            dst_idx_t = torch.tensor(dst_idx, dtype=torch.long, device=h.device)

            src_emb = h[src_idx_t]   # (K, HIDDEN_DIM)
            dst_emb = h[dst_idx_t]   # (K, HIDDEN_DIM)

            action_logits = self.scorer(src_emb, dst_emb, g_emb.unsqueeze(0))
        else:
            action_logits = torch.zeros(0, device=h.device)

        # 价值估计
        value = self.value_net(pooled.unsqueeze(0), g_emb.unsqueeze(0))

        return action_logits, value, h

    def get_action(self, obs, deterministic=False):
        """
        根据观测选择动作。

        Args:
            obs: 环境观测
            deterministic: True=贪心选择, False=按概率采样

        Returns:
            action_idx: int 候选动作索引
            log_prob: float 对数概率
            value: float 状态价值
            entropy: float 策略熵
        """
        logits, value, _ = self.forward(obs)

        if logits.shape[0] == 0:
            return 0, 0.0, value.item(), 0.0

        # 稳定化 logits（防止数值溢出）
        logits = logits - logits.max()

        # softmax → 动作概率
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob.item(), value.item(), entropy.item()

    def evaluate_action(self, obs, action_idx):
        """
        评估指定动作（用于 PPO 更新）。

        **保持计算图**：返回值均为 tensor，用于梯度反向传播。

        Args:
            obs: 环境观测
            action_idx: int 动作索引

        Returns:
            log_prob: (1,) tensor
            entropy: (1,) tensor
            value: (1,) tensor
        """
        logits, value, _ = self.forward(obs)

        if logits.shape[0] == 0:
            return (
                torch.tensor(0.0, device=value.device, requires_grad=True),
                torch.tensor(0.0, device=value.device, requires_grad=True),
                value,
            )

        logits = logits - logits.max()
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_t = torch.tensor(action_idx, device=logits.device)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()

        return log_prob, entropy, value
