"""
RL 模块 — 基于 GNN + PPO 的智能装配匹配

用法:
    from RL import AssemblyEnv, PPOTrainer, train

目录:
    env.py      — RL 环境，封装装配逻辑
    models.py   — GNN 编码器 + 打分网络 + 价值网络
    reward.py   — 四层奖励函数
    ppo.py      — PPO 训练器（GAE + clip）
    train.py    — 训练入口 + curriculum learning
    config.py   — 超参数
"""
