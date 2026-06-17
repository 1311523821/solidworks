"""
RL 超参数配置
=============
所有可调参数集中在此文件，方便实验和调参。
"""

# ====== GNN 网络结构 ======
NODE_FEAT_DIM = 11          # 每个特征节点的原始维度（见 env.py _node_features）
HIDDEN_DIM = 128            # 隐层维度（GNN 层 + MLP 共用）
NUM_GNN_LAYERS = 3          # 消息传递层数
DROPOUT = 0.1               # Dropout 比例

# ====== PPO 算法参数 ======
CLIP_EPSILON = 0.2          # PPO clip 范围 [1-ε, 1+ε]
GAMMA = 0.99                # 折扣因子（远期奖励权重）
GAE_LAMBDA = 0.95           # GAE 平滑参数（越大越看重远期 advantage）
LR_ACTOR = 3e-4             # Actor 学习率
LR_CRITIC = 1e-3            # Critic 学习率（通常比 actor 大）
ENTROPY_COEF = 0.05         # 初始熵正则化系数（鼓励探索，逐步衰减）
ENTROPY_COEF_MIN = 0.005    # 最小熵系数
ENTROPY_DECAY = 0.995       # 每 epoch 衰减因子
VALUE_COEF = 0.5            # 价值损失权重
MAX_GRAD_NORM = 1.0         # 梯度裁剪阈值
REWARD_CLIP = 50.0          # 奖励裁剪（防止极端值冲击训练）

# ====== 训练控制 ======
NUM_EPOCHS = 500            # 总训练轮数（遍历所有数据集）
STEPS_PER_EPOCH = 100       # 每轮收集的经验步数
PPO_EPOCHS = 10             # 每批经验重训练的 PPO 更新次数
BATCH_SIZE = 64             # PPO 更新时的小批次大小
LR_WARMUP_EPOCHS = 5        # 学习率预热 epoch 数

# ====== 奖励权重 ======
W_FEATURE_MATCH = 1.0       # Layer 1: 特征匹配质量
W_PHYSICS = 2.0             # Layer 2: 物理合理性
W_COMPLETENESS = 1.0        # Layer 3: 装配完整性
W_CONSTRAINT = 0.5          # Layer 4: 约束质量

# ====== 几何预过滤阈值（缩小候选动作空间）=====
MATCH_TOL = 0.5             # 圆周长/线段长度匹配容差（比规则版 TOL=0.1 宽松）
CYL_RADIUS_TOL = 1.0        # 圆柱半径匹配容差（比规则版 CYL_TOL=0.5 宽松）
MIN_MATCHED_FEATURES = 1    # 最少匹配特征数（比规则版 MIN_OK=3 低很多）

# ====== Curriculum Learning ======
# 训练数据文件夹，按难度递增
CURRICULUM = ["./1", "./2", "./3"]


