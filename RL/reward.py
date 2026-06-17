"""
奖励函数模块 — 四层奖励信号
===========================

奖励设计原则：
  Layer 1: 每步即时 — 特征匹配质量（密集信号，引导基本行为）
  Layer 2: 每步即时 — 物理合理性（碰撞、间隙、面心距离）
  Layer 3: Episode结束 — 装配完整性（放置率、效率）
  Layer 4: Episode结束 — 约束质量（DOF 约束度、装配刚度）

所有奖励函数独立、可组合，权重在 config.py 中配置。
"""

import math
import cadquery as cq


def reward_feature_match(fa, fb, matched_circles, matched_lines):
    """
    Layer 1: 单次特征面匹配的即时奖励（每步都计算）

    引导 Agent 学到：
      - 匹配越多特征越好（n_matched ↑）
      - 匹配精度越高越好（长度偏差 ↓）
      - 两面装配后法向应对齐（dot → -1）
      - 大面配大面（面积比 ↑）

    Args:
        fa, fb: 两个面的特征字典 (含 circles, lines, area, n, c)
        matched_circles: [(circle_a, circle_b), ...] 匹配的圆列表
        matched_lines: [(line_a, line_b), ...] 匹配的线段列表

    Returns:
        float: 奖励值，范围约 [-20, +30]
    """
    r = 0.0

    # 1. 匹配特征数量（每多一个 +1）
    n_matched = len(matched_circles) + len(matched_lines)
    r += n_matched * 1.0

    # 2. 匹配精度惩罚（圆周长偏差）
    for ca, cb in matched_circles:
        r -= abs(ca["len"] - cb["len"]) * 10.0

    # 3. 匹配精度惩罚（线段长度偏差）
    for la, lb in matched_lines:
        r -= abs(la["len"] - lb["len"]) * 10.0

    # 4. 面法向对齐奖励
    #    装配时两面法向应相反（面对面贴合）→ dot(na, nb) 应接近 -1
    #    dot=-1 → +10, dot=0 → +5, dot=1 → 0
    na = fa["n"]
    nb = fb["n"]
    normal_dot = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]
    r += (1.0 + normal_dot) * 5.0  # dot=-1 时 max(+10), dot=1 时 min(0)

    # 5. 面积相似性（避免大面配小面）
    if fa.get("area", 0) > 0 and fb.get("area", 0) > 0:
        area_ratio = min(fa["area"], fb["area"]) / max(fa["area"], fb["area"])
        r += area_ratio * 3.0

    return r


def reward_physics(new_bb, placed_bbs, face_center_a, face_center_b,
                   face_normal_a, face_normal_b):
    """
    Layer 2: 装配物理合理性（每步即时）

    引导 Agent 学到：
      - 避免零件碰撞（体积交叠 ↓）
      - 配合面心对齐（距离 ↓）
      - 非配合面保持合理间隙

    Args:
        new_bb: 新放置零件的 BoundingBox (xmin, xmax, ymin, ymax, zmin, zmax)
        placed_bbs: 已放置零件的 BoundingBox 列表
        face_center_a, face_center_b: 世界坐标下的面心
        face_normal_a, face_normal_b: 世界坐标下的面法向

    Returns:
        float: 奖励值
    """
    r = 0.0

    # 6. 碰撞检测（粗略 AABB 交叠）
    for p_bb in placed_bbs:
        overlap = _aabb_overlap_volume(new_bb, p_bb)
        new_vol = _aabb_volume(new_bb)
        if new_vol > 0:
            r -= (overlap / new_vol) * 20.0  # 按体积归一化

    # 7. 配合面心距离（世界坐标下两面心应重合）
    face_dist = math.sqrt(
        (face_center_a[0] - face_center_b[0]) ** 2 +
        (face_center_a[1] - face_center_b[1]) ** 2 +
        (face_center_a[2] - face_center_b[2]) ** 2
    )
    r -= face_dist * 2.0  # 每 mm 偏差 -2

    # 8. 法向对齐（装配后法向应相反）
    normal_dot = (
        face_normal_a[0] * face_normal_b[0] +
        face_normal_a[1] * face_normal_b[1] +
        face_normal_a[2] * face_normal_b[2]
    )
    # dot 应 ≈ -1（面对面），偏离越多惩罚越大
    r -= abs(normal_dot + 1.0) * 5.0

    return r


def reward_completeness(n_placed, n_total, step_count):
    """
    Layer 3: 装配完整性（Episode 结束时计算）

    Args:
        n_placed: 已成功放置的零件数
        n_total: 总零件数
        step_count: 使用的步数

    Returns:
        float: 奖励值
    """
    r = 0.0

    if n_total == 0:
        return 0.0

    # 9. 放置成功率
    placement_rate = n_placed / n_total
    r += placement_rate * 100.0  # 全放满 = +100

    # 10. 效率奖励（同样结果越少步数越好）
    if n_placed == n_total:
        # 理想情况：每个零件一步，N 个零件需要 N-1 步
        ideal_steps = n_total - 1
        efficiency = max(0, 1.0 - (step_count - ideal_steps) / max(ideal_steps * 2, 1))
        r += efficiency * 30.0

    # 11. 未放置零件惩罚
    n_unplaced = n_total - n_placed
    r -= n_unplaced * 20.0

    return r


def reward_constraint_quality(placed_parts_data):
    """
    Layer 4: 约束质量（Episode 结束时计算）

    评估装配体的整体约束状态。

    Args:
        placed_parts_data: [(part_name, labels_used, bbox), ...]

    Returns:
        float: 奖励值
    """
    r = 0.0

    if not placed_parts_data:
        return 0.0

    for part_name, labels_used, bbox in placed_parts_data:
        # 12. 约束度检查
        dof = _estimate_constrained_dof(labels_used)
        if dof >= 6:
            r += 10.0  # 全约束
        elif dof >= 4:
            r += 5.0   # 轴对齐 → 至少 4-DOF
        elif dof >= 2:
            r += 1.0   # 1-DOF 太低
        else:
            r -= 15.0  # 欠约束

        # 13. 过约束惩罚
        if dof > 6:
            r -= (dof - 6) * 8.0

    # 14. 装配重心（越低越好，避免头重脚轻）
    if len(placed_parts_data) >= 2:
        total_z = 0.0
        total_vol = 0.0
        min_z = float("inf")
        max_z = float("-inf")
        for _, _, bb in placed_parts_data:
            vol = _aabb_volume(bb)
            cz = (bb[4] + bb[5]) / 2  # z 中心
            total_z += cz * vol
            total_vol += vol
            min_z = min(min_z, bb[4])
            max_z = max(max_z, bb[5])

        if total_vol > 0:
            cog_z = total_z / total_vol
            height = max_z - min_z
            if height > 0:
                # 重心位于下半部（0~0.5）为佳
                cog_ratio = (cog_z - min_z) / height
                r += (1.0 - abs(cog_ratio - 0.4)) * 5.0

    return r


# ====== 辅助函数 ======

def _aabb_volume(bb):
    """BoundingBox 体积，(xmin,xmax,ymin,ymax,zmin,zmax)"""
    if len(bb) != 6:
        return 0
    return (bb[1] - bb[0]) * (bb[3] - bb[2]) * (bb[5] - bb[4])


def _aabb_overlap_volume(bb1, bb2):
    """两个 AABB 的交叠体积"""
    dx = max(0, min(bb1[1], bb2[1]) - max(bb1[0], bb2[0]))
    dy = max(0, min(bb1[3], bb2[3]) - max(bb1[2], bb2[2]))
    dz = max(0, min(bb1[5], bb2[5]) - max(bb1[4], bb2[4]))
    return dx * dy * dz


def _estimate_constrained_dof(labels_used):
    """
    估算已约束的自由度。

    规则:
      - 1 个 PLANAR 标签约束 6 DOF（面贴面）
      - 1 个 CYLINDER 标签约束 4 DOF（轴对齐）
      - 组合约束取并集（粗略估计）
    """
    constrained = set()
    for label in labels_used:
        ud = label.get("userData", {})
        mt = ud.get("matchType")
        if mt == "PLANAR":
            # 面贴面: 约束 3 平移 + 3 旋转 = 6 DOF
            constrained.update(["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"])
        elif mt == "CYLINDER":
            if ud.get("boreToBore"):
                # 孔-孔: 约束 2 平移 + 2 旋转（沿轴可滑可转）= 4 DOF
                constrained.update(["Tx", "Ty", "Rx", "Ry"])  # 径向约束
            else:
                # 轴-孔: 同上 4 DOF
                constrained.update(["Tx", "Ty", "Rx", "Ry"])
    return len(constrained)


# ====== 组合奖励函数 ======

def compute_total_reward(match_info=None, physics_info=None,
                          completeness_info=None, constraint_info=None,
                          weights=None):
    """
    计算加权总奖励。

    Args:
        match_info: (fa, fb, matched_circles, matched_lines) 或 None
        physics_info: (new_bb, placed_bbs, fc_a, fc_b, fn_a, fn_b) 或 None
        completeness_info: (n_placed, n_total, step_count) 或 None
        constraint_info: (placed_parts_data) 或 None
        weights: dict 覆盖默认权重

    Returns:
        dict: {"total": float, "match": float, "physics": float, ...}
    """
    from RL.config import (
        W_FEATURE_MATCH, W_PHYSICS, W_COMPLETENESS, W_CONSTRAINT
    )
    w = weights or {}
    w1 = w.get("match", W_FEATURE_MATCH)
    w2 = w.get("physics", W_PHYSICS)
    w3 = w.get("completeness", W_COMPLETENESS)
    w4 = w.get("constraint", W_CONSTRAINT)

    rewards = {"match": 0.0, "physics": 0.0, "completeness": 0.0, "constraint": 0.0}

    if match_info is not None:
        rewards["match"] = w1 * reward_feature_match(*match_info)

    if physics_info is not None:
        rewards["physics"] = w2 * reward_physics(*physics_info)

    if completeness_info is not None:
        rewards["completeness"] = w3 * reward_completeness(*completeness_info)

    if constraint_info is not None:
        rewards["constraint"] = w4 * reward_constraint_quality(constraint_info)

    rewards["total"] = sum(rewards.values())
    return rewards
