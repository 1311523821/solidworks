"""
RL 装配环境 — AssemblyEnv
=========================

将 STEP 零件装配建模为序列决策问题（MDP）：

  State:  图结构 — 节点=特征面/圆柱, 边=同零件内空间关系
  Action: 选择一个候选配对 (特征面_A, 特征面_B) 并放置新零件
  Reward: 四层奖励信号（见 reward.py）

复用现有模块:
  - feature_extractor.extract_file()  → 提取特征
  - label_generator 中的坐标变换逻辑  → 生成装配标签
"""

import os
import sys
import json
import math
import glob
import random
from typing import Optional

import cadquery as cq
from cadquery import importers

# 将父目录加入 path，以便导入项目模块
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from feature_extractor import extract_file
from label_generator import planar_labels, cylinder_labels, _ortho, _neg, _vec_sub, _bore_face_intersection
from RL.reward import compute_total_reward, _aabb_volume, _aabb_overlap_volume
from RL.config import (
    MATCH_TOL, CYL_RADIUS_TOL, MIN_MATCHED_FEATURES,
    NODE_FEAT_DIM,
)


class AssemblyEnv:
    """
    RL 装配环境。

    每个 episode = 一个文件夹中的 STEP 零件完整装配过程。

    用法:
        env = AssemblyEnv("./1")
        obs = env.reset()
        while True:
            action = agent.select_action(obs)  # 0 ~ env.num_candidates-1
            obs, reward, done, info = env.step(action)
            if done: break
    """

    def __init__(self, folder_path: str, max_steps: int = 50):
        """
        Args:
            folder_path: STEP 文件夹路径（如 "./1", "./2"）
            max_steps: 单 episode 最大步数
        """
        self.folder_path = folder_path
        self.max_steps = max_steps

        # 加载所有 STEP 文件
        self._load_parts()

    # ========== 数据加载 ==========

    def _load_parts(self):
        """加载文件夹中的所有 STEP 文件并提取特征（带缓存）"""
        step_files = list(set(
            glob.glob(os.path.join(self.folder_path, "*.step")) +
            glob.glob(os.path.join(self.folder_path, "*.stp"))
        ))
        step_files = [f for f in step_files if "virtual" not in os.path.basename(f)]
        step_files.sort()

        self.part_names = []         # 零件名列表
        self.part_shapes = {}        # name → cadquery Shape
        self.part_bboxes = {}        # name → (xmin,xmax,ymin,ymax,zmin,zmax)
        self.part_features = {}      # name → {"planar": [...], "cylinders": [...]}
        self.feature_nodes = []      # [(part_idx, feat_type, feat_idx), ...]  全局节点索引
        self.feature_to_global = {}  # (part_name, type, idx) → global_node_idx

        for fp in step_files:
            nm = os.path.splitext(os.path.basename(fp))[0]

            # 特征提取（复用缓存）
            cache_path = os.path.join(self.folder_path, f"{nm}_features.json")
            if os.path.exists(cache_path) and os.path.getmtime(fp) < os.path.getmtime(cache_path):
                import json as _json
                features = _json.load(open(cache_path, encoding="utf-8"))
            else:
                features = extract_file(fp)
                import json as _json
                with open(cache_path, "w", encoding="utf-8") as f:
                    _json.dump(features, f, indent=2)

            # 加载 shape（用于碰撞检测和坐标系变换）
            shape = importers.importStep(fp).val()
            bb = shape.BoundingBox()

            self.part_names.append(nm)
            self.part_shapes[nm] = shape
            self.part_bboxes[nm] = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)
            self.part_features[nm] = features

        # 构建全局节点索引
        node_idx = 0
        for pi, pn in enumerate(self.part_names):
            feats = self.part_features[pn]
            for fi, f in enumerate(feats.get("planar", [])):
                self.feature_nodes.append((pi, "planar", fi))
                self.feature_to_global[(pn, "planar", fi)] = node_idx
                node_idx += 1
            for fi, f in enumerate(feats.get("cylinders", [])):
                self.feature_nodes.append((pi, "cylinder", fi))
                self.feature_to_global[(pn, "cylinder", fi)] = node_idx
                node_idx += 1

        self.n_parts = len(self.part_names)
        self.n_nodes = len(self.feature_nodes)

    # ========== 环境重置 ==========

    def reset(self):
        """
        重置环境到初始状态。

        Returns:
            dict: 初始观测，包含:
              - node_features: (N_nodes, NODE_FEAT_DIM) 节点特征矩阵
              - edge_index: (2, N_edges) 边连接
              - edge_attr: (N_edges, 3) 边特征
              - mask: (N_nodes,) 节点 mask (1=已放置, 0=未放置)
              - candidates: [(src_node, dst_node), ...] 候选动作
              - global_feat: (4,) 全局特征
        """
        self.placed_parts = set()          # 已放置零件名
        self.placed_nodes = set()          # 已放置的全局节点索引
        self.step_count = 0
        self.used_features = {}            # name → set of global_node_idx
        self.placed_labels = {pn: [] for pn in self.part_names}  # 已生成的标签
        self.world_transforms = {}         # name → cadquery Location (世界变换)
        self.placement_history = []        # [(part_name, labels), ...]

        # 选择锚点零件（与现有逻辑一致：选连通度最高的）
        anchor = self._pick_anchor()
        self.placed_parts.add(anchor)
        self.placed_nodes.update(
            self.feature_to_global[(anchor, "planar", fi)]
            for fi in range(len(self.part_features[anchor].get("planar", [])))
        )
        self.placed_nodes.update(
            self.feature_to_global[(anchor, "cylinder", fi)]
            for fi in range(len(self.part_features[anchor].get("cylinders", [])))
        )
        self.world_transforms[anchor] = cq.Location(
            cq.Plane(origin=cq.Vector(0, 0, 0),
                     xDir=cq.Vector(1, 0, 0),
                     normal=cq.Vector(0, 0, 1))
        )

        # 初始化 used_features（锚点特征为"已使用"但可多次用于配合）
        self.used_features[anchor] = set(self.placed_nodes)

        return self._get_observation()

    def _pick_anchor(self):
        """选择锚点零件：连通度最高者"""
        if len(self.part_names) == 1:
            return self.part_names[0]
        # 简化策略：选特征最多的零件
        best = self.part_names[0]
        best_n = 0
        for pn in self.part_names:
            feats = self.part_features[pn]
            n = len(feats.get("planar", [])) + len(feats.get("cylinders", []))
            if n > best_n:
                best_n = n
                best = pn
        return best

    # ========== 观测构建 ==========

    def _get_observation(self):
        """
        构建当前状态的观测。

        Returns:
            dict: 包含图结构数据、候选动作和全局特征
        """
        # 节点特征矩阵
        node_features = self._build_node_features()

        # 边连接（同零件内全连接）
        edge_index, edge_attr = self._build_edges()

        # 节点 mask（1=已放置, 0=未放置, 作为额外特征通道）
        mask = self._build_mask()

        # 候选动作
        candidates = self._get_candidates()

        # 全局特征
        global_feat = self._build_global_features()

        return {
            "node_features": node_features,      # (N_nodes, NODE_FEAT_DIM)
            "edge_index": edge_index,            # (2, N_edges)
            "edge_attr": edge_attr,              # (N_edges, 3)
            "mask": mask,                        # (N_nodes,)
            "candidates": candidates,            # [(src_node_idx, dst_node_idx), ...]
            "global_feat": global_feat,          # (4,)
        }

    def _build_node_features(self):
        """
        为每个特征节点提取固定维度特征向量。

        特征维度 (NODE_FEAT_DIM=11):
          [0]:  类型 (0=planar_face, 1=cylinder)
          [1]:  归一化面积 (planar) 或归一化半径 (cylinder)
          [2-4]: 中心坐标 [x, y, z]（除以包围盒对角线归一化）
          [5-7]: 法向/轴方向 [x, y, z]（已归一化）
          [8]:  特征计数（圆+线段数）
          [9]:  is_ext (0=孔, 1=轴, -1=非圆柱)
          [10]: 所在零件索引（归一化）
        """
        import numpy as np

        # 全局归一化因子
        all_bboxes = list(self.part_bboxes.values())
        global_diag = max(
            math.sqrt((bb[1]-bb[0])**2 + (bb[3]-bb[2])**2 + (bb[5]-bb[4])**2)
            for bb in all_bboxes
        )
        if global_diag < 1e-6:
            global_diag = 1.0

        nf = np.zeros((self.n_nodes, NODE_FEAT_DIM), dtype=np.float32)

        for gi, (pi, ftype, fi) in enumerate(self.feature_nodes):
            pn = self.part_names[pi]
            feats = self.part_features[pn]

            if ftype == "planar":
                f = feats["planar"][fi]
                nf[gi, 0] = 0.0  # planar
                area = f.get("area", 0)
                nf[gi, 1] = math.log1p(area) / 10.0  # log 压缩
                nf[gi, 2] = f["c"][0] / global_diag
                nf[gi, 3] = f["c"][1] / global_diag
                nf[gi, 4] = f["c"][2] / global_diag
                nf[gi, 5] = f["n"][0]
                nf[gi, 6] = f["n"][1]
                nf[gi, 7] = f["n"][2]
                nf[gi, 8] = self._safe_log_count(len(f.get("circles", [])) + len(f.get("lines", [])))
                nf[gi, 9] = -1.0  # N/A for planar

            elif ftype == "cylinder":
                f = feats["cylinders"][fi]
                nf[gi, 0] = 1.0  # cylinder
                r = f.get("r", 0)
                nf[gi, 1] = math.log1p(r) / 5.0  # log 压缩半径
                nf[gi, 2] = f["mid"][0] / global_diag
                nf[gi, 3] = f["mid"][1] / global_diag
                nf[gi, 4] = f["mid"][2] / global_diag
                nf[gi, 5] = f["dir"][0]
                nf[gi, 6] = f["dir"][1]
                nf[gi, 7] = f["dir"][2]
                nf[gi, 8] = len(f.get("ends", []))
                nf[gi, 9] = 1.0 if f.get("ext", False) else 0.0

            nf[gi, 10] = pi / max(self.n_parts - 1, 1)  # 零件索引归一化

        return nf

    @staticmethod
    def _safe_log_count(n):
        return math.log1p(n) / 3.0  # 0→0, 3→0.46, 10→0.80

    def _build_edges(self):
        """构建同零件内的边连接（全连接图）"""
        import numpy as np

        # 为每个零件收集节点
        part_nodes = {}
        for gi, (pi, ftype, fi) in enumerate(self.feature_nodes):
            part_nodes.setdefault(pi, []).append(gi)

        edges = []
        edge_feats = []
        z = np.array([0.0, 0.0, 1.0])

        for pi, node_indices in part_nodes.items():
            pn = self.part_names[pi]
            feats = self.part_features[pn]
            for i in range(len(node_indices)):
                for j in range(i + 1, len(node_indices)):
                    gi, gj = node_indices[i], node_indices[j]
                    edges.append([gi, gj])
                    edges.append([gj, gi])  # 无向图 → 双向

                    # 边特征：距离 + 法向夹角余弦
                    # feature_nodes[gi] = (part_idx, feat_type, feat_idx)
                    pi_i, ftype_i, fii = self.feature_nodes[gi]
                    pi_j, ftype_j, fij = self.feature_nodes[gj]
                    ci = self._get_feature_center(ftype_i, pi_i, fii)
                    cj = self._get_feature_center(ftype_j, pi_j, fij)
                    ni = self._get_feature_normal(ftype_i, pi_i, fii)
                    nj = self._get_feature_normal(ftype_j, pi_j, fij)

                    dist = math.sqrt(sum((ci[k] - cj[k])**2 for k in range(3)))
                    # 法向点积
                    dot = ni[0]*nj[0] + ni[1]*nj[1] + ni[2]*nj[2]

                    edge_feats.append([dist / 100.0, dot, 1.0])  # 归一化距离
                    edge_feats.append([dist / 100.0, dot, 1.0])

        if edges:
            return (
                np.array(edges, dtype=np.int64).T,
                np.array(edge_feats, dtype=np.float32),
            )
        else:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float32),
            )

    def _get_feature_center(self, feat_type, pi, fi):
        pn = self.part_names[pi]
        feats = self.part_features[pn]
        if feat_type == "planar":
            return feats["planar"][fi]["c"]
        else:
            return feats["cylinders"][fi]["mid"]

    def _get_feature_normal(self, feat_type, pi, fi):
        pn = self.part_names[pi]
        feats = self.part_features[pn]
        if feat_type == "planar":
            return feats["planar"][fi]["n"]
        else:
            return feats["cylinders"][fi]["dir"]

    def _get_feature(self, feat_type, pi, fi):
        pn = self.part_names[pi]
        feats = self.part_features[pn]
        if feat_type == "planar":
            return feats["planar"][fi]
        else:
            return feats["cylinders"][fi]

    def _build_mask(self):
        """节点 mask: 1=已放置, 0=未放置"""
        import numpy as np
        mask = np.zeros(self.n_nodes, dtype=np.float32)
        for gi, (pi, ftype, fi) in enumerate(self.feature_nodes):
            pn = self.part_names[pi]
            if pn in self.placed_parts:
                mask[gi] = 1.0
        return mask

    def _build_global_features(self):
        """全局特征：[n_placed, n_remaining, step_count/max_steps, avg_placed_node_degree]"""
        import numpy as np
        n_placed = len(self.placed_parts)
        n_remaining = self.n_parts - n_placed
        step_ratio = self.step_count / max(self.max_steps, 1)
        # 已放置节点的平均度
        deg = 0.0
        if self.n_nodes > 0:
            mask = self._build_mask()
            deg = float(mask.sum()) / max(self.n_nodes, 1)
        return np.array([n_placed, n_remaining, step_ratio, deg], dtype=np.float32)

    # ========== 候选动作生成（几何预过滤）==========

    def _get_candidates(self):
        """
        预过滤：找到几何上兼容的特征面对。

        复用 feature_matcher 的核心逻辑：
          - _match_list(圆的 len) → 匹配圆
          - _match_list(线的 len) → 匹配线
          - match_cylinders(半径) → 匹配圆柱

        候选格式: [(src_node_idx, dst_node_idx), ...]
          src_node: 已放置零件中的特征节点
          dst_node: 未放置零件中的特征节点
        """
        candidates = []

        for src_gi, (spi, stype, sfi) in enumerate(self.feature_nodes):
            src_pn = self.part_names[spi]
            if src_pn not in self.placed_parts:
                continue
            src_feat = self._get_feature(stype, spi, sfi)

            for dst_gi, (dpi, dtype, dfi) in enumerate(self.feature_nodes):
                dst_pn = self.part_names[dpi]
                if dst_pn in self.placed_parts:
                    continue
                dst_feat = self._get_feature(dtype, dpi, dfi)

                # ---- 同类型匹配 ----
                if stype == "planar" and dtype == "planar":
                    if self._geometrically_compatible_planar(src_feat, dst_feat):
                        candidates.append((src_gi, dst_gi))

                elif stype == "cylinder" and dtype == "cylinder":
                    if self._geometrically_compatible_cylinder(src_feat, dst_feat):
                        candidates.append((src_gi, dst_gi))

        # 候选上限：防止框架嵌入等场景下候选数量爆炸（>60000）
        # 按源节点的"特征丰富度"排序，保留最有潜力的候选
        MAX_CANDIDATES = 500
        if len(candidates) > MAX_CANDIDATES:
            # 用简单启发式排序：匹配特征数多的优先
            def _candidate_score(c):
                src_gi, dst_gi = c
                _, st, sf = self.feature_nodes[src_gi]
                _, dt, df = self.feature_nodes[dst_gi]
                src_f = self._get_feature(st, self.feature_nodes[src_gi][0], sf)
                dst_f = self._get_feature(dt, self.feature_nodes[dst_gi][0], df)
                if st == "planar":
                    return len(src_f.get("circles", [])) + len(src_f.get("lines", []))
                else:
                    return 1.0 / max(abs(src_f.get("r", 1) - dst_f.get("r", 1)), 0.01)
            candidates.sort(key=_candidate_score, reverse=True)
            candidates = candidates[:MAX_CANDIDATES]

        return candidates

    def _geometrically_compatible_planar(self, fa, fb):
        """
        两个平面特征是否几何兼容。

        检查：匹配的圆/线段数 ≥ MIN_MATCHED_FEATURES
        """
        mc = self._match_list_simple(
            fa.get("circles", []), fb.get("circles", []), "len", MATCH_TOL
        )
        ml = self._match_list_simple(
            fa.get("lines", []), fb.get("lines", []), "len", MATCH_TOL
        )
        return len(mc) + len(ml) >= MIN_MATCHED_FEATURES

    def _geometrically_compatible_cylinder(self, ca, cb):
        """
        两个圆柱特征是否几何兼容。

        检查：半径相近，且非同类型（不轴配轴、不孔配孔，除非是螺栓对齐的孔-孔）
        """
        if abs(ca["r"] - cb["r"]) > CYL_RADIUS_TOL:
            return False
        # 排除轴-轴（两个 ext 圆柱不需要互配）
        if ca.get("ext") and cb.get("ext"):
            return False
        return True

    @staticmethod
    def _match_list_simple(la, lb, key, tol):
        """简化版匹配列表（贪心）"""
        matched = []
        used = [False] * len(lb)
        for a in la:
            for j, b in enumerate(lb):
                if used[j]:
                    continue
                if abs(a[key] - b[key]) < tol:
                    matched.append((a, b))
                    used[j] = True
                    break
        return matched

    # ========== 环境步进 ==========

    def step(self, action: int):
        """
        执行一步装配。

        Args:
            action: 候选动作索引 (0 ~ len(candidates)-1)

        Returns:
            obs:   下一步观测 (dict, 同 reset)
            reward: float 奖励值
            done:   bool 是否结束
            info:   dict 额外信息
        """
        candidates = self._get_candidates()

        # 无效动作 → 负奖励
        if not candidates or action < 0 or action >= len(candidates):
            obs = self._get_observation()
            done = len(self.placed_parts) >= self.n_parts or self.step_count >= self.max_steps
            return obs, -10.0, done, {"error": "invalid action", "placed": list(self.placed_parts)}

        src_gi, dst_gi = candidates[action]
        _, stype, sfi = self.feature_nodes[src_gi]
        dpi, dtype, dfi = self.feature_nodes[dst_gi]

        src_pn = self.part_names[self.feature_nodes[src_gi][0]]
        dst_pn = self.part_names[dpi]
        src_feat = self._get_feature(stype, self.feature_nodes[src_gi][0], sfi)
        dst_feat = self._get_feature(dtype, dpi, dfi)

        # ---------- 1. 执行装配 ----------
        if stype == "planar" and dtype == "planar":
            success, match_info = self._mate_planar(src_feat, dst_feat, src_pn, dst_pn)
        elif stype == "cylinder" and dtype == "cylinder":
            success, match_info = self._mate_cylinder(src_feat, dst_feat, src_pn, dst_pn)
        else:
            success, match_info = False, None

        if not success:
            obs = self._get_observation()
            self.step_count += 1
            done = len(self.placed_parts) >= self.n_parts or self.step_count >= self.max_steps
            return obs, -5.0, done, {"error": "mate failed"}

        # 标记已放置
        self.placed_parts.add(dst_pn)
        dst_nodes = {
            self.feature_to_global[(dst_pn, "planar", fi)]
            for fi in range(len(self.part_features[dst_pn].get("planar", [])))
        } | {
            self.feature_to_global[(dst_pn, "cylinder", fi)]
            for fi in range(len(self.part_features[dst_pn].get("cylinders", [])))
        }
        self.placed_nodes.update(dst_nodes)
        self.used_features.setdefault(dst_pn, set()).add(dst_gi)
        self.used_features.setdefault(src_pn, set()).add(src_gi)
        self.step_count += 1

        # ---------- 2. 更新世界变换（简化：直接用标签坐标系） ----------
        # 用 label_generator 生成坐标系
        # 检查是否有 face_info 来做投影
        self._update_world_transform(src_pn, dst_pn, src_feat, dst_feat, stype, dtype)

        # ---------- 3. 计算奖励 ----------
        reward_dict = self._compute_rewards(
            src_feat, dst_feat, match_info, src_pn, dst_pn
        )

        # ---------- 4. 检查终止 ----------
        done = len(self.placed_parts) >= self.n_parts or self.step_count >= self.max_steps
        if done and len(self.placed_parts) >= self.n_parts:
            # Episode 完成 → 添加完整性和约束奖励
            completeness = compute_total_reward(
                completeness_info=(len(self.placed_parts), self.n_parts, self.step_count),
                constraint_info=([(pn, self.placed_labels[pn], self.part_bboxes[pn])
                                  for pn in self.placed_parts]),
            )
            reward_dict["total"] += completeness["completeness"] + completeness["constraint"]

        # ---------- 5. 构建下一观测 ----------
        obs = self._get_observation()

        info = {
            "placed": list(self.placed_parts),
            "reward_breakdown": reward_dict,
            "step": self.step_count,
            "src_part": src_pn,
            "dst_part": dst_pn,
        }

        return obs, reward_dict["total"], done, info

    def _mate_planar(self, fa, fb, src_pn, dst_pn):
        """
        两个平面配合。返回 (success, match_info)。
        """
        mc = self._match_list_simple(
            fa.get("circles", []), fb.get("circles", []), "len", MATCH_TOL
        )
        ml = self._match_list_simple(
            fa.get("lines", []), fb.get("lines", []), "len", MATCH_TOL
        )
        # 生成标签（使用 label_generator 的逻辑）
        import copy
        m = {"fa": copy.deepcopy(fa), "fb": copy.deepcopy(fb),
             "mc": mc, "ml": ml, "t": len(mc) + len(ml)}
        try:
            idx = len(self.placed_labels.get(src_pn, [])) + 1
            la, lb = planar_labels(m, src_pn, dst_pn, idx)
            self.placed_labels.setdefault(src_pn, []).append(la)
            self.placed_labels.setdefault(dst_pn, []).append(lb)
            return True, (fa, fb, mc, ml)
        except Exception:
            return False, None

    def _mate_cylinder(self, ca, cb, src_pn, dst_pn):
        """
        两个圆柱配合。返回 (success, match_info)。
        """
        import copy
        shaft_in_a = ca.get("ext", False)
        m = {"shaft": ca if shaft_in_a else cb,
             "bore": cb if shaft_in_a else ca,
             "shaft_in_a": shaft_in_a,
             "bore_to_bore": not ca.get("ext") and not cb.get("ext")}
        try:
            idx = len(self.placed_labels.get(src_pn, [])) + 1
            la, lb = cylinder_labels(m, src_pn, dst_pn, idx)
            self.placed_labels.setdefault(src_pn, []).append(la)
            self.placed_labels.setdefault(dst_pn, []).append(lb)
            return True, (ca, cb)
        except Exception:
            return False, None

    def _update_world_transform(self, src_pn, dst_pn, src_feat, dst_feat, stype, dtype):
        """
        更新 dst_pn 的世界变换。

        规则：world[dst] = world[src] * build_loc(src_label) * build_loc(dst_label)⁻¹
        """
        if src_pn not in self.world_transforms:
            return

        # 获取最近添加的标签对
        src_labels = self.placed_labels.get(src_pn, [])
        dst_labels = self.placed_labels.get(dst_pn, [])
        if not src_labels or not dst_labels:
            return

        src_lbl = src_labels[-1]
        dst_lbl = dst_labels[-1]

        src_loc = self._build_location(src_lbl["geometry"])
        dst_loc = self._build_location(dst_lbl["geometry"])

        try:
            self.world_transforms[dst_pn] = (
                self.world_transforms[src_pn] * src_loc * dst_loc.inverse
            )
        except Exception:
            pass

    @staticmethod
    def _build_location(geo):
        """从标签 geometry 构建 cadquery Location"""
        o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
        x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
        z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
        return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))

    def _get_world_bbox(self, pn):
        """获取零件在世界坐标下的包围盒"""
        if pn not in self.world_transforms:
            return self.part_bboxes[pn]

        bb = self.part_shapes[pn].BoundingBox()
        loc = self.world_transforms[pn]

        # 变换 8 个角点
        corners = [
            (bb.xmin, bb.ymin, bb.zmin), (bb.xmin, bb.ymin, bb.zmax),
            (bb.xmin, bb.ymax, bb.zmin), (bb.xmin, bb.ymax, bb.zmax),
            (bb.xmax, bb.ymin, bb.zmin), (bb.xmax, bb.ymin, bb.zmax),
            (bb.xmax, bb.ymax, bb.zmin), (bb.xmax, bb.ymax, bb.zmax),
        ]
        import cadquery as cq
        xformed = []
        for c in corners:
            v = cq.Vector(*c)
            try:
                tv = loc.wrapped.Transformation() * v.wrapped
                xformed.append((tv.X(), tv.Y(), tv.Z()))
            except Exception:
                xformed.append(c)

        return (
            min(p[0] for p in xformed), max(p[0] for p in xformed),
            min(p[1] for p in xformed), max(p[1] for p in xformed),
            min(p[2] for p in xformed), max(p[2] for p in xformed),
        )

    def _compute_rewards(self, src_feat, dst_feat, match_info, src_pn, dst_pn):
        """
        计算当前步的奖励（Layer 1 + Layer 2）。
        """
        # Layer 1: 特征匹配质量
        if match_info and len(match_info) == 4:
            fa, fb, mc, ml = match_info
            match_reward = compute_total_reward(match_info=(fa, fb, mc, ml))
        else:
            match_reward = {"match": 0.0, "total": 0.0}

        # Layer 2: 物理合理性
        try:
            new_bb = self._get_world_bbox(dst_pn)
            placed_bbs = [self._get_world_bbox(p) for p in self.placed_parts
                          if p != dst_pn and p in self.world_transforms]

            # 世界坐标下的面心和法向
            src_center = src_feat.get("c", src_feat.get("mid", [0, 0, 0]))
            dst_center = dst_feat.get("c", dst_feat.get("mid", [0, 0, 0]))
            src_normal = src_feat.get("n", src_feat.get("dir", [0, 0, 1]))
            dst_normal = dst_feat.get("n", dst_feat.get("dir", [0, 0, 1]))

            physics_reward = compute_total_reward(physics_info=(
                new_bb, placed_bbs, src_center, dst_center, src_normal, dst_normal
            ))
        except Exception:
            physics_reward = {"physics": 0.0, "total": 0.0}

        return {
            "match": match_reward.get("match", 0.0),
            "physics": physics_reward.get("physics", 0.0),
            "total": match_reward.get("total", 0.0) + physics_reward.get("total", 0.0),
        }

    @property
    def num_candidates(self):
        """当前候选动作数量（供 Agent 获取动作空间大小）"""
        return len(self._get_candidates())

    def save_labels(self):
        """
        将 RL 生成的标签保存为 *_RL_label.json 文件。
        格式与现有 label_generator.fmt_json 一致，可直接被 verify_assembly 使用。
        """
        from label_generator import fmt_json
        import json as _json

        saved = []
        for pn in self.part_names:
            labels = self.placed_labels.get(pn, [])
            if not labels:
                continue
            out_path = os.path.join(self.folder_path, f"{pn}_RL_label.json")
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(fmt_json(labels), f, indent=2)
            saved.append(out_path)
        return saved

    def export_step(self):
        """
        使用 verify_assembly.py 的逻辑装配并导出 GLB + STEP。
        输出文件: virtual_assembly_RL.glb, virtual_assembly_RL.step
        """
        from label_generator import fmt_json
        import json as _json

        # 先确保标签已保存
        self.save_labels()

        # 调用 verify_assembly 的装配逻辑
        # 直接内联 BFS 装配（避免子进程调用）
        names = [pn for pn in self.part_names
                 if os.path.exists(os.path.join(self.folder_path, f"{pn}_RL_label.json"))]
        if len(names) < 2:
            print("  [warn] 零件不足，无法导出装配体")
            return None, None

        # 加载标签和 shape
        parts = {}
        for nm in names:
            jp = os.path.join(self.folder_path, f"{nm}_RL_label.json")
            systs = _json.load(open(jp, encoding="utf-8"))["modelAnnotation"]["features"]["featureCoordSyses"]
            parts[nm] = {
                "shape": self.part_shapes.get(nm),
                "labels": systs
            }

        # BFS 装配（复用 verify_assembly 逻辑）
        def _build_loc(geo):
            o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
            x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
            z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
            return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))

        groups = {}
        for nm in names:
            for lbl in parts[nm]["labels"]:
                gid = lbl["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in lbl["identifier"] else lbl["identifier"]
                groups.setdefault(gid, []).append((nm, lbl))

        # 锚点
        target = cq.Location(cq.Plane(origin=cq.Vector(0,0,0), xDir=cq.Vector(1,0,0), normal=cq.Vector(0,0,1)))
        world = {}
        anchor = names[0]
        world[anchor] = target * _build_loc(parts[anchor]["labels"][0]["geometry"]).inverse
        placed = {anchor}

        # BFS
        while len(placed) < len(parts):
            progress = False
            for gid, items in groups.items():
                if len(items) != 2:
                    continue
                n1, l1 = items[0]; n2, l2 = items[1]
                if l1.get("userData", {}).get("boreToBore"):
                    continue
                if n1 in placed and n2 not in placed:
                    world[n2] = world[n1] * _build_loc(l1["geometry"]) * _build_loc(l2["geometry"]).inverse
                    placed.add(n2); progress = True
                elif n2 in placed and n1 not in placed:
                    world[n1] = world[n2] * _build_loc(l2["geometry"]) * _build_loc(l1["geometry"]).inverse
                    placed.add(n1); progress = True
            if not progress:
                break

        # 导出
        palette = [
            cq.Color(0.85,0.20,0.20,0.6), cq.Color(0.20,0.65,0.85,0.6),
            cq.Color(0.30,0.75,0.30,0.6), cq.Color(0.90,0.70,0.15,0.6),
        ]
        assembly = cq.Assembly()
        for i, nm in enumerate(names):
            if nm in world and parts[nm]["shape"] is not None:
                assembly.add(parts[nm]["shape"].located(world[nm]),
                           name=nm, color=palette[i % len(palette)])

        out_glb = os.path.join(self.folder_path, "virtual_assembly_RL.glb")
        out_step = os.path.join(self.folder_path, "virtual_assembly_RL.step")
        assembly.save(out_glb)
        assembly.export(out_step, exportType="STEP")

        print(f"  [RL] GLB 已保存: {out_glb}")
        print(f"  [RL] STEP 已保存: {out_step}")
        return out_glb, out_step

    def seed(self, seed):
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
