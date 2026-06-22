"""
label_generator.py
==================
模块3：从匹配结果生成标签 JSON（独立运行）
用法：
  python label_generator.py <folder> [match_file]
"""
import os, sys, json, math
import cadquery as cq
import numpy as np

# 几何容差常量（制造公差，非几何尺度依赖）
DOT_CROSS_PLANE = 0.3      # 法向点积 < 0.3 视为跨平面匹配
MIN_CROSS_PLANE_T = 5      # 跨平面匹配最少特征数
EPS_DENOM = 1e-9            # 除零保护容差
EPS_ZERO_LEN = 1e-12        # 零长向量判定
EPS_X_DEGENERATE = 1e-9     # x 轴退化判定
MIN_CIRCLES_FOR_CENTROID = 3  # 用圆几何中心的最少圆数


def _kabsch_umeyama(pts_a, pts_b):
    """Kabsch-Umeyama 算法：给定 N≥3 对匹配点，求最优刚体变换 (R, t)。

    min ||R·P_b + t - P_a||_F²  →  闭式 SVD 解

    参数:
        pts_a: list of [x,y,z] — 零件A中匹配点的3D坐标
        pts_b: list of [x,y,z] — 零件B中匹配点的3D坐标
    返回:
        R: 3x3 numpy array — 最优旋转矩阵 (det=+1, B→A)
        t: numpy array(3,)   — 最优平移向量
    """
    a = np.array(pts_a, dtype=float)
    b = np.array(pts_b, dtype=float)
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    H = (b - cb).T @ (a - ca)     # 交叉协方差矩阵 H = B^T A
    U, S, Vt = np.linalg.svd(H)   # H = U S Vt
    R = Vt.T @ U.T                 # 最优旋转 R = V U^T = Vt^T U^T
    if np.linalg.det(R) < 0:       # Kabsch 反射修正
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = ca - R @ cb                # 最优平移
    return R, t


def _ortho(z, x_hint=None):
    z = cq.Vector(z[0], z[1], z[2]).normalized()
    if z.Length < EPS_ZERO_LEN:
        raise ValueError(f"_ortho: zero-length normal vector {z}")
    if x_hint:
        ref = cq.Vector(x_hint[0], x_hint[1], x_hint[2])
    else:
        ref = cq.Vector(1, 0, 0) if abs(z.x) < 0.9 else cq.Vector(0, 1, 0)
    x = ref - z * ref.dot(z)
    if x.Length < EPS_X_DEGENERATE:
        # 退化分支：动态选择不平行于 z 的辅助向量，防止 x 与 z 平行导致叉乘崩溃
        ref2 = cq.Vector(0, 1, 0) if abs(z.x) > 0.9 else cq.Vector(1, 0, 0)
        x = (ref2 - z * z.dot(ref2)).normalized()
    else:
        x = x.normalized()
    y = z.cross(x).normalized()
    return {"x": {"x": x.x, "y": x.y, "z": x.z},
            "y": {"x": y.x, "y": y.y, "z": y.z},
            "z": {"x": z.x, "y": z.y, "z": z.z}}


def _neg(v): return [-v[0], -v[1], -v[2]]


def _vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def mk_label(ident, orig, z_dir, extra=None, x_hint=None):
    ud = {"type": "MATE"}
    if extra: ud.update(extra)
    return {"identifier": ident, "name": ident, "label": "ReferenceSys",
            "geometry": {"origin": {"x": orig[0], "y": orig[1], "z": orig[2]},
                         **_ortho(z_dir, x_hint)},
            "userData": ud}


def planar_labels(m, na, nb, idx, slot_x_a=None, slot_x_b=None):
    fa, fb = m["fa"], m["fb"]
    mc = m["mc"]
    meta = {"matchType": "PLANAR", "total": m["t"]}

    # 原点：≥3个匹配圆时用SVD最优质心（或圆几何中心），否则用面心
    if len(mc) >= MIN_CIRCLES_FOR_CENTROID:
        pts_a = [[c[0]["c"][k] for k in range(3)] for c in mc]
        pts_b = [[c[1]["c"][k] for k in range(3)] for c in mc]
        ca = np.array(pts_a, dtype=float).mean(axis=0)
        cb = np.array(pts_b, dtype=float).mean(axis=0)
        oa = [float(ca[0]), float(ca[1]), float(ca[2])]
        ob_ = [float(cb[0]), float(cb[1]), float(cb[2])]
    else:
        oa, ob_ = fa["c"], fb["c"]

    # 跨平面匹配：法向近垂直且特征匹配强 → 用面几何中心
    dot_n = fa["n"][0]*fb["n"][0] + fa["n"][1]*fb["n"][1] + fa["n"][2]*fb["n"][2]
    if abs(dot_n) < DOT_CROSS_PLANE and m["t"] >= MIN_CROSS_PLANE_T:
        oa, ob_ = fa["c"], fb["c"]
        meta["crossPlane"] = True

    xa = xb = None
    # 优先用槽方向（来自slot检测）
    if slot_x_a:
        xa = slot_x_a
    if slot_x_b:
        xb = slot_x_b

    # SVD最优x轴：匹配圆≥3时用左奇异向量（点集最大方差方向）投影到面平面
    if not xa and not xb and len(mc) >= MIN_CIRCLES_FOR_CENTROID:
        pts_a_arr = np.array([[c[0]["c"][k] for k in range(3)] for c in mc], dtype=float)
        pts_b_arr = np.array([[c[1]["c"][k] for k in range(3)] for c in mc], dtype=float)
        ca = pts_a_arr.mean(axis=0)
        cb = pts_b_arr.mean(axis=0)

        # A面：SVD右奇异向量Vh[0,:]（3D空间主方向）→ 面内投影 → x轴
        Ua, Sa, Vha = np.linalg.svd(pts_a_arr - ca)
        # 检查是否退化（圆形阵列 → 奇异值均匀 → 方向任意，跳过SVD用回退）
        if len(Sa) >= 2 and Sa[0] > Sa[1] * 1.2:  # 非圆形 → SVD方向有意义
            x_svd_a = cq.Vector(float(Vha[0, 0]), float(Vha[0, 1]), float(Vha[0, 2]))
            fn_a = cq.Vector(*fa["n"])
            xa_proj = x_svd_a - fn_a * x_svd_a.dot(fn_a)
            if xa_proj.Length > 0.01:
                xa = [xa_proj.x, xa_proj.y, xa_proj.z]

        # B面同理：Vhb[0,:] 为3D空间主方向
        Ub, Sb, Vhb = np.linalg.svd(pts_b_arr - cb)
        if len(Sb) >= 2 and Sb[0] > Sb[1] * 1.2:
            x_svd_b = cq.Vector(float(Vhb[0, 0]), float(Vhb[0, 1]), float(Vhb[0, 2]))
            fn_b = cq.Vector(*fb["n"])
            xb_proj = x_svd_b - fn_b * x_svd_b.dot(fn_b)
            if xb_proj.Length > 0.01:
                xb = [xb_proj.x, xb_proj.y, xb_proj.z]

    # 回退：用第一个匹配圆/线段方向作为x轴
    if not xa and not xb:
        if mc and (len(mc) >= len(m.get("ml", [])) or not m.get("ml")):
            xa = _vec_sub(mc[0][0]["c"], oa)
            xb = _vec_sub(mc[0][1]["c"], ob_)
        elif m["ml"]:
            xa = _vec_sub(m["ml"][0][0]["m"], oa)
            xb = _vec_sub(m["ml"][0][1]["m"], ob_)
    elif not xa:
        if mc and (len(mc) >= len(m.get("ml", [])) or not m.get("ml")):
            xa = _vec_sub(mc[0][0]["c"], oa)
        elif m["ml"]:
            xa = _vec_sub(m["ml"][0][0]["m"], oa)
    elif not xb:
        if mc and (len(mc) >= len(m.get("ml", [])) or not m.get("ml")):
            xb = _vec_sub(mc[0][1]["c"], ob_)
        elif m["ml"]:
            xb = _vec_sub(m["ml"][0][1]["m"], ob_)
    # 面法向关系：dot>0=键/槽嵌入（同半球）；否则面贴面（B取反）
    # 标签忠实反映零件自身LCS；法向不平行由装配变换自然处理
    dot_n = fa["n"][0]*fb["n"][0] + fa["n"][1]*fb["n"][1] + fa["n"][2]*fb["n"][2]
    nb_z = fb["n"] if dot_n > 0 else _neg(fb["n"])
    if dot_n > 0:
        meta["keywayFit"] = True
    return (mk_label(f"{na}_Mating_{idx}", oa, fa["n"], meta, xa),
            mk_label(f"{nb}_Mating_{idx}", ob_, nb_z, meta, xb))


def _bore_face_intersection(bore, face):
    d = cq.Vector(bore["dir"][0], bore["dir"][1], bore["dir"][2])
    n = cq.Vector(face["n"][0], face["n"][1], face["n"][2])
    c = cq.Vector(face["c"][0], face["c"][1], face["c"][2])
    m = cq.Vector(bore["mid"][0], bore["mid"][1], bore["mid"][2])
    denom = d.dot(n)
    if abs(denom) < EPS_DENOM: return bore["mid"]
    t = (c.sub(m)).dot(n) / denom
    pt = m.add(d.multiply(t))
    return [pt.x, pt.y, pt.z]


def cylinder_labels(m, na, nb, idx, bore_origin=None, shaft_x=None, bore_x=None, shaft_origin=None):
    s, b = m["shaft"], m["bore"]
    bore_to_bore = m.get("bore_to_bore", False)
    meta = {"matchType": "CYLINDER", "radius": s["r"]}
    if bore_to_bore:
        meta["boreToBore"] = True
    if m.get("interference"):
        meta["interference"] = True
    if m.get("clearance", 0) > 0.001:
        meta["clearance"] = round(m["clearance"], 4)
    # 旋转件定向标记：有键槽/槽口约束时不盲旋90°试探
    if shaft_x is not None or bore_x is not None:
        meta["directionFixed"] = True
    o_shaft = shaft_origin if shaft_origin else s["mid"]
    o_bore = bore_origin if bore_origin else b["mid"]
    shaft_nm = na if m["shaft_in_a"] else nb
    bore_nm = nb if m["shaft_in_a"] else na
    la = mk_label(f"{shaft_nm}_Mating_{idx}", o_shaft, s["dir"], meta, shaft_x)
    lb = mk_label(f"{bore_nm}_Mating_{idx}", o_bore, b["dir"], meta, bore_x)
    out = {shaft_nm: la, bore_nm: lb}
    return out[na], out[nb]


def fmt_json(lst, world_step=None):
    result = {"modelAnnotation": {"parameters": {},
        "features": {"featurePoints": [], "featureLines": [], "featurePlanes": [],
                     "featureCoordSyses": lst, "featureSurfaces": [], "featureBodies": []},
        "children": []}}
    if world_step:
        result["modelAnnotation"]["worldStep"] = world_step
    return result
