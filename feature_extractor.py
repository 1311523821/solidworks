"""
feature_extractor.py
====================
模块1：从 STEP 文件中提取结构化特征（独立运行，结果可缓存）
用法：
  python feature_extractor.py <folder>     # 批量提取
  python feature_extractor.py <file.step>  # 单文件提取
"""
import os, sys, json, math
import cadquery as cq
import numpy as np
from cadquery import importers
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Vec

# ===== 无量纲全局常数 =====
# 所有几何阈值表达为零件特征尺寸 L（包围盒对角线长度）的无量纲比率，
# 确保算法自动适应微米级到米级零件。仅有的"可调参数"基于物理含义，非魔术数字。

# --- 面过滤 ---
ALPHA_FACE_AREA = 0.0005        # 面最小面积 = α·L²（L=100mm→5mm²）

# --- 圆柱过滤 ---
BETA_CYL_RADIUS = 0.005         # 最小圆柱半径 = β·L

# --- 匹配容差 ---
GAMMA_CYL_MATCH = 0.005         # 圆柱半径匹配容差 = γ·L（L=100mm→0.5mm）
GAMMA_LINE_MATCH = 0.001        # 线段长度匹配容差（L=100mm→0.1mm）
GAMMA_DIST_NEAR = 0.03          # 近邻距离阈值 = γ_near·L（L=100mm→3mm）
GAMMA_DIST_FAR = 0.05           # 较远距离阈值 = γ_far·L（L=100mm→5mm）

# --- 矩形检测 ---
GAMMA_RECT_PAIR_TOL = 0.2       # 矩形边长对容差比例
GAMMA_RECT_ABS_FLOOR = 0.02     # 矩形边长绝对容差下限 = γ_rect·L

# --- 槽口检测 ---
ETA_SLOT_ASPECT = 3.0           # 槽口长宽比（纯比例，不随尺度变化）
BETA_SLOT_AREA_MIN = 0.0005     # 槽口最小面积
BETA_SLOT_AREA_MAX = 0.01       # 槽口最大面积

# --- 结构尺寸阈值（相对于 L 的比例）---
ETA_SPIGOT_R_MIN = 0.06         # 止口最小半径（L=250mm→15mm）
ETA_SPIGOT_STEP = 0.008         # 止口台阶高度差（L=250mm→2mm）
ETA_FRAME_EDGE_MIN = 0.04       # 框架边最小长度（L=250mm→10mm）
ETA_FRAME_FACE_AREA_MIN = 0.004 # 框架面最小面积（L=250mm→250mm²）

# --- 轴向阈值（无量纲）---
ZETA_AXIAL_ALIGN = 0.7          # 轴向对齐判定（dot > 0.7）
ZETA_AXIAL_END_LO = 0.2         # 端面判定下限
ZETA_AXIAL_END_HI = 0.8         # 端面判定上限

# --- 计数阈值（组合数学，不随尺度变化）---
MIN_CIRCLES_FOR_ARRAY = 3       # 圆周阵列最少圆数
MIN_CYL_FOR_SPLINE = 3          # 花键最少圆柱数
MIN_FACES_FOR_CONCENTRIC = 3    # 同心面检测最少面数
MIN_CIRCLES_FOR_PIN_ARRAY = 4   # 引脚阵列最少圆数
MIN_LINES_FOR_RECT = 4          # 矩形检测最少线段数

# --- 半径分桶（倒排索引，制造标准值，不随尺度变化）---
_STANDARD_RADII = sorted([
    0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0,
    12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 30.0,
    33.0, 36.0, 39.0, 42.0
])
_BUCKET_TOL = 0.5


def quantize_radius(r):
    if r <= 0:
        return "R0"  # 无效半径统一归入 R0 桶
    for std in _STANDARD_RADII:
        if abs(r - std) <= _BUCKET_TOL:
            return f"M{std:.1f}".replace(".0", "")
    return f"R{round(r)}"


def _pt(face):
    es = face.Edges(); return es[0].startPoint() if es else face.Center()


def _characteristic_scale(shape):
    """零件的特征尺寸 L = 包围盒对角线长度（mm）。
    所有几何阈值均表达为 L 的无量纲比率，确保算法适应微米级到米级零件。"""
    bb = shape.val().BoundingBox()
    return ((bb.xmax-bb.xmin)**2 + (bb.ymax-bb.ymin)**2 + (bb.zmax-bb.zmin)**2)**0.5


def _principal_axes(shape):
    """计算零件的质心和主惯性轴（按惯性矩降序排列）。
    利用惯性张量矩阵的特征值分解，得到与零件几何天然绑定的局部坐标系，
    无需任何全局 Z 轴硬编码。

    返回: (centroid, [axis_major, axis_medium, axis_minor])
    其中 axis_major 是惯性矩最大的轴（通常是最长轴方向）。
    """
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    gprops = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.val().wrapped, gprops)

    com = gprops.CentreOfMass()
    centroid = cq.Vector(com.X(), com.Y(), com.Z())

    mat = gprops.MatrixOfInertia()
    I = np.array([
        [mat.Value(1, 1), mat.Value(1, 2), mat.Value(1, 3)],
        [mat.Value(2, 1), mat.Value(2, 2), mat.Value(2, 3)],
        [mat.Value(3, 1), mat.Value(3, 2), mat.Value(3, 3)],
    ], dtype=float)

    eigenvals, eigenvecs = np.linalg.eigh(I)
    order_desc = np.argsort(eigenvals)[::-1]  # 降序：λ₀ ≥ λ₁ ≥ λ₂

    # 用特征值比率检测几何类型，正确选择主轴：
    #   λ₀ ≈ λ₁ ≫ λ₂ → 轴状零件（细长轴）→ 纵向轴 = λ₂（最小惯性矩）
    #   λ₀ ≫ λ₁ ≈ λ₂ → 盘状零件（法兰）  → 法向轴 = λ₀（最大惯性矩）
    #   其他            → 默认取 λ₀（最大惯性矩）
    ev = eigenvals[order_desc]  # [λ_max, λ_mid, λ_min]
    lambda_max, lambda_mid, lambda_min = float(ev[0]), float(ev[1]), float(ev[2])
    ratio_01 = lambda_max / max(lambda_mid, 1e-12)
    ratio_12 = lambda_mid / max(lambda_min, 1e-12)

    if ratio_12 > 2.0 and ratio_01 < 1.5:
        # 轴状：λ_max ≈ λ_mid ≫ λ_min → 主轴 = λ_min（纵向轴）
        primary_idx = order_desc[2]  # 最小惯性矩对应的特征向量
    else:
        # 盘状或其他：主轴 = λ_max（法向轴/最长轴）
        primary_idx = order_desc[0]  # 最大惯性矩对应的特征向量

    # 将主轴放在第一位，其余按降序排列
    remaining = [i for i in order_desc if i != primary_idx]
    final_order = [primary_idx] + remaining

    axes = [cq.Vector(float(eigenvecs[0, i]), float(eigenvecs[1, i]), float(eigenvecs[2, i]))
            for i in final_order]
    return centroid, axes


# ==================== AAG: 属性邻接图 ====================
def _classify_face_type(face):
    """分类 B-Rep 面类型 → 'plane'|'cylinder'|'cone'|'sphere'|'torus'|'spline'|'other'"""
    adaptor = BRepAdaptor_Surface(face.wrapped, True)
    type_map = {
        0: 'plane', 1: 'cylinder', 2: 'cone', 3: 'sphere',
        4: 'torus', 5: 'bezier', 6: 'spline', 7: 'revolution',
        8: 'extrusion', 9: 'offset', 10: 'other'
    }
    return type_map.get(adaptor.GetType(), 'other')


def _classify_edge_convexity(face_a_wrapped, face_b_wrapped, shared_edge_wrapped, centroid):
    """
    参数化无关的边凸凹性分类。

    原理：n_a · u_b 的符号决定凸凹性，其中 u_b = sign * (t_raw × n_b)
    是指向面 B 内部的测地线切向量。用零件质心消除切向量方向二义性。

    返回: 'convex' | 'concave' | 'tangent'
    """
    # 1. 边中点 + 切向量
    adapt = BRepAdaptor_Curve(shared_edge_wrapped)
    mid_param = (adapt.FirstParameter() + adapt.LastParameter()) / 2.0
    pnt = gp_Pnt(); vec = gp_Vec()
    adapt.D1(mid_param, pnt, vec)
    P = cq.Vector(pnt.X(), pnt.Y(), pnt.Z())
    t_raw = cq.Vector(vec.X(), vec.Y(), vec.Z())
    if t_raw.Length < 1e-12:
        return 'tangent'

    # 2. 两面法向
    n_a = cq.Face(face_a_wrapped).normalAt(P)
    n_b = cq.Face(face_b_wrapped).normalAt(P)

    # 3. u_b = t_raw × n_b（面B内垂直于边的切向量）
    u_b = t_raw.cross(n_b)
    if u_b.Length < 1e-9:
        return 'tangent'
    u_b = u_b.normalized()

    # 4. 用质心修正 u_b 方向：指向面内（面B的质心方向）
    #    centroid 是实体的质心 (C - P) 在面B上的投影应指向面内
    to_centroid = centroid - P
    if to_centroid.dot(u_b) < 0:
        u_b = -u_b  # u_b 指向面外 → 取反

    # 5. 凸凹性判定
    dot_val = n_a.dot(u_b)
    if dot_val > 1e-6:
        return 'concave'   # n_a 指向面B内部 → 凹边
    elif dot_val < -1e-6:
        return 'convex'    # n_a 指向面B外部 → 凸边
    else:
        return 'tangent'


def _build_aag(shape, centroid):
    """
    基于 OpenCASCADE TopExp 接口的 O(N) 属性邻接图构建。
    节点包含零件的「所有面」（平面+圆柱+锥面等），确保法兰密封面的圆柱孔邻居能被表达。

    返回: {"nodes": [...], "edges": [...]}
    """
    all_faces = list(shape.faces().vals())
    total_area = sum(f.Area() for f in all_faces)

    # 1. 全量面节点（含非平面）
    nodes = []
    face_map_idx = {}  # CadQuery Face id(f) → AAG node index（Python 对象生命周期稳定）
    for idx, f in enumerate(all_faces):
        surface_type = _classify_face_type(f)

        wire_count = 0
        explorer = TopExp_Explorer(f.wrapped, TopAbs_WIRE)
        while explorer.More():
            wire_count += 1
            explorer.Next()
        n_internal_loops = max(0, wire_count - 1)

        nodes.append({
            "type": surface_type,
            "area_ratio": round(f.Area() / total_area, 4) if total_area > 0 else 0,
            "n_internal_loops": n_internal_loops,
            "area": round(f.Area(), 1),
        })
        face_map_idx[id(f)] = idx  # 稳定的 Python 对象引用

    # 2. TopExp EDGE→FACES 映射（O(N) 线性时间）
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.val().wrapped, TopAbs_EDGE, TopAbs_FACE, edge_map)

    aag_edges = []
    for i in range(1, edge_map.Extent() + 1):
        edge_wrapped = TopoDS.Edge_s(edge_map.FindKey(i))
        face_list = edge_map.FindFromIndex(i)

        # 仅处理两面临接的流形边
        if face_list.Extent() != 2:
            continue

        # 显式下转型：TopExp 返回 TopoDS_Shape，需转为 TopoDS_Face
        fa_wrapped = TopoDS.Face_s(face_list.First())
        fb_wrapped = TopoDS.Face_s(face_list.Last())

        # 通过 IsSame 匹配节点索引（安全避开 id(wrapped) 临时对象陷阱）
        src = dst = None
        for idx, f in enumerate(all_faces):
            if src is None and f.wrapped.IsSame(fa_wrapped):
                src = idx
            if dst is None and f.wrapped.IsSame(fb_wrapped):
                dst = idx
            if src is not None and dst is not None:
                break

        if src is None or dst is None:
            continue

        convexity = 'unknown'; edge_geom = 'unknown'; edge_len = 0.0
        try:
            convexity = _classify_edge_convexity(
                fa_wrapped, fb_wrapped, edge_wrapped, centroid)
            etypes = {0:'line',1:'circle',2:'ellipse',3:'hyperbola',
                      4:'parabola',5:'bezier',6:'spline',7:'offset',8:'other'}
            eadapt = BRepAdaptor_Curve(edge_wrapped)
            edge_geom = etypes.get(eadapt.GetType(), 'unknown')
            from OCP.GCPnts import GCPnts_AbscissaPoint
            edge_len = GCPnts_AbscissaPoint.Length_s(eadapt)
        except Exception:
            pass

        aag_edges.append({
            "src": src, "dst": dst,
            "convexity": convexity,
            "edge_geom": edge_geom,
            "edge_len": round(edge_len, 3),
        })

    return {"nodes": nodes, "edges": aag_edges}


# ==================== planar + 面形描述符 ====================
def _face_shape_descriptors(face_data, L=100.0):
    """计算面的形状描述符：矩形度、长宽比、是否为槽口"""
    lines = face_data.get("lines", [])
    circles = face_data.get("circles", [])
    area = face_data.get("area", 0)
    result = {"aspect_ratio": 1.0, "rect_score": 0.0, "is_slot": False}
    face_scale = max(math.sqrt(area), 1.0)  # 面级特征尺度
    if len(lines) >= MIN_LINES_FOR_RECT:
        lens = sorted([l["len"] for l in lines], reverse=True)
        top4 = lens[:4]
        if min(top4) > 1:
            result["aspect_ratio"] = round(top4[0] / max(top4[2], 1), 2)
            # 矩形度：两对等边差 < 20%（长边对长边、短边对短边）
            tol_rect = max(top4[0] * GAMMA_RECT_PAIR_TOL, face_scale * GAMMA_RECT_ABS_FLOOR)
            pair1_ok = abs(top4[0] - top4[1]) < tol_rect
            pair2_ok = abs(top4[2] - top4[3]) < max(top4[2] * GAMMA_RECT_PAIR_TOL, face_scale * GAMMA_RECT_ABS_FLOOR)
            if pair1_ok and pair2_ok:
                err = abs(top4[0] - top4[1]) / max(top4[0], 1) + abs(top4[2] - top4[3]) / max(top4[2], 1)
                result["rect_score"] = max(0.0, 1.0 - err)
            # 长槽检测：长宽比 > ETA_SLOT_ASPECT 且面积适中的面
            slot_min = BETA_SLOT_AREA_MIN * L * L
            slot_max = BETA_SLOT_AREA_MAX * L * L
            if result["aspect_ratio"] > ETA_SLOT_ASPECT and slot_min < area < slot_max:
                result["is_slot"] = True
    # 孔距不变量：面内同半径圆的圆心距（旋转不变，跨零件比对黄金标准）
    if len(circles) >= 2:
        dists = set()
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dr = abs(circles[i]["r"] - circles[j]["r"])
                if dr > 0.1: continue  # 不同半径的圆不计算间距
                d2 = ((circles[i]["c"][0]-circles[j]["c"][0])**2 +
                      (circles[i]["c"][1]-circles[j]["c"][1])**2 +
                      (circles[i]["c"][2]-circles[j]["c"][2])**2)
                dists.add(round(d2**0.5, 1))
        result["inter_circle_dists"] = sorted(dists)[:20]  # 最多 20 个

    # 引脚阵列候选：≥ MIN_CIRCLES_FOR_PIN_ARRAY 个同半径圆的面
    if len(circles) >= MIN_CIRCLES_FOR_PIN_ARRAY:
        by_r = {}
        for c in circles:
            rk = round(c["r"], 1)
            by_r.setdefault(rk, []).append(c)
        for rk, clist in by_r.items():
            if len(clist) >= MIN_CIRCLES_FOR_PIN_ARRAY:
                if "pin_arrays" not in result:
                    result["pin_arrays"] = []
                result["pin_arrays"].append({
                    "radius": round(rk, 1), "count": len(clist),
                    "bucket": quantize_radius(rk)
                })
    return result


def _sort_by_angle(items, center, normal, key="c"):
    z = cq.Vector(*normal).normalized()
    oc = cq.Vector(*center)
    if len(items) < 2: return items
    best_d = -1; best_pt = None
    for it in items:
        pt = cq.Vector(*it[key]).sub(oc)
        pt_proj = pt - z * pt.dot(z); d = pt_proj.Length
        if d > best_d: best_d = d; best_pt = pt_proj
    x_ref = best_pt.normalized() if best_pt and best_pt.Length > 1e-9 else cq.Vector(1,0,0)
    x_ref = x_ref - z * x_ref.dot(z)
    if x_ref.Length < 1e-9:
        x_ref = cq.Vector(0, 1, 0) - z * z.y
    x_ref = x_ref.normalized()
    y_ref = z.cross(x_ref)
    def _ang(it):
        pt = cq.Vector(*it[key]).sub(oc); pt_proj = pt - z * pt.dot(z)
        return math.atan2(pt_proj.dot(y_ref), pt_proj.dot(x_ref))
    return sorted(items, key=_ang)


def extract_planar(shape, L=None, principal_axis=None):
    if L is None:
        L = _characteristic_scale(shape)
    if principal_axis is None:
        principal_axis = cq.Vector(0, 0, 1)  # 回退到全局 Z
    min_face_area = ALPHA_FACE_AREA * L * L
    out = []
    for f in shape.faces().vals():
        area = f.Area()
        if area < min_face_area: continue
        cs, ls = [], []
        for e in f.Edges():
            if e.geomType() == "CIRCLE":
                l = e.Length()
                cs.append({"len": round(l, 6), "r": round(l / (2 * math.pi), 6),
                           "c": [e.Center().x, e.Center().y, e.Center().z]})
            elif e.geomType() == "LINE":
                ls.append({"len": round(e.Length(), 6),
                           "m": [e.Center().x, e.Center().y, e.Center().z]})
        if not cs and len(ls) < 2: continue
        if cs or ls:
            c = f.Center(); n = f.normalAt(_pt(f))
            if len(cs) > 1: cs = _sort_by_angle(cs, [c.x, c.y, c.z], [n.x, n.y, n.z], "c")
            if len(ls) > 1: ls = _sort_by_angle(ls, [c.x, c.y, c.z], [n.x, n.y, n.z], "m")
            face_data = {
                "c": [c.x, c.y, c.z], "n": [n.x, n.y, n.z],
                "circles": cs, "lines": ls,
                "area": round(area, 1),
                "n_edges": len(cs) + len(ls),
                "z_dominant": abs(n.dot(principal_axis)) > ZETA_AXIAL_ALIGN
            }
            # 附加形状描述符
            face_data.update(_face_shape_descriptors(face_data, L))
            out.append(face_data)
    return out


# ==================== cylinder ====================
def _cyl_geom(face):
    s = BRepAdaptor_Surface(face.wrapped, True); cy = s.Cylinder()
    ax = cy.Axis()
    loc = cq.Vector(ax.Location().X(), ax.Location().Y(), ax.Location().Z())
    d = cq.Vector(ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())
    return loc, d, cy.Radius()


def _is_ext(face):
    loc, d, r = _cyl_geom(face)
    # 注意：cq.Vector(0,0,0) 的布尔值为 False，不能用它判断无效位置
    # _cyl_geom 始终返回有效 Vector，无需 None 检查
    pt = _pt(face); n = face.normalAt(pt)
    v = pt.sub(loc).sub(d.multiply(pt.sub(loc).dot(d)))
    return v.dot(n) > 0


def extract_cylinders(shape, L=None, principal_axis=None):
    if L is None:
        L = _characteristic_scale(shape)
    bb = shape.val().BoundingBox()
    min_r = BETA_CYL_RADIUS * L
    out = []
    for f in shape.faces("%Cylinder").vals():
        try:
            loc, d, r = _cyl_geom(f)
            if r < min_r: continue
            ext = _is_ext(f)
            c = f.Center(); mid = loc.add(d.multiply(c.sub(loc).dot(d)))
            ends = [[e.Center().x, e.Center().y, e.Center().z]
                    for e in f.Edges() if e.geomType() == "CIRCLE"]
            # 轴向归一化位置：0=底面端, 1=顶面端（沿主惯性轴方向）
            if principal_axis is not None and abs(d.dot(principal_axis)) > ZETA_AXIAL_ALIGN:
                # 计算包围盒沿主轴方向的投影范围
                corners = [
                    cq.Vector(bb.xmin, bb.ymin, bb.zmin),
                    cq.Vector(bb.xmax, bb.ymin, bb.zmin),
                    cq.Vector(bb.xmin, bb.ymax, bb.zmin),
                    cq.Vector(bb.xmin, bb.ymin, bb.zmax),
                    cq.Vector(bb.xmax, bb.ymax, bb.zmin),
                    cq.Vector(bb.xmax, bb.ymin, bb.zmax),
                    cq.Vector(bb.xmin, bb.ymax, bb.zmax),
                    cq.Vector(bb.xmax, bb.ymax, bb.zmax),
                ]
                projs = [cv.dot(principal_axis) for cv in corners]
                pa_min, pa_max = min(projs), max(projs)
                pa_len = pa_max - pa_min
                if pa_len > 0:
                    mid_proj = mid.dot(principal_axis)
                    axial_pos = (mid_proj - pa_min) / pa_len
                else:
                    axial_pos = -1
            else:
                axial_pos = -1  # 非主轴向圆柱不计算
            out.append({
                "r": round(r, 4), "ext": ext,
                "mid": [mid.x, mid.y, mid.z],
                "dir": [d.x, d.y, d.z],
                "ends": ends,
                "bucket": quantize_radius(r),
                "axial_pos": round(axial_pos, 3)
            })
            # 锥度检测：两端圆半径差 / 轴线长度
            if len(ends) >= 2:
                end_rs = []
                for ep in ends:
                    ev = cq.Vector(ep[0]-mid.x, ep[1]-mid.y, ep[2]-mid.z)
                    axial = abs(ev.dot(d))
                    radial = (ev - d * ev.dot(d)).Length
                    end_rs.append((radial, axial))
                end_rs.sort(key=lambda x: x[1])
                if len(end_rs) >= 2 and abs(end_rs[0][1] - end_rs[-1][1]) > 1:
                    r1 = end_rs[0][0]; r2 = end_rs[-1][0]
                    length = abs(end_rs[-1][1] - end_rs[0][1])
                    if length > 0 and abs(r1 - r2) > 0.01:
                        out[-1]["taper"] = round(abs(r1 - r2) / length, 6)
                        out[-1]["is_tapered"] = True
        except Exception as e:
            import sys
            print(f"  [warn] cyl extraction failed for a face: {e}", file=sys.stderr)
    out.sort(key=lambda c: (
        c["ext"], c["r"],
        round(c["dir"][0], 2), round(c["dir"][1], 2), round(c["dir"][2], 2),
        round(c["mid"][0], 4), round(c["mid"][1], 4), round(c["mid"][2], 4)))
    return out


# ==================== fingerprint ====================
def _compute_fingerprint(planar, cylinders, L=100.0):
    """零件级指纹：主方向面组、引脚阵列、槽口、止口、花键、锥度、圆柱桶（旋转不变）"""
    fp = {"primary_faces": [], "socket_candidates": [], "slot_candidates": [],
          "pin_arrays": [], "spigot_candidates": [], "spline_groups": [],
          "tapered_cyls": [], "cyl_buckets": {}}

    # 面法向聚类：找主方向面组（替代 z_dominant，旋转不变）
    normal_clusters = {}  # (nx, ny, nz)_rounded → [faces]
    for f in planar:
        n = f["n"]
        n_key = (round(n[0], 2), round(n[1], 2), round(n[2], 2))
        normal_clusters.setdefault(n_key, []).append(f)
    # 取面积最大的聚类作为主方向面组
    primary_n_key = None
    primary_area = 0
    for n_key, group in normal_clusters.items():
        total_area = sum(f["area"] for f in group)
        if total_area > primary_area:
            primary_area = total_area
            primary_n_key = n_key

    for f in planar:
        n = f["n"]
        n_key = (round(n[0], 2), round(n[1], 2), round(n[2], 2))
        if n_key == primary_n_key:
            fp["primary_faces"].append({
                "area": f["area"],
                "n_circles": len(f.get("circles", [])),
                "n_lines": len(f.get("lines", [])),
                "inter_circle_dists": f.get("inter_circle_dists", [])
            })
        for pa in f.get("pin_arrays", []):
            fp["pin_arrays"].append({
                "radius": pa["radius"], "count": pa["count"],
                "bucket": pa["bucket"],
                "face_area": f["area"],
                "is_primary": n_key == primary_n_key
            })
        if f.get("is_slot"):
            fp["slot_candidates"].append({
                "area": f["area"],
                "aspect_ratio": f.get("aspect_ratio", 1),
                "is_primary": n_key == primary_n_key,
                "n_circles": len(f.get("circles", []))
            })

    # 止口检测：大半径 + 短轴 + 有台阶面（同轴不同半径的圆柱面，阈值相对于 L）
    spigot_r_min = ETA_SPIGOT_R_MIN * L
    spigot_step = ETA_SPIGOT_STEP * L
    large_cyls = [c for c in cylinders if c["r"] > spigot_r_min]
    for c in large_cyls:
        # 查找同轴的更大/更小圆柱（台阶特征）
        same_axis = [o for o in cylinders if o is not c and
            abs(c["dir"][0]*o["dir"][0] + c["dir"][1]*o["dir"][1] + c["dir"][2]*o["dir"][2]) > ZETA_AXIAL_ALIGN]
        bigger = any(o["r"] > c["r"] + spigot_step for o in same_axis)
        smaller = any(o["r"] < c["r"] - spigot_step for o in same_axis)
        if bigger or smaller:
            fp["spigot_candidates"].append({
                "radius": c["r"], "bucket": c.get("bucket", "?"),
                "has_step": True, "ext": c["ext"]
            })

    # 花键检测：≥MIN_CYL_FOR_SPLINE 个同半径、同向、同轴的圆柱（内外花键）
    by_bucket_axis = {}
    for c in cylinders:
        bk = c.get("bucket", f"R{round(c['r'],1)}")
        ax_key = (bk, round(c["dir"][0],1), round(c["dir"][1],1), round(c["dir"][2],1))
        by_bucket_axis.setdefault(ax_key, []).append(c)
    for ax_key, group in by_bucket_axis.items():
        if len(group) >= MIN_CYL_FOR_SPLINE:
            fp["spline_groups"].append({
                "bucket": ax_key[0], "count": len(group),
                "radius": group[0]["r"], "ext": group[0]["ext"]
            })

    # 锥度圆柱
    for c in cylinders:
        if c.get("is_tapered"):
            fp["tapered_cyls"].append({
                "radius": c["r"], "taper": c["taper"],
                "ext": c["ext"], "bucket": c.get("bucket", "?")
            })

    # 同心面检测：旋转不变版（同法向面组内找面心投影相同但深度不同的面）
    for n_key, group in normal_clusters.items():
        if len(group) < MIN_FACES_FOR_CONCENTRIC:
            continue
        nv = cq.Vector(n_key[0], n_key[1], n_key[2])
        if nv.Length < 1e-9:
            continue
        nv = nv.normalized()
        # 按面心沿法向的投影 + 面内位置 分组
        by_pos = {}
        for f in group:
            fc = cq.Vector(f["c"][0], f["c"][1], f["c"][2])
            proj_depth = fc.dot(nv)
            proj_in_plane = fc - nv * proj_depth
            ip_key = (round(proj_in_plane.x / 5) * 5,
                      round(proj_in_plane.y / 5) * 5,
                      round(proj_in_plane.z / 5) * 5)
            by_pos.setdefault(ip_key, []).append(proj_depth)
        for ip_key, depths in by_pos.items():
            if len(depths) >= MIN_FACES_FOR_CONCENTRIC and max(depths) - min(depths) > ETA_SPIGOT_STEP * L:
                fp["concentric_step_groups"] = fp.get("concentric_step_groups", 0) + 1

    # 圆柱桶分布 + 端面标记
    for c in cylinders:
        bk = c.get("bucket", f"R{round(c['r'],1)}")
        fp["cyl_buckets"][bk] = fp["cyl_buckets"].get(bk, 0) + 1
    # 统计端面圆柱（axial_pos 基于包围盒最长轴，旋转不变）
    end_cyls = [c for c in cylinders if c.get("axial_pos", -1) >= 0
                and (c["axial_pos"] < ZETA_AXIAL_END_LO or c["axial_pos"] > ZETA_AXIAL_END_HI)]
    if end_cyls:
        fp["end_face_cyls"] = len(end_cyls)

    return fp


# ==================== summary ====================
def summary(features, L=100.0):
    p = features["planar"]
    c = features["cylinders"]
    cyl_by_r = {}
    cyl_by_bucket = {}
    for x in c:
        rk = round(x["r"], 1)
        cyl_by_r[rk] = cyl_by_r.get(rk, 0) + 1
        bk = x.get("bucket", f"R{round(x['r'],1)}")
        cyl_by_bucket[bk] = cyl_by_bucket.get(bk, 0) + 1
    return {
        "n_planar": len(p),
        "n_cylinders": len(c),
        "cyl_by_radius": {str(k): v for k, v in sorted(cyl_by_r.items())[:20]},
        "cyl_buckets": {k: v for k, v in sorted(cyl_by_bucket.items())},
        "max_face_area": max((f["area"] for f in p), default=0),
        "diag_len": round(L, 1),
        "fingerprint": _compute_fingerprint(p, c, L)
    }


# ==================== main ====================
def _import_cached(filepath):
    """导入 STEP，自动缓存为 .brep。二次加载百毫秒级。"""
    brep_path = filepath.rsplit(".", 1)[0] + ".brep"
    if os.path.exists(brep_path) and os.path.getmtime(brep_path) >= os.path.getmtime(filepath):
        shape = cq.Shape.importBrep(brep_path)
        return cq.Workplane("XY").newObject([shape])
    wp = importers.importStep(filepath)
    wp.val().exportBrep(brep_path)
    return wp


def extract_file(filepath):
    shape = _import_cached(filepath)
    L = _characteristic_scale(shape)
    centroid, principal_axes = _principal_axes(shape)
    planar = extract_planar(shape, L, principal_axes[0])
    cylinders = extract_cylinders(shape, L, principal_axes[0])

    # 构建全量面拓扑邻接图（含平面+圆柱面等，法兰的圆柱孔邻居可被表达）
    centroid_vec = cq.Vector(centroid.x, centroid.y, centroid.z)
    topo = _build_aag(shape, centroid_vec)

    # 将 planar faces 映射到全量 AAG 节点中（面中心+面积双重校验）
    all_faces = list(shape.faces().vals())
    for pf in planar:
        pf_c = pf["c"]; pf_area = pf["area"]
        match_idx = -1
        for idx, f in enumerate(all_faces):
            c = f.Center()
            if (abs(c.x - pf_c[0]) < 0.1 and abs(c.y - pf_c[1]) < 0.1
                    and abs(c.z - pf_c[2]) < 0.1 and abs(f.Area() - pf_area) < 1.0):
                match_idx = idx
                break
        if match_idx >= 0:
            node = topo["nodes"][match_idx]
            pf["surface_type"] = node["type"]
            pf["n_internal_loops"] = node["n_internal_loops"]
            # 允许平面邻域包含非平面邻居（核心修正！法兰面→圆柱孔边被正确表达）
            pf["neighbors"] = [
                {"idx": e["dst"] if e["src"] == match_idx else e["src"],
                 "convexity": e["convexity"],
                 "edge_geom": e["edge_geom"]}
                for e in topo["edges"] if e["src"] == match_idx or e["dst"] == match_idx
            ]
        else:
            pf["surface_type"] = 'plane'
            pf["n_internal_loops"] = 0
            pf["neighbors"] = []

    features = {
        "planar": planar, "cylinders": cylinders,
        "diag_len": round(L, 1),
        "principal_axes": [[a.x, a.y, a.z] for a in principal_axes],
        "centroid": [centroid.x, centroid.y, centroid.z],
        "topology_graph": topo,
    }
    features["_summary"] = summary(features, L)
    return features


if __name__ == "__main__":
    import glob
    target = sys.argv[1] if len(sys.argv) > 1 else "./2"
    if os.path.isfile(target):
        files = [target]; out_dir = os.path.dirname(target) or "."
    else:
        files = glob.glob(os.path.join(target, "*.step")) + glob.glob(os.path.join(target, "*.stp"))
        out_dir = target

    for fp in files:
        if "virtual" in os.path.basename(fp): continue
        nm = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_dir, f"{nm}_features.json")

        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                cached = json.load(f)
            # 检查是否有新格式字段（diag_len），无则重新提取
            if cached.get("diag_len") and cached.get("_summary", {}).get("fingerprint"):
                print(f"  [cache] {nm}: {cached['_summary']['n_planar']}p {cached['_summary']['n_cylinders']}c")
                continue

        print(f"  [extracting] {nm}...", end=" ", flush=True)
        features = extract_file(fp)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
        s = features["_summary"]
        print(f"{s['n_planar']}p {s['n_cylinders']}c")
    print("done")
