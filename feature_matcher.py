"""
feature_matcher.py
==================
模块2：在零件特征之间做匹配（独立运行）
用法：
  python feature_matcher.py <folder>
"""
import os, sys, json, math
import cadquery as cq
from label_generator import planar_labels, cylinder_labels, _bore_face_intersection, _neg, _vec_sub, mk_label, fmt_json, _ortho

TOL = 0.1; MIN_OK = 3; MIN_OK_LOOSE = 2; CYL_TOL = 0.5; SLOT_MIN = 2
MULTI_CYL_THRESHOLD = 50  # 超过此数量触发框架嵌入逻辑


def _match_list(la, lb, key, tol):
    m, u = [], [False] * len(lb)
    for a in la:
        for j, b in enumerate(lb):
            if u[j]: continue
            if abs(a[key] - b[key]) < tol: m.append((a, b)); u[j] = True; break
    return m


def match_planar(fa, fb):
    res = []
    for a in fa:
        for b in fb:
            # 圆用半径匹配（CYL_TOL=0.5mm），替代周长 0.1mm（半径仅0.016mm过严）
            mc = _match_list(a["circles"], b["circles"], "r", CYL_TOL)
            ml = _match_list(a["lines"], b["lines"], "len", TOL)
            t = len(mc) + len(ml)
            if t >= MIN_OK or (t >= MIN_OK_LOOSE and len(mc) >= 1):
                res.append({"fa": a, "fb": b, "mc": mc, "ml": ml, "t": t})
    return res


def match_slot(fa, fb):
    res = []
    for a in fa:
        for b in fb:
            mc = _match_list(a["circles"], b["circles"], "r", CYL_TOL)
            ml = _match_list(a["lines"], b["lines"], "len", TOL)
            t = len(mc) + len(ml)
            if t >= SLOT_MIN:
                res.append({"fa": a, "fb": b, "mc": mc, "ml": ml, "t": t})
    return res


def _bucket_key(cyl):
    """获取圆柱的半径桶标签（兼容旧特征文件）"""
    r = cyl["r"]
    bucket = cyl.get("bucket")
    if bucket is not None:
        return bucket
    # 旧文件兼容：现场分桶
    from feature_extractor import quantize_radius
    return quantize_radius(r)


def _classify_match(m):
    """基于配合紧密度分类匹配角色（非绝对半径）。
    螺栓孔：bore > shaft 且间隙 >= 0.3mm（明显间隙配合）
    止口/销：间隙极小(<0.1) 或 过盈的精密配合"""
    if m.get("bore_to_bore"):
        return "bolt-verify"
    if m.get("interference"):
        return "interference"
    clearance = m.get("clearance", 0)
    s_r = m["shaft"]["r"]
    if clearance >= 0.3:  # 超过0.3mm间隙 → 螺栓孔
        return "bolt"
    if s_r > 15:  # 大半径 + 紧密配合 → 止口
        return "spigot"
    if clearance < 0.1 and s_r >= 2 and s_r <= 12:  # 紧密配合 + 中等半径 → 定位销
        return "dowel"
    return "shaft-bore"


def match_cylinders(ca, cb, strict_axis=False, ref_a=None, ref_b=None):
    """圆柱匹配：桶索引 O(k²) 替代笛卡尔积 O(n×m)"""
    # 构建 cb 的桶索引
    cb_index = {}
    for b_idx, b in enumerate(cb):
        bk = _bucket_key(b)
        cb_index.setdefault(bk, []).append((b_idx, b))

    ms = []
    used_a = set()
    used_b = set()

    for a_idx, a in enumerate(ca):
        bk = _bucket_key(a)
        if bk not in cb_index:  # cb 中没有同桶的圆柱 → 跳过
            continue
        for b_idx, b in cb_index[bk]:
            same_type = a["ext"] == b["ext"]
            if same_type and a["ext"]: continue  # 跳过轴-轴
            if abs(a["r"] - b["r"]) > CYL_TOL: continue
            if strict_axis:
                dot = abs(a["dir"][0]*b["dir"][0] + a["dir"][1]*b["dir"][1] + a["dir"][2]*b["dir"][2])
                if dot < 0.9: continue
            if ref_a and ref_b and strict_axis:
                da = sum((a["mid"][k] - ref_a[k])**2 for k in range(3))**0.5
                db = sum((b["mid"][k] - ref_b[k])**2 for k in range(3))**0.5
                if abs(da - db) > 5: continue
            if same_type:  # 孔-孔（bore-to-bore）
                if b_idx in used_b: continue
                if a_idx in used_a: continue
                big_a = max((c for c in ca if not c["ext"]), key=lambda c: c["r"], default=None)
                big_b = max((c for c in cb if not c["ext"]), key=lambda c: c["r"], default=None)
                if big_a and big_b and big_a["r"] > a["r"] and big_b["r"] > b["r"]:
                    import cadquery as cq
                    def _radial(cyl, ref):
                        m = cq.Vector(*cyl["mid"]); rm = cq.Vector(*ref["mid"])
                        rd = cq.Vector(*ref["dir"])
                        return (m - rm - rd * (m - rm).dot(rd)).Length
                    ra = _radial(a, big_a); rb = _radial(b, big_b)
                    if abs(ra - rb) > 1.0:
                        continue
                used_b.add(b_idx); used_a.add(a_idx)
                ms.append({"shaft": a, "bore": b, "shaft_in_a": True, "bore_to_bore": True})
                break
            else:
                if b_idx in used_b: continue
                if a_idx in used_a: continue
                used_b.add(b_idx); used_a.add(a_idx)
                shaft_cyl = a if a["ext"] else b
                bore_cyl = b if a["ext"] else a
                dr = shaft_cyl["r"] - bore_cyl["r"]
                # 过盈配合：轴略大于孔(0~0.05mm) → 设计意图的物理干涉
                # dr > 0.05mm → 不是过盈配合，是尺寸错误/严重干涉
                interference = dr > 0 and dr <= 0.05
                clearance = max(0, -dr)  # 间隙量
                ms.append({"shaft": shaft_cyl, "bore": bore_cyl,
                           "shaft_in_a": a["ext"], "interference": interference,
                           "clearance": clearance})
                break
    return ms


# ========== 过滤函数 ==========
def _bore_filter(cylinders, planar_faces):
    internal = [c for c in cylinders if not c["ext"]]
    if not internal: return planar_faces
    bore = max(internal, key=lambda c: c["r"])
    d = cq.Vector(bore["dir"][0], bore["dir"][1], bore["dir"][2])
    mid = cq.Vector(bore["mid"][0], bore["mid"][1], bore["mid"][2])
    r = bore["r"]
    out = []
    for f in planar_faces:
        fc = cq.Vector(f["c"][0], f["c"][1], f["c"][2])
        radial = fc.sub(mid).sub(d.multiply(fc.sub(mid).dot(d)))
        fn = cq.Vector(f["n"][0], f["n"][1], f["n"][2])
        if abs(radial.Length - r) < 15 and abs(fn.dot(d)) < 0.3:
            out.append(f)
    return out if out else planar_faces


def _shaft_keyway_filter(cylinders, planar_faces):
    cyl = next((c for c in cylinders if c["ext"]), None)
    if not cyl: return planar_faces
    d = cq.Vector(cyl["dir"][0], cyl["dir"][1], cyl["dir"][2])
    mid = cq.Vector(cyl["mid"][0], cyl["mid"][1], cyl["mid"][2])
    r = cyl["r"]
    out = []
    for f in planar_faces:
        fc = cq.Vector(f["c"][0], f["c"][1], f["c"][2])
        radial = fc.sub(mid).sub(d.multiply(fc.sub(mid).dot(d)))
        if abs(radial.Length - r) < 8 and len(f["lines"]) >= 4:
            out.append(f)
    return out if out else planar_faces


# ========== 阵列检测 ==========
def _is_circular_array(face):
    """检测面上的圆是否构成圆周阵列（螺栓孔模式）"""
    circles = face.get("circles", [])
    if len(circles) < 3: return False
    # 检查是否大多数圆半径相同
    radii = [c["len"] / (2*math.pi) for c in circles]
    median_r = sorted(radii)[len(radii)//2]
    same_r = sum(1 for r in radii if abs(r - median_r) < 0.5)
    if same_r < 3: return False
    # 计算圆心的中心点
    cx = sum(c["c"][0] for c in circles) / len(circles)
    cy = sum(c["c"][1] for c in circles) / len(circles)
    # 检查同半径圆到中心的距离是否相近
    dists = []
    for c in circles:
        r = c["len"] / (2*math.pi)
        if abs(r - median_r) < 0.5:
            d = ((c["c"][0]-cx)**2 + (c["c"][1]-cy)**2)**0.5
            dists.append(d)
    if len(dists) < 3: return False
    median_d = sorted(dists)[len(dists)//2]
    consistent = sum(1 for d in dists if abs(d - median_d) < max(5, median_d*0.1))
    return consistent >= 3  # at least 3 holes on the same bolt circle


def _is_linear_array(faces, axis=0):
    """检测面集合是否构成线性阵列（沿 axis 轴等间距排列）"""
    if len(faces) < 2: return False
    # 提取中心坐标 + 面积
    pts = [(f["c"][axis], f.get("area", 0)) for f in faces]
    pts.sort()
    # 计算间距
    gaps = [pts[i+1][0] - pts[i][0] for i in range(len(pts)-1)]
    if not gaps: return False
    median_gap = sorted(gaps)[len(gaps)//2]
    if median_gap < 5: return False
    # 检查间距一致性（间距偏差 < 20%）
    consistent = sum(1 for g in gaps if abs(g - median_gap) < max(median_gap*0.25, 10))
    return consistent >= len(gaps) - 1  # allow at most 1 outlier


# ========== 框架嵌入匹配（Step 0）==========
def _is_frame_face(f):
    """检测是否为矩形框架面：有4条长边构成矩形/方形轮廓"""
    lines = f.get("lines", [])
    if len(lines) < 4:
        return False
    if f.get("area", 0) < 100:
        return False
    sorted_lines = sorted(lines, key=lambda l: l["len"], reverse=True)
    top4 = [l["len"] for l in sorted_lines[:4]]
    if min(top4) < 10:  # 框架边长至少10mm
        return False
    # 方形：4条边长度相近；矩形：两对相等边
    if max(top4) - min(top4) < 2.0:
        return True  # 正方形
    if abs(top4[0] - top4[1]) < 1.0 and abs(top4[2] - top4[3]) < 1.0:
        return True  # 矩形
    # 矩形检查：top4[0]≈top4[2], top4[1]≈top4[3]
    if abs(top4[0] - top4[2]) < 1.0 and abs(top4[1] - top4[3]) < 1.0:
        return True
    return False


def _match_frame_edges(a_lines, b_lines, tol=1.0):
    """匹配框架面的4条外边，允许间隙公差"""
    a_outer = sorted(a_lines, key=lambda l: l["len"], reverse=True)[:4]
    b_outer = sorted(b_lines, key=lambda l: l["len"], reverse=True)[:4]
    used_b = [False] * len(b_outer)
    matched = []
    for al in a_outer:
        best_j, best_diff = -1, float("inf")
        for j, bl in enumerate(b_outer):
            if used_b[j]: continue
            diff = abs(al["len"] - bl["len"])
            if diff < tol and diff < best_diff:
                best_diff, best_j = diff, j
        if best_j >= 0:
            matched.append((al, b_outer[best_j]))
            used_b[best_j] = True
    return matched


def _match_lines_spatial(a_lines, b_lines, a_center, b_center, len_tol=1.0):
    """匹配线段：长度相近 + 相对面中心的方向一致（上方配上方，左方配左方）"""
    matched = []
    used_b = [False] * len(b_lines)

    for al in a_lines:
        if al["len"] < 10:  # 只匹配框架长边（≥10mm），忽略内部短边
            continue
        # FAN线段指向
        adx = al["m"][0] - a_center[0]
        ady = al["m"][1] - a_center[1]
        alen = (adx**2 + ady**2) ** 0.5
        if alen < 1e-6:
            continue
        adir = (adx / alen, ady / alen)

        best_j, best_score = -1, float("inf")
        for j, bl in enumerate(b_lines):
            if used_b[j]:
                continue
            if bl["len"] < 10:
                continue
            len_diff = abs(al["len"] - bl["len"])
            if len_diff > len_tol:
                continue
            # CAGE线段指向
            bdx = bl["m"][0] - b_center[0]
            bdy = bl["m"][1] - b_center[1]
            blen = (bdx**2 + bdy**2) ** 0.5
            if blen < 1e-6:
                continue
            bdir = (bdx / blen, bdy / blen)
            # 方向相似度（dot > 0.7 ≈ 夹角 < 45°，同一象限）
            dot = adir[0] * bdir[0] + adir[1] * bdir[1]
            if dot < 0.7:
                continue
            score = len_diff + (1 - dot) * 10
            if score < best_score:
                best_score, best_j = score, j

        if best_j >= 0:
            matched.append((al, b_lines[best_j]))
            used_b[best_j] = True

    return matched


def _find_cage_frame_candidates(planar_faces):
    """找到所有框架面候选（CAGE的多个槽位）"""
    candidates = []
    for f in planar_faces:
        if not _is_frame_face(f):
            continue
        n = f["n"]
        # 槽面法向应为±Z方向（框架嵌入的面贴面）
        if abs(n[2]) < 0.9:
            continue
        candidates.append(f)
    candidates.sort(key=lambda f: f["area"], reverse=True)
    return candidates


def _frame_in_frame(na, nb, parts, labels, idx_counter, face_info, used_p, fif_slot_centers):
    """
    框架嵌入匹配：一个零件的框架嵌入另一个零件的槽位。
    策略：遍历所有FAN框架面 → 与CAGE槽面匹配 → 选无碰撞的最佳组合。
    返回: True/False
    fif_slot_centers: 记录 cage_name -> [(cx, cy, frame_size), ...] 供Step3同CS过滤用
    """
    fa_list = parts[na]["features"]["planar"]
    fb_list = parts[nb]["features"]["planar"]

    # 找两边框架面候选
    frame_a = [f for f in fa_list if _is_frame_face(f)]
    frame_b = [f for f in fb_list if _is_frame_face(f)]
    if not frame_a or not frame_b:
        return False

    slot_a = [f for f in frame_a if abs(f["n"][2]) > 0.9]
    slot_b = [f for f in frame_b if abs(f["n"][2]) > 0.9]
    if not slot_a or not slot_b:
        return False

    # 判断哪边是CAGE（多槽位）哪边是FAN（单框架）
    if len(slot_a) >= len(slot_b):
        cage_slots, fan_frames_init = slot_a, slot_b
        cage_name, fan_name = na, nb
    else:
        cage_slots, fan_frames_init = slot_b, slot_a
        cage_name, fan_name = nb, na

    # 加载FAN几何体（过大文件跳过，避免OOM/卡死）
    import os as _os
    fan_shape_path = parts[fan_name]["shape_path"]
    if _os.path.getsize(fan_shape_path) > 30 * 1024 * 1024:
        return False  # >30MB 跳过 FIF
    from cadquery import importers as cq_importers
    fan_shape = cq_importers.importStep(fan_shape_path).val()
    fan_bb = fan_shape.BoundingBox()
    fan_centroid = cq.Vector(
        (fan_bb.xmin + fan_bb.xmax) / 2,
        (fan_bb.ymin + fan_bb.ymax) / 2,
        (fan_bb.zmin + fan_bb.zmax) / 2,
    )
    # 过滤FAN框架面：只保留实体在法向反方向的面（法向朝外，实体在法向反方向=正确配合面）
    valid_fan_frames = []
    for ff in fan_frames_init:
        # 3D向量：面心→质心，面法向应指向体外
        body_vec = fan_centroid - cq.Vector(ff["c"][0], ff["c"][1], ff["c"][2])
        face_n = cq.Vector(ff["n"][0], ff["n"][1], ff["n"][2])
        if body_vec.dot(face_n) < 0:  # 质心在法向反侧 = 实体在面内
            valid_fan_frames.append(ff)
    if valid_fan_frames:
        fan_frames_init = valid_fan_frames

    # 面特征Key（用于去重）
    def _fk_face(f):
        return f"p|{f['c'][0]:.4f}|{f['c'][1]:.4f}|{f['c'][2]:.4f}|{f['n'][0]:.4f}|{f['n'][1]:.4f}|{f['n'][2]:.4f}"

    # 过滤已占用的CAGE槽位（1-to-1：每个槽位只放一个FAN）
    # FAN面不过滤——只有一个FAN零件，每个槽位复用它
    cage_slots = [f for f in cage_slots if _fk_face(f) not in used_p.get(cage_name, set())]
    if not cage_slots or not fan_frames_init:
        return False

    # 碰撞检测：法向相反即通过
    def _check_fit(fan_face, cage_face):
        return fan_face["n"][2] * cage_face["n"][2] < -0.25

    # 合并共线碎段线：将同方向相邻短线段重组长边
    def _merge_fragments(lines, center, angle_bin=0.15, prox_tol=3.0):
        if len(lines) < 4:
            return lines
        by_dir = {}
        for l in lines:
            if l["len"] < 1.0:  # skip tiny edges
                continue
            dx = l["m"][0] - center[0]
            dy = l["m"][1] - center[1]
            ang = round(math.atan2(dy, dx) / angle_bin) * angle_bin
            by_dir.setdefault(ang, []).append(l)
        merged = []
        for ang, group in by_dir.items():
            # sort by position along direction
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            group.sort(key=lambda l: l["m"][0]*cos_a + l["m"][1]*sin_a)
            cur_len = 0.0; cur_m = [0.0, 0.0, 0.0]; count = 0
            for l in group:
                cur_len += l["len"]
                for k in range(3):
                    cur_m[k] += l["m"][k] * l["len"]
                count += 1
            if cur_len > 0 and count >= 1:
                for k in range(3):
                    cur_m[k] /= cur_len
                merged.append({"len": round(cur_len, 3), "m": cur_m})
        return merged

    found_any = False
    # 收集所有 (fan_frame, cage_slot) 匹配对，然后贪婪选择+阵列过滤
    all_pairs = []  # (score, fan_frame, cage_cand, ml, mc)
    for fan_frame in fan_frames_init:
        for f in cage_slots:
            cn = cq.Vector(f["n"][0], f["n"][1], f["n"][2])
            fn = cq.Vector(fan_frame["n"][0], fan_frame["n"][1], fan_frame["n"][2])
            if fn.z * cn.z > -0.25:
                continue
            fan_lines = fan_frame.get("lines", [])
            cage_lines = f.get("lines", [])
            ml = _match_lines_spatial(fan_lines, cage_lines,
                                      fan_frame["c"], f["c"], len_tol=5.0)
            # 回退1：线碎段化→放宽长度公差
            if len(ml) < 2:
                ml = _match_lines_spatial(fan_lines, cage_lines,
                                          fan_frame["c"], f["c"], len_tol=999.0)
            # 回退2：合并共线碎段后重试
            if len(ml) < 2:
                fan_merged = _merge_fragments(fan_lines, fan_frame["c"])
                cage_merged = _merge_fragments(cage_lines, f["c"])
                ml = _match_lines_spatial(fan_merged, cage_merged,
                                          fan_frame["c"], f["c"], len_tol=5.0)
            # 回退2：合并共线碎段 + len_tol=999
            if len(ml) < 2:
                ml = _match_lines_spatial(fan_merged, cage_merged,
                                          fan_frame["c"], f["c"], len_tol=999.0)
            if len(ml) < 1:
                continue
            mc = _match_list(fan_frame.get("circles", []), f.get("circles", []), "r", CYL_TOL)
            if not _check_fit(fan_frame, f):
                continue
            len_penalty = sum(abs(al["len"] - bl["len"]) for al, bl in ml)
            score = len(ml) * 10 + len(mc) - len_penalty * 0.5
            all_pairs.append((score, fan_frame, f, ml, mc))

    if not all_pairs:
        return False

    # XY去重+体不对称度优先：
    # 1) 不同FAN面：选body_offset更大的（主体偏向一侧=正确配合面）
    # 2) 同FAN面：选Z更深的内台阶面
    # 3) 同层：按得分
    xy_best = {}  # (x10, y10) -> (score, ff, cage_cand, ml, mc, body_offset, depth_z)
    for score, ff, cage_cand, ml, mc in all_pairs:
        x_key = round(cage_cand["c"][0] / 10) * 10
        y_key = round(cage_cand["c"][1] / 10) * 10
        xy = (x_key, y_key)
        # 深度方向依赖当前FAN面法向（用 ff 而非外循环残留的 fn）
        n_fan_z = ff["n"][2]
        depth_z = -cage_cand["c"][2] * n_fan_z
        body_off = abs(fan_centroid.z - ff["c"][2])
        if xy not in xy_best:
            xy_best[xy] = (score, ff, cage_cand, ml, mc, body_off, depth_z)
        else:
            old_score, _, _, _, _, old_body, old_depth = xy_best[xy]
            # 不同FAN面(body_off差>5mm) → body_off大的优先(主体偏向一侧=正确配合面)
            if abs(body_off - old_body) > 5:
                if body_off > old_body:
                    xy_best[xy] = (score, ff, cage_cand, ml, mc, body_off, depth_z)
            # 同FAN面 → Z深度优先(0.5mm分层)
            elif abs(depth_z - old_depth) > 0.5:
                if depth_z < old_depth:
                    xy_best[xy] = (score, ff, cage_cand, ml, mc, body_off, depth_z)
            # 同层 → 得分优先
            elif score > old_score:
                xy_best[xy] = (score, ff, cage_cand, ml, mc, body_off, depth_z)

    # 只保留最佳 1 个槽位；按 Y 聚类取最大组 → 线性阵列确认
    dedup_pairs = sorted(xy_best.values(), key=lambda x: x[0], reverse=True)
    kept_pairs = [dedup_pairs[0]] if dedup_pairs else []
    if len(dedup_pairs) >= 3:
        # 按 Y 坐标聚类（同行的槽位），取最大的一组
        y_groups = {}
        for _, _, cage_cand, _, _, _, _ in dedup_pairs:
            y10 = round(cage_cand["c"][1] / 10) * 10
            y_groups.setdefault(y10, []).append(cage_cand)
        largest_group = max(y_groups.values(), key=len)
        if len(largest_group) >= 3 and _is_linear_array(largest_group, axis=0):
            print(f"  [ARRAY] {fan_name}<->{cage_name}: {len(largest_group)} of "
                  f"{len(dedup_pairs)} slots form linear array (selected best 1)")

    # 生成标签
    for score, ff, cage_cand, ml, mc, _bo, _dz in kept_pairs:
        t = len(mc) + len(ml)
        m = {"fa": ff, "fb": cage_cand, "mc": mc, "ml": ml, "t": t}
        if fan_name != na:
            na_real, nb_real = fan_name, cage_name
        else:
            na_real, nb_real = na, nb

        idx = idx_counter[0]
        idx_counter[0] += 1
        la, lb = planar_labels(m, na_real, nb_real, idx)
        # 自适应 Z 插深修正：读取 CAGE 槽位内螺栓孔端面 Z，补偿 FAN 原点
        # 找到该槽位(xy)附近的 CAGE 内孔面，其 Z 坐标即真正的定位台阶深度
        slot_x, slot_y = cage_cand["c"][0], cage_cand["c"][1]
        # Z 插深修正：找槽位内 Z 向平行且更深的面（螺栓孔肩面等）
        inner_z = cage_cand["c"][2]
        all_cage_faces = fb_list if cage_name == nb else fa_list
        for cf in all_cage_faces:
            if abs(cf["c"][0] - slot_x) < 30 and abs(cf["c"][1] - slot_y) < 30:
                # 法向平行（同向或反向）且沿法向更深
                nz_prod = cf["n"][2] * cage_cand["n"][2]
                if abs(nz_prod) < 0.5:
                    continue  # 法向不平行
                dz = cf["c"][2] - cage_cand["c"][2]
                if 2.0 < abs(dz) < 80.0:  # 深度差 2~80mm（覆盖厚风扇36.6mm）
                    if (cage_cand["n"][2] > 0 and cf["c"][2] > inner_z) or \
                       (cage_cand["n"][2] < 0 and cf["c"][2] < inner_z):
                        inner_z = cf["c"][2]
                    if (cage_cand["n"][2] > 0 and cf["c"][2] > inner_z) or \
                       (cage_cand["n"][2] < 0 and cf["c"][2] < inner_z):
                        inner_z = cf["c"][2]
        if abs(inner_z - cage_cand["c"][2]) > 1.0:
            lb["geometry"]["origin"]["z"] += (inner_z - cage_cand["c"][2])

        # 修正 FAN Z 面选择：当前 ff 可能是后端面，用螺栓孔 Z 找前端安装面
        all_fan_faces = fa_list if fan_name == na else fb_list
        bolt_zs = []
        for bf in all_fan_faces:
            if abs(bf["c"][0] - ff["c"][0]) < 30 and abs(bf["c"][1] - ff["c"][1]) < 30:
                if abs(bf["n"][2]) > 0.7 and len(bf.get("circles", [])) >= 2:
                    bolt_zs.append(bf["c"][2])
        if bolt_zs:
            # 取 Z 最大（最靠近 CAGE 格栅侧）= 前端面
            front_z = max(bolt_zs)
            if abs(front_z - ff["c"][2]) > 2.0:
                la["geometry"]["origin"]["z"] += (front_z - ff["c"][2])

        labels[na_real].append(la)
        labels[nb_real].append(lb)

        # 锁定CAGE槽位
        used_p.setdefault(nb_real, set()).add(_fk_face(cage_cand))
        if na_real not in face_info:
            face_info[na_real] = {"c": ff["c"], "n": ff["n"]}
        if nb_real not in face_info:
            face_info[nb_real] = {"c": cage_cand["c"], "n": cage_cand["n"]}

        # 记录CAGE槽位中心（均在CAGE CS内），供Step3圆柱过滤用
        # frame_size用匹配到的CAGE框架最短边长估算
        cage_lines = cage_cand.get("lines", [])
        frame_edges = sorted([l["len"] for l in cage_lines], reverse=True)[:4]
        frame_size = min(frame_edges) if len(frame_edges) >= 4 else 30.0
        fif_slot_centers.setdefault(cage_name, []).append(
            (cage_cand["c"][0], cage_cand["c"][1], frame_size))

        print(f"  [{idx}] {na_real}<->{nb_real}: {len(mc)}c+{len(ml)}l"
              f" @({cage_cand['c'][0]:.0f},{cage_cand['c'][1]:.0f}) [FRAME-IN-FRAME]")
        found_any = True


# ========== 验证辅助函数 ==========

def _vec(v):
    """[x,y,z] → cq.Vector"""
    return cq.Vector(v[0], v[1], v[2])


def _xform_point(loc, vec):
    """用 Location 变换点"""
    from OCP.gp import gp_Pnt
    trsf = loc.wrapped.Transformation()
    p = gp_Pnt(vec.x, vec.y, vec.z)
    return cq.Vector(p.Transformed(trsf).X(), p.Transformed(trsf).Y(), p.Transformed(trsf).Z())


def _xform_dir(loc, vec):
    """用 Location 变换方向向量"""
    return _xform_point(loc, vec) - _xform_point(loc, cq.Vector(0, 0, 0))


def _check_collision(shape_a, shape_b, T_AB,
                      min_volume=100.0, min_ratio=0.001):
    """检测两零件装配后是否有体积交叠（布尔求交）。

    shape_a, shape_b: cadquery Shape 对象（各自局部 CS）
    T_AB: A→B 的刚体变换（cq.Location）
    返回: (has_collision, overlap_ratio, info_dict)
    碰撞判定：交集体积 > 100mm³ 且 比率 > 0.1%
    """
    # 变换 A 到 B 的空间
    shape_a_moved = shape_a.located(T_AB)

    # AABB 快速过滤
    bb_a = shape_a_moved.BoundingBox()
    bb_b = shape_b.BoundingBox()
    vol_a = (bb_a.xmax - bb_a.xmin) * (bb_a.ymax - bb_a.ymin) * (bb_a.zmax - bb_a.zmin)
    if vol_a <= 0:
        return False, 0.0, {"aabb_ratio": 0.0, "bool_ratio": 0.0}

    dx = max(0, min(bb_a.xmax, bb_b.xmax) - max(bb_a.xmin, bb_b.xmin))
    dy = max(0, min(bb_a.ymax, bb_b.ymax) - max(bb_a.ymin, bb_b.ymin))
    dz = max(0, min(bb_a.zmax, bb_b.zmax) - max(bb_a.zmin, bb_b.zmin))
    aabb_ratio = (dx * dy * dz) / vol_a

    if aabb_ratio < 0.001:
        return False, 0.0, {"aabb_ratio": aabb_ratio, "bool_ratio": 0.0}

    # 精确布尔求交
    bool_vol = -1.0
    try:
        result = shape_a_moved.intersect(shape_b)
        if result.isValid():
            bool_vol = result.Volume()
    except Exception:
        bool_vol = -1.0

    if bool_vol < 0:
        # 布尔求交失败 → 降级 AABB：< 10% 为面接触/正确嵌套，不算碰撞
        has_collision = aabb_ratio > 0.10
        return has_collision, -1.0, {"aabb_ratio": aabb_ratio, "bool_ratio": -1.0, "bool_vol": -1.0}

    bool_ratio = bool_vol / vol_a if vol_a > 0 else 0.0
    has_collision = bool_vol > min_volume
    return has_collision, bool_ratio, {"aabb_ratio": aabb_ratio, "bool_ratio": bool_ratio, "bool_vol": bool_vol}


def _label_transform(la, lb):
    """从一对标签计算 T_AB：零件A→零件B 的刚体变换。
    公式：world[B] = world[A] * loc_A * loc_B⁻¹，即 T_AB = loc_A * loc_B⁻¹
    """
    def _geo_to_loc(geo):
        o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
        x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
        z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
        return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))
    return _geo_to_loc(la["geometry"]) * _geo_to_loc(lb["geometry"]).inverse


def _loc_from_origin_and_z(origin, z_dir, x_hint=None):
    """从原点和z方向构建 cadquery.Location"""
    z = cq.Vector(z_dir[0], z_dir[1], z_dir[2])
    if x_hint:
        ref = cq.Vector(x_hint[0], x_hint[1], x_hint[2])
    else:
        ref = cq.Vector(1, 0, 0) if abs(z.x) < 0.9 else cq.Vector(0, 1, 0)
    x = ref - z * (ref.dot(z) / z.dot(z))
    x = cq.Vector(1, 0, 0) if x.Length < 1e-9 else x.normalized()
    return cq.Location(cq.Plane(origin=_vec(origin), xDir=x, normal=z))


def _compute_cylinder_transform(m):
    """从CYLINDER匹配dict计算 T_AB: A→B 的刚体变换"""
    s, b = m["shaft"], m["bore"]
    shaft_in_a = m.get("shaft_in_a", True)
    if shaft_in_a:
        loc_a = _loc_from_origin_and_z(s["mid"], s["dir"])
        loc_b = _loc_from_origin_and_z(b["mid"], b["dir"])
    else:
        loc_a = _loc_from_origin_and_z(b["mid"], b["dir"])
        loc_b = _loc_from_origin_and_z(s["mid"], s["dir"])
    return loc_a * loc_b.inverse


def _compute_planar_transform(m):
    """从PLANAR匹配dict计算 T_AB: A→B 的刚体变换"""
    fa, fb = m["fa"], m["fb"]
    loc_a = _loc_from_origin_and_z(fa["c"], fa["n"])
    loc_b = _loc_from_origin_and_z(fb["c"], _neg(fb["n"]))
    return loc_a * loc_b.inverse


def _verify_planar_consistency(kept_match, discarded_matches,
                                tol_distance=5.0, tol_angle_deg=20.0):
    """验证被丢弃的PLANAR匹配是否与kept产生相同的刚体变换。
    如果两对面代表同一装配关系，变换应一致。
    """
    T_k = _compute_planar_transform(kept_match)
    passed, total = 0, 0
    for m in discarded_matches:
        total += 1
        T_d = _compute_planar_transform(m)
        pt_a = _vec(m["fa"]["c"])
        dist = (_xform_point(T_k, pt_a) - _xform_point(T_d, pt_a)).Length
        n_a = _vec(m["fa"]["n"])
        d_k = _xform_dir(T_k, n_a)
        n_expected = _vec(_neg(m["fb"]["n"]))
        if d_k.Length > 1e-9:
            dot_v = max(-1.0, min(1.0, d_k.normalized().dot(n_expected.normalized())))
            angle_deg = math.degrees(math.acos(abs(dot_v)))
        else:
            angle_deg = 0.0
        ok = dist < tol_distance and angle_deg < tol_angle_deg
        if ok: passed += 1
    return passed, total


def _verify_cylinder_consistency(kept_match, discarded_matches, tol_angle_deg=15.0):
    """验证被丢弃的CYLINDER匹配的轴线方向是否与kept一致。

    不同孔/轴位置不同是正常的（螺栓孔分布在法兰不同位置），
    但它们应该共轴——轴线方向必须一致。
    只比较方向，不比较位置。
    """
    T_k = _compute_cylinder_transform(kept_match)
    passed, total = 0, 0
    for m in discarded_matches:
        total += 1
        T_d = _compute_cylinder_transform(m)
        s = m["shaft"]
        d_a = _vec(s["dir"])
        d_k = _xform_dir(T_k, d_a)
        d_d = _xform_dir(T_d, d_a)
        if d_k.Length > 1e-9 and d_d.Length > 1e-9:
            dot_v = max(-1.0, min(1.0, d_k.normalized().dot(d_d.normalized())))
            angle_deg = math.degrees(math.acos(abs(dot_v)))
        else:
            angle_deg = 0.0
        ok = angle_deg < tol_angle_deg
        if ok: passed += 1
    return passed, total


def _run_verification(discarded_planar, discarded_cylinder):
    """用被丢弃的标签验证选中标签的正确性。
    返回 {pair: {type: (passed, total, rate)}} 供回退决策使用。
    """
    results = {}; total_passed = 0; total_count = 0
    lines = []

    for (na, nb), (kept, discarded) in discarded_planar.items():
        if kept is None: continue
        passed, count = _verify_planar_consistency(kept, discarded)
        total_passed += passed; total_count += count
        if count > 0:
            rate = passed / count
            tag = " [OK]" if rate > 0.80 else (" [WARN]" if rate < 0.30 else "")
            lines.append((na, nb, "PLANAR", passed, count, rate, tag))
            results.setdefault((na, nb), {})["PLANAR"] = (passed, count, rate)

    for (na, nb), (kept, discarded) in discarded_cylinder.items():
        if kept is None: continue
        passed, count = _verify_cylinder_consistency(kept, discarded)
        total_passed += passed; total_count += count
        if count > 0:
            rate = passed / count
            tag = " [OK]" if rate > 0.80 else (" [WARN]" if rate < 0.30 else "")
            lines.append((na, nb, "CYL", passed, count, rate, tag))
            results.setdefault((na, nb), {})["CYL"] = (passed, count, rate)

    if total_count == 0:
        return results

    print(f"\n=== Verification: {total_passed}/{total_count} "
          f"consistent ({total_passed/total_count:.1%}) ===")
    for na, nb, mtype, passed, count, rate, tag in lines:
        print(f"  {na}<->{nb} [{mtype}]: {passed}/{count} ({rate:.1%}){tag}")
    return results


# ========== 主入口 ==========
def match_all(parts, world_step=None):
    """
    parts: {name: {"features": {...}, "shape_path": "..."}}
    world_step: 可选，指定作为世界坐标参考的零件名（该零件在装配时保持静止）
    返回: labels_by_part
    """
    names = list(parts.keys())
    labels = {n: [] for n in names}
    idx = [0]

    def _fk(f):
        c, n = f["c"], f["n"]
        return f"p|{c[0]:.4f}|{c[1]:.4f}|{c[2]:.4f}|{n[0]:.4f}|{n[1]:.4f}|{n[2]:.4f}"

    face_info = {}; used_p = {}; slot_faces = {}; face_csys = {}
    discarded_planar_matches = {}  # {(na, nb): (kept_match, [discarded])}
    pairs_with_labels = set()  # 已有标签的零件对，Step 3 跳过
    ky_name = next((n for n in names if "key" in n.lower()), None)
    sh_name = next((n for n in names if "shaft" in n.lower()), None)

    # 懒加载 STEP 形状（仅在需要碰撞检测时加载）
    shapes = {}
    def _get_shape(nm):
        if nm not in shapes and "shape_path" in parts[nm]:
            from cadquery import importers as _imp
            shapes[nm] = _imp.importStep(parts[nm]["shape_path"]).val()
        return shapes.get(nm)

    # === Step 0: 检测多圆柱零件对 → 框架嵌入匹配 ===
    multi_cyl_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ca = parts[names[i]]["features"]["cylinders"]
            cb = parts[names[j]]["features"]["cylinders"]
            # 跳过两面都超多的对（仅一面多时可以匹配）
            pa = len(parts[names[i]]["features"]["planar"])
            pb = len(parts[names[j]]["features"]["planar"])
            if pa > 500 and pb > 500: continue
            if len(ca) > MULTI_CYL_THRESHOLD and len(cb) > MULTI_CYL_THRESHOLD:
                multi_cyl_pairs.append((names[i], names[j]))

    processed_fif = set()
    fif_slot_centers = {}  # cage_name -> [(cx, cy, frame_size), ...] 均在CAGE CS内
    for na, nb in multi_cyl_pairs:
        before_labels = sum(len(labels[n]) for n in [na, nb])
        _frame_in_frame(na, nb, parts, labels, idx, face_info, used_p, fif_slot_centers)
        after_labels = sum(len(labels[n]) for n in [na, nb])
        if after_labels > before_labels:
            processed_fif.add((na, nb))
            pairs_with_labels.add((na, nb))
            print(f"  [OK] frame-in-frame: {after_labels - before_labels} new labels for {na}<->{nb}")
        else:
            print(f"  [warn] frame-in-frame failed for {na}<->{nb}, fallback to planar")

    # === Step 1: planar ===
    planar_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            if (na, nb) in processed_fif:
                continue
            if ky_name and (ky_name in (na, nb) and any("flange" in x for x in (na, nb))):
                continue
            # 跳过两面都超多的对，一面多时只取最大的 100 个面
            pa = len(parts[na]["features"]["planar"])
            pb = len(parts[nb]["features"]["planar"])
            if pa > 200 and pb > 200:
                continue
            fa_list = parts[na]["features"]["planar"]
            fb_list = parts[nb]["features"]["planar"]
            if pa > 100:
                fa_list = sorted(fa_list, key=lambda f: f.get("area", 0), reverse=True)[:100]
            if pb > 100:
                fb_list = sorted(fb_list, key=lambda f: f.get("area", 0), reverse=True)[:100]
            ms = match_planar(fa_list, fb_list)
            if ms:
                ms.sort(key=lambda m: m["t"], reverse=True)
                planar_pairs.append((na, nb, ms))
    planar_pairs.sort(key=lambda x: x[2][0]["t"], reverse=True)

    for na, nb, ms in planar_pairs:
        # 第一遍：收集所有通过检查的有效匹配
        valid = []
        for m in ms:
            ka, kb = _fk(m["fa"]), _fk(m["fb"])
            if ka in used_p.get(na, set()) or kb in used_p.get(nb, set()): continue
            # outward check（仅当形状已缓存时才做，避免大文件加载卡死）
            s_a = shapes.get(na); s_b = shapes.get(nb)
            n_a = cq.Vector(m["fa"]["n"][0], m["fa"]["n"][1], m["fa"]["n"][2])
            n_b = cq.Vector(m["fb"]["n"][0], m["fb"]["n"][1], m["fb"]["n"][2])
            if s_a is not None and s_b is not None:
                bb_a = s_a.BoundingBox(); bb_b = s_b.BoundingBox()
                bc_a = cq.Vector((bb_a.xmin+bb_a.xmax)/2, (bb_a.ymin+bb_a.ymax)/2, (bb_a.zmin+bb_a.zmax)/2)
                bc_b = cq.Vector((bb_b.xmin+bb_b.xmax)/2, (bb_b.ymin+bb_b.ymax)/2, (bb_b.zmin+bb_b.zmax)/2)
                fc_a = cq.Vector(m["fa"]["c"][0], m["fa"]["c"][1], m["fa"]["c"][2])
                fc_b = cq.Vector(m["fb"]["c"][0], m["fb"]["c"][1], m["fb"]["c"][2])
                if fc_a.sub(bc_a).dot(n_a) < 0 or fc_b.sub(bc_b).dot(n_b) < 0: continue
            # 面法向检查：t≥3的高质量匹配跳过（管法兰、弯头等非直对场景）
            if m["t"] < MIN_OK and n_a.dot(n_b) > -0.5:
                continue
            ca = len(parts[na]["features"]["cylinders"])
            cb = len(parts[nb]["features"]["cylinders"])
            # 已由 FIF 处理的多圆柱对：跳过重复 PLANAR（用 3D 法向替代 Z 轴）
            if ca > 50 and cb > 50 and n_a.dot(n_b) < -0.7: continue
            valid.append(m)
        if not valid: continue

        # 排序：t 得分 + 圆周阵列加权（螺栓孔面优先）
        def _score(m):
            bonus = 100 if (_is_circular_array(m["fa"])
                        and _is_circular_array(m["fb"])) else 0
            return m["t"] + bonus
        valid.sort(key=_score, reverse=True)
        primary = valid[0]

        # 法向多样性：>30° 的视为不同接触面（如键的底面+侧壁），保留最多3个
        kept = [primary]
        for m in valid[1:]:
            n_candidate = _vec(m["fa"]["n"])
            is_new_face = True
            for k in kept:
                n_existing = _vec(k["fa"]["n"])
                dot_v = max(-1.0, min(1.0, n_candidate.normalized().dot(n_existing.normalized())))
                angle = math.degrees(math.acos(abs(dot_v)))
                if angle < 30:
                    is_new_face = False
                    break
            if is_new_face and len(kept) < 3:
                kept.append(m)

        # 生成标签：PRIMARY 输出到 JSON，SIDE 仅用于验证
        for m in kept:
            used_p.setdefault(na, set()).add(_fk(m["fa"]))
            used_p.setdefault(nb, set()).add(_fk(m["fb"]))
            is_primary = (m is primary)
            if is_primary:
                idx[0] += 1
            is_array_a = _is_circular_array(m["fa"])
            is_array_b = _is_circular_array(m["fb"])
            array_bonus = " [ARRAY]" if (is_array_a and is_array_b) else ""
            tag = " [PRIMARY]" if is_primary else " [SIDE-verify]"
            print(f"  [{idx[0]}] {na} <-> {nb}: {len(m['mc'])}c+{len(m['ml'])}l{array_bonus}{tag}")
            la, lb = planar_labels(m, na, nb, idx[0])
            if "flange" in na.lower() and "flange" in nb.lower():
                _flange_face_labels = getattr(match_all, '_flange_face_labels', None)
                if _flange_face_labels is None:
                    match_all._flange_face_labels = []
                match_all._flange_face_labels.append((na, nb, la, lb, m))
            if is_primary:
                labels[na].append(la); labels[nb].append(lb)
            if na not in face_info:
                face_info[na] = {"c": m["fa"]["c"], "n": m["fa"]["n"]}
                face_csys[na] = {"c": m["fa"]["c"], "n": m["fa"]["n"],
                                 "x": la["geometry"]["x"], "y": la["geometry"]["y"]}
            if nb not in face_info:
                face_info[nb] = {"c": m["fb"]["c"], "n": m["fb"]["n"]}
                face_csys[nb] = {"c": m["fb"]["c"], "n": m["fb"]["n"],
                                 "x": lb["geometry"]["x"], "y": lb["geometry"]["y"]}

        pairs_with_labels.add((na, nb))

        # 收集被丢弃的匹配用于验证
        remaining = [m for m in valid if m not in kept]
        if remaining:
            discarded_planar_matches[(na, nb)] = (primary, remaining)

    # === Step 2: slot detection (store slot face positions for CYLINDER label xDir) ===
    if sh_name:
        sh_feat = parts[sh_name]["features"]
        sh_filtered = _shaft_keyway_filter(sh_feat["cylinders"], sh_feat["planar"])
        for fl_name in [n for n in names if "flange" in n.lower()]:
            fl_feat = parts[fl_name]["features"]
            fl_filt = _bore_filter(fl_feat["cylinders"], fl_feat["planar"])
            ms = match_slot(sh_filtered, fl_filt)
            if ms:
                ms.sort(key=lambda m: m["t"], reverse=True)
                for m in ms:
                    if m["t"] < 2: continue
                    fk = _fk(m["fb"])
                    if fk in used_p.get(fl_name, set()): continue
                    used_p.setdefault(fl_name, set()).add(fk)
                    if fl_name not in face_info:
                        face_info[fl_name] = {"c": m["fb"]["c"], "n": m["fb"]["n"]}
                    if sh_name not in face_info:
                        face_info[sh_name] = {"c": m["fa"]["c"], "n": m["fa"]["n"]}
                    slot_faces[fl_name] = m["fb"]["c"]
                    slot_faces[sh_name] = m["fa"]["c"]
                    break

    # === Post-process: 修正 flange-flange 面标签的 xDir（用槽方向替代圆方向）===
    if hasattr(match_all, '_flange_face_labels'):
        for na, nb, la, lb, m in match_all._flange_face_labels:
            fa_has_slot = na in slot_faces
            fb_has_slot = nb in slot_faces
            if not fa_has_slot and not fb_has_slot:
                continue

            def _slot_x(part_name, face_center, face_normal, slot_center):
                """面心到槽面心的方向，投影到面平面"""
                fc = cq.Vector(*face_center)
                fn = cq.Vector(*face_normal)
                sc = cq.Vector(*slot_center)
                dir_to_slot = sc - fc
                # 投影到面平面
                proj = dir_to_slot - fn * (dir_to_slot.dot(fn) / fn.dot(fn))
                if proj.Length < 0.1:
                    return None
                return [proj.x, proj.y, proj.z]

            oa = m["fa"]["c"]; ob = m["fb"]["c"]
            fn_a = m["fa"]["n"]; fn_b = m["fb"]["n"]

            if fa_has_slot:
                sx_a = _slot_x(na, oa, fn_a, slot_faces[na])
                if sx_a:
                    la["geometry"].update(_ortho(
                    [la["geometry"]["z"]["x"], la["geometry"]["z"]["y"], la["geometry"]["z"]["z"]], sx_a))
            if fb_has_slot:
                sx_b = _slot_x(nb, ob, fn_b, slot_faces[nb])
                if sx_b:
                    lb["geometry"].update(_ortho(
                        [lb["geometry"]["z"]["x"], lb["geometry"]["z"]["y"], lb["geometry"]["z"]["z"]], sx_b))
        del match_all._flange_face_labels

    # === Step 3: cylinder matching ===
    cyl_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            ca = parts[na]["features"]["cylinders"]
            cb = parts[nb]["features"]["cylinders"]
            is_fif = (na, nb) in processed_fif
            strict = len(ca) > 50 and len(cb) > 50
            ref_a = face_info[na]["c"] if na in face_info else None
            ref_b = face_info[nb]["c"] if nb in face_info else None
            # 两面来自不同配合时（如都>200面），空间偏移验证无意义
            if (na, nb) not in pairs_with_labels:
                ref_a = ref_b = None
            ms = match_cylinders(ca, cb, strict_axis=strict,
                                 ref_a=(None if is_fif else ref_a),
                                 ref_b=(None if is_fif else ref_b))
            # 框架嵌入对：用CAGE槽位中心过滤（均在CAGE CS内），非FAN面中心（FAN CS内）
            if is_fif and ms:
                cage_name = na if len(ca) > len(cb) else nb
                fan_name = nb if len(ca) > len(cb) else na
                slot_centers = fif_slot_centers.get(cage_name, [])
                if slot_centers:
                    ms_filtered = []
                    for m in ms:
                        cage_cyl = m["shaft"] if (m["shaft_in_a"] and na == cage_name) or (not m["shaft_in_a"] and nb == cage_name) else m["bore"]
                        cx, cy = cage_cyl["mid"][0], cage_cyl["mid"][1]
                        # 对每个圆柱，检查是否在任一匹配槽位的范围内
                        for scx, scy, frame_size in slot_centers:
                            dxy = ((cx - scx)**2 + (cy - scy)**2)**0.5
                            # 自适应阈值：基于框架最短边长的0.7倍（螺栓孔在边框附近）
                            threshold = max(frame_size * 0.7, 15.0)
                            if dxy < threshold:
                                ms_filtered.append(m)
                                break
                    ms = ms_filtered
            if ms: cyl_pairs.append((na, nb, ms))

    def _ck(cyl):
        m = cyl["mid"]
        return f"c|{cyl['r']:.4f}|{m[0]:.4f}|{m[1]:.4f}|{m[2]:.4f}"

    # 按半径降序排列：大框架/主轴优先
    for pair in cyl_pairs:
        # 按角色优先级排序：过盈 > 止口 > 销 > 普通 > 螺栓
        role_order = {"interference": 0, "spigot": 1, "dowel": 2, "shaft-bore": 3, "bolt": 4, "bolt-verify": 5}
        pair[2].sort(key=lambda m: (
            role_order.get(_classify_match(m), 3),
            -m["shaft"]["r"]  # 同角色按半径降序
        ))

    used_c = {}
    used_sf = set()
    discarded_cylinder_matches = {}  # {(na, nb): (kept_match|None, [discarded])}

    for na, nb, ms in cyl_pairs:
        # 按 boreToBore 拆分：shaft-bore 优先用于装配
        shaft_bore = [m for m in ms if not m.get("bore_to_bore", False)]
        bore_to_bore = [m for m in ms if m.get("bore_to_bore", False)]

        # 选择最佳 1 个：优先 shaft-bore（最大半径），fallback bore-to-bore
        all_valid = shaft_bore + bore_to_bore
        if all_valid:
            discarded_cylinder_matches[(na, nb)] = (
                all_valid[0], all_valid[1:])

        # 已有 PLANAR/框架标签的零件对不再追加 CYLINDER 标签
        # 已有 PLANAR/框架标签的零件对不再追加 CYLINDER 标签
        if (na, nb) in pairs_with_labels:
            continue

        # 只为 kept 生成标签
        kept_ms = [all_valid[0]] if all_valid else []
        for m in kept_ms:
            s_has = na if m["shaft_in_a"] else nb
            b_has = nb if m["shaft_in_a"] else na
            is_sf = (sh_name and (s_has == sh_name or b_has == sh_name)
                     and any("flange" in x for x in (na, nb)))
            if is_sf:
                sf_key = "shaft-flange"
                if sf_key in used_sf: continue
                used_sf.add(sf_key)
            kb = _ck(m["bore"])
            if kb in used_c.get(b_has, set()): continue
            used_c.setdefault(b_has, set()).add(kb)
            s_cyl = m["shaft"]; b_cyl = m["bore"]
            bore_pt = None; shaft_pt = None
            if b_has in face_info:
                bore_pt = _bore_face_intersection(m["bore"], face_info[b_has])
            if s_has in face_info:
                shaft_pt = _bore_face_intersection(m["shaft"], face_info[s_has])
            if is_sf and s_has in face_info:
                fc_s = face_info[s_has]["c"]
                sd = cq.Vector(s_cyl["dir"][0], s_cyl["dir"][1], s_cyl["dir"][2])
                sm = cq.Vector(s_cyl["mid"][0], s_cyl["mid"][1], s_cyl["mid"][2])
                fc_v = cq.Vector(fc_s[0], fc_s[1], fc_s[2])
                proj = fc_v.sub(sm).dot(sd)
                shaft_pt = [sm.x + sd.x*proj, sm.y + sd.y*proj, sm.z + sd.z*proj]
            s_x = None; b_x = None
            if is_sf:
                if s_has in slot_faces:
                    s_x = _vec_sub(slot_faces[s_has], s_cyl["mid"])
                if b_has in slot_faces:
                    b_x = _vec_sub(slot_faces[b_has], b_cyl["mid"])
            idx[0] += 1
            slot_note = " [SLOT-x]" if (s_x or b_x) else ""
            role = _classify_match(m)
            role_note = f" [{role}]" if role not in ("shaft-bore", "bolt-verify") else ""
            print(f"  [{idx[0]}] {na}<->{nb}: R={m['shaft']['r']:.2f}{slot_note}{role_note}")
            la, lb = cylinder_labels(m, na, nb, idx[0], bore_pt, s_x, b_x, shaft_pt)
            labels[na].append(la); labels[nb].append(lb)

    # === Step 4: 用被丢弃的标签验证选中标签的正确性 ===
    verify_results = _run_verification(discarded_planar_matches, discarded_cylinder_matches)

    # === Step 4b: 碰撞检测（PRIMARY 标签）===
    collided_pairs = set()
    for (na, nb) in pairs_with_labels:
        # 找出这对零件的 PRIMARY 标签 mating ID
        gid_to_labels = {}
        for nm in (na, nb):
            for l in labels[nm]:
                gid = l["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in l["identifier"] else ""
                gid_to_labels.setdefault(gid, []).append((nm, l))
        for gid, items in gid_to_labels.items():
            if len(items) != 2: continue
            (n1, l1), (n2, l2) = items
            # 过盈配合：跳过碰撞检测（物理干涉是设计意图）
            if l1.get("userData", {}).get("interference") or l2.get("userData", {}).get("interference"):
                print(f"  [INTERFERENCE] {na}<->{nb}: intentional fit, skip collision")
                continue
            s1, s2 = _get_shape(n1), _get_shape(n2)
            if s1 is None or s2 is None: continue
            # 跳过超大文件的碰撞检测（加载太慢）
            import os as _os2
            sp1 = parts.get(n1, {}).get("shape_path","")
            sp2 = parts.get(n2, {}).get("shape_path","")
            if (_os2.path.getsize(sp1) if sp1 else 0) > 30*1024*1024: continue
            if (_os2.path.getsize(sp2) if sp2 else 0) > 30*1024*1024: continue
            # T_n1→n2 = loc(l2) * loc(l1)⁻¹（装配公式的逆）
            T_AB = _label_transform(l2, l1)
            has_coll, ratio, info = _check_collision(s1, s2, T_AB)
            if has_coll:
                print(f"  [COLLISION] {na}<->{nb}: vol={info.get('bool_vol',0):.0f}mm3"
                      f" ratio={info['bool_ratio']:.4f}")
                collided_pairs.add((na, nb))
            elif info['aabb_ratio'] > 0.001:
                tag = " [NESTED]" if info['aabb_ratio'] > 0.80 else ""
                print(f"  [FIT] {na}<->{nb}: aabb={info['aabb_ratio']:.3f}"
                      f" bool={info.get('bool_vol',0):.0f}mm3{tag}")

    # === Step 4c: 验证失败或碰撞 → 从丢弃的候选中寻找替代 ===
    # 收集所有需要回退的零件对
    fallback_pairs = set()
    for (na, nb), vdata in verify_results.items():
        cyl = vdata.get("CYL")
        if cyl is None: continue
        _, _, cyl_rate = cyl
        if cyl_rate < 0.50 or (na, nb) in collided_pairs:
            fallback_pairs.add((na, nb))
    # 也要处理 verify_results 未覆盖的碰撞对
    fallback_pairs.update(collided_pairs)

    for (na, nb) in fallback_pairs:
        # 不替换 FIF 标签（框架嵌入的物理位置是正确的）
        if (na, nb) in processed_fif or (nb, na) in processed_fif:
            continue

        cyl_info = discarded_cylinder_matches.get((na, nb))
        if cyl_info is None: continue
        best_cyl_ref, other_cyls = cyl_info

        # 候选序列：其他 PLANAR → 其他 CYL（限20个，防止碰撞检测卡死）
        candidates = []
        planar_info = discarded_planar_matches.get((na, nb))
        if planar_info:
            _, discarded_pl = planar_info
            discarded_pl_sorted = sorted(discarded_pl, key=lambda m: m["t"], reverse=True)[:10]
            for m in discarded_pl_sorted:
                candidates.append(("PLANAR", m))
        cyl_candidates = [m for m in other_cyls if not m.get("bore_to_bore", False)]
        if best_cyl_ref and not best_cyl_ref.get("bore_to_bore", False):
            cyl_candidates.append(best_cyl_ref)
        for m in cyl_candidates[:10]:
            candidates.append(("CYL", m))

        found = False
        for cand_type, cand_match in candidates:
            # 计算该候选的变换（compute 给出 nb→na，碰撞需要 na→nb）
            if cand_type == "PLANAR":
                T_cand = _compute_planar_transform(cand_match).inverse
            else:
                T_cand = _compute_cylinder_transform(cand_match).inverse

            # 过盈配合：跳过碰撞检测（设计意图的物理干涉）
            if not cand_match.get("interference"):
                s_a_fb, s_b_fb = _get_shape(na), _get_shape(nb)
            else:
                s_a_fb = s_b_fb = None
            if s_a_fb is not None and s_b_fb is not None:
                shape_moved = s_a_fb.located(T_cand)
                bb_a = shape_moved.BoundingBox(); bb_b = s_b_fb.BoundingBox()
                dx = max(0, min(bb_a.xmax, bb_b.xmax) - max(bb_a.xmin, bb_b.xmin))
                dy = max(0, min(bb_a.ymax, bb_b.ymax) - max(bb_a.ymin, bb_b.ymin))
                dz = max(0, min(bb_a.zmax, bb_b.zmax) - max(bb_a.zmin, bb_b.zmin))
                aabb_overlap = (dx * dy * dz) / max(1, (bb_a.xmax-bb_a.xmin)*(bb_a.ymax-bb_a.ymin)*(bb_a.zmax-bb_a.zmin))
                if aabb_overlap > 0.50:  # 大面积AABB交叠 → 跳过
                    print(f"    skip: AABB overlap={aabb_overlap:.2f}")
                    continue

            # CYL 轴线一致性验证
            if cand_type == "PLANAR":
                verify_ref = [best_cyl_ref] if best_cyl_ref else []
                verify_ref += [m for m in other_cyls if not m.get("bore_to_bore", False)]
                if not verify_ref:
                    verify_ref = [m for m in other_cyls]
                all_ok = True
                for cyl_m in verify_ref:
                    T_cyl = _compute_cylinder_transform(cyl_m)
                    s = cyl_m["shaft"]
                    d_a = _vec(s["dir"])
                    d_cand = _xform_dir(T_cand, d_a)
                    d_cyl = _xform_dir(T_cyl, d_a)
                    if d_cand.Length > 1e-9 and d_cyl.Length > 1e-9:
                        dot_v = max(-1.0, min(1.0,
                            d_cand.normalized().dot(d_cyl.normalized())))
                        angle = math.degrees(math.acos(abs(dot_v)))
                        if angle > 15.0:
                            all_ok = False
                            break
                if not all_ok:
                    continue
            else:
                if best_cyl_ref and cand_match is not best_cyl_ref:
                    T_ref = _compute_cylinder_transform(best_cyl_ref)
                    s = cand_match["shaft"]
                    d_a = _vec(s["dir"])
                    d_cand = _xform_dir(T_cand, d_a)
                    d_ref = _xform_dir(T_ref, d_a)
                    if d_cand.Length > 1e-9 and d_ref.Length > 1e-9:
                        dot_v = max(-1.0, min(1.0,
                            d_cand.normalized().dot(d_ref.normalized())))
                        angle = math.degrees(math.acos(abs(dot_v)))
                        if angle > 15.0:
                            continue

            found = True
            break

        if not found: continue

        # 找到候选 → 生成标签 → 最终碰撞验证 → 确认无碰撞才替换
        idx[0] += 1
        if cand_type == "PLANAR":
            la, lb = planar_labels(cand_match, na, nb, idx[0])
        else:
            s_cyl = cand_match["shaft"]; b_cyl = cand_match["bore"]
            s_has = na if cand_match["shaft_in_a"] else nb
            b_has = nb if cand_match["shaft_in_a"] else na
            bore_pt = (_bore_face_intersection(b_cyl, face_info[b_has])
                       if b_has in face_info else None)
            shaft_pt = (_bore_face_intersection(s_cyl, face_info[s_has])
                        if s_has in face_info else None)
            la, lb = cylinder_labels(cand_match, na, nb, idx[0], bore_pt, None, None, shaft_pt)

        # 快速 AABB 最终验证（过盈配合跳过）
        if not la.get("userData", {}).get("interference"):
            T_real = _label_transform(lb, la)
            s_a_r, s_b_r = _get_shape(na), _get_shape(nb)
        else:
            s_a_r = s_b_r = None
        if s_a_r is not None and s_b_r is not None:
            shape_moved = s_a_r.located(T_real)
            bb_a = shape_moved.BoundingBox(); bb_b = s_b_r.BoundingBox()
            dx = max(0, min(bb_a.xmax, bb_b.xmax) - max(bb_a.xmin, bb_b.xmin))
            dy = max(0, min(bb_a.ymax, bb_b.ymax) - max(bb_a.ymin, bb_b.ymin))
            dz = max(0, min(bb_a.zmax, bb_b.zmax) - max(bb_a.zmin, bb_b.zmin))
            aabb_overlap = (dx * dy * dz) / max(1, (bb_a.xmax-bb_a.xmin)*(bb_a.ymax-bb_a.ymin)*(bb_a.zmax-bb_a.zmin))
            if aabb_overlap > 0.80:  # >80% = 严重交叠 → 跳过
                print(f"    [COLLISION] AABB={aabb_overlap:.2f} — try next")
                continue

        # 确认无碰撞 → 移除旧标签，追加新标签
        print(f"  [FALLBACK] {na}<->{nb}: {cand_type}"
              + (f" (R={s_cyl['r']:.2f})" if cand_type == "CYL" else f" (t={cand_match['t']})")
              + " [OK]")
        planar_ids = set()
        for nm in (na, nb):
            for l in labels[nm]:
                if l.get("userData", {}).get("matchType") == "PLANAR":
                    gid = l["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in l["identifier"] else ""
                    if gid: planar_ids.add(gid)
        for gid in planar_ids:
            for nm in (na, nb):
                labels[nm] = [l for l in labels[nm]
                              if not l["identifier"].endswith(f"_Mating_{gid}")]
        labels[na].append(la); labels[nb].append(lb)
        # 继续处理 fallback_pairs 中其他碰撞对（不 break）

    return labels


if __name__ == "__main__":
    import os as _os
    # 解析 CLI: [folder] [--world-step <name>]
    target = "./2"
    world_step_arg = None
    skip_next = False
    for i, a in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if a == '--world-step' and i + 1 < len(sys.argv):
            world_step_arg = sys.argv[i + 1]
            skip_next = True
        elif not a.startswith('--'):
            target = a

    # 解析零件名
    def _resolve_part(name_hint, names):
        hint = _os.path.splitext(name_hint)[0]
        if hint in names: return hint
        matches = [n for n in names if hint.lower() in n.lower()]
        if len(matches) == 1: return matches[0]
        return None

    parts = {}
    for fp in os.listdir(target):
        if not (fp.endswith(".step") or fp.endswith(".stp")): continue
        if "virtual" in fp: continue
        nm = os.path.splitext(os.path.basename(fp))[0]
        feat_path = os.path.join(target, f"{nm}_features.json")
        if not os.path.exists(feat_path):
            print(f"  [skip] {nm}: no features (run feature_extractor first)")
            continue
        parts[nm] = {"features": json.load(open(feat_path, encoding="utf-8")),
                     "shape_path": os.path.join(target, fp)}

    # 解析 world_step（需先有 parts 才有 names）
    world_step = None
    if world_step_arg:
        world_step = _resolve_part(world_step_arg, list(parts.keys()))
        if world_step:
            print(f"world-step: {world_step}")
        else:
            print(f"  [warn] --world-step '{world_step_arg}' 未找到匹配零件，忽略")

    labels = match_all(parts, world_step=world_step)
    for nm, lst in labels.items():
        out_path = os.path.join(target, f"{nm}_label.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fmt_json(lst, world_step=world_step), f, indent=2)
        print(f"  [OK] {nm}_label.json ({len(lst)} labels)")
