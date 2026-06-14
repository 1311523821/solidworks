"""
feature_matcher.py
==================
模块2：在零件特征之间做匹配（独立运行）
用法：
  python feature_matcher.py <folder>
"""
import os, sys, json, math
import cadquery as cq
from label_generator import planar_labels, cylinder_labels, _bore_face_intersection, _neg, _vec_sub, mk_label, fmt_json

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
            mc = _match_list(a["circles"], b["circles"], "len", TOL)
            ml = _match_list(a["lines"], b["lines"], "len", TOL)
            t = len(mc) + len(ml)
            if t >= MIN_OK or (t >= MIN_OK_LOOSE and len(mc) >= 1):
                res.append({"fa": a, "fb": b, "mc": mc, "ml": ml, "t": t})
    return res


def match_slot(fa, fb):
    res = []
    for a in fa:
        for b in fb:
            mc = _match_list(a["circles"], b["circles"], "len", TOL)
            ml = _match_list(a["lines"], b["lines"], "len", TOL)
            t = len(mc) + len(ml)
            if t >= SLOT_MIN:
                res.append({"fa": a, "fb": b, "mc": mc, "ml": ml, "t": t})
    return res


def match_cylinders(ca, cb, strict_axis=False, ref_a=None, ref_b=None):
    """圆柱匹配：半径相等 + 可选轴过滤 + 可选空间偏移验证"""
    ms = []
    for a in ca:
        for b in cb:
            if a["ext"] == b["ext"]: continue
            if abs(a["r"] - b["r"]) > CYL_TOL: continue
            if strict_axis:
                dot = abs(a["dir"][0]*b["dir"][0] + a["dir"][1]*b["dir"][1] + a["dir"][2]*b["dir"][2])
                if dot < 0.9: continue
            # 空间偏移验证：仅当两面共享平面匹配（法向平行）时才做
            if ref_a and ref_b and strict_axis:
                da = sum((a["mid"][k] - ref_a[k])**2 for k in range(3))**0.5
                db = sum((b["mid"][k] - ref_b[k])**2 for k in range(3))**0.5
                if abs(da - db) > 5: continue
            ms.append({"shaft": a if a["ext"] else b,
                       "bore": b if a["ext"] else a, "shaft_in_a": a["ext"]})
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


def _frame_in_frame(na, nb, parts, labels, idx_counter, face_info, used_p):
    """
    框架嵌入匹配：一个零件的框架嵌入另一个零件的槽位。
    策略：遍历所有FAN框架面 → 与CAGE槽面匹配 → 选无碰撞的最佳组合。
    返回: True/False
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

    # 计算FAN和CAGE的z范围（用于碰撞检测）
    all_fan_faces = parts[fan_name]["features"]["planar"]
    fan_z_min = min(f["c"][2] for f in all_fan_faces)
    fan_z_max = max(f["c"][2] for f in all_fan_faces)
    all_cage_faces = parts[cage_name]["features"]["planar"]
    cage_z_min = min(f["c"][2] for f in all_cage_faces)
    cage_z_max = max(f["c"][2] for f in all_cage_faces)

    def _check_collision(fan_face, cage_face):
        """检查配对后FAN是否在CAGE内（不碰撞）"""
        fn = cq.Vector(fan_face["n"][0], fan_face["n"][1], fan_face["n"][2])
        cn = cq.Vector(cage_face["n"][0], cage_face["n"][1], cage_face["n"][2])
        # 法向必须相反
        if fn.dot(cn) > -0.5:
            return False
        # 计算Z方向偏移
        dz = cage_face["c"][2] - fan_face["c"][2]
        fan_top_world = fan_z_max + dz
        fan_bot_world = fan_z_min + dz
        # FAN必须在CAGE z范围内
        if fan_top_world < cage_z_max and fan_bot_world > cage_z_min:
            return True  # 无碰撞
        return False  # 有碰撞

    # 按z从高到低排序FAN框架面（优先选顶部面→FAN嵌入更深）
    fan_frames = sorted(fan_frames_init, key=lambda f: f["c"][2], reverse=True)

    # 对每个FAN框架面，尝试匹配所有CAGE槽面（一对多：每个槽一个FAN）
    found_any = False
    face_info_note = {}
    used_xy = set()  # 去重：同一XY位置只保留最佳匹配
    for fan_frame in fan_frames:
        fan_center = cq.Vector(fan_frame["c"][0], fan_frame["c"][1], fan_frame["c"][2])

        cage_candidates = []
        for f in cage_slots:
            cn = cq.Vector(f["n"][0], f["n"][1], f["n"][2])
            fn = cq.Vector(fan_frame["n"][0], fan_frame["n"][1], fan_frame["n"][2])
            if fn.dot(cn) < -0.5:
                cage_candidates.append(f)
        if not cage_candidates:
            cage_candidates = cage_slots

        cage_candidates.sort(key=lambda f: (f["c"][0] - fan_center.x)**2 + (f["c"][1] - fan_center.y)**2)

        # 收集每个槽位的最佳匹配，按XY去重
        slot_matches = []  # (xy_key, score, fan_frame, cage_cand, ml, mc)
        for cage_cand in cage_candidates:
            ml = _match_lines_spatial(fan_frame.get("lines", []), cage_cand.get("lines", []),
                                      fan_frame["c"], cage_cand["c"], len_tol=20.0)
            if len(ml) < 1:
                continue
            mc = _match_list(fan_frame.get("circles", []), cage_cand.get("circles", []), "len", 1.0)
            if not _check_collision(fan_frame, cage_cand):
                continue
            x_key = round(cage_cand["c"][0] / 10) * 10  # 10mm精度分组
            y_key = round(cage_cand["c"][1] / 10) * 10
            xy_key = (x_key, y_key)
            score = len(ml) * 10 + len(mc)
            # 同一位置只保留最高分
            replaced = False
            for i, (ek, es, _, _, _, _) in enumerate(slot_matches):
                if ek == xy_key:
                    if score > es:
                        slot_matches[i] = (xy_key, score, fan_frame, cage_cand, ml, mc)
                    replaced = True
                    break
            if not replaced:
                slot_matches.append((xy_key, score, fan_frame, cage_cand, ml, mc))

        # 为每个去重后的槽位生成面标签
        for xy_key, score, ff, cage_cand, ml, mc in slot_matches:
            t = len(mc) + len(ml)
            m = {"fa": ff, "fb": cage_cand, "mc": mc, "ml": ml, "t": t}
            if fan_name != na:
                na_real, nb_real = fan_name, cage_name
            else:
                na_real, nb_real = na, nb

            idx = idx_counter[0]
            idx_counter[0] += 1
            la, lb = planar_labels(m, na_real, nb_real, idx)
            labels[na_real].append(la)
            labels[nb_real].append(lb)

            if na_real not in face_info:
                face_info[na_real] = {"c": ff["c"], "n": ff["n"]}
            if nb_real not in face_info:
                face_info[nb_real] = {"c": cage_cand["c"], "n": cage_cand["n"]}

            print(f"  [{idx}] {na_real}<->{nb_real}: {len(mc)}c+{len(ml)}l"
                  f" @({cage_cand['c'][0]:.0f},{cage_cand['c'][1]:.0f}) [FRAME-IN-FRAME]")
            found_any = True

        break  # 只用第一个匹配的FAN框架面


# ========== 主入口 ==========
def match_all(parts):
    """
    parts: {name: {"features": {...}, "shape_path": "..."}}
    返回: (labels_by_part, stats)
    """
    names = list(parts.keys())
    labels = {n: [] for n in names}
    idx = [0]

    def _fk(f):
        c, n = f["c"], f["n"]
        return f"p|{c[0]:.4f}|{c[1]:.4f}|{c[2]:.4f}|{n[0]:.4f}|{n[1]:.4f}|{n[2]:.4f}"

    face_info = {}; used_p = {}; slot_faces = {}; face_csys = {}
    ky_name = next((n for n in names if "key" in n.lower()), None)
    sh_name = next((n for n in names if "shaft" in n.lower()), None)

    # === Step 0: 检测多圆柱零件对 → 框架嵌入匹配 ===
    multi_cyl_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ca = parts[names[i]]["features"]["cylinders"]
            cb = parts[names[j]]["features"]["cylinders"]
            if len(ca) > MULTI_CYL_THRESHOLD and len(cb) > MULTI_CYL_THRESHOLD:
                multi_cyl_pairs.append((names[i], names[j]))

    processed_fif = set()
    for na, nb in multi_cyl_pairs:
        before_labels = sum(len(labels[n]) for n in [na, nb])
        _frame_in_frame(na, nb, parts, labels, idx, face_info, used_p)
        after_labels = sum(len(labels[n]) for n in [na, nb])
        if after_labels > before_labels:
            processed_fif.add((na, nb))
            print(f"  [OK] frame-in-frame: {after_labels - before_labels} new labels for {na}<->{nb}")
        else:
            print(f"  [warn] frame-in-frame failed for {na}<->{nb}, fallback to planar")

    # === Step 1: planar ===
    planar_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            if (na, nb) in processed_fif:
                continue  # 已通过框架嵌入处理
            if ky_name and (ky_name in (na, nb) and any("flange" in x for x in (na, nb))):
                continue
            ms = match_planar(parts[na]["features"]["planar"], parts[nb]["features"]["planar"])
            if ms:
                ms.sort(key=lambda m: m["t"], reverse=True)
                planar_pairs.append((na, nb, ms))
    planar_pairs.sort(key=lambda x: x[2][0]["t"], reverse=True)

    for na, nb, ms in planar_pairs:
        for m in ms:
            ka, kb = _fk(m["fa"]), _fk(m["fb"])
            if ka in used_p.get(na, set()) or kb in used_p.get(nb, set()): continue
            used_p.setdefault(na, set()).add(ka)
            used_p.setdefault(nb, set()).add(kb)
            idx[0] += 1
            # outward check
            from cadquery import importers
            shape_a = importers.importStep(parts[na]["shape_path"]).val()
            shape_b = importers.importStep(parts[nb]["shape_path"]).val()
            bb_a = shape_a.BoundingBox(); bb_b = shape_b.BoundingBox()
            bc_a = cq.Vector((bb_a.xmin+bb_a.xmax)/2, (bb_a.ymin+bb_a.ymax)/2, (bb_a.zmin+bb_a.zmax)/2)
            bc_b = cq.Vector((bb_b.xmin+bb_b.xmax)/2, (bb_b.ymin+bb_b.ymax)/2, (bb_b.zmin+bb_b.zmax)/2)
            fc_a = cq.Vector(m["fa"]["c"][0], m["fa"]["c"][1], m["fa"]["c"][2])
            fc_b = cq.Vector(m["fb"]["c"][0], m["fb"]["c"][1], m["fb"]["c"][2])
            n_a = cq.Vector(m["fa"]["n"][0], m["fa"]["n"][1], m["fa"]["n"][2])
            n_b = cq.Vector(m["fb"]["n"][0], m["fb"]["n"][1], m["fb"]["n"][2])
            if fc_a.sub(bc_a).dot(n_a) < 0 or fc_b.sub(bc_b).dot(n_b) < 0: continue
            # multi-hole filter
            ca = len(parts[na]["features"]["cylinders"])
            cb = len(parts[nb]["features"]["cylinders"])
            if ca > 50 and cb > 50 and abs(n_a.dot(n_b)) < 0.7: continue
            print(f"  [{idx[0]}] {na} <-> {nb}: {len(m['mc'])}c+{len(m['ml'])}l")
            la, lb = planar_labels(m, na, nb, idx[0])
            labels[na].append(la); labels[nb].append(lb)
            if na not in face_info:
                face_info[na] = {"c": m["fa"]["c"], "n": m["fa"]["n"]}
                face_csys[na] = {"c": m["fa"]["c"], "n": m["fa"]["n"],
                                 "x": la["geometry"]["x"], "y": la["geometry"]["y"]}
            if nb not in face_info:
                face_info[nb] = {"c": m["fb"]["c"], "n": m["fb"]["n"]}
                face_csys[nb] = {"c": m["fb"]["c"], "n": m["fb"]["n"],
                                 "x": lb["geometry"]["x"], "y": lb["geometry"]["y"]}
            break

    # === Step 2: slot ===
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
                    idx[0] += 1
                    print(f"  [{idx[0]}] {sh_name} <-> {fl_name}: {len(m['mc'])}c+{len(m['ml'])}l [SLOT]")
                    la, lb = planar_labels(m, sh_name, fl_name, idx[0])
                    labels[sh_name].append(la); labels[fl_name].append(lb)
                    if fl_name not in face_info:
                        face_info[fl_name] = {"c": m["fb"]["c"], "n": m["fb"]["n"]}
                    slot_faces[fl_name] = m["fb"]["c"]
                    slot_faces[sh_name] = m["fa"]["c"]
                    break

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
            # FIF对不用空间偏移验证（面中心z差36mm），列过滤替代
            ms = match_cylinders(ca, cb, strict_axis=strict,
                                 ref_a=(None if is_fif else ref_a),
                                 ref_b=(None if is_fif else ref_b))
            # 框架嵌入对：CAGE侧圆柱的XY必须接近CAGE槽面中心（过滤跨列误匹配）
            if is_fif and ms:
                cage_name = na if len(ca) > len(cb) else nb
                fc = face_info[cage_name]["c"]
                ms_filtered = []
                for m in ms:
                    cage_cyl = m["shaft"] if (m["shaft_in_a"] and na == cage_name) or (not m["shaft_in_a"] and nb == cage_name) else m["bore"]
                    dxy = ((cage_cyl["mid"][0] - fc[0])**2 + (cage_cyl["mid"][1] - fc[1])**2)**0.5
                    if dxy < 20:  # 20mm内才算同列
                        ms_filtered.append(m)
                ms = ms_filtered
            if ms: cyl_pairs.append((na, nb, ms))

    def _ck(cyl):
        m = cyl["mid"]
        return f"c|{cyl['r']:.4f}|{m[0]:.4f}|{m[1]:.4f}|{m[2]:.4f}"

    # 按半径降序排列：大框架/主轴优先
    for pair in cyl_pairs:
        pair[2].sort(key=lambda m: m["shaft"]["r"], reverse=True)

    used_c = {}
    for na, nb, ms in cyl_pairs:
        for m in ms:
            s_has = na if m["shaft_in_a"] else nb
            b_has = nb if m["shaft_in_a"] else na
            kb = _ck(m["bore"])
            if kb in used_c.get(b_has, set()): continue
            used_c.setdefault(b_has, set()).add(kb)
            # 孔/轴都投影到配合面上（避免轴中点差造成z偏移）
            bore_pt = None; shaft_pt = None
            if b_has in face_info:
                bore_pt = _bore_face_intersection(m["bore"], face_info[b_has])
            if s_has in face_info:
                shaft_pt = _bore_face_intersection(m["shaft"], face_info[s_has])
            s_cyl = m["shaft"]; b_cyl = m["bore"]
            s_x = None; b_x = None
            if s_has in slot_faces:
                s_x = _vec_sub(slot_faces[s_has], s_cyl["mid"])
            if b_has in slot_faces:
                b_x = _vec_sub(slot_faces[b_has], b_cyl["mid"])
            idx[0] += 1
            print(f"  [{idx[0]}] {na}<->{nb}: R={m['shaft']['r']:.2f}")
            la, lb = cylinder_labels(m, na, nb, idx[0], bore_pt, s_x, b_x, shaft_pt)
            labels[na].append(la); labels[nb].append(lb)

    return labels


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./2"
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

    labels = match_all(parts)
    for nm, lst in labels.items():
        out_path = os.path.join(target, f"{nm}_label.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fmt_json(lst), f, indent=2)
        print(f"  [OK] {nm}_label.json ({len(lst)} labels)")
