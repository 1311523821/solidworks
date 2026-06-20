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
from cadquery import importers
from OCP.BRepAdaptor import BRepAdaptor_Surface

TOL = 0.1; CYL_TOL = 0.5
MIN_FACE_AREA = 50
MIN_CYL_R_RATIO = 0.005  # 圆柱半径 < 对角线*0.5% 跳过

# ==================== 半径分桶（倒排索引） ====================
_STANDARD_RADII = sorted([
    0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0,
    12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 30.0,
    33.0, 36.0, 39.0, 42.0
])
_BUCKET_TOL = 0.5


def quantize_radius(r):
    if r <= 0: return f"R{round(r, 1):.1f}"
    for std in _STANDARD_RADII:
        if abs(r - std) <= _BUCKET_TOL:
            return f"M{std:.1f}".replace(".0", "")
    return f"R{round(r)}"


def _pt(face):
    es = face.Edges(); return es[0].startPoint() if es else face.Center()


# ==================== planar + 面形描述符 ====================
def _face_shape_descriptors(face_data):
    """计算面的形状描述符：矩形度、长宽比、是否为槽口"""
    lines = face_data.get("lines", [])
    circles = face_data.get("circles", [])
    area = face_data.get("area", 0)
    result = {"aspect_ratio": 1.0, "rect_score": 0.0, "is_slot": False}
    if len(lines) >= 4:
        lens = sorted([l["len"] for l in lines], reverse=True)
        top4 = lens[:4]
        if min(top4) > 1:
            result["aspect_ratio"] = round(top4[0] / max(top4[2], 1), 2)
            # 矩形度：两对等边差 < 20%
            pair1_ok = abs(top4[0] - top4[2]) < max(top4[0] * 0.2, 2.0)
            pair2_ok = abs(top4[1] - top4[3]) < max(top4[1] * 0.2, 2.0)
            if pair1_ok and pair2_ok:
                result["rect_score"] = min(1.0, abs(top4[0] - top4[2]) / max(top4[0], 1) +
                                              abs(top4[1] - top4[3]) / max(top4[1], 1))
            # 长槽检测：长宽比 > 3 且面积适中的面
            if result["aspect_ratio"] > 3 and 50 < area < 1000:
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

    # 引脚阵列候选：> 4 个同半径圆的面
    if len(circles) >= 4:
        by_r = {}
        for c in circles:
            rk = round(c["r"], 1)
            by_r.setdefault(rk, []).append(c)
        for rk, clist in by_r.items():
            if len(clist) >= 4:
                if "pin_arrays" not in result:
                    result["pin_arrays"] = []
                result["pin_arrays"].append({
                    "radius": round(rk, 1), "count": len(clist),
                    "bucket": quantize_radius(rk)
                })
    return result


def _sort_by_angle(items, center, normal, key="c"):
    z = cq.Vector(*normal)
    oc = cq.Vector(*center)
    if len(items) < 2: return items
    best_d = -1; best_pt = None
    for it in items:
        pt = cq.Vector(*it[key]).sub(oc)
        pt_proj = pt - z * pt.dot(z); d = pt_proj.Length
        if d > best_d: best_d = d; best_pt = pt_proj
    x_ref = best_pt.normalized() if best_pt and best_pt.Length > 1e-9 else cq.Vector(1,0,0)
    x_ref = x_ref - z * (x_ref.dot(z) / z.dot(z))
    if x_ref.Length < 1e-9:
        x_ref = cq.Vector(0, 1, 0) - z * (z.y / z.dot(z))
    x_ref = x_ref.normalized()
    y_ref = z.cross(x_ref)
    def _ang(it):
        pt = cq.Vector(*it[key]).sub(oc); pt_proj = pt - z * pt.dot(z)
        return math.atan2(pt_proj.dot(y_ref), pt_proj.dot(x_ref))
    return sorted(items, key=_ang)


def extract_planar(shape):
    bb = shape.val().BoundingBox()
    out = []
    for f in shape.faces().vals():
        area = f.Area()
        if area < MIN_FACE_AREA: continue
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
                "z_dominant": abs(n.z) > 0.7
            }
            # 附加形状描述符
            face_data.update(_face_shape_descriptors(face_data))
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
    if not loc: return False
    pt = _pt(face); n = face.normalAt(pt)
    v = pt.sub(loc).sub(d.multiply(pt.sub(loc).dot(d)))
    return v.dot(n) > 0


def extract_cylinders(shape):
    bb = shape.val().BoundingBox()
    diag = ((bb.xmax-bb.xmin)**2 + (bb.ymax-bb.ymin)**2 + (bb.zmax-bb.zmin)**2)**0.5
    min_r = diag * MIN_CYL_R_RATIO
    out = []
    for f in shape.faces("%Cylinder").vals():
        try:
            loc, d, r = _cyl_geom(f)
            if r < min_r: continue
            ext = _is_ext(f)
            c = f.Center(); mid = loc.add(d.multiply(c.sub(loc).dot(d)))
            ends = [[e.Center().x, e.Center().y, e.Center().z]
                    for e in f.Edges() if e.geomType() == "CIRCLE"]
            # 轴向归一化位置：0=底面端, 1=顶面端（区分端面孔 vs 中段孔）
            if abs(d.z) > 0.7 and (bb.zmax - bb.zmin) > 0:
                axial_pos = (mid.z - bb.zmin) / (bb.zmax - bb.zmin)
            else:
                axial_pos = -1  # 非 Z 向圆柱不计算
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
                    radial = ((ep[0]-mid.x)**2 + (ep[1]-mid.y)**2 + (ep[2]-mid.z)**2)**0.5
                    axial = abs((ep[0]-mid.x)*d.x + (ep[1]-mid.y)*d.y + (ep[2]-mid.z)*d.z)
                    end_rs.append((radial, axial))
                end_rs.sort(key=lambda x: x[1])
                if len(end_rs) >= 2 and abs(end_rs[0][1] - end_rs[-1][1]) > 1:
                    r1 = end_rs[0][0]; r2 = end_rs[-1][0]
                    length = abs(end_rs[-1][1] - end_rs[0][1])
                    if length > 0 and abs(r1 - r2) > 0.01:
                        out[-1]["taper"] = round(abs(r1 - r2) / length, 6)
                        out[-1]["is_tapered"] = True
        except Exception: pass
    out.sort(key=lambda c: (
        c["ext"], c["r"],
        round(c["dir"][0], 2), round(c["dir"][1], 2), round(c["dir"][2], 2),
        round(c["mid"][0], 4), round(c["mid"][1], 4), round(c["mid"][2], 4)))
    return out


# ==================== fingerprint ====================
def _compute_fingerprint(planar, cylinders):
    """零件级指纹：Z面、引脚阵列、槽口、止口、花键、锥度、圆柱桶"""
    fp = {"z_faces": [], "socket_candidates": [], "slot_candidates": [],
          "pin_arrays": [], "spigot_candidates": [], "spline_groups": [],
          "tapered_cyls": [], "cyl_buckets": {}}

    for f in planar:
        if f.get("z_dominant"):
            fp["z_faces"].append({
                "area": f["area"], "z": round(f["c"][2], 1),
                "n_circles": len(f.get("circles", [])),
                "n_lines": len(f.get("lines", [])),
                "inter_circle_dists": f.get("inter_circle_dists", [])
            })
        for pa in f.get("pin_arrays", []):
            fp["pin_arrays"].append({
                "radius": pa["radius"], "count": pa["count"],
                "bucket": pa["bucket"],
                "face_area": f["area"], "face_z": round(f["c"][2], 1),
                "z_dominant": f.get("z_dominant", False)
            })
        if f.get("is_slot"):
            fp["slot_candidates"].append({
                "area": f["area"],
                "aspect_ratio": f.get("aspect_ratio", 1),
                "z": round(f["c"][2], 1),
                "z_dominant": f.get("z_dominant", False),
                "n_circles": len(f.get("circles", []))
            })

    # 止口检测：大半径(>15mm) + 短轴 + 有台阶面（同轴不同半径的圆柱面）
    large_cyls = [c for c in cylinders if c["r"] > 15]
    for c in large_cyls:
        # 查找同轴的更大/更小圆柱（台阶特征）
        same_axis = [o for o in cylinders if o is not c and
            abs(c["dir"][0]*o["dir"][0] + c["dir"][1]*o["dir"][1] + c["dir"][2]*o["dir"][2]) > 0.9]
        bigger = any(o["r"] > c["r"] + 2 for o in same_axis)
        smaller = any(o["r"] < c["r"] - 2 for o in same_axis)
        if bigger or smaller:
            fp["spigot_candidates"].append({
                "radius": c["r"], "bucket": c.get("bucket", "?"),
                "has_step": True, "ext": c["ext"]
            })

    # 花键检测：≥3 个同半径、同向、同轴的圆柱（内外花键）
    by_bucket_axis = {}
    for c in cylinders:
        bk = c.get("bucket", f"R{round(c['r'],1)}")
        ax_key = (bk, round(c["dir"][0],1), round(c["dir"][1],1), round(c["dir"][2],1))
        by_bucket_axis.setdefault(ax_key, []).append(c)
    for ax_key, group in by_bucket_axis.items():
        if len(group) >= 3:
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

    # 同心面检测：同XY(±5mm)不同Z(>2mm)的面组 = 台阶/止口/槽位
    by_xy = {}
    for f in planar:
        if not f.get("z_dominant"): continue
        xy_key = (round(f["c"][0] / 5) * 5, round(f["c"][1] / 5) * 5)
        by_xy.setdefault(xy_key, []).append(f)
    for xy_key, group in by_xy.items():
        zs = sorted([g["c"][2] for g in group])
        if len(zs) >= 3 and zs[-1] - zs[0] > 2:
            fp["concentric_step_groups"] = fp.get("concentric_step_groups", 0) + 1

    # 圆柱桶分布 + 端面标记
    for c in cylinders:
        bk = c.get("bucket", f"R{round(c['r'],1)}")
        fp["cyl_buckets"][bk] = fp["cyl_buckets"].get(bk, 0) + 1
    # 统计 Z 向端面圆柱（axial_pos ≈ 0 或 ≈ 1）
    end_cyls = [c for c in cylinders if c.get("axial_pos", -1) >= 0
                and (c["axial_pos"] < 0.2 or c["axial_pos"] > 0.8)]
    if end_cyls:
        fp["end_face_cyls"] = len(end_cyls)

    return fp


# ==================== summary ====================
def summary(features):
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
        "fingerprint": _compute_fingerprint(p, c)
    }


# ==================== main ====================
def extract_file(filepath):
    shape = importers.importStep(filepath)
    planar = extract_planar(shape)
    cylinders = extract_cylinders(shape)
    features = {"planar": planar, "cylinders": cylinders}
    features["_summary"] = summary(features)
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
            cached = json.load(open(out_path, encoding="utf-8"))
            # 检查是否有新字段（fingerprint），无则重新提取
            if cached.get("_summary", {}).get("fingerprint"):
                print(f"  [cache] {nm}: {cached['_summary']['n_planar']}p {cached['_summary']['n_cylinders']}c")
                continue

        print(f"  [extracting] {nm}...", end=" ", flush=True)
        features = extract_file(fp)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
        s = features["_summary"]
        print(f"{s['n_planar']}p {s['n_cylinders']}c")
    print("done")
