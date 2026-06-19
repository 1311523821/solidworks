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
# 标准机械尺寸 + 0.5mm 公差带
_STANDARD_RADII = sorted([
    0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0,
    12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 30.0,
    33.0, 36.0, 39.0, 42.0
])
_BUCKET_TOL = 0.5  # 标准尺寸容差


def quantize_radius(r):
    """将圆柱半径归入最近的机械标准尺寸桶，返回桶标签字符串"""
    if r <= 0:
        return f"R{round(r, 1):.1f}"
    # 查找最近的标准尺寸
    for std in _STANDARD_RADII:
        if abs(r - std) <= _BUCKET_TOL:
            return f"M{std:.1f}".replace(".0", "")
    # 非标准尺寸：按 1mm 粒度分桶
    bucket = round(r)
    return f"R{bucket}"


def _pt(face):
    es = face.Edges(); return es[0].startPoint() if es else face.Center()


# ==================== planar ====================
def _sort_by_angle(items, center, normal, key="c"):
    """按角度排序——用最远点作参考方向，确保不同面坐标系一致"""
    z = cq.Vector(*normal)
    oc = cq.Vector(*center)
    if len(items) < 2:
        return items
    # 用面内距离中心最远的点作参考方向（面自身几何特征，跨零件一致）
    best_d = -1; best_pt = None
    for it in items:
        pt = cq.Vector(*it[key]).sub(oc)
        pt_proj = pt - z * pt.dot(z)
        d = pt_proj.Length
        if d > best_d:
            best_d = d; best_pt = pt_proj
    x_ref = best_pt.normalized() if best_pt and best_pt.Length > 1e-9 else cq.Vector(1,0,0)
    x_ref = x_ref - z * (x_ref.dot(z) / z.dot(z))
    if x_ref.Length < 1e-9:
        x_ref = cq.Vector(0, 1, 0) - z * (z.y / z.dot(z))
    x_ref = x_ref.normalized()
    y_ref = z.cross(x_ref)
    def _ang(it):
        pt = cq.Vector(*it[key]).sub(oc)
        pt_proj = pt - z * pt.dot(z)
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
            # 按角度排序，保证跨零件匹配顺序一致（法兰孔对齐关键）
            if len(cs) > 1:
                cs = _sort_by_angle(cs, [c.x, c.y, c.z], [n.x, n.y, n.z], "c")
            if len(ls) > 1:
                ls = _sort_by_angle(ls, [c.x, c.y, c.z], [n.x, n.y, n.z], "m")
            out.append({
                "c": [c.x, c.y, c.z], "n": [n.x, n.y, n.z],
                "circles": cs, "lines": ls,
                "area": round(area, 1),
                "n_edges": len(cs) + len(ls)
            })
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
            out.append({
                "r": round(r, 4), "ext": ext,
                "mid": [mid.x, mid.y, mid.z],
                "dir": [d.x, d.y, d.z],
                "ends": ends,
                "bucket": quantize_radius(r)
            })
        except Exception: pass
    # 排序圆柱：确保跨零件匹配顺序一致（螺栓孔对齐关键）
    out.sort(key=lambda c: (
        c["ext"], c["r"],
        round(c["dir"][0], 2), round(c["dir"][1], 2), round(c["dir"][2], 2),
        round(c["mid"][0], 4), round(c["mid"][1], 4), round(c["mid"][2], 4)))
    return out


# ==================== summary ====================
def summary(features):
    """特征统计"""
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
        "max_face_area": max((f["area"] for f in p), default=0)
    }


# ==================== main ====================
def extract_file(filepath):
    """提取单个 STEP 文件的特征"""
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

        # 检查缓存
        if os.path.exists(out_path):
            cached = json.load(open(out_path, encoding="utf-8"))
            age = os.path.getmtime(fp) - os.path.getmtime(out_path)
            if age < 0:  # STEP 比缓存新
                pass
            else:
                print(f"  [cache] {nm}: {cached['_summary']['n_planar']}p {cached['_summary']['n_cylinders']}c")
                continue

        print(f"  [extracting] {nm}...", end=" ", flush=True)
        features = extract_file(fp)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
        s = features["_summary"]
        print(f"{s['n_planar']}p {s['n_cylinders']}c")
    print("done")
