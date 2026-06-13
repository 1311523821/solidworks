"""
auto_labeler.py
==============
1. 平面面匹配（边长相等）→ 法兰面贴面、键入轴槽
2. 槽口匹配（孔轴过滤 + 宽松阈值）→ 轴槽↔法兰槽（X方向约束）
3. 圆柱面匹配（半径相等）→ 轴入法兰孔（原点=孔轴×法兰面交点）

X 方向：轴心→面中心（径向向量，无 180° 歧义）
"""
import os, sys, glob, json, math
import cadquery as cq
from cadquery import importers
from OCP.BRepAdaptor import BRepAdaptor_Surface

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

TOL = 0.1; MIN_OK = 3; MIN_OK_LOOSE = 2; CYL_TOL = 0.5; SLOT_MIN = 2


def _pt(face):
    es = face.Edges(); return es[0].startPoint() if es else face.Center()

def _ortho(z_dict, x_hint=None):
    z = cq.Vector(z_dict["x"], z_dict["y"], z_dict["z"])
    if x_hint:
        ref = cq.Vector(x_hint["x"], x_hint["y"], x_hint["z"])
    else:
        ref = cq.Vector(1, 0, 0) if abs(z.x) < 0.9 else cq.Vector(0, 1, 0)
    x = ref - z * (ref.dot(z) / z.dot(z))
    x = cq.Vector(1, 0, 0) if x.Length < 1e-9 else x.normalized()
    y = z.cross(x).normalized()
    return {"x": {"x": x.x, "y": x.y, "z": x.z},
            "y": {"y": y.y, "x": y.x, "z": y.z},
            "z": {"x": z.x, "y": z.y, "z": z.z}}

def _neg(v): return {"x": -v["x"], "y": -v["y"], "z": -v["z"]}
def _vec_sub(a, b): return {"x": a["x"] - b["x"], "y": a["y"] - b["y"], "z": a["z"] - b["z"]}

def mk_label(ident, orig, z_dir, extra=None, x_hint=None):
    ud = {"type": "MATE"}
    if extra: ud.update(extra)
    return {"identifier": ident, "name": ident, "label": "ReferenceSys",
            "geometry": {"origin": orig, **_ortho(z_dir, x_hint)}, "userData": ud}


# ========== planar ==========
def extract_planar(shape):
    out = []
    for f in shape.faces().vals():
        cs, ls = [], []
        for e in f.Edges():
            if e.geomType() == "CIRCLE":
                cs.append({"len": round(e.Length(),6),
                           "c": {"x": e.Center().x, "y": e.Center().y, "z": e.Center().z}})
            elif e.geomType() == "LINE":
                ls.append({"len": round(e.Length(),6),
                           "m": {"x": e.Center().x, "y": e.Center().y, "z": e.Center().z}})
        if cs or ls:
            c = f.Center(); n = f.normalAt(_pt(f))
            out.append({"c": {"x": c.x, "y": c.y, "z": c.z},
                        "n": {"x": n.x, "y": n.y, "z": n.z},
                        "circles": cs, "lines": ls,
                        "area": round(f.Area(), 1)})
    return out

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

def planar_labels(m, na, nb, idx):
    fa, fb = m["fa"], m["fb"]
    oa, ob_ = fa["c"], fb["c"]
    meta = {"matchType": "PLANAR", "total": m["t"]}
    if m["mc"]:
        xa = _vec_sub(m["mc"][0][0]["c"], oa)
        xb = _vec_sub(m["mc"][0][1]["c"], ob_)
    elif m["ml"]:
        xa = _vec_sub(m["ml"][0][0]["m"], oa)
        xb = _vec_sub(m["ml"][0][1]["m"], ob_)
    else:
        xa = xb = None
    return (mk_label(f"{na}_Mating_{idx}", oa, fa["n"], meta, xa),
            mk_label(f"{nb}_Mating_{idx}", ob_, _neg(fb["n"]), meta, xb))


# ========== cylinder ==========
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
    cyls = []
    for f in shape.faces("%Cylinder").vals():
        try:
            loc, d, r = _cyl_geom(f); ext = _is_ext(f)
            c = f.Center(); mid = loc.add(d.multiply(c.sub(loc).dot(d)))
            ends = [{"x": e.Center().x, "y": e.Center().y, "z": e.Center().z}
                    for e in f.Edges() if e.geomType() == "CIRCLE"]
            cyls.append({"r": round(r,4), "ext": ext,
                         "mid": {"x": mid.x, "y": mid.y, "z": mid.z},
                         "dir": {"x": d.x, "y": d.y, "z": d.z}, "ends": ends})
        except Exception: pass
    return cyls

def match_cylinders(ca, cb):
    ms = []
    for a in ca:
        for b in cb:
            if a["ext"] == b["ext"]: continue
            if abs(a["r"] - b["r"]) < CYL_TOL:
                ms.append({"shaft": a if a["ext"] else b,
                           "bore": b if a["ext"] else a,
                           "shaft_in_a": a["ext"]})
    return ms

def cylinder_labels(m, na, nb, idx, bore_origin=None, shaft_x=None, bore_x=None):
    s, b = m["shaft"], m["bore"]
    meta = {"matchType": "CYLINDER", "radius": s["r"]}
    o_shaft = s["mid"]
    o_bore = bore_origin if bore_origin else b["mid"]
    shaft_nm = na if m["shaft_in_a"] else nb
    bore_nm  = nb if m["shaft_in_a"] else na
    la = mk_label(f"{shaft_nm}_Mating_{idx}", o_shaft, s["dir"], meta, shaft_x)
    lb = mk_label(f"{bore_nm}_Mating_{idx}", o_bore, b["dir"], meta, bore_x)
    out = {shaft_nm: la, bore_nm: lb}
    return out[na], out[nb]

def _bore_face_intersection(bore_cyl, face):
    d = cq.Vector(bore_cyl["dir"]["x"], bore_cyl["dir"]["y"], bore_cyl["dir"]["z"])
    n = cq.Vector(face["n"]["x"], face["n"]["y"], face["n"]["z"])
    c = cq.Vector(face["c"]["x"], face["c"]["y"], face["c"]["z"])
    m = cq.Vector(bore_cyl["mid"]["x"], bore_cyl["mid"]["y"], bore_cyl["mid"]["z"])
    denom = d.dot(n)
    if abs(denom) < 1e-9: return bore_cyl["mid"]
    t = (c.sub(m)).dot(n) / denom
    pt = m.add(d.multiply(t))
    return {"x": pt.x, "y": pt.y, "z": pt.z}

def _bore_filter(cylinders, planar_faces):
    # 取最大内孔（非硬编码 R 范围）
    internal = [c for c in cylinders if not c["ext"]]
    if not internal: return planar_faces
    bore = max(internal, key=lambda c: c["r"])
    d = cq.Vector(bore["dir"]["x"], bore["dir"]["y"], bore["dir"]["z"])
    mid = cq.Vector(bore["mid"]["x"], bore["mid"]["y"], bore["mid"]["z"])
    r = bore["r"]
    out = []
    for f in planar_faces:
        fc = cq.Vector(f["c"]["x"], f["c"]["y"], f["c"]["z"])
        radial = fc.sub(mid).sub(d.multiply(fc.sub(mid).dot(d)))
        fn = cq.Vector(f["n"]["x"], f["n"]["y"], f["n"]["z"])
        # 面在孔壁附近（r±10mm）且法向⊥孔轴
        if abs(radial.Length - r) < 15 and abs(fn.dot(d)) < 0.3:
            out.append(f)
    return out if out else planar_faces


def _shaft_keyway_filter(cylinders, planar_faces):
    cyl = next((c for c in cylinders if c["ext"]), None)
    if not cyl: return planar_faces
    d = cq.Vector(cyl["dir"]["x"], cyl["dir"]["y"], cyl["dir"]["z"])
    mid = cq.Vector(cyl["mid"]["x"], cyl["mid"]["y"], cyl["mid"]["z"])
    r = cyl["r"]
    out = []
    for f in planar_faces:
        fc = cq.Vector(f["c"]["x"], f["c"]["y"], f["c"]["z"])
        radial = fc.sub(mid).sub(d.multiply(fc.sub(mid).dot(d)))
        # 面在轴表面附近（r±5mm）且至少有 4 条线
        if abs(radial.Length - r) < 8 and len(f["lines"]) >= 4:
            out.append(f)
    return out if out else planar_faces

def _shaft_keyway_filter(cylinders, planar_faces):
    cyl = next((c for c in cylinders if c["ext"]), None)
    if not cyl: return planar_faces
    d = cq.Vector(cyl["dir"]["x"], cyl["dir"]["y"], cyl["dir"]["z"])
    mid = cq.Vector(cyl["mid"]["x"], cyl["mid"]["y"], cyl["mid"]["z"])
    out = []
    for f in planar_faces:
        fc = cq.Vector(f["c"]["x"], f["c"]["y"], f["c"]["z"])
        radial = fc.sub(mid).sub(d.multiply(fc.sub(mid).dot(d)))
        if 10 < radial.Length < 25 and len(f["lines"]) >= 4:
            out.append(f)
    return out if out else planar_faces


def fmt_json(lst):
    return {"modelAnnotation": {"parameters": {},
        "features": {"featurePoints": [], "featureLines": [], "featurePlanes": [],
                     "featureCoordSyses": lst, "featureSurfaces": [], "featureBodies": []},
        "children": []}}


# ========== main ==========
if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(root, './2')
    fps = list(set(os.path.normpath(f) for f in
        glob.glob(os.path.join(folder, "*.step")) + glob.glob(os.path.join(folder, "*.stp"))
        if "virtual" not in os.path.basename(f)))
    if len(fps) < 2: print("[ER] >=2 files"); exit(1)

    print("=== Step 1 ===")
    parts = {}; names = []
    for fp in fps:
        nm = os.path.splitext(os.path.basename(fp))[0]
        try:
            sh = importers.importStep(fp)
            parts[nm] = {"shape": sh, "planar": extract_planar(sh), "cyls": extract_cylinders(sh)}
            names.append(nm)
            print(f"  [OK] {nm}: {len(parts[nm]['planar'])}p {len(parts[nm]['cyls'])}c")
        except Exception as e: print(f"  [ER] {nm}: {e}")

    labels = {n: [] for n in names}; idx = 0
    def _fk(f): return f"p|{f['c']['x']:.4f}|{f['c']['y']:.4f}|{f['c']['z']:.4f}|{f['n']['x']:.4f}|{f['n']['y']:.4f}|{f['n']['z']:.4f}"
    face_info = {}; used_p = {}; slot_faces = {}  # slot_faces: part_name -> face center for X direction
    ky_name = next((n for n in names if "key" in n.lower()), None)
    sh_name = next((n for n in names if "shaft" in n.lower()), None)

    # === Step 2a: planar ===
    print("\n=== Step 2a: planar ===")
    planar_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            if ky_name and (ky_name in (na, nb) and any("flange" in x for x in (na, nb))):
                continue
            ms = match_planar(parts[na]["planar"], parts[nb]["planar"])
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
            idx += 1
            # 只选朝外正面：面法向应指向远离包围盒中心
            bb_a = parts[na]["shape"].val().BoundingBox()
            bc_a = cq.Vector((bb_a.xmin+bb_a.xmax)/2, (bb_a.ymin+bb_a.ymax)/2, (bb_a.zmin+bb_a.zmax)/2)
            fc_a = cq.Vector(m["fa"]["c"]["x"], m["fa"]["c"]["y"], m["fa"]["c"]["z"])
            n_a = cq.Vector(m["fa"]["n"]["x"], m["fa"]["n"]["y"], m["fa"]["n"]["z"])
            bb_b = parts[nb]["shape"].val().BoundingBox()
            bc_b = cq.Vector((bb_b.xmin+bb_b.xmax)/2, (bb_b.ymin+bb_b.ymax)/2, (bb_b.zmin+bb_b.zmax)/2)
            fc_b = cq.Vector(m["fb"]["c"]["x"], m["fb"]["c"]["y"], m["fb"]["c"]["z"])
            n_b = cq.Vector(m["fb"]["n"]["x"], m["fb"]["n"]["y"], m["fb"]["n"]["z"])
            if fc_a.sub(bc_a).dot(n_a) < 0 or fc_b.sub(bc_b).dot(n_b) < 0:
                continue  # 跳过朝内的面
            print(f"  [{idx}] {na} <-> {nb}: {len(m['mc'])}c+{len(m['ml'])}l")
            la, lb = planar_labels(m, na, nb, idx)
            labels[na].append(la); labels[nb].append(lb)
            if na not in face_info: face_info[na] = {"c": m["fa"]["c"], "n": m["fa"]["n"]}
            if nb not in face_info: face_info[nb] = {"c": m["fb"]["c"], "n": m["fb"]["n"]}
            break

    # === Step 2b: shaft keyway <-> flange slot ===
    print("\n=== Step 2b: slot ===")
    if sh_name:
        sh_filtered = _shaft_keyway_filter(parts[sh_name]["cyls"], parts[sh_name]["planar"])
        for fl_name in [n for n in names if "flange" in n.lower()]:
            fl_filt = _bore_filter(parts[fl_name]["cyls"], parts[fl_name]["planar"])
            ms = match_slot(sh_filtered, fl_filt)
            if ms:
                ms.sort(key=lambda m: m["t"], reverse=True)
                for m in ms:
                    if m["t"] < 2: continue
                    fk = _fk(m["fb"])
                    if fk in used_p.get(fl_name, set()): continue
                    used_p.setdefault(fl_name, set()).add(fk)
                    idx += 1
                    print(f"  [{idx}] {sh_name} <-> {fl_name}: {len(m['mc'])}c+{len(m['ml'])}l [SLOT]")
                    la, lb = planar_labels(m, sh_name, fl_name, idx)
                    labels[sh_name].append(la); labels[fl_name].append(lb)
                    # 记录槽口面中心 → X 方向 = 轴心→面中心
                    if fl_name not in face_info:
                        face_info[fl_name] = {"c": m["fb"]["c"], "n": m["fb"]["n"]}
                    # 存储面中心用于 cylinder_labels 计算 X
                    slot_faces[fl_name] = m["fb"]["c"]
                    slot_faces[sh_name] = m["fa"]["c"]
                    break

    # === Step 2c: cylinder ===
    print("\n=== Step 2c: cylinder ===")
    def _ck(cyl): return f"c|{cyl['r']:.4f}|{cyl['mid']['x']:.4f}|{cyl['mid']['y']:.4f}|{cyl['mid']['z']:.4f}"
    cyl_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            ms = match_cylinders(parts[na]["cyls"], parts[nb]["cyls"])
            if ms: cyl_pairs.append((na, nb, ms))

    used_c = {}
    for na, nb, ms in cyl_pairs:
        for m in ms:
            s_has = na if m["shaft_in_a"] else nb
            b_has = nb if m["shaft_in_a"] else na
            kb = _ck(m["bore"])
            if kb in used_c.get(b_has, set()): continue
            used_c.setdefault(b_has, set()).add(kb)
            bore_pt = None
            if b_has in face_info:
                bore_pt = _bore_face_intersection(m["bore"], face_info[b_has])
            # X方向: 轴心→槽口面中心 (径向，无180°歧义)
            s_x = None; b_x = None
            s_cyl = m["shaft"]; b_cyl = m["bore"]
            if s_has in slot_faces:
                s_x = _vec_sub(slot_faces[s_has], s_cyl["mid"])
            if b_has in slot_faces:
                b_x = _vec_sub(slot_faces[b_has], b_cyl["mid"])
            idx += 1
            print(f"  [{idx}] {na}<->{nb}: R={m['shaft']['r']:.2f}")
            la, lb = cylinder_labels(m, na, nb, idx, bore_pt, s_x, b_x)
            labels[na].append(la); labels[nb].append(lb)
            break

    # === Step 3 ===
    print("\n=== Step 3 ===")
    for nm in names:
        lst = labels[nm]
        if not lst: print(f"  [!!] {nm}: 0"); continue
        with open(os.path.join(folder, f"{nm}_label.json"), "w", encoding="utf-8") as f:
            json.dump(fmt_json(lst), f, indent=2)
        print(f"  [OK] {nm}_label.json ({len(lst)} labels)")
    print("\n=== done ===")
