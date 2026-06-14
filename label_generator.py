"""
label_generator.py
==================
模块3：从匹配结果生成标签 JSON（独立运行）
用法：
  python label_generator.py <folder> [match_file]
"""
import os, sys, json, math
import cadquery as cq

TOL = 0.1


def _ortho(z, x_hint=None):
    z = cq.Vector(z[0], z[1], z[2])
    if x_hint:
        ref = cq.Vector(x_hint[0], x_hint[1], x_hint[2])
    else:
        ref = cq.Vector(1, 0, 0) if abs(z.x) < 0.9 else cq.Vector(0, 1, 0)
    x = ref - z * (ref.dot(z) / z.dot(z))
    x = cq.Vector(1, 0, 0) if x.Length < 1e-9 else x.normalized()
    y = z.cross(x).normalized()
    return {"x": {"x": x.x, "y": x.y, "z": x.z},
            "y": {"y": y.y, "x": y.x, "z": y.z},
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


def planar_labels(m, na, nb, idx):
    fa, fb = m["fa"], m["fb"]
    oa, ob_ = fa["c"], fb["c"]
    meta = {"matchType": "PLANAR", "total": m["t"]}
    xa = xb = None
    # 优先用线（空间匹配保证方向一致），线为空时才用圆
    if m["ml"]:
        xa = _vec_sub(m["ml"][0][0]["m"], oa)
        xb = _vec_sub(m["ml"][0][1]["m"], ob_)
    elif m["mc"]:
        xa = _vec_sub(m["mc"][0][0]["c"], oa)
        xb = _vec_sub(m["mc"][0][1]["c"], ob_)
    return (mk_label(f"{na}_Mating_{idx}", oa, fa["n"], meta, xa),
            mk_label(f"{nb}_Mating_{idx}", ob_, _neg(fb["n"]), meta, xb))


def _bore_face_intersection(bore, face):
    d = cq.Vector(bore["dir"][0], bore["dir"][1], bore["dir"][2])
    n = cq.Vector(face["n"][0], face["n"][1], face["n"][2])
    c = cq.Vector(face["c"][0], face["c"][1], face["c"][2])
    m = cq.Vector(bore["mid"][0], bore["mid"][1], bore["mid"][2])
    denom = d.dot(n)
    if abs(denom) < 1e-9: return bore["mid"]
    t = (c.sub(m)).dot(n) / denom
    pt = m.add(d.multiply(t))
    return [pt.x, pt.y, pt.z]


def cylinder_labels(m, na, nb, idx, bore_origin=None, shaft_x=None, bore_x=None, shaft_origin=None):
    s, b = m["shaft"], m["bore"]
    meta = {"matchType": "CYLINDER", "radius": s["r"]}
    o_shaft = shaft_origin if shaft_origin else s["mid"]
    o_bore = bore_origin if bore_origin else b["mid"]
    shaft_nm = na if m["shaft_in_a"] else nb
    bore_nm = nb if m["shaft_in_a"] else na
    la = mk_label(f"{shaft_nm}_Mating_{idx}", o_shaft, s["dir"], meta, shaft_x)
    lb = mk_label(f"{bore_nm}_Mating_{idx}", o_bore, b["dir"], meta, bore_x)
    out = {shaft_nm: la, bore_nm: lb}
    return out[na], out[nb]


def fmt_json(lst):
    return {"modelAnnotation": {"parameters": {},
        "features": {"featurePoints": [], "featureLines": [], "featurePlanes": [],
                     "featureCoordSyses": lst, "featureSurfaces": [], "featureBodies": []},
        "children": []}}
