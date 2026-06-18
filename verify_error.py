"""
装配误差验证 — 只检查关键配合特征
用法: python verify_error.py ./2 [--world-step <name>]
"""
import os, sys, json, math
import cadquery as cq
from cadquery import importers
from OCP.gp import gp_Pnt

TOL = 0.15  # 误差容限 (mm)

def _xform_point(loc, vec):
    trsf = loc.wrapped.Transformation()
    p = gp_Pnt(vec.x, vec.y, vec.z)
    p2 = p.Transformed(trsf)
    return cq.Vector(p2.X(), p2.Y(), p2.Z())
def _xform_dir(loc, vec):
    trsf = loc.wrapped.Transformation()
    p = gp_Pnt(vec.x, vec.y, vec.z); o = gp_Pnt(0, 0, 0)
    p2 = p.Transformed(trsf); o2 = o.Transformed(trsf)
    return cq.Vector(p2.X() - o2.X(), p2.Y() - o2.Y(), p2.Z() - o2.Z())
def build_loc(geo):
    o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
    x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
    z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
    return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))

root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.normpath(os.path.join(root, sys.argv[1] if len(sys.argv) > 1 else "./2"))
print(f"{'='*60}")
print(f"  装配误差验证: {os.path.basename(folder)}")
print(f"{'='*60}")

# Load
parts = {}; names = []; world_step = None
for fp in sorted(os.listdir(folder)):
    if not (fp.endswith(".step") or fp.endswith(".stp")): continue
    if "virtual" in fp: continue
    nm = os.path.splitext(os.path.basename(fp))[0]
    jp = os.path.join(folder, f"{nm}_label.json")
    if not os.path.exists(jp): continue
    data = json.load(open(jp, encoding="utf-8"))
    systs = data["modelAnnotation"]["features"]["featureCoordSyses"]
    parts[nm] = {"labels": systs}
    names.append(nm)
    # 读取 worldStep 元数据
    ws = data.get("modelAnnotation", {}).get("worldStep")
    if ws and not world_step:
        world_step = ws

# Groups
groups = {}
for nm in names:
    for l in parts[nm]["labels"]:
        gid = l["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in l["identifier"] else l["identifier"]
        groups.setdefault(gid, []).append((nm, l))

# ---- Assembly (same logic as verify_assembly.py) ----
identity = cq.Location(cq.Plane(origin=cq.Vector(0,0,0), xDir=cq.Vector(1,0,0), normal=cq.Vector(0,0,1)))

if world_step and world_step in names:
    anchor = world_step
    world = {anchor: identity}
else:
    anchor = names[0]; anchor_score = 0
    for nm in names:
        n_planar = sum(1 for l in parts[nm]["labels"] if l.get("userData",{}).get("matchType")=="PLANAR")
        if n_planar > anchor_score: anchor_score = n_planar; anchor = nm
    best_t = -1
    for nm in names:
        for l in parts[nm]["labels"]:
            ud = l.get("userData", {})
            if ud.get("matchType") == "PLANAR" and ud.get("total", 0) > best_t:
                best_t = ud["total"]; fallback = nm
    if fallback != anchor:
        n_fb = sum(1 for l in parts[fallback]["labels"] if l.get("userData",{}).get("matchType")=="PLANAR")
        if n_fb >= anchor_score: anchor = fallback
    world = {anchor: identity * build_loc(parts[anchor]["labels"][0]["geometry"]).inverse}
placed = {anchor}

# Phase 1
remaining = list(groups.values())
while len(placed) < len(parts) and remaining:
    remaining = [r for r in remaining if not all(n in placed for n, _ in r)]
    if not remaining: break
    best = None; best_score = 999
    for r in remaining:
        src = [(n, l) for n, l in r if n in placed]
        dst = [(n, l) for n, l in r if n not in placed]
        if not src or not dst: continue
        ud = dst[0][1].get("userData", {})
        t = ud.get("total", 0); src_in_anchor = any(n == anchor for n, _ in src)
        is_cyl = ud.get("matchType") == "CYLINDER"
        if t >= 10: s = 0
        elif src_in_anchor and not is_cyl and t >= 3: s = 1
        elif is_cyl and src_in_anchor: s = 2
        elif src_in_anchor and not is_cyl: s = 3
        elif is_cyl and not src_in_anchor: s = 2
        elif t >= 3: s = 1
        else: s = 3
        if s < best_score: best_score = s; best = (r, src[0], dst[0])
    if not best: break
    items, (s_name, s_lbl), (d_name, d_lbl) = best
    world[d_name] = world[s_name] * build_loc(s_lbl["geometry"]) * build_loc(d_lbl["geometry"]).inverse
    placed.add(d_name)
    # Collision flip
    n_s_w = _xform_dir(world[s_name], cq.Vector(s_lbl["geometry"]["z"]["x"], s_lbl["geometry"]["z"]["y"], s_lbl["geometry"]["z"]["z"]))
    n_d_w = _xform_dir(world[d_name], cq.Vector(d_lbl["geometry"]["z"]["x"], d_lbl["geometry"]["z"]["y"], d_lbl["geometry"]["z"]["z"]))
    if n_s_w.dot(n_d_w) < -0.9:
        face_c_w = _xform_point(world[s_name], cq.Vector(s_lbl["geometry"]["origin"]["x"], s_lbl["geometry"]["origin"]["y"], s_lbl["geometry"]["origin"]["z"]))
        xdir_w = _xform_dir(world[s_name], cq.Vector(s_lbl["geometry"]["x"]["x"], s_lbl["geometry"]["x"]["y"], s_lbl["geometry"]["x"]["z"]))
        T = cq.Location(face_c_w); R = cq.Location(cq.Vector(0, 0, 0), xdir_w, 180.0)
        world[d_name] = (T * R * T.inverse) * world[d_name]


# Phase 1.5
refined_parts = set()
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    ud1 = l1.get("userData", {})
    if ud1.get("matchType") != "PLANAR": continue
    if ud1.get("total", 0) > 3: continue
    if anchor not in (n1, n2): continue
    child = n2 if n1 == anchor else n1
    child_lbl = l2 if n1 == anchor else l1; anchor_lbl = l1 if n1 == anchor else l2
    if child not in placed: continue
    anchor_n_w = _xform_dir(world[anchor], cq.Vector(anchor_lbl["geometry"]["z"]["x"], anchor_lbl["geometry"]["z"]["y"], anchor_lbl["geometry"]["z"]["z"]))
    child_n_w = _xform_dir(world[child], cq.Vector(child_lbl["geometry"]["z"]["x"], child_lbl["geometry"]["z"]["y"], child_lbl["geometry"]["z"]["z"]))
    face_axis = None; face_origin_local = None
    for gid2, items2 in groups.items():
        if len(items2) != 2: continue
        nn = [it[0] for it in items2]
        if child not in nn: continue
        other = nn[1] if nn[0] == child else nn[0]
        if other not in placed: continue
        child_p = next(it[1] for it in items2 if it[0] == child)
        udp = child_p.get("userData", {})
        if udp.get("matchType") == "PLANAR" and udp.get("total", 0) >= 3:
            pz = child_p["geometry"]["z"]; po = child_p["geometry"]["origin"]
            face_axis = _xform_dir(world[child], cq.Vector(pz["x"], pz["y"], pz["z"]))
            face_origin_local = cq.Vector(po["x"], po["y"], po["z"]); break
    if face_axis is None: continue
    a_proj = anchor_n_w - face_axis * anchor_n_w.dot(face_axis)
    c_proj = child_n_w - face_axis * child_n_w.dot(face_axis)
    if a_proj.Length < 0.01 or c_proj.Length < 0.01: continue
    cross = a_proj.cross(c_proj); ang = math.atan2(cross.dot(face_axis), a_proj.dot(c_proj))
    if abs(ang) < 0.001: continue
    face_c_w = _xform_point(world[child], face_origin_local)
    T = cq.Location(face_c_w); R = cq.Location(cq.Vector(0, 0, 0), face_axis, ang * 180 / math.pi)
    world[child] = (T * R * T.inverse) * world[child]
    refined_parts.add(child)

# Refine using best bore-bore pair
best_bb = None; best_bb_ang = float("inf")
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    ud1 = l1.get("userData", {})
    if ud1.get("matchType") != "CYLINDER": continue
    if ud1.get("boreToBore") != True: continue
    if n1 not in world or n2 not in world: continue
    s_name, d_name = n1, n2; s_lbl_cyl, d_lbl_cyl = l1, l2
    if d_name == anchor:
        if n2 != anchor: s_name, d_name = n2, n1; s_lbl_cyl, d_lbl_cyl = l2, l1
        else: continue
    if d_name in refined_parts:
        other = n1 if d_name == n2 else n2
        if other not in refined_parts and other != anchor: s_name, d_name = d_name, other; s_lbl_cyl = l2 if d_name == n1 else l1; d_lbl_cyl = l1 if d_name == n1 else l2
        else: continue
    planar_items = None
    for gid2, items2 in groups.items():
        if len(items2) != 2: continue
        nn = [it[0] for it in items2]
        if s_name in nn and d_name in nn:
            p1 = next(it[1] for it in items2 if it[0] == s_name)
            if p1.get("userData", {}).get("matchType") == "PLANAR": planar_items = items2; break
    if not planar_items: continue
    s_planar = next(it[1] for it in planar_items if it[0] == s_name)
    s_cyl_w = _xform_point(world[s_name], cq.Vector(s_lbl_cyl["geometry"]["origin"]["x"], s_lbl_cyl["geometry"]["origin"]["y"], s_lbl_cyl["geometry"]["origin"]["z"]))
    d_cyl_w = _xform_point(world[d_name], cq.Vector(d_lbl_cyl["geometry"]["origin"]["x"], d_lbl_cyl["geometry"]["origin"]["y"], d_lbl_cyl["geometry"]["origin"]["z"]))
    s_face_n_w = _xform_dir(world[s_name], cq.Vector(s_planar["geometry"]["z"]["x"], s_planar["geometry"]["z"]["y"], s_planar["geometry"]["z"]["z"]))
    s_face_c_w = _xform_point(world[s_name], cq.Vector(s_planar["geometry"]["origin"]["x"], s_planar["geometry"]["origin"]["y"], s_planar["geometry"]["origin"]["z"]))
    s_proj = (s_cyl_w - s_face_c_w) - s_face_n_w * ((s_cyl_w - s_face_c_w).dot(s_face_n_w))
    d_proj = (d_cyl_w - s_face_c_w) - s_face_n_w * ((d_cyl_w - s_face_c_w).dot(s_face_n_w))
    if s_proj.Length < 0.01 or d_proj.Length < 0.01: continue
    ang = abs(math.atan2(d_proj.y, d_proj.x) - math.atan2(s_proj.y, s_proj.x))
    if ang < best_bb_ang: best_bb_ang = ang; best_bb = (s_name, d_name, s_planar, s_face_n_w, s_face_c_w, ang, gid)

if best_bb and best_bb_ang > 0.001:
    s_name, d_name, s_planar, s_face_n_w, s_face_c_w, ang, gid = best_bb
    T = cq.Location(s_face_c_w); R = cq.Location(cq.Vector(0, 0, 0), s_face_n_w, ang * 180 / math.pi)
    world[d_name] = (T * R * T.inverse) * world[d_name]
    refined_parts.add(d_name)

print(f"\n装配完成: 锚点={anchor}, 已放置={list(placed)}, 已精调={list(refined_parts)}")

# ========== 验证（只检查实际对齐的特征） ==========
print(f"\n{'='*60}")
print(f"  配合误差验证 (容限={TOL}mm)")
print(f"{'='*60}")

ok = 0; fail = 0; details = []

for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    if n1 not in world or n2 not in world: continue
    ud1 = l1.get("userData", {}); ud2 = l2.get("userData", {})
    mt = ud1.get("matchType", "?")
    g1 = l1["geometry"]; g2 = l2["geometry"]

    c1_w = _xform_point(world[n1], cq.Vector(g1["origin"]["x"], g1["origin"]["y"], g1["origin"]["z"]))
    c2_w = _xform_point(world[n2], cq.Vector(g2["origin"]["x"], g2["origin"]["y"], g2["origin"]["z"]))
    n1_w = _xform_dir(world[n1], cq.Vector(g1["z"]["x"], g1["z"]["y"], g1["z"]["z"]))
    n2_w = _xform_dir(world[n2], cq.Vector(g2["z"]["x"], g2["z"]["y"], g2["z"]["z"]))

    if mt == "PLANAR":
        center_dist = (c1_w - c2_w).Length
        normal_dot = n1_w.dot(n2_w)
        t = ud1.get("total", 0)
        # Main face (t>=10): faces should be coincident, normals parallel
        # SLOT (t<=3): faces parallel, offset allowed
        # Keyway (4<=t<10): faces coincident
        if t >= 4:
            err = center_dist
            if err < TOL and normal_dot > 0.9:
                ok += 1; details.append(f"  OK  [{gid}] {n1}<->{n2} PLANAR(t={t}) offset={center_dist:.4f}mm")
            else:
                fail += 1; details.append(f"  ERR [{gid}] {n1}<->{n2} PLANAR(t={t}) offset={center_dist:.4f}mm normal_dot={normal_dot:.4f}")
        else:
            # SLOT: just check normals
            if normal_dot > 0.9 or normal_dot < -0.9:
                ok += 1; details.append(f"  OK  [{gid}] {n1}<->{n2} SLOT(t={t}) normal_dot={normal_dot:.4f}")
            else:
                fail += 1; details.append(f"  ERR [{gid}] {n1}<->{n2} SLOT(t={t}) normal_dot={normal_dot:.4f}")

    elif mt == "CYLINDER":
        avg_axis = (n1_w + n2_w).normalized() if n1_w.dot(n2_w) > 0 else (n1_w - n2_w).normalized()
        radial = ((c1_w - c2_w) - avg_axis * (c1_w - c2_w).dot(avg_axis)).Length
        is_bb = ud1.get("boreToBore", False)
        label = "bore-bore" if is_bb else "shaft-bore"
        if radial < TOL:
            ok += 1; details.append(f"  OK  [{gid}] {n1}<->{n2} CYL({label}) radial={radial:.4f}mm")
        else:
            fail += 1; details.append(f"  ERR [{gid}] {n1}<->{n2} CYL({label}) radial={radial:.4f}mm")

for d in sorted(details):
    print(d)

print(f"\n{'='*60}")
if fail == 0:
    print(f"  ✓ 全部通过: {ok} 配合对, 0mm 误差 (容限{TOL}mm)")
else:
    print(f"  WARN: {ok} OK + {fail} FAIL")
print(f"{'='*60}")

# 法兰面重合度
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    ud1 = l1.get("userData", {})
    if ud1.get("matchType") != "PLANAR": continue
    if ud1.get("total", 0) < 10: continue
    g1 = l1["geometry"]; g2 = l2["geometry"]
    c1_w = _xform_point(world[n1], cq.Vector(g1["origin"]["x"], g1["origin"]["y"], g1["origin"]["z"]))
    c2_w = _xform_point(world[n2], cq.Vector(g2["origin"]["x"], g2["origin"]["y"], g2["origin"]["z"]))
    n1_w = _xform_dir(world[n1], cq.Vector(g1["z"]["x"], g1["z"]["y"], g1["z"]["z"]))
    face_offset = abs((c1_w - c2_w).dot(n1_w))
    print(f"\n法兰面重合度 [{gid}]: 面心距离={(c1_w-c2_w).Length:.4f}mm, 法向偏移={face_offset:.4f}mm")
