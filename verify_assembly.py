import os, sys, glob, json
import cadquery as cq
from cadquery import importers

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.normpath(os.path.join(root, './3'))

def build_loc(geo):
    o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
    x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
    z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
    return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))

parts = {}; names = []
for fp in sorted(set(os.path.normpath(f) for f in
    glob.glob(os.path.join(folder, "*.step")) + glob.glob(os.path.join(folder, "*.stp"))
    if "virtual" not in os.path.basename(f))):
    nm = os.path.splitext(os.path.basename(fp))[0]
    jp = os.path.join(folder, f"{nm}_label.json")
    if not os.path.exists(jp): continue
    systs = json.load(open(jp, encoding="utf-8"))["modelAnnotation"]["features"]["featureCoordSyses"]
    parts[nm] = {"shape": importers.importStep(fp).val(), "labels": systs}
    names.append(nm)

groups = {}
for nm in names:
    for l in parts[nm]["labels"]:
        gid = l["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in l["identifier"] else l["identifier"]
        groups.setdefault(gid, []).append((nm, l))

print("=== groups ===")
for gid, items in groups.items():
    print(f"  {gid}: {' + '.join(it[0] for it in items)}")

target = cq.Location(cq.Plane(origin=cq.Vector(0,0,0), xDir=cq.Vector(1,0,0), normal=cq.Vector(0,0,1)))

# 选锚点：计数每个零件在面标签中出现的次数，最多的为底座
# 同一零件在多个配对中出现 → 底座（CAGE），只在少数几次出现 → 被嵌入件（FAN）
anchor = names[0]; anchor_score = 0
for nm in names:
    n_planar = sum(1 for l in parts[nm]["labels"] if l.get("userData",{}).get("matchType")=="PLANAR")
    if n_planar > anchor_score:
        anchor_score = n_planar; anchor = nm
# 出现次数相同时，选面标签 total 更大的
best_t = -1; fallback = anchor
for nm in names:
    for l in parts[nm]["labels"]:
        ud = l.get("userData", {})
        if ud.get("matchType") == "PLANAR" and ud.get("total", 0) > best_t:
            best_t = ud["total"]; fallback = nm
if fallback != anchor:
    # 检查 fallback 是否有更多标签
    n_fb = sum(1 for l in parts[fallback]["labels"] if l.get("userData",{}).get("matchType")=="PLANAR")
    if n_fb >= anchor_score:
        anchor = fallback
        anchor_score = n_fb

world = {anchor: target * build_loc(parts[anchor]["labels"][0]["geometry"]).inverse}
placed = {anchor}
print(f"\nanchor: {anchor}  ({anchor_score} planar labels)")

# === Phase 1: standard chaining (place each part once) ===
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
        t = ud.get("total", 0)
        if t >= 10: s = 0
        elif t >= 3: s = 1
        elif ud.get("matchType") == "CYLINDER": s = 2
        else: s = 3
        if s < best_score: best_score = s; best = (r, src[0], dst[0])

    if not best: break
    items, (s_name, s_lbl), (d_name, d_lbl) = best
    world[d_name] = world[s_name] * build_loc(s_lbl["geometry"]) * build_loc(d_lbl["geometry"]).inverse
    placed.add(d_name)
    print(f"  {d_name} <- {s_name}  (primary)")

# === Phase 2: one-to-many — 同一零件对的多个面配对 ===
# 先统计每对零件有多少个PLANAR标签
pair_planar_count = {}
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    if l1.get("userData", {}).get("matchType") != "PLANAR": continue
    key = tuple(sorted([n1, n2]))
    pair_planar_count[key] = pair_planar_count.get(key, 0) + 1

instances = []
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    # 只处理同一零件对有多于1个PLANAR标签的情况
    key = tuple(sorted([n1, n2]))
    if pair_planar_count.get(key, 0) <= 1: continue
    if n1 not in placed or n2 not in placed: continue
    ud = l1.get("userData", {})
    if ud.get("matchType") != "PLANAR": continue

    # Compute this instance's world position
    # One part is the "source" (anchor/placed first), the other is "target"
    # The target part at this new position = source_world * source_label * target_label.inverse
    # Determine which is source (typically the cage, staying put) and which is target (fan, moving)
    # The source is the part that's closer to the anchor in the dependency chain
    s_name, s_lbl = items[0]
    d_name, d_lbl = items[1]
    src_loc = world[s_name] * build_loc(s_lbl["geometry"])
    tgt_world = src_loc * build_loc(d_lbl["geometry"]).inverse

    # Skip if this position is the same as the primary placement (within tolerance)
    primary_pos = world[d_name].wrapped.Transformation().TranslationPart()
    this_pos = tgt_world.wrapped.Transformation().TranslationPart()
    dx = primary_pos.X() - this_pos.X()
    dy = primary_pos.Y() - this_pos.Y()
    dz = primary_pos.Z() - this_pos.Z()
    if abs(dx) < 0.01 and abs(dy) < 0.01 and abs(dz) < 0.01:
        continue

    inst_name = f"{d_name}@{gid}"
    print(f"  {inst_name} <- {s_name}  (instance at slot {gid})")
    instances.append((parts[d_name]["shape"], tgt_world, inst_name))

# === Assembly ===
palette = [
    cq.Color(0.85, 0.20, 0.20, 0.6),  # 红
    cq.Color(0.20, 0.65, 0.85, 0.6),  # 蓝
    cq.Color(0.30, 0.75, 0.30, 0.6),  # 绿
    cq.Color(0.90, 0.70, 0.15, 0.6),  # 黄
    cq.Color(0.75, 0.35, 0.85, 0.6),  # 紫
    cq.Color(0.95, 0.55, 0.20, 0.6),  # 橙
    cq.Color(0.25, 0.75, 0.75, 0.6),  # 青
    cq.Color(0.75, 0.45, 0.55, 0.6),  # 粉
]
assembly = cq.Assembly()
i = 0
for nm in names:
    xf = world.get(nm)
    if not xf: continue
    color = palette[i % len(palette)]
    assembly.add(parts[nm]["shape"].located(xf), name=nm, color=color)
    i += 1
for shape, loc, name in instances:
    color = palette[i % len(palette)]
    assembly.add(shape.located(loc), name=name, color=color)
    i += 1

out = os.path.join(folder, "virtual_assembly_test.glb")
assembly.save(out)
print(f"\nsaved: {out}")
print(f"  {len(names)} primary parts + {len(instances)} additional instances = {len(names) + len(instances)} total")
