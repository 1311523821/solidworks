"""
verify_assembly.py — 纯装配器
读取 label，按依赖链放置零件，输出 GLB + STEP。
不含任何匹配或精调逻辑——所有智能都在 feature_matcher/label_generator。
"""
import os, sys, glob, json, math
import cadquery as cq
from cadquery import importers

root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.normpath(os.path.join(root, sys.argv[1] if len(sys.argv) > 1 else './3'))

def build_loc(geo):
    o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
    x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
    z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
    return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))

# ---- load parts & labels ----
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

# ---- group labels by mating ID ----
groups = {}
for nm in names:
    for l in parts[nm]["labels"]:
        gid = l["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in l["identifier"] else l["identifier"]
        groups.setdefault(gid, []).append((nm, l))

print(f"=== verify_assembly: {os.path.basename(folder)} ===")
print(f"parts: {names}  groups: {len(groups)}")

# ---- pick anchor: part connected to most OTHER parts (highest degree in group graph)
degree = {nm: set() for nm in names}
for gid, items in groups.items():
    for n1, _ in items:
        for n2, _ in items:
            if n1 != n2:
                degree[n1].add(n2)
anchor = max(names, key=lambda nm: (len(degree[nm]),
    sum(1 for l in parts[nm]["labels"] if l.get("userData",{}).get("matchType")=="CYLINDER"
        and not l.get("userData",{}).get("boreToBore")),
    # fewer bore-bore labels = more likely a hub (shaft), not a flange
    -sum(1 for l in parts[nm]["labels"] if l.get("userData",{}).get("boreToBore"))))
print(f"anchor: {anchor}")

# ---- BFS placement ----
target = cq.Location(cq.Plane(origin=cq.Vector(0,0,0), xDir=cq.Vector(1,0,0), normal=cq.Vector(0,0,1)))
world = {}
placed = set()

# place anchor using its first label
world[anchor] = target * build_loc(parts[anchor]["labels"][0]["geometry"]).inverse
placed.add(anchor)

# BFS: repeatedly find a group with one placed part, place the other
# Skip bore-bore CYLINDER labels — they're for verification only, not placement
while len(placed) < len(parts):
    progress = False
    for gid, items in groups.items():
        if len(items) != 2: continue
        n1, l1 = items[0]; n2, l2 = items[1]
        # Never use bore-bore for primary placement
        if l1.get("userData", {}).get("boreToBore"): continue
        if n1 in placed and n2 not in placed:
            world[n2] = world[n1] * build_loc(l1["geometry"]) * build_loc(l2["geometry"]).inverse
            placed.add(n2); progress = True
        elif n2 in placed and n1 not in placed:
            world[n1] = world[n2] * build_loc(l2["geometry"]) * build_loc(l1["geometry"]).inverse
            placed.add(n1); progress = True
    if not progress: break

print(f"placed: {list(placed)}")

# ---- multi-instance: 多个 PLANAR 标签 = 同一零件对的多个实例 ----
pair_counts = {}
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    if l1.get("userData", {}).get("matchType") != "PLANAR": continue
    key = tuple(sorted([n1, n2]))
    pair_counts[key] = pair_counts.get(key, 0) + 1

instances = []
for gid, items in groups.items():
    if len(items) != 2: continue
    n1, l1 = items[0]; n2, l2 = items[1]
    ud = l1.get("userData", {})
    if ud.get("matchType") != "PLANAR": continue
    key = tuple(sorted([n1, n2]))
    if pair_counts.get(key, 0) <= 1: continue  # only one label → already placed
    if n1 not in placed or n2 not in placed: continue

    # skip the primary placement label (first one used in BFS)
    # determine which label was used for primary placement by checking if the
    # dest part's current world position matches this label pair
    s_name, d_name = n1, n2
    s_lbl, d_lbl = l1, l2

    # compute what world[d_name] would be if placed via THIS label
    alt_world = world[s_name] * build_loc(s_lbl["geometry"]) * build_loc(d_lbl["geometry"]).inverse
    primary_pos = world[d_name].wrapped.Transformation().TranslationPart()
    alt_pos = alt_world.wrapped.Transformation().TranslationPart()
    if abs(primary_pos.X() - alt_pos.X()) < 0.01 and abs(primary_pos.Y() - alt_pos.Y()) < 0.01 and abs(primary_pos.Z() - alt_pos.Z()) < 0.01:
        continue  # this IS the primary label, skip

    inst_name = f"{d_name}@{gid}"
    instances.append((parts[d_name]["shape"], alt_world, inst_name))
    print(f"  instance: {inst_name}")

# ---- visualize ----
palette = [
    cq.Color(0.85, 0.20, 0.20, 0.6), cq.Color(0.20, 0.65, 0.85, 0.6),
    cq.Color(0.30, 0.75, 0.30, 0.6), cq.Color(0.90, 0.70, 0.15, 0.6),
    cq.Color(0.75, 0.35, 0.85, 0.6), cq.Color(0.95, 0.55, 0.20, 0.6),
    cq.Color(0.25, 0.75, 0.75, 0.6), cq.Color(0.75, 0.45, 0.55, 0.6),
]
assembly = cq.Assembly()
i = 0
for nm in names:
    if nm not in world: continue
    assembly.add(parts[nm]["shape"].located(world[nm]), name=nm, color=palette[i % len(palette)])
    i += 1
for shape, loc, name in instances:
    assembly.add(shape.located(loc), name=name, color=palette[i % len(palette)])
    i += 1

out_glb = os.path.join(folder, "virtual_assembly_test.glb")
out_step = os.path.join(folder, "virtual_assembly_test.step")
assembly.save(out_glb)
assembly.export(out_step, exportType="STEP")
print(f"saved: {out_glb}")
print(f"saved: {out_step}")
print(f"total: {len(names)} primary + {len(instances)} instances = {len(names)+len(instances)}")