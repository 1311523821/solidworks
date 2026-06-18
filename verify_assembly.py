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
world_step = None  # from label metadata
for fp in sorted(set(os.path.normpath(f) for f in
    glob.glob(os.path.join(folder, "*.step")) + glob.glob(os.path.join(folder, "*.stp"))
    if "virtual" not in os.path.basename(f))):
    nm = os.path.splitext(os.path.basename(fp))[0]
    jp = os.path.join(folder, f"{nm}_label.json")
    if not os.path.exists(jp): continue
    data = json.load(open(jp, encoding="utf-8"))
    systs = data["modelAnnotation"]["features"]["featureCoordSyses"]
    parts[nm] = {"shape": importers.importStep(fp).val(), "labels": systs}
    names.append(nm)
    # 读取标签中的 worldStep 元数据（由 feature_matcher 写入）
    ws = data.get("modelAnnotation", {}).get("worldStep")
    if ws and not world_step:
        world_step = ws

# ---- group labels by mating ID ----
groups = {}
for nm in names:
    for l in parts[nm]["labels"]:
        gid = l["identifier"].rsplit("_Mating_", 1)[1] if "_Mating_" in l["identifier"] else l["identifier"]
        groups.setdefault(gid, []).append((nm, l))

print(f"=== verify_assembly: {os.path.basename(folder)} ===")
print(f"parts: {names}  groups: {len(groups)}")

# ---- pick anchor ----
if world_step and world_step in names:
    anchor = world_step
    print(f"world-step: {anchor} (来自标签元数据)")
else:
    degree = {nm: set() for nm in names}
    for gid, items in groups.items():
        for n1, _ in items:
            for n2, _ in items:
                if n1 != n2:
                    degree[n1].add(n2)
    anchor = max(names, key=lambda nm: (len(degree[nm]),
        sum(1 for l in parts[nm]["labels"] if l.get("userData",{}).get("matchType")=="CYLINDER"
            and not l.get("userData",{}).get("boreToBore")),
        -sum(1 for l in parts[nm]["labels"] if l.get("userData",{}).get("boreToBore"))))
print(f"anchor: {anchor}")

# ---- BFS placement ----
identity = cq.Location(cq.Plane(origin=cq.Vector(0,0,0), xDir=cq.Vector(1,0,0), normal=cq.Vector(0,0,1)))
world = {}
placed = set()

# 如果标签中指定了 worldStep，该零件使用 identity（静止不动）
if world_step and world_step in names:
    world[anchor] = identity
else:
    world[anchor] = identity * build_loc(parts[anchor]["labels"][0]["geometry"]).inverse
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

suffix = "_world_step" if world_step else ""
out_glb = os.path.join(folder, f"virtual_assembly_test{suffix}.glb")
out_step = os.path.join(folder, f"virtual_assembly_test{suffix}.step")
assembly.save(out_glb)
assembly.export(out_step, exportType="STEP")
print(f"saved: {out_glb}")
print(f"saved: {out_step}")
print(f"total: {len(names)} primary = {len(names)}")
