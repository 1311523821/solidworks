"""
verify_assembly.py — 纯装配器
读取 label，按依赖链放置零件，输出 GLB + STEP。
不含任何匹配或精调逻辑——所有智能都在 feature_matcher/label_generator。
"""
import os, sys, glob, json, math
import cadquery as cq
from cadquery import importers

root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.normpath(os.path.join(root, sys.argv[1] if len(sys.argv) > 1 
                                       else './2'))

# ---- BREP 缓存：加载速度提升 10-50x ----
def _import_cached(filepath):
    """导入 STEP，自动缓存为 .brep 二进制格式。二次加载百毫秒级。"""
    brep_path = filepath.rsplit(".", 1)[0] + ".brep"
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    if os.path.exists(brep_path) and os.path.getmtime(brep_path) >= os.path.getmtime(filepath):
        from OCP.TopoDS import TopoDS_Shape
        ts = TopoDS_Shape()
        BRepTools.Read_s(ts, brep_path, BRep_Builder())
        return cq.Shape.cast(ts)
    shape = importers.importStep(filepath)
    BRepTools.Write_s(shape.val().wrapped, brep_path)
    return shape.val()


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
    with open(jp, encoding="utf-8") as f:
        data = json.load(f)
    systs = data["modelAnnotation"]["features"]["featureCoordSyses"]
    parts[nm] = {"shape": _import_cached(fp), "labels": systs}
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
    # 优先使用 PRIMARY PLANAR 标签（非 boreToBore）做 anchor 放置
    anchor_label = None
    for l in parts[anchor]["labels"]:
        ud = l.get("userData", {})
        if ud.get("matchType") == "PLANAR" and not ud.get("boreToBore"):
            anchor_label = l
            break
    if anchor_label is None:
        # 回退：使用第一个非 boreToBore 的标签
        for l in parts[anchor]["labels"]:
            if not l.get("userData", {}).get("boreToBore"):
                anchor_label = l
                break
    if anchor_label is None:
        anchor_label = parts[anchor]["labels"][0]
    world[anchor] = identity * build_loc(anchor_label["geometry"]).inverse
placed.add(anchor)

# BFS: repeatedly find a group with one placed part, place the other
# Skip bore-bore CYLINDER labels — they're for verification only, not placement
def _axes_parallel(l1, l2, threshold=0.9):
    """检查两个标签的Z轴方向是否足够平行（用于CYLINDER标签可靠性检查）"""
    z1 = l1["geometry"]["z"]; z2 = l2["geometry"]["z"]
    dot = abs(z1["x"]*z2["x"] + z1["y"]*z2["y"] + z1["z"]*z2["z"])
    return dot >= threshold

while len(placed) < len(parts):
    progress = False
    # 第一遍：优先使用非 boreToBore 标签
    for gid, items in groups.items():
        if len(items) != 2: continue
        n1, l1 = items[0]; n2, l2 = items[1]
        if l1.get("userData", {}).get("boreToBore"): continue
        # boreToBore 标签轴线不平行时变换不可靠（两个孔方向不同→旋转不确定），跳过
        # 非 boreToBore 的 CYLINDER 标签变换本身能处理方向差异，不跳过
        if (l1.get("userData", {}).get("matchType") == "CYLINDER"
            and l1.get("userData", {}).get("boreToBore")
            and not _axes_parallel(l1, l2)):
            continue
        if n1 in placed and n2 not in placed:
            world[n2] = world[n1] * build_loc(l1["geometry"]) * build_loc(l2["geometry"]).inverse
            placed.add(n2); progress = True
        elif n2 in placed and n1 not in placed:
            world[n1] = world[n2] * build_loc(l2["geometry"]) * build_loc(l1["geometry"]).inverse
            placed.add(n1); progress = True
    if progress: continue
    # 第二遍：用 boreToBore 标签放置剩余零件（仅当无法通过主标签放置时）
    # 但 boreToBore 标签的轴线方向必须一致（dot > 0.9），否则变换不可靠
    for gid, items in groups.items():
        if len(items) != 2: continue
        n1, l1 = items[0]; n2, l2 = items[1]
        if not l1.get("userData", {}).get("boreToBore"): continue
        # boreToBore 轴线方向检查
        z1 = l1["geometry"]["z"]; z2 = l2["geometry"]["z"]
        z_dot = abs(z1["x"]*z2["x"] + z1["y"]*z2["y"] + z1["z"]*z2["z"])
        if z_dot < 0.9: continue  # 轴线不平行，boreToBore 变换不可靠
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
assembly.save(out_step)
print(f"saved: {out_step}")
print(f"saved: {out_glb}")
print(f"total: {len(names)} primary = {len(names)}")
