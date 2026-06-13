import os, sys, glob, json
import cadquery as cq
from cadquery import importers

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.normpath(os.path.join(root, './2'))

def build_loc(geo):
    o = cq.Vector(geo["origin"]["x"], geo["origin"]["y"], geo["origin"]["z"])
    x = cq.Vector(geo["x"]["x"], geo["x"]["y"], geo["x"]["z"])
    z = cq.Vector(geo["z"]["x"], geo["z"]["y"], geo["z"]["z"])
    return cq.Location(cq.Plane(origin=o, xDir=x, normal=z))

parts = {}; names = []
for fp in set(os.path.normpath(f) for f in
    glob.glob(os.path.join(folder, "*.step")) + glob.glob(os.path.join(folder, "*.stp"))
    if "virtual" not in os.path.basename(f)):
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
best_t = -1; anchor = names[0]
for nm in names:
    for l in parts[nm]["labels"]:
        ud = l.get("userData", {})
        if ud.get("matchType") == "PLANAR" and ud.get("total", 0) > best_t:
            best_t = ud["total"]; anchor = nm

world = {anchor: target * build_loc(parts[anchor]["labels"][0]["geometry"]).inverse}
placed = {anchor}
print(f"\nanchor: {anchor}")

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
    print(f"  {d_name} <- {s_name}")

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
for i, nm in enumerate(names):
    xf = world.get(nm)
    if not xf: continue
    color = palette[i % len(palette)]
    assembly.add(parts[nm]["shape"].located(xf), name=nm, color=color)

out = os.path.join(folder, "virtual_assembly_test.glb")
assembly.save(out)
print(f"\nsaved: {out}")
