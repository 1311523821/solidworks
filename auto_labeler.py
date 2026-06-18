"""
auto_labeler.py
==============
智能特征打标与装配匹配

用法: python auto_labeler.py <folder> [--world-step <name>]
示例: python auto_labeler.py ./2 --world-step shaft_with_keyway

流程:
  1. feature_extractor: 提取每个 STEP 的特征 → 缓存 *_features.json
  2. feature_matcher:   特征间匹配
  3. label_generator:   生成 *_label.json（含 worldStep 元数据）
"""
import os, sys, json, glob
from feature_extractor import extract_file
from feature_matcher import match_all
from label_generator import fmt_json
from cadquery import importers

# ---- 解析 CLI: <folder> [--world-step <name>] ----
folder = None
world_step_arg = None
for i, a in enumerate(sys.argv[1:], 1):
    if a == '--world-step' and i + 1 < len(sys.argv):
        world_step_arg = sys.argv[i + 1]
    elif not a.startswith('--') and not (i > 1 and sys.argv[i-1] == '--world-step'):
        folder = a

if folder:
    folder = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder))
else:
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), './2')

# 解析零件名（需要先加载 STEP 列表才能做模糊匹配）
def _resolve_part(name_hint, part_names):
    hint = os.path.splitext(name_hint)[0]
    if hint in part_names:
        return hint
    matches = [n for n in part_names if hint.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    return None

print(f"=== auto_labeler: {folder} ===")

# Step 1: 提取特征（带缓存）
print("\n--- Step 1: extract features ---")
step_files = [os.path.normpath(f) for f in
    glob.glob(os.path.join(folder, "*.step")) + glob.glob(os.path.join(folder, "*.stp"))
    if "virtual" not in os.path.basename(f)]
step_files = list(set(step_files))

parts = {}
for fp in step_files:
    nm = os.path.splitext(os.path.basename(fp))[0]
    cache_path = os.path.join(folder, f"{nm}_features.json")
    # 检查缓存
    if os.path.exists(cache_path) and os.path.getmtime(fp) < os.path.getmtime(cache_path):
        features = json.load(open(cache_path, encoding="utf-8"))
        s = features["_summary"]
        print(f"  [cache] {nm}: {s['n_planar']}p {s['n_cylinders']}c")
    else:
        features = extract_file(fp)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
        s = features["_summary"]
        print(f"  [extract] {nm}: {s['n_planar']}p {s['n_cylinders']}c")
    parts[nm] = {"features": features, "shape_path": fp}

# 解析 world_step（指定世界坐标参考零件）
world_step = None
if world_step_arg:
    world_step = _resolve_part(world_step_arg, list(parts.keys()))
    if world_step:
        print(f"\n[world-step] {world_step} — 该零件将作为世界坐标参考（静止）")
    else:
        print(f"\n  [warn] --world-step '{world_step_arg}' 未找到匹配零件: {list(parts.keys())}")

# Step 2: 匹配
print("\n--- Step 2: match ---")
labels = match_all(parts, world_step=world_step)

# Step 3: 生成标签（含 worldStep 元数据）
print("\n--- Step 3: labels ---")
for nm, lst in labels.items():
    out_path = os.path.join(folder, f"{nm}_label.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fmt_json(lst, world_step=world_step), f, indent=2)
    print(f"  [OK] {nm}_label.json ({len(lst)} labels)")

print("\n=== done ===")
