"""
auto_labeler.py
==============
智能特征打标与装配匹配

用法: python auto_labeler.py <folder>
示例: python auto_labeler.py ./2

流程:
  1. feature_extractor: 提取每个 STEP 的特征 → 缓存 *_features.json
  2. feature_matcher:   特征间匹配
  3. label_generator:   生成 *_label.json
"""
import os, sys, json, glob
from feature_extractor import extract_file
from feature_matcher import match_all
from label_generator import fmt_json
from cadquery import importers

if len(sys.argv) > 1:
    folder = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1]))
else:
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), './2')

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
        import json
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

# Step 2: 匹配
print("\n--- Step 2: match ---")
labels = match_all(parts)

# Step 3: 生成标签
print("\n--- Step 3: labels ---")
for nm, lst in labels.items():
    out_path = os.path.join(folder, f"{nm}_label.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fmt_json(lst), f, indent=2)
    print(f"  [OK] {nm}_label.json ({len(lst)} labels)")

print("\n=== done ===")
