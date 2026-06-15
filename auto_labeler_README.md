# auto_labeler — 智能特征打标与装配匹配

## 核心原则

1. **不同 STEP 文件的坐标系完全独立**——绝不跨零件做向量点积/叉积/加减
2. **所有匹配基于零件内禀几何量**：边长、半径、面内角度、径向距离
3. **匹配智能全在 feature_matcher，verify_assembly 是纯装配器（~80行）**

---

## 整体流程

```
*.step
  → feature_extractor.py: 提取面+圆柱特征，面内最远点排序
  → feature_matcher.py: 四层匹配（框架嵌入→平面→槽检测→圆柱）
  → label_generator.py: 生成 *_label.json（面/圆柱→坐标系标签）
  → verify_assembly.py: BFS 链式装配 + 多实例 → GLB/STEP
```

---

## 用法

```bash
# 1. 生成标签
python auto_labeler.py ./1    # 法兰对
python auto_labeler.py ./2    # 轴系装配
python auto_labeler.py ./3    # 风扇模块

# 2. 装配可视化
python verify_assembly.py ./1
python verify_assembly.py ./2
python verify_assembly.py ./3

# 输出：virtual_assembly_test.glb + virtual_assembly_test.step
```

---

## 文件职责

| 文件 | 行数 | 职责 |
|------|------|------|
| `feature_extractor.py` | ~150 | 从 STEP 提取 planar 面（圆+线）+ cylinder（半径/方向/内外），面内最远点作参考排序 |
| `feature_matcher.py` | ~650 | **全部匹配逻辑**：框架嵌入、平面匹配、槽检测、圆柱匹配、阵列过滤 |
| `label_generator.py` | ~100 | 面/圆柱特征 → JSON 标签（origin, xDir, zDir） |
| `verify_assembly.py` | ~80 | **纯装配器**：BFS 放置 + 多实例，无评分/精调/碰撞检测 |
| `auto_labeler.py` | ~60 | 入口脚本：调用 feature_extractor → feature_matcher → label_generator |

---

## 匹配分层

### Step 0: 框架嵌入 (FRAME-IN-FRAME)
只处理多圆柱零件对（>50 cylinders），检测矩形框架面匹配：

- `_is_frame_face`: 4 条长边构成矩形/方形
- `_match_lines_spatial`: 长度 + 面内方向一致匹配
- `_check_collision`: Z 轴深度碰撞检测（FIF 面均 Z 对齐）
- **滑动窗口线性阵列过滤**：找间距一致的子集，剔除离群假阳性槽位

### Step 1: 平面匹配 (PLANAR)
两面间的圆周长 + 线长度匹配：

- 阈值：≥3 特征直接过 / =2 特征须含圆
- **阵列检测**：`_is_circular_array` 识别螺栓孔法兰面
- xDir 优先用槽方向（slot_faces），否则用匹配圆方向
- 面法向：A=面法向 / B=-(面法向)

### Step 2: 槽检测
检测轴-法兰键槽面，**不生成单独标签**，只为 CYLINDER 标签提供 xDir 参考：

- `_shaft_keyway_filter`: 轴圆柱面附近的面（距轴心 < r±8mm，≥4 条线）
- `_bore_filter`: 孔圆柱面附近的面（距孔心 < r±15mm，法向⊥孔轴）

### Step 3: 圆柱匹配 (CYLINDER)
- **轴-孔** (ext 不同)：半径匹配 ±0.5mm，xDir 复用槽检测方向
- **孔-孔** (bore-to-bore)：半径 + 到中央孔径向距离一致 + used_b 去重
- **shaft-flange 原点**：轴 keyway 面心沿轴方向投影到轴线上（均在轴 CS 内计算）

---

## 装配逻辑 (verify_assembly.py)

1. **锚点选择**：连接最多零件者，同度数优先非 bore-bore CYLINDER 多者
2. **BFS 链式放置**：遍历 group，已放置→未放置（bore-bore 标签跳过）
3. **多实例**：同一零件对多 PLANAR 标签时，额外标签生成实例

---

## 匹配经验总结

| # | 经验 |
|---|------|
| 1 | **边长相等即匹配**——圆配圆、线配线，不猜零件类型 |
| 2 | **坐标隔离是铁律**——两个 STEP 文件的 CS 完全独立，绝不跨零件做向量运算（dot/cross/sub），只比较标量 |
| 3 | **面内排序用自身特征作参考**——最远点定零度角，不依赖全局轴 |
| 4 | **槽方向优先于螺栓孔方向**——法兰面 xDir 用槽面心方向，否则槽位被螺栓孔角度带偏 |
| 5 | **槽检测只存信息不生成标签**——槽匹配结果只给 CYLINDER 标签设 xDir，BFS 不把它当放置依据 |
| 6 | **轴点投影在轴的 CS 内做**——keyway 面心沿轴方向投影，绝不跨 CS 用法兰面心 |
| 7 | **bore-bore 只验证不放置**——BFS 跳过 boreToBore，避免 4-DOF 约束破坏 6-DOF 面贴面 |
| 8 | **阵列过滤踢出假阳性**——滑动窗口找间距一致子集，CAGE 只保留等距真内框 |
| 9 | **PLANAR 面是 6-DOF 主约束**——面心+法向+xDir，BFS 链式放置首选 |
| 10 | **CYLINDER 标签 4-DOF + slot xDir**——轴对齐 + 槽方向定旋转，配合面标签完成全约束 |

---

## 标签格式

```json
{
  "modelAnnotation": {
    "features": {
      "featureCoordSyses": [{
        "identifier": "零件名_Mating_N",
        "geometry": {
          "origin": {"x": ..., "y": ..., "z": ...},
          "x": {"x": ..., "y": ..., "z": ...},
          "y": {"x": ..., "y": ..., "z": ...},
          "z": {"x": ..., "y": ..., "z": ...}
        },
        "userData": {
          "type": "MATE",
          "matchType": "PLANAR | CYLINDER",
          "total": N,           // PLANAR: 匹配特征数
          "radius": R,          // CYLINDER: 圆柱半径
          "boreToBore": true    // CYLINDER: 孔-孔对齐标记
        }
      }]
    }
  }
}
```

**坐标系说明**：
- `origin`：特征中心在**零件局部 CS** 下的位置（原点在 STEP 文件原点）
- `z`：特征法向/轴线方向（PLANAR: 面法向；CYLINDER: 圆柱轴方向）
- `x`：参考方向（PLANAR: 面心→槽面心或第一个匹配圆；CYLINDER: 轴心→槽面心）
- `y`：自动计算 `y = z × x`

**跨零件变换**：两零件的 geometry 各自在局部 CS 下，通过 `build_loc(label_A) * build_loc(label_B)⁻¹` 链式装配对齐。

---

## 坐标系隔离原则

跨零件**只能比较标量**：

| 允许 | 禁止 |
|------|------|
| 半径相等 `abs(r1-r2)` | 向量点积 `v1·v2` |
| 长度相等 `abs(len1-len2)` | 向量叉积 `v1×v2` |
| 面内角度（同零件CS） | 跨零件距离 `(p1-p2).Length` |
| 径向距离（同零件CS） | 跨零件投影 `p1.dot(n2)` |

---

## 阵列检测

| 函数 | 用途 |
|------|------|
| `_is_circular_array(face)` | 检测面是否含圆周螺栓孔阵列（≥3 同半径圆，距离中心一致） |
| `_is_linear_array(faces, axis)` | 检测面集合是否沿轴等间距排列（用于 CAGE 槽位过滤） |

---

## 参数

| 参数 | 值 | 含义 |
|------|-----|------|
| TOL | 0.1mm | 边长方差 |
| MIN_OK | 3 | 平面匹配直接接受 |
| MIN_OK_LOOSE | 2 | 平面匹配最低（需含圆） |
| SLOT_MIN | 2 | 槽口匹配最低 |
| CYL_TOL | 0.5mm | 圆柱半径公差 |
| MULTI_CYL_THRESHOLD | 50 | 触发 FIF 匹配的圆柱数 |
