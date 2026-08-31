# Released Models — 分层飞行专家学生（30Hz + 5Hz）

本目录发布最新训练的深度-控制学生模型，用于模仿 C++ 分层专家
（`HierarchicalExpert`）的飞行策略。模型都是 **schema v25** 格式的因果
LSTM 策略，输入一张 D435i 深度图 + 7 维状态，输出机体系（FLU）控制。

| 文件 | 架构 | 职责 | 训练数据 |
|------|------|------|----------|
| `30hz_v32_origgoal_v3complete_mirror_nonoise_best.pt` | `ViTFlyLSTMPolicy` | **30 Hz 端到端学生**（原始导航目标，`student30` 栈默认） | `il_data_d435i_col_v3_complete`（554 ep / 46 scenes / 234,903 frames） |
| `30hz_v31_v3complete_mirror_nonoise_best.pt` | `ViTFlyLSTMPolicy` | **30 Hz 局部避障学生**（有效目标，分层栈低层执行器） | 同一数据集 |
| `5hz_macro_v5_v3complete_mirror_nonoise_best.pt` | `MacroPlannerPolicy` | **5 Hz 上层规划学生**（宏观接管/绕障） | 同一数据集（39,371 个 5 Hz 决策帧） |

> **v32 vs v31**：`v32_origgoal` 是原始导航目标消融——30 Hz 学生直接吃
> 原始导航目标（`navigation_goal_*`），自己规划整个绕行，不依赖 5 Hz 修正，
> 适合 `stack:=student30` 单模型部署（长墙/长绕行场景表现最好）。
> `v31` 输入的是 5 Hz 修正后的有效目标，适合 `stack:=student5_student30`
> 分层部署。两者 7-D 字段顺序一致，checkpoint 可互换加载（区别在训练目标源）。

两个模型使用**相同的训练配方**：镜像增强开启、深度噪声关闭（采集存的是干净深度，
噪声在部署/训练时按需叠加）、100 epoch、stateful truncated BPTT（LSTM 状态跨
chunk 传递）。

---

## 0. 系统背景（先读这个）

分层架构（与 C++ 专家一致）：

```
                 ┌─────────────────────────────┐
                 │  5 Hz 上层（宏观规划）        │
                 │  MacroPlannerPolicy 学生     │
                 │  输入: 深度 + 原始导航目标     │
                 │  输出: 修正后目标(方向+距离)   │
                 └──────────────┬──────────────┘
                                │ 世界锁存(decision-time yaw)
                                │ 每 30 Hz tick 重投影到当前机体
                 ┌──────────────▼──────────────┐
                 │ 30 Hz 低层（局部执行）         │
                 │  ViTFlyLSTMPolicy 学生        │
                 │  输入: 深度 + 有效目标          │
                 │  输出: FLU 速度 + yaw rate     │
                 └─────────────────────────────┘
```

- **30 Hz 学生**：可单独部署（直接飞原始导航目标，无上层），也能作为低层执行器
  跟随上层给出的"有效目标"。
- **5 Hz 学生**：只能与 30 Hz 学生组合（`student5_student30` 栈），给出修正目标。

**传感器（D435i，两个模型通用）**：

| 参数 | 值 | 说明 |
|------|----|----|
| 分辨率 | 640 × 360 | 16:9，D435i 满水平 |
| 垂直 FOV | 58° | Unity `Camera.fieldOfView` |
| 水平 FOV | ~89.16° | `2·atan(tan(29°)·640/360)` |
| near / far | 0.28 m / 10.0 m | D435i 有效深度范围 |
| **max_depth（使用范围）** | **5.0 m** | 深度 >5m 一律当作"远处"，编码为 far 标记 |
| 相机外参 T_BC | 单位旋转 + 前向 0.15 m | 相机在机头前方 15 cm |

---

## 1. 30 Hz 局部学生 `ViTFlyLSTMPolicy`

### 1.1 输入（每 30 Hz tick 一次）

**① 深度图**：形状 `[1, 1, 360, 640]`（B=1, C=1, H=360, W=640），值域 **[0, 1]**。
模型内部会**双线性缩放到 60 × 90** 再进视觉编码器（训练时同样缩放）。

深度归一化流程（`rollout.canonicalize_unity_depth`，与训练 loader 完全一致）：

1. Unity 深度 payload 单位是**百米**，`×100` → 米
2. `flipud`（Unity 图像是上下翻转的，翻回正常）
3. 无效 / 零 / 负像素 → 置为 `max_depth_m = 5.0`
4. 截断到 `[0, 5.0]` m
5. 取整到厘米（`uint16`），再 `cm × 0.01 / 5.0` → **[0, 1] 归一化深度**

> 即：`depth_normalized = clip(round(depth_m×100) × 0.01 / 5.0, 0, 1)`。
> 只有 0–5 m 有语义，5 m 之外全是 1.0（far）。

**② 7 维状态**：形状 `[1, 7]`，字段顺序（= `student_input_fields`）：

| 索引 | 字段 | 含义 | 归一化 |
|----|------|------|--------|
| 0–2 | `gravity_flu` | 重力在机体系 FLU 的 3 分量（水平悬停 = `[0,0,-1]`） | 除 `state_scale[0..2]=1` |
| 3–5 | `goal_direction_flu` | **有效目标**方向的单位向量（FLU 系，前/左/上） | 除 `state_scale[3..5]=1` |
| 6 | `goal_distance_norm` | 有效目标归一化距离 | 除 `state_scale[6]=1` |

- `goal_distance_norm = min(实际水平距离, R−0.5) / R`，其中 **R = 5.0 m**（`depth_max_m`），
  因此最大值 **0.9**（=4.5m/5m）。TURN 纯旋转时该值恒为 **1.0**（特殊标记，见下）。
- `goal_direction_flu` 是单位向量，投影在水平面（z=0 附近）。到达目标时 `[1,0,0]`（正前方）。
- **没有速度 / yaw_rate 输入**——刻意移除（2026-08-26），让策略必须读深度图做避障，
  不能靠当前运动状态"短路"。
- `state_scale = [1,1,1,1,1,1,1]`（checkpoint `normalization.state_scale`），
  归一化即 `state / state_scale`。

**有效目标（effective target）的语义**（决定 `goal_direction_flu` / `goal_distance_norm`）：

| 来源 | 含义 |
|------|------|
| **PASS_THROUGH** | 直接飞原始导航目标（无修正），方向=原目标方向 |
| **NORMAL_CORRECTION** | 上层给出一个修正的世界点航点，30 Hz 每 tick 重新投影到当前机体 |
| **TURN_LEFT / TURN_RIGHT** | 纯原地旋转（有限角度步进），`goal_distance_norm == 1.0` 恒成立 |

> 单独部署 30 Hz 学生（无上层）时，`goal_*` 就填**原始导航目标**。

### 1.2 输出（每 30 Hz tick）

**FLU 机体系命令** `[vx, vy, vz, yaw_rate]`：

| 分量 | 含义 | 物理范围 |
|----|------|---------|
| `vx` | 前向速度 | ~±2.5 m/s |
| `vy` | 左向速度 | ~±2.5 m/s |
| `vz` | 上向速度（FLU +up） | ~±2.5 m/s |
| `yaw_rate` | 偏航角速度（左正） | ~±1.5 rad/s |

网络输出是**归一化命令** `normalized ∈ ~[-1,1]`，实际命令 =
`normalized × command_scale`，其中 **`command_scale = [2.5, 2.5, 2.5, 1.5]`**
（checkpoint `normalization.command_scale`）。

### 1.3 时序语义

- 因果 LSTM（2 层），`(h, c)` 状态跨 30 Hz tick 持续传递（streaming），
  只在新的 episode 重置。
- 一次前向 = 一帧深度 + 一帧 7 维状态 → 一组 4 维命令。

---

## 2. 5 Hz 上层规划学生 `MacroPlannerPolicy`

### 2.1 输入（每 5 Hz 决策一次，即每 6 个 30 Hz tick）

**① 深度图**：与 30 Hz 完全相同（640×360 → 内部缩放 60×90，[0,1] 归一化）。

**② 7 维状态**：形状 `[1, 7]`，字段顺序（= `student_input_fields`）：

| 索引 | 字段 | 含义 |
|----|------|------|
| 0–2 | `gravity_flu` | 重力 FLU 3 分量（`[0,0,-1]`） |
| 3–5 | `navigation_goal_direction_flu` | **原始导航目标**方向的单位向量（FLU） |
| 6 | `navigation_goal_distance_norm` | 原始目标归一化距离 `min(d, R−0.5)/R` |

> **关键差异**：5 Hz 学生输入的是**原始导航目标**（`navigation_goal_*`），
> 而不是 30 Hz 学生看到的"有效目标"。它必须自己判断要不要修正/转向。

### 2.2 输出（每 5 Hz 决策）

**纯回归，无类型 token**（没有 PASS/NORMAL/TURN 分类）：

| 输出 | 形状 | 含义 |
|------|------|------|
| `direction` | `[3]` | **修正后目标**的单位方向（FLU，L2 归一化） |
| `distance_norm` | `[1]` | 修正后目标归一化距离（sigmoid，∈ (0,1)） |

**语义解码**（`decode_directive` / 运行时适配器）：

| 条件 | 解码为 | 含义 |
|------|--------|------|
| `direction ≈ 原始目标方向` | **PASS_THROUGH** | 无需修正，直飞原目标 |
| `direction 明显偏离` | **NORMAL_CORRECTION** | 修正目标（世界点） |
| `distance_norm ≈ 1.0` | **TURN_LEFT / TURN_RIGHT** | 纯旋转（按 direction 左右判定） |

- `distance_norm == 1.0` 是**纯旋转标记**：TURN 时水平位移为 0，只有旋转。
- 运行时适配器把预测的 `direction` **世界锁存**（decision-time yaw），
  然后在每个 30 Hz tick 重新投影到 30 Hz 学生当前的机体 FLU 系——所以
  一次 5 Hz 决策在 30 Hz 学生看来是连续更新的目标。

### 2.3 时序语义

- LSTM 状态**只在 5 Hz 携带**（每 6 个 30 Hz tick 更新一次）；
  中间 5 个 30 Hz tick 用零阶保持（zero-order hold）的同一决策，**不算**独立决策。
- 训练时只使用 `macro_update_mask == 1` 的真实决策帧（39,371 帧）。

---

## 3. 如何加载与推理

加载（checkpoint 自带全部归一化元数据，无需手动配置）：

```python
import torch
from model.model import ViTFlyPolicyConfig, ViTFlyLSTMPolicy
from model.model import MacroPolicyConfig, MacroPlannerPolicy

# 30 Hz
ck30 = torch.load("released_models/30hz_v31_v3complete_mirror_nonoise_best.pt",
                  map_location="cpu", weights_only=False)
m30 = ViTFlyLSTMPolicy(ViTFlyPolicyConfig(**ck30["model_config"]))
m30.load_state_dict({k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                     for k, v in ck30["model_state"].items()}, strict=True)
m30.eval()
# ck30["normalization"] = {depth_max_m:5.0, state_scale:[1]*7, command_scale:[2.5,2.5,2.5,1.5]}

# 5 Hz
ck5 = torch.load("released_models/5hz_macro_v5_v3complete_mirror_nonoise_best.pt",
                 map_location="cpu", weights_only=False)
m5 = MacroPlannerPolicy(MacroPolicyConfig(**ck5["model_config"]))
m5.load_state_dict({k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                    for k, v in ck5["model_state"].items()}, strict=True)
m5.eval()
```

单步推理（深度先按 §1.1 归一化到 `[0,1]`，形状 `[1,1,360,640]`；状态按 §1.1 除 scale）：

```python
import torch
import numpy as np
from rollout import preprocess_depth, build_normalized_state

depth_norm = ...                 # np.ndarray [360,640], 值域[0,1]
gravity_flu = np.array([0., 0., -1.])
goal_dir_flu = np.array([1., 0., 0.])       # 或上层给的 effective target 方向
goal_dist_norm = 0.8

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = preprocess_depth(depth_norm, dev)                       # [1,1,360,640]
st = build_normalized_state(gravity_flu, goal_dir_flu,
                            goal_dist_norm, ck30["normalization"]["state_scale"], dev)

with torch.no_grad():
    out = m30.step(dt, st, hidden)          # hidden 跨 tick 传递
    cmd = out.command[0].cpu().numpy()      # [vx,vy,vz,yaw_rate]（已乘 command_scale）
```

完整闭环使用请用仓库工具（已内置全部归一化 / 时序 / 世界锁存逻辑）：

```bash
# 30 Hz 学生单独（飞原始目标）
python3 rollout.py --checkpoint released_models/30hz_v31_v3complete_mirror_nonoise_best.pt \
  --depth-fov 58.0 --depth-near 0.28 --depth-far 10.0 --depth-max-m 5.0

# 5 Hz 学生 + 30 Hz 学生（完整学习栈）
python3 rollout_stack.py --stack student5_student30 \
  --checkpoint released_models/30hz_v31_v3complete_mirror_nonoise_best.pt \
  --macro-checkpoint released_models/5hz_macro_v5_v3complete_mirror_nonoise_best.pt \
  --expert-config ../il_dataset/config/il_dataset_joint_v2_config.yaml \
  --depth-fov 58.0 --depth-near 0.28 --depth-far 10.0 --depth-max-m 5.0
```

---

## 4. checkpoint 内部结构（schema v25）

```jsonc
{
  "schema_version": 25,
  "architecture": "ViTFlyLSTMPolicy" | "MacroPlannerPolicy",
  "student_input_fields": ["depth_file",
      "gravity_flu_x","gravity_flu_y","gravity_flu_z",
      "goal_direction_flu_x","goal_direction_flu_y","goal_direction_flu_z",
      "goal_distance_norm"],                      // 5Hz 用 navigation_goal_* 前缀
  "normalization": {
    "depth_max_m": 5.0,
    "state_scale": [1,1,1,1,1,1,1],               // 30Hz
    "command_scale": [2.5,2.5,2.5,1.5],           // 仅 30Hz
    "macro_state_scale": [1,1,1,1,1,1,1],         // 仅 5Hz
    "macro_type_names": ["PASS_THROUGH","NORMAL_CORRECTION","TURN_LEFT","TURN_RIGHT"] // 仅 5Hz
  },
  "model_config": { "image_height":60, "image_width":90, "state_dim":7, ... },
  "model_state": { ... }
}
```

> `architecture` 与 `schema_version` 被 `load_policy_checkpoint` 严格校验；
> 输入字段顺序必须与 `student_input_fields` 完全一致。

---

## 5. 已知边界（诚实声明）

- **室内障碍区**（间距≥1.6m 的小/中/大/混合障碍）：30 Hz 学生 6/6 通过，
  加 5 Hz 学生后 6/6 通过（0.5m 到达判定）。
- **室外 20m 级大障碍绕行**：当前两个学生都无法可靠完成（完整 C++ 专家可以）。
  原因是训练数据缺少这种极端长绕障样本——这是已知短板，非 bug。
- 到达判定：距离目标 < 0.5 m 即判成功（`rollout_stack --goal-immediate 0.5`）。
