
https://github.com/user-attachments/assets/2590fba6-25ec-4662-ae1b-45827cb58f90
# lerobot_hilserl

外壳仓库：配置与脚本在本目录；**不改** `lerobot` / `hilserl_piper` 源码。通过 `piper_bridge` 复用 `hilserl_piper` 的连臂 / 键盘 / 相机逻辑。

默认场景：**单从臂 + 键盘开环 Δxyz** → 录 demo → BC（三种观测）→ 可选 SAC。

```bash
conda activate lerobot_latest
cd ~/桌面/lerobot_hilserl
```

---

## 默认控制方式（开环）

| 项 | 当前默认 |
|---|---|
| 硬件 | 仅从臂 `follower_only`，CAN=`can0` |
| 主臂 | 无（不用固件主从、不用 MS 配置脚本） |
| 遥操 | 键盘，**开环**：每帧根据按键算 Δxyz，累加到内部 `cmd_xyz`，再下发末端位姿 |
| 姿态 | **rpy 锁定**：开局读一次姿态后全程不变，只动平移 |
| 夹爪 | **无夹爪动作/观测**；下发时固定闭合 `gripper_bound[0]` |
| 控制频率 | `fps=20` |
| 键盘步进 | `keyboard_ee_step_m=0.001` → **每帧最多 1mm**（可多键叠加） |
| 动作归一化 | `action_scale_m=0.0015` → **±1 对应 ±1.5mm** |
| 越界 | `workspace.out_of_bounds=clip`，用 `robot.end_effector_bounds` 裁剪目标点 |
| 复位 | 关节模式，走到 `reset.fixed_reset_joint_positions` |

物理关系：

```text
键盘物理增量:  delta_m = held_axis * 1mm
写入数据集:    cmd_action = clip(delta_m / 1.5mm, -1, 1)
执行时还原:    delta_m   = cmd_action * 1.5mm
```

配置文件：

- `cfg/hardware.json` — CAN、相机设备
- `cfg/robot_shared.json` — 动作/观测/键盘/工作区/复位
- `cfg/train_config.json` — BC/SAC 训练底座（BC 会用 `--obs-mode` 覆盖输入维）

---

## 键盘映射（采集 / 挪起点）

**焦点必须在运行脚本的终端。**

| 键 | 作用 |
|---|---|
| `W` / `↑` | +X |
| `S` / `↓` | −X |
| `A` / `←` | +Y |
| `D` / `→` | −Y |
| `+` / `=` | +Z |
| `-` | −Z |
| `Space` | 从当前位置开始录制 |
| `Enter` | 保存本回合（写 raw + 投影 mdp）并复位 |
| `Esc` | 取消本回合（不保存）并复位 |
| `R` | 立刻复位到 cfg 关节 |
| `Q` | 退出 |

未按 `Space` 时也可 WASD 挪到起点；按 `Space` 后同样用这些键示教，边动边记盘。

---

## 数据落盘

```text
datasets/
  raw/episode_XXX/          # 完整示教（含图像、全量字段）
    data.npz
    images_wrist/…          # 若 observation.images.wrist=true
  mdp/episode_XXX/          # 从 raw 投影：按 cfg 裁剪字段、去掉零动作帧
```

- 采集保存时自动：`raw` → `mdp`（同名 `episode_XXX`）
- BC / SAC 读的是 **`datasets/mdp`**
- 当前观测开关（`robot_shared.json`）：`eef_xyz`、`eef_rpy`、腕部图 `wrist`；关节与夹爪关
- 当前动作：仅 `dx_eff, dy_eff, dz_eff`（3 维）

---

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `scripts/can_activate.sh` | 激活 USB-CAN（默认读 `hardware.json` → `can0`，1Mbps） |
| `scripts/setup.py` | 首次/改硬件时写 cfg（本仓库固定：仅从臂、1mm、1.5mm、无夹爪） |
| `scripts/record_demo.py` | 键盘开环采集 → raw + mdp |
| `scripts/replay.py` | 开环回放存盘 Δxyz（同一 `apply_delta_ee`） |
| `scripts/train_bc.py` | BC，三种 `--obs-mode` |
| `scripts/eval.py` | 开环推理，按 `--obs-mode` 加载对应权重 |
| `scripts/train_sac.py` | SAC：`--role learner` / `--role actor` |

---

## 完整操作流程

### 0. 环境与 CAN

```bash
conda activate lerobot_latest
cd ~/桌面/lerobot_hilserl
# 上电从臂，插好 USB-CAN / 腕部相机
bash scripts/can_activate.sh
```

无参数时按 `cfg/hardware.json` 的 `control.follower_can`（当前 `can0`）激活。

### 1. （可选）Setup / 改硬件

相机路径、工作区 bounds、复位关节等需要重绑时：

```bash
python scripts/setup.py --skip-workspace
```

- 本仓库补丁会**不问**控制模式：固定 `follower_only`、键盘 1mm、`action_scale` 1.5mm、无夹爪
- `--skip-workspace`：不跑绿灯拖动扫区，**沿用** `robot_shared.json` 里已有 `xyz_min/max` 与 `end_effector_bounds`
- 日常只采数据一般**不必**重跑 setup

### 2. 采集 demo

```bash
python scripts/record_demo.py
```

建议节奏：

1. 启动后自动走到复位关节，打印当前 `xyz`
2. 用 WASD 挪到任务起点（未录制也可动）
3. `Space` 开录 → 边按键边示教
4. `Enter` 保存并自动复位；或 `Esc` 丢弃
5. 重复；`Q` 退出

保存后应看到成对目录，例如：

```text
datasets/raw/episode_000/
datasets/mdp/episode_000/
```

### 3. 回放核对

```bash
python scripts/replay.py                         # 列出回合，交互选
python scripts/replay.py --source raw --index 1  # 第 1 条 = episode_000
python scripts/replay.py --source mdp --index 1
python scripts/replay.py --episode 0             # 按目录号 episode_000
```

- 默认 `--delta-source action`：用 `cmd_action * action_scale` 开环执行
- `--delta-source physical`：用存盘的 `physical_delta_xyz_m`
- 与采集一样：**开环 + rpy 锁定**

### 4. BC 训练（三种观测，三套权重）

动作一律是归一化 **Δxyz(3)**。观测用 `--obs-mode`：

| `--obs-mode` | 输入 | 默认输出目录 |
|---|---|---|
| `eef` | 末端 `eef_xyz` (3) | `outputs/bc_eef/` |
| `image` | 腕部图 → 默认 resize `128×128` | `outputs/bc_image/` |
| `image_eef` | 图像 + `eef_xyz` | `outputs/bc_image_eef/` |

```bash
python scripts/train_bc.py --obs-mode eef
python scripts/train_bc.py --obs-mode image
python scripts/train_bc.py --obs-mode image_eef --steps 30000
```

常用参数：

- `--mdp-root` 默认 `datasets/mdp`
- `--image-role wrist` / `--image-size 128 128`
- `--steps` 默认 20000；`--batch-size` 默认 64
- 默认丢掉空动作帧；需要保留时加 `--keep-idle`

检查点位置：

```text
outputs/bc_<mode>/checkpoints/last/pretrained_model/
  model.safetensors
  config.json
  bc_meta.json            # 记录 obs_mode、image_size 等
  bc_train_policy.json    # 含 dataset_stats
```

### 5. 开环推理（eval）

与训练同一套 `--obs-mode`，默认加载对应目录：

```bash
# 不连真机，只跑一步策略
python scripts/eval.py --obs-mode image --dry-run

# 真机开环
python scripts/eval.py --obs-mode image
python scripts/eval.py --obs-mode eef
python scripts/eval.py --obs-mode image_eef

# 指定权重
python scripts/eval.py --obs-mode image \
  --ckpt outputs/bc_image/checkpoints/last/pretrained_model
```

真机时终端：

- `Enter` / `Space`：开始本轮开环
- `R`：复位
- `Esc`：停本轮；`Q`：退出

其它：`--max-steps 400`、`--init-from-raw 1`（用某条 raw 的起点）、`--image-role wrist`。

策略输出 `a∈[-1,1]^3` → `Δxyz = a * 1.5mm` → 与采集相同的 `apply_delta_ee`。

### 6. （可选）SAC 在线

先保证 `cfg/train_config.json` 里 `policy.pretrained_path` 指向要用的 BC 权重（例如 `outputs/bc_image/.../pretrained_model`）。然后开两个终端：

```bash
# 终端 1：更新网络
python scripts/train_sac.py --role learner

# 终端 2：真机交互 + 写经验
python scripts/train_sac.py --role actor
```

---

## 当前观测 / 相机约定

`cfg/robot_shared.json`：

- `observation.images.wrist = true`，`front = false`
- 采集分辨率用法：`usage_size` / `resize_size` 为 **480×640**（高×宽）
- BC `image` / `image_eef` 训练时再缩到 **128×128**（`train_bc --image-size`）

`cfg/hardware.json`：腕部相机设备路径写在 `cameras.wrist.index_or_path`；换机器或换 USB 口后需改或重跑 setup。

---

## 注意

1. **开环不是闭环伺服**：键盘/策略只给增量；不根据误差修正，越界会被 clip。
2. **rpy 全程锁定**：任务若依赖姿态变化，当前管线不支持（`action.include_rpy=false`）。
3. **无夹爪维**：示教与推理都不动夹爪；保持闭合。
4. **终端焦点**：采集键鼠事件读本终端 tty；切到别的窗口会收不到键。
5. **退出后终端无回显**：执行 `stty sane`。
6. **三种 BC 模式互不覆盖**：换 `--obs-mode` 就换一套 `outputs/bc_*`；推理必须用同一 mode 才能对上输入维与权重。

## 效果演示
Uploading hilserl_piper插插座.mp4…
