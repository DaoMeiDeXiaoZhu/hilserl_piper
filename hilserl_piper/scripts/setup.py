#!/usr/bin/env python3
"""本仓库 Setup：复用 hilserl_piper/setup.py 的标定流程，写入本仓库 cfg。

固定默认（不询问）:
  - 控制：仅从臂 follower_only + 键盘开环 Δxyz
  - 键盘步进：1mm；动作归一化 action_scale：1.5mm
  - 无夹爪动作/观测；示教时夹爪保持闭合

工作区可在交互里跳过（``--skip-workspace``），沿用 cfg 里已有 bounds。

不改 hilserl_piper / lerobot 源码。

用法::

    conda activate lerobot_latest
    cd ~/桌面/lerobot_hilserl
    bash scripts/can_activate.sh
    python scripts/setup.py --skip-workspace
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QT_LOGGING_RULES", "*=false")

from lerobot_hilserl.paths import HARDWARE_CFG, SHARED_CFG
from lerobot_hilserl.piper_bridge import load_setup_mod
from lerobot_hilserl.setup_overlay import apply_patches, restamp_hardware_file, sync_gym_configs


def main() -> int:
    if "--hardware" not in sys.argv:
        sys.argv[1:1] = ["--hardware", str(HARDWARE_CFG), "--shared", str(SHARED_CFG)]

    mod = load_setup_mod()
    apply_patches(mod)
    rc = int(mod.main() or 0)
    if rc != 0:
        return rc

    restamp_hardware_file(HARDWARE_CFG)
    hw = json.loads(HARDWARE_CFG.read_text(encoding="utf-8"))
    shared = json.loads(SHARED_CFG.read_text(encoding="utf-8"))
    sync_gym_configs(hw, shared)
    print(f"本仓库 cfg: {HARDWARE_CFG}", flush=True)
    print(f"            {SHARED_CFG}", flush=True)
    print("固定: 仅从臂 / 键盘1mm / action_scale=1.5mm / 无夹爪", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCtrl+C 退出", flush=True)
        raise SystemExit(130)
