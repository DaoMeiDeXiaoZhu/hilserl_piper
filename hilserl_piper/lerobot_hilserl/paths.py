from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT / "cfg"
SHARED_CFG = CFG_DIR / "robot_shared.json"
HARDWARE_CFG = CFG_DIR / "hardware.json"
DATASETS = PROJECT / "datasets"
RAW_ROOT = DATASETS / "raw"
MDP_ROOT = DATASETS / "mdp"
DEMO_ROOT = DATASETS / "demo"  # 旧键盘路径，不再写入
SAC_ROOT = DATASETS / "sac"

HILSERL_ROOT = Path(
    os.environ.get("HILSERL_PIPER_ROOT", "/home/siasunds/zgy/hilserl_piper")
).expanduser()
RECORD_DEMO_PY = HILSERL_ROOT / "hilserl_piper" / "scripts" / "record_demo.py"
REPLAY_PY = HILSERL_ROOT / "hilserl_piper" / "scripts" / "replay.py"
SETUP_PY = HILSERL_ROOT / "hilserl_piper" / "setup.py"

SOURCE_DEMO = "demo"
SOURCE_EXPLORE = "explore"
SOURCE_INTERVENE = "intervene"
SOURCES = (SOURCE_DEMO, SOURCE_EXPLORE, SOURCE_INTERVENE)
