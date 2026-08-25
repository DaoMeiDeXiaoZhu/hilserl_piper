"""加载 hilserl_piper 的 record_demo / replay，并把配置指到本仓库 cfg。不改那边源码。"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from .paths import HARDWARE_CFG, HILSERL_ROOT, RECORD_DEMO_PY, REPLAY_PY, SETUP_PY, SHARED_CFG


def _exec_script(path, module_name: str) -> Any:
    if str(HILSERL_ROOT) not in sys.path:
        sys.path.insert(0, str(HILSERL_ROOT))
    if not path.is_file():
        raise SystemExit(f"找不到 {path}（检查 HILSERL_PIPER_ROOT）")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SHARED_CFG = SHARED_CFG
    mod.HARDWARE_CFG = HARDWARE_CFG
    return mod


def load_record_mod() -> Any:
    return _exec_script(RECORD_DEMO_PY, "hilserl_record_demo")


def load_replay_mod() -> Any:
    return _exec_script(REPLAY_PY, "hilserl_replay")


def load_setup_mod() -> Any:
    return _exec_script(SETUP_PY, "hilserl_setup")
