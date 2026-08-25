#!/usr/bin/env python3
"""SAC：一个入口，两个进程。网络更新（critic + actor）在 learner 里；actor 只负责真机交互和写经验。

用法::

    python scripts/train_sac.py --role learner
    python scripts/train_sac.py --role actor

同一份 ``--config_path``。先起 learner，再起 actor。
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DEFAULT_CFG = PROJECT / "cfg" / "train_config_piper_hilserl.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="SAC learner / actor")
    parser.add_argument("--role", choices=("learner", "actor"), required=True)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CFG)
    args, rest = parser.parse_known_args()

    cfg = Path(args.config_path).resolve()
    if not cfg.is_file():
        raise SystemExit(f"没有配置：{cfg}")

    argv = ["--config_path", str(cfg), *rest]
    if args.role == "learner":
        sys.argv = ["lerobot.rl.learner", *argv]
        runpy.run_module("lerobot.rl.learner", run_name="__main__")
        return 0

    from lerobot_hilserl.zero_actions import install_official_actor_send_filter

    install_official_actor_send_filter()
    sys.argv = ["lerobot.rl.actor", *argv]
    runpy.run_module("lerobot.rl.actor", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
