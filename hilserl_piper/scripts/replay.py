#!/usr/bin/env python3
"""开环回放：Δxyz（rpy 锁定），与 record_demo 同一路径 apply_delta_ee。

用法::

    python scripts/replay.py                  # 列出回合，交互选择
    python scripts/replay.py --index 1        # 第 1 条（episode_000）
    python scripts/replay.py --episode 0      # 按目录号 episode_000
    python scripts/replay.py --source mdp --index 2
    python scripts/replay.py --delta-source physical
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QT_LOGGING_RULES", "*=false")

from lerobot_hilserl.mdp_io import list_mdp_episodes, load_mdp, mdp_root
from lerobot_hilserl.paths import RAW_ROOT, SHARED_CFG
from lerobot_hilserl.piper_bridge import load_record_mod
from lerobot_hilserl.raw_io import list_raw_episodes, load_raw


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _ep_steps(ep: Path) -> int:
    try:
        z = np.load(ep / "data.npz", allow_pickle=False)
        for k in ("cmd_action", "actions", "physical_delta_xyz_m"):
            if k in z.files:
                return int(np.asarray(z[k]).shape[0])
    except Exception:
        pass
    return -1


def _pick_episode(eps: list[Path], *, index: int | None, episode: int | None) -> Path:
    if not eps:
        raise SystemExit("没有可回放的回合")

    _say(f"共 {len(eps)} 条：")
    for i, ep in enumerate(eps):
        n = _ep_steps(ep)
        n_s = str(n) if n >= 0 else "?"
        _say(f"  [{i + 1}] {ep.name}  steps={n_s}")

    if episode is not None:
        name = f"episode_{int(episode):03d}"
        for ep in eps:
            if ep.name == name:
                return ep
        raise SystemExit(f"找不到 {name}。可选: {[p.name for p in eps]}")

    if index is not None:
        i = int(index) - 1
        if i < 0 or i >= len(eps):
            raise SystemExit(f"--index 请给 1..{len(eps)}")
        return eps[i]

    if len(eps) == 1:
        _say(f"只有 1 条，自动选 [{1}] {eps[0].name}")
        return eps[0]

    raw = input(f"回放哪一条？[1..{len(eps)}] > ").strip()
    if not raw:
        raise SystemExit("未选择")
    try:
        i = int(raw) - 1
    except ValueError as exc:
        raise SystemExit(f"无效输入: {raw}") from exc
    if i < 0 or i >= len(eps):
        raise SystemExit(f"请给 1..{len(eps)}")
    return eps[i]


def main() -> int:
    parser = argparse.ArgumentParser(description="开环回放 Δxyz")
    parser.add_argument("--source", choices=("raw", "mdp"), default="raw")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="从 1 起的序号；省略则交互选择",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="按目录号，如 0 → episode_000",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--delta-source",
        choices=("action", "physical"),
        default="action",
        help="action=cmd_action*scale；physical=存盘 physical_delta_xyz_m",
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    args = parser.parse_args()

    shared = json.loads(SHARED_CFG.read_text(encoding="utf-8"))
    hw = json.loads((PROJECT / "cfg" / "hardware.json").read_text(encoding="utf-8"))
    scale_m = float((shared.get("action") or {}).get("action_scale_m") or 0.0015)

    if args.source == "raw":
        eps = list_raw_episodes(args.raw_root)
        if not eps:
            raise SystemExit(f"没有 raw：{args.raw_root}。请先 python scripts/record_demo.py")
        ep = _pick_episode(eps, index=args.index, episode=args.episode)
        data = load_raw(ep)
        label = f"raw/{ep.name}"
    else:
        mdp_dir = mdp_root(shared)
        eps = list_mdp_episodes(mdp_dir)
        if not eps:
            raise SystemExit(f"没有 MDP：{mdp_dir}")
        ep = _pick_episode(eps, index=args.index, episode=args.episode)
        data = load_mdp(ep)
        label = f"mdp/{ep.name}"

    if args.delta_source == "physical" and "physical_delta_xyz_m" in data:
        deltas = np.asarray(data["physical_delta_xyz_m"], dtype=np.float32)
    elif "cmd_action" in data:
        deltas = np.asarray(data["cmd_action"], dtype=np.float32) * scale_m
        if deltas.ndim == 1:
            deltas = deltas.reshape(1, -1)
        deltas = deltas[:, :3]
    elif "actions" in data:
        deltas = np.asarray(data["actions"], dtype=np.float32) * scale_m
        if deltas.ndim == 1:
            deltas = deltas.reshape(1, -1)
        deltas = deltas[:, :3]
    elif "physical_delta_xyz_m" in data:
        deltas = np.asarray(data["physical_delta_xyz_m"], dtype=np.float32)
    else:
        raise SystemExit("数据里没有 cmd_action / actions / physical_delta_xyz_m")

    n = int(deltas.shape[0])
    fps = float(np.asarray(data.get("fps", 20)).reshape(-1)[0])
    period = 1.0 / max(fps * float(args.speed), 0.1)
    _say(f"[replay] {label}  steps={n}  fps={fps}  delta_source={args.delta_source}")

    rec = load_record_mod()
    robot = None
    try:
        robot, reset = rec.connect_robot(shared, hw)
        grip = float(robot.gripper_bound[0])
        pose = rec.init_pose_from_arrays(data)
        cmd_xyz, rpy = rec.restore_init_pose(
            robot, pose or None, grip, hz=max(fps, 1.0), fallback_reset=reset
        )
        time.sleep(0.3)

        for t in range(n):
            t0 = time.perf_counter()
            d = np.asarray(deltas[t], dtype=np.float32).reshape(3)
            cmd_xyz, _ = rec.apply_delta_ee(robot, cmd_xyz, d, rpy, grip)
            if float(np.linalg.norm(d)) <= 1e-12:
                rec.send_eef(robot, rec.clip_xyz(robot, cmd_xyz), rpy, grip)
            slept = period - (time.perf_counter() - t0)
            if slept > 0:
                time.sleep(slept)
            if t % max(1, n // 10) == 0 or t == n - 1:
                act, _ = rec.read_xyz_rpy(robot)
                _say(f"  [{t + 1}/{n}] cmd={cmd_xyz.round(4).tolist()}  act={act.round(4).tolist()}")
        time.sleep(0.4)
    except KeyboardInterrupt:
        _say("\n中断")
        return 130
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                _say(f"disconnect: {exc}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
