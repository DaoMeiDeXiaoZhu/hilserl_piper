#!/usr/bin/env python3
"""键盘开环采集：Δxyz + 锁定 rpy，写入 datasets/raw。

按键（焦点在本终端）::

    Space   从当前位置开始录制
    Enter   保存(raw+mdp) 并复位
    Esc     取消并复位
    WASD / 方向键 / +/-   平移（未录制时可先挪起点）
    R       复位到 cfg 关节
    Q       退出

用法::

    bash scripts/can_activate.sh
    python scripts/record_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QT_LOGGING_RULES", "*=false")

from lerobot_hilserl.mdp_io import save_mdp_from_raw_episode
from lerobot_hilserl.paths import HARDWARE_CFG, RAW_ROOT, SHARED_CFG, SOURCE_DEMO
from lerobot_hilserl.piper_bridge import load_record_mod
from lerobot_hilserl.raw_io import RawBuffer, next_episode_dir, save_raw_episode


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    rec = load_record_mod()
    shared = json.loads(SHARED_CFG.read_text(encoding="utf-8"))
    hw = json.loads(HARDWARE_CFG.read_text(encoding="utf-8"))

    fps = float(shared.get("fps") or 20)
    period = 1.0 / max(fps, 1.0)
    hold_s = max(1.25 / fps, period + 0.01)
    hold_cfg = (shared.get("keyboard") or {}).get("hold_s")
    if hold_cfg is not None:
        hold_s = max(hold_s, float(hold_cfg))
    hold_s = max(hold_s, 0.18)

    step_m = float((shared.get("keyboard") or {}).get("keyboard_ee_step_m") or 0.001)
    scale_m = float((shared.get("action") or {}).get("action_scale_m") or 0.003)
    obs_cfg = shared.get("observation") or {}

    _say("=" * 60)
    _say("record_demo  键盘开环 Δxyz")
    _say(f"  out={RAW_ROOT}")
    _say(f"  fps={fps:.0f}  step={step_m * 1000:.2f}mm  action_scale={scale_m * 1000:.2f}mm")
    _say("  Space=开录  Enter=保存  Esc=取消  R=复位  Q=退出")
    _say("  WASD/方向键/+/- 平移（rpy 锁定）")
    _say("=" * 60)

    fd = rec.setup_tty()
    intent = rec.SharedIntent(hold_s=hold_s)
    robot = None
    cams: dict = {}
    try:
        robot, reset = rec.connect_robot(shared, hw)
        cams = rec.open_enabled_cameras(shared, hw)
        grip = float(robot.gripper_bound[0])

        cmd_xyz, rpy = rec.home_and_arm_ee(robot, reset, grip, hz=fps, fd=fd, intent=intent)
        buf = RawBuffer()
        ep_idx = int(next_episode_dir(RAW_ROOT).name.split("_")[1])
        recording = False
        init_pose: dict[str, np.ndarray] | None = None

        _say(f"就绪  xyz={cmd_xyz.round(4).tolist()}  下一回合 episode_{ep_idx:03d}")

        kb = threading.Thread(target=rec.keyboard_loop, args=(fd, intent), daemon=True)
        kb.start()

        z3 = np.zeros(3, dtype=np.float32)
        while not intent.should_stop():
            t0 = time.perf_counter()

            if intent.take_cancel():
                if recording:
                    buf.clear()
                    init_pose = None
                    recording = False
                    intent.set_recording(False)
                    cmd_xyz, rpy = rec.home_and_arm_ee(
                        robot, reset, grip, hz=fps, fd=fd, intent=intent
                    )
                    _say(f"已取消，未保存。Space 录 episode_{ep_idx:03d}")
                else:
                    _say("当前未在录制")

            if intent.take_save():
                if recording:
                    n_steps = len(buf)
                    recording = False
                    intent.set_recording(False)
                    pose_to_save = init_pose
                    init_pose = None
                    cmd_xyz, rpy = rec.home_and_arm_ee(
                        robot, reset, grip, hz=fps, fd=fd, intent=intent
                    )
                    if n_steps > 0:
                        ep_dir = RAW_ROOT / f"episode_{ep_idx:03d}"
                        init = {}
                        if pose_to_save:
                            if pose_to_save.get("eef_xyz") is not None:
                                init["eef_xyz"] = pose_to_save["eef_xyz"]
                            if pose_to_save.get("eef_rpy") is not None:
                                init["eef_rpy"] = pose_to_save["eef_rpy"]
                            if pose_to_save.get("joint_pos") is not None:
                                init["joint_pos"] = pose_to_save["joint_pos"]
                            init["grip"] = np.array([grip], dtype=np.float32)
                        save_raw_episode(
                            ep_dir,
                            buf,
                            shared=shared,
                            hw=hw,
                            episode_index=ep_idx,
                            source_default=SOURCE_DEMO,
                            init=init or None,
                        )
                        try:
                            save_mdp_from_raw_episode(ep_dir, shared, drop_zero_actions=True)
                        except SystemExit as exc:
                            _say(f"[mdp] 未写出: {exc}")
                        ep_idx += 1
                    else:
                        _say("空回合，不保存")
                    buf.clear()
                    _say(f"已复位。Space 录 episode_{ep_idx:03d}  xyz={cmd_xyz.round(4).tolist()}")
                else:
                    _say("未在录制")

            if intent.take_start():
                if not recording:
                    intent.clear_ee()
                    rec.flush_stdin(fd)
                    init_pose = rec.snapshot_init_pose(robot)
                    cmd_xyz = init_pose["eef_xyz"].copy()
                    rpy = init_pose["eef_rpy"].copy()
                    rec.send_eef(robot, cmd_xyz, rpy, grip)
                    recording = True
                    intent.set_recording(True)
                    buf.clear()
                    _say(f"--- 记盘中 episode_{ep_idx:03d}  Enter 保存 / Esc 取消 ---")
                else:
                    _say("已在录制中")

            if intent.take_reset():
                was = recording
                intent.set_recording(False)
                recording = False
                init_pose = None
                cmd_xyz, rpy = rec.home_and_arm_ee(
                    robot, reset, grip, hz=fps, fd=fd, intent=intent
                )
                if was:
                    buf.clear()
                _say(f"复位完成  xyz={cmd_xyz.round(4).tolist()}")

            if not recording:
                delta_cmd = rec.held_to_delta_m(intent.held(), step_m)
                if float(np.linalg.norm(delta_cmd)) > 1e-12:
                    cmd_xyz, _ = rec.apply_delta_ee(robot, cmd_xyz, delta_cmd, rpy, grip)
                else:
                    rec.send_eef(robot, rec.clip_xyz(robot, cmd_xyz), rpy, grip)
                slept = period - (time.perf_counter() - t0)
                if slept > 0:
                    time.sleep(slept)
                continue

            observation = rec.collect_observation(robot, cams, obs_cfg)
            delta_cmd = rec.held_to_delta_m(intent.held(), step_m)
            if float(np.linalg.norm(delta_cmd)) > 1e-12:
                cmd_xyz, applied = rec.apply_delta_ee(robot, cmd_xyz, delta_cmd, rpy, grip)
            else:
                cmd_xyz = rec.clip_xyz(robot, cmd_xyz)
                rec.send_eef(robot, cmd_xyz, rpy, grip)
                applied = z3.copy()

            slept = period - (time.perf_counter() - t0)
            if slept > 0:
                time.sleep(slept)

            action_norm = rec.physical_to_norm(applied, scale_m)
            buf.add(observation, action_norm, applied, z3, 0.0, SOURCE_DEMO)

    except KeyboardInterrupt:
        intent.request_stop()
        _say("\nKeyboardInterrupt")
    finally:
        intent.request_stop()
        rec.restore_tty()
        for c in cams.values():
            try:
                c.close()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                _say(f"disconnect: {exc}")
        _say("bye（若无回显: stty sane）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
