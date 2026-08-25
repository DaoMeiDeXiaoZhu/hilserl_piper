#!/usr/bin/env python3
"""开环评测 BC：按 ``--obs-mode`` 加载对应检查点。

三种模式与训练一致（动作=Δxyz）::

    eef         只用末端 xyz
    image       只用腕部图像
    image_eef   图像 + xyz

用法::

    python scripts/eval.py --obs-mode eef --dry-run
    python scripts/eval.py --obs-mode image
    python scripts/eval.py --obs-mode image_eef --ckpt outputs/bc_image_eef/checkpoints/last/pretrained_model
    python scripts/eval.py --obs-mode eef --init-from-raw 1
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QT_LOGGING_RULES", "*=false")

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_hilserl.mdp_io import image_feature_key, resolve_bc_obs_mode
from lerobot_hilserl.paths import HARDWARE_CFG, PROJECT as PROJECT_ROOT, RAW_ROOT, SHARED_CFG
from lerobot_hilserl.piper_bridge import load_record_mod
from lerobot_hilserl.raw_io import list_raw_episodes, load_raw

DEFAULT_TRAIN_CFG = PROJECT_ROOT / "cfg" / "train_config.json"


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _minmax(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0


def _inv_minmax(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x + 1.0) * 0.5 * (hi - lo + 1e-8) + lo).astype(np.float32)


def default_ckpt_for_mode(obs_mode: str) -> Path:
    return PROJECT_ROOT / "outputs" / f"bc_{obs_mode}" / "checkpoints" / "last" / "pretrained_model"


def resolve_ckpt(obs_mode: str, train_cfg: dict, cli: Path | None) -> Path:
    if cli is not None:
        p = Path(cli)
    else:
        # 优先本模式默认目录；否则退回 train_config.pretrained_path
        cand = default_ckpt_for_mode(obs_mode)
        if (cand / "model.safetensors").is_file():
            p = cand
        else:
            raw = (train_cfg.get("policy") or {}).get("pretrained_path")
            p = Path(raw) if raw else cand
    if not (p / "model.safetensors").is_file():
        raise SystemExit(
            f"找不到权重：{p}/model.safetensors\n"
            f"请先: python scripts/train_bc.py --obs-mode {obs_mode}"
        )
    return p


def load_bc_side_meta(ckpt: Path) -> dict:
    meta: dict = {}
    for name in ("bc_meta.json", "bc_train_policy.json"):
        p = ckpt / name
        if p.is_file():
            meta[name] = json.loads(p.read_text(encoding="utf-8"))
    # config.json 是 lerobot save_pretrained 写出的
    cfg_p = ckpt / "config.json"
    if cfg_p.is_file():
        meta["config.json"] = json.loads(cfg_p.read_text(encoding="utf-8"))
    return meta


def stats_from_ckpt(ckpt: Path, train_cfg: dict, obs_mode: str) -> tuple[dict, tuple[int, int], str]:
    """返回 dataset_stats、image_hw、image_key。"""
    side = load_bc_side_meta(ckpt)
    pol = side.get("bc_train_policy.json") or (train_cfg.get("policy") or {})
    stats = dict(pol.get("dataset_stats") or {})
    # config.json 里也可能有
    cfgj = side.get("config.json") or {}
    if not stats and isinstance(cfgj.get("dataset_stats"), dict):
        stats = dict(cfgj["dataset_stats"])

    bc_meta = side.get("bc_meta.json") or {}
    img_hw = bc_meta.get("image_size") or [128, 128]
    img_key = bc_meta.get("image_key") or image_feature_key("wrist")
    # 若 ckpt 记录了 obs_mode，与 CLI 不一致则警告
    saved_mode = bc_meta.get("obs_mode")
    if saved_mode and resolve_bc_obs_mode(saved_mode) != obs_mode:
        _say(f"[warn] ckpt 内 obs_mode={saved_mode}，但 CLI 指定={obs_mode}，仍按 CLI 组观测")
    return stats, (int(img_hw[0]), int(img_hw[1])), str(img_key)


def wait_command(rec, fd: int) -> str:
    _say("Enter=开始  R=复位  Q=退出  （跑的时候 Esc 停本轮）")
    rec.flush_stdin(fd)
    while True:
        r, _, _ = select.select([fd], [], [], 0.2)
        if not r:
            continue
        name = rec.read_key(fd)
        if name in ("save_episode", "start_record"):
            return "start"
        if name == "reset":
            return "reset"
        if name in ("quit", "cancel"):
            return "quit"


def grab_wrist_chw(stream, resize_hw: tuple[int, int]) -> np.ndarray:
    """CamStream.read_rgb → float CHW [0,1]。"""
    rgb = stream.read_rgb()
    rh, rw = int(resize_hw[0]), int(resize_hw[1])
    if rgb.shape[0] != rh or rgb.shape[1] != rw:
        rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


def policy_action(
    policy,
    *,
    obs_mode: str,
    eef_xyz: np.ndarray | None,
    image_chw: np.ndarray | None,
    image_key: str,
    lo_s: torch.Tensor | None,
    hi_s: torch.Tensor | None,
    lo_a: np.ndarray,
    hi_a: np.ndarray,
    device: str,
) -> np.ndarray:
    obs: dict[str, torch.Tensor] = {}
    if obs_mode in ("eef", "image_eef"):
        assert eef_xyz is not None and lo_s is not None and hi_s is not None
        st = torch.as_tensor(np.asarray(eef_xyz, dtype=np.float32).reshape(1, 3), device=device)
        obs[OBS_STATE] = _minmax(st, lo_s, hi_s)
    if obs_mode in ("image", "image_eef"):
        assert image_chw is not None
        obs[image_key] = torch.as_tensor(image_chw, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        a_n = policy.actor.mode(obs)
    a_n = a_n.detach().float().cpu().numpy().reshape(-1)
    return np.clip(_inv_minmax(a_n, lo_a, hi_a), -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="开环评测 BC（按 obs-mode 选权重）")
    parser.add_argument(
        "--obs-mode",
        default="eef",
        choices=("eef", "image", "image_eef"),
        help="与 train_bc 一致；默认加载 outputs/bc_<mode>/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CFG)
    parser.add_argument("--ckpt", type=Path, default=None, help="覆盖默认路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--init-from-raw", type=int, default=None, help="从 1 起的 raw episode")
    parser.add_argument("--image-role", default="wrist")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    obs_mode = resolve_bc_obs_mode(args.obs_mode)
    train_cfg = json.loads(Path(args.train_config).read_text(encoding="utf-8"))
    shared = json.loads(SHARED_CFG.read_text(encoding="utf-8"))
    hw = json.loads(HARDWARE_CFG.read_text(encoding="utf-8"))
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    ckpt = resolve_ckpt(obs_mode, train_cfg, args.ckpt)
    stats, img_hw, img_key = stats_from_ckpt(ckpt, train_cfg, obs_mode)
    if args.image_role and obs_mode != "eef":
        img_key = image_feature_key(str(args.image_role))

    if ACTION not in stats:
        raise SystemExit(f"ckpt/config 缺 action dataset_stats: {ckpt}")
    lo_a = np.asarray(stats[ACTION]["min"], dtype=np.float32).reshape(-1)
    hi_a = np.asarray(stats[ACTION]["max"], dtype=np.float32).reshape(-1)
    lo_s = hi_s = None
    if obs_mode in ("eef", "image_eef"):
        if OBS_STATE not in stats:
            raise SystemExit(f"obs_mode={obs_mode} 需要 observation.state 的 dataset_stats")
        lo_s = torch.tensor(stats[OBS_STATE]["min"], device=device, dtype=torch.float32)
        hi_s = torch.tensor(stats[OBS_STATE]["max"], device=device, dtype=torch.float32)

    scale_m = float((shared.get("action") or {}).get("action_scale_m") or 0.0015)
    _say(f"obs_mode={obs_mode}  ckpt={ckpt}")
    _say(f"加载权重…")
    policy = GaussianActorPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()

    if args.dry_run:
        eef = None
        img = None
        if obs_mode in ("eef", "image_eef"):
            eef = (0.5 * (lo_s + hi_s)).detach().cpu().numpy()
        if obs_mode in ("image", "image_eef"):
            img = np.zeros((3, img_hw[0], img_hw[1]), dtype=np.float32)
        a = policy_action(
            policy,
            obs_mode=obs_mode,
            eef_xyz=eef,
            image_chw=img,
            image_key=img_key,
            lo_s=lo_s,
            hi_s=hi_s,
            lo_a=lo_a,
            hi_a=hi_a,
            device=device,
        )
        _say(f"[dry-run] a={a.round(3).tolist()}  Δmm≈{(a * scale_m * 1000).round(2).tolist()}")
        return 0

    rec = load_record_mod()
    need_cam = obs_mode in ("image", "image_eef")
    robot = None
    cams: dict = {}
    fd = rec.setup_tty()
    try:
        robot, reset = rec.connect_robot(shared, hw)
        grip = float(robot.gripper_bound[0])
        if need_cam:
            cams = rec.open_enabled_cameras(shared, hw)
            if str(args.image_role) not in cams:
                raise SystemExit(f"需要相机 {args.image_role}，当前打开: {list(cams)}")

        fps = float(shared.get("fps") or 20)
        period = 1.0 / max(fps, 1.0)
        cmd_xyz, rpy = rec.home_and_arm_ee(robot, reset, grip, hz=fps, fd=fd)

        while True:
            cmd = wait_command(rec, fd)
            if cmd == "quit":
                break
            if cmd == "reset":
                cmd_xyz, rpy = rec.home_and_arm_ee(robot, reset, grip, hz=fps, fd=fd)
                continue

            if args.init_from_raw is not None:
                eps = list_raw_episodes(RAW_ROOT)
                i = int(args.init_from_raw) - 1
                if i < 0 or i >= len(eps):
                    raise SystemExit(f"--init-from-raw 请给 1..{len(eps)}")
                data = load_raw(eps[i])
                pose = rec.init_pose_from_arrays(data)
                cmd_xyz, rpy = rec.restore_init_pose(
                    robot, pose or None, grip, hz=fps, fallback_reset=reset, fd=fd
                )

            policy.reset()
            _say(f"--- 开环最多 {args.max_steps} 步  Esc 停 ---")
            rec.flush_stdin(fd)
            stop = False
            for i in range(int(args.max_steps)):
                t0 = time.perf_counter()
                if select.select([fd], [], [], 0.0)[0]:
                    name = rec.read_key(fd)
                    if name in ("quit", "cancel"):
                        stop = name == "quit"
                        break

                xyz_now, _ = rec.read_xyz_rpy(robot)
                eef = np.asarray(xyz_now, dtype=np.float32).reshape(3)
                img = None
                if need_cam:
                    img = grab_wrist_chw(cams[str(args.image_role)], img_hw)

                a = policy_action(
                    policy,
                    obs_mode=obs_mode,
                    eef_xyz=eef if obs_mode != "image" else None,
                    image_chw=img,
                    image_key=img_key,
                    lo_s=lo_s,
                    hi_s=hi_s,
                    lo_a=lo_a,
                    hi_a=hi_a,
                    device=device,
                )
                # 归一化动作 → 物理 Δxyz
                dxyz = (a[:3] * scale_m).astype(np.float32)
                cmd_xyz, applied = rec.apply_delta_ee(robot, cmd_xyz, dxyz, rpy, grip)
                if float(np.linalg.norm(applied)) <= 1e-12:
                    rec.send_eef(robot, rec.clip_xyz(robot, cmd_xyz), rpy, grip)

                slept = period - (time.perf_counter() - t0)
                if slept > 0:
                    time.sleep(slept)
                if i % max(1, int(args.max_steps) // 10) == 0:
                    _say(
                        f"  [{i + 1}] a={a.round(3).tolist()}  "
                        f"Δmm={(applied * 1000).round(1).tolist()}  "
                        f"cmd={cmd_xyz.round(4).tolist()}"
                    )
            if stop:
                break
            cmd_xyz, rpy = rec.home_and_arm_ee(robot, reset, grip, hz=fps, fd=fd)
    except KeyboardInterrupt:
        _say("\nKeyboardInterrupt")
    finally:
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
