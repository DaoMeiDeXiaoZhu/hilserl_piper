"""键盘 demo 存盘格式与 hilserl_piper 一致：episode_*/data.npz。

若目录里还是旧的 LeRobotDataset（meta/info.json），第一次录/回放时转成 npz。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .paths import DEMO_ROOT


def list_episodes(root: Path | None = None) -> list[Path]:
    root = Path(root or DEMO_ROOT)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("episode_*") if p.is_dir() and (p / "data.npz").is_file())


def load_episode_npz(ep_dir: Path) -> dict[str, Any]:
    z = np.load(Path(ep_dir) / "data.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_bc_pairs(
    root: Path | None = None,
    *,
    skip_idle: bool = True,
    idle_abs: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """从 demo npz 取出 (eef_xyz, action_norm) 供 BC。

    action 已是 [-1, 1]。默认丢掉接近 0 的保持帧，避免策略学成「什么都不做」；
    每条回合最后一帧仍保留。
    """
    root = Path(root or DEMO_ROOT)
    eps = list_episodes(root)
    if not eps:
        raise SystemExit(f"没有 demo：{root}/episode_*/data.npz。请先 python scripts/record_demo.py")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    n_raw = 0
    n_keep = 0
    for ep_dir in eps:
        data = load_episode_npz(ep_dir)
        if "eef_xyz" not in data or "actions" not in data:
            raise SystemExit(f"{ep_dir} 缺少 eef_xyz 或 actions")
        eef = np.asarray(data["eef_xyz"], dtype=np.float32).reshape(-1, 3)
        act = np.asarray(data["actions"], dtype=np.float32).reshape(-1, 3)
        n = min(len(eef), len(act))
        n_raw += n
        for i in range(n):
            a = act[i]
            if skip_idle and float(np.linalg.norm(a)) < idle_abs and i < n - 1:
                continue
            states.append(eef[i])
            actions.append(a)
            n_keep += 1
    if not states:
        raise SystemExit("demo 里没有可用帧（全是空动作？）")
    return (
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        {"episodes": len(eps), "raw_frames": n_raw, "kept_frames": n_keep},
    )


def migrate_lerobot_demo_if_needed(root: Path | None = None) -> None:
    """把外壳早期写入的 LeRobotDataset 转成 episode_*/data.npz，避免和官方键盘回放混用。"""
    root = Path(root or DEMO_ROOT)
    info = root / "meta" / "info.json"
    if not info.is_file():
        return

    print(f"[migrate] 发现旧 LeRobot 数据集 {root}，转为 hilserl_piper 的 episode_*/data.npz …", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STATE, REWARD

    def _as_np(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _as_int(x: Any) -> int:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)

    dataset = LeRobotDataset("local/piper_demo", root=root)
    n_ep = int(dataset.num_episodes)
    for i in range(n_ep):
        ep_meta = dataset.meta.episodes[i]
        a = _as_int(ep_meta["dataset_from_index"])
        b = _as_int(ep_meta["dataset_to_index"])
        ep_dir = root / f"episode_{i:03d}"
        out = ep_dir / "data.npz"
        if out.is_file():
            print(f"[migrate] 已有 {out}，跳过", flush=True)
            continue

        actions, physical, eef, rewards = [], [], [], []
        init_xyz = init_rpy = init_q = None
        scale = 0.0015
        for idx in range(a, b):
            row = dataset[idx]
            actions.append(_as_np(row[ACTION]).astype(np.float32).reshape(-1))
            if "complementary_info.physical_delta_m" in row:
                physical.append(
                    _as_np(row["complementary_info.physical_delta_m"]).astype(np.float32).reshape(3)
                )
            if OBS_STATE in row:
                eef.append(_as_np(row[OBS_STATE]).astype(np.float32).reshape(-1)[:3])
            if REWARD in row:
                rewards.append(float(_as_np(row[REWARD]).reshape(-1)[0]))
            if init_xyz is None and "complementary_info.init_eef_xyz" in row:
                init_xyz = _as_np(row["complementary_info.init_eef_xyz"]).astype(np.float32).reshape(3)
                init_rpy = _as_np(row["complementary_info.init_eef_rpy"]).astype(np.float32).reshape(3)
                init_q = _as_np(row["complementary_info.init_joint_pos"]).astype(np.float32).reshape(6)
            if "complementary_info.action_scale_m" in row:
                scale = float(_as_np(row["complementary_info.action_scale_m"]).reshape(-1)[0])

        n = len(actions)
        payload: dict[str, Any] = {
            "actions": np.stack(actions) if actions else np.zeros((0, 3), np.float32),
            "fps": np.array(float(dataset.meta.fps), dtype=np.float32),
            "action_scale_m": np.array(float(scale), dtype=np.float32),
            "episode_index": np.array(int(i), dtype=np.int32),
            "is_intervention": np.ones(n, dtype=np.bool_),
            "success": np.array(True),
            "rewards": np.asarray(rewards, dtype=np.float32)
            if rewards
            else np.zeros(n, dtype=np.float32),
        }
        if physical:
            payload["physical_delta_m"] = np.stack(physical)
        if eef:
            payload["eef_xyz"] = np.stack(eef)
        if init_xyz is not None:
            payload["init_eef_xyz"] = init_xyz
            payload["init_eef_rpy"] = init_rpy
            payload["init_joint_pos"] = init_q
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **payload)
        print(f"[migrate] 写出 {out}  frames={n}", flush=True)

    backup = root.parent / "_backup_lerobot_demo"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("meta", "data", "videos", "images"):
        src = root / name
        if src.exists():
            dst = backup / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"[migrate] 旧 LeRobot 文件移到 {dst}", flush=True)
    print("[migrate] 完成。之后走 datasets/raw 与 scripts/replay.py。", flush=True)
