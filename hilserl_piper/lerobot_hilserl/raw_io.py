"""原始经验：时间序、可回放。向量进 data.npz，图像进 images_*/。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .paths import RAW_ROOT, SOURCE_DEMO

VEC_ALWAYS = (
    "eef_xyz",
    "eef_rpy",
    "gripper_gap",
    "joint_pos",
    "joint_vel",
    "joint_current",
    "cmd_xyz",
    "cmd_rpy",
    "cmd_grip",
    "leader_eef_xyz",
    "leader_eef_rpy",
    "leader_gripper",
    "leader_joint_pos",
)


def next_episode_dir(root: Path | None = None) -> Path:
    root = Path(root or RAW_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        p = root / f"episode_{idx:03d}"
        if not p.exists():
            return p
        idx += 1


def list_raw_episodes(root: Path | None = None) -> list[Path]:
    root = Path(root or RAW_ROOT)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("episode_*") if p.is_dir() and (p / "data.npz").is_file())


def load_raw(ep_dir: Path) -> dict[str, Any]:
    z = np.load(Path(ep_dir) / "data.npz", allow_pickle=False)
    data = {k: z[k] for k in z.files}
    meta_p = Path(ep_dir) / "meta.json"
    if meta_p.is_file():
        data["_meta"] = json.loads(meta_p.read_text(encoding="utf-8"))
    return data


class RawBuffer:
    def __init__(self) -> None:
        self.cmd_action: list[np.ndarray] = []
        self.physical_xyz: list[np.ndarray] = []
        self.physical_rpy: list[np.ndarray] = []
        self.physical_grip: list[float] = []
        self.source: list[str] = []
        self.obs: dict[str, list[Any]] = {}

    def __len__(self) -> int:
        return len(self.cmd_action)

    def add(
        self,
        observation: dict[str, Any],
        cmd_action: np.ndarray,
        dxyz: np.ndarray,
        drpy: np.ndarray,
        dgrip: float,
        source: str,
    ) -> None:
        self.cmd_action.append(np.asarray(cmd_action, dtype=np.float32).reshape(-1).copy())
        self.physical_xyz.append(np.asarray(dxyz, dtype=np.float32).reshape(3).copy())
        self.physical_rpy.append(np.asarray(drpy, dtype=np.float32).reshape(3).copy())
        self.physical_grip.append(float(dgrip))
        self.source.append(str(source))
        for k, v in observation.items():
            self.obs.setdefault(k, []).append(v)

    def clear(self) -> None:
        self.cmd_action.clear()
        self.physical_xyz.clear()
        self.physical_rpy.clear()
        self.physical_grip.clear()
        self.source.clear()
        self.obs.clear()


def save_raw_episode(
    ep_dir: Path,
    buf: RawBuffer,
    *,
    shared: dict[str, Any],
    hw: dict[str, Any],
    episode_index: int,
    source_default: str = SOURCE_DEMO,
    init: dict[str, np.ndarray] | None = None,
) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = len(buf)
    if n == 0:
        raise SystemExit("空回合，不保存")
    act = shared.get("action") or {}
    payload: dict[str, Any] = {
        "cmd_action": np.stack(buf.cmd_action, axis=0),
        "physical_delta_xyz_m": np.stack(buf.physical_xyz, axis=0),
        "physical_delta_rpy_rad": np.stack(buf.physical_rpy, axis=0),
        "physical_delta_grip_m": np.asarray(buf.physical_grip, dtype=np.float32),
        "source": np.asarray(buf.source, dtype="U16"),
        "fps": np.array(float(shared.get("fps") or 20), dtype=np.float32),
        "action_scale_m": np.array(float(act.get("action_scale_m") or 0.0015), dtype=np.float32),
        "action_scale_deg": np.array(float(act.get("action_scale_deg") or 2.0), dtype=np.float32),
        "action_scale_gripper_m": np.array(float(act.get("action_scale_gripper_m") or 0.002), dtype=np.float32),
        "episode_index": np.array(int(episode_index), dtype=np.int32),
    }
    if init:
        for k, v in init.items():
            payload[f"init_{k}"] = np.asarray(v, dtype=np.float32)
    for key, series in buf.obs.items():
        if key.startswith("images_"):
            role = key[len("images_") :]
            img_dir = ep_dir / f"images_{role}"
            img_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(series):
                bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_dir / f"{i:06d}.png"), bgr)
        else:
            payload[key] = np.stack([np.asarray(x, dtype=np.float32) for x in series], axis=0)
    np.savez_compressed(ep_dir / "data.npz", **payload)
    meta = {
        "n": n,
        "source_default": source_default,
        "fps": float(shared.get("fps") or 20),
        "control": (hw.get("control") or {}),
        "action": {
            "type": act.get("type"),
            "names": act.get("names"),
        },
        "raw_keys": [k for k in payload if k != "source"],
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[raw] 保存 {ep_dir}  steps={n}", flush=True)
