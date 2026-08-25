"""从 raw 投影 MDP：同级 episode_*、同轨迹格式。

与 raw 相同字段风格（cmd_action / eef_* / physical_* / init_*），区别仅是：
  - 只保留 observation.* / action.include_* 打开的字段
  - 丢掉静止（零动作）帧
  - cmd_action 为任务动作空间上的归一化动作
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .paths import MDP_ROOT, SOURCE_DEMO
from .raw_io import load_raw

OBS_VEC_ORDER = (
    "eef_xyz",
    "eef_rpy",
    "gripper_gap",
    "joint_pos",
    "joint_vel",
    "joint_current",
)

ZERO_ACTION_ABS = 1e-8


def mdp_root(shared: dict[str, Any] | None = None, root: Path | None = None) -> Path:
    """datasets/mdp（与 datasets/raw 平级，下面直接是 episode_*）。"""
    if root is not None:
        return Path(root)
    if shared:
        layout = shared.get("dataset_layout") or {}
        base = layout.get("root")
        name = layout.get("mdp") or "mdp"
        if base:
            return Path(base) / str(name)
    return MDP_ROOT


# 兼容旧调用名
def mdp_task_dir(shared: dict[str, Any], root: Path | None = None) -> Path:
    return mdp_root(shared, root)


def obs_enabled(shared: dict[str, Any]) -> list[str]:
    obs = shared.get("observation") or {}
    keys = [k for k in OBS_VEC_ORDER if bool(obs.get(k))]
    if not keys:
        raise SystemExit("observation 全是 false，MDP 没有观测字段")
    return keys


def images_enabled(shared: dict[str, Any]) -> list[str]:
    images = (shared.get("observation") or {}).get("images") or {}
    if not isinstance(images, dict):
        return []
    return [k for k, on in images.items() if on]


def action_slice(shared: dict[str, Any]) -> tuple[list[int], list[str]]:
    act = shared.get("action") or {}
    names = list(act.get("names") or [])
    include_rpy = bool(act.get("include_rpy", True))
    include_g = bool(act.get("include_gripper", True))
    idx = [0, 1, 2]
    out_names = ["x_off", "y_off", "z_off"] if str(act.get("type") or "").startswith("abs") else ["dx_eff", "dy_eff", "dz_eff"]
    if include_rpy:
        idx.extend([3, 4, 5])
        out_names.extend(
            ["roll_off", "pitch_off", "yaw_off"]
            if str(act.get("type") or "").startswith("abs")
            else ["droll_deg", "dpitch_deg", "dyaw_deg"]
        )
    if include_g:
        idx.append(6)
        out_names.append("grip_off" if str(act.get("type") or "").startswith("abs") else "dgripper_m")
    if names and len(names) == len(idx):
        out_names = names
    return idx, out_names


def pack_state(row: dict[str, Any], keys: list[str]) -> np.ndarray:
    parts = []
    for k in keys:
        if k not in row:
            raise SystemExit(f"缺观测字段 {k}，无法组 state")
        parts.append(np.asarray(row[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def _nonzero_mask(actions: np.ndarray, *, abs_eps: float = ZERO_ACTION_ABS) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return np.linalg.norm(a, axis=-1) > float(abs_eps)


def project_raw_data_to_mdp(
    data: dict[str, Any],
    shared: dict[str, Any],
    *,
    raw_episode_name: str,
    episode_index: int,
    drop_zero_actions: bool = True,
    zero_abs: float = ZERO_ACTION_ABS,
) -> tuple[dict[str, Any], dict[str, int]]:
    """投影成与 raw 同结构的轨迹 payload（字段子集 + 去静止 + 归一化动作）。"""
    keys = obs_enabled(shared)
    aidx, anames = action_slice(shared)
    actions_full = np.asarray(data["cmd_action"], dtype=np.float32)
    n = int(actions_full.shape[0])
    if n == 0:
        raise SystemExit("空回合，无法写 MDP")

    src = data.get("source")
    if src is None:
        src = np.array([SOURCE_DEMO] * n, dtype="U16")
    else:
        src = np.asarray(src, dtype="U16").reshape(-1)[:n]

    ac = actions_full[:, aidx] if actions_full.ndim == 2 else actions_full.reshape(n, -1)[:, aidx]
    if drop_zero_actions:
        # 绝对动作：静止用帧间 physical Δ；delta 动作：用 cmd_action 范数
        if (
            "physical_delta_xyz_m" in data
            and "physical_delta_rpy_rad" in data
            and "physical_delta_grip_m" in data
        ):
            dxyz = np.asarray(data["physical_delta_xyz_m"], dtype=np.float32)
            drpy = np.asarray(data["physical_delta_rpy_rad"], dtype=np.float32)
            dg = np.asarray(data["physical_delta_grip_m"], dtype=np.float32).reshape(n, -1)
            move = np.linalg.norm(dxyz.reshape(n, -1), axis=-1)
            move = move + np.linalg.norm(drpy.reshape(n, -1), axis=-1) + np.abs(dg.reshape(n, -1)[:, 0])
            keep = move > float(zero_abs)
        else:
            keep = _nonzero_mask(ac, abs_eps=zero_abs)
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise SystemExit("该回合动作全为 0，MDP 无有效帧（raw 仍已保存）")
        frame_index = np.nonzero(keep)[0].astype(np.int32)
    else:
        n_keep = n
        frame_index = np.arange(n, dtype=np.int32)

    act_cfg = shared.get("action") or {}
    payload: dict[str, Any] = {
        # 与 raw 同名：归一化动作（维度=任务动作空间）
        "cmd_action": ac[frame_index].astype(np.float32),
        "source": src[frame_index],
        "fps": np.asarray(data.get("fps", shared.get("fps", 20)), dtype=np.float32),
        "action_scale_m": np.asarray(
            data.get("action_scale_m", act_cfg.get("action_scale_m", 0.0015)), dtype=np.float32
        ),
        "action_scale_deg": np.asarray(
            data.get("action_scale_deg", act_cfg.get("action_scale_deg", 2.0)), dtype=np.float32
        ),
        "action_scale_gripper_m": np.asarray(
            data.get("action_scale_gripper_m", act_cfg.get("action_scale_gripper_m", 0.002)),
            dtype=np.float32,
        ),
        "episode_index": np.array(int(episode_index), dtype=np.int32),
        "raw_episode": np.array(raw_episode_name),
        "frame_index": frame_index,
        "action_index": np.asarray(aidx, dtype=np.int32),
        "action_names": np.asarray(anames),
        "observation_keys": np.asarray(keys),
    }
    for k in keys:
        if k not in data:
            raise SystemExit(f"raw 缺 {k}，无法写 MDP")
        payload[k] = np.asarray(data[k], dtype=np.float32)[frame_index]
    for phys_key in (
        "physical_delta_xyz_m",
        "physical_delta_rpy_rad",
        "physical_delta_grip_m",
        "cmd_xyz",
        "cmd_rpy",
        "cmd_grip",
        "leader_eef_xyz",
        "leader_eef_rpy",
        "leader_gripper",
    ):
        if phys_key in data:
            payload[phys_key] = np.asarray(data[phys_key], dtype=np.float32)[frame_index]
    for k, v in data.items():
        if k.startswith("init_"):
            payload[k] = np.asarray(v, dtype=np.float32)

    stats = {"raw_frames": n, "kept_frames": n_keep, "dropped_zero": n - n_keep}
    return payload, stats


def save_mdp_episode(
    ep_dir: Path,
    payload: dict[str, Any],
    *,
    shared: dict[str, Any],
    stats: dict[str, int] | None = None,
    raw_ep_dir: Path | None = None,
) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    # 图像：若 observation.images.* 打开，从 raw 拷贝对应帧
    img_roles = images_enabled(shared)
    frame_index = np.asarray(payload.get("frame_index"), dtype=np.int32).reshape(-1)
    if img_roles and raw_ep_dir is not None:
        import cv2

        for role in img_roles:
            src_dir = Path(raw_ep_dir) / f"images_{role}"
            if not src_dir.is_dir():
                continue
            dst_dir = ep_dir / f"images_{role}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for j, src_i in enumerate(frame_index.tolist()):
                src = src_dir / f"{int(src_i):06d}.png"
                if not src.is_file():
                    continue
                img = cv2.imread(str(src))
                if img is None:
                    continue
                cv2.imwrite(str(dst_dir / f"{j:06d}.png"), img)

    np_payload = {k: v for k, v in payload.items()}
    np.savez_compressed(ep_dir / "data.npz", **np_payload)

    keys = list(np.asarray(payload.get("observation_keys", [])).tolist())
    anames = list(np.asarray(payload.get("action_names", [])).tolist())
    n = int(np.asarray(payload["cmd_action"]).shape[0])
    adim = int(np.asarray(payload["cmd_action"]).shape[-1])
    sdim = int(sum(np.asarray(payload[k]).reshape(n, -1).shape[-1] for k in keys))
    meta = {
        "n": n,
        "observation_keys": keys,
        "action_names": anames,
        "state_dim": sdim,
        "action_dim": adim,
        "fps": float(np.asarray(payload.get("fps", 20)).reshape(-1)[0]),
        "format": "trajectory_like_raw",
        "note": "与 raw 同结构；字段子集；已滤静止帧；cmd_action 已归一化。",
    }
    if stats:
        meta.update(stats)
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    drop = (stats or {}).get("dropped_zero", 0)
    print(
        f"[mdp] 保存 {ep_dir}  steps={n}"
        + (f"  (丢掉静止 {drop})" if drop else ""),
        flush=True,
    )


def save_mdp_from_raw_episode(
    raw_ep_dir: Path,
    shared: dict[str, Any],
    *,
    mdp_root_path: Path | None = None,
    drop_zero_actions: bool = True,
) -> Path:
    data = load_raw(raw_ep_dir)
    ep_name = Path(raw_ep_dir).name
    try:
        episode_index = int(ep_name.split("_")[-1])
    except ValueError:
        episode_index = 0
    payload, stats = project_raw_data_to_mdp(
        data,
        shared,
        raw_episode_name=ep_name,
        episode_index=episode_index,
        drop_zero_actions=drop_zero_actions,
    )
    out_root = mdp_root(shared, mdp_root_path)
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / ep_name
    if dest.exists():
        shutil.rmtree(dest)
    save_mdp_episode(dest, payload, shared=shared, stats=stats, raw_ep_dir=Path(raw_ep_dir))
    return dest


def list_mdp_episodes(root: Path | None = None) -> list[Path]:
    root = Path(root or MDP_ROOT)
    # 兼容误放在 mdp/current/ 下的旧数据
    if root.is_dir() and not any(root.glob("episode_*")) and (root / "current").is_dir():
        root = root / "current"
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("episode_*") if (p / "data.npz").is_file())


def load_mdp(ep_dir: Path) -> dict[str, Any]:
    z = np.load(Path(ep_dir) / "data.npz", allow_pickle=False)
    data = {k: z[k] for k in z.files}
    meta_p = Path(ep_dir) / "meta.json"
    if meta_p.is_file():
        data["_meta"] = json.loads(meta_p.read_text(encoding="utf-8"))
    return data


def load_mdp_pairs(
    root: Path | None = None,
    *,
    sources: tuple[str, ...] = (SOURCE_DEMO,),
    skip_idle: bool = False,
    idle_abs: float = ZERO_ACTION_ABS,
    shared: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """把轨迹字段拼成 (state, action) 供 BC。兼容旧 observation.state/action 格式。"""
    root = Path(root or (mdp_root(shared) if shared else MDP_ROOT))
    eps = list_mdp_episodes(root)
    if not eps:
        raise SystemExit(f"没有 MDP：{root}。请先 python scripts/record_demo.py")
    want = set(sources)
    keys = obs_enabled(shared) if shared else None
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    n_raw = 0
    n_keep = 0
    for ep in eps:
        z = np.load(ep / "data.npz", allow_pickle=True)
        files = set(z.files)
        if "cmd_action" in files:
            ac = np.asarray(z["cmd_action"], dtype=np.float32)
            if keys is None:
                if "observation_keys" not in files:
                    raise SystemExit(f"{ep} 缺 observation_keys，且未传入 shared")
                keys = list(np.asarray(z["observation_keys"]).tolist())
            st_list = []
            n = int(ac.shape[0])
            for t in range(n):
                row = {k: z[k][t] for k in keys}
                st_list.append(pack_state(row, keys))
            st = np.stack(st_list, axis=0)
        elif "observation.state" in files and "action" in files:
            # 旧格式
            st = np.asarray(z["observation.state"], dtype=np.float32)
            ac = np.asarray(z["action"], dtype=np.float32)
        else:
            raise SystemExit(f"{ep} 既无 cmd_action 也无 observation.state")
        src = np.asarray(z["source"]).astype(str).reshape(-1) if "source" in files else np.array([SOURCE_DEMO] * len(ac))
        n = min(len(st), len(ac), len(src))
        n_raw += n
        for i in range(n):
            if src[i] not in want:
                continue
            a = ac[i]
            if skip_idle and float(np.linalg.norm(a)) < idle_abs:
                continue
            states.append(st[i])
            actions.append(a)
            n_keep += 1
    if not states:
        raise SystemExit(f"MDP 里没有 source={sorted(want)} 的帧")
    return (
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        {"episodes": len(eps), "raw_frames": n_raw, "kept_frames": n_keep},
    )


# ---------------------------------------------------------------------------
# BC 三种观测模式
# ---------------------------------------------------------------------------

OBS_MODES = ("image", "image_eef", "eef")


def resolve_bc_obs_mode(mode: str) -> str:
    m = str(mode or "eef").strip().lower().replace("-", "_")
    aliases = {
        "img": "image",
        "images": "image",
        "image_only": "image",
        "img_eef": "image_eef",
        "image_state": "image_eef",
        "vision_state": "image_eef",
        "state": "eef",
        "eef_xyz": "eef",
        "xyz": "eef",
    }
    m = aliases.get(m, m)
    if m not in OBS_MODES:
        raise SystemExit(f"--obs-mode 应为 {OBS_MODES}，收到: {mode}")
    return m


def image_feature_key(role: str) -> str:
    return f"observation.images.{role}"


def load_rgb_chw(
    path: Path,
    *,
    resize_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """读 PNG → float32 CHW ∈ [0,1]。resize_hw=(H,W)。"""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读图: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize_hw is not None:
        rh, rw = int(resize_hw[0]), int(resize_hw[1])
        rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


def index_mdp_bc_frames(
    root: Path | None = None,
    *,
    sources: tuple[str, ...] = (SOURCE_DEMO,),
    skip_idle: bool = True,
    idle_abs: float = ZERO_ACTION_ABS,
    shared: dict[str, Any] | None = None,
    obs_mode: str = "eef",
    image_role: str = "wrist",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """列出 BC 可用帧索引（不预加载图像，省内存）。

    每条: {ep_dir, t, action, eef_xyz?}
    """
    mode = resolve_bc_obs_mode(obs_mode)
    need_img = mode in ("image", "image_eef")
    need_eef = mode in ("eef", "image_eef")
    root = Path(root or (mdp_root(shared) if shared else MDP_ROOT))
    eps = list_mdp_episodes(root)
    if not eps:
        raise SystemExit(f"没有 MDP：{root}")

    want = set(sources)
    frames: list[dict[str, Any]] = []
    n_raw = 0
    for ep in eps:
        z = np.load(ep / "data.npz", allow_pickle=True)
        files = set(z.files)
        if "cmd_action" not in files:
            raise SystemExit(f"{ep} 缺 cmd_action")
        ac = np.asarray(z["cmd_action"], dtype=np.float32)
        n = int(ac.shape[0])
        src = (
            np.asarray(z["source"]).astype(str).reshape(-1)
            if "source" in files
            else np.array([SOURCE_DEMO] * n)
        )
        eef = None
        if need_eef:
            if "eef_xyz" not in files:
                raise SystemExit(f"{ep} 缺 eef_xyz（末端位置），无法用 obs_mode={mode}")
            eef = np.asarray(z["eef_xyz"], dtype=np.float32).reshape(n, -1)
            if eef.shape[-1] < 3:
                raise SystemExit(f"{ep} eef_xyz 维数异常: {eef.shape}")
            eef = eef[:, :3]
        img_dir = ep / f"images_{image_role}"
        if need_img and not img_dir.is_dir():
            raise SystemExit(f"{ep} 缺 {img_dir.name}/，无法用图像模式")

        n_raw += n
        for i in range(n):
            if src[i] not in want:
                continue
            a = ac[i]
            if skip_idle and float(np.linalg.norm(a)) < idle_abs:
                continue
            if need_img and not (img_dir / f"{i:06d}.png").is_file():
                continue
            row: dict[str, Any] = {"ep_dir": ep, "t": i, "action": a.astype(np.float32)}
            if need_eef and eef is not None:
                row["eef_xyz"] = eef[i].astype(np.float32)
            if need_img:
                row["image_path"] = img_dir / f"{i:06d}.png"
            frames.append(row)

    if not frames:
        raise SystemExit(
            f"MDP 无可用帧 mode={mode} source={sorted(want)}。"
            "请确认已录 wrist 图像且 eef_xyz 存在。"
        )
    meta = {
        "episodes": len(eps),
        "raw_frames": n_raw,
        "kept_frames": len(frames),
        "obs_mode": mode,
        "image_role": image_role if need_img else "",
    }
    return frames, meta
