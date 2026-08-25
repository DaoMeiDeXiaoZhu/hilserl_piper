"""丢掉动作全为 0 的单步经验，保留其余轨迹。不改 hilserl_piper / lerobot 源码。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .paths import DEMO_ROOT

# 录制里真正的零动作是 float32 全 0；用极小阈值避免浮点噪声误伤 1mm 步
ZERO_ABS = 1e-8


def is_zero_action(action: Any, *, abs_tol: float = ZERO_ABS) -> bool:
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return True
    return bool(np.max(np.abs(a)) <= abs_tol)


def nonzero_mask(actions: np.ndarray, *, abs_tol: float = ZERO_ABS) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return np.max(np.abs(a), axis=-1) > abs_tol


def filter_record_buffer(buf: Any, *, abs_tol: float = ZERO_ABS) -> int:
    """就地删掉 EpisodeBuffer 里的零动作帧。返回删除条数。"""
    n = len(buf.actions)
    keep = [i for i, act in enumerate(buf.actions) if not is_zero_action(act, abs_tol=abs_tol)]
    dropped = n - len(keep)
    if dropped <= 0:
        return 0
    buf.actions = [buf.actions[i] for i in keep]
    if getattr(buf, "physical_delta", None) is not None:
        buf.physical_delta = [buf.physical_delta[i] for i in keep]
    for key, series in list(buf.obs.items()):
        buf.obs[key] = [series[i] for i in keep]
    return dropped


def _index_keep(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(arr)[mask]


def strip_zero_actions_npz(ep_dir: Path, *, abs_tol: float = ZERO_ABS) -> dict[str, int]:
    """改写 episode_*/data.npz，删掉动作全 0 的帧。全是 0 则删除该回合目录。"""
    ep_dir = Path(ep_dir)
    npz_path = ep_dir / "data.npz"
    if not npz_path.is_file():
        return {"raw": 0, "kept": 0, "dropped": 0, "deleted": 0}

    data = dict(np.load(npz_path, allow_pickle=False))
    actions = np.asarray(data["actions"])
    n = int(actions.shape[0])
    mask = nonzero_mask(actions, abs_tol=abs_tol)
    kept = int(mask.sum())
    dropped = n - kept
    if dropped == 0:
        return {"raw": n, "kept": n, "dropped": 0, "deleted": 0}

    if kept == 0:
        import shutil

        shutil.rmtree(ep_dir)
        print(f"[zero] 删除全零回合 {ep_dir}", flush=True)
        return {"raw": n, "kept": 0, "dropped": dropped, "deleted": 1}

    out: dict[str, Any] = {}
    for key, val in data.items():
        arr = np.asarray(val)
        if arr.shape[:1] == (n,):
            out[key] = _index_keep(arr, mask)
        else:
            out[key] = arr
    np.savez_compressed(npz_path, **out)

    # 若有按帧存的 png，删掉被丢掉的编号太麻烦；本任务不开图
    print(f"[zero] {ep_dir.name}: {n} → {kept}（丢掉 {dropped} 帧零动作）", flush=True)
    return {"raw": n, "kept": kept, "dropped": dropped, "deleted": 0}


def strip_zero_actions_in_root(root: Path | None = None, *, abs_tol: float = ZERO_ABS) -> dict[str, int]:
    root = Path(root or DEMO_ROOT)
    if not root.is_dir():
        return {"episodes": 0, "raw": 0, "kept": 0, "dropped": 0, "deleted": 0}
    totals = {"episodes": 0, "raw": 0, "kept": 0, "dropped": 0, "deleted": 0}
    for ep in sorted(p for p in root.glob("episode_*") if p.is_dir()):
        if not (ep / "data.npz").is_file():
            continue
        totals["episodes"] += 1
        st = strip_zero_actions_npz(ep, abs_tol=abs_tol)
        for k in ("raw", "kept", "dropped", "deleted"):
            totals[k] += st[k]
    return totals


def filter_transition_list(transitions: Sequence[Any], *, abs_tol: float = ZERO_ABS) -> tuple[list[Any], int]:
    """SAC 回合结束发送/存盘前：丢掉零动作步，保留其余。"""
    kept: list[Any] = []
    dropped = 0
    for tr in transitions:
        if hasattr(tr, "action"):
            act = tr.action
        elif isinstance(tr, dict):
            act = tr.get("action")
            if act is None:
                from lerobot.utils.constants import ACTION

                act = tr.get(ACTION)
        else:
            kept.append(tr)
            continue
        if is_zero_action(act, abs_tol=abs_tol):
            dropped += 1
            continue
        kept.append(tr)
    return kept, dropped


def install_record_save_filter(rec: Any) -> None:
    orig = rec.save_episode

    def save_episode(ep_dir, buf, *args, **kwargs):
        dropped = filter_record_buffer(buf)
        if len(buf) == 0:
            rec.say("[save] 本回合动作全为 0，不写盘（不影响其它回合）")
            return None
        if dropped:
            rec.say(f"[save] 丢掉 {dropped} 帧零动作，保留 {len(buf)}")
        return orig(ep_dir, buf, *args, **kwargs)

    rec.save_episode = save_episode


def install_official_actor_send_filter() -> None:
    """官方 actor：回合结束才 push 到 learner；在这里丢掉零动作步。"""
    import lerobot.rl.actor as actor_mod

    orig = actor_mod.push_transitions_to_transport_queue

    def wrapped(transitions, transitions_queue):
        kept, dropped = filter_transition_list(transitions)
        if dropped:
            logging.info(
                "[ACTOR] 丢掉 %s 步零动作，本回合仍发送 %s 步（未丢整条轨迹）",
                dropped,
                len(kept),
            )
        if not kept:
            logging.info("[ACTOR] 本回合过滤后为空，不发送")
            return None
        return orig(kept, transitions_queue)

    actor_mod.push_transitions_to_transport_queue = wrapped


def install_hilserl_sac_save_filter() -> None:
    """hilserl_piper actor：回合结束 save_episode_npz 时丢掉零动作步。"""
    from hilserl_piper.algorithms.common import dataset_io as dio
    import hilserl_piper.algorithms.sac.actor as sac_actor

    orig = dio.save_episode_npz

    def wrapped(ep_dir, *, actions, **kwargs):
        act = np.asarray(actions)
        n = int(act.shape[0])
        mask = nonzero_mask(act)
        dropped = int(n - int(mask.sum()))
        if dropped:
            print(
                f"[sac-save] 丢掉 {dropped} 步零动作，保留 {int(mask.sum())}（未丢整条轨迹）",
                flush=True,
            )
        if not bool(mask.any()):
            print("[sac-save] 本回合动作全为 0，不写盘", flush=True)
            return None

        def _take(x):
            if x is None:
                return None
            arr = np.asarray(x) if not isinstance(x, list) else None
            if arr is not None and arr.shape[:1] == (n,):
                return arr[mask]
            if isinstance(x, list) and len(x) == n:
                return [x[i] for i, k in enumerate(mask) if k]
            return x

        kwargs = dict(kwargs)
        extra = kwargs.get("extra")
        if extra:
            extra = dict(extra)
            for k, v in list(extra.items()):
                extra[k] = _take(v)
            kwargs["extra"] = extra
        vec = kwargs.get("vec_obs")
        if vec:
            kwargs["vec_obs"] = {k: _take(v) for k, v in vec.items()}
        return orig(
            ep_dir,
            actions=act[mask],
            physical_delta_m=_take(kwargs.pop("physical_delta_m", None)),
            rewards=_take(kwargs.pop("rewards", None)),
            observations=_take(kwargs.pop("observations", None)),
            **kwargs,
        )

    dio.save_episode_npz = wrapped
    sac_actor.save_episode_npz = wrapped
