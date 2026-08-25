#!/usr/bin/env python3
"""BC：三种观测模式训官方 GaussianActorPolicy（动作统一为 Δxyz）。

观测模式 ``--obs-mode``::

    eef         只用末端位置 eef_xyz (3)
    image       只用腕部图像
    image_eef   图像 + eef_xyz

用法::

    python scripts/train_bc.py --obs-mode eef
    python scripts/train_bc.py --obs-mode image
    python scripts/train_bc.py --obs-mode image_eef --steps 30000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_hilserl.gaussian_actor import apply_obs_mode_to_policy_cfg, make_gaussian_actor
from lerobot_hilserl.mdp_io import (
    image_feature_key,
    index_mdp_bc_frames,
    load_rgb_chw,
    mdp_root,
    resolve_bc_obs_mode,
)
from lerobot_hilserl.paths import PROJECT as PROJECT_ROOT, SHARED_CFG, SOURCE_DEMO

DEFAULT_TRAIN_CFG = PROJECT_ROOT / "cfg" / "train_config.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "bc"


def _minmax(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0


def write_pretrained_path(train_cfg_path: Path, pretrained: Path) -> None:
    cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("policy", {})["pretrained_path"] = str(pretrained)
    train_cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class MdpBCDataset(Dataset):
    def __init__(
        self,
        frames: list[dict],
        *,
        obs_mode: str,
        image_key: str,
        resize_hw: tuple[int, int],
    ) -> None:
        self.frames = frames
        self.obs_mode = resolve_bc_obs_mode(obs_mode)
        self.image_key = image_key
        self.resize_hw = resize_hw
        self.need_img = self.obs_mode in ("image", "image_eef")
        self.need_eef = self.obs_mode in ("eef", "image_eef")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frames[idx]
        out: dict[str, torch.Tensor] = {
            ACTION: torch.from_numpy(np.asarray(row["action"], dtype=np.float32)),
        }
        if self.need_eef:
            out[OBS_STATE] = torch.from_numpy(np.asarray(row["eef_xyz"], dtype=np.float32).reshape(3))
        if self.need_img:
            img = load_rgb_chw(Path(row["image_path"]), resize_hw=self.resize_hw)
            out[self.image_key] = torch.from_numpy(img)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="BC 预训练 gaussian_actor（三种观测）")
    parser.add_argument(
        "--obs-mode",
        default="eef",
        choices=("eef", "image", "image_eef"),
        help="eef=末端xyz | image=仅图像 | image_eef=图像+xyz",
    )
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CFG)
    parser.add_argument("--mdp-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="默认 outputs/bc_<mode>")
    parser.add_argument("--image-role", default="wrist")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-idle", action="store_true", help="保留空动作帧（默认丢掉）")
    parser.add_argument("--no-write-config", action="store_true")
    args = parser.parse_args()

    obs_mode = resolve_bc_obs_mode(args.obs_mode)
    train_cfg_path = Path(args.train_config).resolve()
    if not train_cfg_path.is_file():
        raise SystemExit(f"没有训练配置：{train_cfg_path}")
    train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    shared = json.loads(SHARED_CFG.read_text(encoding="utf-8"))
    mdp_dir = Path(args.mdp_root) if args.mdp_root else mdp_root(shared)
    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "outputs" / f"bc_{obs_mode}")
    resize_hw = (int(args.image_size[0]), int(args.image_size[1]))
    img_key = image_feature_key(str(args.image_role))

    frames, meta = index_mdp_bc_frames(
        mdp_dir,
        sources=(SOURCE_DEMO,),
        skip_idle=not args.keep_idle,
        shared=shared,
        obs_mode=obs_mode,
        image_role=str(args.image_role),
    )
    actions = np.stack([f["action"] for f in frames], axis=0)
    action_dim = int(actions.shape[-1])
    state_dim = 3
    state_min = state_max = None
    if obs_mode in ("eef", "image_eef"):
        eefs = np.stack([f["eef_xyz"] for f in frames], axis=0)
        state_min = eefs.min(axis=0).tolist()
        state_max = eefs.max(axis=0).tolist()

    apply_obs_mode_to_policy_cfg(
        train_cfg,
        obs_mode=obs_mode,
        state_dim=state_dim,
        action_dim=action_dim,
        image_key=img_key,
        image_shape=(3, resize_hw[0], resize_hw[1]),
        state_min=state_min,
        state_max=state_max,
    )

    print(
        f"obs_mode={obs_mode}  mdp={mdp_dir}  episodes={meta['episodes']}  "
        f"kept={meta['kept_frames']}  action_dim={action_dim}",
        flush=True,
    )
    if obs_mode in ("eef", "image_eef"):
        print(f"  eef_xyz range min={state_min} max={state_max}", flush=True)
    if obs_mode in ("image", "image_eef"):
        print(f"  image={img_key} resize={resize_hw}", flush=True)

    policy = make_gaussian_actor(train_cfg, device=device)
    policy.actor.encoder_is_shared = False
    policy.train()

    pol = train_cfg["policy"]
    stats = pol["dataset_stats"]
    lo_a = torch.tensor(stats[ACTION]["min"], device=device, dtype=torch.float32)
    hi_a = torch.tensor(stats[ACTION]["max"], device=device, dtype=torch.float32)
    lo_s = hi_s = None
    if OBS_STATE in stats:
        lo_s = torch.tensor(stats[OBS_STATE]["min"], device=device, dtype=torch.float32)
        hi_s = torch.tensor(stats[OBS_STATE]["max"], device=device, dtype=torch.float32)

    ds = MdpBCDataset(frames, obs_mode=obs_mode, image_key=img_key, resize_hw=resize_hw)
    bs = min(int(args.batch_size), len(ds))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        drop_last=len(ds) > bs,
        num_workers=int(args.num_workers) if obs_mode != "eef" else 0,
        pin_memory=device.startswith("cuda"),
    )
    params = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr)

    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    print(f"BC steps={args.steps} device={device} batch={bs} out={out_dir}", flush=True)

    step = 0
    data_iter = iter(loader)
    running = 0.0
    last_loss = 0.0
    pbar = tqdm(total=args.steps, desc=f"BC/{obs_mode}")

    def _save(tag: str, mse: float) -> Path:
        save_dir = ckpt_root / tag / "pretrained_model"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        policy.actor.encoder_is_shared = bool(pol.get("shared_encoder", True))
        policy.save_pretrained(save_dir)
        policy.actor.encoder_is_shared = False
        (save_dir / "bc_meta.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "bc_mse": mse,
                    "obs_mode": obs_mode,
                    "mdp_root": str(mdp_dir.resolve()),
                    "kept_frames": meta["kept_frames"],
                    "action_dim": action_dim,
                    "image_key": img_key if obs_mode != "eef" else None,
                    "image_size": list(resize_hw) if obs_mode != "eef" else None,
                    "train_config": str(train_cfg_path),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        # 同步一份当时的 policy 配置，方便 eval 对齐维数
        (save_dir / "bc_train_policy.json").write_text(
            json.dumps(pol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"saved {save_dir}", flush=True)
        return save_dir

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        act = batch[ACTION].to(device=device, dtype=torch.float32)
        obs: dict[str, torch.Tensor] = {}
        if OBS_STATE in batch and lo_s is not None and hi_s is not None:
            st = batch[OBS_STATE].to(device=device, dtype=torch.float32)
            obs[OBS_STATE] = _minmax(st, lo_s, hi_s)
        if img_key in batch:
            # [0,1] CHW；DefaultImageEncoder 直接吃像素
            obs[img_key] = batch[img_key].to(device=device, dtype=torch.float32)

        target = _minmax(act, lo_a, hi_a)
        pred = policy.actor.mode(obs)
        loss = F.mse_loss(pred, target)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optim.step()

        step += 1
        last_loss = float(loss.item())
        running += last_loss
        pbar.update(1)
        pbar.set_postfix(loss=f"{last_loss:.4f}")

        if step % args.log_freq == 0:
            print(f"\nstep={step} bc_mse={running / args.log_freq:.6f}", flush=True)
            running = 0.0

        if step % args.save_freq == 0:
            _save(f"{step:07d}", last_loss)

    pbar.close()
    last = _save("last", last_loss)

    loaded = GaussianActorPolicy.from_pretrained(last)
    n_ok = sum(
        int(torch.allclose(a, b))
        for (a, b) in zip(policy.state_dict().values(), loaded.state_dict().values(), strict=True)
        if a.shape == b.shape
    )
    print(f"[check] from_pretrained 可加载  tensors_close≈{n_ok}/{len(loaded.state_dict())}", flush=True)

    if not args.no_write_config:
        write_pretrained_path(train_cfg_path, last)
        print(f"已写入 {train_cfg_path}  policy.pretrained_path={last}", flush=True)

    print("\n========== BC done ==========", flush=True)
    print(f"mode={obs_mode}\n  {last}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
