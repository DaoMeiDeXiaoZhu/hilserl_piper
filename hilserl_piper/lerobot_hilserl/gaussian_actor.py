"""从本仓库 train_config 构建官方 GaussianActorPolicy，保证 BC / SAC 同一套权重格式。"""

from __future__ import annotations

from typing import Any

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorNetworkConfig,
    GaussianActorConfig,
    PolicyConfig,
    is_image_feature,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def policy_cfg_from_train_json(train_cfg: dict[str, Any], *, device: str) -> GaussianActorConfig:
    p = train_cfg["policy"]
    actor_kw = p.get("actor_network_kwargs") or {"hidden_dims": [256, 256], "activate_final": True}
    pol_kw = p.get("policy_kwargs") or {}
    in_feats = {}
    for k, v in (p.get("input_features") or {}).items():
        in_feats[k] = PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
    out_feats = {}
    for k, v in (p.get("output_features") or {}).items():
        out_feats[k] = PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))

    has_img = any(is_image_feature(k) for k in in_feats)
    has_state = OBS_STATE in in_feats
    if not has_img and not has_state:
        raise SystemExit("policy.input_features 需要 observation.state 和/或 observation.images.*")
    if ACTION not in out_feats:
        raise SystemExit("policy.output_features 需要 action")

    return GaussianActorConfig(
        n_obs_steps=int(p.get("n_obs_steps", 1)),
        normalization_mapping=p.get("normalization_mapping"),
        dataset_stats=p.get("dataset_stats"),
        input_features=in_feats,
        output_features=out_feats,
        device=device,
        storage_device=str(p.get("storage_device") or "cpu"),
        push_to_hub=False,
        vision_encoder_name=p.get("vision_encoder_name"),
        freeze_vision_encoder=bool(p.get("freeze_vision_encoder", True)),
        image_encoder_hidden_dim=int(p.get("image_encoder_hidden_dim", 32)),
        shared_encoder=bool(p.get("shared_encoder", True)),
        num_discrete_actions=p.get("num_discrete_actions"),
        image_embedding_pooling_dim=int(p.get("image_embedding_pooling_dim", 8)),
        state_encoder_hidden_dim=int(p.get("state_encoder_hidden_dim", 256)),
        latent_dim=int(p.get("latent_dim", 64)),
        actor_network_kwargs=ActorNetworkConfig(
            hidden_dims=list(actor_kw.get("hidden_dims") or [256, 256]),
            activate_final=bool(actor_kw.get("activate_final", True)),
        ),
        policy_kwargs=PolicyConfig(
            use_tanh_squash=bool(pol_kw.get("use_tanh_squash", True)),
            std_min=float(pol_kw.get("std_min", -5.0)),
            std_max=float(pol_kw.get("std_max", 2.0)),
            init_final=float(pol_kw.get("init_final", 0.05)),
        ),
    )


def apply_obs_mode_to_policy_cfg(
    train_cfg: dict[str, Any],
    *,
    obs_mode: str,
    state_dim: int,
    action_dim: int,
    image_key: str,
    image_shape: tuple[int, int, int],
    state_min: list[float] | None = None,
    state_max: list[float] | None = None,
) -> dict[str, Any]:
    """按 BC 观测模式改写 policy.input_features / dataset_stats / output 维数。"""
    from .mdp_io import resolve_bc_obs_mode

    mode = resolve_bc_obs_mode(obs_mode)
    pol = train_cfg.setdefault("policy", {})
    in_feats: dict[str, Any] = {}
    stats: dict[str, Any] = dict(pol.get("dataset_stats") or {})

    if mode in ("image", "image_eef"):
        in_feats[image_key] = {"type": "VISUAL", "shape": list(image_shape)}
        stats[image_key] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        # DefaultImageEncoder 需要训练
        if not pol.get("vision_encoder_name"):
            pol["freeze_vision_encoder"] = False

    if mode in ("eef", "image_eef"):
        in_feats[OBS_STATE] = {"type": "STATE", "shape": [int(state_dim)]}
        lo = state_min if state_min is not None else [0.0] * int(state_dim)
        hi = state_max if state_max is not None else [1.0] * int(state_dim)
        stats[OBS_STATE] = {"min": list(lo), "max": list(hi)}

    # 清掉未用的旧 state/image stats，避免维数残留
    for k in list(stats.keys()):
        if k.startswith("observation.images.") and k not in in_feats:
            stats.pop(k, None)
        if k == OBS_STATE and OBS_STATE not in in_feats:
            stats.pop(k, None)

    ad = int(action_dim)
    stats[ACTION] = {"min": [-1.0] * ad, "max": [1.0] * ad}
    pol["input_features"] = in_feats
    pol["output_features"] = {ACTION: {"type": "ACTION", "shape": [ad]}}
    pol["dataset_stats"] = stats
    return train_cfg


def make_gaussian_actor(train_cfg: dict[str, Any], *, device: str) -> GaussianActorPolicy:
    cfg = policy_cfg_from_train_json(train_cfg, device=device)
    policy = GaussianActorPolicy(cfg)
    policy.to(device)
    return policy
