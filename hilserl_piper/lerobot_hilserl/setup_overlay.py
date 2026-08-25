"""覆盖 hilserl_piper setup 的默认：仅从臂、键盘 Δxyz、无夹爪。

固定：
  - 控制：follower_only（不询问）
  - 键盘步进：1mm
  - SAC/动作归一化 action_scale：1.5mm
  - 无夹爪动作/观测；示教时夹爪保持闭合下发

不改 hilserl_piper 源码；由 ``scripts/setup.py`` 在加载 setup 模块后打补丁。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import CFG_DIR, HARDWARE_CFG, SHARED_CFG

XYZ_NAMES = ["dx_eff", "dy_eff", "dz_eff"]
KEYBOARD_STEP_M = 0.001  # 1mm
ACTION_SCALE_MM = 1.5  # SAC / 归一化满量程

GYM_JSONS = (
    CFG_DIR / "record_config_piper_hilserl.json",
    CFG_DIR / "train_config_piper_hilserl.json",
)


def apply_patches(mod: Any) -> dict[str, Any]:
    """把本仓库默认打进 hilserl_piper 的 setup 模块。"""
    ctx: dict[str, Any] = {}

    orig_bind = mod.bind_can_ports
    orig_sync = mod.sync_robot_shared
    orig_build_hw = mod.build_hardware
    orig_ws = mod.record_workspace_and_reset
    orig_help = mod.print_observation_help
    orig_save = mod.save_json

    def choose_control_mode(existing: str | None = None) -> str:
        del existing
        mod.say("=" * 56)
        mod.say("任务控制：仅从臂 (follower_only) + 键盘开环 Δxyz（不询问）")
        return "follower_only"

    def bind_can_ports(
        mode: str,
        existing_follower: str | None = None,
        existing_leader: str | None = None,
    ) -> tuple[str, str | None]:
        follower, _ = orig_bind(
            "follower_only",
            existing_follower=existing_follower,
            existing_leader=None,
        )
        mod.say(f"  follower_can={follower}  （无主臂）")
        return follower, None

    def ask_action_scale_mm(shared: dict[str, Any]) -> float:
        del shared
        mod.say("=" * 56)
        mod.say("动作空间（固定，不询问）:")
        mod.say(f"  键盘步进 keyboard_ee_step_m = {KEYBOARD_STEP_M * 1000:.1f}mm")
        mod.say(f"  归一化满量程 action_scale = {ACTION_SCALE_MM:.1f}mm  （a∈[-1,1]）")
        mod.say("  动作=Δxyz 三维；夹爪不控制（示教时保持闭合）")
        return float(ACTION_SCALE_MM)

    def sync_robot_shared(
        shared: dict[str, Any],
        roles_out: dict[str, dict[str, Any]],
        *,
        fps: int,
        action_scale_mm: float,
        workspace: dict[str, Any] | None,
        cameras_retuned: bool,
    ) -> dict[str, Any]:
        out = orig_sync(
            shared,
            roles_out,
            fps=fps,
            action_scale_mm=action_scale_mm,
            workspace=workspace,
            cameras_retuned=cameras_retuned,
        )
        scale_m = float(action_scale_mm) / 1000.0
        action = out.setdefault("action", {})
        action.clear()
        action.update(
            {
                "type": "delta_ee_xyz",
                "include_rpy": False,
                "include_gripper": False,
                "action_scale_mm": float(action_scale_mm),
                "action_scale_m": scale_m,
                "names": list(XYZ_NAMES),
                "space": "[-1, 1]",
                "physical": "delta_m = a * action_scale_m",
            }
        )
        # 去掉历史夹爪/旋转 scale 字段
        for k in (
            "action_scale_deg",
            "action_scale_gripper_mm",
            "action_scale_gripper_m",
        ):
            action.pop(k, None)

        obs = out.setdefault("observation", {})
        obs["gripper_gap"] = False
        obs["eef_rpy"] = bool(obs.get("eef_rpy", True))  # 可观测，但不进动作
        ws = out.setdefault("workspace", {})
        ws["include_rpy"] = False

        kb = out.setdefault("keyboard", {})
        kb["keyboard_ee_control"] = True
        kb["keyboard_ee_step_m"] = float(KEYBOARD_STEP_M)
        kb["hold_s"] = float(kb.get("hold_s") or 0.06)
        kb["_comment_step"] = "每 fps 步最多平移这么多（米）。键盘固定 1mm。"
        kb.pop("keyboard_control_hz", None)

        out["teleop"] = {
            "type": "keyboard",
            "use_gripper": False,
            "_comment": "仅从臂 + 键盘 Δxyz；夹爪保持闭合，不进动作空间。",
        }
        layout = out.setdefault("dataset_layout", {})
        layout.setdefault("root", str(SHARED_CFG.parent.parent / "datasets"))
        layout.setdefault("raw", "raw")
        layout.setdefault("mdp", "mdp")
        layout.pop("mdp_task", None)
        layout.setdefault("demo", "demo")
        layout.setdefault("sac", "sac")
        layout.setdefault("classifier", "classifier")
        out["_comment"] = (
            "键盘开环 Δxyz。step=1mm，action_scale=1.5mm。"
            "无夹爪动作/观测；示教时夹爪闭合。"
        )
        return out

    def build_hardware(
        control_mode: str,
        follower_can: str,
        leader_can: str | None,
        roles_out: dict[str, dict[str, Any]],
        template: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = orig_build_hw("follower_only", follower_can, None, roles_out, template)
        _stamp_hardware_control(out, "follower_only", follower_can, None)
        return out

    def record_workspace_and_reset(follower_can: str) -> dict[str, Any] | None:
        mod.say("=" * 56)
        mod.say("工作区：建议跳过（--skip-workspace），沿用 cfg 已有 bounds。")
        mod.say("  此处若继续，仍是绿灯拖动扫 XYZ（与键盘开环姿态可能不一致）。")
        return orig_ws(follower_can)

    def print_observation_help(obs: dict[str, Any]) -> None:
        orig_help(obs)
        mod.say("  夹爪：已禁用（observation.gripper_gap=false，动作无夹爪维）")

    def save_json(path, cfg: dict[str, Any]) -> None:
        p = Path(path)
        if p.resolve() == HARDWARE_CFG.resolve() or p.name == "hardware.json":
            ctrl = cfg.get("control") or {}
            _stamp_hardware_control(
                cfg,
                "follower_only",
                str(ctrl.get("follower_can") or "can0"),
                None,
            )
        orig_save(path, cfg)

    mod.choose_control_mode = choose_control_mode
    mod.bind_can_ports = bind_can_ports
    mod.ask_action_scale_mm = ask_action_scale_mm
    mod.sync_robot_shared = sync_robot_shared
    mod.build_hardware = build_hardware
    mod.record_workspace_and_reset = record_workspace_and_reset
    mod.print_observation_help = print_observation_help
    mod.save_json = save_json

    flags = dict(getattr(mod, "DEFAULT_OBS_FLAGS", {}))
    flags["gripper_gap"] = False
    mod.DEFAULT_OBS_FLAGS = flags
    return ctx


def _stamp_hardware_control(
    hw: dict[str, Any],
    mode: str,
    follower_can: str,
    leader_can: str | None,
) -> None:
    del mode, leader_can
    hw["_comment"] = "本机硬件。运行: python scripts/setup.py"
    hw["control"] = {
        "_comment": "仅从臂 + 键盘。follower 执行。",
        "mode": "follower_only",
        "follower_can": follower_can,
        "leader_can": None,
    }


def restamp_hardware_file(path: Path) -> None:
    if not path.is_file():
        return
    hw = json.loads(path.read_text(encoding="utf-8"))
    ctrl = hw.get("control") or {}
    _stamp_hardware_control(
        hw,
        "follower_only",
        str(ctrl.get("follower_can") or "can0"),
        None,
    )
    path.write_text(json.dumps(hw, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _env_block(cfg: dict[str, Any]) -> dict[str, Any]:
    env = cfg.get("env")
    if isinstance(env, dict) and ("robot" in env or "teleop" in env):
        return env
    return cfg


def sync_gym_configs(hw: dict[str, Any], shared: dict[str, Any]) -> None:
    """把从臂 CAN、3D Δxyz、无夹爪写进 gym record/train json。"""
    ctrl = hw.get("control") or {}
    follower = str(ctrl.get("follower_can") or "can0")
    act = shared.get("action") or {}
    scale_m = float(act.get("action_scale_m") or ACTION_SCALE_MM / 1000.0)

    for path in GYM_JSONS:
        if not path.is_file():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        env = _env_block(cfg)
        robot = env.setdefault("robot", {})
        robot["can_name"] = follower
        # 键盘任务：gym teleop 置空，避免误开主臂/夹爪
        env["teleop"] = None
        proc = env.setdefault("processor", {})
        grip = proc.setdefault("gripper", {})
        grip["use_gripper"] = False
        grip["force_closed"] = True
        feats = env.setdefault("features", {})
        if "action" in feats and isinstance(feats["action"], dict):
            feats["action"]["shape"] = [3]
        # state：eef_xyz(+eef_rpy) 不含 gripper；与 observation 开关对齐
        obs = shared.get("observation") or {}
        state_dim = 0
        if obs.get("eef_xyz"):
            state_dim += 3
        if obs.get("eef_rpy"):
            state_dim += 3
        if state_dim <= 0:
            state_dim = 3
        if "observation.state" in feats and isinstance(feats["observation.state"], dict):
            feats["observation.state"]["shape"] = [state_dim]
        proc["control_mode"] = "piper_delta_ee"
        # 若有显式 action scale 字段则同步
        if "ee_max_step_delta" in robot:
            robot["ee_max_step_delta"] = [scale_m, scale_m, scale_m]
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"已同步 gym  {path.name}  follower={follower}  "
            f"action=3DΔxyz scale={scale_m * 1000:.1f}mm  gripper=off",
            flush=True,
        )
