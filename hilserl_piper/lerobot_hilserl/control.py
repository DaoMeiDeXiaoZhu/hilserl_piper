"""主从控制。

- firmware_ms=true（共一线 CAN）：读主臂 JointCtrl，PC 以高跟随(0xAD)转发给从臂。
  （PiperFollower.connect 会抢走 CAN 位控，纯固件跟随不可靠；故用软件镜像。）
- 否则：软件绝对末端跟随（分 CAN）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .paths import HARDWARE_CFG, SHARED_CFG
from .piper_bridge import load_record_mod

GRIP_MAX_M = 0.08
HOME_HZ = 20.0
HOME_S = 4.0
MS_MIT = 0xAD  # 高跟随
JOINT_LIMITS = np.array(
    [
        [-2.6179, 2.6179],
        [0.0, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.09439, 2.09439],
    ],
    dtype=np.float32,
)


def wrap_pi(rad: np.ndarray) -> np.ndarray:
    x = np.asarray(rad, dtype=np.float32)
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def action_scales(shared: dict[str, Any]) -> dict[str, float]:
    act = shared.get("action") or {}
    return {
        "xyz_m": float(act.get("action_scale_m") or 0.15),
        "rpy_deg": float(act.get("action_scale_deg") or 45.0),
        "grip_m": float(act.get("action_scale_gripper_m") or 0.08),
    }


def is_abs_action(shared: dict[str, Any]) -> bool:
    return str((shared.get("action") or {}).get("type") or "").lower() in {
        "abs_ee_pose",
        "absolute_ee",
        "abs_eef",
    }


def pose_to_norm(
    xyz: np.ndarray,
    rpy: np.ndarray,
    grip: float,
    init: CmdPose | dict[str, Any],
    scales: dict[str, float],
) -> np.ndarray:
    """绝对位姿 → 相对 init 的归一化动作 ∈ [-1,1]。"""
    if isinstance(init, CmdPose):
        i_xyz, i_rpy, i_g = init.xyz, init.rpy, init.grip
    else:
        i_xyz = np.asarray(init.get("eef_xyz", init.get("xyz")), dtype=np.float32).reshape(3)
        i_rpy = np.asarray(init.get("eef_rpy", init.get("rpy")), dtype=np.float32).reshape(3)
        i_g = float(np.asarray(init.get("grip", init.get("gripper_gap", 0.0))).reshape(-1)[0])
    xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
    rpy = np.asarray(rpy, dtype=np.float32).reshape(3)
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip((xyz - i_xyz) / max(scales["xyz_m"], 1e-9), -1.0, 1.0)
    drpy_deg = wrap_pi(rpy - i_rpy) * (180.0 / np.pi)
    a[3:6] = np.clip(drpy_deg / max(scales["rpy_deg"], 1e-9), -1.0, 1.0)
    a[6] = float(np.clip((float(grip) - i_g) / max(scales["grip_m"], 1e-9), -1.0, 1.0))
    return a


def norm_to_pose(
    a: np.ndarray,
    init: CmdPose | dict[str, Any],
    scales: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(init, CmdPose):
        i_xyz, i_rpy, i_g = init.xyz, init.rpy, init.grip
    else:
        i_xyz = np.asarray(init.get("eef_xyz", init.get("xyz")), dtype=np.float32).reshape(3)
        i_rpy = np.asarray(init.get("eef_rpy", init.get("rpy")), dtype=np.float32).reshape(3)
        i_g = float(np.asarray(init.get("grip", 0.0)).reshape(-1)[0])
    a = np.clip(np.asarray(a, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if a.size < 7:
        a = np.pad(a, (0, 7 - a.size))
    xyz = i_xyz + a[:3] * scales["xyz_m"]
    rpy = wrap_pi(i_rpy + a[3:6] * scales["rpy_deg"] * (np.pi / 180.0))
    grip = float(i_g + a[6] * scales["grip_m"])
    return xyz.astype(np.float32), rpy.astype(np.float32), grip


# 兼容旧 delta API 名
def physical_to_norm(dxyz_m, drpy_rad, dgrip_m, scales):
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip(np.asarray(dxyz_m, np.float32).reshape(3) / max(scales["xyz_m"], 1e-9), -1, 1)
    a[3:6] = np.clip(
        np.asarray(drpy_rad, np.float32).reshape(3) * (180 / np.pi) / max(scales["rpy_deg"], 1e-9), -1, 1
    )
    a[6] = float(np.clip(float(dgrip_m) / max(scales["grip_m"], 1e-9), -1, 1))
    return a


def norm_to_physical(a, scales):
    a = np.clip(np.asarray(a, np.float32).reshape(-1), -1, 1)
    if a.size < 7:
        a = np.pad(a, (0, 7 - a.size))
    return (
        (a[:3] * scales["xyz_m"]).astype(np.float32),
        (a[3:6] * scales["rpy_deg"] * (np.pi / 180.0)).astype(np.float32),
        float(a[6] * scales["grip_m"]),
    )


@dataclass
class CmdPose:
    xyz: np.ndarray
    rpy: np.ndarray
    grip: float

    def copy(self) -> CmdPose:
        return CmdPose(self.xyz.copy(), self.rpy.copy(), float(self.grip))


@dataclass
class Runtime:
    rec: Any
    shared: dict[str, Any]
    hw: dict[str, Any]
    follower: Any
    reset: Any
    leader: Any | None = None
    cams: dict[str, Any] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    cmd: CmdPose | None = None
    prev_leader: CmdPose | None = None
    init_pose: CmdPose | None = None
    coupled: bool = False
    firmware_ms: bool = False  # 共 CAN：软件高跟随镜像主臂控制帧
    _ms_motion_ctrl: tuple[int, ...] | None = None
    _ms_no_ctrl_t: float = 0.0
    _ms_dbg_t: float = 0.0

    @property
    def fps(self) -> float:
        return float(self.shared.get("fps") or 20)

    @property
    def grip_lo(self) -> float:
        b = getattr(self.follower, "gripper_bound", [0.0, GRIP_MAX_M])
        return float(b[0])

    @property
    def grip_hi(self) -> float:
        b = getattr(self.follower, "gripper_bound", [0.0, GRIP_MAX_M])
        return float(b[1] if len(b) > 1 else GRIP_MAX_M)

    @property
    def piper(self) -> Any:
        return self.follower.bus.port_handler


def load_cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    shared = json.loads(SHARED_CFG.read_text(encoding="utf-8"))
    hw = json.loads(HARDWARE_CFG.read_text(encoding="utf-8"))
    return shared, hw


def _firmware_ms_enabled(hw: dict[str, Any]) -> bool:
    ctrl = hw.get("control") or {}
    if bool(ctrl.get("firmware_ms")):
        return True
    can = ctrl.get("can") or ctrl.get("follower_can")
    lead = ctrl.get("leader_can")
    return bool(can and lead and str(can) == str(lead))


def connect_runtime(*, cameras: bool = True, force_software: bool = False) -> Runtime:
    rec = load_record_mod()
    shared, hw = load_cfgs()
    ctrl = dict(hw.get("control") or {})
    firmware_ms = False if force_software else _firmware_ms_enabled(hw)
    can = str(ctrl.get("can") or ctrl.get("follower_can") or "can0")

    hw_use = dict(hw)
    ctrl_use = dict(ctrl)
    ctrl_use["follower_can"] = can
    if firmware_ms:
        ctrl_use["leader_can"] = can
    hw_use["control"] = ctrl_use
    hw_use["cameras"] = {}
    follower, reset = rec.connect_robot(shared, hw_use)
    follower.use_end_effector_bounds = False

    leader = None
    if not firmware_ms:
        lead_can = ctrl.get("leader_can")
        if str(ctrl.get("mode") or "") == "leader_follower" and lead_can and str(lead_can) != can:
            hw_lead = {"control": {"follower_can": str(lead_can)}, "cameras": {}}
            try:
                leader, _ = rec.connect_robot(shared, hw_lead)
                leader.use_end_effector_bounds = False
            except Exception as exc:
                rec.say(f"[warn] 主臂 {lead_can} 连接失败: {exc}")
                leader = None
    else:
        rec.say(
            f"[ms] 共 CAN 软件高跟随  can={can}  "
            "读主臂 JointCtrl → MotionCtrl_2(0xAD)+JointCtrl 写从臂"
        )
        try:
            follower.bus.port_handler.MasterSlaveConfig(0xFC, 0, 0, 0)
            rec.say("[ms] 已 MasterSlaveConfig(0xFC) 确认从臂角色")
        except Exception as exc:
            rec.say(f"[ms] MasterSlaveConfig(0xFC) 失败: {exc}")

    cams: dict[str, Any] = {}
    if cameras:
        cams = open_hardware_cameras(rec, shared, hw)
    return Runtime(
        rec=rec,
        shared=shared,
        hw=hw,
        follower=follower,
        reset=reset,
        leader=leader,
        cams=cams,
        scales=action_scales(shared),
        firmware_ms=firmware_ms,
        coupled=firmware_ms,
    )


def open_hardware_cameras(rec: Any, shared: dict, hw: dict) -> dict[str, Any]:
    obs = shared.get("observation") or {}
    resize = obs.get("resize_size") or [128, 128]
    rh, rw = int(resize[0]), int(resize[1])
    crops = obs.get("crop_params_dict") or {}
    out = {}
    for role, cam in (hw.get("cameras") or {}).items():
        crop = crops.get(role) or crops.get(f"observation.images.{role}")
        stream = rec.CamStream(role, cam, crop, (rh, rw))
        try:
            stream.open()
        except Exception as exc:
            rec.say(f"[cam] {role} 打开失败: {exc}")
            continue
        out[role] = stream
        rec.say(f"[cam] raw 保存 {role} ← {cam.get('index_or_path')}")
    return out


def clip_grip(rt: Runtime, g: float) -> float:
    return float(np.clip(g, rt.grip_lo, rt.grip_hi))


def send_cmd(rt: Runtime, pose: CmdPose | None = None) -> None:
    """分 CAN 软件模式写末端。共 CAN(ms) 用关节镜像，不走 EndPoseCtrl。"""
    p = pose or rt.cmd
    assert p is not None
    xyz = np.asarray(p.xyz, dtype=np.float32).reshape(3)
    rpy = np.asarray(p.rpy, dtype=np.float32).reshape(3)
    g = clip_grip(rt, p.grip)
    if not rt.firmware_ms:
        rt.rec.send_eef(rt.follower, xyz, rpy, g)
    rt.cmd = CmdPose(xyz, rpy, g)


def _ms_set_motion(rt: Runtime, *, high_follow: bool, spd: int = 100) -> None:
    """切换从臂运动模式；缓存避免每拍重复 MotionCtrl_2。"""
    ph = rt.piper
    mit = MS_MIT if high_follow else 0x00
    ctrl = (0x01, 0x01, int(spd), int(mit))
    if rt._ms_motion_ctrl == ctrl:
        return
    try:
        ph.MotionCtrl_2(*ctrl)
        rt._ms_motion_ctrl = ctrl
    except Exception as exc:
        rt.rec.say(f"[ms] MotionCtrl_2 失败: {exc}")


def soft_mirror_master_joints(rt: Runtime) -> bool:
    """把主臂控制帧关节/夹爪以高跟随转发给从臂。"""
    q = read_leader_joints(rt)
    if q is None:
        return False
    ph = rt.piper
    _ms_set_motion(rt, high_follow=True, spd=100)
    deg = np.asarray(q, dtype=np.float64) * (180.0 / np.pi) * 1e3
    try:
        ph.JointCtrl(*map(int, np.round(deg)))
    except Exception:
        return False
    try:
        g_um = int(ph.GetArmGripperCtrl().gripper_ctrl.grippers_angle)
        ph.GripperCtrl(abs(g_um), 1000, 0x01, 0)
    except Exception:
        pass
    return True


def apply_physical(rt: Runtime, dxyz_m, drpy_rad, dgrip_m):
    assert rt.cmd is not None
    dxyz = np.asarray(dxyz_m, dtype=np.float32).reshape(3)
    drpy = np.asarray(drpy_rad, dtype=np.float32).reshape(3)
    nxt = CmdPose(
        (rt.cmd.xyz + dxyz).astype(np.float32),
        wrap_pi(rt.cmd.rpy + drpy),
        clip_grip(rt, rt.cmd.grip + float(dgrip_m)),
    )
    send_cmd(rt, nxt)
    return dxyz, drpy, float(dgrip_m)


def _fk_link6_to_pose(fk: Any) -> tuple[np.ndarray, np.ndarray]:
    """GetFK → 第6关节相对 base 的 xyz(m)+rpy(rad)。"""
    arr = np.asarray(fk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 6)
    row = arr[-1]
    xyz = (row[:3] * 1e-3).astype(np.float32)
    rpy = (row[3:6] * (np.pi / 180.0)).astype(np.float32)
    return xyz, rpy


def read_master_ctrl_pose(rt: Runtime) -> CmdPose | None:
    """从共享 CAN 上主臂发出的控制帧解析主臂目标 EE（GetFK control）。"""
    try:
        ph = rt.piper
        xyz, rpy = _fk_link6_to_pose(ph.GetFK("control"))
        g_raw = ph.GetArmGripperCtrl().gripper_ctrl.grippers_angle
        # 文档: 0.001 mm
        grip = float(g_raw) * 1e-6
        return CmdPose(xyz, rpy, grip)
    except Exception:
        return None


def read_leader_pose(rt: Runtime) -> CmdPose | None:
    if rt.firmware_ms:
        return read_master_ctrl_pose(rt)
    if rt.leader is None:
        return None
    xyz, rpy = rt.rec.read_xyz_rpy(rt.leader)
    grip = float(rt.rec.read_gripper(rt.leader))
    return CmdPose(xyz.astype(np.float32).reshape(3), rpy.astype(np.float32).reshape(3), grip)


def read_leader_joints(rt: Runtime) -> np.ndarray | None:
    if rt.firmware_ms:
        try:
            jc = rt.piper.GetArmJointCtrl().joint_ctrl
            deg = np.array([getattr(jc, f"joint_{i+1}") for i in range(6)], dtype=np.float32) * 1e-3
            return (deg * (np.pi / 180.0)).astype(np.float32)
        except Exception:
            return None
    if rt.leader is None:
        return None
    try:
        return rt.rec.read_joints_q(rt.leader)
    except Exception:
        return None


def couple_leader(rt: Runtime) -> None:
    rt.prev_leader = read_leader_pose(rt)
    rt.coupled = True if rt.firmware_ms else (rt.prev_leader is not None)
    if rt.cmd is not None:
        rt.init_pose = rt.cmd.copy()


def decouple_leader(rt: Runtime) -> None:
    if rt.firmware_ms:
        # 固件主从保持耦合；仅刷新 prev
        rt.prev_leader = read_leader_pose(rt)
        rt.coupled = True
        return
    rt.coupled = False
    rt.prev_leader = read_leader_pose(rt)


def sync_cmd_from_follower(rt: Runtime) -> CmdPose:
    xyz, rpy = rt.rec.read_xyz_rpy(rt.follower)
    grip = clip_grip(rt, rt.rec.read_gripper(rt.follower))
    rt.cmd = CmdPose(xyz.astype(np.float32).reshape(3), rpy.astype(np.float32).reshape(3), grip)
    return rt.cmd


def teleop_step(rt: Runtime) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """共 CAN：关节高跟随镜像；分 CAN：绝对 EE 下发。"""
    now = read_leader_pose(rt)
    if now is None or not rt.coupled:
        if not rt.firmware_ms and rt.cmd is not None:
            send_cmd(rt)
        if rt.firmware_ms:
            soft_mirror_master_joints(rt)
            t = time.perf_counter()
            if t - rt._ms_no_ctrl_t > 2.0:
                rt._ms_no_ctrl_t = t
                rt.rec.say(
                    "[ms] 未读到主臂控制帧/未耦合。请确认：主臂已 configure --role master(0xFA)，"
                    "上电顺序先从后主；拖主臂时应有 JointCtrl。"
                )
        z = np.zeros(3, np.float32)
        return z, z.copy(), 0.0, np.zeros(7, np.float32)

    prev = rt.prev_leader or now
    dxyz = (now.xyz - prev.xyz).astype(np.float32)
    drpy = wrap_pi(now.rpy - prev.rpy)
    dgrip = float(now.grip - prev.grip)
    rt.prev_leader = now

    if rt.firmware_ms:
        ok = soft_mirror_master_joints(rt)
        sync_cmd_from_follower(rt)
        t = time.perf_counter()
        if not ok and t - rt._ms_no_ctrl_t > 2.0:
            rt._ms_no_ctrl_t = t
            rt.rec.say("[ms] JointCtrl 镜像失败（无主臂控制帧？先 configure --role master）")
        elif ok and t - rt._ms_dbg_t > 1.0:
            rt._ms_dbg_t = t
            lj = read_leader_joints(rt)
            fj = None
            try:
                fj = rt.rec.read_joints_q(rt.follower)
            except Exception:
                pass
            if lj is not None and fj is not None:
                dq = float(np.max(np.abs(lj - fj)))
                rt.rec.say(f"[ms][follow] max|Δq_leader-follower|={dq:.4f} rad  |Δxyz|={float(np.linalg.norm(dxyz))*1e3:.1f}mm")
    else:
        send_cmd(rt, now)

    init = rt.init_pose or now
    a = pose_to_norm(now.xyz, now.rpy, now.grip, init, rt.scales)
    return dxyz, drpy, dgrip, a


def collect_raw_vectors(rt: Runtime, *, with_images: bool = True) -> dict[str, np.ndarray]:
    rec = rt.rec
    xyz, rpy = rec.read_xyz_rpy(rt.follower)
    out: dict[str, np.ndarray] = {
        "eef_xyz": xyz.astype(np.float32).reshape(3),
        "eef_rpy": rpy.astype(np.float32).reshape(3),
        "gripper_gap": np.array([rec.read_gripper(rt.follower)], dtype=np.float32),
        "joint_pos": rec.read_joints_q(rt.follower),
    }
    vel, cur = rec.read_joint_vel_current(rt.follower)
    out["joint_vel"] = vel
    out["joint_current"] = cur
    if rt.cmd is not None:
        out["cmd_xyz"] = rt.cmd.xyz.astype(np.float32)
        out["cmd_rpy"] = rt.cmd.rpy.astype(np.float32)
        out["cmd_grip"] = np.array([rt.cmd.grip], dtype=np.float32)
    lead = read_leader_pose(rt)
    if lead is not None:
        out["leader_eef_xyz"] = lead.xyz
        out["leader_eef_rpy"] = lead.rpy
        out["leader_gripper"] = np.array([lead.grip], dtype=np.float32)
        lj = read_leader_joints(rt)
        if lj is not None:
            out["leader_joint_pos"] = lj
    if with_images:
        for role, stream in rt.cams.items():
            out[f"images_{role}"] = stream.read_rgb()
    return out


def reload_reset_from_cfg(rt: Runtime) -> dict[str, Any]:
    shared, hw = load_cfgs()
    rt.shared = shared
    rt.hw = hw
    rt.scales = action_scales(shared)
    rt.firmware_ms = _firmware_ms_enabled(hw)
    rt.reset = rt.rec.parse_reset(shared)
    return rt.reset


def send_joint_home(robot: Any, joints6: np.ndarray, grip: float) -> None:
    home = np.concatenate([np.asarray(joints6, dtype=np.float32).reshape(6), [float(grip)]])
    robot.send_action(home, "abs_joint")


def home_arm_can(rt: Runtime, robot: Any, *, label: str, can_name: str) -> None:
    joints = np.asarray(rt.reset.get("joints") or [], dtype=np.float32).reshape(-1)
    if joints.size != 6:
        raise SystemExit(f"cfg reset.fixed_reset_joint_positions 需要 6 维，当前 {joints.size}")
    clipped = np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    if float(np.max(np.abs(clipped - joints))) > 1e-4:
        rt.rec.say(
            f"[reset][{label}/{can_name}] cfg 关节超限位已夹紧："
            f"j5 {joints[4]:.4f}→{clipped[4]:.4f}"
        )
    rt.rec.say(f"[reset][{label}/{can_name}] abs_joint q={np.round(clipped, 4).tolist()}")
    grip = rt.grip_lo
    period = 1.0 / HOME_HZ
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < HOME_S:
        send_joint_home(robot, clipped, grip)
        time.sleep(period)
    q_now = np.asarray(rt.rec.read_joints_q(robot), dtype=np.float32).reshape(6)
    rt.rec.say(
        f"[reset][{label}/{can_name}] max|Δq|={float(np.max(np.abs(q_now - clipped))):.4f}  "
        f"q_now={np.round(q_now, 4).tolist()}"
    )


def home_firmware_ms(rt: Runtime, fd: int | None = None) -> CmdPose:
    """共 CAN 复位：PC 按 cfg 关节驱动从臂；不广播改角色，以免把主臂 0xFA 冲掉。"""
    reload_reset_from_cfg(rt)
    ph = rt.piper
    can = str((rt.hw.get("control") or {}).get("can") or "can0")

    # 普通关节位控回初始位（connect 已使能从臂）
    rt._ms_motion_ctrl = None
    _ms_set_motion(rt, high_follow=False, spd=80)
    rt.rec.say(f"[reset][ms] 从臂按 robot_shared.json 关节回初始位 → {can}")
    home_arm_can(rt, rt.follower, label="从臂(共CAN)", can_name=can)

    sync_cmd_from_follower(rt)
    rt.init_pose = rt.cmd.copy() if rt.cmd else None

    # 切回高跟随，准备镜像主臂控制帧
    rt._ms_motion_ctrl = None
    try:
        ph.MasterSlaveConfig(0xFC, 0, 0, 0)
    except Exception:
        pass
    _ms_set_motion(rt, high_follow=True, spd=100)
    soft_mirror_master_joints(rt)
    couple_leader(rt)

    q0 = read_leader_joints(rt)
    time.sleep(0.15)
    q1 = read_leader_joints(rt)
    if q0 is None and q1 is None:
        rt.rec.say(
            "[ms] 警告: 读不到主臂 JointCtrl。"
            "请断电后：只上从臂 → configure --role slave；"
            "再上主臂 → configure --role master；然后重跑。"
            "（你之前若只跑过 slave，主臂不会发控制帧。）"
        )
    else:
        rt.rec.say(
            "[teleop][ms] 从臂已回初始位；拖主臂应由 PC 高跟随镜像。"
            "主臂不会被 PC 写关节——请手动摆到相近姿态，或松开绿灯后对齐。"
        )
    if fd is not None:
        rt.rec.flush_stdin(fd)
    assert rt.cmd is not None
    return rt.cmd


def home_both(rt: Runtime, fd: int | None = None, *, timeout_s: float | None = None) -> CmdPose:
    del timeout_s
    if rt.firmware_ms:
        return home_firmware_ms(rt, fd=fd)

    reload_reset_from_cfg(rt)
    decouple_leader(rt)
    ctrl = rt.hw.get("control") or {}
    home_arm_can(rt, rt.follower, label="从臂", can_name=str(ctrl.get("follower_can") or "can0"))
    xyz, rpy = rt.rec.read_xyz_rpy(rt.follower)
    rt.cmd = CmdPose(
        np.asarray(xyz, np.float32).reshape(3),
        np.asarray(rpy, np.float32).reshape(3),
        rt.grip_lo,
    )
    send_cmd(rt)
    rt.init_pose = rt.cmd.copy()

    if rt.leader is not None:
        home_arm_can(rt, rt.leader, label="主臂", can_name=str(ctrl.get("leader_can") or "can1"))
        couple_leader(rt)
        rt.rec.say("[teleop] 软件绝对跟随（分 CAN）。共 CAN 请设 firmware_ms=true。")
    if fd is not None:
        rt.rec.flush_stdin(fd)
    assert rt.cmd is not None
    return rt.cmd


def home_follower(rt: Runtime, fd: int | None = None) -> CmdPose:
    return home_both(rt, fd=fd)


def close_runtime(rt: Runtime) -> None:
    for stream in rt.cams.values():
        try:
            stream.close()
        except Exception:
            pass
    arms = [rt.follower]
    if rt.leader is not None:
        arms.append(rt.leader)
    for arm in arms:
        try:
            arm.disconnect()
        except Exception as exc:
            rt.rec.say(f"disconnect: {exc}")
