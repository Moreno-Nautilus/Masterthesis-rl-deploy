"""SERL HTTP backend for the real iiwa7 — reuses the deploy stack.

Bridges the SERL HTTP robot server interface to the existing LBR FRI ROS2 stack:
  - Reads joint state + torques from /lbr_dual_arm_y_gripper/{joint_states,state}
  - FK via KDL (same kinematics.py as inference_node)
  - Force from measured joint torques via Jacobian (same as deploy, no wrist F/T)
  - Commands via LBRJointPositionCommand to /rl_deploy/command_15hz → upsampled to FRI rate
  - Gripper via /gripper/open_cmd (Float64, meters)

Usage
-----
  # 1. Start bringup + command upsampler as usual (see below)
  # 2. Start SERL server pointing at this backend:
  python -m iiwa_serl.robot_servers.iiwa_server \\
      --backend="rl_deploy_inference.serl_backend:LbrIiwaBackend"

Bringup sequence (same as RL deployment)
-----------------------------------------
  Terminal 1 (KRC PC): start sunrise application (FRI mode)
  Terminal 2: ros2 launch lbr_bringup move_group.launch.py mode:=hardware rviz:=false
  Terminal 3: python -m rl_deploy_inference.command_upsampler  (keeps FRI fed at 200 Hz)
  Terminal 4: python -m iiwa_serl.robot_servers.iiwa_server --backend="rl_deploy_inference.serl_backend:LbrIiwaBackend"
  Terminal 5: python record_demo.py  OR  bash run_learner.sh + run_actor.sh
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import rclpy
from lbr_fri_idl.msg import LBRJointPositionCommand, LBRState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from iiwa_serl.robot.api import RobotState

from .ik import (
    get_delta_dof_pos,
    get_pose_error,
    quat_wxyz_from_rotmat,
    wrench_from_external_torque,
)
from .kinematics import KdlConfig, KdlKinematics, KinematicsUnavailable

# Joint limits (rad)
_LOWER = np.deg2rad([-170, -120, -170, -120, -170, -120, -175], dtype=np.float64)
_UPPER = np.deg2rad([ 170,  120,  170,  120,  170,  120,  175], dtype=np.float64)

# Joint order expected by LBR FRI. Arm prefix is settable so the same backend drives
# either arm: right = lbr_two, left = lbr_one. Default left (single-arm bringup in use).
_ARM_PREFIX = os.environ.get("SERL_ARM_PREFIX", "lbr_one")
_LBR_JOINT_NAMES = [f"{_ARM_PREFIX}_A{i}" for i in range(1, 8)]

# SERL 10 Hz control → upsampler handles FRI 200 Hz refresh
_CMD_TOPIC      = "/rl_deploy/command_15hz"
_HOLD_TOPIC     = "/rl_deploy/hold"
_JS_TOPIC       = "/lbr_dual_arm_y_gripper/joint_states"
_STATE_TOPIC    = "/lbr_dual_arm_y_gripper/state"
_GRIPPER_TOPIC  = "/gripper/open_cmd"
_ROBOT_DESC_SVC = "/lbr_dual_arm_y_gripper/robot_state_publisher/get_parameters"

_GRIPPER_OPEN_M  = 0.04   # metres (fully open)
_GRIPPER_CLOSE_M = 0.00   # metres (closed)

# Reset joint config matches iiwa_serl/config.py
# Captured from the real left arm at the operator's chosen insertion home posture (2026-09-03).
# Captured on THIS left arm at the operator's PRE-INSERT pose (2026-09-02), matching
# iiwa_serl config reset_pose/reset_joints. The previous value ([-1.7,53.4,-5.9,-57.6,6.6,
# 66.5,2.8] deg) was a DIFFERENT posture (A7 off by ~103 deg) — reset yanked the arm across
# that gap and dropped FRI. These radians place reset at the pre-insert pose = tiny safe move.
_DEFAULT_RESET_JOINTS = np.array(
    [-0.17289, 1.04601, 0.63629, -1.06604, -0.60515, 1.1218, 1.85074], dtype=np.float64
)


def _reset_joints_from_env() -> np.ndarray:
    """Per-insert reset joint config.

    The assembly handoff gives a different pre-insert `reset_move_arm_q` (7 rad)
    per insert. Since we launch a fresh T6 server per insert-run anyway, pass it via
    SERL_RESET_JOINTS="q1,q2,...,q7" (mirrors SERL_ARM_PREFIX). Falls back to the
    hand-captured left-arm pose if unset/malformed (keeps old behaviour)."""
    raw = os.environ.get("SERL_RESET_JOINTS")
    if not raw:
        return _DEFAULT_RESET_JOINTS.copy()
    try:
        q = np.array([float(x) for x in raw.replace(";", ",").split(",")], dtype=np.float64)
        if q.shape == (7,):
            return q
        print(f"[serl_backend] SERL_RESET_JOINTS has {q.shape} values, need 7 — using default.")
    except ValueError as exc:
        print(f"[serl_backend] bad SERL_RESET_JOINTS ({exc!r}) — using default.")
    return _DEFAULT_RESET_JOINTS.copy()


_RESET_JOINTS = _reset_joints_from_env()


class _IiwaStateNode(Node):
    """Background ROS2 node: subscribes to state topics, publishes commands."""

    def __init__(self):
        super().__init__("serl_iiwa_backend")

        self._q   = _RESET_JOINTS.copy()
        self._dq  = np.zeros(7, dtype=np.float64)
        self._tau = np.zeros(7, dtype=np.float64)   # measured joint torques
        self._lock = threading.Lock()
        self._seen_all_joints = False               # set once a msg carries ALL 7 selected-arm joints
        self._last_js_time = 0.0

        self.create_subscription(JointState, _JS_TOPIC,   self._on_js,    qos_profile_sensor_data)
        self.create_subscription(LBRState,   _STATE_TOPIC, self._on_state, qos_profile_sensor_data)

        self._cmd_pub = self.create_publisher(LBRJointPositionCommand, _CMD_TOPIC, 1)
        self._hold_pub = self.create_publisher(LBRJointPositionCommand, _HOLD_TOPIC, 1)
        self._grip_pub = self.create_publisher(Float64, _GRIPPER_TOPIC, 1)

    # ------------------------------------------------------------------ callbacks
    def _on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            n_matched = 0
            for k, name in enumerate(_LBR_JOINT_NAMES):
                if name in idx:
                    i = idx[name]
                    self._q[k]  = msg.position[i]
                    if len(msg.velocity) > i:
                        self._dq[k] = msg.velocity[i]
                    n_matched += 1
            if n_matched == 7:                       # a full reading of THIS arm's joints (#15)
                self._seen_all_joints = True
                self._last_js_time = time.time()

    def _on_state(self, msg: LBRState):
        # Use EXTERNAL torque (gravity/dynamics compensated by the KUKA controller), NOT
        # measured_torque. measured_torque includes the static weight of gripper+peg (~99 N in Z
        # free-floating) which swamps contact forces and made force-based success impossible.
        # external_torque is ~0 free-floating and reflects true CONTACT wrench.
        if len(msg.external_torque) == 7:
            with self._lock:
                self._tau[:] = msg.external_torque
        elif len(msg.measured_torque) == 7:
            with self._lock:
                self._tau[:] = msg.measured_torque  # fallback

    # ------------------------------------------------------------------ commands
    def send_joints(self, q: np.ndarray):
        q = np.clip(np.asarray(q, dtype=np.float64), _LOWER, _UPPER)
        msg = LBRJointPositionCommand()
        msg.joint_position = q.tolist()
        self._cmd_pub.publish(msg)

    def send_hold(self, q: np.ndarray):
        q = np.clip(np.asarray(q, dtype=np.float64), _LOWER, _UPPER)
        msg = LBRJointPositionCommand()
        msg.joint_position = q.tolist()
        self._hold_pub.publish(msg)

    def send_gripper(self, width_m: float):
        self._grip_pub.publish(Float64(data=float(width_m)))

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            return self._q.copy(), self._dq.copy(), self._tau.copy()


class LbrIiwaBackend:
    """SERL backend: LBR FRI ROS2 stack + deploy KDL kinematics."""

    def __init__(
        self,
        base_link: str = "base_link",
        tip_link: str = f"{_ARM_PREFIX}_gripper_tcp",
        robot_prefix: str = _ARM_PREFIX,
        ik_damping: float = 0.01,
        ik_damping_max: float = 0.05,
        ik_sigma_soft: float = 0.12,
        ik_sigma_hard: float = 0.05,
        ik_iterations: int = 4,
        ft_sign: float = -1.0,     # flip to match sim convention (see deploy.yaml)
        **_kwargs: Any,
    ):
        # KdlConfig needs the robot_description URDF, which we only have after the
        # robot_state_publisher is up — so store params here and build it in _load_kdl().
        self._base_link = base_link
        self._tip_link = tip_link
        self._fallback_tip_link = f"{robot_prefix}_link_ee"
        self._ik_damping = max(0.0, float(ik_damping))
        self._ik_damping_max = max(self._ik_damping, float(ik_damping_max))
        self._ik_sigma_soft = max(0.0, float(ik_sigma_soft))
        self._ik_sigma_hard = max(
            0.0, min(float(ik_sigma_hard), self._ik_sigma_soft)
        )
        self._ik_iterations = max(1, int(ik_iterations))
        self._ft_sign = ft_sign
        self._node: _IiwaStateNode | None = None
        self._spin_thread: threading.Thread | None = None
        self._kdl: KdlKinematics | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self._node = _IiwaStateNode()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

        # Wait for a FULL, FRESH joint reading of the selected arm before advertising ready.
        # FAIL-CLOSED (#15): a timeout with no real reading raises instead of silently running
        # on the synthetic _RESET_JOINTS initializer (which would corrupt FK/IK).
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._node._seen_all_joints:
                break
            time.sleep(0.1)
        if not self._node._seen_all_joints:
            raise RuntimeError(
                f"No complete joint state for arm '{_ARM_PREFIX}' on {_JS_TOPIC} within 10 s. "
                "Check bringup / SERL_ARM_PREFIX matches the running arm."
            )

        # Load KDL robot description
        self._kdl = self._load_kdl()
        print(f"[LbrIiwaBackend] ready — tip={self._tip_link}, arm={_ARM_PREFIX}")

    def stop(self) -> None:
        if self._node:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def clear_errors(self) -> None:
        pass   # FRI errors are handled at the Sunrise level

    # ------------------------------------------------------------------ state
    def get_state(self) -> RobotState:
        q, dq, tau = self._node.snapshot()
        pos, R = self._kdl.fk(q)                   # base-frame TCP pose (R = 3x3 rot matrix)
        quat_wxyz = quat_wxyz_from_rotmat(R)
        jac = self._kdl.jacobian(q)                 # (6, 7) base-frame Jacobian

        wrench = wrench_from_external_torque(jac, tau)
        force  = self._ft_sign * wrench[:3]
        torque = self._ft_sign * wrench[3:6]

        # SERL uses [x,y,z,qx,qy,qz,qw]; deploy uses wxyz → convert
        quat_xyzw = np.r_[quat_wxyz[1:], quat_wxyz[0]]

        return RobotState(
            pose=np.r_[pos, quat_xyzw].astype(np.float64),
            vel=np.zeros(6, dtype=np.float64),     # not needed for SERL obs
            force=force.astype(np.float64),
            torque=torque.astype(np.float64),
            q=q.astype(np.float64),
            dq=dq.astype(np.float64),
            jacobian=jac.astype(np.float64),
            gripper=0.0,
            timestamp=time.time(),
        )

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward-kinematics of an ARBITRARY joint vector -> base-frame TCP pose7 (xyzw).

        Used to convert the handoff's goal_move_arm_q (7 rad) into a goal pose in THIS
        robot's own base/TCP frame WITHOUT moving the arm. Same frame as get_state().pose,
        so the insertion axis = goal_fk - reset_measured is frame-consistent (no handoff
        world->base transform, no flange/TCP mismatch). See HIL_RESIDUAL_SPEC.md §3."""
        q = np.asarray(q, dtype=np.float64).reshape(7)
        pos, R = self._kdl.fk(q)
        quat_wxyz = quat_wxyz_from_rotmat(R)
        quat_xyzw = np.r_[quat_wxyz[1:], quat_wxyz[0]]
        return np.r_[pos, quat_xyzw].astype(np.float64)

    # ------------------------------------------------------------------ motion
    def move_pose(self, pose7: np.ndarray) -> None:
        """Iterated adaptive-DLS Cartesian target → joint position command.

        Iterating on a virtual joint state removes the first-order Cartesian
        linearization residual before the command is sent. The nominal damping is
        low in the well-conditioned insertion postures and rises smoothly near a
        singularity. There is deliberately no nullspace/posture term.
        """
        pose7 = np.asarray(pose7, dtype=np.float64)
        target_pos  = pose7[:3]
        # SERL pose7 quaternion is xyzw; deploy FK returns wxyz
        qx, qy, qz, qw = pose7[3:7]
        target_quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)

        q, _, _ = self._node.snapshot()
        q_virtual = q.copy()
        max_step = 0.01  # rad per command (~0.57 deg), including virtual IK steps

        for _ in range(self._ik_iterations):
            cur_pos, cur_R = self._kdl.fk(q_virtual)
            cur_quat_wxyz = quat_wxyz_from_rotmat(cur_R)
            delta_pose = get_pose_error(
                cur_pos, cur_quat_wxyz, target_pos, target_quat_wxyz
            )
            if (
                float(np.linalg.norm(delta_pose[:3])) < 1e-5
                and float(np.linalg.norm(delta_pose[3:])) < 1e-4
            ):
                break

            jac = self._kdl.jacobian(q_virtual)
            sigma_min = float(np.min(np.linalg.svd(jac, compute_uv=False)))
            if sigma_min >= self._ik_sigma_soft:
                damping = self._ik_damping
            elif sigma_min <= self._ik_sigma_hard:
                damping = self._ik_damping_max
            else:
                span = max(self._ik_sigma_soft - self._ik_sigma_hard, 1e-12)
                alpha = (self._ik_sigma_soft - sigma_min) / span
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                damping = self._ik_damping + smooth * (
                    self._ik_damping_max - self._ik_damping
                )

            step_q = get_delta_dof_pos(delta_pose, jac, damping=damping)
            virtual_peak = float(np.max(np.abs(step_q)))
            if virtual_peak > max_step:
                step_q *= max_step / virtual_peak
            q_next = np.clip(q_virtual + step_q, _LOWER, _UPPER)
            if np.allclose(q_next, q_virtual, atol=1e-10, rtol=0.0):
                break
            q_virtual = q_next

        delta_q = q_virtual - q

        # Safety: limit the per-command joint step to stay inside the FRI CommandGuard velocity
        # limit. Scale the WHOLE delta_q vector by one factor (not per-joint clip!) so the IK's
        # coordinated multi-joint solution keeps its DIRECTION — per-joint clipping breaks the
        # coordination and turns pure translation into tilt (was the teleop tilt bug).
        # HALVED from 0.02: FRI was dropping at varied points (contact / sudden move / jog) —
        # i.e. the CommandGuard tripping on timing jitter, not one bad motion. Keep this final
        # whole-vector clamp even though each virtual step is bounded above.
        peak = float(np.max(np.abs(delta_q)))
        if peak > max_step:
            delta_q = delta_q * (max_step / peak)

        self._node.send_joints(q + delta_q)

    def move_joint_delta(self, dq7: np.ndarray) -> None:
        """Add a delta directly to the current joint positions (bypasses Cartesian IK).

        Used for direct single-joint teleop, e.g. d-pad -> A7 spin with no other joint motion.
        Clamped per-command like move_pose to stay inside the FRI velocity guard.
        """
        dq7 = np.asarray(dq7, dtype=np.float64).reshape(7)
        max_step = 0.02
        np.clip(dq7, -max_step, max_step, out=dq7)
        q, _, _ = self._node.snapshot()
        self._node.send_joints(q + dq7)

    def hold_position(self) -> None:
        """Cancel interpolation and pin the currently measured joints immediately."""
        q, _, _ = self._node.snapshot()
        self._node.send_hold(q)

    #: max per-command joint increment during a reset move (rad). Matches the move_pose FRI
    #: clamp intent — never step a joint more than this in one 10 Hz command.
    _RESET_MAX_STEP_RAD = 0.01

    def reset_joints(self, target_q=None) -> None:
        """Move to a target joint config via BOUNDED small steps, then wait for convergence.

        target_q : optional (7,) explicit per-insert reset (handoff reset_move_arm_q). If None,
        falls back to the module _RESET_JOINTS (env-configured). FAIL-CLOSED: a malformed
        target raises instead of silently driving somewhere wrong (#14).

        Every per-command increment is bounded to _RESET_MAX_STEP_RAD so a far reset can never
        step-jump the arm and drop FRI; the number of steps scales with the distance. Waits
        until the measured joints converge (or a timeout) before returning."""
        if target_q is None:
            q_home = _RESET_JOINTS.copy()
        else:
            q_home = np.asarray(target_q, dtype=np.float64).reshape(-1)
            if q_home.shape != (7,) or not np.all(np.isfinite(q_home)):
                raise ValueError(f"reset_joints target_q must be finite (7,), got {q_home!r}")
        # #3 FAIL-CLOSED: REJECT an out-of-range target instead of silently clipping it (a clipped
        # reset would put the peg at the wrong pre-insert -> wrong nominal line/axis).
        if np.any(q_home < _LOWER - 1e-9) or np.any(q_home > _UPPER + 1e-9):
            raise ValueError(
                f"reset_joints target out of joint limits: {q_home} not in [{_LOWER}, {_UPPER}]"
            )
        # Require a FRESH full joint reading before moving (no stale/synthetic origin).
        if not self._node._seen_all_joints or (time.time() - self._node._last_js_time) > 1.0:
            raise RuntimeError("reset_joints: no fresh full joint state (stale/absent) — aborting")

        q, _, _ = self._node.snapshot()
        max_disp = float(np.max(np.abs(q_home - q)))
        steps = max(1, int(np.ceil(max_disp / self._RESET_MAX_STEP_RAD)))
        for i in range(1, steps + 1):
            q_interp = q + (q_home - q) * (i / steps)
            self._node.send_joints(q_interp)
            time.sleep(0.05)

        # wait for convergence; RAISE if it never converges (fail-closed, not silent return).
        deadline = time.time() + float(os.environ.get("SERL_RESET_TIMEOUT_S", "10.0"))
        # Convergence tolerance (rad). The compliant Cartesian-impedance controller yields, so it
        # can't always close the last ~0.001 rad; a too-tight tol fail-closes on a ~0.05deg miss.
        conv_tol = float(os.environ.get("SERL_RESET_TOL_RAD", "0.01"))
        converged = False
        while time.time() < deadline:
            qm, _, _ = self._node.snapshot()
            if float(np.max(np.abs(qm - q_home))) < conv_tol:
                converged = True
                break
            self._node.send_joints(q_home)
            time.sleep(0.05)
        if not converged:
            qm, _, _ = self._node.snapshot()
            raise RuntimeError(
                f"reset_joints did not converge to target within timeout "
                f"(max joint err {float(np.max(np.abs(qm - q_home))):.4f} rad)"
            )
            time.sleep(0.05)

    # ------------------------------------------------------------------ gripper
    def activate_gripper(self) -> None:
        self.open_gripper()

    def reset_gripper(self) -> None:
        self.open_gripper()

    def open_gripper(self) -> None:
        self._node.send_gripper(_GRIPPER_OPEN_M)

    def close_gripper(self) -> None:
        self._node.send_gripper(_GRIPPER_CLOSE_M)

    def move_gripper(self, value: float) -> None:
        width = float(np.clip(value, 0.0, 1.0)) * _GRIPPER_OPEN_M
        self._node.send_gripper(width)

    def update_params(self, params: dict) -> None:
        pass

    # ------------------------------------------------------------------ helpers
    def _load_kdl(self) -> KdlKinematics:
        from rcl_interfaces.srv import GetParameters

        urdf = ""
        client = self._node.create_client(GetParameters, _ROBOT_DESC_SVC)
        if client.wait_for_service(timeout_sec=5.0):
            req = GetParameters.Request()
            req.names = ["robot_description"]
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
            if future.done() and future.result() and future.result().values:
                urdf = future.result().values[0].string_value or ""

        if not urdf:
            raise RuntimeError(
                "Could not fetch robot_description from "
                f"{_ROBOT_DESC_SVC}. Start the bringup (robot_state_publisher) before the backend."
            )

        cfg = KdlConfig(
            robot_description=urdf,
            base_link=self._base_link,
            tip_link=self._tip_link,
            joint_names=tuple(_LBR_JOINT_NAMES),
            fallback_tip_link=self._fallback_tip_link,
        )
        return KdlKinematics(cfg)
