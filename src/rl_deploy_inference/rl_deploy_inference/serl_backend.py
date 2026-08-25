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

import math
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

from .ik import get_delta_dof_pos, get_pose_error, wrench_from_external_torque
from .kinematics import KdlConfig, KdlKinematics, KinematicsUnavailable

# Joint limits (rad)
_LOWER = np.deg2rad([-170, -120, -170, -120, -170, -120, -175], dtype=np.float64)
_UPPER = np.deg2rad([ 170,  120,  170,  120,  170,  120,  175], dtype=np.float64)

# Joint order expected by LBR FRI
_LBR_JOINT_NAMES = [f"lbr_two_A{i}" for i in range(1, 8)]

# SERL 10 Hz control → upsampler handles FRI 200 Hz refresh
_CMD_TOPIC      = "/rl_deploy/command_15hz"
_JS_TOPIC       = "/lbr_dual_arm_y_gripper/joint_states"
_STATE_TOPIC    = "/lbr_dual_arm_y_gripper/state"
_GRIPPER_TOPIC  = "/gripper/open_cmd"
_ROBOT_DESC_SVC = "/lbr_dual_arm_y_gripper/robot_state_publisher/get_parameters"

_GRIPPER_OPEN_M  = 0.04   # metres (fully open)
_GRIPPER_CLOSE_M = 0.00   # metres (closed)

# Reset joint config matches iiwa_serl/config.py
_RESET_JOINTS = np.array([
    0.0,
    105.0 * math.pi / 180.0,
    0.0,
    -60.0 * math.pi / 180.0,
    0.0,
    118.0 * math.pi / 180.0,
    0.0,
], dtype=np.float64)


class _IiwaStateNode(Node):
    """Background ROS2 node: subscribes to state topics, publishes commands."""

    def __init__(self):
        super().__init__("serl_iiwa_backend")

        self._q   = _RESET_JOINTS.copy()
        self._dq  = np.zeros(7, dtype=np.float64)
        self._tau = np.zeros(7, dtype=np.float64)   # measured joint torques
        self._lock = threading.Lock()

        self.create_subscription(JointState, _JS_TOPIC,   self._on_js,    qos_profile_sensor_data)
        self.create_subscription(LBRState,   _STATE_TOPIC, self._on_state, qos_profile_sensor_data)

        self._cmd_pub = self.create_publisher(LBRJointPositionCommand, _CMD_TOPIC, 1)
        self._grip_pub = self.create_publisher(Float64, _GRIPPER_TOPIC, 1)

    # ------------------------------------------------------------------ callbacks
    def _on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            for k, name in enumerate(_LBR_JOINT_NAMES):
                if name in idx:
                    i = idx[name]
                    self._q[k]  = msg.position[i]
                    if msg.velocity:
                        self._dq[k] = msg.velocity[i]

    def _on_state(self, msg: LBRState):
        if msg.measured_joint_torque:
            with self._lock:
                self._tau[:] = msg.measured_joint_torque

    # ------------------------------------------------------------------ commands
    def send_joints(self, q: np.ndarray):
        q = np.clip(np.asarray(q, dtype=np.float64), _LOWER, _UPPER)
        msg = LBRJointPositionCommand()
        msg.joint_position = q.tolist()
        self._cmd_pub.publish(msg)

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
        tip_link: str = "lbr_two_gripper_tcp",
        robot_prefix: str = "lbr_two",
        ik_damping: float = 0.05,
        ft_sign: float = -1.0,     # flip to match sim convention (see deploy.yaml)
        **_kwargs: Any,
    ):
        self._kdl_cfg = KdlConfig(
            base_link=base_link,
            tip_link=tip_link,
            fallback_tip_link=f"{robot_prefix}_link_ee",
        )
        self._ik_damping = ik_damping
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

        # Wait for first joint state
        deadline = time.time() + 10.0
        while time.time() < deadline:
            q, _, _ = self._node.snapshot()
            if not np.allclose(q, _RESET_JOINTS):   # received at least one real reading
                break
            time.sleep(0.1)

        # Load KDL robot description
        self._kdl = self._load_kdl()
        print(f"[LbrIiwaBackend] ready — tip={self._kdl_cfg.tip_link}")

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
        pos, quat_wxyz = self._kdl.fk(q)          # base-frame TCP pose
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

    # ------------------------------------------------------------------ motion
    def move_pose(self, pose7: np.ndarray) -> None:
        """DLS IK: Cartesian target → joint positions → LBRJointPositionCommand."""
        pose7 = np.asarray(pose7, dtype=np.float64)
        target_pos  = pose7[:3]
        # SERL pose7 quaternion is xyzw; deploy FK returns wxyz
        qx, qy, qz, qw = pose7[3:7]
        target_quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)

        q, _, _ = self._node.snapshot()
        cur_pos, cur_quat_wxyz = self._kdl.fk(q)
        jac = self._kdl.jacobian(q)

        delta_pose = get_pose_error(cur_pos, cur_quat_wxyz, target_pos, target_quat_wxyz)
        delta_q    = get_delta_dof_pos(delta_pose, jac, damping=self._ik_damping)

        self._node.send_joints(q + delta_q)

    def reset_joints(self) -> None:
        """Move to home joint config via small steps (10 iterations at 10 Hz)."""
        q_home = _RESET_JOINTS.copy()
        q, _, _ = self._node.snapshot()
        steps = 20
        for i in range(1, steps + 1):
            q_interp = q + (q_home - q) * (i / steps)
            self._node.send_joints(q_interp)
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
        client = self._node.create_client(GetParameters, _ROBOT_DESC_SVC)
        if client.wait_for_service(timeout_sec=5.0):
            req = GetParameters.Request()
            req.names = ["robot_description"]
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
            if future.done() and future.result() and future.result().values:
                urdf = future.result().values[0].string_value
                if urdf:
                    return KdlKinematics(self._kdl_cfg, urdf=urdf)

        # Fallback: check ROS_PACKAGE_PATH / file
        print("[LbrIiwaBackend] Warning: could not fetch robot_description from service, "
              "trying KdlKinematics without explicit URDF (must be on URDF path).")
        try:
            return KdlKinematics(self._kdl_cfg)
        except KinematicsUnavailable as exc:
            raise RuntimeError(
                "KDL kinematics unavailable. Either:\n"
                "  a) Start the robot_state_publisher before the backend, or\n"
                "  b) Set the URDF path manually.\n"
                f"Original error: {exc}"
            ) from exc
