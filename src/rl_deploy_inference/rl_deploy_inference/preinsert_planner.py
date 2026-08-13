"""Headless MoveIt preinsert planner/executor for the RIGHT arm (``lbr_two`` / group ``arm_two``).

Moves the right arm to a safe *preinsert* pose hovering above the perceived socket, using MoveIt for
the gross motion -- NOT the RL ``reset_preinsert`` IK servo (which is a local damped-IK servo that
rattles from far away). The RL policy then handles only the final local insertion from this hover.

What it does (steps mirror the request):
  1. Waits for a fresh socket pose on ``/perception/fp/pose_base/fused/assembly``.
  2. Filters ``assembly_name == socket_assembly_name`` (default ``cooling_manifold``); ``part_id`` is
     matched only when ``socket_part_id >= 0`` (``-1`` accepts any, with a warning -- dry-run bring-up).
  3. Target = perceived socket position + ``[0, 0, hover_z_m]`` in the global ``base_link`` frame
     (perception is already in ``base_link``, which is also MoveIt's planning frame). Orientation is
     the CURRENT TCP orientation by default (``orientation_mode: current_tcp``) or a fixed configured
     quaternion (``fixed``) -- the perceived object orientation is deliberately NOT trusted yet.
  4. Plans for the right arm only via MoveIt (only the inserting arm moves).
  5. Executes MoveIt-natively through the ``joint_trajectory_controller``.
  6. Two modes: plan-only (default, never moves) and execute.
  7. Logs current TCP pose, socket pose, target pose, plan success/failure, first/last joint targets.
  8. Conservative velocity/acceleration scaling (<= 0.1).
  9. Reusable: the underlying MotionCommander plans/executes to any pose or joint target.

Interfaces:
  * One-shot CLI (default): plan, print a full report, exit. Add ``--execute`` (+ confirmation) to move.
  * Long-lived services (``--service``): ``~/plan_preinsert`` (dry-run) and ``~/move_preinsert``
    (plan+execute, gated behind the ``allow_execute`` param).

SAFETY: motion is OFF by default. Before anything that can move: remaster A1 if the pendant says it
is unmastered, and be ready on the physical e-stop. Start with the plan-only path.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Trigger

from fp_debug_msgs.msg import DebugPoseItem
from geometry_msgs.msg import PoseStamped

from .motion_commander import (
    MotionCommander,
    MotionCommanderConfig,
    PlanResult,
    QuaternionXYZW,
    PositionXYZ,
    moveit_available,
)
from .ik import get_delta_dof_pos, get_pose_error, quat_wxyz_from_rotmat
from .kinematics import KdlConfig, KdlKinematics, KinematicsUnavailable


IIWA7_LOWER = np.deg2rad(np.array([-170, -120, -170, -120, -170, -120, -175], dtype=np.float64))
IIWA7_UPPER = np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175], dtype=np.float64))

def warn_banner(arm: str, group: str) -> str:
    return (
        "\n" + "!" * 78 + "\n"
        "! ROBOT MAY MOVE NOW. Be ready on the physical E-STOP.\n"
        f"! Only the {arm.upper()} arm ({group}) is commanded; the other arm is not touched.\n"
        "! If the pendant shows A1 UNMASTERED, ABORT and remaster before moving.\n"
        + "!" * 78 + "\n"
    )


def _rotmat_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_xyzw_from_rotmat(R: np.ndarray) -> QuaternionXYZW:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _rotmat_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def closest_down_quat_xyzw(current_quat_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    """The tool-straight-down orientation CLOSEST to the current one (minimal wrist reorientation).

    The gripper approach axis is gripper_tcp +z; "down" means that axis points to world -z. There is a
    whole family of down orientations (any yaw about the vertical) -- this picks the one nearest the
    current wrist, so the tool ends up exactly vertical with the smallest possible reorientation
    (avoids the ~180deg wrist flip a fixed [1,0,0,0] would force when the current roll is opposite).
    """
    R = _rotmat_from_quat_xyzw(current_quat_xyzw)
    a = R[:, 2]  # current tool approach axis in base_link
    b = np.array([0.0, 0.0, -1.0])  # world down
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-8:
        delta = np.eye(3) if c > 0 else _rotmat_about(np.array([1.0, 0.0, 0.0]), math.pi)
    else:
        delta = _rotmat_about(v / s, math.atan2(s, c))
    return _quat_xyzw_from_rotmat(delta @ R)


def compute_preinsert_target(
    socket_pos: PositionXYZ,
    current_tcp_quat_xyzw: QuaternionXYZW | None,
    *,
    hover_z_m: float,
    orientation_mode: str,
    fixed_quat_xyzw: QuaternionXYZW,
) -> tuple[PositionXYZ, QuaternionXYZW]:
    """Pure preinsert-target math (no ROS): hover ``hover_z_m`` above the socket in global z.

    Position is the socket position raised by ``hover_z_m`` in the base/global frame. Orientation:
      ``down``       -> tool exactly straight down, yaw closest to the current wrist (minimal move).
      ``current_tcp``-> keep the current TCP orientation as-is.
      ``fixed``      -> the configured fixed quaternion.
    The perceived object orientation is intentionally ignored in all modes.
    """
    target_pos = (float(socket_pos[0]), float(socket_pos[1]), float(socket_pos[2]) + float(hover_z_m))
    mode = str(orientation_mode).strip().lower()
    if mode == "fixed":
        quat = fixed_quat_xyzw
    elif mode == "current_tcp":
        if current_tcp_quat_xyzw is None:
            raise ValueError("orientation_mode=current_tcp needs the current TCP orientation (TF)")
        quat = current_tcp_quat_xyzw
    elif mode == "down":
        if current_tcp_quat_xyzw is None:
            raise ValueError("orientation_mode=down needs the current TCP orientation (TF)")
        quat = closest_down_quat_xyzw(current_tcp_quat_xyzw)
    else:
        raise ValueError(f"unknown orientation_mode {orientation_mode!r} (use 'down', 'current_tcp' or 'fixed')")
    return target_pos, (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


@dataclass
class _TimedPose:
    pos: PositionXYZ
    quat: QuaternionXYZW
    wall_t: float


def _fmt_xyz(v) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


def _fmt_quat(q) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in q) + "]"


def _fmt_rad_deg(rad: float) -> str:
    return f"{rad:.3f} rad ({math.degrees(rad):.1f} deg)"


def _wxyz_from_xyzw(q: QuaternionXYZW) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# arm -> (planning group, tip link). Both arms plan through the SAME move_group; only the group
# (hence which 7 of the 14 joints move) and the tip link differ. base_link stays the frame for both.
ARM_DEFAULTS = {
    "right": ("arm_two", "lbr_two_gripper_tcp"),
    "left": ("arm_one", "lbr_one_gripper_tcp"),
}


ARM_JOINTS = {
    "right": [f"lbr_two_A{i}" for i in range(1, 8)],
    "left": [f"lbr_one_A{i}" for i in range(1, 8)],
}


class PreinsertPlanner(Node):
    def __init__(self, *, namespace: str, background: bool, arm_override: str | None = None) -> None:
        super().__init__("preinsert_planner", namespace=namespace)
        self._background = background
        self._declare_parameters()

        arm = (arm_override or self.get_parameter("arm").value or "right").strip().lower()
        if arm not in ARM_DEFAULTS:
            raise ValueError(f"arm must be one of {sorted(ARM_DEFAULTS)}, got {arm!r}")
        self._arm = arm
        default_group, default_tip = ARM_DEFAULTS[arm]
        # Empty group_name/tip_link params derive from `arm`; non-empty overrides them explicitly.
        group_name = self.get_parameter("group_name").value or default_group
        tip_link = self.get_parameter("tip_link").value or default_tip

        cfg = MotionCommanderConfig(
            group_name=group_name,
            base_frame=self.get_parameter("base_frame").value,
            tip_link=tip_link,
            move_group_action=self.get_parameter("move_group_action").value,
            joint_states_topic=self.get_parameter("joint_states_topic").value,
            pipeline_id=self.get_parameter("pipeline_id").value,
            planner_id=self.get_parameter("planner_id").value,
            num_planning_attempts=int(self.get_parameter("num_planning_attempts").value),
            allowed_planning_time_s=float(self.get_parameter("planning_time_s").value),
            max_velocity_scaling=float(self.get_parameter("velocity_scaling").value),
            max_acceleration_scaling=float(self.get_parameter("acceleration_scaling").value),
        )
        self.commander = MotionCommander(self, cfg, background_spin=background)

        self._socket: _TimedPose | None = None
        # Perception publishes the socket with best-effort sensor QoS (same as inference_node's socket
        # sub); a RELIABLE subscriber would silently receive NOTHING from it. Match it.
        self.create_subscription(
            DebugPoseItem,
            self.get_parameter("socket_pose_topic").value,
            self._on_socket,
            qos_profile_sensor_data,
        )
        # Service callbacks block while a plan/execute future resolves. Put them in a reentrant group
        # so the MultiThreadedExecutor can keep servicing the socket/joint_states/action-result
        # callbacks (default group) concurrently -- otherwise the blocking service would deadlock.
        srv_group = ReentrantCallbackGroup()
        self.create_service(Trigger, "~/plan_preinsert", self._srv_plan, callback_group=srv_group)
        self.create_service(Trigger, "~/move_preinsert", self._srv_move, callback_group=srv_group)

        self.get_logger().info(
            f"preinsert_planner ready (namespace={self.get_namespace()}, arm={self._arm}, "
            f"group={cfg.group_name}, tip={cfg.tip_link}, frame={cfg.base_frame}, "
            f"moveit={'yes' if moveit_available() else 'NO'})."
        )
        if not moveit_available():
            self.get_logger().error(
                "moveit_msgs is NOT importable here -- planning will fail. Install/source MoveIt "
                "(ros-humble-moveit) on the machine that runs move_group + this node."
            )

    # ---- parameters ---------------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        # "right" -> group arm_two / tip lbr_two_gripper_tcp; "left" -> arm_one / lbr_one_gripper_tcp.
        self.declare_parameter("arm", "right")
        # Leave group_name/tip_link empty to derive from `arm`; set them to override explicitly.
        self.declare_parameter("group_name", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tip_link", "")
        self.declare_parameter("move_group_action", "move_action")
        self.declare_parameter("joint_states_topic", "joint_states")

        self.declare_parameter("socket_pose_topic", "/perception/fp/pose_base/fused/assembly")
        self.declare_parameter("socket_assembly_name", "cooling_manifold")
        self.declare_parameter("socket_part_id", -1)
        self.declare_parameter("socket_timeout_s", 3.0)

        # Socket pose is ~1 cm off physically -> hover HIGH and let the RL policy close the last cm.
        self.declare_parameter("hover_z_m", 0.15)
        # "down" (default): tool exactly straight down, yaw closest to the current wrist (minimal
        # reorientation). "current_tcp": keep the current orientation. "fixed": fixed_orientation_xyzw.
        self.declare_parameter("orientation_mode", "down")
        self.declare_parameter("fixed_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])

        self.declare_parameter("position_tolerance_m", 0.01)
        self.declare_parameter("orientation_tolerance_rad", 0.1)
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("acceleration_scaling", 0.05)
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("num_planning_attempts", 10)
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planner_id", "")
        # Keep MoveIt on the current elbow/wrist branch. Without this, OMPL can solve a nearby TCP
        # target with a wildly different IK branch (large A1/A2/A5/A7 flips), which is unacceptable
        # for a preinsert hover move next to real cameras/fixtures.
        self.declare_parameter("joint_near_current_tolerance_rad", 0.80)
        self.declare_parameter("max_plan_joint_delta_rad", 0.85)
        # "pose_goal": normal MoveIt Cartesian pose goal. "local_ik_joint": compute a nearby joint
        # target with KDL/DLS from the current branch, then let MoveIt execute a joint-space path.
        self.declare_parameter("target_mode", "pose_goal")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robot_description_service", "robot_state_publisher/get_parameters")
        self.declare_parameter("local_ik_max_iters", 120)
        self.declare_parameter("local_ik_damping", 0.1)
        self.declare_parameter("local_ik_joint_step_rad", 0.04)
        self.declare_parameter("local_ik_cart_step_m", 0.025)
        self.declare_parameter("local_ik_rot_step_rad", 0.08)
        self.declare_parameter("local_ik_max_total_delta_rad", 1.20)
        self.declare_parameter("local_ik_joint_goal_tolerance_rad", 0.01)
        self.declare_parameter("tf_timeout_s", 5.0)
        # Absorb move_group's dirty-current-state startup window with a plan-only probe before planning.
        self.declare_parameter("warmup_plan", True)
        # Gate for the ~/move_preinsert SERVICE (the CLI --execute path has its own confirmation).
        self.declare_parameter("allow_execute", False)

    # ---- perception ---------------------------------------------------------------------------
    def _on_socket(self, msg: DebugPoseItem) -> None:
        want_assembly = self.get_parameter("socket_assembly_name").value
        want_part = int(self.get_parameter("socket_part_id").value)
        if want_assembly and msg.assembly_name != want_assembly:
            return
        if want_part >= 0 and int(msg.part_id) != want_part:
            return
        if want_part < 0:
            self.get_logger().warn(
                f"socket_part_id=-1 wildcard accepted {msg.assembly_name}/{msg.part_id}; "
                "set socket_part_id explicitly for real runs.",
                throttle_duration_sec=5.0,
            )
        p = msg.pose_base.pose.position
        q = msg.pose_base.pose.orientation
        self._socket = _TimedPose(
            pos=(p.x, p.y, p.z), quat=(q.x, q.y, q.z, q.w), wall_t=time.monotonic()
        )

    def _pump(self, timeout_s: float) -> None:
        if self._background:
            time.sleep(min(timeout_s, 0.05))
        else:
            rclpy.spin_once(self, timeout_sec=timeout_s)

    def _wait_for_socket(self, timeout_s: float) -> _TimedPose | None:
        fresh_age = float(self.get_parameter("socket_timeout_s").value)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            s = self._socket
            if s is not None and (time.monotonic() - s.wall_t) <= fresh_age:
                return s
            self._pump(0.1)
        return self._socket  # may be stale/None; caller decides

    # ---- core flow ----------------------------------------------------------------------------
    def run_preinsert(self, *, execute: bool) -> PlanResult:
        """Plan (and optionally execute) the preinsert move. Returns the (dry-run) PlanResult.

        Always plans first and logs the full report. Only when ``execute`` is True (and the plan
        succeeds) does it issue the MoveIt-native execute.
        """
        if not moveit_available():
            return PlanResult(False, 0, "NO_MOVEIT", message="moveit_msgs not importable")
        if not self.commander.wait_for_server(timeout_s=10.0):
            return PlanResult(False, 0, "NO_MOVE_GROUP", message="move_group action server not found")

        socket = self._wait_for_socket(float(self.get_parameter("socket_timeout_s").value) + 2.0)
        fresh_age = float(self.get_parameter("socket_timeout_s").value)
        if socket is None:
            return PlanResult(False, 0, "NO_SOCKET", message="no socket pose received")
        if (time.monotonic() - socket.wall_t) > fresh_age:
            return PlanResult(False, 0, "STALE_SOCKET", message="socket pose older than socket_timeout_s")

        mode = str(self.get_parameter("orientation_mode").value).strip().lower()
        current_tcp: _TimedPose | None = None
        try:
            tcp_pos, tcp_quat = self.commander.get_link_pose(
                timeout_s=float(self.get_parameter("tf_timeout_s").value)
            )
            current_tcp = _TimedPose(tcp_pos, tcp_quat, time.monotonic())
        except LookupError as exc:
            if mode == "current_tcp":
                return PlanResult(False, 0, "NO_TCP_TF", message=f"current TCP pose unavailable: {exc}")
            self.get_logger().warn(f"TCP pose via TF unavailable ({exc}); continuing (orientation_mode=fixed).")

        try:
            target_pos, target_quat = compute_preinsert_target(
                socket.pos,
                current_tcp.quat if current_tcp else None,
                hover_z_m=float(self.get_parameter("hover_z_m").value),
                orientation_mode=mode,
                fixed_quat_xyzw=tuple(self.get_parameter("fixed_orientation_xyzw").value),
            )
        except ValueError as exc:
            return PlanResult(False, 0, "BAD_TARGET", message=str(exc))

        pos_tol = float(self.get_parameter("position_tolerance_m").value)
        ori_tol = float(self.get_parameter("orientation_tolerance_rad").value)

        self.get_logger().info("---- preinsert target report ----")
        if current_tcp:
            self.get_logger().info(
                f"current TCP  (base_link): pos={_fmt_xyz(current_tcp.pos)} quat_xyzw={_fmt_quat(current_tcp.quat)}"
            )
        self.get_logger().info(
            f"socket pose  (base_link): pos={_fmt_xyz(socket.pos)} quat_xyzw={_fmt_quat(socket.quat)} "
            "(orientation ignored)"
        )
        self.get_logger().info(
            f"target pose  (base_link): pos={_fmt_xyz(target_pos)} quat_xyzw={_fmt_quat(target_quat)} "
            f"(hover +{float(self.get_parameter('hover_z_m').value):.3f} m z, orientation_mode={mode})"
        )

        current_branch = self._current_arm_joint_map()
        joint_tol = float(self.get_parameter("joint_near_current_tolerance_rad").value)
        if current_branch:
            self.get_logger().info(
                f"joint branch constraint: keep {self._arm} arm within +/-{_fmt_rad_deg(joint_tol)} "
                "of current joints during planning"
            )
        else:
            self.get_logger().warn("no current arm joint map available; planning without branch constraint")

        target_mode = str(self.get_parameter("target_mode").value).strip().lower()
        if target_mode == "local_ik_joint":
            plan = self._plan_local_ik_joint(current_branch, target_pos, target_quat)
            self._log_plan(plan)
            if not self._plan_stays_near_current(plan, current_branch):
                return plan
            if not plan.success:
                self.get_logger().error(f"plan FAILED ({plan.error_name}); not executing.")
                return plan
            if not execute:
                self.get_logger().info("plan-only mode: NOT executing. Re-run with execute to move.")
                return plan
            self.get_logger().warn(warn_banner(self._arm, self.commander.cfg.group_name))
            q_goal = self._last_local_ik_goal
            if not q_goal:
                return PlanResult(False, 0, "NO_LOCAL_IK_GOAL", message="internal local IK goal missing")
            moved = self.commander.move_to_joint(
                q_goal,
                tolerance_rad=float(self.get_parameter("local_ik_joint_goal_tolerance_rad").value),
            )
            self._log_plan(moved, executed=True)
            if not moved.success:
                self.get_logger().error(
                    f"execution FAILED ({moved.error_name}). Check joint_trajectory_controller state."
                )
            else:
                self.get_logger().info("preinsert execution SUCCEEDED; RL policy can take over the final insertion.")
            return moved

        if self.get_parameter("warmup_plan").value:
            self.commander.wait_until_plannable(
                target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
                path_joint_targets=current_branch or None,
                path_joint_tolerance_rad=joint_tol if current_branch else None,
            )

        plan = self.commander.plan_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
            path_joint_targets=current_branch or None,
            path_joint_tolerance_rad=joint_tol if current_branch else None,
        )
        self._log_plan(plan)
        if not self._plan_stays_near_current(plan, current_branch):
            return plan
        if not plan.success:
            self.get_logger().error(f"plan FAILED ({plan.error_name}); not executing.")
            return plan
        if not execute:
            self.get_logger().info("plan-only mode: NOT executing. Re-run with execute to move.")
            return plan

        self.get_logger().warn(warn_banner(self._arm, self.commander.cfg.group_name))
        moved = self.commander.move_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
            path_joint_targets=current_branch or None,
            path_joint_tolerance_rad=joint_tol if current_branch else None,
        )
        self._log_plan(moved, executed=True)
        if not moved.success:
            self.get_logger().error(
                f"execution FAILED ({moved.error_name}). If it was rejected, check that "
                "joint_trajectory_controller is active AND has allow_partial_joints_goal: true."
            )
        else:
            self.get_logger().info("preinsert execution SUCCEEDED; RL policy can take over the final insertion.")
        return moved

    _last_local_ik_goal: dict[str, float] | None = None

    def _current_arm_joint_map(self) -> dict[str, float]:
        current = self.commander.current_joint_positions()
        names = ARM_JOINTS[self._arm]
        if not all(name in current for name in names):
            return {}
        return {name: current[name] for name in names}

    def _retrieve_robot_description(self, timeout_s: float = 1.0) -> str | None:
        value = self.get_parameter("robot_description").value
        if value:
            return str(value)
        service = self.get_parameter("robot_description_service").value
        client = self.create_client(GetParameters, service)
        if not client.wait_for_service(timeout_sec=timeout_s):
            return None
        request = GetParameters.Request(names=["robot_description"])
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if future.result() is None or not future.result().values:
            return None
        return future.result().values[0].string_value or None

    def _make_kinematics(self) -> KdlKinematics | None:
        desc = self._retrieve_robot_description(timeout_s=1.0)
        if not desc:
            self.get_logger().error("local_ik_joint: no robot_description available")
            return None
        try:
            return KdlKinematics(
                KdlConfig(
                    robot_description=desc,
                    base_link=self.get_parameter("base_frame").value,
                    tip_link=self.commander.cfg.tip_link,
                    joint_names=tuple(ARM_JOINTS[self._arm]),
                )
            )
        except KinematicsUnavailable as exc:
            self.get_logger().error(f"local_ik_joint: {exc}")
            return None

    def _plan_local_ik_joint(
        self,
        current_branch: dict[str, float],
        target_pos: PositionXYZ,
        target_quat: QuaternionXYZW,
    ) -> PlanResult:
        self._last_local_ik_goal = None
        if not current_branch:
            return PlanResult(False, 0, "NO_JOINT_STATE", message="no current arm joint state")
        kin = self._make_kinematics()
        if kin is None:
            return PlanResult(False, 0, "NO_KDL", message="KDL unavailable")

        names = ARM_JOINTS[self._arm]
        q0 = np.array([current_branch[n] for n in names], dtype=np.float64)
        q = q0.copy()
        target_p = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_q = _wxyz_from_xyzw(target_quat)
        pos_tol = float(self.get_parameter("position_tolerance_m").value)
        ori_tol = float(self.get_parameter("orientation_tolerance_rad").value)
        damping = float(self.get_parameter("local_ik_damping").value)
        max_joint_step = float(self.get_parameter("local_ik_joint_step_rad").value)
        max_cart_step = float(self.get_parameter("local_ik_cart_step_m").value)
        max_rot_step = float(self.get_parameter("local_ik_rot_step_rad").value)
        total_limit = float(self.get_parameter("local_ik_max_total_delta_rad").value)
        margin = 0.03

        final_pos_err = float("inf")
        final_rot_err = float("inf")
        for _ in range(int(self.get_parameter("local_ik_max_iters").value)):
            pos, R = kin.fk(q)
            quat = quat_wxyz_from_rotmat(R)
            err = get_pose_error(pos, quat, target_p, target_q)
            final_pos_err = float(np.linalg.norm(err[:3]))
            final_rot_err = float(np.linalg.norm(err[3:]))
            if final_pos_err <= pos_tol and final_rot_err <= ori_tol:
                break

            step = err.copy()
            trans_norm = float(np.linalg.norm(step[:3]))
            if trans_norm > max_cart_step:
                step[:3] *= max_cart_step / trans_norm
            rot_norm = float(np.linalg.norm(step[3:]))
            if rot_norm > max_rot_step:
                step[3:] *= max_rot_step / rot_norm

            dq = get_delta_dof_pos(step, kin.jacobian(q), damping=damping)
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q = np.clip(q + dq, IIWA7_LOWER + margin, IIWA7_UPPER - margin)

        delta = q - q0
        max_delta = float(np.max(np.abs(delta)))
        q_goal = {name: float(val) for name, val in zip(names, q)}
        self.get_logger().info(
            "local IK joint target: "
            f"pos_err={final_pos_err:.4f} m rot_err={final_rot_err:.4f} rad "
            f"max_delta={_fmt_rad_deg(max_delta)} q={_fmt_xyz(q)}"
        )
        if final_pos_err > pos_tol or final_rot_err > ori_tol:
            return PlanResult(
                False,
                0,
                "LOCAL_IK_FAILED",
                message=f"pos_err={final_pos_err:.4f} rot_err={final_rot_err:.4f}",
            )
        if max_delta > total_limit:
            return PlanResult(
                False,
                0,
                "LOCAL_IK_JUMP",
                message=f"local IK max_delta {_fmt_rad_deg(max_delta)} > {_fmt_rad_deg(total_limit)}",
            )
        self._last_local_ik_goal = q_goal
        return self.commander.plan_to_joint(
            q_goal,
            tolerance_rad=float(self.get_parameter("local_ik_joint_goal_tolerance_rad").value),
        )

    def _plan_stays_near_current(self, plan: PlanResult, current: dict[str, float]) -> bool:
        if not plan.success or not current:
            return True
        names, _first, last = plan.joint_targets()
        if not names:
            return True
        deltas = [abs(float(v) - current.get(name, float(v))) for name, v in zip(names, last)]
        max_delta = max(deltas) if deltas else 0.0
        limit = float(self.get_parameter("max_plan_joint_delta_rad").value)
        if max_delta <= limit:
            self.get_logger().info(f"plan joint delta check OK: max_delta={_fmt_rad_deg(max_delta)}")
            return True
        plan.success = False
        plan.message = f"rejected branch flip: max final joint delta {_fmt_rad_deg(max_delta)} > {_fmt_rad_deg(limit)}"
        self.get_logger().error(plan.message)
        return False

    def _log_plan(self, plan: PlanResult, *, executed: bool = False) -> None:
        tag = "execute" if executed else "plan"
        self.get_logger().info(f"{tag}: {plan.summary()}")
        names, first, last = plan.joint_targets()
        if names:
            self.get_logger().info(f"{tag} joints: {list(names)}")
            self.get_logger().info(f"{tag} first joint target: {_fmt_xyz(first)}")
            self.get_logger().info(f"{tag} last  joint target: {_fmt_xyz(last)}")

    # ---- services -----------------------------------------------------------------------------
    def _srv_plan(self, _request, response):
        plan = self.run_preinsert(execute=False)
        response.success = plan.success
        response.message = plan.summary()
        return response

    def _srv_move(self, _request, response):
        if not self.get_parameter("allow_execute").value:
            response.success = False
            response.message = "allow_execute param is false; refusing to move. Set it true to enable ~/move_preinsert."
            self.get_logger().error(response.message)
            return response
        plan = self.run_preinsert(execute=True)
        response.success = plan.success and plan.moved
        response.message = plan.summary()
        return response


def _parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(description="Headless MoveIt preinsert planner for the right arm.")
    parser.add_argument("--namespace", default="lbr_dual_arm_y_gripper", help="Robot namespace (move_group / tf / joint_states).")
    parser.add_argument("--arm", choices=["right", "left"], default=None, help="Which arm to move (default: right, or the 'arm' param).")
    parser.add_argument("--service", action="store_true", help="Run as a long-lived service server instead of one-shot.")
    parser.add_argument("--execute", action="store_true", help="One-shot: plan AND execute (default is plan-only).")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive execute confirmation (for non-interactive use).")
    return parser.parse_args(argv)


def _confirm_execute() -> bool:
    if not sys.stdin.isatty():
        print("Refusing to execute non-interactively without --yes.", file=sys.stderr)
        return False
    try:
        return input("Type MOVE to execute on the real robot: ").strip() == "MOVE"
    except (EOFError, KeyboardInterrupt):
        return False


def main(args: list[str] | None = None) -> None:
    cli = _parse_args(remove_ros_args(args=sys.argv)[1:])
    namespace = cli.namespace if cli.namespace.startswith("/") else f"/{cli.namespace}"
    rclpy.init(args=args)

    if cli.service:
        node = PreinsertPlanner(namespace=namespace, background=True, arm_override=cli.arm)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info(
            "service mode: call ~/plan_preinsert (dry-run) or ~/move_preinsert (needs allow_execute:=true)."
        )
        try:
            executor.spin()
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    node = PreinsertPlanner(namespace=namespace, background=False, arm_override=cli.arm)
    try:
        execute = cli.execute
        if execute and not cli.yes:
            # Show the plan first (plan-only), then confirm before moving.
            node.run_preinsert(execute=False)
            if not _confirm_execute():
                node.get_logger().warn("execute not confirmed; staying in plan-only. Nothing moved.")
                execute = False
        if execute:
            node.run_preinsert(execute=True)
        elif not cli.execute:
            node.run_preinsert(execute=False)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
