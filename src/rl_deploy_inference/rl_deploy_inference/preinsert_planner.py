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

import sys
import time
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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

WARN_BANNER = (
    "\n" + "!" * 78 + "\n"
    "! ROBOT MAY MOVE NOW. Be ready on the physical E-STOP.\n"
    "! Only the RIGHT arm (arm_two) is commanded; the left arm is not touched.\n"
    "! If the pendant shows A1 UNMASTERED, ABORT and remaster before moving.\n"
    + "!" * 78 + "\n"
)


def compute_preinsert_target(
    socket_pos: PositionXYZ,
    current_tcp_quat_xyzw: QuaternionXYZW | None,
    *,
    hover_z_m: float,
    orientation_mode: str,
    fixed_quat_xyzw: QuaternionXYZW,
) -> tuple[PositionXYZ, QuaternionXYZW]:
    """Pure preinsert-target math (no ROS): hover ``hover_z_m`` above the socket in global z.

    Position is the socket position raised by ``hover_z_m`` in the base/global frame. Orientation is
    the current TCP quaternion (``current_tcp``) or the configured fixed quaternion (``fixed``); the
    perceived object orientation is intentionally ignored.
    """
    target_pos = (float(socket_pos[0]), float(socket_pos[1]), float(socket_pos[2]) + float(hover_z_m))
    mode = str(orientation_mode).strip().lower()
    if mode == "fixed":
        quat = fixed_quat_xyzw
    elif mode == "current_tcp":
        if current_tcp_quat_xyzw is None:
            raise ValueError("orientation_mode=current_tcp needs the current TCP orientation (TF)")
        quat = current_tcp_quat_xyzw
    else:
        raise ValueError(f"unknown orientation_mode {orientation_mode!r} (use 'current_tcp' or 'fixed')")
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


# arm -> (planning group, tip link). Both arms plan through the SAME move_group; only the group
# (hence which 7 of the 14 joints move) and the tip link differ. base_link stays the frame for both.
ARM_DEFAULTS = {
    "right": ("arm_two", "lbr_two_gripper_tcp"),
    "left": ("arm_one", "lbr_one_gripper_tcp"),
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
        self.create_subscription(
            DebugPoseItem, self.get_parameter("socket_pose_topic").value, self._on_socket, 10
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
        # "current_tcp" (default) keeps whatever orientation the TCP currently has (do not trust the
        # perceived object orientation yet). "fixed" uses fixed_orientation_xyzw instead.
        self.declare_parameter("orientation_mode", "current_tcp")
        self.declare_parameter("fixed_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])

        self.declare_parameter("position_tolerance_m", 0.01)
        self.declare_parameter("orientation_tolerance_rad", 0.1)
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("acceleration_scaling", 0.05)
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("num_planning_attempts", 10)
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planner_id", "")
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

        if self.get_parameter("warmup_plan").value:
            self.commander.wait_until_plannable(
                target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
            )

        plan = self.commander.plan_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
        )
        self._log_plan(plan)
        if not plan.success:
            self.get_logger().error(f"plan FAILED ({plan.error_name}); not executing.")
            return plan
        if not execute:
            self.get_logger().info("plan-only mode: NOT executing. Re-run with execute to move.")
            return plan

        self.get_logger().warn(WARN_BANNER)
        moved = self.commander.move_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
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
