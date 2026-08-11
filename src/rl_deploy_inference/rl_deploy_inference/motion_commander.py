"""Reusable headless MoveIt client for the dual-arm iiwa (RIGHT arm = ``arm_two``).

A thin, synchronous wrapper on top of an already-running ``move_group`` (launched separately, e.g.
``ros2 launch lbr_dual_arm_y_gripper_bringup move_group.launch.py mode:=hardware rviz:=false``). It is
NOT a MoveIt process itself -- it only talks to the running one via the standard
``moveit_msgs/action/MoveGroup`` action, so it needs no robot_description / SRDF / kinematics params
of its own. This mirrors the supervisor's ``Masterthesis-vision/src/calibration/moveit_dual_arm.py``
(``DualArmMoveitClient``), which drives single arms on this exact rig the same way -- we deliberately
reuse that proven approach rather than invent one.

Key facts that shaped this (all verified against the kuka stack + the supervisor's notes):

* **Single-arm only -- only the inserting arm moves.** A goal for planning group ``arm_two`` covers
  just the right arm's 7 joints; ``arm_one`` is never commanded. Execution is MoveIt-native
  (``plan_only=False``), NOT a padded full-controller goal.
* **Execution needs ``allow_partial_joints_goal: true``** on the dual-arm ``joint_trajectory_controller``
  (it spans all 14 joints). Without it the controller rejects a 7-joint trajectory with
  *"Joints on incoming trajectory don't match the controller joints"*. The supervisor added this flag
  to ``dual_arm_controllers.yaml`` (0/14 -> 13/14 single-arm executes). Plan-only is unaffected.
* **Planning frame == perception frame == ``base_link``** (the URDF root; ``lbr_two_link_0`` is a fixed
  child at ``xyz=0 0.42 0``). So a socket pose from perception (already in ``base_link``) and a
  ``+z`` global hover offset go straight into the goal with no frame conversion.
* **TF is namespaced** under ``/lbr_dual_arm_y_gripper/tf``; run the owning node in that namespace so
  the ``TransformListener`` (relative ``tf``) resolves, and use relative action/topic names.

ROS quaternions are ``xyzw`` everywhere here (geometry_msgs / TF convention). Do NOT mix in the RL
stack's ``wxyz`` helpers.

Spinning: the class never owns an executor. For one-shot CLI use, pass nothing -- it drives the node
inline via ``spin_until_future_complete``. For a long-lived node already spun by an executor (serving
services), pass ``background_spin=True`` so it blocks on futures via an event instead of nested-spin.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

# MoveIt messages live in a MoveIt install (ros-humble-moveit). They are only needed once we actually
# plan, so import them softly: the module still imports (and TF / joint-state readback still work) on a
# box without MoveIt, and plan/move raise a clear, actionable error instead of an ImportError at import.
try:
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        BoundingVolume,
        Constraints,
        JointConstraint,
        MotionPlanRequest,
        MoveItErrorCodes,
        OrientationConstraint,
        PlanningOptions,
        PositionConstraint,
        WorkspaceParameters,
    )

    _MOVEIT_AVAILABLE = True
    _MOVEIT_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on runtime install
    _MOVEIT_AVAILABLE = False
    _MOVEIT_IMPORT_ERROR = exc


PositionXYZ = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def moveit_available() -> bool:
    """True if moveit_msgs could be imported (i.e. planning is possible in this environment)."""
    return _MOVEIT_AVAILABLE


def error_code_name(code: int) -> str:
    """Human-readable name for a MoveItErrorCodes value (e.g. 1 -> 'SUCCESS')."""
    if not _MOVEIT_AVAILABLE:
        return f"CODE_{code}"
    for name in dir(MoveItErrorCodes):
        if name.isupper() and getattr(MoveItErrorCodes, name) == code:
            return name
    return f"CODE_{code}"


@dataclass
class PlanResult:
    success: bool
    error_code: int
    error_name: str
    moved: bool = False  # True only if this was an execute (plan_only=False) that actually ran
    planning_time_s: float = 0.0
    trajectory: object | None = None  # trajectory_msgs/JointTrajectory (arm-group joints only)
    message: str = ""

    @property
    def num_points(self) -> int:
        traj = self.trajectory
        return len(traj.points) if traj is not None and traj.points else 0

    @property
    def duration_s(self) -> float:
        traj = self.trajectory
        if traj is None or not traj.points:
            return 0.0
        t = traj.points[-1].time_from_start
        return t.sec + t.nanosec * 1e-9

    def joint_targets(self) -> tuple[list[str], list[float], list[float]]:
        """(joint_names, first_point_positions, last_point_positions); empty if no trajectory."""
        traj = self.trajectory
        if traj is None or not traj.points:
            return [], [], []
        return list(traj.joint_names), list(traj.points[0].positions), list(traj.points[-1].positions)

    def summary(self) -> str:
        return (
            f"success={self.success} moved={self.moved} code={self.error_name}({self.error_code}) "
            f"points={self.num_points} duration={self.duration_s:.2f}s "
            f"planning_time={self.planning_time_s:.3f}s"
            + (f" -- {self.message}" if self.message else "")
        )


@dataclass
class MotionCommanderConfig:
    group_name: str = "arm_two"
    base_frame: str = "base_link"
    tip_link: str = "lbr_two_gripper_tcp"
    # Relative names -- the owning node must run in the robot namespace (e.g. /lbr_dual_arm_y_gripper)
    # so these and the TransformListener's tf/tf_static resolve there.
    move_group_action: str = "move_action"
    joint_states_topic: str = "joint_states"
    pipeline_id: str = "ompl"
    planner_id: str = ""
    num_planning_attempts: int = 10
    allowed_planning_time_s: float = 5.0
    max_velocity_scaling: float = 0.05
    max_acceleration_scaling: float = 0.05


class MotionCommander:
    """Plan (dry-run) and execute right-arm motions through a running move_group."""

    def __init__(self, node: Node, config: MotionCommanderConfig, *, background_spin: bool = False) -> None:
        self._node = node
        self._log = node.get_logger()
        self.cfg = config
        self._background_spin = background_spin

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node, spin_thread=False)

        self._joint_positions: dict[str, float] = {}
        self._joint_lock = threading.Lock()
        node.create_subscription(JointState, config.joint_states_topic, self._on_joint_state, 10)

        self._client = (
            ActionClient(node, MoveGroup, config.move_group_action) if _MOVEIT_AVAILABLE else None
        )

    # ---- spin helpers -------------------------------------------------------------------------
    def _wait_future(self, future, timeout_s: float) -> bool:
        if self._background_spin:
            done = threading.Event()
            future.add_done_callback(lambda _f: done.set())
            if not done.wait(timeout_s):
                return False
            return future.done()
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
        return future.done()

    def _pump(self, timeout_s: float) -> None:
        if self._background_spin:
            time.sleep(min(timeout_s, 0.05))
        else:
            rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def _sleep(self, duration_s: float) -> None:
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self._pump(0.05)

    # ---- inputs -------------------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        with self._joint_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_positions[name] = float(pos)

    def current_joint_positions(self) -> dict[str, float]:
        with self._joint_lock:
            return dict(self._joint_positions)

    def wait_for_server(self, timeout_s: float = 10.0) -> bool:
        if self._client is None:
            return False
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def get_link_pose(
        self, source_link: str | None = None, target_frame: str | None = None, timeout_s: float = 3.0
    ) -> tuple[PositionXYZ, QuaternionXYZW]:
        """Pose of ``source_link`` in ``target_frame`` via TF -> (xyz, quat xyzw).

        Defaults to (tip_link in base_frame) = the current TCP pose in the planning frame.
        """
        source_link = source_link or self.cfg.tip_link
        target_frame = target_frame or self.cfg.base_frame
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if self._tf_buffer.can_transform(target_frame, source_link, RclpyTime()):
                    tf = self._tf_buffer.lookup_transform(target_frame, source_link, RclpyTime())
                    t = tf.transform.translation
                    q = tf.transform.rotation
                    return ((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
            except TransformException as exc:  # transient while the tree fills
                last_err = exc
            self._pump(0.1)
        raise LookupError(
            f"TF {target_frame} <- {source_link} unavailable after {timeout_s:.1f}s"
            + (f" ({last_err})" if last_err else "")
        )

    # ---- goal construction --------------------------------------------------------------------
    def _require_moveit(self) -> None:
        if not _MOVEIT_AVAILABLE:
            raise RuntimeError(
                "moveit_msgs is not importable, so planning is impossible. Install MoveIt "
                "(e.g. `sudo apt install ros-humble-moveit`) and source it before running. "
                f"Original import error: {_MOVEIT_IMPORT_ERROR}"
            )

    def _pose_constraints(
        self, position_xyz: PositionXYZ, quaternion_xyzw: QuaternionXYZW,
        position_tolerance_m: float, orientation_tolerance_rad: float,
    ):
        header = Header(frame_id=self.cfg.base_frame)

        pos = PositionConstraint()
        pos.header = header
        pos.link_name = self.cfg.tip_link
        pos.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(position_tolerance_m)]
        pos.constraint_region = BoundingVolume(
            primitives=[sphere],
            primitive_poses=[
                Pose(
                    position=Point(x=float(position_xyz[0]), y=float(position_xyz[1]), z=float(position_xyz[2])),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            ],
        )
        pos.weight = 1.0

        ori = OrientationConstraint()
        ori.header = header
        ori.link_name = self.cfg.tip_link
        ori.orientation = Quaternion(
            x=float(quaternion_xyzw[0]), y=float(quaternion_xyzw[1]),
            z=float(quaternion_xyzw[2]), w=float(quaternion_xyzw[3]),
        )
        ori.absolute_x_axis_tolerance = float(orientation_tolerance_rad)
        ori.absolute_y_axis_tolerance = float(orientation_tolerance_rad)
        ori.absolute_z_axis_tolerance = float(orientation_tolerance_rad)
        ori.weight = 1.0

        c = Constraints()
        c.name = "preinsert_pose"
        c.position_constraints = [pos]
        c.orientation_constraints = [ori]
        return c

    def _joint_constraints(self, joint_targets: dict[str, float], tolerance_rad: float):
        c = Constraints()
        c.name = "joint_goal"
        for name, val in joint_targets.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = float(tolerance_rad)
            jc.tolerance_below = float(tolerance_rad)
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _build_goal(self, goal_constraints: list, plan_only: bool):
        req = MotionPlanRequest()
        req.group_name = self.cfg.group_name
        req.goal_constraints = goal_constraints
        req.num_planning_attempts = int(self.cfg.num_planning_attempts)
        req.allowed_planning_time = float(self.cfg.allowed_planning_time_s)
        req.max_velocity_scaling_factor = float(self.cfg.max_velocity_scaling)
        req.max_acceleration_scaling_factor = float(self.cfg.max_acceleration_scaling)
        req.pipeline_id = self.cfg.pipeline_id
        req.planner_id = self.cfg.planner_id
        req.start_state.is_diff = True  # start from move_group's live current state
        ws = WorkspaceParameters()
        ws.header = Header(frame_id=self.cfg.base_frame)
        ws.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        ws.max_corner = Vector3(x=2.0, y=2.0, z=2.0)
        req.workspace_parameters = ws

        goal = MoveGroup.Goal()
        goal.request = req
        opts = PlanningOptions()
        opts.plan_only = plan_only
        opts.replan = False
        opts.planning_scene_diff.is_diff = True
        opts.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options = opts
        return goal

    # ---- send/wait ----------------------------------------------------------------------------
    def _send_and_wait(self, goal, plan_only: bool, timeout_s: float) -> PlanResult:
        if not self._client.wait_for_server(timeout_sec=5.0):
            return PlanResult(False, 0, "NO_MOVE_GROUP", message="move_group action server unavailable")
        send_future = self._client.send_goal_async(goal)
        if not self._wait_future(send_future, timeout_s):
            return PlanResult(False, 0, "SEND_TIMEOUT", message="timed out sending goal")
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return PlanResult(False, 0, "REJECTED", message="move_group rejected the goal")
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, timeout_s):
            return PlanResult(False, 0, "RESULT_TIMEOUT", message="timed out waiting for result")
        wrapped = result_future.result()
        if wrapped is None:
            return PlanResult(False, 0, "NO_RESULT", message="result future resolved empty")
        result = wrapped.result
        code = int(result.error_code.val)
        status_ok = wrapped.status == GoalStatus.STATUS_SUCCEEDED
        traj = result.planned_trajectory.joint_trajectory
        ok = status_ok and code == MoveItErrorCodes.SUCCESS and bool(traj.points)
        return PlanResult(
            success=ok,
            error_code=code,
            error_name=error_code_name(code),
            moved=(ok and not plan_only),
            planning_time_s=float(result.planning_time),
            trajectory=traj if traj.points else None,
        )

    # ---- public API ---------------------------------------------------------------------------
    def plan_to_pose(
        self, position_xyz: PositionXYZ, quaternion_xyzw: QuaternionXYZW, *,
        position_tolerance_m: float = 0.01, orientation_tolerance_rad: float = 0.1,
        plan_timeout_s: float = 15.0,
    ) -> PlanResult:
        """Plan the tip to a Cartesian pose in ``base_frame``. Never moves the robot."""
        self._require_moveit()
        goal = self._build_goal(
            [self._pose_constraints(position_xyz, quaternion_xyzw, position_tolerance_m, orientation_tolerance_rad)],
            plan_only=True,
        )
        return self._send_and_wait(goal, plan_only=True, timeout_s=plan_timeout_s)

    def move_to_pose(
        self, position_xyz: PositionXYZ, quaternion_xyzw: QuaternionXYZW, *,
        position_tolerance_m: float = 0.01, orientation_tolerance_rad: float = 0.1,
        exec_timeout_s: float = 120.0,
    ) -> PlanResult:
        """Plan AND execute the tip to a Cartesian pose (MoveIt-native, single arm).

        Needs ``allow_partial_joints_goal: true`` on the dual-arm joint_trajectory_controller, and
        that controller must be the ACTIVE one (the RL command controller inactive).
        """
        self._require_moveit()
        goal = self._build_goal(
            [self._pose_constraints(position_xyz, quaternion_xyzw, position_tolerance_m, orientation_tolerance_rad)],
            plan_only=False,
        )
        return self._send_and_wait(goal, plan_only=False, timeout_s=exec_timeout_s)

    def plan_to_joint(self, joint_targets: dict[str, float], *, tolerance_rad: float = 0.001, plan_timeout_s: float = 15.0) -> PlanResult:
        """Plan the group to explicit joint targets (reuse: named poses, retract, etc.). No motion."""
        self._require_moveit()
        goal = self._build_goal([self._joint_constraints(joint_targets, tolerance_rad)], plan_only=True)
        return self._send_and_wait(goal, plan_only=True, timeout_s=plan_timeout_s)

    def move_to_joint(self, joint_targets: dict[str, float], *, tolerance_rad: float = 0.001, exec_timeout_s: float = 120.0) -> PlanResult:
        """Plan AND execute the group to explicit joint targets (skips IK)."""
        self._require_moveit()
        goal = self._build_goal([self._joint_constraints(joint_targets, tolerance_rad)], plan_only=False)
        return self._send_and_wait(goal, plan_only=False, timeout_s=exec_timeout_s)

    def wait_until_plannable(
        self, position_xyz: PositionXYZ, quaternion_xyzw: QuaternionXYZW, *,
        position_tolerance_m: float = 0.01, orientation_tolerance_rad: float = 0.1,
        timeout_s: float = 30.0, retry_interval_s: float = 1.0,
    ) -> bool:
        """Retry plan-only toward a pose until move_group's current-state monitor is ready.

        Right after move_group starts, its planning-scene monitor can be "dirty" (no valid link
        transforms yet) even once the action server answers, giving a spurious generic FAILURE on the
        first real goal. Probing plan-only until it succeeds once absorbs that startup window. (Learned
        from the supervisor's DualArmMoveitClient.wait_for_valid_state.)
        """
        self._require_moveit()
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while True:
            attempt += 1
            res = self.plan_to_pose(
                position_xyz, quaternion_xyzw,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
            if res.success:
                if attempt > 1:
                    self._log.info(f"move_group ready after {attempt} plan-only probes.")
                return True
            if time.monotonic() >= deadline:
                self._log.warn(
                    f"move_group not plannable after {timeout_s:.0f}s "
                    f"(last {res.error_name}); proceeding may still fail."
                )
                return False
            self._log.warn(
                f"[probe {attempt}] not plannable yet ({res.error_name}); retrying in {retry_interval_s:.1f}s..."
            )
            self._sleep(retry_interval_s)
