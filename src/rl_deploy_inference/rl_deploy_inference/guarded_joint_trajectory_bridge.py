"""Guarded 15 Hz policy-command bridge for ros2_control's trajectory controller."""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from lbr_fri_idl.msg import LBRJointPositionCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .guarded_joint_smoother import GuardedJointSmoother


IIWA7_LOWER = np.deg2rad(np.array([-170, -120, -170, -120, -170, -120, -175], dtype=np.float64))
IIWA7_UPPER = np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175], dtype=np.float64))


def _duration(seconds: float) -> Duration:
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


class GuardedJointTrajectoryBridge(Node):
    """Convert policy setpoints into bounded, continuous q/dq/ddq horizons."""

    def __init__(self) -> None:
        super().__init__("guarded_joint_trajectory_bridge")
        self.declare_parameter("input_topic", "/rl_deploy/command_15hz")
        self.declare_parameter("joint_state_topic", "/lbr_dual_arm_y_gripper/joint_states")
        self.declare_parameter(
            "output_topic",
            "/lbr_dual_arm_y_gripper/joint_trajectory_controller/joint_trajectory",
        )
        self.declare_parameter("robot_prefix", "lbr_two")
        self.declare_parameter("input_hz", 15.0)
        self.declare_parameter("robot_sample_period_s", 0.01)
        self.declare_parameter("command_horizon_s", 0.08)
        self.declare_parameter("joint_state_max_age_s", 0.05)
        self.declare_parameter("max_target_lead_deg", 0.5)
        self.declare_parameter("max_tracking_error_deg", 0.35)
        self.declare_parameter("max_velocity_deg_s", 3.0)
        self.declare_parameter("max_acceleration_deg_s2", 30.0)
        self.declare_parameter("max_jerk_deg_s3", 300.0)
        self.declare_parameter("min_duration_s", 0.5)
        self.declare_parameter("max_duration_s", 8.0)
        self.declare_parameter("joint_limit_margin_rad", 0.03)

        prefix = str(self.get_parameter("robot_prefix").value)
        self.joint_names = tuple(f"{prefix}_A{i}" for i in range(1, 8))
        self._measured_q: np.ndarray | None = None
        self._measured_t = 0.0
        self._smoother: GuardedJointSmoother | None = None
        self._latched_reason = ""

        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LBRJointPositionCommand,
            self.get_parameter("input_topic").value,
            self._on_command,
            qos_profile_sensor_data,
        )
        self._publisher = self.create_publisher(
            JointTrajectory, self.get_parameter("output_topic").value, 10
        )
        self.create_service(Trigger, "/rl_deploy/clear_bridge_latch", self._clear_latch)
        self.get_logger().info(
            "guarded bridge ready: 15 Hz targets -> JTC q/dq/ddq horizons; "
            "motion waits for a fresh measured joint state"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in self.joint_names):
            return
        self._measured_q = np.array(
            [positions[name] for name in self.joint_names], dtype=np.float64
        )
        self._measured_t = time.monotonic()

    def _measured_is_fresh(self) -> bool:
        max_age = float(self.get_parameter("joint_state_max_age_s").value)
        return self._measured_q is not None and time.monotonic() - self._measured_t <= max_age

    def _new_smoother(self, q_start: np.ndarray) -> GuardedJointSmoother:
        margin = float(self.get_parameter("joint_limit_margin_rad").value)
        return GuardedJointSmoother(
            q_start,
            np.deg2rad(float(self.get_parameter("max_velocity_deg_s").value)),
            np.deg2rad(float(self.get_parameter("max_acceleration_deg_s2").value)),
            np.deg2rad(float(self.get_parameter("max_jerk_deg_s3").value)),
            min_duration_s=float(self.get_parameter("min_duration_s").value),
            max_duration_s=float(self.get_parameter("max_duration_s").value),
            lower=IIWA7_LOWER + margin,
            upper=IIWA7_UPPER - margin,
        )

    def _on_command(self, msg: LBRJointPositionCommand) -> None:
        if self._latched_reason:
            return
        target = np.asarray(msg.joint_position, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            self._latch("policy target must contain seven finite joint positions")
            return
        if not self._measured_is_fresh():
            if self._smoother is None:
                self.get_logger().warn(
                    "waiting for fresh measured joints before forwarding policy commands",
                    throttle_duration_sec=2.0,
                )
            else:
                self._latch("measured joint state went stale")
            return

        measured = self._measured_q.copy()
        if self._smoother is None:
            try:
                self._smoother = self._new_smoother(measured)
            except ValueError as exc:
                self._latch(f"cannot initialize smoother: {exc}")
                return
        else:
            tracking_limit = np.deg2rad(
                float(self.get_parameter("max_tracking_error_deg").value)
            )
            tracking_error = float(np.max(np.abs(measured - self._smoother.state.q)))
            if tracking_limit > 0.0 and tracking_error > tracking_limit:
                self._latch(
                    f"joint tracking error {np.rad2deg(tracking_error):.3f} deg exceeds "
                    f"{np.rad2deg(tracking_limit):.3f} deg"
                )
                return

        lead_limit = np.deg2rad(float(self.get_parameter("max_target_lead_deg").value))
        target_lead = float(np.max(np.abs(target - measured)))
        if lead_limit > 0.0 and target_lead > lead_limit:
            self._latch(
                f"target lead {np.rad2deg(target_lead):.3f} deg exceeds "
                f"{np.rad2deg(lead_limit):.3f} deg"
            )
            return

        sample_dt = float(self.get_parameter("robot_sample_period_s").value)
        input_period = 1.0 / float(self.get_parameter("input_hz").value)
        horizon = max(
            float(self.get_parameter("command_horizon_s").value), input_period + sample_dt
        )
        sample_count = max(3, int(math.ceil(horizon / sample_dt)))
        try:
            plan = self._smoother.plan(
                target,
                sample_dt_s=sample_dt,
                sample_count=sample_count,
                handoff_time_s=input_period,
            )
        except (ValueError, RuntimeError) as exc:
            self._latch(f"trajectory smoothing failed: {exc}")
            return

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        for q, t, dq, ddq in plan.samples:
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            point.velocities = dq.tolist()
            point.accelerations = ddq.tolist()
            point.time_from_start = _duration(t)
            trajectory.points.append(point)
        self._publisher.publish(trajectory)
        self._smoother.commit(plan.handoff_state)

    def _publish_measured_hold(self) -> None:
        if not self._measured_is_fresh():
            return
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        for seconds in (0.01, 0.10):
            point = JointTrajectoryPoint()
            point.positions = self._measured_q.tolist()
            point.velocities = [0.0] * 7
            point.accelerations = [0.0] * 7
            point.time_from_start = _duration(seconds)
            trajectory.points.append(point)
        self._publisher.publish(trajectory)

    def _latch(self, reason: str) -> None:
        if self._latched_reason:
            return
        self._latched_reason = reason
        self.get_logger().error(
            f"GUARDED BRIDGE LATCHED: {reason}; publishing measured hold. "
            "Inspect the cause, then call /rl_deploy/clear_bridge_latch."
        )
        self._publish_measured_hold()

    def _clear_latch(self, _request, response):
        if not self._measured_is_fresh():
            response.success = False
            response.message = "cannot clear bridge latch without a fresh measured joint state"
            return response
        previous = self._latched_reason or "none"
        self._latched_reason = ""
        self._smoother = None
        self._publish_measured_hold()
        response.success = True
        response.message = f"bridge latch cleared (previous reason: {previous})"
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GuardedJointTrajectoryBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
