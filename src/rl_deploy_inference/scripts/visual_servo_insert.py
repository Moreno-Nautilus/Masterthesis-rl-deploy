#!/usr/bin/env python3
"""Vision-only insertion baseline: corrected hole-align preinsert, then visual-servo descent.

This is a comparison path for the RL policy. It deliberately does not load or call the actor. The
motion sequence is:

  1. Detect the real socket opening with the same wrist RGB-D hole detector used by hole_align.
  2. Move to the corrected preinsert hover with MoveIt / joint_trajectory_controller.
  3. Switch to the LBR joint-position command controller.
  4. Servo the TCP down toward the detected opening with damped IK at 200 Hz, keeping XY centered.

The default is dry-run. Use --execute to move the real robot; without --yes it asks for typed
confirmation before the MoveIt preinsert and again before the servo descent.

Typical run after sourcing ROS + kuka + vision + this workspace:

    python3 src/rl_deploy_inference/scripts/visual_servo_insert.py --execute --ros-args \
      --params-file src/rl_deploy_inference/config/hole_align.yaml

The script publishes directly to command/joint_position in the chosen robot namespace at 200 Hz, so do
not run command_upsampler at the same time unless you override command_topic intentionally.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow direct execution from the source checkout without requiring this script to be installed as a
# console entry point. If the workspace is installed/sourced, the installed package still wins normally.
_PKG_SRC = Path(__file__).resolve().parents[1]
if (_PKG_SRC / "rl_deploy_inference").is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import message_filters
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import WrenchStamped
from lbr_fri_idl.msg import LBRJointPositionCommand
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time as RclpyTime
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

from rl_deploy_inference.hole_align_planner import (
    ARM_CAMERA,
    _DEFAULT_EXTRINSICS,
    _decode_depth_m,
    _decode_rgb,
    HoleAlignPlanner,
)
from rl_deploy_inference.ik import get_delta_dof_pos, get_pose_error, quat_wxyz_from_rotmat
from rl_deploy_inference.kinematics import KdlConfig, KdlKinematics, KinematicsUnavailable
from rl_deploy_inference.motion_commander import PlanResult, moveit_available
from rl_deploy_inference.preinsert_planner import (
    ARM_DEFAULTS,
    ARM_JOINTS,
    IIWA7_LOWER,
    IIWA7_UPPER,
    _fmt_quat,
    _fmt_xyz,
    compute_preinsert_target,
    warn_banner,
)
from rl_deploy_inference import hole_detector as hd


@dataclass
class TimedValue:
    value: object | None = None
    wall_t: float = 0.0

    def fresh(self, max_age_s: float) -> bool:
        return self.value is not None and (time.monotonic() - self.wall_t) <= float(max_age_s)


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    wall_t: float


@dataclass
class ServoResult:
    success: bool
    outcome: str
    message: str
    duration_s: float = 0.0

    def summary(self) -> str:
        return (
            f"success={self.success} outcome={self.outcome} duration={self.duration_s:.2f}s"
            + (f" -- {self.message}" if self.message else "")
        )


class VisualServoInsert(Node):
    """Damped-IK visual-servo insertion node.

    The node recomputes IK setpoints and streams them at the high-rate command frequency. It can use a
    frozen hole captured by hole_align, live detector updates, or live updates with frozen fallback for
    the normal centered/occluded view.
    """

    def __init__(
        self,
        *,
        namespace: str,
        arm_override: str | None,
        hole_base: np.ndarray | None,
    ) -> None:
        super().__init__("visual_servo_insert", namespace=namespace)
        self._declare_parameters()

        arm = (arm_override or self.get_parameter("arm").value or "right").strip().lower()
        if arm not in ARM_DEFAULTS:
            raise ValueError(f"arm must be one of {sorted(ARM_DEFAULTS)}, got {arm!r}")
        self._arm = arm
        _default_group, default_tip = ARM_DEFAULTS[arm]
        default_cam, default_flange = ARM_CAMERA[arm]
        self._tip_link = self.get_parameter("tip_link").value or default_tip
        self._cam_id = self.get_parameter("camera_id").value or default_cam
        self._flange_link = self.get_parameter("flange_link").value or default_flange
        self._joint_names = tuple(ARM_JOINTS[arm])
        self._joint_state_topic = (
            self.get_parameter("joint_state_topic").value
            or self.get_parameter("joint_states_topic").value
        )
        self._command_topic = self.get_parameter("command_topic").value

        self._hole_base = None if hole_base is None else np.asarray(hole_base, dtype=np.float64).reshape(3)
        self._kinematics: KdlKinematics | None = None
        self._last_q_cmd: np.ndarray | None = None
        self._last_stream_t = 0.0
        self._last_control_t = 0.0
        self._last_status_t = 0.0
        self._last_debug_t = 0.0
        self._last_vision_t = 0.0
        self._last_vision_attempt_t = 0.0
        self._last_detected_uv: tuple[float, float] | None = None
        self._started_t = 0.0
        self._ft_baseline: np.ndarray | None = None
        self._manual_estop = False
        self._stream_enabled = False

        self.joint_pos = TimedValue()
        self.ft_force = TimedValue()
        self.frame = TimedValue()
        self.intrinsics: hd.CameraIntrinsics | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self._R_ee_cam = None
        self._t_ee_cam = None
        try:
            self._R_ee_cam, self._t_ee_cam = hd.load_cam_flange_extrinsics(
                self.get_parameter("extrinsics_yaml").value,
                self._cam_id,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"live visual target updates unavailable: could not load extrinsics for {self._cam_id} "
                f"from {self.get_parameter('extrinsics_yaml').value!r}: {exc}"
            )

        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            WrenchStamped,
            self.get_parameter("ft_topic").value,
            self._on_ft,
            qos_profile_sensor_data,
        )
        self.create_subscription(Bool, self.get_parameter("estop_topic").value, self._on_estop, 1)

        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("rgb_topic").value, qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=qos_profile_sensor_data
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=int(self.get_parameter("img_sync_queue").value),
            slop=float(self.get_parameter("img_sync_slop_s").value),
        )
        self._sync.registerCallback(self._on_rgbd)
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        self.command_pub = self.create_publisher(
            LBRJointPositionCommand,
            self._command_topic,
            1,
        )
        self.status_pub = self.create_publisher(String, self.get_parameter("status_topic").value, 10)
        self.seat_pub = self.create_publisher(Bool, self.get_parameter("seat_topic").value, 1)

        self.get_logger().info(
            f"visual_servo_insert ready (namespace={self.get_namespace()}, arm={self._arm}, "
            f"tip={self._tip_link}, joint_state_topic={self._joint_state_topic}, "
            f"command_topic={self._command_topic}, "
            f"target_mode={self.get_parameter('target_mode').value})."
        )

    # ---- parameters -------------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("arm", "right")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tip_link", "")
        self.declare_parameter("joint_state_topic", "")  # singular alias used by deploy.yaml
        self.declare_parameter("joint_states_topic", "joint_states")  # plural alias used by hole_align.yaml
        self.declare_parameter("command_topic", "command/joint_position")
        self.declare_parameter("status_topic", "/rl_deploy/visual_servo_status")
        self.declare_parameter("seat_topic", "/rl_deploy/visual_servo_seat_detected")

        self.declare_parameter("robot_description", "")
        self.declare_parameter(
            "robot_description_service",
            "robot_state_publisher/get_parameters",
        )
        self.declare_parameter("fallback_tip_link", "")
        self.declare_parameter("flange_to_fingertip_xyz", [0.0, 0.0, 0.1463])
        self.declare_parameter("flange_to_fingertip_quat_wxyz", [1.0, 0.0, 0.0, 0.0])

        self.declare_parameter("control_hz", 200.0)
        self.declare_parameter("stream_hz", 200.0)
        self.declare_parameter("input_timeout_s", 0.35)
        self.declare_parameter("startup_timeout_s", 8.0)
        self.declare_parameter("trial_timeout_s", 35.0)
        self.declare_parameter("dry_run_s", 2.0)
        self.declare_parameter("debug_print", True)
        self.declare_parameter("debug_print_hz", 5.0)
        self.declare_parameter("debug_print_q_cmd", True)

        self.declare_parameter("ik_damping", 0.1)
        self.declare_parameter("max_joint_step_rad", 0.00075)
        self.declare_parameter("joint_limit_margin_rad", 0.03)
        self.declare_parameter("xy_gain", 0.6)
        self.declare_parameter("max_xy_step_m", 0.00025)
        self.declare_parameter("descent_speed_mps", 0.012)
        self.declare_parameter("descent_xy_gate_m", 0.004)
        self.declare_parameter("abort_xy_error_m", 0.020)
        self.declare_parameter("insertion_depth_m", 0.0175)
        self.declare_parameter("seat_z_tolerance_m", 0.002)
        self.declare_parameter("seat_xy_tolerance_m", 0.004)
        self.declare_parameter("trial_max_overtravel_m", 0.002)

        self.declare_parameter("ft_topic", "force_torque_broadcaster/wrench")
        self.declare_parameter("require_ft", True)
        self.declare_parameter("ft_baseline_on_start", True)
        self.declare_parameter("ft_force_cap_n", 20.0)
        self.declare_parameter("seat_force_n", 14.0)
        self.declare_parameter("seat_force_z_window_m", 0.008)
        self.declare_parameter("estop_topic", "/rl_deploy/e_stop")

        self.declare_parameter("controller_manager_service", "")
        self.declare_parameter("activate_controller", "lbr_joint_position_command_controller")
        self.declare_parameter("deactivate_controller", "joint_trajectory_controller")
        self.declare_parameter("verify_command_controller", True)
        self.declare_parameter("controller_switch_timeout_s", 8.0)
        self.declare_parameter("pre_switch_hold_s", 0.5)
        self.declare_parameter("post_switch_hold_s", 0.5)

        # Live visual target updates. Defaults to live-fallback because the centered screw can occlude
        # the socket; the frozen target captured before the descent remains the safe reference.
        self.declare_parameter("target_mode", "live-fallback")  # frozen | live | live-fallback
        self.declare_parameter("vision_update_hz", 4.0)
        self.declare_parameter("live_target_alpha", 0.35)
        self.declare_parameter("live_timeout_s", 1.5)

        self.declare_parameter("rgb_topic", "/realsense_2/camera/color/image_rect")
        self.declare_parameter("depth_topic", "/realsense_2/camera/aligned_depth_to_color/image_rect")
        self.declare_parameter("camera_info_topic", "/realsense_2/camera/color/camera_info")
        self.declare_parameter("camera_id", "")
        self.declare_parameter("flange_link", "")
        self.declare_parameter("extrinsics_yaml", _DEFAULT_EXTRINSICS)
        self.declare_parameter("img_sync_slop_s", 0.05)
        self.declare_parameter("img_sync_queue", 30)
        self.declare_parameter("frame_timeout_s", 5.0)
        self.declare_parameter("tf_timeout_s", 1.0)
        self.declare_parameter("max_center_dist_px", 60.0)

        # Same detector knobs as hole_align.yaml, so that params file can tune this script too.
        self.declare_parameter("method", "contour")
        self.declare_parameter("hole_diameter_m", 0.014)
        self.declare_parameter("radius_tol_frac", 0.45)
        self.declare_parameter("socket_spacing_m", 0.060)
        self.declare_parameter("socket_spacing_tol_m", 0.012)
        self.declare_parameter("use_socket_pair", True)
        self.declare_parameter("use_saturation_mask", True)
        self.declare_parameter("sat_min", 60)
        self.declare_parameter("val_min", 40)
        self.declare_parameter("adaptive_block", 51)
        self.declare_parameter("adaptive_C", 5)
        self.declare_parameter("morph_open", False)
        self.declare_parameter("min_circularity", 0.65)
        self.declare_parameter("max_aspect", 1.8)
        self.declare_parameter("min_solidity", 0.80)
        self.declare_parameter("min_fill", 0.55)
        self.declare_parameter("require_part_surround", True)
        self.declare_parameter("surround_part_frac", 0.85)
        self.declare_parameter("blur_ksize", 5)
        self.declare_parameter("hough_dp", 1.2)
        self.declare_parameter("hough_param1", 120.0)
        self.declare_parameter("hough_param2", 18.0)
        self.declare_parameter("detect_on", "rgb")
        self.declare_parameter("depth_valid_min_m", 0.05)
        self.declare_parameter("depth_valid_max_m", 0.60)

    # ---- callbacks --------------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        by_name = {name: pos for name, pos in zip(msg.name, msg.position)}
        if not all(name in by_name for name in self._joint_names):
            return
        self.joint_pos = TimedValue(
            np.array([by_name[name] for name in self._joint_names], dtype=np.float64),
            time.monotonic(),
        )

    def _on_ft(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        self.ft_force = TimedValue(np.array([f.x, f.y, f.z], dtype=np.float64), time.monotonic())

    def _on_estop(self, msg: Bool) -> None:
        self._manual_estop = bool(msg.data)

    def _on_rgbd(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            rgb = _decode_rgb(rgb_msg)
            depth = _decode_depth_m(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"RGB-D decode failed: {exc}", throttle_duration_sec=5.0)
            return
        self.frame = TimedValue(Frame(rgb, depth, time.monotonic()), time.monotonic())

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.intrinsics = hd.CameraIntrinsics.from_k(msg.k)

    # ---- wait/spin helpers ------------------------------------------------------------------
    def _pump(self, timeout_s: float) -> None:
        rclpy.spin_once(self, timeout_sec=timeout_s)

    def _wait_for(self, predicate, timeout_s: float, label: str) -> bool:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return True
            self._maybe_stream()
            self._pump(0.02)
        ok = bool(predicate())
        if not ok:
            self.get_logger().error(f"timed out waiting for {label} ({timeout_s:.1f}s)")
        return ok

    def _status(self, text: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_t < 1.0:
            return
        self._last_status_t = now
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    # ---- setup ------------------------------------------------------------------------------
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
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self._maybe_stream()
            self._pump(0.02)
        if future.result() is None or not future.result().values:
            return None
        return future.result().values[0].string_value or None

    def _init_kinematics(self) -> bool:
        if self._kinematics is not None:
            return True
        desc = self._retrieve_robot_description(timeout_s=1.5)
        if not desc:
            self.get_logger().error("no robot_description available for KDL visual servo IK")
            return False
        fallback_tip = self.get_parameter("fallback_tip_link").value
        if not fallback_tip:
            fallback_tip = self._flange_link
        try:
            self._kinematics = KdlKinematics(
                KdlConfig(
                    robot_description=desc,
                    base_link=self.get_parameter("base_frame").value,
                    tip_link=self._tip_link,
                    joint_names=self._joint_names,
                    fallback_tip_link=fallback_tip,
                    fallback_tip_offset_xyz=tuple(self.get_parameter("flange_to_fingertip_xyz").value),
                    fallback_tip_offset_quat_wxyz=tuple(
                        self.get_parameter("flange_to_fingertip_quat_wxyz").value
                    ),
                )
            )
            self.get_logger().info(
                f"KDL ready for visual servo: {self.get_parameter('base_frame').value} -> {self._tip_link}"
            )
            return True
        except KinematicsUnavailable as exc:
            self.get_logger().error(str(exc))
            return False

    def _prepare_inputs(self) -> bool:
        startup = float(self.get_parameter("startup_timeout_s").value)
        timeout = float(self.get_parameter("input_timeout_s").value)
        if not self._wait_for(lambda: self.joint_pos.fresh(timeout), startup, "fresh joint state"):
            return False
        self._last_q_cmd = np.asarray(self.joint_pos.value, dtype=np.float64).reshape(7).copy()
        if bool(self.get_parameter("require_ft").value):
            if not self._wait_for(lambda: self.ft_force.fresh(timeout), startup, "fresh F/T wrench"):
                return False
        if self.ft_force.value is not None and bool(self.get_parameter("ft_baseline_on_start").value):
            self._ft_baseline = np.asarray(self.ft_force.value, dtype=np.float64).reshape(3).copy()
            self.get_logger().warn(
                "captured visual-servo F/T baseline: "
                f"[{self._ft_baseline[0]:+.2f}, {self._ft_baseline[1]:+.2f}, {self._ft_baseline[2]:+.2f}] N"
            )
        if not self._init_kinematics():
            return False
        return True

    # ---- controller handoff -----------------------------------------------------------------
    def _controller_switch_service(self) -> str:
        configured = str(self.get_parameter("controller_manager_service").value or "").strip()
        if configured:
            if configured.endswith("/controller_manager"):
                return f"{configured}/switch_controller"
            return configured
        ns = self.get_namespace().rstrip("/")
        return f"{ns}/controller_manager/switch_controller" if ns else "/controller_manager/switch_controller"

    def _controller_list_service(self) -> str:
        configured = str(self.get_parameter("controller_manager_service").value or "").strip()
        if configured:
            if configured.endswith("/switch_controller"):
                return f"{configured.rsplit('/', 1)[0]}/list_controllers"
            if configured.endswith("/controller_manager"):
                return f"{configured}/list_controllers"
            return f"{configured.rstrip('/')}/list_controllers"
        ns = self.get_namespace().rstrip("/")
        return f"{ns}/controller_manager/list_controllers" if ns else "/controller_manager/list_controllers"

    def _controller_state(self, name: str, timeout_s: float) -> str | None:
        try:
            from controller_manager_msgs.srv import ListControllers
        except ImportError as exc:
            self.get_logger().warn(f"controller_manager_msgs unavailable; cannot verify controllers: {exc}")
            return None

        service = self._controller_list_service()
        client = self.create_client(ListControllers, service)
        if not client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().warn(f"controller list service not available: {service}")
            return None
        future = client.call_async(ListControllers.Request())
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self._maybe_stream()
            self._pump(0.02)
        result = future.result() if future.done() else None
        if result is None:
            self.get_logger().warn("controller list request timed out or returned no result")
            return None
        states = {ctrl.name: ctrl.state for ctrl in result.controller}
        interesting = [
            self.get_parameter("activate_controller").value,
            self.get_parameter("deactivate_controller").value,
        ]
        summary = ", ".join(f"{n}={states.get(n, 'missing')}" for n in interesting if n)
        self.get_logger().info(f"controller states: {summary}")
        return states.get(name)

    def _command_controller_is_active(self) -> bool:
        name = str(self.get_parameter("activate_controller").value or "").strip()
        if not name:
            return True
        state = self._controller_state(name, timeout_s=2.0)
        return state == "active"

    def switch_to_command_controller(self) -> bool:
        try:
            from controller_manager_msgs.srv import SwitchController
        except ImportError as exc:
            self.get_logger().error(f"controller_manager_msgs unavailable; cannot switch controllers: {exc}")
            return False

        service = self._controller_switch_service()
        timeout = float(self.get_parameter("controller_switch_timeout_s").value)
        client = self.create_client(SwitchController, service)
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"controller switch service not available: {service}")
            return False

        req = SwitchController.Request()
        activate = str(self.get_parameter("activate_controller").value or "").strip()
        deactivate = str(self.get_parameter("deactivate_controller").value or "").strip()
        req.activate_controllers = [activate] if activate else []
        req.deactivate_controllers = [deactivate] if deactivate else []
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9))

        self.get_logger().warn(
            f"switching controllers via {service}: activate={req.activate_controllers} "
            f"deactivate={req.deactivate_controllers}"
        )
        future = client.call_async(req)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self._maybe_stream()
            self._pump(0.02)
        result = future.result() if future.done() else None
        ok = bool(result is not None and result.ok)
        if ok:
            self.get_logger().info("controller handoff complete; visual servo command controller is active.")
        else:
            self.get_logger().error("controller handoff failed; refusing visual-servo motion.")
        return ok

    # ---- live target detection ---------------------------------------------------------------
    def _detector_config(self) -> hd.HoleDetectorConfig:
        gp = self.get_parameter
        return hd.HoleDetectorConfig(
            method=str(gp("method").value).strip().lower(),
            hole_diameter_m=float(gp("hole_diameter_m").value),
            radius_tol_frac=float(gp("radius_tol_frac").value),
            socket_spacing_m=float(gp("socket_spacing_m").value),
            socket_spacing_tol_m=float(gp("socket_spacing_tol_m").value),
            use_socket_pair=bool(gp("use_socket_pair").value),
            use_saturation_mask=bool(gp("use_saturation_mask").value),
            sat_min=int(gp("sat_min").value),
            val_min=int(gp("val_min").value),
            adaptive_block=int(gp("adaptive_block").value),
            adaptive_C=int(gp("adaptive_C").value),
            morph_open=bool(gp("morph_open").value),
            min_circularity=float(gp("min_circularity").value),
            max_aspect=float(gp("max_aspect").value),
            min_solidity=float(gp("min_solidity").value),
            min_fill=float(gp("min_fill").value),
            require_part_surround=bool(gp("require_part_surround").value),
            surround_part_frac=float(gp("surround_part_frac").value),
            blur_ksize=int(gp("blur_ksize").value),
            hough_dp=float(gp("hough_dp").value),
            hough_param1=float(gp("hough_param1").value),
            hough_param2=float(gp("hough_param2").value),
            detect_on=str(gp("detect_on").value).strip().lower(),
            depth_valid_min_m=float(gp("depth_valid_min_m").value),
            depth_valid_max_m=float(gp("depth_valid_max_m").value),
        )

    def _get_link_pose(self, source_link: str, timeout_s: float) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        target_frame = self.get_parameter("base_frame").value
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                if self._tf_buffer.can_transform(target_frame, source_link, RclpyTime()):
                    tf = self._tf_buffer.lookup_transform(target_frame, source_link, RclpyTime())
                    t = tf.transform.translation
                    q = tf.transform.rotation
                    return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)
            except TransformException as exc:
                last_err = exc
            self._maybe_stream()
            self._pump(0.02)
        raise LookupError(
            f"TF {target_frame} <- {source_link} unavailable after {timeout_s:.1f}s"
            + (f" ({last_err})" if last_err else "")
        )

    def _capture_live_hole(self, *, wait: bool) -> np.ndarray | None:
        if not hd.opencv_available() or self._R_ee_cam is None or self._t_ee_cam is None:
            return None
        frame_timeout = float(self.get_parameter("frame_timeout_s").value)
        if wait:
            if not self._wait_for(
                lambda: self.frame.fresh(frame_timeout) and self.intrinsics is not None,
                frame_timeout,
                "synced RGB-D frame + CameraInfo",
            ):
                return None
        if not self.frame.fresh(frame_timeout) or self.intrinsics is None:
            return None

        frame = self.frame.value
        intr = self.intrinsics
        assert isinstance(frame, Frame)
        try:
            base_ee_pos, base_ee_quat = self._get_link_pose(
                self._flange_link,
                timeout_s=float(self.get_parameter("tf_timeout_s").value),
            )
        except LookupError as exc:
            self.get_logger().warn(f"live hole update skipped: {exc}", throttle_duration_sec=2.0)
            return None

        expected_uv = None
        if self._hole_base is not None:
            p_cam = hd.base_point_to_cam(
                self._hole_base,
                self._R_ee_cam,
                self._t_ee_cam,
                base_ee_pos,
                base_ee_quat,
            )
            if p_cam[2] > 1e-3:
                expected_uv = hd.project_point_cam(p_cam, intr)

        cfg = self._detector_config()
        ref_depth = hd.reference_depth(frame.depth, cfg)
        detections = hd.detect_holes(frame.rgb, frame.depth, intr, cfg, ref_depth_m=ref_depth)
        gate = float(self.get_parameter("max_center_dist_px").value)
        chosen = hd.select_hole(
            detections,
            intr,
            expected_uv=expected_uv,
            max_center_dist_px=(gate if gate > 0 else None),
        )
        self._last_vision_attempt_t = time.monotonic()
        if chosen is None:
            self.get_logger().warn(
                f"live hole update: no selected detection ({len(detections)} candidate(s)); keeping frozen target",
                throttle_duration_sec=2.0,
            )
            return None

        hole = hd.cam_point_to_base(
            chosen.point_cam,
            self._R_ee_cam,
            self._t_ee_cam,
            base_ee_pos,
            base_ee_quat,
        )
        self._last_vision_t = time.monotonic()
        self._last_detected_uv = (float(chosen.u), float(chosen.v))
        self.get_logger().info(
            f"live hole update: uv=({chosen.u:.0f},{chosen.v:.0f}) pos={_fmt_xyz(hole)}",
            throttle_duration_sec=1.0,
        )
        return hole

    def _project_hole_pixel_now(self, hole_base: np.ndarray) -> tuple[float, float, float] | None:
        if self.intrinsics is None or self._R_ee_cam is None or self._t_ee_cam is None:
            return None
        target_frame = self.get_parameter("base_frame").value
        try:
            if not self._tf_buffer.can_transform(target_frame, self._flange_link, RclpyTime()):
                return None
            tf = self._tf_buffer.lookup_transform(target_frame, self._flange_link, RclpyTime())
        except TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        p_cam = hd.base_point_to_cam(
            hole_base,
            self._R_ee_cam,
            self._t_ee_cam,
            (t.x, t.y, t.z),
            (q.x, q.y, q.z, q.w),
        )
        if p_cam[2] <= 1e-3:
            return None
        u, v = hd.project_point_cam(p_cam, self.intrinsics)
        return float(u), float(v), float(p_cam[2])

    def _maybe_update_live_target(self) -> bool:
        mode = str(self.get_parameter("target_mode").value).strip().lower()
        if mode == "frozen":
            return True
        now = time.monotonic()
        hz = max(0.1, float(self.get_parameter("vision_update_hz").value))
        if now - self._last_vision_attempt_t < 1.0 / hz:
            return True
        new_hole = self._capture_live_hole(wait=False)
        if new_hole is not None:
            alpha = float(np.clip(float(self.get_parameter("live_target_alpha").value), 0.0, 1.0))
            self._hole_base = new_hole if self._hole_base is None else (alpha * new_hole + (1.0 - alpha) * self._hole_base)
            return True
        if mode == "live":
            live_age = now - self._last_vision_t if self._last_vision_t > 0.0 else float("inf")
            if live_age > float(self.get_parameter("live_timeout_s").value):
                self.get_logger().error(f"live target stale for {live_age:.2f}s; aborting visual servo.")
                return False
        return True

    # ---- servo loop -------------------------------------------------------------------------
    def _force_norm(self) -> float | None:
        if self.ft_force.value is None:
            return None
        force = np.asarray(self.ft_force.value, dtype=np.float64).reshape(3)
        if self._ft_baseline is not None:
            force = force - self._ft_baseline
        return float(np.linalg.norm(force))

    def _limit_command(self, q_measured: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        max_step = float(self.get_parameter("max_joint_step_rad").value)
        q_cmd = q_measured + np.clip(q_target - q_measured, -max_step, max_step)
        margin = float(self.get_parameter("joint_limit_margin_rad").value)
        return np.clip(q_cmd, IIWA7_LOWER + margin, IIWA7_UPPER - margin)

    def _publish_q(self, q: np.ndarray) -> None:
        msg = LBRJointPositionCommand()
        msg.joint_position = [float(x) for x in np.asarray(q, dtype=np.float64).reshape(7)]
        self.command_pub.publish(msg)

    def _maybe_stream(self) -> None:
        if not self._stream_enabled or self._last_q_cmd is None:
            return
        now = time.monotonic()
        if now - self._last_stream_t < 1.0 / max(1.0, float(self.get_parameter("stream_hz").value)):
            return
        self._last_stream_t = now
        self._publish_q(self._last_q_cmd)

    def _debug_servo_print(
        self,
        *,
        now: float,
        action: str,
        hole: np.ndarray,
        xy_err: float,
        z_remaining: float,
        force_norm: float | None,
        cart_step: np.ndarray,
        dq_cmd: np.ndarray,
        q_cmd: np.ndarray,
    ) -> None:
        if not bool(self.get_parameter("debug_print").value):
            return
        hz = max(0.1, float(self.get_parameter("debug_print_hz").value))
        if now - self._last_debug_t < 1.0 / hz:
            return
        self._last_debug_t = now

        px = self._project_hole_pixel_now(hole)
        if px is None or self.intrinsics is None:
            px_txt = "px=n/a"
        else:
            u, v, z_cam = px
            du = u - float(self.intrinsics.cx)
            dv = v - float(self.intrinsics.cy)
            norm = math.hypot(du, dv)
            px_txt = (
                f"uv=({u:.1f},{v:.1f}) px_err=({du:+.1f},{dv:+.1f}) |px|={norm:.1f} "
                f"z_cam={z_cam:.3f}m"
            )
            if self._last_detected_uv is not None:
                du_det = float(self._last_detected_uv[0]) - float(self.intrinsics.cx)
                dv_det = float(self._last_detected_uv[1]) - float(self.intrinsics.cy)
                px_txt += f" det_px=({du_det:+.1f},{dv_det:+.1f})"

        ftxt = "n/a" if force_norm is None else f"{force_norm:.2f}N"
        cart_mm = np.asarray(cart_step, dtype=np.float64).reshape(3) * 1000.0
        dq_mrad = np.asarray(dq_cmd, dtype=np.float64).reshape(7) * 1000.0
        msg = (
            f"servo_dbg {action}: {px_txt} xy={xy_err*1000:.1f}mm "
            f"z_rem={z_remaining*1000:.1f}mm force={ftxt} "
            f"cart_cmd_mm=[{cart_mm[0]:+.2f},{cart_mm[1]:+.2f},{cart_mm[2]:+.2f}] "
            f"dq_cmd_mrad=[{','.join(f'{x:+.2f}' for x in dq_mrad)}]"
        )
        if bool(self.get_parameter("debug_print_q_cmd").value):
            q = np.asarray(q_cmd, dtype=np.float64).reshape(7)
            msg += f" q_cmd=[{','.join(f'{x:+.4f}' for x in q)}]"
        self.get_logger().info(msg)

    def _servo_step(self, execute: bool) -> ServoResult | None:
        timeout = float(self.get_parameter("input_timeout_s").value)
        if self._manual_estop:
            return ServoResult(False, "estop", "manual e-stop asserted", time.monotonic() - self._started_t)
        if not self.joint_pos.fresh(timeout):
            self._status("visual servo waiting for fresh joint state")
            return None
        if bool(self.get_parameter("require_ft").value) and not self.ft_force.fresh(timeout):
            return ServoResult(False, "stale_ft", "F/T wrench went stale", time.monotonic() - self._started_t)
        if self._hole_base is None:
            return ServoResult(False, "no_hole", "no visual hole target available", time.monotonic() - self._started_t)
        if not self._maybe_update_live_target():
            return ServoResult(False, "stale_live_target", "live visual target went stale", time.monotonic() - self._started_t)
        assert self._kinematics is not None

        now = time.monotonic()
        dt = now - self._last_control_t if self._last_control_t > 0.0 else 1.0 / float(self.get_parameter("control_hz").value)
        self._last_control_t = now

        q = np.asarray(self.joint_pos.value, dtype=np.float64).reshape(7)
        pos, R = self._kinematics.fk(q)
        quat = quat_wxyz_from_rotmat(R)
        jac = self._kinematics.jacobian(q)
        hole = np.asarray(self._hole_base, dtype=np.float64).reshape(3)
        final_z = float(hole[2] - float(self.get_parameter("insertion_depth_m").value))
        xy_err_vec = hole[:2] - pos[:2]
        xy_err = float(np.linalg.norm(xy_err_vec))
        z_remaining = float(pos[2] - final_z)
        force_norm = self._force_norm()

        force_cap = float(self.get_parameter("ft_force_cap_n").value)
        if force_norm is not None and force_cap > 0.0 and force_norm >= force_cap:
            return ServoResult(
                False,
                "force_cap",
                f"force cap exceeded: {force_norm:.2f} N",
                now - self._started_t,
            )

        seat_force = float(self.get_parameter("seat_force_n").value)
        if force_norm is not None and seat_force > 0.0 and force_norm >= seat_force:
            near_bottom = pos[2] <= final_z + float(self.get_parameter("seat_force_z_window_m").value)
            if near_bottom and xy_err <= float(self.get_parameter("seat_xy_tolerance_m").value):
                return ServoResult(True, "seated_force", f"seat force reached: {force_norm:.2f} N", now - self._started_t)
            return ServoResult(
                False,
                "early_contact",
                f"seat force {force_norm:.2f} N before bottom/alignment (z_remaining={z_remaining:.4f}, xy={xy_err:.4f})",
                now - self._started_t,
            )

        if xy_err > float(self.get_parameter("abort_xy_error_m").value):
            return ServoResult(False, "xy_abort", f"lateral error too large: {xy_err:.4f} m", now - self._started_t)

        overtravel = float(self.get_parameter("trial_max_overtravel_m").value)
        if overtravel > 0.0 and pos[2] < final_z - overtravel:
            return ServoResult(False, "overtravel", f"z passed socket bottom by {-z_remaining:.4f} m", now - self._started_t)

        z_tol = float(self.get_parameter("seat_z_tolerance_m").value)
        xy_tol = float(self.get_parameter("seat_xy_tolerance_m").value)
        if z_remaining <= z_tol and xy_err <= xy_tol:
            return ServoResult(True, "seated_geometry", f"bottom reached: z_remaining={z_remaining:.4f}, xy={xy_err:.4f}", now - self._started_t)

        timeout_s = float(self.get_parameter("trial_timeout_s").value)
        if execute and timeout_s > 0.0 and now - self._started_t > timeout_s:
            return ServoResult(False, "timeout", f"trial exceeded {timeout_s:.1f}s", now - self._started_t)

        xy_step = float(self.get_parameter("xy_gain").value) * xy_err_vec
        xy_step_norm = float(np.linalg.norm(xy_step))
        max_xy_step = float(self.get_parameter("max_xy_step_m").value)
        if xy_step_norm > max_xy_step > 0.0:
            xy_step *= max_xy_step / xy_step_norm

        dz = 0.0
        if xy_err <= float(self.get_parameter("descent_xy_gate_m").value):
            dz = -min(max(0.0, float(self.get_parameter("descent_speed_mps").value) * dt), max(0.0, z_remaining - z_tol))

        action = "descending" if dz < 0.0 else "centering"
        target_pos = pos + np.array([xy_step[0], xy_step[1], dz], dtype=np.float64)
        target_pos[2] = max(target_pos[2], final_z)
        cart_step = target_pos - pos
        delta_pose = get_pose_error(pos, quat, target_pos, quat)
        dq = get_delta_dof_pos(delta_pose, jac, damping=float(self.get_parameter("ik_damping").value))
        q_cmd = self._limit_command(q, q + dq)
        self._last_q_cmd = q_cmd
        self._debug_servo_print(
            now=now,
            action=action,
            hole=hole,
            xy_err=xy_err,
            z_remaining=z_remaining,
            force_norm=force_norm,
            cart_step=cart_step,
            dq_cmd=q_cmd - q,
            q_cmd=q_cmd,
        )

        ftxt = "n/a" if force_norm is None else f"{force_norm:.2f} N"
        suffix = " (dry-run)" if not execute else ""
        self._status(
            f"visual servo {action}{suffix}: xy={xy_err*1000:.1f} mm "
            f"z_remaining={z_remaining*1000:.1f} mm force={ftxt}"
        )
        return None

    def run(self, *, execute: bool, switch_controller: bool) -> ServoResult:
        if not self._prepare_inputs():
            return ServoResult(False, "not_ready", "required inputs unavailable")

        mode = str(self.get_parameter("target_mode").value).strip().lower()
        if mode not in {"frozen", "live", "live-fallback"}:
            return ServoResult(False, "bad_target_mode", f"unknown target_mode {mode!r}")

        if self._hole_base is None:
            self.get_logger().info("no frozen hole target was supplied; trying to capture one from the wrist camera.")
            self._hole_base = self._capture_live_hole(wait=True)
            if self._hole_base is None:
                return ServoResult(False, "no_hole", "could not capture initial visual hole target")
        if mode in {"live", "live-fallback"}:
            self._last_vision_t = time.monotonic()
        self.get_logger().warn(f"visual-servo target opening (base_link): pos={_fmt_xyz(self._hole_base)}")

        if execute:
            self._status("streaming current joint hold before controller handoff", force=True)
            self._stream_enabled = True
            self._stream_for(float(self.get_parameter("pre_switch_hold_s").value))
            if switch_controller and not self.switch_to_command_controller():
                self._stream_enabled = False
                return ServoResult(False, "controller_switch_failed", "could not activate command controller")
            if bool(self.get_parameter("verify_command_controller").value) and not self._command_controller_is_active():
                self._stream_enabled = False
                return ServoResult(
                    False,
                    "controller_inactive",
                    f"{self.get_parameter('activate_controller').value} is not active; commands would not move the robot",
                )
            self._stream_for(float(self.get_parameter("post_switch_hold_s").value))
            self.get_logger().warn(warn_banner(self._arm, str(self.get_parameter("activate_controller").value)))
        else:
            self.get_logger().info("dry-run mode: computing visual-servo setpoints but NOT publishing motion commands.")

        self._started_t = time.monotonic()
        duration = float(self.get_parameter("trial_timeout_s").value if execute else self.get_parameter("dry_run_s").value)
        self.seat_pub.publish(Bool(data=False))
        result: ServoResult | None = None
        while rclpy.ok():
            now = time.monotonic()
            if not execute and now - self._started_t >= duration:
                result = ServoResult(True, "dry_run_complete", "dry-run finished without commanding motion", now - self._started_t)
                break
            if now - self._last_control_t >= 1.0 / max(1.0, float(self.get_parameter("control_hz").value)):
                result = self._servo_step(execute=execute)
                if result is not None:
                    break
            if execute:
                self._maybe_stream()
            self._pump(0.005)

        if result is None:
            result = ServoResult(False, "interrupted", "ROS shutdown/interrupted", time.monotonic() - self._started_t)
        if result.success and result.outcome.startswith("seated"):
            self.seat_pub.publish(Bool(data=True))
        if execute and self._last_q_cmd is not None:
            # Keep publishing the final hold briefly so the command controller does not see a gap at stop.
            self._stream_for(0.5)
            self._stream_enabled = False
        self._status(result.summary(), force=True)
        return result

    def _stream_for(self, duration_s: float) -> None:
        end = time.monotonic() + max(0.0, float(duration_s))
        while rclpy.ok() and time.monotonic() < end:
            self._maybe_stream()
            self._pump(0.005)


def _confirm(prompt: str, expected: str) -> bool:
    if not sys.stdin.isatty():
        print(f"Refusing to execute non-interactively without --yes. Expected confirmation: {expected}", file=sys.stderr)
        return False
    try:
        return input(prompt).strip() == expected
    except (EOFError, KeyboardInterrupt):
        return False


def _run_corrected_preinsert(node: HoleAlignPlanner, *, execute: bool) -> tuple[PlanResult, np.ndarray | None]:
    """Run hole-align using one captured hole, and return that same hole for the servo phase."""
    if not moveit_available():
        return PlanResult(False, 0, "NO_MOVEIT", message="moveit_msgs not importable"), None
    if not hd.opencv_available():
        return PlanResult(False, 0, "NO_OPENCV", message="cv2 not importable"), None
    if not node.commander.wait_for_server(timeout_s=10.0):
        return PlanResult(False, 0, "NO_MOVE_GROUP", message="move_group action server not found"), None

    hole = node._capture_hole_base()  # reuse the same detector/debug path as hole_align_planner
    if hole is None:
        return PlanResult(False, 0, "NO_HOLE", message="no socket hole detected; see debug_dir"), None
    node.get_logger().warn(f"captured corrected hole for visual servo (base_link): pos={_fmt_xyz(hole)}")

    mode = str(node.get_parameter("orientation_mode").value).strip().lower()
    try:
        _tcp_pos, tcp_quat = node.commander.get_link_pose(timeout_s=float(node.get_parameter("tf_timeout_s").value))
    except LookupError as exc:
        if mode in {"current_tcp", "down"}:
            return PlanResult(False, 0, "NO_TCP_TF", message=f"current TCP pose unavailable: {exc}"), hole
        tcp_quat = None
    try:
        target_pos, target_quat = compute_preinsert_target(
            (float(hole[0]), float(hole[1]), float(hole[2])),
            tcp_quat,
            hover_z_m=float(node.get_parameter("hover_z_m").value),
            orientation_mode=mode,
            fixed_quat_xyzw=tuple(node.get_parameter("fixed_orientation_xyzw").value),
        )
    except ValueError as exc:
        return PlanResult(False, 0, "BAD_TARGET", message=str(exc)), hole

    node.get_logger().info(
        f"visual-servo preinsert target (base_link): pos={_fmt_xyz(target_pos)} "
        f"quat_xyzw={_fmt_quat(target_quat)}"
    )
    plan = node._plan_and_maybe_execute(target_pos, target_quat, execute=execute)
    return plan, hole


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected hole-align preinsert followed by a visual-servo insertion baseline."
    )
    parser.add_argument("--namespace", default="lbr_dual_arm_y_gripper", help="Robot namespace.")
    parser.add_argument("--arm", choices=["right", "left"], default=None, help="Which arm to move.")
    parser.add_argument("--execute", action="store_true", help="Move the robot. Default is dry-run.")
    parser.add_argument("--yes", action="store_true", help="Skip typed confirmations.")
    parser.add_argument(
        "--skip-hole-align",
        action="store_true",
        help="Assume the robot is already at corrected preinsert and only run the visual-servo phase.",
    )
    parser.add_argument(
        "--hole-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Use a fixed base_link socket-opening position for the servo phase.",
    )
    parser.add_argument(
        "--no-controller-switch",
        action="store_true",
        help="Do not switch JTC -> lbr_joint_position_command_controller before servoing.",
    )
    return parser.parse_args(argv)


def main(args: list[str] | None = None) -> None:
    raw_args = sys.argv if args is None else args
    cli = _parse_args(remove_ros_args(args=raw_args)[1:])
    namespace = cli.namespace if cli.namespace.startswith("/") else f"/{cli.namespace}"

    if cli.execute and not cli.yes and not sys.stdin.isatty():
        print("Refusing to execute non-interactively without --yes.", file=sys.stderr)
        return

    rclpy.init(args=args)
    hole_base = np.asarray(cli.hole_xyz, dtype=np.float64).reshape(3) if cli.hole_xyz is not None else None
    try:
        if not cli.skip_hole_align:
            align = HoleAlignPlanner(namespace=namespace, background=False, arm_override=cli.arm)
            try:
                if cli.execute and not cli.yes:
                    dry, _dry_hole = _run_corrected_preinsert(align, execute=False)
                    if not dry.success:
                        align.get_logger().error(f"preinsert dry-run failed: {dry.summary()}")
                        return
                    if not _confirm("Type MOVE to execute corrected preinsert: ", "MOVE"):
                        align.get_logger().warn("corrected preinsert not confirmed; nothing moved.")
                        return
                plan, captured_hole = _run_corrected_preinsert(align, execute=cli.execute)
                if captured_hole is not None and cli.hole_xyz is None:
                    hole_base = captured_hole
                if not plan.success:
                    align.get_logger().error(f"corrected preinsert failed: {plan.summary()}")
                    return
                if cli.execute and not plan.moved:
                    align.get_logger().error(f"corrected preinsert did not execute: {plan.summary()}")
                    return
                if not cli.execute:
                    align.get_logger().info("dry-run complete: corrected preinsert planned; visual servo not started.")
                    return
            finally:
                align.destroy_node()

        servo = VisualServoInsert(namespace=namespace, arm_override=cli.arm, hole_base=hole_base)
        try:
            if cli.execute and not cli.yes:
                action = (
                    "verify controller and start visual-servo descent"
                    if cli.no_controller_switch
                    else "switch controller and start visual-servo descent"
                )
                if not _confirm(f"Type SERVO to {action}: ", "SERVO"):
                    servo.get_logger().warn("visual-servo descent not confirmed; holding at preinsert.")
                    return
            servo.run(execute=cli.execute, switch_controller=(cli.execute and not cli.no_controller_switch))
        finally:
            servo.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
