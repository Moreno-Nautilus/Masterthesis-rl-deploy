"""Vision-corrected MoveIt preinsert: hover the gripper exactly above the DETECTED socket hole.

This is the "clean preinsert" the deploy was missing. The plain ``preinsert_planner`` hovers above the
*perception* socket pose (tracked ``cooling_base`` CAD centre), which is ~1 cm off -- enough to defeat
the 1 mm-clearance insertion. This node instead:

  1. Grabs one time-synced wrist D405 RGB-D frame (+ CameraInfo intrinsics).
  2. Localizes the socket OPENING with a Hough-circle detector sized by the KNOWN 14 mm hole
     (``hole_detector``), sampling the opening depth from the aligned depth image.
  3. Deprojects the detected circle to a metric 3D point and transforms it into ``base_link`` using the
     live flange pose (TF) composed with the calibrated camera-to-flange extrinsics.
  4. Re-aims the MoveIt preinsert hover at that DETECTED hole (not the estimated one) and plans it with
     the same ``MotionCommander`` + branch-safety the preinsert planner uses.

Perception's socket pose is still used (when available) only to DISAMBIGUATE the base's two sockets and
to reject spurious circles -- never as the final target. It also prints the estimate-vs-detected delta.

DEBUG-FIRST / DRY-RUN: debug is ON and motion is OFF by default. Every run writes annotated images
(all circles, the chosen one, the projected perception estimate) + a depth colormap to ``debug_dir``
and logs the full geometry, so you can inspect a dry run before anything moves. ``--execute`` (with an
interactive ``MOVE`` confirmation, or the ``allow_execute`` gate on the service) is required to move.

SAFETY: remaster A1 if the pendant says unmastered; keep the physical e-stop in hand; run dry first.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Trigger

import message_filters
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from fp_debug_msgs.msg import DebugPoseItem

from rcl_interfaces.srv import GetParameters

from .motion_commander import MotionCommander, MotionCommanderConfig, PlanResult, moveit_available
from .preinsert_planner import (
    ARM_DEFAULTS,
    ARM_JOINTS,
    IIWA7_LOWER,
    IIWA7_UPPER,
    compute_preinsert_target,
    warn_banner,
    _fmt_xyz,
    _fmt_quat,
    _fmt_rad_deg,
    _wxyz_from_xyzw,
)
from .ik import get_delta_dof_pos, get_pose_error, quat_wxyz_from_rotmat
from .kinematics import KdlConfig, KdlKinematics, KinematicsUnavailable
from . import hole_detector as hd


# arm -> (wrist RealSense id in the extrinsics map, flange link the extrinsics are relative to).
ARM_CAMERA = {
    "right": ("realsense_2", "lbr_two_link_ee"),
    "left": ("realsense_1", "lbr_one_link_ee"),
}

_DEFAULT_EXTRINSICS = os.path.expanduser("~/Masterthesis-vision/config/camera_extrinsics_realsense.yaml")


def _decode_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        ch = 3
    elif enc in ("rgba8", "bgra8"):
        ch = 4
    else:
        raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = row[:, : msg.width * ch].reshape(msg.height, msg.width, ch)[:, :, :3]
    if enc.startswith("bgr"):
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _decode_depth_m(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        row = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
        return row[:, : msg.width].astype(np.float32) * 0.001
    if enc == "32fc1":
        row = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        return row[:, : msg.width].astype(np.float32)
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


@dataclass
class _Frame:
    rgb: np.ndarray
    depth: np.ndarray
    wall_t: float


@dataclass
class _Socket:
    pos: np.ndarray          # perceived CAD centre in base_link (3,)
    quat_xyzw: np.ndarray    # perceived orientation (4,)
    wall_t: float


class HoleAlignPlanner(Node):
    def __init__(self, *, namespace: str, background: bool, arm_override: str | None = None) -> None:
        super().__init__("hole_align_planner", namespace=namespace)
        self._background = background
        self._declare_parameters()

        arm = (arm_override or self.get_parameter("arm").value or "right").strip().lower()
        if arm not in ARM_DEFAULTS:
            raise ValueError(f"arm must be one of {sorted(ARM_DEFAULTS)}, got {arm!r}")
        self._arm = arm
        default_group, default_tip = ARM_DEFAULTS[arm]
        default_cam, default_flange = ARM_CAMERA[arm]
        group_name = self.get_parameter("group_name").value or default_group
        tip_link = self.get_parameter("tip_link").value or default_tip
        self._cam_id = self.get_parameter("camera_id").value or default_cam
        self._flange_link = self.get_parameter("flange_link").value or default_flange

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

        # Load the calibrated camera-to-flange extrinsics once (static mount transform).
        self._R_ee_cam = None
        self._t_ee_cam = None
        try:
            self._R_ee_cam, self._t_ee_cam = hd.load_cam_flange_extrinsics(
                self.get_parameter("extrinsics_yaml").value, self._cam_id
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"could not load extrinsics for {self._cam_id} from "
                f"{self.get_parameter('extrinsics_yaml').value!r}: {exc}"
            )

        self._frame: _Frame | None = None
        self._intr: hd.CameraIntrinsics | None = None
        self._socket: _Socket | None = None

        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("rgb_topic").value, qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=qos_profile_sensor_data
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=int(self.get_parameter("img_sync_queue").value),
            slop=float(self.get_parameter("img_sync_slop_s").value),
        )
        self._sync.registerCallback(self._on_rgbd)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            DebugPoseItem, self.get_parameter("socket_pose_topic").value, self._on_socket,
            qos_profile_sensor_data,
        )

        srv_group = ReentrantCallbackGroup()
        self.create_service(Trigger, "~/plan_hole_align", self._srv_plan, callback_group=srv_group)
        self.create_service(Trigger, "~/move_hole_align", self._srv_move, callback_group=srv_group)

        self.get_logger().info(
            f"hole_align_planner ready (namespace={self.get_namespace()}, arm={self._arm}, "
            f"group={cfg.group_name}, tip={cfg.tip_link}, cam={self._cam_id}, flange={self._flange_link}, "
            f"moveit={'yes' if moveit_available() else 'NO'}, opencv={'yes' if hd.opencv_available() else 'NO'})."
        )
        if not hd.opencv_available():
            self.get_logger().error("OpenCV (cv2) not importable -- hole detection will fail.")

    # ---- parameters ---------------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("arm", "right")
        self.declare_parameter("group_name", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tip_link", "")
        self.declare_parameter("move_group_action", "move_action")
        self.declare_parameter("joint_states_topic", "joint_states")

        # ---- camera ----
        self.declare_parameter("rgb_topic", "/realsense_2/camera/color/image_rect")
        self.declare_parameter("depth_topic", "/realsense_2/camera/aligned_depth_to_color/image_rect")
        self.declare_parameter("camera_info_topic", "/realsense_2/camera/color/camera_info")
        self.declare_parameter("camera_id", "")            # "" -> derive from arm (realsense_2/1)
        self.declare_parameter("flange_link", "")          # "" -> derive from arm (lbr_two/one_link_ee)
        self.declare_parameter("extrinsics_yaml", _DEFAULT_EXTRINSICS)
        self.declare_parameter("img_sync_slop_s", 0.05)
        self.declare_parameter("img_sync_queue", 30)
        self.declare_parameter("frame_timeout_s", 5.0)

        # ---- hole geometry / detector (see hole_detector.HoleDetectorConfig) ----
        self.declare_parameter("method", "contour")        # "contour" (robust) or "hough" (legacy)
        # 14 mm = the socket OPENING diameter (CAD cooling_base.obj inner rim + sim asset_size).
        self.declare_parameter("hole_diameter_m", 0.014)
        self.declare_parameter("radius_tol_frac", 0.45)
        # CAD cross-check: the two sockets are 60 mm apart (x = +/-30 mm). Pair them geometrically.
        self.declare_parameter("socket_spacing_m", 0.060)
        self.declare_parameter("socket_spacing_tol_m", 0.012)
        self.declare_parameter("use_socket_pair", True)
        # Part segmentation (contour method): keep only the vividly-coloured part.
        self.declare_parameter("use_saturation_mask", True)
        self.declare_parameter("sat_min", 60)
        self.declare_parameter("val_min", 40)
        # Dark-blob shape filters (contour method): reject the elongated cooling fins.
        self.declare_parameter("adaptive_block", 51)
        self.declare_parameter("adaptive_C", 5)
        self.declare_parameter("morph_open", True)
        self.declare_parameter("min_circularity", 0.65)
        self.declare_parameter("max_aspect", 1.8)
        self.declare_parameter("min_solidity", 0.80)
        self.declare_parameter("min_fill", 0.55)
        # Reject edge/mounting through-holes: the socket must be embedded in the coloured part.
        self.declare_parameter("require_part_surround", True)
        self.declare_parameter("surround_part_frac", 0.85)
        # Hough (legacy method) knobs.
        self.declare_parameter("hough_dp", 1.2)
        self.declare_parameter("hough_param1", 120.0)
        self.declare_parameter("hough_param2", 18.0)
        self.declare_parameter("blur_ksize", 5)
        self.declare_parameter("detect_on", "rgb")         # "rgb" or "depth"
        self.declare_parameter("depth_valid_min_m", 0.05)
        self.declare_parameter("depth_valid_max_m", 0.60)
        # Gate: chosen circle must be within this many px of the projected perception estimate (when a
        # socket pose is available). Keeps the two sockets / stray circles apart. <=0 disables the gate.
        self.declare_parameter("max_center_dist_px", 60.0)

        # ---- perception socket (selection + comparison only; NEVER the final target) ----
        self.declare_parameter("socket_pose_topic", "/perception/fp/pose_base/fused/assembly")
        self.declare_parameter("socket_assembly_name", "cooling_base")
        self.declare_parameter("socket_part_id", -1)
        self.declare_parameter("socket_timeout_s", 3.0)
        # Which socket opening on the base to aim at, in the perceived CAD frame (rotated by the
        # perceived orientation). RIGHT socket default; flip x sign for the LEFT socket.
        self.declare_parameter("socket_opening_offset_cad_xyz", [0.030, 0.0, 0.0175])
        # If detection fails, fall back to the perception opening instead of aborting (default: abort,
        # because the whole point here is the vision correction). Turn on only knowingly.
        self.declare_parameter("fallback_to_perception", False)

        # ---- preinsert target (same contract as preinsert_planner) ----
        self.declare_parameter("hover_z_m", 0.06)
        self.declare_parameter("orientation_mode", "down")
        self.declare_parameter("fixed_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])

        # ---- planning tolerances / conservatism ----
        self.declare_parameter("position_tolerance_m", 0.005)
        self.declare_parameter("orientation_tolerance_rad", 0.1)
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("acceleration_scaling", 0.05)
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("num_planning_attempts", 10)
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planner_id", "")
        self.declare_parameter("joint_near_current_tolerance_rad", 0.80)
        self.declare_parameter("max_plan_joint_delta_rad", 0.85)
        # Keep the plan near the current joint branch (safety vs. wild elbow/wrist flips). Turn OFF when
        # the arm must make a LARGE gross move to the socket (e.g. localizing with the OTHER arm's
        # camera) -- the near-branch guard makes such a move unplannable. Off => OMPL may pick any branch.
        self.declare_parameter("constrain_branch", True)
        self.declare_parameter("tf_timeout_s", 5.0)
        self.declare_parameter("warmup_plan", True)
        # Bounded warmup so a genuine planning FAILURE doesn't spin for the full 30 s before the real
        # plan + diagnosis runs. Only meant to absorb move_group's dirty-startup window.
        self.declare_parameter("warmup_timeout_s", 8.0)
        # "local_ik_joint" (DEFAULT): compute a nearby joint target with KDL/DLS from the CURRENT branch,
        # then plan a joint-space path -- keeps a small Cartesian correction a small IN-BRANCH move (a
        # plain pose_goal lets OMPL branch-flip: a ~1 cm correction became a 70 deg swing). "pose_goal":
        # normal MoveIt Cartesian pose goal (subject to branch flips; use only if KDL is unavailable).
        self.declare_parameter("target_mode", "local_ik_joint")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robot_description_service", "robot_state_publisher/get_parameters")
        self.declare_parameter("local_ik_max_iters", 120)
        self.declare_parameter("local_ik_damping", 0.1)
        self.declare_parameter("local_ik_joint_step_rad", 0.04)
        self.declare_parameter("local_ik_cart_step_m", 0.025)
        self.declare_parameter("local_ik_rot_step_rad", 0.08)
        self.declare_parameter("local_ik_max_total_delta_rad", 1.20)
        self.declare_parameter("local_ik_joint_goal_tolerance_rad", 0.01)

        # ---- corrected-anchor republisher (--republish-anchor) ----
        # Capture the detected hole ONCE, then continuously publish it as a DebugPoseItem so the RL node
        # can use the VISION-corrected opening as its socket anchor (point its socket_pose_topic here and
        # set socket_opening_offset_cad_xyz:=[0,0,0]). The hole is static in base_link, so a one-time
        # capture republished at a steady rate is correct through the whole trial.
        self.declare_parameter("anchor_topic", "/rl_deploy/corrected_socket")
        self.declare_parameter("anchor_assembly_name", "cooling_base")
        self.declare_parameter("anchor_part_id", 0)
        self.declare_parameter("anchor_publish_hz", 20.0)
        # Fixed-anchor bypass (use when the centered screw occludes re-detection): publish this base_link
        # position directly instead of detecting. Feed the corrected preinsert's "DETECTED hole" value.
        self.declare_parameter("anchor_use_fixed", False)
        self.declare_parameter("anchor_xyz", [0.0, 0.0, 0.0])

        # ---- debug ----
        self.declare_parameter("debug", True)
        self.declare_parameter("debug_dir", "/tmp/hole_align")

        # ---- execute gate (service path only) ----
        self.declare_parameter("allow_execute", False)

    # ---- callbacks ----------------------------------------------------------------------------
    def _on_rgbd(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            rgb = _decode_rgb(rgb_msg)
            depth = _decode_depth_m(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"RGB-D decode failed: {exc}", throttle_duration_sec=5.0)
            return
        self._frame = _Frame(rgb, depth, time.monotonic())

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._intr = hd.CameraIntrinsics.from_k(msg.k)

    def _on_socket(self, msg: DebugPoseItem) -> None:
        want_assembly = self.get_parameter("socket_assembly_name").value
        want_part = int(self.get_parameter("socket_part_id").value)
        if want_assembly and msg.assembly_name != want_assembly:
            return
        if want_part >= 0 and int(msg.part_id) != want_part:
            return
        p = msg.pose_base.pose.position
        q = msg.pose_base.pose.orientation
        self._socket = _Socket(
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
            time.monotonic(),
        )

    def _pump(self, timeout_s: float) -> None:
        if self._background:
            time.sleep(min(timeout_s, 0.05))
        else:
            rclpy.spin_once(self, timeout_sec=timeout_s)

    def _wait_for(self, predicate, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            self._pump(0.1)
        return predicate()

    # ---- detector config ----------------------------------------------------------------------
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

    def _perceived_opening_base(self, socket: _Socket) -> np.ndarray:
        """Perceived socket OPENING in base = centre + R(perceived_quat) @ CAD-frame offset."""
        offset = np.asarray(
            self.get_parameter("socket_opening_offset_cad_xyz").value, dtype=np.float64
        ).reshape(3)
        return socket.pos + hd.rotmat_from_quat_xyzw(socket.quat_xyzw) @ offset

    # ---- core flow ----------------------------------------------------------------------------
    def run(self, *, execute: bool) -> PlanResult:
        if not moveit_available():
            return PlanResult(False, 0, "NO_MOVEIT", message="moveit_msgs not importable")
        if not hd.opencv_available():
            return PlanResult(False, 0, "NO_OPENCV", message="cv2 not importable")
        if self._R_ee_cam is None:
            return PlanResult(False, 0, "NO_EXTRINSICS", message="camera-to-flange extrinsics not loaded")
        if not self.commander.wait_for_server(timeout_s=10.0):
            return PlanResult(False, 0, "NO_MOVE_GROUP", message="move_group action server not found")

        ftimeout = float(self.get_parameter("frame_timeout_s").value)
        if not self._wait_for(lambda: self._frame is not None and self._intr is not None, ftimeout):
            return PlanResult(False, 0, "NO_FRAME", message="no synced RGB-D + CameraInfo received")
        frame, intr = self._frame, self._intr
        assert frame is not None and intr is not None

        # Flange pose in base (TF) -> compose camera-to-base with the calibrated extrinsics.
        try:
            base_ee_pos, base_ee_quat = self.commander.get_link_pose(
                source_link=self._flange_link, target_frame=self.get_parameter("base_frame").value,
                timeout_s=float(self.get_parameter("tf_timeout_s").value),
            )
        except LookupError as exc:
            return PlanResult(False, 0, "NO_FLANGE_TF", message=f"flange TF unavailable: {exc}")

        # Optional perception socket -> project the estimated opening into the image (selection + compare).
        socket = self._socket
        if socket is not None and (time.monotonic() - socket.wall_t) > float(self.get_parameter("socket_timeout_s").value):
            socket = None
        expected_opening_base = None
        expected_uv = None
        if socket is not None:
            expected_opening_base = self._perceived_opening_base(socket)
            p_cam = hd.base_point_to_cam(
                expected_opening_base, self._R_ee_cam, self._t_ee_cam, base_ee_pos, base_ee_quat
            )
            if p_cam[2] > 1e-3:
                expected_uv = hd.project_point_cam(p_cam, intr)

        # Detect + select.
        dcfg = self._detector_config()
        ref_depth = hd.reference_depth(frame.depth, dcfg)
        detections = hd.detect_holes(frame.rgb, frame.depth, intr, dcfg, ref_depth_m=ref_depth)
        gate = float(self.get_parameter("max_center_dist_px").value)
        chosen = hd.select_hole(
            detections, intr, expected_uv=expected_uv,
            max_center_dist_px=(gate if gate > 0 else None),
        )

        self._log_and_dump(frame, intr, detections, chosen, expected_uv, ref_depth)

        if chosen is None:
            if self.get_parameter("fallback_to_perception").value and expected_opening_base is not None:
                self.get_logger().warn(
                    "NO hole detected -> falling back to the PERCEPTION opening (fallback_to_perception=true)."
                )
                hole_base = expected_opening_base
            else:
                return PlanResult(
                    False, 0, "NO_HOLE",
                    message=f"no socket hole detected ({len(detections)} raw circle(s); see debug_dir)",
                )
        else:
            hole_base = hd.cam_point_to_base(
                chosen.point_cam, self._R_ee_cam, self._t_ee_cam, base_ee_pos, base_ee_quat
            )

        # Report the correction vs the perception estimate.
        self.get_logger().info(f"DETECTED hole (base_link): pos={_fmt_xyz(hole_base)}")
        # Remember this detection (taken from the clean, pre-descent view) so --republish-anchor can
        # reuse it instead of a SECOND CV detection at the 4.5 cm hover, where the centered screw
        # occludes/shifts the hole and gives a several-cm-off anchor.
        self._executed_hole_base = np.asarray(hole_base, dtype=np.float64).reshape(3)
        if expected_opening_base is not None:
            delta = hole_base - expected_opening_base
            self.get_logger().info(
                f"perception opening (base_link): pos={_fmt_xyz(expected_opening_base)}  "
                f"correction delta={_fmt_xyz(delta)} |xy|={math.hypot(delta[0], delta[1])*1000:.1f} mm "
                f"|z|={abs(delta[2])*1000:.1f} mm"
            )

        # Build the hover target above the DETECTED hole and plan it (same contract as preinsert_planner).
        mode = str(self.get_parameter("orientation_mode").value).strip().lower()
        try:
            tcp_pos, tcp_quat = self.commander.get_link_pose(
                timeout_s=float(self.get_parameter("tf_timeout_s").value)
            )
        except LookupError as exc:
            if mode in ("current_tcp", "down"):
                return PlanResult(False, 0, "NO_TCP_TF", message=f"current TCP pose unavailable: {exc}")
            tcp_quat = None
        try:
            target_pos, target_quat = compute_preinsert_target(
                (float(hole_base[0]), float(hole_base[1]), float(hole_base[2])),
                tcp_quat,
                hover_z_m=float(self.get_parameter("hover_z_m").value),
                orientation_mode=mode,
                fixed_quat_xyzw=tuple(self.get_parameter("fixed_orientation_xyzw").value),
            )
        except ValueError as exc:
            return PlanResult(False, 0, "BAD_TARGET", message=str(exc))

        self.get_logger().info(
            f"preinsert target (base_link): pos={_fmt_xyz(target_pos)} quat_xyzw={_fmt_quat(target_quat)} "
            f"(hover +{float(self.get_parameter('hover_z_m').value):.3f} m z, orientation_mode={mode})"
        )

        return self._plan_and_maybe_execute(target_pos, target_quat, execute=execute)

    # ---- corrected-anchor capture + republisher ----------------------------------------------
    def _capture_hole_base(self) -> np.ndarray | None:
        """Detect the socket hole once and return its position in base_link (or None)."""
        if not hd.opencv_available() or self._R_ee_cam is None:
            self.get_logger().error("cannot capture hole: OpenCV or extrinsics unavailable.")
            return None
        if not self._wait_for(lambda: self._frame is not None and self._intr is not None,
                              float(self.get_parameter("frame_timeout_s").value)):
            self.get_logger().error("cannot capture hole: no synced RGB-D + CameraInfo.")
            return None
        frame, intr = self._frame, self._intr
        try:
            base_ee_pos, base_ee_quat = self.commander.get_link_pose(
                source_link=self._flange_link, target_frame=self.get_parameter("base_frame").value,
                timeout_s=float(self.get_parameter("tf_timeout_s").value),
            )
        except LookupError as exc:
            self.get_logger().error(f"cannot capture hole: flange TF unavailable: {exc}")
            return None
        expected_uv = None
        socket = self._socket
        if socket is not None and (time.monotonic() - socket.wall_t) <= float(self.get_parameter("socket_timeout_s").value):
            p_cam = hd.base_point_to_cam(
                self._perceived_opening_base(socket), self._R_ee_cam, self._t_ee_cam, base_ee_pos, base_ee_quat
            )
            if p_cam[2] > 1e-3:
                expected_uv = hd.project_point_cam(p_cam, intr)
        dcfg = self._detector_config()
        ref_depth = hd.reference_depth(frame.depth, dcfg)
        detections = hd.detect_holes(frame.rgb, frame.depth, intr, dcfg, ref_depth_m=ref_depth)
        gate = float(self.get_parameter("max_center_dist_px").value)
        chosen = hd.select_hole(detections, intr, expected_uv=expected_uv, max_center_dist_px=(gate if gate > 0 else None))
        self._log_and_dump(frame, intr, detections, chosen, expected_uv, ref_depth)
        if chosen is None:
            return None
        return hd.cam_point_to_base(chosen.point_cam, self._R_ee_cam, self._t_ee_cam, base_ee_pos, base_ee_quat)

    def run_republish_anchor(self) -> bool:
        """Capture the hole once, then republish it as a DebugPoseItem for the RL node's socket anchor.

        Blocks (spins) until shutdown. The pose is static in base_link, so the RL node sees a rock-steady
        vision-corrected anchor for the whole trial. Point the RL node's socket_pose_topic here and set
        socket_opening_offset_cad_xyz:=[0,0,0], socket_assembly_name/socket_part_id to match below.
        """
        # Fixed-anchor bypass: once the arm is CENTERED over the hole, the held screw occludes the wrist
        # view, so re-detection fails. Feed the base_link hole the corrected preinsert already logged
        # ("DETECTED hole (base_link): pos=[...]") via anchor_xyz to publish it directly, no detection.
        fixed = [float(v) for v in self.get_parameter("anchor_xyz").value]
        if bool(self.get_parameter("anchor_use_fixed").value) or any(abs(v) > 1e-9 for v in fixed):
            hole = np.asarray(fixed, dtype=np.float64).reshape(3)
            self.get_logger().warn(f"using FIXED anchor from anchor_xyz: pos={_fmt_xyz(hole)} (no detection).")
        else:
            if not self.commander.wait_for_server(timeout_s=10.0):
                self.get_logger().warn("move_group not found; continuing (only TF + camera are needed to capture).")
            hole = self._capture_hole_base()
            if hole is None:
                self.get_logger().error(
                    "anchor capture FAILED (no hole detected -- screw likely occludes the centered view). "
                    "Pass the corrected-preinsert 'DETECTED hole (base_link)' via -p anchor_xyz:=\"[x, y, z]\"."
                )
                return False
        return self._spin_publish_anchor(hole)

    def _spin_publish_anchor(self, hole) -> bool:
        """Publish a fixed base_link hole as the RL socket anchor at anchor_publish_hz until Ctrl-C."""
        topic = self.get_parameter("anchor_topic").value
        assembly = self.get_parameter("anchor_assembly_name").value
        part_id = int(self.get_parameter("anchor_part_id").value)
        base_frame = self.get_parameter("base_frame").value
        pub = self.create_publisher(DebugPoseItem, topic, qos_profile_sensor_data)

        item = DebugPoseItem()
        item.assembly_name = assembly
        item.part_id = part_id
        ps = PoseStamped()
        ps.header.frame_id = base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(hole[0]), float(hole[1]), float(hole[2]))
        ps.pose.orientation.w = 1.0  # identity: policy ignores socket orientation; opening plane = world +z
        item.pose_base = ps

        hz = max(1.0, float(self.get_parameter("anchor_publish_hz").value))

        def _tick():
            ps.header.stamp = self.get_clock().now().to_msg()
            pub.publish(item)

        self.create_timer(1.0 / hz, _tick)
        self.get_logger().warn(
            f"PUBLISHING vision-corrected anchor on {topic} @ {hz:.0f} Hz: "
            f"assembly={assembly} part_id={part_id} pos={_fmt_xyz(hole)} (base_link).\n"
            f"  -> launch the RL node with: socket_pose_topic:={topic}  socket_assembly_name:={assembly}  "
            f"socket_part_id:={part_id}  socket_opening_offset_cad_xyz:=[0.0,0.0,0.0]\n"
            "  Keep this running for the whole trial. Ctrl-C to stop."
        )
        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            pass
        return True

    def _plan_and_maybe_execute(self, target_pos, target_quat, *, execute: bool) -> PlanResult:
        pos_tol = float(self.get_parameter("position_tolerance_m").value)
        ori_tol = float(self.get_parameter("orientation_tolerance_rad").value)
        current_branch = self._current_arm_joint_map()
        constrain = bool(self.get_parameter("constrain_branch").value)
        joint_tol = float(self.get_parameter("joint_near_current_tolerance_rad").value)
        if not current_branch:
            self.get_logger().warn("no current arm joint map; planning without a branch constraint")

        if str(self.get_parameter("target_mode").value).strip().lower() == "local_ik_joint":
            return self._plan_local_ik_and_maybe_execute(current_branch, target_pos, target_quat, execute=execute)
        # Branch constraint is applied only when requested AND we have a current joint map.
        branch = current_branch if (constrain and current_branch) else None
        btol = joint_tol if branch else None
        if current_branch and not constrain:
            self.get_logger().warn("constrain_branch=false: planning a FREE (possibly large) move; verify the plan.")

        if self.get_parameter("warmup_plan").value:
            self.commander.wait_until_plannable(
                target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
                path_joint_targets=branch, path_joint_tolerance_rad=btol,
                timeout_s=float(self.get_parameter("warmup_timeout_s").value),
            )
        plan = self.commander.plan_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
            path_joint_targets=branch, path_joint_tolerance_rad=btol,
        )
        self._log_plan(plan)
        if branch and not self._plan_stays_near_current(plan, current_branch):
            return plan
        if not plan.success:
            self.get_logger().error(f"plan FAILED ({plan.error_name}); not executing.")
            self._diagnose_plan_failure(target_pos, target_quat, pos_tol, ori_tol, branch is not None)
            return plan
        if not execute:
            self.get_logger().info("plan-only mode: NOT executing. Re-run with execute to move.")
            return plan

        self.get_logger().warn(warn_banner(self._arm, self.commander.cfg.group_name))
        moved = self.commander.move_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
            path_joint_targets=branch, path_joint_tolerance_rad=btol,
        )
        self._log_plan(moved, executed=True)
        if not moved.success:
            self.get_logger().error(
                f"execution FAILED ({moved.error_name}). Check joint_trajectory_controller is active "
                "AND has allow_partial_joints_goal: true."
            )
        else:
            self.get_logger().info("hole-aligned preinsert SUCCEEDED; RL policy can take over the final insertion.")
        return moved

    # ---- local-IK joint target (branch-safe small correction) --------------------------------
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
            return KdlKinematics(KdlConfig(
                robot_description=desc,
                base_link=self.get_parameter("base_frame").value,
                tip_link=self.commander.cfg.tip_link,
                joint_names=tuple(ARM_JOINTS[self._arm]),
            ))
        except KinematicsUnavailable as exc:
            self.get_logger().error(f"local_ik_joint: {exc}")
            return None

    def _solve_local_ik(self, current_branch, target_pos, target_quat):
        """DLS servo from the CURRENT joints toward the target pose -> (q_goal dict | None, max_delta)."""
        if not current_branch:
            self.get_logger().error("local_ik_joint: no current arm joint state")
            return None, 0.0
        kin = self._make_kinematics()
        if kin is None:
            return None, 0.0
        gp = self.get_parameter
        names = ARM_JOINTS[self._arm]
        q0 = np.array([current_branch[n] for n in names], dtype=np.float64)
        q = q0.copy()
        target_p = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_q = _wxyz_from_xyzw(target_quat)
        pos_tol = float(gp("position_tolerance_m").value)
        ori_tol = float(gp("orientation_tolerance_rad").value)
        damping = float(gp("local_ik_damping").value)
        max_joint_step = float(gp("local_ik_joint_step_rad").value)
        max_cart_step = float(gp("local_ik_cart_step_m").value)
        max_rot_step = float(gp("local_ik_rot_step_rad").value)
        margin = 0.03
        pos_err = rot_err = float("inf")
        for _ in range(int(gp("local_ik_max_iters").value)):
            pos, R = kin.fk(q)
            err = get_pose_error(pos, quat_wxyz_from_rotmat(R), target_p, target_q)
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err <= pos_tol and rot_err <= ori_tol:
                break
            step = err.copy()
            tn = float(np.linalg.norm(step[:3]))
            if tn > max_cart_step:
                step[:3] *= max_cart_step / tn
            rn = float(np.linalg.norm(step[3:]))
            if rn > max_rot_step:
                step[3:] *= max_rot_step / rn
            dq = np.clip(get_delta_dof_pos(step, kin.jacobian(q), damping=damping), -max_joint_step, max_joint_step)
            q = np.clip(q + dq, IIWA7_LOWER + margin, IIWA7_UPPER - margin)
        delta = float(np.max(np.abs(q - q0)))
        self.get_logger().info(
            f"local IK: pos_err={pos_err:.4f} m rot_err={rot_err:.4f} rad max_delta={_fmt_rad_deg(delta)} q={_fmt_xyz(q)}"
        )
        if pos_err > pos_tol or rot_err > ori_tol:
            self.get_logger().error(f"local IK did NOT converge (pos_err={pos_err:.4f}, rot_err={rot_err:.4f}).")
            return None, delta
        if delta > float(gp("local_ik_max_total_delta_rad").value):
            self.get_logger().error(
                f"local IK max_delta {_fmt_rad_deg(delta)} > local_ik_max_total_delta_rad "
                f"{_fmt_rad_deg(float(gp('local_ik_max_total_delta_rad').value))}; refusing (branch jump)."
            )
            return None, delta
        return {n: float(v) for n, v in zip(names, q)}, delta

    def _plan_local_ik_and_maybe_execute(self, current_branch, target_pos, target_quat, *, execute: bool) -> PlanResult:
        q_goal, _delta = self._solve_local_ik(current_branch, target_pos, target_quat)
        if q_goal is None:
            return PlanResult(False, 0, "LOCAL_IK_FAILED", message="local IK did not yield a safe in-branch joint goal")
        jtol = float(self.get_parameter("local_ik_joint_goal_tolerance_rad").value)
        plan = self.commander.plan_to_joint(q_goal, tolerance_rad=jtol)
        self._log_plan(plan)
        if not plan.success:
            self.get_logger().error(f"joint-space plan FAILED ({plan.error_name}); not executing.")
            return plan
        if not execute:
            self.get_logger().info("plan-only mode: NOT executing. Re-run with execute to move.")
            return plan
        self.get_logger().warn(warn_banner(self._arm, self.commander.cfg.group_name))
        moved = self.commander.move_to_joint(q_goal, tolerance_rad=jtol)
        self._log_plan(moved, executed=True)
        if not moved.success:
            self.get_logger().error(
                f"execution FAILED ({moved.error_name}). Check joint_trajectory_controller is active "
                "AND has allow_partial_joints_goal: true."
            )
        else:
            self.get_logger().info("hole-aligned preinsert SUCCEEDED; RL policy can take over the final insertion.")
        return moved

    def _diagnose_plan_failure(self, target_pos, target_quat, pos_tol, ori_tol, was_constrained: bool) -> None:
        """Explain WHY the plan failed: branch guard (reachable-but-large) vs. genuinely unreachable.

        Runs a plan-only probe with NO branch constraint (safe -- never moves). If that succeeds, the
        target is reachable and the failure was the near-branch guard -> a large gross move is needed.
        If it also fails, the pose itself is the problem (reach / orientation / start-state collision).
        """
        self.get_logger().info("---- plan-failure diagnosis (plan-only, never moves) ----")
        free = self.commander.plan_to_pose(
            target_pos, target_quat, position_tolerance_m=pos_tol, orientation_tolerance_rad=ori_tol,
        )
        if free.success:
            names, _first, last = free.joint_targets()
            current = self._current_arm_joint_map()
            max_delta = 0.0
            if names and current:
                max_delta = max(abs(float(v) - current.get(n, float(v))) for n, v in zip(names, last))
            self.get_logger().warn(
                "target IS reachable with a FREE plan "
                f"(max joint delta {_fmt_rad_deg(max_delta)}). The constrained plan failed because this "
                "is a LARGE move for the current branch."
                if was_constrained else
                "target is reachable; the earlier failure was likely a transient move_group start-state window."
            )
            self.get_logger().warn(
                "This usually means the arm you are MOVING (arm_two/right) is not near the socket -- e.g. "
                "you localized with the OTHER arm's camera. Fix by either (A) moving the RIGHT arm over "
                "the socket first and localizing with realsense_2, or (B) re-running with "
                "-p constrain_branch:=false to allow the large gross move (verify the plan before executing)."
            )
        else:
            self.get_logger().error(
                f"target is NOT reachable even unconstrained ({free.error_name}). Likely causes: pose out "
                "of reach / bad orientation for this arm, start-state in collision (check the "
                "scene_collisions boxes), or the base_link transform is off (verify DETECTED hole sits on "
                "the real socket; check realsense_1/lbr_one_link_ee extrinsics). Try orientation_mode:="
                "current_tcp or a higher hover_z_m."
            )

    def _current_arm_joint_map(self) -> dict[str, float]:
        current = self.commander.current_joint_positions()
        names = ARM_JOINTS[self._arm]
        if not all(name in current for name in names):
            return {}
        return {name: current[name] for name in names}

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

    # ---- debug --------------------------------------------------------------------------------
    def _log_and_dump(self, frame, intr, detections, chosen, expected_uv, ref_depth) -> None:
        self.get_logger().info("---- hole detection report ----")
        self.get_logger().info(
            f"intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}; "
            f"ref_depth={ref_depth*1000:.0f} mm; raw circles={len(detections)}"
        )
        for i, d in enumerate(detections[:8]):
            tag = " <== CHOSEN" if d is chosen else ""
            self.get_logger().info(
                f"  circle[{i}] uv=({d.u:.0f},{d.v:.0f}) r={d.radius_px:.1f}px z={d.depth_m*1000:.0f}mm "
                f"r_err={d.radius_err_frac*100:.0f}% cam={_fmt_xyz(d.point_cam)}{tag}"
            )
        if expected_uv is not None:
            self.get_logger().info(f"perception opening projected to pixel: ({expected_uv[0]:.0f}, {expected_uv[1]:.0f})")
        if chosen is None:
            self.get_logger().warn("no hole selected (empty detections or all outside the center-distance gate).")

        paired = sum(1 for d in detections if d.paired)
        if paired:
            self.get_logger().info(f"CAD 60mm pairing: {paired} candidate(s) flagged as the socket pair.")
        elif self.get_parameter("use_socket_pair").value:
            self.get_logger().warn("CAD 60mm pairing: no matching pair found (only one socket in view, or noisy).")

        if not self.get_parameter("debug").value:
            return
        try:
            import cv2
            dcfg = self._detector_config()
            ddir = self.get_parameter("debug_dir").value
            os.makedirs(ddir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            mask = hd.part_mask(frame.rgb, frame.depth, dcfg) if dcfg.method == "contour" else None
            annotated = hd.render_debug(frame.rgb, detections, chosen, expected_uv=expected_uv, mask=mask)
            cv2.imwrite(os.path.join(ddir, f"holes_{stamp}.png"), annotated)
            cv2.imwrite(os.path.join(ddir, f"depth_{stamp}.png"), hd.depth_colormap(frame.depth, dcfg))
            # RAW frames for offline detector tuning: RGB (as BGR png) + depth in millimetres (16-bit png).
            cv2.imwrite(os.path.join(ddir, f"raw_rgb_{stamp}.png"), cv2.cvtColor(frame.rgb.astype("uint8"), cv2.COLOR_RGB2BGR))
            depth_mm = np.clip(np.nan_to_num(frame.depth) * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(ddir, f"raw_depth_mm_{stamp}.png"), depth_mm)
            with open(os.path.join(ddir, f"intrinsics_{stamp}.txt"), "w", encoding="utf-8") as f:
                f.write(f"{intr.fx} {intr.fy} {intr.cx} {intr.cy}\n")
            self.get_logger().info(
                f"debug written to {ddir}: holes_{stamp}.png depth_{stamp}.png "
                f"raw_rgb_{stamp}.png raw_depth_mm_{stamp}.png intrinsics_{stamp}.txt"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"debug image dump failed: {exc}")

    # ---- services -----------------------------------------------------------------------------
    def _srv_plan(self, _request, response):
        plan = self.run(execute=False)
        response.success = plan.success
        response.message = plan.summary()
        return response

    def _srv_move(self, _request, response):
        if not self.get_parameter("allow_execute").value:
            response.success = False
            response.message = "allow_execute is false; refusing to move. Set it true to enable ~/move_hole_align."
            self.get_logger().error(response.message)
            return response
        plan = self.run(execute=True)
        response.success = plan.success and plan.moved
        response.message = plan.summary()
        return response


def _parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(description="Vision-corrected MoveIt preinsert (hover above the detected hole).")
    parser.add_argument("--namespace", default="lbr_dual_arm_y_gripper", help="Robot namespace.")
    parser.add_argument("--arm", choices=["right", "left"], default=None, help="Which arm (default: right).")
    parser.add_argument("--service", action="store_true", help="Run as a long-lived service server.")
    parser.add_argument("--execute", action="store_true", help="One-shot: plan AND execute (default plan-only).")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive execute confirmation.")
    parser.add_argument(
        "--republish-anchor", action="store_true",
        help="Capture the hole once, then continuously publish it as the RL socket anchor (never plans/moves).",
    )
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
        node = HoleAlignPlanner(namespace=namespace, background=True, arm_override=cli.arm)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info(
            "service mode: call ~/plan_hole_align (dry-run) or ~/move_hole_align (needs allow_execute:=true)."
        )
        try:
            executor.spin()
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    if cli.republish_anchor and not cli.execute:
        node = HoleAlignPlanner(namespace=namespace, background=False, arm_override=cli.arm)
        try:
            node.run_republish_anchor()
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    node = HoleAlignPlanner(namespace=namespace, background=False, arm_override=cli.arm)
    try:
        execute = cli.execute
        if execute and not cli.yes:
            node.run(execute=False)  # show the plan + debug first
            if not _confirm_execute():
                node.get_logger().warn("execute not confirmed; staying in plan-only. Nothing moved.")
                execute = False
        if execute:
            res = node.run(execute=True)
            if cli.republish_anchor:
                # Reuse the hole detected DURING the gross move (clean ~15 cm view) as the anchor -- do
                # NOT re-detect at the 4.5 cm hover, where the centered screw occludes/shifts it.
                hole = getattr(node, "_executed_hole_base", None)
                if res.success and res.moved and hole is not None:
                    node.get_logger().warn(
                        "republishing the DETECTED hole from the gross move as the anchor (no re-detection)."
                    )
                    node._spin_publish_anchor(hole)  # blocks until Ctrl-C
                else:
                    node.get_logger().error(
                        "NOT republishing anchor: execute did not move, or no hole was detected."
                    )
        elif not cli.execute:
            node.run(execute=False)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
