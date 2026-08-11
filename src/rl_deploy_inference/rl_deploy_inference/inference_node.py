
"""ROS2 node that runs the trained E2E insertion actor on the real iiwa."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import message_filters
import numpy as np
import rclpy
from fp_debug_msgs.msg import DebugPoseItem
from geometry_msgs.msg import PoseStamped, WrenchStamped
from lbr_fri_idl.msg import LBRJointPositionCommand, LBRState
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from std_msgs.msg import Bool, Float32MultiArray, Float64, String
import yaml

from .ik import (
    action_to_target_pose,
    get_delta_dof_pos,
    get_pose_error,
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    quat_wxyz_from_rotmat,
    rotmat_from_quat_wxyz,
    wrench_from_external_torque,
)
from .kinematics import KdlConfig, KdlKinematics, KinematicsUnavailable
from .obs_preprocessing import (
    ForceSmoother,
    FrameStacker,
    ObsPreprocessConfig,
    build_policy_vector,
    make_actor_obs,
    preprocess_rgbd,
)
from .policy import PolicyLoadError, RlGamesActor


IIWA7_LOWER = np.deg2rad(np.array([-170, -120, -170, -120, -170, -120, -175], dtype=np.float64))
IIWA7_UPPER = np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175], dtype=np.float64))
MODE_HOLD = "hold"
MODE_POLICY = "policy"
MODE_RESET_PREINSERT = "reset_preinsert"


@dataclass
class TimedValue:
    value: object | None = None
    wall_t: float = 0.0

    def fresh(self, max_age_s: float) -> bool:
        return self.value is not None and (time.monotonic() - self.wall_t) <= float(max_age_s)


@dataclass
class SeatGeometry:
    xy_error_m: float
    z_above_bottom_m: float


@dataclass
class TrialStop:
    outcome: str
    reason: str
    seated: bool = False


def _pose_to_pos_quat_wxyz(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    p = msg.pose.position
    q = msg.pose.orientation
    return (
        np.array([p.x, p.y, p.z], dtype=np.float64),
        normalize_quat_wxyz(np.array([q.w, q.x, q.y, q.z], dtype=np.float64)),
    )


def _decode_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        channels = 3
    elif enc in ("rgba8", "bgra8"):
        channels = 4
    else:
        raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    arr = arr[:, :, :3]
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


def _finite_diff_angvel(prev_quat: np.ndarray, curr_quat: np.ndarray, dt: float) -> np.ndarray:
    from .ik import axis_angle_from_quat_wxyz

    qdiff = quat_mul_wxyz(curr_quat, quat_conjugate_wxyz(prev_quat))
    if qdiff[0] < 0.0:
        qdiff = -qdiff
    return axis_angle_from_quat_wxyz(qdiff) / max(float(dt), 1e-6)


class RLDeployInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_deploy_inference")
        self._declare_parameters()

        self.rgb = TimedValue()
        self.depth = TimedValue()
        self.socket_pos = TimedValue()
        self.flange_pose = TimedValue()
        self.ft_force = TimedValue()
        self.joint_pos = TimedValue()
        self.external_torque = TimedValue()
        self.cam_intrinsics: tuple | None = None

        self.prev_ft_pos: np.ndarray | None = None
        self.prev_ft_quat: np.ndarray | None = None
        self.prev_loop_t: float | None = None
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.last_command_q: np.ndarray | None = None
        self.manual_estop = False
        self.force_cap_latched = False
        self.seat_detected = False
        self.policy_start_t: float | None = None
        self.trial_outcome = ""
        self.mode = MODE_POLICY if self.get_parameter("policy_active_on_start").value else MODE_HOLD
        self._last_status_t = 0.0
        self._obs_dumped = False

        obs_cfg = self._make_obs_config()
        self.obs_cfg = obs_cfg
        self.frame_stack = FrameStacker(obs_cfg)
        self.force_smoother = ForceSmoother(obs_cfg.ft_smoothing_factor)

        self.joint_names = tuple(
            f"{self.get_parameter('robot_prefix').value}_A{i}" for i in range(1, 8)
        )
        self.kinematics: KdlKinematics | None = None
        self._robot_description: str | None = self.get_parameter("robot_description").value or None

        self.actor = self._load_actor()
        if getattr(self.actor, "_aux_label_dim", 0) > 0:
            self.get_logger().info(
                f"Explicit-estimator checkpoint: feeding dummy '{self.actor._aux_label_key}' "
                f"zeros (dim={self.actor._aux_label_dim}) each step; it has no effect on the action."
            )
        self._init_kinematics_once()
        self._setup_ros_io()

        period = 1.0 / float(self.get_parameter("control_hz").value)
        self.timer = self.create_timer(period, self._control_tick)
        self.get_logger().info(
            "rl_deploy_inference ready: motion is "
            + ("ENABLED" if self.get_parameter("enable_motion").value else "DISABLED")
            + f", mode={self.mode}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("enable_motion", False)
        self.declare_parameter("policy_active_on_start", False)
        self.declare_parameter("stream_hold_when_disabled", True)
        self.declare_parameter("control_hz", 15.0)
        self.declare_parameter(
            "policy_checkpoint",
            "/home/moreno/Masterthesis-rl-train/logs/rl_games/Forge/w2_estimator_192/nn/last_Forge_ep_2000_rew_162.13815.pth",
        )
        self.declare_parameter(
            "agent_config",
            "/home/moreno/Masterthesis-rl-train/logs/rl_games/Forge/w2_estimator_192/params/agent.yaml",
        )
        self.declare_parameter("train_repo", "/home/moreno/Masterthesis-rl-train")
        self.declare_parameter("env_config", "")
        self.declare_parameter("auto_obs_config_from_env_yaml", True)
        self.declare_parameter("policy_device", "cuda:0")
        self.declare_parameter("deterministic_policy", True)

        self.declare_parameter("rgb_topic", "/realsense_2/camera/color/image_rect")
        self.declare_parameter("depth_topic", "/realsense_2/camera/aligned_depth_to_color/image_rect")
        self.declare_parameter("rgb_camera_info_topic", "/realsense_2/camera/color/camera_info")
        # Time-sync the RGB + aligned-depth pair (both carry a header stamp) so the obs is built from
        # one coherent frame. FRI state is high-rate (latest is ~synchronous); socket is slow (latest).
        self.declare_parameter("img_sync_slop_s", 0.02)
        self.declare_parameter("img_sync_queue", 30)
        # FOV match: crop the real color+aligned-depth to the sim pinhole FOV before the 224 resize.
        self.declare_parameter("fov_match", True)
        self.declare_parameter("socket_pose_topic", "/perception/fp/pose_base/fused/assembly")
        self.declare_parameter("socket_pose_type", "debug_pose_item")
        self.declare_parameter("socket_assembly_name", "cooling_manifold")
        self.declare_parameter("socket_part_id", -1)
        # Perception reports the tracked CAD's CENTER pose (FoundationPose uses the centred mesh);
        # sim's anchor (fixed_pos_obs_frame) is the target socket OPENING. Offset is measured in the
        # CAD frame and rotated by the perceived orientation (so a lateral hole offset rotates with
        # yaw, exactly as sim rotates socket_offset_local by fixed_quat). cooling_base is symmetric ->
        # openings at RIGHT (+0.030, 0, +0.0175) / LEFT (-0.030, 0, +0.0175) m -> pick your socket.
        self.declare_parameter("socket_opening_offset_cad_xyz", [0.0, 0.0, 0.0])

        self.declare_parameter("lbr_state_topic", "state")
        self.declare_parameter("joint_state_topic", "/lbr_dual_arm/joint_states")
        self.declare_parameter("lbr_command_topic", "command/joint_position")
        self.declare_parameter("flange_pose_topic", "/right/ee_pose")
        # Fingertip pose source. "fk" (default) = KDL FK(q) to tip_link=gripper_tcp, exactly matching
        # sim (fingertip = FK of joint state to the gripper_tcp body). "flange_pose" = legacy topic +
        # flange_to_fingertip offset (needs /right/ee_pose to be the flange, verified against CAD).
        self.declare_parameter("fingertip_source", "fk")
        # Contact force source. The arm has NO wrist F/T sensor, so default = estimate the external
        # end-effector force from the FRI external joint torque via pinv(J^T). "wrench_topic" = legacy
        # WrenchStamped. ft_sign flips the convention to match sim (validate with a press-test).
        self.declare_parameter("force_source", "arm_external_torque")
        self.declare_parameter("ft_sign", 1.0)
        # Constant base-frame force bias to match sim's gravity-inclusive force_sensor DC (see
        # _contact_force_base). Set to sim's no-contact hover force if the real force is grav-comp'd.
        self.declare_parameter("ft_bias_base_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("ft_topic", "/wrist_ft")
        self.declare_parameter("ft_frame", "base")
        self.declare_parameter("estop_topic", "/rl_deploy/e_stop")

        self.declare_parameter("robot_prefix", "lbr_two")
        self.declare_parameter("base_link", "lbr_two_link_0")
        self.declare_parameter("tip_link", "lbr_two_gripper_tcp")
        self.declare_parameter("fallback_tip_link", "lbr_two_link_ee")
        self.declare_parameter("flange_to_fingertip_xyz", [0.0, 0.0, 0.1463])
        self.declare_parameter("flange_to_fingertip_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("robot_description", "")
        self.declare_parameter(
            "robot_description_service",
            "/lbr_dual_arm/robot_state_publisher/get_parameters",
        )

        self.declare_parameter("image_height", 224)
        self.declare_parameter("image_width", 224)
        self.declare_parameter("frame_stack", 1)
        self.declare_parameter("depth_far_m", 0.3)
        self.declare_parameter("ft_smoothing_factor", 0.25)
        self.declare_parameter("force_noise_std", 0.0)

        # Sim trained with per-episode DR ema_factor_range [0.025, 0.1] (forge_env). 1.0 (no
        # smoothing) is 10-40x outside that range: both the applied action and the prev_action obs
        # would be out-of-distribution. 0.0625 is the mid of the trained range (lower = gentler).
        self.declare_parameter("ema_factor", 0.0625)
        self.declare_parameter("e2e_pos_action_scale", 0.01)
        self.declare_parameter("e2e_rot_action_scale", 0.1)
        self.declare_parameter("socket_action_bound", 0.08)
        self.declare_parameter("ik_damping", 0.1)
        self.declare_parameter("max_joint_step_rad", 0.010)
        self.declare_parameter("joint_limit_margin_rad", 0.03)
        # Preinsert reset target. "joint" = move to a fixed measured 7-joint pose (needs it set).
        # "socket_hover" = Cartesian: hover preinsert_hover_z_m ABOVE the perceived socket opening
        # (socket estimate + hole offset + hover_z), keeping the current fingertip orientation. Sim
        # starts ~6.3 cm above the opening, so 0.06 matches. NEEDS a good socket estimate + a roughly
        # upright start pose; UNTESTED ON HARDWARE -> walk it guarded.
        self.declare_parameter("preinsert_mode", "joint")
        self.declare_parameter("preinsert_hover_z_m", 0.06)
        self.declare_parameter("preinsert_pos_tol_m", 0.005)
        self.declare_parameter("preinsert_joint_position", [0.0] * 7)
        self.declare_parameter("preinsert_joint_position_set", False)
        self.declare_parameter("preinsert_tolerance_rad", 0.01)
        self.declare_parameter("preinsert_max_joint_step_rad", 0.006)

        self.declare_parameter("input_timeout_s", 0.35)
        self.declare_parameter("socket_timeout_s", 0.75)
        self.declare_parameter("ft_force_cap_n", 20.0)
        self.declare_parameter("ft_warn_n", 10.0)
        self.declare_parameter("seat_force_n", 14.0)
        # Policy-trial stop gates. Geometry is evaluated in the perceived socket frame, where the socket
        # opening is z=0 and the bottom is z=-seat_socket_depth_m. Defaults mirror the sim cooling socket:
        # 17.5 mm depth and a 25%-of-depth seated tolerance (~4.4 mm above bottom).
        self.declare_parameter("seat_socket_depth_m", 0.0175)
        self.declare_parameter("seat_z_tolerance_m", 0.0045)
        self.declare_parameter("seat_xy_tolerance_m", 0.004)
        # If geometry is available, a seat-force hit above the bottom zone is treated as a jam/failure
        # instead of success. Set false to recover the legacy "force alone means seated" behavior.
        self.declare_parameter("seat_force_requires_geometry", True)
        self.declare_parameter("trial_timeout_s", 10.0)
        self.declare_parameter("trial_max_overtravel_m", 0.002)
        self.declare_parameter("seat_depth_roi_max_m", 0.0)
        self.declare_parameter("freeze_on_seat", True)
        self.declare_parameter("open_gripper_on_seat", False)
        self.declare_parameter("gripper_open_topic", "/gripper/open_cmd")
        self.declare_parameter("gripper_open_value", 0.04)

        self.declare_parameter("publish_debug_obs", False)
        self.declare_parameter("obs_dump_path", "")

    def _make_obs_config(self) -> ObsPreprocessConfig:
        values = {
            "image_height": int(self.get_parameter("image_height").value),
            "image_width": int(self.get_parameter("image_width").value),
            "image_channels": 4,
            "frame_stack": int(self.get_parameter("frame_stack").value),
            "depth_far_m": float(self.get_parameter("depth_far_m").value),
            "ft_smoothing_factor": float(self.get_parameter("ft_smoothing_factor").value),
            "force_noise_std": float(self.get_parameter("force_noise_std").value),
            "fov_match": bool(self.get_parameter("fov_match").value),
        }
        if self.get_parameter("auto_obs_config_from_env_yaml").value:
            path = self._env_config_path()
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
                    values["image_height"] = int(env_cfg.get("image_height", values["image_height"]))
                    values["image_width"] = int(env_cfg.get("image_width", values["image_width"]))
                    values["image_channels"] = int(env_cfg.get("image_channels", values["image_channels"]))
                    values["frame_stack"] = int(env_cfg.get("frame_stack", values["frame_stack"]))
                    values["ft_smoothing_factor"] = float(
                        env_cfg.get("ft_smoothing_factor", values["ft_smoothing_factor"])
                    )
                    spawn = ((env_cfg.get("tiled_camera") or {}).get("spawn")) or {}
                    clipping = spawn.get("clipping_range")
                    if clipping and len(clipping) >= 2:
                        values["depth_far_m"] = float(clipping[1])
                    # Sim pinhole intrinsics for the FOV-match crop (fall back to D405-tuned defaults).
                    values["sim_focal_length_mm"] = float(spawn.get("focal_length", 19.0))
                    values["sim_horizontal_aperture_mm"] = float(spawn.get("horizontal_aperture", 20.955))
                    self.get_logger().info(
                        "Loaded obs config from "
                        f"{path}: H={values['image_height']} W={values['image_width']} "
                        f"C={values['image_channels']} frame_stack={values['frame_stack']} "
                        f"depth_far_m={values['depth_far_m']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"Could not read env_config for obs sizing: {exc}")
        if values["image_channels"] != 4:
            raise ValueError(
                f"Expected per-frame RGB-D image_channels=4 from training env, got {values['image_channels']}"
            )
        return ObsPreprocessConfig(**values)

    def _env_config_path(self) -> str:
        explicit = os.path.expanduser(str(self.get_parameter("env_config").value or ""))
        if explicit:
            return explicit
        agent_cfg = Path(os.path.expanduser(str(self.get_parameter("agent_config").value)))
        candidate = agent_cfg.parent / "env.yaml"
        return str(candidate) if candidate.is_file() else ""

    def _load_actor(self) -> RlGamesActor:
        try:
            return RlGamesActor(
                checkpoint=self.get_parameter("policy_checkpoint").value,
                agent_config=self.get_parameter("agent_config").value,
                train_repo=self.get_parameter("train_repo").value,
                device=self.get_parameter("policy_device").value,
                deterministic=self.get_parameter("deterministic_policy").value,
                policy_dim=21,
                image_shape=(
                    int(self.obs_cfg.image_height),
                    int(self.obs_cfg.image_width),
                    int(self.obs_cfg.image_channels) * int(self.obs_cfg.frame_stack),
                ),
                action_dim=5,
            )
        except PolicyLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PolicyLoadError(f"Policy load failed: {exc}") from exc

    def _setup_ros_io(self) -> None:
        # RGB + aligned-depth are time-synchronized (approximate, on header stamp) so every obs is
        # built from one coherent camera frame instead of two independently-latched images.
        self._rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("rgb_topic").value, qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=qos_profile_sensor_data
        )
        self._img_sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=int(self.get_parameter("img_sync_queue").value),
            slop=float(self.get_parameter("img_sync_slop_s").value),
        )
        self._img_sync.registerCallback(self._on_rgbd_synced)
        self.create_subscription(
            CameraInfo, self.get_parameter("rgb_camera_info_topic").value, self._on_camera_info, qos_profile_sensor_data
        )
        if self.get_parameter("socket_pose_type").value == "pose_stamped":
            self.create_subscription(
                PoseStamped,
                self.get_parameter("socket_pose_topic").value,
                self._on_socket_pose_stamped,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                DebugPoseItem,
                self.get_parameter("socket_pose_topic").value,
                self._on_socket_debug_pose,
                qos_profile_sensor_data,
            )
        self.create_subscription(LBRState, self.get_parameter("lbr_state_topic").value, self._on_lbr_state, 1)
        self.create_subscription(
            JointState, self.get_parameter("joint_state_topic").value, self._on_joint_state, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("flange_pose_topic").value, self._on_flange_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            WrenchStamped, self.get_parameter("ft_topic").value, self._on_ft, qos_profile_sensor_data
        )
        self.create_subscription(Bool, self.get_parameter("estop_topic").value, self._on_estop, 1)
        self.create_service(Trigger, "/rl_deploy/start_policy", self._srv_start_policy)
        self.create_service(Trigger, "/rl_deploy/stop_policy", self._srv_stop_policy)
        self.create_service(Trigger, "/rl_deploy/reset_preinsert", self._srv_reset_preinsert)
        self.create_service(Trigger, "/rl_deploy/clear_latches", self._srv_clear_latches)

        self.command_pub = self.create_publisher(
            LBRJointPositionCommand, self.get_parameter("lbr_command_topic").value, 1
        )
        self.status_pub = self.create_publisher(String, "/rl_deploy/status", 10)
        self.debug_obs_pub = self.create_publisher(Float32MultiArray, "/rl_deploy/policy_obs", 1)
        self.seat_pub = self.create_publisher(Bool, "/rl_deploy/seat_detected", 1)
        self.gripper_open_pub = self.create_publisher(Float64, self.get_parameter("gripper_open_topic").value, 1)

    def _retrieve_robot_description(self, timeout_s: float = 0.5) -> str | None:
        service = self.get_parameter("robot_description_service").value
        client = self.create_client(GetParameters, service)
        if not client.wait_for_service(timeout_sec=timeout_s):
            return None
        request = GetParameters.Request(names=["robot_description"])
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if future.result() is None or not future.result().values:
            return None
        value = future.result().values[0]
        return value.string_value or None

    def _init_kinematics_once(self) -> None:
        if self.kinematics is not None:
            return
        if not self._robot_description:
            self._robot_description = self._retrieve_robot_description(timeout_s=0.75)
        if not self._robot_description:
            self.get_logger().warn("No robot_description yet; holding until kinematics can initialize.")
            return
        cfg = KdlConfig(
            robot_description=self._robot_description,
            base_link=self.get_parameter("base_link").value,
            tip_link=self.get_parameter("tip_link").value,
            joint_names=self.joint_names,
            fallback_tip_link=self.get_parameter("fallback_tip_link").value,
            fallback_tip_offset_xyz=tuple(self.get_parameter("flange_to_fingertip_xyz").value),
            fallback_tip_offset_quat_wxyz=tuple(self.get_parameter("flange_to_fingertip_quat_wxyz").value),
        )
        try:
            self.kinematics = KdlKinematics(cfg)
            self.get_logger().info(f"KDL ready: {cfg.base_link} -> {cfg.tip_link}")
        except KinematicsUnavailable as exc:
            self.get_logger().error(str(exc))
            self.kinematics = None

    def _on_rgbd_synced(self, rgb_msg: Image, depth_msg: Image) -> None:
        stamp = time.monotonic()
        try:
            rgb = _decode_rgb(rgb_msg)
            depth = _decode_depth_m(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"RGB-D decode failed: {exc}")
            return
        self.rgb = TimedValue(rgb, stamp)
        self.depth = TimedValue(depth, stamp)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]; aligned_depth_to_color shares these intrinsics.
        self.cam_intrinsics = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def _on_socket_debug_pose(self, msg: DebugPoseItem) -> None:
        want_assembly = self.get_parameter("socket_assembly_name").value
        want_part = int(self.get_parameter("socket_part_id").value)
        if want_assembly and msg.assembly_name != want_assembly:
            return
        if want_part >= 0 and int(msg.part_id) != want_part:
            return
        if want_part < 0:
            self.get_logger().warn(
                f"socket_part_id=-1 wildcard accepted {msg.assembly_name}/{msg.part_id}; set this parameter explicitly.",
                throttle_duration_sec=5.0,
            )
        # Store (center_pos, orientation): the orientation places the socket opening (see _socket_opening).
        self.socket_pos = TimedValue(_pose_to_pos_quat_wxyz(msg.pose_base), time.monotonic())

    def _on_socket_pose_stamped(self, msg: PoseStamped) -> None:
        self.socket_pos = TimedValue(_pose_to_pos_quat_wxyz(msg), time.monotonic())

    def _on_lbr_state(self, msg: LBRState) -> None:
        stamp = time.monotonic()
        self.joint_pos = TimedValue(np.asarray(msg.measured_joint_position, dtype=np.float64), stamp)
        self.external_torque = TimedValue(np.asarray(msg.external_torque, dtype=np.float64), stamp)

    def _on_joint_state(self, msg: JointState) -> None:
        by_name = {name: pos for name, pos in zip(msg.name, msg.position)}
        if not all(name in by_name for name in self.joint_names):
            return
        q = np.array([by_name[name] for name in self.joint_names], dtype=np.float64)
        self.joint_pos = TimedValue(q, time.monotonic())

    def _on_flange_pose(self, msg: PoseStamped) -> None:
        self.flange_pose = TimedValue(_pose_to_pos_quat_wxyz(msg), time.monotonic())

    def _on_ft(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        force = np.array([f.x, f.y, f.z], dtype=np.float64)
        if self.get_parameter("ft_frame").value == "flange" and self.flange_pose.value is not None:
            _, q = self.flange_pose.value
            force = rotmat_from_quat_wxyz(q) @ force
        self.ft_force = TimedValue(force, time.monotonic())

    def _on_estop(self, msg: Bool) -> None:
        self.manual_estop = bool(msg.data)
        if self.manual_estop:
            self._status("manual e-stop asserted")

    def _srv_start_policy(self, _request, response):
        if not self.get_parameter("enable_motion").value:
            response.success = False
            response.message = "enable_motion is false; arm the node first."
            return response
        if (
            self.get_parameter("socket_pose_type").value == "debug_pose_item"
            and int(self.get_parameter("socket_part_id").value) < 0
        ):
            response.success = False
            response.message = "socket_part_id must be set before starting policy motion."
            return response
        if self.manual_estop or self.force_cap_latched:
            response.success = False
            response.message = "safety latch/e-stop active; stop motion and clear before starting."
            return response
        self.mode = MODE_POLICY
        self.prev_action[:] = 0.0
        self.seat_detected = False
        self.seat_pub.publish(Bool(data=False))
        self.policy_start_t = time.monotonic()
        self.trial_outcome = "running"
        self.frame_stack.reset()
        self.force_smoother.reset()
        # The policy is recurrent (LSTM); sim zeroes the hidden state on every episode reset. Do the
        # same here or a fresh insertion starts with stale recurrent memory from the previous run.
        self.actor.reset()
        # Drop the finite-difference history so the first post-start velocity isn't a huge spike.
        self.prev_ft_pos = None
        self.prev_ft_quat = None
        self.prev_loop_t = None
        response.success = True
        response.message = "policy mode requested; node will stream when all inputs are fresh."
        self._status(response.message)
        return response

    def _srv_stop_policy(self, _request, response):
        self.mode = MODE_HOLD
        self.prev_action[:] = 0.0
        self.policy_start_t = None
        self.trial_outcome = "stopped"
        response.success = True
        response.message = "policy stopped; holding measured joint position."
        self._status(response.message)
        return response

    def _srv_reset_preinsert(self, _request, response):
        if not self.get_parameter("enable_motion").value:
            response.success = False
            response.message = "enable_motion is false; reset-to-preinsert will not move."
            return response
        socket_hover = self.get_parameter("preinsert_mode").value == "socket_hover"
        if not socket_hover and not self.get_parameter("preinsert_joint_position_set").value:
            response.success = False
            response.message = "preinsert_joint_position_set is false; configure the 7 joint target first."
            return response
        if socket_hover and not self.socket_pos.fresh(float(self.get_parameter("socket_timeout_s").value)):
            response.success = False
            response.message = "socket_hover preinsert needs a fresh socket pose first."
            return response
        if self.manual_estop or self.force_cap_latched:
            response.success = False
            response.message = "safety latch/e-stop active; reset-to-preinsert refused."
            return response
        self.mode = MODE_RESET_PREINSERT
        self.prev_action[:] = 0.0
        self.policy_start_t = None
        self.trial_outcome = "reset_preinsert"
        response.success = True
        response.message = "reset-to-preinsert requested."
        self._status(response.message)
        return response

    def _srv_clear_latches(self, _request, response):
        self.force_cap_latched = False
        self.seat_detected = False
        self.prev_action[:] = 0.0
        self.policy_start_t = None
        self.trial_outcome = ""
        self.force_smoother.reset()
        self.seat_pub.publish(Bool(data=False))
        response.success = True
        response.message = "force/seat latches cleared; manual e-stop still follows /rl_deploy/e_stop."
        self._status(response.message)
        return response

    def _compute_fingertip(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fingertip (gripper_tcp) pose in base_link.

        Default source "fk": KDL FK(q) to tip_link, matching sim (fingertip = FK of joint state to the
        gripper_tcp body) and keeping the obs, IK error, and Jacobian on ONE kinematic source.
        Source "flange_pose": legacy /right/ee_pose + flange_to_fingertip offset.
        """
        if self.get_parameter("fingertip_source").value == "flange_pose":
            flange_pos, flange_quat = self.flange_pose.value
            offset = np.asarray(self.get_parameter("flange_to_fingertip_xyz").value, dtype=np.float64).reshape(3)
            offset_quat = normalize_quat_wxyz(
                np.asarray(self.get_parameter("flange_to_fingertip_quat_wxyz").value, dtype=np.float64)
            )
            ft_pos = flange_pos + rotmat_from_quat_wxyz(flange_quat) @ offset
            ft_quat = quat_mul_wxyz(flange_quat, offset_quat)
            return ft_pos, ft_quat
        pos, R = self.kinematics.fk(q)
        return pos, quat_wxyz_from_rotmat(R)

    def _contact_force_base(self, jac: np.ndarray) -> np.ndarray:
        """3-axis contact force in the base frame (sim ft_force = world-frame contact force).

        Default: estimate the external end-effector force from the FRI external joint torque,
        ``F = ft_sign * pinv(J^T) tau_ext [0:3]``. Legacy: the WrenchStamped topic value.
        """
        if self.get_parameter("force_source").value == "wrench_topic":
            force = np.asarray(self.ft_force.value, dtype=np.float64).reshape(3)
        else:
            tau_ext = np.asarray(self.external_torque.value, dtype=np.float64).reshape(7)
            sign = float(self.get_parameter("ft_sign").value)
            force = sign * wrench_from_external_torque(jac, tau_ext)[0:3]
        # Sim's force_sensor sits above the gripper -> it INCLUDES the constant distal-weight term
        # (gripper+screw), world frame. If the real force is gravity/payload-compensated, add it back
        # here so the DC matches sim (measure it as sim's no-contact hover force). Default 0 = no bias.
        return force + np.asarray(self.get_parameter("ft_bias_base_xyz").value, dtype=np.float64).reshape(3)

    def _socket_opening(self, center: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """Socket OPENING in base frame = perceived center + R(perceived_quat) @ CAD-frame offset.

        Mirrors sim's fixed_pos_obs_frame = fixed_pos + R(fixed_quat) @ socket_offset_local(+height):
        the lateral hole offset rotates with the perceived yaw, so it stays correct under socket yaw.
        """
        offset_cad = np.asarray(
            self.get_parameter("socket_opening_offset_cad_xyz").value, dtype=np.float64
        ).reshape(3)
        center = np.asarray(center, dtype=np.float64).reshape(3)
        return center + rotmat_from_quat_wxyz(quat_wxyz) @ offset_cad

    def _ready(self) -> tuple[bool, str]:
        if (
            self.get_parameter("enable_motion").value
            and self.get_parameter("socket_pose_type").value == "debug_pose_item"
            and int(self.get_parameter("socket_part_id").value) < 0
        ):
            return False, "socket_part_id must be set explicitly before motion"
        timeout = float(self.get_parameter("input_timeout_s").value)
        checks = [
            (self.rgb.fresh(timeout), "rgb"),
            (self.depth.fresh(timeout), "depth"),
            (self.joint_pos.fresh(timeout), "joint_state"),
            (self.socket_pos.fresh(float(self.get_parameter("socket_timeout_s").value)), "socket_pose"),
        ]
        if self.get_parameter("fingertip_source").value == "flange_pose":
            checks.append((self.flange_pose.fresh(timeout), "flange_pose"))
        if self.get_parameter("force_source").value == "wrench_topic":
            checks.append((self.ft_force.fresh(timeout), "ft"))
        else:
            checks.append((self.external_torque.fresh(timeout), "external_torque"))
        missing = [name for ok, name in checks if not ok]
        if missing:
            return False, "stale/missing: " + ", ".join(missing)
        if self.kinematics is None:
            return False, "kinematics unavailable"
        return True, ""

    def _control_tick(self) -> None:
        if self.mode == MODE_RESET_PREINSERT:
            self._preinsert_tick()
            return

        if self.mode != MODE_POLICY:
            reason = (
                f"trial={self.trial_outcome}"
                if self.trial_outcome.startswith(("succeeded_", "failed_"))
                else f"mode={self.mode}"
            )
            self._hold(reason)
            return

        if self.kinematics is None:
            self._init_kinematics_once()

        ready, reason = self._ready()
        if not ready:
            self._hold(reason)
            return

        now = time.monotonic()
        dt = now - self.prev_loop_t if self.prev_loop_t is not None else 1.0 / float(self.get_parameter("control_hz").value)
        self.prev_loop_t = now

        q = np.asarray(self.joint_pos.value, dtype=np.float64).reshape(7)
        jac = self.kinematics.jacobian(q)  # one kinematic source for force estimate + IK
        fingertip_pos, fingertip_quat = self._compute_fingertip(q)
        if self.prev_ft_pos is None:
            ee_linvel = np.zeros(3, dtype=np.float32)
            ee_angvel = np.zeros(3, dtype=np.float32)
        else:
            ee_linvel = ((fingertip_pos - self.prev_ft_pos) / max(dt, 1e-6)).astype(np.float32)
            ee_angvel = _finite_diff_angvel(self.prev_ft_quat, fingertip_quat, dt).astype(np.float32)
        self.prev_ft_pos = fingertip_pos.copy()
        self.prev_ft_quat = fingertip_quat.copy()
        # SIM PARITY (non-negotiable): Forge zeroes the roll/pitch angular-velocity channels every
        # step (forge_env._compute_intermediate_values: `self.ee_angvel_fd[:, 0:2] = 0.0`), and the
        # E2E env inherits that unchanged, so the trained policy only ever saw ee_angvel = [0, 0, wz].
        # Leaving the real roll/pitch rates in makes two of the 21 obs dims out-of-distribution.
        ee_angvel[0:2] = 0.0

        ft_smooth = self.force_smoother.update(self._contact_force_base(jac))
        force_norm = float(np.linalg.norm(ft_smooth))
        socket_center, socket_quat = self.socket_pos.value
        socket_pos = self._socket_opening(socket_center, socket_quat)

        stop = self._evaluate_trial_stop(fingertip_pos, socket_pos, socket_quat, force_norm)
        if stop is not None:
            if stop.seated:
                self.seat_detected = True
                self.seat_pub.publish(Bool(data=True))
                if self.get_parameter("open_gripper_on_seat").value:
                    self.gripper_open_pub.publish(Float64(data=float(self.get_parameter("gripper_open_value").value)))
                if not self.get_parameter("freeze_on_seat").value:
                    self._status(stop.reason)
                else:
                    self._finish_policy_trial(stop)
                    return
            else:
                self._finish_policy_trial(stop)
                return
        if force_norm > float(self.get_parameter("ft_warn_n").value):
            self._status(f"force warning: {force_norm:.2f} N")

        image = self.frame_stack.push(
            preprocess_rgbd(self.rgb.value, self.depth.value, self.obs_cfg, self.cam_intrinsics)
        )
        policy = build_policy_vector(
            fingertip_pos=fingertip_pos,
            socket_pos_estimate=socket_pos,
            fingertip_quat_wxyz=fingertip_quat,
            ee_linvel=ee_linvel,
            ee_angvel=ee_angvel,
            ft_force=ft_smooth,
            prev_action=self.prev_action,
        )
        if self.get_parameter("publish_debug_obs").value:
            msg = Float32MultiArray()
            msg.data = [float(x) for x in policy]
            self.debug_obs_pub.publish(msg)
        self._maybe_dump_obs(policy, image)

        raw_action = np.clip(self.actor.act(make_actor_obs(policy, image)), -1.0, 1.0)
        ema = float(self.get_parameter("ema_factor").value)
        action = ema * raw_action + (1.0 - ema) * self.prev_action
        self.prev_action = action.astype(np.float32)

        target_pos, target_quat = action_to_target_pose(
            action=action,
            fingertip_pos=fingertip_pos,
            fingertip_quat_wxyz=fingertip_quat,
            socket_pos=socket_pos,
            pos_scale=float(self.get_parameter("e2e_pos_action_scale").value),
            rot_scale=float(self.get_parameter("e2e_rot_action_scale").value),
            socket_action_bound=float(self.get_parameter("socket_action_bound").value),
        )
        delta_pose = get_pose_error(fingertip_pos, fingertip_quat, target_pos, target_quat)
        dq = get_delta_dof_pos(delta_pose, jac, damping=float(self.get_parameter("ik_damping").value))
        q_cmd = self._limit_command(q, q + dq)

        if self.manual_estop or self.force_cap_latched:
            self._hold("e-stop latched")
            return
        if not self.get_parameter("enable_motion").value:
            self._hold("motion disabled")
            return
        self._publish_q(q_cmd)
        self.last_command_q = q_cmd

    def _preinsert_tick(self) -> None:
        timeout = float(self.get_parameter("input_timeout_s").value)
        if not self.joint_pos.fresh(timeout):
            self._hold("reset-to-preinsert waiting for joint_state")
            return
        if self.manual_estop or self.force_cap_latched or not self.get_parameter("enable_motion").value:
            self.mode = MODE_HOLD
            self._hold("reset-to-preinsert stopped by safety")
            return

        q = np.asarray(self.joint_pos.value, dtype=np.float64).reshape(7)
        if self.get_parameter("preinsert_mode").value == "socket_hover":
            self._preinsert_socket_hover(q)
            return
        q_goal = np.asarray(self.get_parameter("preinsert_joint_position").value, dtype=np.float64).reshape(7)
        err = q_goal - q
        if float(np.max(np.abs(err))) <= float(self.get_parameter("preinsert_tolerance_rad").value):
            self.mode = MODE_HOLD
            self._publish_q(q_goal)
            self.last_command_q = q_goal
            self._status("preinsert reached; holding")
            return
        max_step = float(self.get_parameter("preinsert_max_joint_step_rad").value)
        q_cmd = self._limit_command(q, q + np.clip(err, -max_step, max_step))
        self._publish_q(q_cmd)
        self.last_command_q = q_cmd
        self._status(f"reset-to-preinsert moving; max_err={float(np.max(np.abs(err))):.4f} rad")

    def _preinsert_socket_hover(self, q: np.ndarray) -> None:
        """Cartesian preinsert: move the fingertip to hover_z above the perceived socket opening.

        Target = socket opening (center + R(quat)@offset) + [0,0,hover_z], base frame. Orientation is
        held at the CURRENT fingertip quat (no reorientation commanded), so this only translates the
        peg over the hole. Steps are joint-limited; a force cap aborts if it unexpectedly touches down.
        """
        if self.kinematics is None or not self.socket_pos.fresh(float(self.get_parameter("socket_timeout_s").value)):
            self._hold("socket_hover preinsert waiting for socket + kinematics")
            return
        jac = self.kinematics.jacobian(q)
        timeout = float(self.get_parameter("input_timeout_s").value)
        force_fresh = (
            self.ft_force.fresh(timeout)
            if self.get_parameter("force_source").value == "wrench_topic"
            else self.external_torque.fresh(timeout)
        )
        if force_fresh:
            force_norm = float(np.linalg.norm(self._contact_force_base(jac)))
            if force_norm > float(self.get_parameter("ft_force_cap_n").value):
                self.force_cap_latched = True
                self.mode = MODE_HOLD
                self._hold(f"socket_hover preinsert force cap: {force_norm:.2f} N")
                return

        center, socket_quat = self.socket_pos.value
        hover = np.array([0.0, 0.0, float(self.get_parameter("preinsert_hover_z_m").value)])
        target_pos = self._socket_opening(center, socket_quat) + hover
        fingertip_pos, fingertip_quat = self._compute_fingertip(q)
        pos_err = float(np.linalg.norm(target_pos - fingertip_pos))
        if pos_err <= float(self.get_parameter("preinsert_pos_tol_m").value):
            self.mode = MODE_HOLD
            self._publish_q(q)
            self.last_command_q = q
            self._status("socket_hover preinsert reached; holding")
            return
        # Position-only servo: target orientation == current (delta orientation ~0 -> keeps pose upright).
        delta_pose = get_pose_error(fingertip_pos, fingertip_quat, target_pos, fingertip_quat)
        dq = get_delta_dof_pos(delta_pose, jac, damping=float(self.get_parameter("ik_damping").value))
        max_step = float(self.get_parameter("preinsert_max_joint_step_rad").value)
        q_cmd = self._limit_command(q, q + np.clip(dq, -max_step, max_step))
        self._publish_q(q_cmd)
        self.last_command_q = q_cmd
        self._status(f"socket_hover preinsert moving; pos_err={pos_err:.4f} m")

    def _limit_command(self, q_measured: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        max_step = float(self.get_parameter("max_joint_step_rad").value)
        q_cmd = q_measured + np.clip(q_target - q_measured, -max_step, max_step)
        margin = float(self.get_parameter("joint_limit_margin_rad").value)
        return np.clip(q_cmd, IIWA7_LOWER + margin, IIWA7_UPPER - margin)

    def _finish_policy_trial(self, stop: TrialStop) -> None:
        self.mode = MODE_HOLD
        self.policy_start_t = None
        self.prev_action[:] = 0.0
        self.trial_outcome = stop.outcome
        if stop.seated:
            self.seat_detected = True
        self._last_status_t = 0.0
        self._hold(f"{stop.outcome}: {stop.reason}")

    def _seat_geometry(
        self, fingertip_pos: np.ndarray, socket_opening_pos: np.ndarray, socket_quat: np.ndarray
    ) -> SeatGeometry | None:
        socket_depth = float(self.get_parameter("seat_socket_depth_m").value)
        if socket_depth <= 0.0:
            return None
        rel_socket = rotmat_from_quat_wxyz(socket_quat).T @ (
            np.asarray(fingertip_pos, dtype=np.float64).reshape(3)
            - np.asarray(socket_opening_pos, dtype=np.float64).reshape(3)
        )
        return SeatGeometry(
            xy_error_m=float(np.linalg.norm(rel_socket[0:2])),
            z_above_bottom_m=float(rel_socket[2] + socket_depth),
        )

    def _depth_roi_seat_trigger(self) -> bool:
        depth_limit = float(self.get_parameter("seat_depth_roi_max_m").value)
        if depth_limit <= 0.0 or self.depth.value is None:
            return False
        d = np.asarray(self.depth.value, dtype=np.float32)
        h, w = d.shape[:2]
        roi = d[h // 2 - h // 12 : h // 2 + h // 12, w // 2 - w // 12 : w // 2 + w // 12]
        finite = roi[np.isfinite(roi) & (roi > 0.0)]
        return bool(finite.size > 16 and float(np.median(finite)) <= depth_limit)

    def _evaluate_trial_stop(
        self, fingertip_pos: np.ndarray, socket_opening_pos: np.ndarray, socket_quat: np.ndarray, force_norm: float
    ) -> TrialStop | None:
        force_cap = float(self.get_parameter("ft_force_cap_n").value)
        if force_cap > 0.0 and force_norm > force_cap:
            self.force_cap_latched = True
            return TrialStop("failed_force_cap", f"force cap exceeded: {force_norm:.2f} N")

        geom = self._seat_geometry(fingertip_pos, socket_opening_pos, socket_quat)
        max_overtravel = float(self.get_parameter("trial_max_overtravel_m").value)
        if geom is not None and max_overtravel > 0.0 and geom.z_above_bottom_m < -max_overtravel:
            return TrialStop(
                "failed_overtravel",
                f"z={geom.z_above_bottom_m:.4f} m below socket bottom limit, xy={geom.xy_error_m:.4f} m",
            )

        geometry_seated = False
        if geom is not None:
            z_tol = float(self.get_parameter("seat_z_tolerance_m").value)
            xy_tol = float(self.get_parameter("seat_xy_tolerance_m").value)
            geometry_seated = geom.z_above_bottom_m <= z_tol and (xy_tol <= 0.0 or geom.xy_error_m <= xy_tol)
            if geometry_seated:
                return TrialStop(
                    "succeeded_seated",
                    f"geometry seat: z_above_bottom={geom.z_above_bottom_m:.4f} m, xy={geom.xy_error_m:.4f} m",
                    seated=True,
                )

        if self._depth_roi_seat_trigger():
            return TrialStop("succeeded_seated", "depth ROI seat trigger", seated=True)

        seat_force = float(self.get_parameter("seat_force_n").value)
        force_trigger = seat_force > 0.0 and force_norm >= seat_force
        if force_trigger:
            if self.get_parameter("seat_force_requires_geometry").value and geom is not None and not geometry_seated:
                return TrialStop(
                    "failed_early_contact",
                    f"seat force {force_norm:.2f} N before seated geometry"
                    f" (z_above_bottom={geom.z_above_bottom_m:.4f} m, xy={geom.xy_error_m:.4f} m)",
                )
            return TrialStop("succeeded_seated", f"seat force reached: {force_norm:.2f} N", seated=True)

        timeout_s = float(self.get_parameter("trial_timeout_s").value)
        if timeout_s > 0.0 and self.policy_start_t is not None and (time.monotonic() - self.policy_start_t) > timeout_s:
            return TrialStop("failed_timeout", f"policy trial exceeded {timeout_s:.1f} s")

        return None

    def _publish_q(self, q: np.ndarray) -> None:
        msg = LBRJointPositionCommand()
        msg.joint_position = [float(x) for x in np.asarray(q, dtype=np.float64).reshape(7)]
        self.command_pub.publish(msg)

    def _hold(self, reason: str) -> None:
        q = self.joint_pos.value
        should_stream = self.get_parameter("stream_hold_when_disabled").value
        if should_stream and q is not None:
            self._publish_q(np.asarray(q, dtype=np.float64).reshape(7))
        self._status("holding: " + reason)

    def _status(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_status_t < 1.0:
            return
        self._last_status_t = now
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _maybe_dump_obs(self, policy: np.ndarray, image: np.ndarray) -> None:
        path = os.path.expanduser(str(self.get_parameter("obs_dump_path").value or ""))
        if not path or self._obs_dumped:
            return
        np.savez(path, policy=policy.astype(np.float32), image=image.astype(np.float32))
        self._obs_dumped = True
        self.get_logger().info(f"Wrote first deploy obs dump to {path}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RLDeployInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
