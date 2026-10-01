"""Dedicated real-robot inference node for we_A_resnet18_baseline."""

from __future__ import annotations

import time

import numpy as np
import rclpy

from . import inference_node as base
from .ik import get_delta_dof_pos, get_pose_error, rotmat_from_quat_wxyz
from .ik_we_a import (
    action_to_target_pose_we_a,
    force_base_to_tcp,
    goal_delta_tcp_we_a,
    shaft_socket_axis_error_deg,
    screw_tip_position_base,
)
from .obs_preprocessing_we_a import (
    WeAFrameStacker,
    WeAObsPreprocessConfig,
    build_policy_vector_we_a,
    make_actor_obs_we_a,
    preprocess_rgb_we_a,
)
from .policy_we_a import PolicyLoadError, RlGamesActorWeA


class WeAInferenceNode(base.RLDeployInferenceNode):
    """Parallel deployment stack with we_A's fixed observation/action contract."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.yaw_accum_rad = 0.0
        self.frame_stack = WeAFrameStacker(self.obs_cfg)
        self.get_logger().info(
            "we_A contract active: policy=15, image=180x320x3 ImageNet RGB, action=6, TCP-relative"
        )
        if not bool(self.get_parameter("we_a_geometry_calibrated").value):
            self.get_logger().warn(
                "we_A screw geometry uses the nominal training transform (tip TCP +23 mm, shaft TCP -Z). "
                "Motion start is gated until the real grasp is measured and we_a_geometry_calibrated=true."
            )
        if not bool(self.get_parameter("we_a_force_bias_calibrated").value):
            self.get_logger().warn(
                "we_A env.yaml has use_gravity_comp=false, but policy force bias is not certified. "
                "Motion start is gated until we_a_force_bias_calibrated=true."
            )
        if bool(self.get_parameter("zero_force_obs").value):
            self.get_logger().warn(
                "zero_force_obs=true: policy force channels [6:9] are ZEROED before every actor forward "
                "(diagnostic ablation). Real-contact safety and seat thresholds are unaffected."
            )

    def _declare_parameters(self) -> None:
        super()._declare_parameters()
        self.declare_parameter("screw_tip_offset_tcp_xyz", [0.0, 0.0, 0.023])
        self.declare_parameter("shaft_axis_tcp_xyz", [0.0, 0.0, -1.0])
        self.declare_parameter("socket_axis_mode", "base_axis")
        self.declare_parameter("socket_axis_base_xyz", [0.0, 0.0, 1.0])
        self.declare_parameter("socket_depth_m", 0.0175)
        self.declare_parameter("e2e_action_safety_box", 0.05)
        self.declare_parameter("e2e_yaw_action_bound", 1.5708)
        self.declare_parameter("we_a_geometry_calibrated", False)
        self.declare_parameter("we_a_axis_guard_deg", 60.0)
        self.declare_parameter("we_a_force_bias_calibrated", False)
        # A/B DIAGNOSTIC: feed the actor a BLANK (zero) image instead of the live wrist RGB, so the
        # policy runs on force + socket-anchored goal_delta only. Isolates whether the real wrist view
        # is STEERING the policy wrong (blank should stop the vision-driven XY drift) vs helping.
        self.declare_parameter("blank_image", False)
        # Diagnostic: zero the 3 wrist-force channels (policy indices [6:9]) before every actor forward.
        # pdz_overnight trained with a broken gravity-comp baseline (~0.75*gripper_weight bias baked into
        # the force obs), so the real gravity-compensated F/T does not match training. Zeroing removes the
        # force channel entirely to isolate whether force helps or hurts on the real robot. This does NOT
        # touch the real-contact safety/seat thresholds (those use force_base, not the policy vector).
        self.declare_parameter("zero_force_obs", False)

    def _make_obs_config(self) -> WeAObsPreprocessConfig:
        return WeAObsPreprocessConfig(
            image_height=180,
            image_width=320,
            image_channels=3,
            frame_stack=1,
            ft_smoothing_factor=float(self.get_parameter("ft_smoothing_factor").value),
            force_noise_std=0.0,
            fov_match=bool(self.get_parameter("fov_match").value),
            sim_focal_length_mm=11.0,
            sim_horizontal_aperture_mm=20.955,
        )

    def _load_actor(self) -> RlGamesActorWeA:
        try:
            return RlGamesActorWeA(
                checkpoint=self.get_parameter("policy_checkpoint").value,
                agent_config=self.get_parameter("agent_config").value,
                train_repo=self.get_parameter("train_repo").value,
                device=self.get_parameter("policy_device").value,
                deterministic=self.get_parameter("deterministic_policy").value,
            )
        except PolicyLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PolicyLoadError(f"we_A policy load failed: {exc}") from exc

    def _warmup_actor(self) -> None:
        try:
            dummy_policy = np.zeros(15, dtype=np.float32)
            dummy_image = np.zeros((180, 320, 3), dtype=np.float32)
            self.actor.act(make_actor_obs_we_a(dummy_policy, dummy_image))
            self.get_logger().info("we_A actor warmup complete")
        except Exception as exc:  # noqa: BLE001
            raise PolicyLoadError(f"we_A actor warmup failed: {exc}") from exc

    def _setup_ros_io(self) -> None:
        self._rgb_sub = self.create_subscription(
            base.Image,
            self.get_parameter("rgb_topic").value,
            self._on_rgb_we_a,
            base.qos_profile_sensor_data,
        )
        self.create_subscription(
            base.CameraInfo,
            self.get_parameter("rgb_camera_info_topic").value,
            self._on_camera_info,
            base.qos_profile_sensor_data,
        )
        if self.get_parameter("socket_pose_type").value == "pose_stamped":
            self.create_subscription(
                base.PoseStamped,
                self.get_parameter("socket_pose_topic").value,
                self._on_socket_pose_stamped,
                base.qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                base.DebugPoseItem,
                self.get_parameter("socket_pose_topic").value,
                self._on_socket_debug_pose,
                base.qos_profile_sensor_data,
            )
        self.create_subscription(base.LBRState, self.get_parameter("lbr_state_topic").value, self._on_lbr_state, 1)
        self.create_subscription(
            base.JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            base.qos_profile_sensor_data,
        )
        self.create_subscription(
            base.PoseStamped,
            self.get_parameter("flange_pose_topic").value,
            self._on_flange_pose,
            base.qos_profile_sensor_data,
        )
        self.create_subscription(
            base.WrenchStamped,
            self.get_parameter("ft_topic").value,
            self._on_ft,
            base.qos_profile_sensor_data,
        )
        self.create_subscription(base.Bool, self.get_parameter("estop_topic").value, self._on_estop, 1)
        self.create_service(base.Trigger, "/rl_deploy/start_policy", self._srv_start_policy)
        self.create_service(base.Trigger, "/rl_deploy/stop_policy", self._srv_stop_policy)
        self.create_service(base.Trigger, "/rl_deploy/reset_preinsert", self._srv_reset_preinsert)
        self.create_service(base.Trigger, "/rl_deploy/clear_latches", self._srv_clear_latches)

        self.command_pub = self.create_publisher(
            base.LBRJointPositionCommand, self.get_parameter("lbr_command_topic").value, 1
        )
        self.status_pub = self.create_publisher(base.String, "/rl_deploy/status", 10)
        self.debug_obs_pub = self.create_publisher(base.Float32MultiArray, "/rl_deploy/policy_obs", 1)
        self.seat_pub = self.create_publisher(base.Bool, "/rl_deploy/seat_detected", 1)
        self.gripper_open_pub = self.create_publisher(
            base.Float64, self.get_parameter("gripper_open_topic").value, 1
        )

    def _on_rgb_we_a(self, msg: base.Image) -> None:
        try:
            rgb = base._decode_rgb(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"we_A RGB decode failed: {exc}")
            return
        self.rgb = base.TimedValue(rgb, time.monotonic())

    def _ready(self) -> tuple[bool, str]:
        if (
            self.get_parameter("enable_motion").value
            and self.get_parameter("socket_pose_type").value == "debug_pose_item"
            and int(self.get_parameter("socket_part_id").value) < 0
            and not self.get_parameter("allow_wildcard_part_id").value
        ):
            return False, "socket_part_id must be set explicitly before motion"
        timeout = float(self.get_parameter("input_timeout_s").value)
        checks = [
            (self.rgb.fresh(timeout), "rgb"),
            (self.joint_pos.fresh(timeout), "joint_state"),
            (self.socket_pos.fresh(float(self.get_parameter("socket_timeout_s").value)), "socket_pose"),
        ]
        if self.get_parameter("fingertip_source").value == "flange_pose":
            checks.append((self.flange_pose.fresh(timeout), "flange_pose"))
        force_source = self.get_parameter("force_source").value
        if force_source == "wrench_topic":
            checks.append((self.ft_force.fresh(timeout), "ft"))
            if self.get_parameter("ft_frame").value == "flange":
                checks.append((self.flange_pose.fresh(timeout), "flange_pose"))
        elif force_source == "arm_measured_torque":
            checks.append((self.joint_effort.fresh(timeout), "joint_effort"))
        else:
            checks.append((self.external_torque.fresh(timeout), "external_torque"))
        missing = [name for ok, name in checks if not ok]
        if missing:
            return False, "stale/missing: " + ", ".join(missing)
        if self.kinematics is None:
            return False, "kinematics unavailable"
        fri_problem = self._fri_problem()
        if fri_problem:
            return False, "FRI: " + fri_problem
        return True, ""

    def _socket_axis_base(self, socket_quat: np.ndarray) -> np.ndarray:
        mode = str(self.get_parameter("socket_axis_mode").value)
        if mode == "pose":
            axis = rotmat_from_quat_wxyz(socket_quat) @ np.array([0.0, 0.0, 1.0])
        elif mode == "base_axis":
            axis = np.asarray(self.get_parameter("socket_axis_base_xyz").value, dtype=np.float64)
        else:
            raise ValueError(f"socket_axis_mode must be 'pose' or 'base_axis', got {mode!r}")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError("configured socket axis is zero")
        return axis / norm

    def _control_tick(self) -> None:
        if self.mode == base.MODE_RESET_PREINSERT:
            self._preinsert_tick()
            return
        if self.mode != base.MODE_POLICY:
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

        q = np.asarray(self.joint_pos.value, dtype=np.float64).reshape(7)
        jac = self.kinematics.jacobian(q)
        fingertip_pos, fingertip_quat = self._compute_fingertip(q)
        screw_tip_pos = screw_tip_position_base(
            fingertip_pos,
            fingertip_quat,
            self.get_parameter("screw_tip_offset_tcp_xyz").value,
        )
        raw_force_base = self._contact_force_base(jac)
        force_base = self.force_smoother.update(raw_force_base)
        # ft_bias_base_xyz recreates the uncompensated sim sensor's distal-weight term for the policy.
        # Keep that synthetic load out of real-contact safety and seat thresholds.
        force_bias_base = np.asarray(self.get_parameter("ft_bias_base_xyz").value, dtype=np.float64).reshape(3)
        contact_force_norm = float(np.linalg.norm(raw_force_base - force_bias_base))
        force_tcp = force_base_to_tcp(force_base, fingertip_quat)
        # Diagnostic ablation: feed zero force to the policy (indices [6:9]) while leaving the real-contact
        # safety norm above untouched. The debug/dump obs then also read zero, confirming the flag is live.
        if bool(self.get_parameter("zero_force_obs").value):
            force_tcp = np.zeros(3, dtype=np.float32)
        socket_center, socket_quat = self.socket_pos.value
        socket_opening = self._socket_opening(socket_center, socket_quat)

        stop = self._evaluate_trial_stop(screw_tip_pos, socket_opening, socket_quat, contact_force_norm)
        if stop is not None:
            if stop.seated:
                self.seat_detected = True
                self.seat_pub.publish(base.Bool(data=True))
                if self.get_parameter("open_gripper_on_seat").value:
                    self.gripper_open_pub.publish(
                        base.Float64(data=float(self.get_parameter("gripper_open_value").value))
                    )
                if not self.get_parameter("freeze_on_seat").value:
                    self._status(stop.reason)
                else:
                    self._finish_policy_trial(stop)
                    return
            else:
                self._finish_policy_trial(stop)
                return
        if contact_force_norm > float(self.get_parameter("ft_warn_n").value):
            self._status(f"force warning: {contact_force_norm:.2f} N")

        socket_axis = self._socket_axis_base(socket_quat)
        try:
            axis_angle_deg = shaft_socket_axis_error_deg(
                fingertip_quat,
                self.get_parameter("shaft_axis_tcp_xyz").value,
                socket_axis,
            )
        except ValueError as exc:
            self._finish_policy_trial(
                base.TrialStop("failed_axis_config", str(exc))
            )
            return
        axis_guard_deg = float(self.get_parameter("we_a_axis_guard_deg").value)
        if axis_guard_deg > 0.0 and axis_angle_deg > axis_guard_deg:
            self._finish_policy_trial(
                base.TrialStop(
                    "failed_axis_mismatch",
                    f"shaft/socket axis error {axis_angle_deg:.1f} deg exceeds {axis_guard_deg:.1f} deg guard",
                )
            )
            return
        goal_delta, socket_bottom = goal_delta_tcp_we_a(
            fingertip_pos=fingertip_pos,
            fingertip_quat_wxyz=fingertip_quat,
            socket_opening_pos=socket_opening,
            socket_axis_base=socket_axis,
            socket_depth_m=float(self.get_parameter("socket_depth_m").value),
            screw_tip_offset_tcp_xyz=self.get_parameter("screw_tip_offset_tcp_xyz").value,
            shaft_axis_tcp_xyz=self.get_parameter("shaft_axis_tcp_xyz").value,
        )
        image = self.frame_stack.push(preprocess_rgb_we_a(self.rgb.value, self.obs_cfg, self.cam_intrinsics))
        if self.get_parameter("blank_image").value:
            image = np.zeros_like(image)  # A/B: run on force + anchor only, no wrist vision
        policy = build_policy_vector_we_a(goal_delta, force_tcp, self.prev_action)
        if self.get_parameter("publish_debug_obs").value:
            msg = base.Float32MultiArray()
            msg.data = [float(value) for value in policy]
            self.debug_obs_pub.publish(msg)
        self._maybe_dump_obs(policy, image)

        raw_action = np.clip(self.actor.act(make_actor_obs_we_a(policy, image)), -1.0, 1.0)
        ema = float(self.get_parameter("ema_factor").value)
        action = ema * raw_action + (1.0 - ema) * self.prev_action
        action = base.limit_action_step(
            action,
            position_scale_m=float(self.get_parameter("e2e_pos_action_scale").value),
            rotation_scale_rad=float(self.get_parameter("e2e_rot_action_scale").value),
            max_position_step_m=float(self.get_parameter("max_policy_position_step_m").value),
            max_rotation_step_rad=float(self.get_parameter("max_policy_rotation_step_rad").value),
        )
        self.prev_action = action.copy()
        target_pos, target_quat, self.yaw_accum_rad = action_to_target_pose_we_a(
            action=action,
            fingertip_pos=fingertip_pos,
            fingertip_quat_wxyz=fingertip_quat,
            goal_center_base=socket_bottom,
            pos_scale=float(self.get_parameter("e2e_pos_action_scale").value),
            rot_scale=float(self.get_parameter("e2e_rot_action_scale").value),
            action_safety_box_m=float(self.get_parameter("e2e_action_safety_box").value),
            yaw_accum_rad=self.yaw_accum_rad,
            yaw_bound_rad=float(self.get_parameter("e2e_yaw_action_bound").value),
        )

        if self.get_parameter("z_force_limit_enable").value:
            limit = float(self.get_parameter("z_force_limit_n").value)
            attenuation = float(np.clip(1.0 - contact_force_norm / max(limit, 1e-6), 0.0, 1.0))
            target_pos = np.asarray(target_pos, dtype=np.float64).copy()
            dz = float(target_pos[2]) - float(fingertip_pos[2])
            if dz < 0.0:
                target_pos[2] = float(fingertip_pos[2]) + dz * attenuation

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

    def _srv_start_policy(self, request, response):
        self.yaw_accum_rad = 0.0
        if not self.get_parameter("enable_motion").value:
            return super()._srv_start_policy(request, response)
        if not bool(self.get_parameter("we_a_geometry_calibrated").value):
            response.success = False
            response.message = (
                "we_A held-part geometry is not certified; measure screw_tip_offset_tcp_xyz, confirm "
                "shaft_axis_tcp_xyz, then set we_a_geometry_calibrated:=true."
            )
            return response
        if not bool(self.get_parameter("we_a_force_bias_calibrated").value):
            response.success = False
            response.message = (
                "we_A force bias is not certified; set ft_bias_base_xyz and then "
                "we_a_force_bias_calibrated:=true."
            )
            return response
        return super()._srv_start_policy(request, response)

    def _srv_stop_policy(self, request, response):
        self.yaw_accum_rad = 0.0
        return super()._srv_stop_policy(request, response)

    def _srv_reset_preinsert(self, request, response):
        self.yaw_accum_rad = 0.0
        return super()._srv_reset_preinsert(request, response)

    def _srv_clear_latches(self, request, response):
        self.yaw_accum_rad = 0.0
        return super()._srv_clear_latches(request, response)

    def _depth_roi_seat_trigger(self) -> bool:
        return False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WeAInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
