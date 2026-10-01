"""Launch the pdz_overnight deployment node (reuses the we_A inference node) + command bridge.

The pdz_overnight checkpoints share the we_A observation/action contract, so this launch starts the
same ``inference_node_we_a`` executable and only swaps the parameter file. Defaults to episode 1500;
override for the other spread checkpoints, e.g.::

    ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py \
        params_file:=$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/deploy_pdz_ep500.yaml

COMMAND PATH: this launch ALSO starts ``guarded_joint_trajectory_bridge``. The inference node emits
15 Hz joint setpoints on ``/rl_deploy/command_15hz``; the bridge smooths them into bounded q/dq/ddq
horizons and forwards them to ``joint_trajectory_controller`` (the SAME controller MoveIt / hole-align
use, which survives contact on this robot). Keep ``joint_trajectory_controller`` ACTIVE -- do NOT
switch to lbr_joint_position_command_controller, and do NOT run the legacy 200 Hz command_upsampler
(bare position steps through the direct LBR controller are the documented drop-prone path).

Set start_bridge:=false to launch the node alone (e.g. to run the bridge manually with overrides).
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rl_deploy_inference")
    default_params = os.path.join(package_share, "config", "deploy_pdz_ep1500.yaml")
    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="YAML parameters for the pdz_overnight deployment node "
        "(deploy_pdz_ep500.yaml / deploy_pdz_ep1000.yaml / deploy_pdz_ep1500.yaml).",
    )
    start_bridge_arg = DeclareLaunchArgument(
        "start_bridge",
        default_value="true",
        description="Start guarded_joint_trajectory_bridge alongside the node (JTC command path). "
        "Set false to run the bridge manually.",
    )
    robot_prefix_arg = DeclareLaunchArgument(
        "robot_prefix",
        default_value="lbr_two",
        description="FRI joint prefix for the bridge: lbr_two (right arm) or lbr_one (left arm). "
        "Must match the params_file's robot_prefix.",
    )
    return LaunchDescription(
        [
            params_arg,
            start_bridge_arg,
            robot_prefix_arg,
            Node(
                package="rl_deploy_inference",
                executable="inference_node_we_a",
                name="rl_deploy_inference",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
            Node(
                package="rl_deploy_inference",
                executable="guarded_joint_trajectory_bridge",
                name="guarded_joint_trajectory_bridge",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_bridge")),
                parameters=[
                    {
                        # Loosened commissioning guards. ISOLATED ON HARDWARE (left arm, 2026-08-31):
                        # the bridge latched on `target lead ... exceeds 2.5 deg` and silently dropped
                        # ALL policy commands -> arm never moved. The policy commands joint TARGETS that
                        # lead the current measured position by more than the old 2.5 deg cap, so the
                        # lead guard tripped every tick. Raised to 15 deg (lead + tracking) so real
                        # policy motion flows; verified the arm then tracks a repeating command cleanly.
                        # Baked in so they persist across every relaunch (ros2 param set was resetting).
                        # The step-size caps in the deploy YAML (max_policy_position_step_m,
                        # max_joint_step_rad) still bound per-tick motion; this guard is a safety trip,
                        # not the motion limiter. Lower if you want a tighter runaway trip once seated.
                        "max_tracking_error_deg": 15.0,
                        "max_target_lead_deg": 15.0,
                        "max_velocity_deg_s": 10.0,
                        "max_acceleration_deg_s2": 60.0,
                        "max_jerk_deg_s3": 600.0,
                        "joint_state_max_age_s": 0.2,
                        "robot_prefix": LaunchConfiguration("robot_prefix"),
                    }
                ],
            ),
        ]
    )
