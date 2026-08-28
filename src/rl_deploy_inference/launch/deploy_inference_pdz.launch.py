"""Launch the pdz_overnight deployment node (reuses the we_A inference node).

The pdz_overnight checkpoints share the we_A observation/action contract, so this launch starts the
same ``inference_node_we_a`` executable and only swaps the parameter file. Defaults to episode 1500;
override for the other spread checkpoints, e.g.::

    ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py \
        params_file:=$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/deploy_pdz_ep500.yaml

NODE-ONLY (no bundled command bridge): for hardware, keep ``joint_trajectory_controller`` active and
run ``ros2 run rl_deploy_inference guarded_joint_trajectory_bridge``. Do not also start the legacy
direct-position upsampler.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
    return LaunchDescription(
        [
            params_arg,
            Node(
                package="rl_deploy_inference",
                executable="inference_node_we_a",
                name="rl_deploy_inference",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
