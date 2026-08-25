"""Launch the dedicated we_A_resnet18_baseline deployment node."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rl_deploy_inference")
    default_params = os.path.join(package_share, "config", "deploy_we_a.yaml")
    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="YAML parameters for the dedicated we_A deployment node.",
    )
    return LaunchDescription(
        [
            params_arg,
            Node(
                package="rl_deploy_inference",
                executable="command_upsampler",
                name="command_upsampler",
                output="screen",
            ),
            Node(
                package="rl_deploy_inference",
                executable="inference_node_we_a",
                name="rl_deploy_inference",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
