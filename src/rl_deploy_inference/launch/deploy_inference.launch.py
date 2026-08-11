"""Launch the real-robot RL deployment inference node."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("rl_deploy_inference")
    default_params = os.path.join(pkg_share, "config", "deploy.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="YAML file with rl_deploy_inference ROS parameters.",
    )

    return LaunchDescription(
        [
            params_arg,
            Node(
                package="rl_deploy_inference",
                executable="inference_node",
                name="rl_deploy_inference",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
