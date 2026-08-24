"""Launch the real-robot RL deployment node with the EXPLICIT-ESTIMATOR checkpoint (w2_estimator_192).

Identical to deploy_inference.launch.py but defaults params_file to config/deploy_w2_estimator.yaml.
The node/policy code is unchanged: policy.py auto-detects the aux_head/aux_label in the w2 agent.yaml
and handles the explicit-estimator obs contract. Override params_file:= to point elsewhere.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("rl_deploy_inference")
    default_params = os.path.join(pkg_share, "config", "deploy_w2_estimator.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="YAML file with rl_deploy_inference ROS parameters (explicit-estimator default).",
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
