"""Launch the headless MoveIt preinsert planner as a long-lived service server.

  ros2 launch rl_deploy_inference preinsert_planner.launch.py

Then, from another terminal:
  ros2 service call /lbr_dual_arm_y_gripper/preinsert_planner/plan_preinsert std_srvs/srv/Trigger   # dry-run
  # to allow the execute service, launch with allow_execute:=true, then:
  ros2 service call /lbr_dual_arm_y_gripper/preinsert_planner/move_preinsert std_srvs/srv/Trigger   # MOVES the arm

For a one-shot plan/execute-and-exit instead, prefer `ros2 run rl_deploy_inference preinsert_planner`
(add --execute to move). This launch file is the reusable service form.

Requires (separate terminals): hardware.launch.py, move_group.launch.py (mode:=hardware rviz:=false),
and the perception pipeline publishing the socket pose. See docs/PREINSERT_MOVEIT.md.
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
    default_params = os.path.join(pkg_share, "config", "preinsert.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params, description="YAML params for preinsert_planner."
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="lbr_dual_arm_y_gripper",
        description="Robot namespace (move_group / tf / joint_states live here).",
    )
    arm_arg = DeclareLaunchArgument(
        "arm",
        default_value="right",
        description="Which arm to move: 'right' (arm_two) or 'left' (arm_one).",
    )
    allow_execute_arg = DeclareLaunchArgument(
        "allow_execute",
        default_value="false",
        description="Allow the ~/move_preinsert service to actually move the robot.",
    )

    return LaunchDescription(
        [
            params_arg,
            namespace_arg,
            arm_arg,
            allow_execute_arg,
            Node(
                package="rl_deploy_inference",
                executable="preinsert_planner",
                name="preinsert_planner",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {"arm": LaunchConfiguration("arm")},
                    {"allow_execute": LaunchConfiguration("allow_execute")},
                ],
                arguments=["--service", "--namespace", LaunchConfiguration("namespace")],
            ),
        ]
    )
