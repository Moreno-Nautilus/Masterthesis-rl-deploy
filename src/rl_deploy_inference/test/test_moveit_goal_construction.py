"""Validate MoveGroup goal construction against the real moveit_msgs.

Skips where moveit_msgs is not installed (e.g. CI without MoveIt). This is the field-name check that
could not run before MoveIt was installed: it builds the exact goals plan_to_pose/plan_to_joint send.
"""

import pytest

from rl_deploy_inference.motion_commander import (
    MotionCommander,
    MotionCommanderConfig,
    moveit_available,
)

pytestmark = pytest.mark.skipif(not moveit_available(), reason="moveit_msgs not installed")


@pytest.fixture(scope="module")
def commander():
    import rclpy

    rclpy.init()
    node = rclpy.create_node("test_motion_commander", namespace="/lbr_dual_arm_y_gripper")
    try:
        yield MotionCommander(node, MotionCommanderConfig())
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_pose_goal_has_expected_fields(commander):
    from moveit_msgs.action import MoveGroup

    goal = commander._build_goal(
        [commander._pose_constraints((0.4, 0.1, 0.5), (0.0, 0.0, 0.0, 1.0), 0.01, 0.1)],
        plan_only=True,
    )
    assert isinstance(goal, MoveGroup.Goal)
    req = goal.request
    assert req.group_name == "arm_two"
    assert req.start_state.is_diff is True
    assert req.max_velocity_scaling_factor == pytest.approx(0.05)
    assert req.max_acceleration_scaling_factor == pytest.approx(0.05)
    assert goal.planning_options.plan_only is True
    assert goal.planning_options.planning_scene_diff.is_diff is True

    c = req.goal_constraints[0]
    assert len(c.position_constraints) == 1
    assert len(c.orientation_constraints) == 1

    pc = c.position_constraints[0]
    assert pc.link_name == "lbr_two_gripper_tcp"
    assert pc.header.frame_id == "base_link"
    assert list(pc.constraint_region.primitives[0].dimensions) == pytest.approx([0.01])
    pose = pc.constraint_region.primitive_poses[0]
    assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx((0.4, 0.1, 0.5))

    oc = c.orientation_constraints[0]
    assert (oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w) == pytest.approx(
        (0.0, 0.0, 0.0, 1.0)
    )
    assert oc.absolute_x_axis_tolerance == pytest.approx(0.1)
    assert oc.absolute_z_axis_tolerance == pytest.approx(0.1)


def test_joint_goal_has_expected_fields(commander):
    targets = {"lbr_two_A1": 0.2, "lbr_two_A2": -0.3}
    goal = commander._build_goal([commander._joint_constraints(targets, 0.01)], plan_only=True)
    jcs = goal.request.goal_constraints[0].joint_constraints
    assert {j.joint_name: j.position for j in jcs} == pytest.approx(targets)
    assert all(j.tolerance_above == pytest.approx(0.01) for j in jcs)


def test_left_arm_config_targets_arm_one(commander):
    import rclpy

    node = rclpy.create_node("test_left", namespace="/lbr_dual_arm_y_gripper")
    try:
        left = MotionCommander(
            node, MotionCommanderConfig(group_name="arm_one", tip_link="lbr_one_gripper_tcp")
        )
        goal = left._build_goal(
            [left._pose_constraints((0.0, -0.1, 0.5), (0.0, 0.0, 0.0, 1.0), 0.01, 0.1)], plan_only=True
        )
        assert goal.request.group_name == "arm_one"
        assert goal.request.goal_constraints[0].position_constraints[0].link_name == "lbr_one_gripper_tcp"
    finally:
        node.destroy_node()
