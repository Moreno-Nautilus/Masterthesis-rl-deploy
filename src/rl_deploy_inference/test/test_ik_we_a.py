from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_deploy_inference.ik import axis_angle_from_quat_wxyz, quat_conjugate_wxyz, quat_mul_wxyz  # noqa: E402
from rl_deploy_inference.ik_we_a import (  # noqa: E402
    action_to_target_pose_we_a,
    goal_delta_tcp_we_a,
    shaft_socket_axis_error_deg,
)


def test_goal_delta_targets_socket_bottom_in_tcp_frame() -> None:
    delta, bottom = goal_delta_tcp_we_a(
        # Tool-down pose: TCP +Z is base -Z. Screw tip extends along TCP +Z while
        # screw/shaft +Z is TCP -Z, so the shaft and socket +Z axes are parallel.
        fingertip_pos=np.array([0.0, 0.0, 0.023]),
        fingertip_quat_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
        socket_opening_pos=np.array([0.0, 0.0, 0.0175]),
        socket_axis_base=np.array([0.0, 0.0, 1.0]),
        socket_depth_m=0.0175,
        screw_tip_offset_tcp_xyz=np.array([0.0, 0.0, 0.023]),
        shaft_axis_tcp_xyz=np.array([0.0, 0.0, -1.0]),
    )
    np.testing.assert_allclose(bottom, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(delta, np.zeros(6), atol=1e-7)


def test_axis_sign_guard_rejects_old_tcp_plus_z_assumption() -> None:
    tool_down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    socket_axis = np.array([0.0, 0.0, 1.0])

    corrected = shaft_socket_axis_error_deg(tool_down_quat, [0.0, 0.0, -1.0], socket_axis)
    old_assumption = shaft_socket_axis_error_deg(tool_down_quat, [0.0, 0.0, 1.0], socket_axis)

    assert corrected == 0.0
    assert old_assumption == 180.0


def test_action_is_tcp_relative_and_cumulative_yaw_is_bounded() -> None:
    half = np.sqrt(0.5)
    fingertip_quat = np.array([half, 0.0, 0.0, half])  # TCP +X points along base +Y.
    target_pos, target_quat, yaw = action_to_target_pose_we_a(
        action=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        fingertip_pos=np.zeros(3),
        fingertip_quat_wxyz=fingertip_quat,
        goal_center_base=np.zeros(3),
        pos_scale=0.01,
        rot_scale=0.1,
        action_safety_box_m=0.05,
        yaw_accum_rad=0.08,
        yaw_bound_rad=0.1,
    )
    np.testing.assert_allclose(target_pos, [0.0, 0.01, 0.0], atol=1e-12)
    assert yaw == 0.1
    delta_quat = quat_mul_wxyz(quat_conjugate_wxyz(fingertip_quat), target_quat)
    np.testing.assert_allclose(axis_angle_from_quat_wxyz(delta_quat), [0.0, 0.0, 0.02], atol=1e-7)
