from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_deploy_inference.ik import (  # noqa: E402
    action_to_target_pose,
    axis_angle_from_quat_wxyz,
    get_delta_dof_pos,
    get_pose_error,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    quat_wxyz_from_rotmat,
    rotmat_from_quat_wxyz,
    wrench_from_external_torque,
)


def test_dls_matches_identity_jacobian_solution_with_damping() -> None:
    delta_pose = np.array([1.0, -2.0, 3.0, 0.2, -0.3, 0.4])
    jac = np.eye(6)

    dq = get_delta_dof_pos(delta_pose, jac, damping=0.1)

    np.testing.assert_allclose(dq, delta_pose / 1.01, atol=1e-12)


def test_pose_error_translation_and_tilt_quaternion() -> None:
    current_pos = np.zeros(3)
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([0.01, -0.02, 0.03])
    action = np.array([0.0, 0.0, 0.0, 0.5, -0.25])

    _, target_quat = action_to_target_pose(
        action=action,
        fingertip_pos=current_pos,
        fingertip_quat_wxyz=current_quat,
        socket_pos=np.zeros(3),
        pos_scale=0.01,
        rot_scale=0.1,
        socket_action_bound=0.08,
    )
    err = get_pose_error(current_pos, current_quat, target_pos, target_quat)

    np.testing.assert_allclose(err[:3], target_pos, atol=1e-12)
    np.testing.assert_allclose(err[3:], [0.05, -0.025, 0.0], atol=1e-12)


def test_quat_wxyz_from_rotmat_roundtrips() -> None:
    for q in (
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, -0.5, 0.5]),
        np.array([0.0, 0.7071, 0.0, 0.7071]),
    ):
        R = rotmat_from_quat_wxyz(q)
        q_rt = quat_wxyz_from_rotmat(R)
        if np.dot(q_rt, q) < 0.0:
            q_rt = -q_rt
        np.testing.assert_allclose(q_rt, q / np.linalg.norm(q), atol=1e-6)


def test_wrench_from_external_torque_recovers_applied_wrench() -> None:
    # jac (6x7): a full-rank 6-dof map padded with a null 7th column. tau = J^T W -> pinv(J^T) tau = W.
    jac = np.hstack([np.eye(6), np.zeros((6, 1))])
    wrench = np.array([3.0, -4.0, 5.0, 0.1, -0.2, 0.3])
    tau = jac.T @ wrench
    recovered = wrench_from_external_torque(jac, tau)
    np.testing.assert_allclose(recovered, wrench, atol=1e-9)


def test_action_translation_clips_around_socket_and_has_no_yaw_delta() -> None:
    fingertip_pos = np.array([0.2, -0.2, 0.4])
    socket_pos = np.array([0.0, 0.0, 0.3])
    action = np.array([100.0, -100.0, 100.0, 0.5, -0.5])

    target_pos, target_quat = action_to_target_pose(
        action=action,
        fingertip_pos=fingertip_pos,
        fingertip_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        socket_pos=socket_pos,
        pos_scale=0.01,
        rot_scale=0.1,
        socket_action_bound=0.08,
    )
    delta_quat = quat_mul_wxyz(target_quat, quat_conjugate_wxyz(np.array([1.0, 0.0, 0.0, 0.0])))
    rotvec = axis_angle_from_quat_wxyz(delta_quat)

    np.testing.assert_allclose(target_pos, [0.08, -0.08, 0.38], atol=1e-12)
    assert abs(rotvec[2]) < 1e-12
