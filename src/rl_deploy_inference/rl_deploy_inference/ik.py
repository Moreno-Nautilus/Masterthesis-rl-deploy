"""Isaac-free port of the Factory differential-IK controller."""

from __future__ import annotations

import math

import numpy as np


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize_quat_wxyz(a)
    b = normalize_quat_wxyz(b)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return normalize_quat_wxyz(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def quat_from_rotvec_wxyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    return normalize_quat_wxyz(np.r_[math.cos(half), axis * math.sin(half)])


def rotmat_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z). Matches the KDL FK rotation convention."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_wxyz(np.array([w, x, y, z], dtype=np.float64))


def wrench_from_external_torque(jacobian: np.ndarray, tau_ext: np.ndarray) -> np.ndarray:
    """Estimate the external end-effector wrench [Fx,Fy,Fz,Tx,Ty,Tz] (base frame) from joint torques.

    Static relation tau = J^T W (7 = 7x6 @ 6); invert least-squares: W = pinv(J^T) tau_ext. The FORCE
    part (0:3) is the net external force in the base frame and is reference-point invariant, so the
    fingertip Jacobian is fine even though sim reads the wrist ``force_sensor`` link. NOTE: sim's
    ``get_link_incoming_joint_force`` also carries the distal gravity/inertia term, while KUKA's
    ``external_torque`` is gravity-compensated -> validate sign/scale/DC on hardware (press-test).
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape[0] != 6:
        raise ValueError(f"jacobian must have shape (6, n), got {jacobian.shape}")
    tau_ext = np.asarray(tau_ext, dtype=np.float64).reshape(jacobian.shape[1])
    return np.linalg.pinv(jacobian.T) @ tau_ext


def axis_angle_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:4]
    sin_half = float(np.linalg.norm(v))
    if sin_half < 1e-9:
        return 2.0 * v
    angle = 2.0 * math.atan2(sin_half, float(q[0]))
    return v / sin_half * angle


def get_pose_error(
    current_pos: np.ndarray,
    current_quat_wxyz: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Match factory_control.get_pose_error(..., jacobian_type='geometric')."""
    current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    current_quat = normalize_quat_wxyz(current_quat_wxyz)
    target_quat = normalize_quat_wxyz(target_quat_wxyz)

    if float(np.dot(target_quat, current_quat)) < 0.0:
        target_quat = -target_quat

    quat_error = quat_mul_wxyz(target_quat, quat_conjugate_wxyz(current_quat))
    return np.r_[target_pos - current_pos, axis_angle_from_quat_wxyz(quat_error)]


def get_delta_dof_pos(delta_pose: np.ndarray, jacobian: np.ndarray, damping: float = 0.1) -> np.ndarray:
    """Damped least-squares IK: J^T (J J^T + lambda^2 I)^-1 dx."""
    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape[0] != 6:
        raise ValueError(f"jacobian must have shape (6, n), got {jacobian.shape}")
    jj_t = jacobian @ jacobian.T
    reg = (float(damping) ** 2) * np.eye(6, dtype=np.float64)
    return jacobian.T @ np.linalg.solve(jj_t + reg, delta_pose)


def action_to_target_pose(
    action: np.ndarray,
    fingertip_pos: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    socket_pos: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    socket_action_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Port InsertionEnvE2EIiwa._pre_physics_step for one real robot."""
    a = np.asarray(action, dtype=np.float64).reshape(5)
    fingertip_pos = np.asarray(fingertip_pos, dtype=np.float64).reshape(3)
    socket_pos = np.asarray(socket_pos, dtype=np.float64).reshape(3)

    target_pos = fingertip_pos + float(pos_scale) * a[:3]
    target_pos = socket_pos + np.clip(
        target_pos - socket_pos,
        -float(socket_action_bound),
        float(socket_action_bound),
    )

    rotvec = np.array([a[3] * float(rot_scale), a[4] * float(rot_scale), 0.0], dtype=np.float64)
    target_quat = quat_mul_wxyz(fingertip_quat_wxyz, quat_from_rotvec_wxyz(rotvec))
    return target_pos, target_quat

