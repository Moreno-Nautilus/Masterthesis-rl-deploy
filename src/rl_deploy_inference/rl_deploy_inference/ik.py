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


def get_delta_dof_pos_weighted(
    delta_pose: np.ndarray,
    jacobian: np.ndarray,
    joint_weights: np.ndarray,
    damping: float = 0.05,
    q: np.ndarray | None = None,
    q_rest: np.ndarray | None = None,
    null_gain: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Weighted damped least-squares IK with optional nullspace posture task.

    dq = Winv Jt (J Winv Jt + lambda^2 I)^-1 dx  +  (I - J^# J) k (q_rest - q)

    joint_weights (len n): PER-JOINT COST. High weight = "expensive / avoid moving this joint"
    for the Cartesian task, so a high weight on A1/A2/A3 stops the shoulder from dominating
    translation and dragging the TCP down. Winv = diag(1/joint_weights).
    """
    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    w = np.asarray(joint_weights, dtype=np.float64).reshape(-1)
    n = jacobian.shape[1]
    if jacobian.shape[0] != 6:
        raise ValueError(f"jacobian must have shape (6, n), got {jacobian.shape}")

    winv = np.diag(1.0 / w)                              # (n, n) inverse weight
    jwjt = jacobian @ winv @ jacobian.T
    reg = (float(damping) ** 2) * np.eye(6, dtype=np.float64)
    jwjt_inv = np.linalg.solve(jwjt + reg, np.eye(6))
    j_sharp = winv @ jacobian.T @ jwjt_inv               # (n, 6) weighted pseudo-inverse
    dq_task = j_sharp @ delta_pose

    if q is not None and q_rest is not None and np.any(np.asarray(null_gain) != 0.0):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        q_rest = np.asarray(q_rest, dtype=np.float64).reshape(-1)
        null_proj = np.eye(n) - j_sharp @ jacobian
        k = np.asarray(null_gain, dtype=np.float64)
        dq_task = dq_task + null_proj @ (k * (q_rest - q))
    return dq_task


def get_delta_dof_pos_nullspace(
    delta_pose: np.ndarray,
    jacobian: np.ndarray,
    q: np.ndarray,
    q_rest: np.ndarray,
    damping: float = 0.05,
    null_gain: np.ndarray | float = 0.1,
) -> np.ndarray:
    """DLS IK with a nullspace posture task that pins the redundant DoF (elbow) toward q_rest.

    dq = J^dls dx + (I - J^dls J) k (q_rest - q)

    The nullspace term is projected so it does not disturb the TCP pose, but resolves the
    7-DoF redundancy by pulling joints toward a rest posture. Use a per-joint `null_gain`
    array to weight specific joints harder (e.g. joints 3/4 to stop the elbow collapsing).
    """
    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    q_rest = np.asarray(q_rest, dtype=np.float64).reshape(-1)
    n = jacobian.shape[1]
    if jacobian.shape[0] != 6:
        raise ValueError(f"jacobian must have shape (6, n), got {jacobian.shape}")

    reg = (float(damping) ** 2) * np.eye(6, dtype=np.float64)
    jj_t_inv = np.linalg.solve(jacobian @ jacobian.T + reg, np.eye(6))
    j_dls = jacobian.T @ jj_t_inv                       # (n, 6) damped pseudo-inverse
    dq_task = j_dls @ delta_pose                        # primary Cartesian task

    null_proj = np.eye(n) - j_dls @ jacobian            # (I - J^dls J)
    k = np.asarray(null_gain, dtype=np.float64)
    dq_null = null_proj @ (k * (q_rest - q))            # posture bias, TCP-preserving
    return dq_task + dq_null


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

