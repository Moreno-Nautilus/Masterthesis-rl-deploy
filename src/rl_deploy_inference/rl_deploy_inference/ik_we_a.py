"""Geometry and six-axis action port for we_A_resnet18_baseline."""

from __future__ import annotations

import numpy as np

from .ik import quat_from_rotvec_wxyz, quat_mul_wxyz, rotmat_from_quat_wxyz


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def socket_bottom_from_opening(
    socket_opening_pos: np.ndarray,
    socket_axis_base: np.ndarray,
    socket_depth_m: float,
) -> np.ndarray:
    """Training uses socket bottom as G; perception supplies the socket opening."""
    opening = np.asarray(socket_opening_pos, dtype=np.float64).reshape(3)
    return opening - _unit(socket_axis_base, "socket_axis_base") * float(socket_depth_m)


def screw_tip_position_base(
    fingertip_pos: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    screw_tip_offset_tcp_xyz: np.ndarray,
) -> np.ndarray:
    """Transform the calibrated TCP-to-screw-tip offset into the robot base frame."""
    ft_pos = np.asarray(fingertip_pos, dtype=np.float64).reshape(3)
    rotation = rotmat_from_quat_wxyz(fingertip_quat_wxyz)
    offset = np.asarray(screw_tip_offset_tcp_xyz, dtype=np.float64).reshape(3)
    return ft_pos + rotation @ offset


def shaft_socket_axis_error_deg(
    fingertip_quat_wxyz: np.ndarray,
    shaft_axis_tcp_xyz: np.ndarray,
    socket_axis_base: np.ndarray,
) -> float:
    """Return the unsigned shaft/socket axis angle used by the deployment sign guard."""
    rotation = rotmat_from_quat_wxyz(fingertip_quat_wxyz)
    held_axis = rotation @ _unit(shaft_axis_tcp_xyz, "shaft_axis_tcp_xyz")
    cosine = float(np.clip(np.dot(held_axis, _unit(socket_axis_base, "socket_axis_base")), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def goal_delta_tcp_we_a(
    fingertip_pos: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    socket_opening_pos: np.ndarray,
    socket_axis_base: np.ndarray,
    socket_depth_m: float,
    screw_tip_offset_tcp_xyz: np.ndarray,
    shaft_axis_tcp_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ([tip-to-goal position, shaft-axis error] in TCP, socket bottom)."""
    rotation = rotmat_from_quat_wxyz(fingertip_quat_wxyz)
    tip_pos = screw_tip_position_base(fingertip_pos, fingertip_quat_wxyz, screw_tip_offset_tcp_xyz)
    socket_axis = _unit(socket_axis_base, "socket_axis_base")
    held_axis = _unit(rotation @ _unit(shaft_axis_tcp_xyz, "shaft_axis_tcp_xyz"), "held_axis")
    socket_bottom = socket_bottom_from_opening(socket_opening_pos, socket_axis, socket_depth_m)

    pos_world = socket_bottom - tip_pos
    cross = np.cross(held_axis, socket_axis)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(held_axis, socket_axis), -1.0, 1.0))
    angle = float(np.arctan2(sine, cosine))
    rot_world = np.zeros(3, dtype=np.float64) if sine < 1e-9 else (cross / sine) * angle
    delta_tcp = np.concatenate([rotation.T @ pos_world, rotation.T @ rot_world])
    return delta_tcp.astype(np.float32), socket_bottom


def force_base_to_tcp(force_base: np.ndarray, fingertip_quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = rotmat_from_quat_wxyz(fingertip_quat_wxyz)
    return (rotation.T @ np.asarray(force_base, dtype=np.float64).reshape(3)).astype(np.float32)


def action_to_target_pose_we_a(
    action: np.ndarray,
    fingertip_pos: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    goal_center_base: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    action_safety_box_m: float,
    yaw_accum_rad: float,
    yaw_bound_rad: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Port we_A's TCP-relative translation, body rotation, and cumulative yaw bound."""
    command = np.asarray(action, dtype=np.float64).reshape(6).copy()
    ft_pos = np.asarray(fingertip_pos, dtype=np.float64).reshape(3)
    rotation = rotmat_from_quat_wxyz(fingertip_quat_wxyz)

    target_pos = ft_pos + rotation @ (float(pos_scale) * command[:3])
    box = float(action_safety_box_m)
    if box > 0.0:
        center = np.asarray(goal_center_base, dtype=np.float64).reshape(3)
        target_pos = center + np.clip(target_pos - center, -box, box)

    yaw_increment = command[5] * float(rot_scale)
    new_yaw = float(yaw_accum_rad) + yaw_increment
    bound = float(yaw_bound_rad)
    if bound > 0.0:
        bounded_yaw = float(np.clip(new_yaw, -bound, bound))
        yaw_increment = bounded_yaw - float(yaw_accum_rad)
        new_yaw = bounded_yaw
    rotvec = np.array(
        [command[3] * float(rot_scale), command[4] * float(rot_scale), yaw_increment],
        dtype=np.float64,
    )
    target_quat = quat_mul_wxyz(fingertip_quat_wxyz, quat_from_rotvec_wxyz(rotvec))
    return target_pos, target_quat, new_yaw
