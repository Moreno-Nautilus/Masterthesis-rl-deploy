"""Pure safety helpers shared by the real-robot deployment nodes."""

from __future__ import annotations

import numpy as np


FRI_STATE_EXPECTED = {
    "session_state": (4, "COMMANDING_ACTIVE"),
    "connection_quality": (3, "EXCELLENT"),
    "safety_state": (0, "NORMAL_OPERATION"),
    "drive_state": (2, "ACTIVE"),
    "client_command_mode": (1, "POSITION"),
    "overlay_type": (1, "JOINT"),
}


def fri_state_problem(state: object, expected_control_mode: int) -> str:
    """Return a concise FRI-state mismatch, or an empty string when safe."""
    if expected_control_mode not in (-1, 0, 1):
        return (
            "expected_control_mode must be -1 (either), 0 (position), "
            "or 1 (Cartesian impedance)"
        )

    expected = dict(FRI_STATE_EXPECTED)
    if expected_control_mode >= 0:
        label = "POSITION_CONTROL" if expected_control_mode == 0 else "CARTESIAN_IMPEDANCE_CONTROL"
        expected["control_mode"] = (expected_control_mode, label)

    bad = []
    for field, (value, label) in expected.items():
        actual = getattr(state, field, None)
        if actual != value:
            bad.append(f"{field}={actual} (need {value}/{label})")
    return "; ".join(bad)


def limit_action_step(
    action: np.ndarray,
    *,
    position_scale_m: float,
    rotation_scale_rad: float,
    max_position_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    """Norm-limit the Cartesian action while preserving its direction."""
    limited = np.asarray(action, dtype=np.float64).copy()
    if limited.ndim != 1 or limited.size < 3:
        raise ValueError("action must be a one-dimensional vector with at least 3 values")

    def _limit(values: np.ndarray, scale: float, cap: float, name: str) -> None:
        if cap <= 0.0:
            return
        if scale <= 0.0:
            raise ValueError(f"{name} scale must be positive when its step guard is enabled")
        physical_norm = float(np.linalg.norm(values)) * scale
        if physical_norm > cap:
            values *= cap / physical_norm

    _limit(limited[:3], float(position_scale_m), float(max_position_step_m), "position")
    if limited.size > 3:
        _limit(limited[3:], float(rotation_scale_rad), float(max_rotation_step_rad), "rotation")
    return limited.astype(np.float32)
