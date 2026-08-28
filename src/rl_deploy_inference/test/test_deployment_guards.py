from types import SimpleNamespace

import numpy as np

from rl_deploy_inference.deployment_guards import fri_state_problem, limit_action_step


def _healthy_fri_state(**overrides):
    values = {
        "session_state": 4,
        "connection_quality": 3,
        "safety_state": 0,
        "drive_state": 2,
        "client_command_mode": 1,
        "overlay_type": 1,
        "control_mode": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fri_state_problem_accepts_expected_position_mode():
    assert fri_state_problem(_healthy_fri_state(), expected_control_mode=0) == ""


def test_fri_state_problem_reports_all_mismatches():
    problem = fri_state_problem(
        _healthy_fri_state(session_state=2, connection_quality=1, control_mode=1),
        expected_control_mode=0,
    )
    assert "session_state=2" in problem
    assert "connection_quality=1" in problem
    assert "control_mode=1" in problem


def test_action_limit_caps_translation_and_rotation_norms():
    action = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    limited = limit_action_step(
        action,
        position_scale_m=0.005,
        rotation_scale_rad=0.1,
        max_position_step_m=0.001,
        max_rotation_step_rad=np.deg2rad(0.5),
    )
    assert np.isclose(np.linalg.norm(limited[:3]) * 0.005, 0.001)
    assert np.isclose(np.linalg.norm(limited[3:]) * 0.1, np.deg2rad(0.5))
    assert np.allclose(limited[:3] / limited[0], action[:3] / action[0])


def test_disabled_action_limits_preserve_action():
    action = np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.7], dtype=np.float32)
    limited = limit_action_step(
        action,
        position_scale_m=0.005,
        rotation_scale_rad=0.1,
        max_position_step_m=0.0,
        max_rotation_step_rad=0.0,
    )
    assert np.array_equal(limited, action)
