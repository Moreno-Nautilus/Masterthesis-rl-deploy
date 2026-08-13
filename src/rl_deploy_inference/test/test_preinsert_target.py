"""Unit tests for the pure preinsert-target math (no ROS runtime needed)."""

import pytest

from rl_deploy_inference.preinsert_planner import compute_preinsert_target


def test_hover_offset_adds_to_global_z_only():
    pos, quat = compute_preinsert_target(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
        hover_z_m=0.15,
        orientation_mode="current_tcp",
        fixed_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert pos[0] == pytest.approx(1.0)
    assert pos[1] == pytest.approx(2.0)
    assert pos[2] == pytest.approx(3.15)
    assert quat == (0.0, 0.0, 0.0, 1.0)


def test_current_tcp_mode_passes_through_current_orientation():
    q = (0.1, 0.2, 0.3, 0.9)
    _, quat = compute_preinsert_target(
        (0.0, 0.0, 0.0), q, hover_z_m=0.1, orientation_mode="current_tcp", fixed_quat_xyzw=(0.0, 0.0, 0.0, 1.0)
    )
    assert quat == pytest.approx(q)


def test_fixed_mode_uses_fixed_quat_and_ignores_current():
    fixed = (0.0, 1.0, 0.0, 0.0)
    _, quat = compute_preinsert_target(
        (0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5), hover_z_m=0.1, orientation_mode="fixed", fixed_quat_xyzw=fixed
    )
    assert quat == pytest.approx(fixed)


def test_current_tcp_mode_requires_a_current_orientation():
    with pytest.raises(ValueError):
        compute_preinsert_target(
            (0.0, 0.0, 0.0), None, hover_z_m=0.1, orientation_mode="current_tcp", fixed_quat_xyzw=(0.0, 0.0, 0.0, 1.0)
        )


def test_unknown_orientation_mode_raises():
    with pytest.raises(ValueError):
        compute_preinsert_target(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), hover_z_m=0.1, orientation_mode="bogus", fixed_quat_xyzw=(0.0, 0.0, 0.0, 1.0)
        )


def test_down_mode_points_tool_straight_down():
    import math

    from rl_deploy_inference.preinsert_planner import _rotmat_from_quat_xyzw

    # an arbitrary awkward current orientation (tool NOT vertical)
    cur = (-0.0852, 0.9778, 0.1044, 0.1603)
    _, quat = compute_preinsert_target(
        (0.5, 0.0, 0.0), cur, hover_z_m=0.15, orientation_mode="down", fixed_quat_xyzw=(0.0, 0.0, 0.0, 1.0)
    )
    tool_z = _rotmat_from_quat_xyzw(quat)[:, 2]
    assert tool_z[0] == pytest.approx(0.0, abs=1e-6)
    assert tool_z[1] == pytest.approx(0.0, abs=1e-6)
    assert tool_z[2] == pytest.approx(-1.0, abs=1e-6)  # exactly straight down


def test_down_mode_requires_current_orientation():
    with pytest.raises(ValueError):
        compute_preinsert_target(
            (0.0, 0.0, 0.0), None, hover_z_m=0.1, orientation_mode="down", fixed_quat_xyzw=(0.0, 0.0, 0.0, 1.0)
        )
