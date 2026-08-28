import numpy as np

from rl_deploy_inference.guarded_joint_smoother import GuardedJointSmoother


def _smoother():
    return GuardedJointSmoother(
        np.zeros(7),
        max_velocity=np.deg2rad(3.0),
        max_acceleration=np.deg2rad(30.0),
        max_jerk=np.deg2rad(300.0),
        min_duration_s=0.5,
        max_duration_s=8.0,
        lower=np.full(7, -2.0),
        upper=np.full(7, 2.0),
    )


def test_plan_respects_velocity_acceleration_and_jerk_limits():
    smoother = _smoother()
    plan = smoother.plan(
        np.full(7, 0.02),
        sample_dt_s=0.01,
        sample_count=8,
        handoff_time_s=1.0 / 15.0,
    )
    assert len(plan.samples) == 8
    assert np.all(plan.peak_velocity <= np.deg2rad(3.0) * (1.0 + 1e-8))
    assert np.all(plan.peak_acceleration <= np.deg2rad(30.0) * (1.0 + 1e-8))
    assert np.all(plan.peak_jerk <= np.deg2rad(300.0) * (1.0 + 1e-8))


def test_replan_starts_from_committed_q_dq_ddq_handoff():
    smoother = _smoother()
    first = smoother.plan(
        np.full(7, 0.01),
        sample_dt_s=0.01,
        sample_count=8,
        handoff_time_s=1.0 / 15.0,
    )
    smoother.commit(first.handoff_state)
    committed = smoother.state
    second = smoother.plan(
        np.full(7, 0.015),
        sample_dt_s=0.01,
        sample_count=8,
        handoff_time_s=1.0 / 15.0,
    )
    assert np.allclose(committed.q, first.handoff_state.q)
    assert np.allclose(committed.dq, first.handoff_state.dq)
    assert np.allclose(committed.ddq, first.handoff_state.ddq)
    assert np.all(np.isfinite(second.samples[0][0]))


def test_plan_rejects_goal_outside_joint_limits():
    smoother = _smoother()
    goal = np.zeros(7)
    goal[3] = 2.1
    try:
        smoother.plan(
            goal,
            sample_dt_s=0.01,
            sample_count=8,
            handoff_time_s=1.0 / 15.0,
        )
    except ValueError as exc:
        assert "joint limit" in str(exc)
    else:
        raise AssertionError("out-of-limit goal was accepted")
