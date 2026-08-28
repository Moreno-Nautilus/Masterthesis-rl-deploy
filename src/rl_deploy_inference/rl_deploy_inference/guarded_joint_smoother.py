"""Stateful quintic smoothing for guarded 15 Hz to JTC command handoffs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointMotionState:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


@dataclass(frozen=True)
class GuardedJointPlan:
    samples: tuple[tuple[np.ndarray, float, np.ndarray, np.ndarray], ...]
    handoff_state: JointMotionState
    duration_s: float
    peak_velocity: np.ndarray
    peak_acceleration: np.ndarray
    peak_jerk: np.ndarray


class GuardedJointSmoother:
    """Receding quintic planner that carries position, velocity, and acceleration."""

    def __init__(
        self,
        q_start: np.ndarray,
        max_velocity: float | np.ndarray,
        max_acceleration: float | np.ndarray,
        max_jerk: float | np.ndarray,
        *,
        min_duration_s: float = 0.5,
        max_duration_s: float = 8.0,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> None:
        q_start = np.asarray(q_start, dtype=np.float64)
        if q_start.ndim != 1 or not np.all(np.isfinite(q_start)):
            raise ValueError("q_start must be a finite one-dimensional joint vector")
        self.n = q_start.size
        self.max_velocity = self._positive_vector(max_velocity, "max_velocity")
        self.max_acceleration = self._positive_vector(max_acceleration, "max_acceleration")
        self.max_jerk = self._positive_vector(max_jerk, "max_jerk")
        self.min_duration_s = float(min_duration_s)
        self.max_duration_s = float(max_duration_s)
        if not 0.0 < self.min_duration_s <= self.max_duration_s:
            raise ValueError("need 0 < min_duration_s <= max_duration_s")
        self.lower = self._limit_vector(lower, -np.inf)
        self.upper = self._limit_vector(upper, np.inf)
        if np.any(self.lower >= self.upper):
            raise ValueError("each lower joint limit must be below its upper limit")
        if np.any(q_start < self.lower) or np.any(q_start > self.upper):
            raise ValueError("q_start lies outside the configured joint limits")
        self._state = JointMotionState(q_start.copy(), np.zeros(self.n), np.zeros(self.n))

    def _positive_vector(self, value: float | np.ndarray, name: str) -> np.ndarray:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), (self.n,)).copy()
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"{name} must contain finite positive values")
        return result

    def _limit_vector(self, value: np.ndarray | None, fill: float) -> np.ndarray:
        if value is None:
            return np.full(self.n, fill, dtype=np.float64)
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), (self.n,)).copy()
        if np.any(np.isnan(result)):
            raise ValueError("joint-limit vectors must not contain NaN")
        return result

    @property
    def state(self) -> JointMotionState:
        return JointMotionState(
            self._state.q.copy(), self._state.dq.copy(), self._state.ddq.copy()
        )

    def commit(self, state: JointMotionState) -> None:
        q = np.asarray(state.q, dtype=np.float64)
        dq = np.asarray(state.dq, dtype=np.float64)
        ddq = np.asarray(state.ddq, dtype=np.float64)
        if q.shape != (self.n,) or dq.shape != (self.n,) or ddq.shape != (self.n,):
            raise ValueError("handoff state has the wrong joint-vector shape")
        if np.any(q < self.lower - 1e-9) or np.any(q > self.upper + 1e-9):
            raise ValueError("handoff position violates a joint limit")
        if np.any(np.abs(dq) > self.max_velocity + 1e-9):
            raise ValueError("handoff velocity violates its limit")
        if np.any(np.abs(ddq) > self.max_acceleration + 1e-9):
            raise ValueError("handoff acceleration violates its limit")
        self._state = JointMotionState(q.copy(), dq.copy(), ddq.copy())

    @staticmethod
    def _coefficients(state: JointMotionState, q_goal: np.ndarray, duration: float) -> np.ndarray:
        q0, v0, a0 = state.q, state.dq, state.ddq
        t = float(duration)
        delta = q_goal - q0
        return np.stack(
            (
                q0,
                v0,
                0.5 * a0,
                (20.0 * delta - 12.0 * v0 * t - 3.0 * a0 * t**2) / (2.0 * t**3),
                (-30.0 * delta + 16.0 * v0 * t + 3.0 * a0 * t**2) / (2.0 * t**4),
                (12.0 * delta - 6.0 * v0 * t - a0 * t**2) / (2.0 * t**5),
            ),
            axis=0,
        )

    @staticmethod
    def _evaluate(coeff: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, ...]:
        t = np.asarray(times, dtype=np.float64).reshape(-1, 1)
        c0, c1, c2, c3, c4, c5 = coeff
        q = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
        dq = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
        ddq = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2 + 20.0 * c5 * t**3
        jerk = 6.0 * c3 + 24.0 * c4 * t + 60.0 * c5 * t**2
        return q, dq, ddq, jerk

    def plan(
        self,
        q_goal: np.ndarray,
        *,
        sample_dt_s: float,
        sample_count: int,
        handoff_time_s: float,
    ) -> GuardedJointPlan:
        q_goal = np.asarray(q_goal, dtype=np.float64)
        if q_goal.shape != (self.n,) or not np.all(np.isfinite(q_goal)):
            raise ValueError("q_goal has the wrong shape or contains a non-finite value")
        if np.any(q_goal < self.lower) or np.any(q_goal > self.upper):
            raise ValueError("q_goal violates a joint limit")
        sample_dt_s = float(sample_dt_s)
        sample_count = int(sample_count)
        handoff_time_s = float(handoff_time_s)
        if sample_dt_s <= 0.0 or sample_count < 1:
            raise ValueError("sample_dt_s and sample_count must be positive")
        if not 0.0 < handoff_time_s <= sample_count * sample_dt_s:
            raise ValueError("handoff_time_s lies outside the published horizon")

        duration = max(self.min_duration_s, sample_count * sample_dt_s)
        while True:
            coeff = self._coefficients(self._state, q_goal, duration)
            dense_t = np.linspace(0.0, duration, 2001)
            q_all, dq_all, ddq_all, jerk_all = self._evaluate(coeff, dense_t)
            peak_v = np.max(np.abs(dq_all), axis=0)
            peak_a = np.max(np.abs(ddq_all), axis=0)
            peak_j = np.max(np.abs(jerk_all), axis=0)
            if (
                np.all(peak_v <= self.max_velocity * (1.0 + 1e-9))
                and np.all(peak_a <= self.max_acceleration * (1.0 + 1e-9))
                and np.all(peak_j <= self.max_jerk * (1.0 + 1e-9))
                and np.all(q_all >= self.lower - 1e-9)
                and np.all(q_all <= self.upper + 1e-9)
            ):
                break
            duration *= 1.25
            if duration > self.max_duration_s + 1e-12:
                raise RuntimeError(
                    "could not fit a quintic within limits before "
                    f"max_duration_s={self.max_duration_s:g}"
                )

        sample_times = sample_dt_s * np.arange(1, sample_count + 1, dtype=np.float64)
        q_s, dq_s, ddq_s, _ = self._evaluate(coeff, sample_times)
        q_h, dq_h, ddq_h, _ = self._evaluate(coeff, np.array([handoff_time_s]))
        return GuardedJointPlan(
            samples=tuple(
                (q_s[i].copy(), float(sample_times[i]), dq_s[i].copy(), ddq_s[i].copy())
                for i in range(sample_count)
            ),
            handoff_state=JointMotionState(q_h[0].copy(), dq_h[0].copy(), ddq_h[0].copy()),
            duration_s=duration,
            peak_velocity=peak_v,
            peak_acceleration=peak_a,
            peak_jerk=peak_j,
        )
