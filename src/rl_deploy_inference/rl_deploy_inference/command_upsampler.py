"""Republish 15 Hz policy setpoints at FRI rate."""

from __future__ import annotations

import argparse
from collections import deque
import statistics

import rclpy
from lbr_fri_idl.msg import LBRJointPositionCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class CommandUpsampler(Node):
    """Zero-order hold or interpolate low-rate joint-position commands."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("command_upsampler")
        self._interp = args.interpolate
        self._configured_input_period = 1.0 / float(args.input_hz)
        self._input_period = self._configured_input_period
        self._adaptive_period = not getattr(args, "fixed_input_period", False)
        self._minimum_input_period = 2.0 / float(args.rate_hz)
        self._observed_periods: deque[float] = deque(maxlen=5)
        self._last_command_t: float | None = None
        self._holding = False
        self._hold_t = 0.0

        self._target: list[float] | None = None
        self._ramp_start: list[float] | None = None
        self._target_t = 0.0
        self._output: list[float] | None = None

        self.sub = self.create_subscription(
            LBRJointPositionCommand,
            args.input_topic,
            self._on_cmd,
            qos_profile_sensor_data,
        )
        self.hold_sub = self.create_subscription(
            LBRJointPositionCommand,
            args.hold_topic,
            self._on_hold,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(LBRJointPositionCommand, args.output_topic, 1)
        self.create_timer(1.0 / float(args.rate_hz), self._tick)
        self.get_logger().info(
            f"upsampling {args.input_topic} ({args.input_hz} Hz) -> {args.output_topic} "
            f"@ {args.rate_hz} Hz ({'adaptive ramp' if self._interp and self._adaptive_period else 'ramp' if self._interp else 'zero-order hold'})"
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd(self, msg: LBRJointPositionCommand) -> None:
        now = self._now()
        if self._holding:
            stale_guard = max(self._minimum_input_period, 0.5 * self._input_period)
            if now - self._hold_t < stale_guard:
                return
            self._holding = False
        self._observe_input_period(now)
        new = list(msg.joint_position)
        self._ramp_start = list(self._output) if self._output is not None else new
        self._target = new
        self._target_t = now

    def _on_hold(self, msg: LBRJointPositionCommand) -> None:
        now = self._now()
        hold = list(msg.joint_position)
        self._holding = True
        self._hold_t = now
        self._target = hold
        self._ramp_start = hold
        self._output = hold
        self._target_t = now
        output = LBRJointPositionCommand()
        output.joint_position = hold
        self.pub.publish(output)

    def _observe_input_period(self, now: float) -> None:
        if not self._adaptive_period:
            return
        if self._last_command_t is not None:
            observed = now - self._last_command_t
            if self._minimum_input_period <= observed <= 0.5:
                self._observed_periods.append(observed)
                self._input_period = statistics.median(self._observed_periods)
            elif observed > 0.5:
                # A paused source is a new stream, not a half-second interpolation.
                self._observed_periods.clear()
                self._input_period = self._configured_input_period
        self._last_command_t = now

    def _tick(self) -> None:
        if self._target is None:
            return
        if self._interp and self._ramp_start is not None:
            alpha = min(1.0, max(0.0, (self._now() - self._target_t) / self._input_period))
            out = [s + alpha * (t - s) for s, t in zip(self._ramp_start, self._target)]
        else:
            out = list(self._target)
        self._output = out
        msg = LBRJointPositionCommand()
        msg.joint_position = out
        self.pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsample RL 15 Hz joint commands to FRI rate.")
    parser.add_argument("--input-topic", default="/rl_deploy/command_15hz")
    parser.add_argument("--hold-topic", default="/rl_deploy/hold")
    parser.add_argument(
        "--output-topic",
        default="/lbr_dual_arm_y_gripper/command/joint_position",
    )
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--input-hz", type=float, default=15.0)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument(
        "--fixed-input-period",
        action="store_true",
        help="Use exactly 1/input-hz for every ramp instead of following observed command cadence.",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = CommandUpsampler(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
