#!/usr/bin/env python3
"""Upsample the RL node's 15 Hz LBRJointPositionCommand to FRI rate so FRI isn't starved.

The LBR FRI needs a fresh joint command every control cycle (~100 Hz+); the RL policy only produces
one at 15 Hz. Feeding 15 Hz straight into the command controller starves FRI between setpoints -> the
gear-grinding rattle. This node sits in between:

    RL node --(15 Hz)--> /rl_deploy/command_15hz --[this]--(200 Hz)--> command/joint_position --> FRI

Default is zero-order hold (republish the latest setpoint every tick -> no added latency, FRI always
fed). `--interpolate` linearly ramps from the previous output to each new setpoint over one input
period, killing the residual 15 Hz step jumps (costs up to ~one input period of latency).

Run it after sourcing ROS + the deploy workspace (needs lbr_fri_idl):
    python3 command_upsampler.py                 # zero-order hold @ 200 Hz
    python3 command_upsampler.py --interpolate    # ramped (smoother)
"""
from __future__ import annotations

import argparse

import rclpy
from lbr_fri_idl.msg import LBRJointPositionCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class CommandUpsampler(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("command_upsampler")
        self._interp = args.interpolate
        self._input_period = 1.0 / float(args.input_hz)

        self._target: list[float] | None = None   # latest received setpoint
        self._ramp_start: list[float] | None = None
        self._target_t: float = 0.0
        self._output: list[float] | None = None    # last published value

        self.sub = self.create_subscription(
            LBRJointPositionCommand, args.input_topic, self._on_cmd, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(LBRJointPositionCommand, args.output_topic, 1)
        self.create_timer(1.0 / float(args.rate_hz), self._tick)
        self.get_logger().info(
            f"upsampling {args.input_topic} ({args.input_hz} Hz) -> {args.output_topic} "
            f"@ {args.rate_hz} Hz ({'ramp' if self._interp else 'zero-order hold'})"
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd(self, msg: LBRJointPositionCommand) -> None:
        new = list(msg.joint_position)
        self._ramp_start = list(self._output) if self._output is not None else new
        self._target = new
        self._target_t = self._now()

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
    parser.add_argument("--output-topic", default="/lbr_dual_arm_y_gripper/command/joint_position")
    parser.add_argument("--rate-hz", type=float, default=200.0, help="FRI-side output rate (>= FRI cycle).")
    parser.add_argument("--input-hz", type=float, default=15.0, help="Expected RL command rate (for ramp timing).")
    parser.add_argument("--interpolate", action="store_true", help="Linear ramp between setpoints (smoother).")
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
