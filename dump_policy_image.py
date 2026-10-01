#!/usr/bin/env python3
"""Dump the EXACT 320x180 RGB image the policy receives, for sim-vs-real comparison.

Subscribes to the live RGB topic + camera_info, runs the deploy preprocessing
(preprocess_rgb_we_a: optional fov-match crop -> resize 320x180 -> ImageNet-RGB
normalization is done in the actor, so here we save the pre-normalization 320x180
RGB that the CNN sees, plus the raw frame for reference), and writes PNGs to repo root.

Usage (sourced deploy env):
    python3 dump_policy_image.py --rgb /realsense_1/camera/color/image_raw \
        --info /realsense_1/camera/color/camera_info --fov-match false

Compare policy_image_320x180.png against a sim render at a matched pose.
"""
from __future__ import annotations

import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import qos_profile_sensor_data

from rl_deploy_inference.obs_preprocessing_we_a import (
    WeAObsPreprocessConfig,
    preprocess_rgb_we_a,
)


def _img_to_np(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding in ("bgr8", "bgra8"):
        arr = arr[:, :, ::-1] if arr.shape[2] == 3 else arr[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(arr[:, :, :3])


class Dumper(Node):
    def __init__(self, args):
        super().__init__("dump_policy_image")
        self.args = args
        self.intr = None
        self.done = False
        self.create_subscription(CameraInfo, args.info, self._info, qos_profile_sensor_data)
        self.create_subscription(Image, args.rgb, self._rgb, qos_profile_sensor_data)
        self.get_logger().info(f"waiting for {args.rgb} + {args.info} ...")

    def _info(self, msg: CameraInfo):
        self.intr = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx,fy,cx,cy

    def _rgb(self, msg: Image):
        if self.done or self.intr is None:
            return
        import cv2
        raw = _img_to_np(msg)
        cfg = WeAObsPreprocessConfig(
            image_height=180, image_width=320, image_channels=3,
            fov_match=self.args.fov_match,
        )
        proc = preprocess_rgb_we_a(raw.astype(np.float32), cfg, self.intr)  # HxWx3 float [0,1] or [0,255]
        # normalize to 0-255 uint8 for viewing
        p = proc.astype(np.float32)
        if p.max() <= 1.5:
            p = p * 255.0
        p = np.clip(p, 0, 255).astype(np.uint8)
        cv2.imwrite("policy_image_320x180.png", p[:, :, ::-1])   # save BGR for cv2
        cv2.imwrite("policy_image_raw_full.png", raw[:, :, ::-1])
        self.get_logger().info(
            f"WROTE policy_image_320x180.png (fov_match={self.args.fov_match}) "
            f"and policy_image_raw_full.png  intr fx,fy,cx,cy={self.intr}"
        )
        self.done = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default="/realsense_1/camera/color/image_raw")
    ap.add_argument("--info", default="/realsense_1/camera/color/camera_info")
    ap.add_argument("--fov-match", dest="fov_match", default="false",
                    type=lambda s: s.lower() in ("1", "true", "yes"))
    args = ap.parse_args()
    rclpy.init()
    node = Dumper(args)
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
