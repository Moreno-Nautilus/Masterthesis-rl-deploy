#!/usr/bin/env python3
"""Preview the FOV crop the policy will see. Needs ONLY a camera (no robot) -- or a saved frame.

The real D405 color view is ~88 deg wide; sim is a ~58 deg square. The node center-crops the color +
aligned-depth to the sim FOV before the 224 resize. This tool shows exactly that cropped 224 image so
you can eyeball it against the sim wrist renders.

OFFLINE (works now, no ROS, no camera) -- run the crop on a saved color frame + its intrinsics:
    python3 preview_camera_crop.py --image color.png --fx 437.42 --fy 436.81 --cx 427.52 --cy 238.64

LIVE (needs the RealSense node running, no robot) -- read the color + camera_info topics:
    python3 preview_camera_crop.py --ros \\
        --rgb-topic /realsense_2/camera/color/image_rect \\
        --info-topic /realsense_2/camera/color/camera_info

Both write <out>_crop.png (the 224x224 view the CNN gets) and print the raw size, crop box, and the
horizontal FOV before/after.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_deploy_inference.obs_preprocessing import ObsPreprocessConfig, _fov_match_crop  # noqa: E402


def _hfov_deg(fx: float, width: int) -> float:
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def _crop_and_resize(rgb: np.ndarray, intr: tuple, cfg: ObsPreprocessConfig, out: str) -> None:
    import cv2

    h, w = rgb.shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)  # dummy; the same box crops the aligned depth live
    rgb_c, _ = _fov_match_crop(rgb, depth, cfg, intr)
    resized = cv2.resize(rgb_c, (cfg.image_width, cfg.image_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{out}_crop.png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    fx = intr[0]
    print(f"raw {w}x{h} (HFOV {_hfov_deg(fx, w):.1f} deg) -> crop {rgb_c.shape[1]}x{rgb_c.shape[0]} "
          f"(HFOV {_hfov_deg(fx, rgb_c.shape[1]):.1f} deg) -> {cfg.image_width}x{cfg.image_height}")
    print(f"wrote {out}_crop.png  (sim target ~57.7 deg square)")


def run_offline(args) -> int:
    import cv2

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"could not read {args.image}")
        return 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cfg = ObsPreprocessConfig(fov_match=True)
    _crop_and_resize(rgb, (args.fx, args.fy, args.cx, args.cy), cfg, args.out)
    return 0


def run_ros(args) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    cfg = ObsPreprocessConfig(fov_match=True)

    class Preview(Node):
        def __init__(self):
            super().__init__("preview_camera_crop")
            self.intr = None
            self.create_subscription(CameraInfo, args.info_topic, self._on_info, qos_profile_sensor_data)
            self.create_subscription(Image, args.rgb_topic, self._on_rgb, qos_profile_sensor_data)

        def _on_info(self, m):
            self.intr = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]))

        def _on_rgb(self, m):
            if self.intr is None:
                self.get_logger().info("waiting for camera_info...", throttle_duration_sec=2.0)
                return
            enc = m.encoding.lower()
            arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width * 3]
            arr = arr.reshape(m.height, m.width, 3)
            rgb = arr[:, :, ::-1] if enc.startswith("bgr") else arr
            _crop_and_resize(np.ascontiguousarray(rgb), self.intr, cfg, args.out)
            self.get_logger().info("wrote preview; shutting down.")
            rclpy.shutdown()

    rclpy.init()
    rclpy.spin(Preview())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="offline: a saved color frame (png/jpg)")
    ap.add_argument("--fx", type=float, default=437.42)
    ap.add_argument("--fy", type=float, default=436.81)
    ap.add_argument("--cx", type=float, default=427.52)
    ap.add_argument("--cy", type=float, default=238.64)
    ap.add_argument("--ros", action="store_true", help="live: read camera topics instead of a file")
    ap.add_argument("--rgb-topic", default="/realsense_2/camera/color/image_rect")
    ap.add_argument("--info-topic", default="/realsense_2/camera/color/camera_info")
    ap.add_argument("--out", default="/tmp/cam_preview")
    args = ap.parse_args()
    if args.ros:
        return run_ros(args)
    if not args.image:
        ap.error("give --image <file> (offline) or --ros (live)")
    return run_offline(args)


if __name__ == "__main__":
    raise SystemExit(main())
