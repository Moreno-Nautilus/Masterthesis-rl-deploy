#!/usr/bin/env python3
"""Offline hole-detector tuner: run ``hole_detector`` on a captured raw frame (no ROS, no robot).

The node dumps ``raw_rgb_<ts>.png`` + ``raw_depth_mm_<ts>.png`` + ``intrinsics_<ts>.txt`` to its
debug_dir. Point this at that timestamp to iterate on the detector params against the EXACT frame the
robot saw, then copy the winning params into config/hole_align.yaml.

Examples:
  # newest capture in /tmp/hole_align, contour method:
  python3 tune_hole_detection.py --dir /tmp/hole_align
  # a specific capture, tweak a couple of knobs:
  python3 tune_hole_detection.py --dir /tmp/hole_align --stamp 20260812_165158 \
      --set min_circularity=0.6 sat_min=50 depth_valid_max_m=0.14
  # compare the legacy Hough detector:
  python3 tune_hole_detection.py --dir /tmp/hole_align --set method=hough

Writes ``tuned_<ts>.png`` (annotated) next to the inputs and prints every candidate.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# Allow running straight from the source tree (no colcon install needed).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl_deploy_inference import hole_detector as hd  # noqa: E402

import cv2  # noqa: E402


def _newest_stamp(ddir: str) -> str:
    files = sorted(glob.glob(os.path.join(ddir, "raw_rgb_*.png")))
    if not files:
        sys.exit(f"no raw_rgb_*.png in {ddir} -- run the node once with debug:=true to capture a frame.")
    return os.path.basename(files[-1])[len("raw_rgb_"):-len(".png")]


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/tmp/hole_align", help="debug_dir with the raw_* captures.")
    ap.add_argument("--stamp", default=None, help="capture timestamp (default: newest).")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value", help="override HoleDetectorConfig fields.")
    args = ap.parse_args()

    stamp = args.stamp or _newest_stamp(args.dir)
    rgb_bgr = cv2.imread(os.path.join(args.dir, f"raw_rgb_{stamp}.png"), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(os.path.join(args.dir, f"raw_depth_mm_{stamp}.png"), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth_mm is None:
        sys.exit(f"missing raw_rgb/raw_depth for stamp {stamp} in {args.dir}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth_m = depth_mm.astype(np.float32) / 1000.0

    intr_path = os.path.join(args.dir, f"intrinsics_{stamp}.txt")
    if os.path.exists(intr_path):
        fx, fy, cx, cy = (float(x) for x in open(intr_path).read().split())
    else:
        h, w = rgb.shape[:2]
        fx = fy = 0.5 * w  # rough fallback
        cx, cy = w / 2.0, h / 2.0
        print(f"[warn] no intrinsics_{stamp}.txt; using rough fallback fx={fx:.0f}")
    intr = hd.CameraIntrinsics(fx, fy, cx, cy)

    cfg = hd.HoleDetectorConfig()
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if not hasattr(cfg, k):
            sys.exit(f"unknown config field {k!r}; fields: {list(cfg.__dataclass_fields__)}")
        setattr(cfg, k, _coerce(v))

    ref = hd.reference_depth(depth_m, cfg)
    dets = hd.detect_holes(rgb, depth_m, intr, cfg, ref_depth_m=ref)
    chosen = hd.select_hole(dets, intr)

    print(f"stamp={stamp}  method={cfg.method}  intrinsics fx={intr.fx:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
    print(f"ref_depth={ref*1000:.0f} mm  candidates={len(dets)}")
    for i, d in enumerate(dets):
        tag = " <== CHOSEN" if d is chosen else (" [paired]" if d.paired else "")
        print(f"  [{i}] uv=({d.u:.0f},{d.v:.0f}) r={d.radius_px:.1f}px z={d.depth_m*1000:.0f}mm "
              f"circ={d.circularity:.2f} r_err={d.radius_err_frac*100:.0f}% cam={np.round(d.point_cam,4).tolist()}{tag}")

    mask = hd.part_mask(rgb, depth_m, cfg) if cfg.method == "contour" else None
    out = hd.render_debug(rgb, dets, chosen, mask=mask)
    out_path = os.path.join(args.dir, f"tuned_{stamp}.png")
    cv2.imwrite(out_path, out)
    if mask is not None:
        cv2.imwrite(os.path.join(args.dir, f"mask_{stamp}.png"), mask)
    print(f"wrote {out_path}" + (f" and mask_{stamp}.png" if mask is not None else ""))


if __name__ == "__main__":
    main()
