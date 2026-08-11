#!/usr/bin/env python3
"""Inspect a deploy obs dump (written via the node's `obs_dump_path` param).

Prints the 21-D policy vector split into its sim-order blocks (with plausible-range flags) and saves
the RGB + depth channels of the (last) stacked frame as PNGs so you can eyeball framing/FOV against
the sim wrist renders. This is the obs-parity gate you run BEFORE enabling motion.

    python3 inspect_obs_dump.py /path/to/deploy_obs.npz --out /tmp/obs

The npz holds `policy` (21,) and `image` (H, W, 4*frame_stack), exactly as the actor consumes them.
"""
from __future__ import annotations

import argparse

import numpy as np


BLOCKS = [
    ("fingertip_pos - socket_opening (m)", 3, 0.15),
    ("fingertip_quat wxyz", 4, 1.01),
    ("ee_linvel (m/s)", 3, 2.0),
    ("ee_angvel (rad/s) [x,y must be 0]", 3, 5.0),
    ("ft_force (N)", 3, 50.0),
    ("prev_action [-1,1]", 5, 1.01),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="/tmp/obs", help="prefix for the saved rgb/depth PNGs")
    args = ap.parse_args()

    data = np.load(args.npz)
    policy = np.asarray(data["policy"], dtype=np.float64).reshape(-1)
    image = np.asarray(data["image"], dtype=np.float32)
    print(f"policy shape={policy.shape}  image shape={image.shape}")
    if policy.shape[0] != 21:
        print(f"  !! expected 21-D policy, got {policy.shape[0]}")

    i = 0
    for name, n, lim in BLOCKS:
        vals = policy[i : i + n]
        flag = "" if np.all(np.abs(vals) <= lim) else "   <-- OUT OF RANGE"
        print(f"  {name:38s} {np.round(vals, 4)}{flag}")
        i += n
    # explicit ee_angvel roll/pitch-zero parity check
    ang = policy[10:12]
    print("  ee_angvel[x,y] == 0 (Forge parity):", "OK" if np.allclose(ang, 0.0) else f"FAIL {ang}")

    # Save the newest frame's channels (image is oldest->newest on the channel axis, 4 ch/frame).
    frame = image[:, :, -4:]
    rgb = frame[:, :, :3]
    depth = frame[:, :, 3]
    try:
        import cv2

        rgb_u8 = np.clip((rgb - rgb.min()) / (rgb.ptp() + 1e-9) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{args.out}_rgb.png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{args.out}_depth.png", (np.clip(depth, 0, 1) * 255).astype(np.uint8))
        print(f"  wrote {args.out}_rgb.png and {args.out}_depth.png")
    except ImportError:
        print("  (install opencv to save PNGs)")
    print(f"  depth[0,1] frac non-zero={np.mean(depth > 0):.2f}  rgb mean~0 check={np.round(rgb.mean(),4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
