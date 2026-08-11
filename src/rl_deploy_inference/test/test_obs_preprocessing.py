from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_deploy_inference.obs_preprocessing import (  # noqa: E402
    ForceSmoother,
    FrameStacker,
    ObsPreprocessConfig,
    _fov_match_crop,
    build_policy_vector,
    preprocess_rgbd,
)


def test_preprocess_rgbd_matches_sim_order_and_scaling() -> None:
    cfg = ObsPreprocessConfig(image_height=2, image_width=2, depth_far_m=0.3)
    rgb = np.array(
        [
            [[0, 128, 255], [255, 128, 0]],
            [[64, 64, 64], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[np.inf, -1.0], [0.15, 0.6]], dtype=np.float32)

    out = preprocess_rgbd(rgb, depth, cfg)

    rgb01 = rgb.astype(np.float32) / 255.0
    expected_rgb = rgb01 - rgb01.mean(axis=(0, 1), keepdims=True)
    expected_depth = np.array([[0.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(out[:, :, :3], expected_rgb, atol=1e-6)
    np.testing.assert_allclose(out[:, :, 3], expected_depth, atol=1e-6)
    assert out.dtype == np.float32


def test_frame_stacker_uses_oldest_to_newest_channel_order() -> None:
    cfg = ObsPreprocessConfig(image_height=1, image_width=1, frame_stack=3)
    stacker = FrameStacker(cfg)

    one = np.ones((1, 1, 4), dtype=np.float32)
    two = 2.0 * one
    three = 3.0 * one

    np.testing.assert_allclose(stacker.push(one), np.concatenate([one, one, one], axis=2))
    np.testing.assert_allclose(stacker.push(two), np.concatenate([one, one, two], axis=2))
    np.testing.assert_allclose(stacker.push(three), np.concatenate([one, two, three], axis=2))


def test_fov_match_crop_hits_sim_focal_length() -> None:
    # Real D405 color intrinsics + sim pinhole -> the crop side must make the post-resize focal match.
    cfg = ObsPreprocessConfig(image_height=224, image_width=224, fov_match=True)
    intr = (437.42, 436.81, 427.52, 238.64)  # fx, fy, cx, cy for 848x480
    rgb = np.zeros((480, 848, 3), dtype=np.uint8)
    depth = np.zeros((480, 848), dtype=np.float32)

    rgb_c, depth_c = _fov_match_crop(rgb, depth, cfg, intr)

    f_out = (cfg.sim_focal_length_mm / cfg.sim_horizontal_aperture_mm) * cfg.image_width
    assert rgb_c.shape[:2] == depth_c.shape[:2]
    # Effective focal after resizing the crop back to 224 must match sim's within ~1 px.
    eff_fx = intr[0] * cfg.image_width / rgb_c.shape[1]
    assert abs(eff_fx - f_out) <= 1.0
    # And it is a genuine crop of the wide image, not the full frame.
    assert rgb_c.shape[1] < 848


def test_fov_match_crop_is_noop_without_intrinsics() -> None:
    cfg = ObsPreprocessConfig(fov_match=True)
    rgb = np.zeros((480, 848, 3), dtype=np.uint8)
    depth = np.zeros((480, 848), dtype=np.float32)
    rgb_c, depth_c = _fov_match_crop(rgb, depth, cfg, None)
    assert rgb_c.shape == rgb.shape and depth_c.shape == depth.shape


def test_force_smoother_starts_from_zero_like_sim_reset() -> None:
    smoother = ForceSmoother(alpha=0.25)

    np.testing.assert_allclose(smoother.update([4.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(smoother.update([4.0, 0.0, 0.0]), [1.75, 0.0, 0.0])


def test_policy_vector_order_is_e2e_sim_order() -> None:
    obs = build_policy_vector(
        fingertip_pos=np.array([1.0, 2.0, 3.0]),
        socket_pos_estimate=np.array([0.1, 0.2, 0.3]),
        fingertip_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        ee_linvel=np.array([4.0, 5.0, 6.0]),
        ee_angvel=np.array([7.0, 8.0, 9.0]),
        ft_force=np.array([10.0, 11.0, 12.0]),
        prev_action=np.array([13.0, 14.0, 15.0, 16.0, 17.0]),
    )

    np.testing.assert_allclose(
        obs,
        [
            0.9,
            1.8,
            2.7,
            1.0,
            0.0,
            0.0,
            0.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
        ],
        atol=1e-6,
    )
