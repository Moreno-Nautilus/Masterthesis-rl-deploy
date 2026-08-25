from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_deploy_inference.obs_preprocessing_we_a import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    WeAObsPreprocessConfig,
    build_policy_vector_we_a,
    preprocess_rgb_we_a,
)


def test_rgb_uses_imagenet_normalization_without_mean_subtraction() -> None:
    cfg = WeAObsPreprocessConfig(image_height=1, image_width=2, fov_match=False)
    rgb = np.array([[[0, 128, 255], [255, 64, 0]]], dtype=np.uint8)

    actual = preprocess_rgb_we_a(rgb, cfg)

    expected = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (1, 2, 3)
    assert actual.dtype == np.float32


def test_policy_vector_is_exact_we_a_order() -> None:
    actual = build_policy_vector_we_a(
        goal_delta_tcp=np.arange(1, 7),
        ft_force_tcp=np.arange(7, 10),
        prev_action=np.arange(10, 16),
    )
    np.testing.assert_allclose(actual, np.arange(1, 16, dtype=np.float32))
    assert actual.shape == (15,)
