"""Observation preprocessing for the we_A_resnet18_baseline policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class WeAObsPreprocessConfig:
    image_height: int = 180
    image_width: int = 320
    image_channels: int = 3
    frame_stack: int = 1
    ft_smoothing_factor: float = 0.25
    force_noise_std: float = 0.0
    fov_match: bool = True
    sim_focal_length_mm: float = 11.0
    sim_horizontal_aperture_mm: float = 20.955


def _fov_match_crop_rgb(
    rgb: np.ndarray,
    cfg: WeAObsPreprocessConfig,
    intrinsics: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Crop a real RGB frame to the pinhole FOV used by the training camera."""
    if not cfg.fov_match or intrinsics is None:
        return rgb
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    if fx <= 0.0 or fy <= 0.0 or cfg.sim_horizontal_aperture_mm <= 0.0:
        return rgb
    f_out = (cfg.sim_focal_length_mm / cfg.sim_horizontal_aperture_mm) * cfg.image_width
    if f_out <= 0.0:
        return rgb
    h, w = rgb.shape[:2]
    crop_w = min(max(1, int(round(fx * cfg.image_width / f_out))), w)
    crop_h = min(max(1, int(round(fy * cfg.image_height / f_out))), h)
    x0 = max(0, min(int(round(cx - crop_w / 2.0)), w - crop_w))
    y0 = max(0, min(int(round(cy - crop_h / 2.0)), h - crop_h))
    return rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return rgb
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - live frames normally require OpenCV.
        raise RuntimeError("OpenCV is required to resize the we_A RGB observation.") from exc
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_rgb_we_a(
    rgb: np.ndarray,
    cfg: WeAObsPreprocessConfig,
    intrinsics: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Return the HWC ImageNet-normalized RGB tensor expected by we_A."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb must be HxWx3/4, got {rgb.shape}")
    rgb = _fov_match_crop_rgb(rgb[:, :, :3], cfg, intrinsics)
    rgb = _resize_rgb(rgb, cfg.image_height, cfg.image_width)
    rgb01 = rgb.astype(np.float32)
    if rgb01.max(initial=0.0) > 1.5:
        rgb01 *= 1.0 / 255.0
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return ((rgb01 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32, copy=False)


class WeAFrameStacker:
    """Frame stack kept explicit even though we_A is fixed to one RGB frame."""

    def __init__(self, cfg: WeAObsPreprocessConfig):
        self.cfg = cfg
        self._history: list[np.ndarray] = []

    def reset(self) -> None:
        self._history.clear()

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        n = max(1, int(self.cfg.frame_stack))
        if n == 1:
            return frame
        if not self._history:
            self._history = [frame.copy() for _ in range(n)]
        else:
            self._history = (self._history + [frame.copy()])[-n:]
        return np.concatenate(self._history, axis=2).astype(np.float32, copy=False)


def build_policy_vector_we_a(
    goal_delta_tcp: np.ndarray,
    ft_force_tcp: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Build we_A's 15-D [goal_delta_tcp, force_tcp, previous_action] vector."""
    out = np.concatenate(
        [
            np.asarray(goal_delta_tcp, dtype=np.float32).reshape(6),
            np.asarray(ft_force_tcp, dtype=np.float32).reshape(3),
            np.asarray(prev_action, dtype=np.float32).reshape(6),
        ]
    ).astype(np.float32, copy=False)
    if out.shape != (15,):
        raise AssertionError(f"we_A policy vector must be 15-D, got {out.shape}")
    return out


def make_actor_obs_we_a(policy: np.ndarray, image: np.ndarray) -> dict[str, np.ndarray]:
    policy = np.asarray(policy, dtype=np.float32).reshape(15)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"we_A image must be HxWx3, got {image.shape}")
    return {"policy": policy[None, :], "image": image[None, :, :, :]}
