"""Standalone observation preprocessing ported from the Isaac insertion env."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsPreprocessConfig:
    image_height: int = 224
    image_width: int = 224
    image_channels: int = 4
    frame_stack: int = 1
    depth_far_m: float = 0.3
    ft_smoothing_factor: float = 0.25
    force_noise_std: float = 0.0
    # FOV matching: the sim camera is a 224x224 pinhole (focal_length_mm / horizontal_aperture_mm),
    # HFOV ~= 57.7 deg. The real D405 color stream is much wider (~88 deg), so a plain resize would
    # feed the policy a wider, aspect-distorted scene. When fov_match is on AND real intrinsics are
    # supplied, we first center-crop to the window whose post-resize focal length equals sim's.
    fov_match: bool = True
    sim_focal_length_mm: float = 19.0
    sim_horizontal_aperture_mm: float = 20.955


def _fov_match_crop(
    rgb: np.ndarray, depth: np.ndarray, cfg: ObsPreprocessConfig, intrinsics: tuple | None
):
    """Center-crop RGB + aligned depth to the sim camera FOV before the 224 resize.

    ``intrinsics = (fx, fy, cx, cy)`` are the REAL color intrinsics (from camera_info);
    ``aligned_depth_to_color`` shares them, so the SAME pixel box applies to both. The crop side is
    chosen so that, after resizing to the sim output size, the effective focal length equals sim's
    ``f_out = (focal_length_mm / horizontal_aperture_mm) * out_width``. No-op if fov_match is off or
    intrinsics are missing/invalid (falls back to the plain resize, with a wider-than-sim FOV).
    """
    if not cfg.fov_match or intrinsics is None:
        return rgb, depth
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    if fx <= 0.0 or fy <= 0.0 or cfg.sim_horizontal_aperture_mm <= 0.0:
        return rgb, depth
    f_out = (cfg.sim_focal_length_mm / cfg.sim_horizontal_aperture_mm) * cfg.image_width
    if f_out <= 0.0:
        return rgb, depth
    h, w = rgb.shape[:2]
    crop_w = min(int(round(fx * cfg.image_width / f_out)), w)
    crop_h = min(int(round(fy * cfg.image_height / f_out)), h)
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x0 = max(0, min(x0, w - crop_w))
    y0 = max(0, min(y0, h - crop_h))
    rgb_c = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    depth_c = depth[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return rgb_c, depth_c


def _resize_if_needed(image: np.ndarray, height: int, width: int, interpolation: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - only used when live image size differs.
        raise RuntimeError("OpenCV is required when RGB-D frames are not already 224x224.") from exc
    return cv2.resize(image, (width, height), interpolation=interpolation)


def preprocess_rgbd(
    rgb: np.ndarray, depth_m: np.ndarray, cfg: ObsPreprocessConfig, intrinsics: tuple | None = None
) -> np.ndarray:
    """Return HWC float32 RGB-D exactly as the actor expects before rl_games normalization.

    Sim provenance: insertion_env.py::_get_camera_image with deploy DR/noise off:
    RGB->[0,1], per-image mean subtract; depth inf->0, clamp [0, far], divide by far;
    concatenate RGB + depth. When ``intrinsics=(fx,fy,cx,cy)`` is given, the frames are first
    center-cropped to the sim FOV (see ``_fov_match_crop``) so the resize doesn't feed a wider view.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb must be HxWx3/4, got {rgb.shape}")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW or HxWx1, got {depth.shape}")

    rgb, depth = _fov_match_crop(rgb[:, :, :3], depth, cfg, intrinsics)

    try:
        import cv2

        rgb_interp = cv2.INTER_AREA
        depth_interp = cv2.INTER_NEAREST
    except ImportError:  # pragma: no cover - exact-size tests do not need cv2.
        rgb_interp = 3
        depth_interp = 0

    rgb = _resize_if_needed(rgb[:, :, :3], cfg.image_height, cfg.image_width, rgb_interp)
    depth = _resize_if_needed(depth, cfg.image_height, cfg.image_width, depth_interp)

    rgb01 = rgb.astype(np.float32)
    if rgb01.max(initial=0.0) > 1.5:
        rgb01 *= 1.0 / 255.0
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    rgb_policy = rgb01 - rgb01.mean(axis=(0, 1), keepdims=True)

    depth = depth.astype(np.float32, copy=True)
    depth[~np.isfinite(depth)] = 0.0
    depth = np.clip(depth, 0.0, float(cfg.depth_far_m)) / float(cfg.depth_far_m)
    return np.concatenate([rgb_policy, depth[:, :, None]], axis=2).astype(np.float32, copy=False)


class FrameStacker:
    """Temporal stack in oldest->newest channel order, matching InsertionEnv._stack_frames."""

    def __init__(self, cfg: ObsPreprocessConfig):
        self.cfg = cfg
        self._history: list[np.ndarray] = []

    def reset(self) -> None:
        self._history.clear()

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        n = max(1, int(self.cfg.frame_stack))
        if n <= 1:
            return frame
        if not self._history:
            self._history = [frame.copy() for _ in range(n)]
        else:
            self._history.append(frame.copy())
            self._history = self._history[-n:]
            while len(self._history) < n:
                self._history.insert(0, frame.copy())
        return np.concatenate(self._history, axis=2).astype(np.float32, copy=False)


class ForceSmoother:
    """EMA for wrist F/T, mirroring Forge's ft_smoothing_factor path."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._smooth: np.ndarray | None = None

    def reset(self) -> None:
        self._smooth = None

    def update(self, force_xyz: np.ndarray) -> np.ndarray:
        force = np.asarray(force_xyz, dtype=np.float32).reshape(3)
        if self._smooth is None:
            self._smooth = np.zeros(3, dtype=np.float32)
        self._smooth = self.alpha * force + (1.0 - self.alpha) * self._smooth
        return self._smooth.astype(np.float32, copy=True)


def build_policy_vector(
    fingertip_pos: np.ndarray,
    socket_pos_estimate: np.ndarray,
    fingertip_quat_wxyz: np.ndarray,
    ee_linvel: np.ndarray,
    ee_angvel: np.ndarray,
    ft_force: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Concatenate the 21-D E2E policy vector in sim order."""
    parts = [
        np.asarray(fingertip_pos, dtype=np.float32).reshape(3)
        - np.asarray(socket_pos_estimate, dtype=np.float32).reshape(3),
        np.asarray(fingertip_quat_wxyz, dtype=np.float32).reshape(4),
        np.asarray(ee_linvel, dtype=np.float32).reshape(3),
        np.asarray(ee_angvel, dtype=np.float32).reshape(3),
        np.asarray(ft_force, dtype=np.float32).reshape(3),
        np.asarray(prev_action, dtype=np.float32).reshape(5),
    ]
    out = np.concatenate(parts).astype(np.float32, copy=False)
    if out.shape != (21,):
        raise AssertionError(f"policy vector must be 21-D, got {out.shape}")
    return out


def make_actor_obs(policy: np.ndarray, image: np.ndarray) -> dict[str, np.ndarray]:
    policy = np.asarray(policy, dtype=np.float32).reshape(21)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image must be HWC, got {image.shape}")
    return {"policy": policy[None, :], "image": image[None, :, :, :]}

