"""Vision-based cooling-base HOLE (socket-opening) localizer -- pure, ROS-free, unit-testable.

The perception pipeline publishes the *tracked CAD centre* of ``cooling_base`` (FoundationPose), which
is ~1 cm off physically -- enough to defeat the 1 mm-clearance socket. This module localizes the actual
socket OPENING directly in the wrist D405 image, then deprojects it to a metric 3D point. The ROS node
``hole_align_planner`` composes that with the live flange pose + calibrated camera-to-flange extrinsics
to get the opening in ``base_link`` and re-aims the MoveIt preinsert hover.

Two detectors (config ``method``):
  * ``"contour"`` (DEFAULT, robust): isolate the vividly-coloured part with a SATURATION mask (the
    metal table + black gripper are gray -> dropped), find dark blobs inside it, and keep only ROUND
    ones (circularity / aspect / solidity). The cooling FINS are thin elongated slots -> rejected by
    shape; only the circular sockets survive. Then a CAD cross-check pairs the two sockets by their
    known 60 mm spacing (x = +/-30 mm), which uniquely fixes them without perception or colour tuning.
  * ``"hough"`` (legacy): cv2.HoughCircles sized by the known hole diameter. Kept for comparison; it
    over-triggers badly on the fin texture, so it is not the default.

Geometry (measured from CAD ``cooling_base.obj`` and matching sim ``CoolingInsert.asset_size``):
  * socket OPENING (inner rim) diameter  ~= 14.0 mm  -> the circle we detect / aim at
  * outer counterbore rim diameter       ~= 17.5 mm  (secondary ring; not used by default)
  * two sockets at local x = +/-30 mm  -> centre-to-centre spacing = 60 mm (the pairing cue)

Only numpy + OpenCV here (no rclpy), so the detector runs in a plain pytest. The ROS glue lives in
``hole_align_planner.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # OpenCV is a hard runtime dep for detection, but keep import errors legible.
    import cv2
except Exception as _cv_exc:  # noqa: BLE001  pragma: no cover
    cv2 = None  # type: ignore
    _CV_IMPORT_ERROR = _cv_exc
else:
    _CV_IMPORT_ERROR = None


def opencv_available() -> bool:
    return cv2 is not None


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics from ``sensor_msgs/CameraInfo.k`` (aligned depth shares these)."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, k) -> "CameraIntrinsics":
        return cls(float(k[0]), float(k[4]), float(k[2]), float(k[5]))


@dataclass
class HoleDetectorConfig:
    # --- detector selection ---
    method: str = "contour"          # "contour" (robust, default) or "hough" (legacy)

    # --- known geometry (metres) ---
    # 14 mm socket-opening diameter: measured from cooling_base.obj (inner rim r~=7.0 mm) and equal to
    # the sim CoolingInsert.asset_size (14 mm). Override to 0.0175 to chase the outer counterbore rim.
    hole_diameter_m: float = 0.014
    # Two sockets sit at x = +/-30 mm on the base -> 60 mm centre-to-centre. Used to pair the sockets
    # geometrically (a strong CAD-derived filter that rejects lone false positives).
    socket_spacing_m: float = 0.060
    socket_spacing_tol_m: float = 0.012
    use_socket_pair: bool = True

    # How far the detected circle radius may deviate (fraction) from the geometric prediction, in px.
    radius_tol_frac: float = 0.45
    min_radius_px: int = 4
    max_radius_px: int = 120

    # --- part segmentation (contour method) ---
    # The base is vividly coloured; the metal table + black gripper are gray/low-saturation. Keep only
    # colourful pixels -> the part. Set use_saturation_mask False for a monochrome / pale part.
    use_saturation_mask: bool = True
    sat_min: int = 60                # HSV S threshold (0..255) for "colourful enough to be the part"
    val_min: int = 40                # HSV V threshold; also drops the near-black gripper
    part_min_area_frac: float = 0.01 # drop colour blobs smaller than this * image area

    # --- dark-blob hole finding (contour method) ---
    blur_ksize: int = 5              # odd; median blur before thresholding
    adaptive_block: int = 51         # odd; adaptive-threshold neighbourhood (local darkness)
    adaptive_C: int = 5              # adaptive-threshold bias; higher => only clearly-dark pixels
    morph_open: bool = False         # sever thin fin slots -- OFF: it also erodes real socket rims; the
                                     # circularity/aspect/solidity filters reject fins on their own.
    min_circularity: float = 0.65    # 4*pi*A/P^2; 1.0 = perfect circle. Fins fail this.
    max_aspect: float = 1.8          # bbox long/short side; fins are very elongated.
    min_solidity: float = 0.80       # area / convex-hull area.
    min_fill: float = 0.55           # area / min-enclosing-circle area.
    # A real insertion socket is EMBEDDED in the coloured part (fins all around it); a mounting hole
    # sits at the part EDGE with the table showing through/around it. Require the ring just outside the
    # detected circle to be mostly coloured part -> rejects edge/through-holes. Off for a monochrome part.
    require_part_surround: bool = True
    surround_part_frac: float = 0.85
    surround_inner_frac: float = 1.25  # annulus inner radius = this * detected radius
    surround_outer_frac: float = 1.75  # annulus outer radius = this * detected radius

    # --- Hough (legacy method) ---
    hough_dp: float = 1.2
    hough_param1: float = 120.0
    hough_param2: float = 18.0
    min_dist_frac: float = 1.0
    detect_on: str = "rgb"           # "rgb" (grayscale) or "depth" (normalized) input to Hough

    # --- depth sampling for deprojection ---
    depth_sample_frac: float = 1.0
    depth_valid_min_m: float = 0.05
    depth_valid_max_m: float = 0.60

    # --- reference depth used to SIZE the radius window (perception-independent) ---
    ref_depth_roi_frac: float = 0.5


@dataclass
class HoleDetection:
    u: float                       # circle centre pixel (col)
    v: float                       # circle centre pixel (row)
    radius_px: float               # detected radius
    depth_m: float                 # sampled opening depth
    point_cam: np.ndarray          # (3,) 3D opening centre in the camera OPTICAL frame (x right, y down, z fwd)
    radius_err_frac: float         # |detected - predicted| / predicted  (lower is better)
    circularity: float = 1.0       # shape score (contour method); 1.0 for Hough
    paired: bool = False           # part of the 60 mm socket pair (CAD cross-check)

    def as_dict(self) -> dict:
        return {
            "u": float(self.u), "v": float(self.v), "radius_px": float(self.radius_px),
            "depth_m": float(self.depth_m), "point_cam": [float(x) for x in self.point_cam],
            "radius_err_frac": float(self.radius_err_frac), "circularity": float(self.circularity),
            "paired": bool(self.paired),
        }


# ---- geometry helpers (pure) ------------------------------------------------------------------

def deproject(u: float, v: float, z: float, intr: CameraIntrinsics) -> np.ndarray:
    """Pixel (u, v) + depth z -> 3D point in the camera optical frame (REP-104: x right, y down, z fwd)."""
    x = (float(u) - intr.cx) / intr.fx * z
    y = (float(v) - intr.cy) / intr.fy * z
    return np.array([x, y, float(z)], dtype=np.float64)


def predicted_radius_px(diameter_m: float, depth_m: float, intr: CameraIntrinsics) -> float:
    """Apparent radius (px) of a metric circle of ``diameter_m`` at ``depth_m`` under this pinhole."""
    f = 0.5 * (intr.fx + intr.fy)
    return 0.5 * diameter_m * f / max(depth_m, 1e-6)


def rotmat_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_point(point: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rigid transform p_dst = R @ p_src + t."""
    return R @ np.asarray(point, dtype=np.float64).reshape(3) + np.asarray(t, dtype=np.float64).reshape(3)


def cam_point_to_base(
    point_cam: np.ndarray,
    R_ee_cam: np.ndarray, t_ee_cam: np.ndarray,
    base_ee_pos, base_ee_quat_xyzw,
) -> np.ndarray:
    """Camera-optical point -> base frame: p_base = T_base_ee @ (R_ee_cam @ p_cam + t_ee_cam).

    ``R_ee_cam``/``t_ee_cam`` are the calibrated camera-to-flange offset (dst = lbr_two_link_ee, from
    ``camera_extrinsics_realsense.yaml``). ``base_ee_*`` is the live flange pose in base (from TF).
    """
    p_ee = transform_point(point_cam, R_ee_cam, t_ee_cam)
    R_base_ee = rotmat_from_quat_xyzw(base_ee_quat_xyzw)
    return transform_point(p_ee, R_base_ee, np.asarray(base_ee_pos, dtype=np.float64))


def base_point_to_cam(
    point_base: np.ndarray,
    R_ee_cam: np.ndarray, t_ee_cam: np.ndarray,
    base_ee_pos, base_ee_quat_xyzw,
) -> np.ndarray:
    """Inverse of :func:`cam_point_to_base`: base-frame point -> camera-optical point."""
    R_base_ee = rotmat_from_quat_xyzw(base_ee_quat_xyzw)
    p_ee = R_base_ee.T @ (np.asarray(point_base, dtype=np.float64).reshape(3)
                          - np.asarray(base_ee_pos, dtype=np.float64).reshape(3))
    return R_ee_cam.T @ (p_ee - np.asarray(t_ee_cam, dtype=np.float64).reshape(3))


def project_point_cam(point_cam: np.ndarray, intr: CameraIntrinsics) -> tuple[float, float]:
    """Camera-optical point -> pixel (u, v). Inverse of :func:`deproject`."""
    z = max(float(point_cam[2]), 1e-6)
    u = intr.fx * float(point_cam[0]) / z + intr.cx
    v = intr.fy * float(point_cam[1]) / z + intr.cy
    return u, v


# ---- extrinsics loading -----------------------------------------------------------------------

def load_cam_flange_extrinsics(yaml_path: str, cam_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``camera_extrinsics_realsense.yaml`` -> (R_ee_cam 3x3, t_ee_cam 3).

    For the wrist RealSenses (realsense_1/2) that file stores the STATIC camera-optical -> flange
    (lbr_link_ee) offset: p_ee = R @ p_cam + t, with R flattened ROW-MAJOR. (zed2i_1 is a different,
    camera-to-base entry -- not what we want here.)
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if cam_id not in data:
        raise KeyError(f"cam_id {cam_id!r} not in {yaml_path}; have {sorted(data)}")
    entry = data[cam_id]
    R = np.asarray(entry["R"], dtype=np.float64).reshape(3, 3)  # row-major
    t = np.asarray(entry["t"], dtype=np.float64).reshape(3)
    return R, t


# ---- depth sampling ---------------------------------------------------------------------------

def sample_opening_depth(
    depth_m: np.ndarray, u: float, v: float, radius_px: float, cfg: HoleDetectorConfig
) -> float:
    """Median of valid depths in a disc around (u, v); NaN if nothing valid.

    The narrow hole centre frequently returns nothing on the D405, so we sample the whole disc
    (rim + top face) and take the median of in-range returns -- a robust proxy for the opening plane.
    """
    h, w = depth_m.shape[:2]
    r = max(1, int(round(radius_px * cfg.depth_sample_frac)))
    u0, u1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    v0, v1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    if patch.size == 0:
        return float("nan")
    yy, xx = np.mgrid[v0:v1, u0:u1]
    disc = (xx - u) ** 2 + (yy - v) ** 2 <= float(r) ** 2
    vals = patch[disc]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= cfg.depth_valid_min_m) & (vals <= cfg.depth_valid_max_m)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def reference_depth(depth_m: np.ndarray, cfg: HoleDetectorConfig) -> float:
    """Median valid depth over a central ROI -- used to size the radius window a priori."""
    h, w = depth_m.shape[:2]
    fh, fw = int(h * cfg.ref_depth_roi_frac), int(w * cfg.ref_depth_roi_frac)
    v0, u0 = (h - fh) // 2, (w - fw) // 2
    roi = depth_m[v0:v0 + fh, u0:u0 + fw]
    vals = roi[np.isfinite(roi)]
    vals = vals[(vals >= cfg.depth_valid_min_m) & (vals <= cfg.depth_valid_max_m)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


# ---- image prep -------------------------------------------------------------------------------

def _to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)


def _depth_to_u8(depth_m: np.ndarray, cfg: HoleDetectorConfig) -> np.ndarray:
    lo, hi = cfg.depth_valid_min_m, cfg.depth_valid_max_m
    d = np.clip(depth_m, lo, hi)
    d = (d - lo) / max(hi - lo, 1e-6)
    d[~np.isfinite(depth_m)] = 0.0
    return (d * 255.0).astype(np.uint8)


def part_mask(rgb: np.ndarray, depth_m: np.ndarray, cfg: HoleDetectorConfig) -> np.ndarray:
    """Binary mask (uint8 0/255) of the coloured part: saturated pixels within the working depth band.

    The metal table and black gripper are low-saturation -> excluded, so downstream hole finding never
    even looks at them. When ``use_saturation_mask`` is off, falls back to the depth band alone (or the
    whole frame if depth is unusable).
    """
    h, w = rgb.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    if cfg.use_saturation_mask and rgb.ndim == 3:
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        mask = np.where((s >= cfg.sat_min) & (v >= cfg.val_min), 255, 0).astype(np.uint8)
    if depth_m is not None:
        band = np.isfinite(depth_m) & (depth_m >= cfg.depth_valid_min_m) & (depth_m <= cfg.depth_valid_max_m)
        if band.any():
            mask = cv2.bitwise_and(mask, band.astype(np.uint8) * 255)
    # Clean + keep only large connected blobs (the part, not speckle).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = cfg.part_min_area_frac * h * w
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    if not keep.any():
        keep = mask
    # FILL interior holes: the socket openings are dark/low-saturation (and often no-return in depth),
    # so they were punched OUT of the part region above. Fill each external contour solid so the holes
    # inside the part boundary count as "on the part" for the dark-blob search.
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


# ---- socket pairing (CAD cross-check) ---------------------------------------------------------

def _surround_part_fraction(rgb: np.ndarray, u: float, v: float, r: float, cfg: HoleDetectorConfig) -> float:
    """Fraction of an annulus just OUTSIDE the circle that is coloured part (HSV saturation >= sat_min).

    ~1.0 for a socket embedded in the fins; low for an edge/through mounting hole (table shows around).
    """
    if rgb.ndim != 3:
        return 1.0
    h, w = rgb.shape[:2]
    r_out = r * cfg.surround_outer_frac
    u0, u1 = max(0, int(u - r_out)), min(w, int(u + r_out) + 1)
    v0, v1 = max(0, int(v - r_out)), min(h, int(v + r_out) + 1)
    if u1 <= u0 or v1 <= v0:
        return 0.0
    patch = rgb[v0:v1, u0:u1]
    hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2HSV)
    yy, xx = np.mgrid[v0:v1, u0:u1]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    ring = (d2 >= (r * cfg.surround_inner_frac) ** 2) & (d2 <= r_out ** 2)
    if ring.sum() == 0:
        return 0.0
    colored = (hsv[:, :, 1] >= cfg.sat_min) & (hsv[:, :, 2] >= cfg.val_min)
    return float(colored[ring].sum()) / float(ring.sum())


def mark_socket_pair(dets: list[HoleDetection], cfg: HoleDetectorConfig) -> tuple[int, int] | None:
    """Flag the candidate pair whose 3D spacing best matches the known 60 mm socket spacing."""
    best = None
    best_err = cfg.socket_spacing_tol_m
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            d = float(np.linalg.norm(dets[i].point_cam - dets[j].point_cam))
            err = abs(d - cfg.socket_spacing_m)
            if err <= best_err:
                best_err, best = err, (i, j)
    if best is not None:
        dets[best[0]].paired = True
        dets[best[1]].paired = True
    return best


# ---- detection --------------------------------------------------------------------------------

def _finalize_candidate(u, v, r, circ, depth_m, intr, cfg) -> HoleDetection | None:
    z = sample_opening_depth(depth_m, u, v, r, cfg)
    if not math.isfinite(z):
        return None
    pr = predicted_radius_px(cfg.hole_diameter_m, z, intr)
    err = abs(r - pr) / max(pr, 1e-6)
    return HoleDetection(u, v, r, z, deproject(u, v, z, intr), err, circularity=circ)


def _detect_contour(rgb, depth_m, intr, cfg, ref_depth_m) -> list[HoleDetection]:
    pred_r = predicted_radius_px(cfg.hole_diameter_m, ref_depth_m, intr)
    r_min = max(cfg.min_radius_px, pred_r * (1.0 - cfg.radius_tol_frac))
    r_max = min(cfg.max_radius_px, pred_r * (1.0 + cfg.radius_tol_frac))
    area_min = math.pi * (r_min * 0.7) ** 2   # allow partially-occluded blobs

    mask = part_mask(rgb, depth_m, cfg)
    gray = _to_gray_u8(rgb)
    k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
    blurred = cv2.medianBlur(gray, k)
    block = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
    dark = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, cfg.adaptive_C
    )
    dark = cv2.bitwise_and(dark, mask)
    if cfg.morph_open:
        # Kernel ~ half the predicted hole radius: severs the thin fin slots, keeps the round holes.
        ok = max(3, int(round(pred_r * 0.5)))
        ok += 1 - (ok % 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[HoleDetection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < r_min or r > r_max:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 1e-6:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        fill = area / max(math.pi * r * r, 1e-6)
        hull = cv2.convexHull(c)
        harea = cv2.contourArea(hull)
        solidity = area / harea if harea > 0 else 0.0
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if circ < cfg.min_circularity or aspect > cfg.max_aspect:
            continue
        if solidity < cfg.min_solidity or fill < cfg.min_fill:
            continue
        if cfg.require_part_surround and _surround_part_fraction(rgb, x, y, r, cfg) < cfg.surround_part_frac:
            continue  # edge / mounting through-hole: table shows around it
        det = _finalize_candidate(x, y, r, circ, depth_m, intr, cfg)
        if det is not None:
            out.append(det)

    if cfg.use_socket_pair:
        mark_socket_pair(out, cfg)
    # Paired sockets first, then rounder + better radius match.
    out.sort(key=lambda d: (not d.paired, -d.circularity, d.radius_err_frac))
    return out


def _detect_hough(rgb, depth_m, intr, cfg, ref_depth_m) -> list[HoleDetection]:
    pred_r = predicted_radius_px(cfg.hole_diameter_m, ref_depth_m, intr)
    r_min = int(max(cfg.min_radius_px, math.floor(pred_r * (1.0 - cfg.radius_tol_frac))))
    r_max = int(min(cfg.max_radius_px, math.ceil(pred_r * (1.0 + cfg.radius_tol_frac))))
    r_min = max(1, min(r_min, r_max - 1))
    min_dist = max(1.0, cfg.min_dist_frac * 2.0 * pred_r)

    src = _depth_to_u8(depth_m, cfg) if cfg.detect_on == "depth" else _to_gray_u8(rgb)
    k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
    blurred = cv2.medianBlur(src, k)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=cfg.hough_dp, minDist=min_dist,
        param1=cfg.hough_param1, param2=cfg.hough_param2, minRadius=r_min, maxRadius=r_max,
    )
    out: list[HoleDetection] = []
    if circles is None:
        return out
    for u, v, r in np.asarray(circles[0], dtype=np.float64):
        det = _finalize_candidate(u, v, r, 1.0, depth_m, intr, cfg)
        if det is not None:
            out.append(det)
    if cfg.use_socket_pair:
        mark_socket_pair(out, cfg)
    out.sort(key=lambda d: (not d.paired, d.radius_err_frac))
    return out


def detect_holes(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    cfg: HoleDetectorConfig,
    *,
    ref_depth_m: float | None = None,
) -> list[HoleDetection]:
    """Detect socket-opening circles, best-first. Dispatches on ``cfg.method``."""
    if cv2 is None:
        raise RuntimeError(f"OpenCV (cv2) is required for hole detection: {_CV_IMPORT_ERROR}")
    if ref_depth_m is None or not math.isfinite(ref_depth_m):
        ref_depth_m = reference_depth(depth_m, cfg)
    if not math.isfinite(ref_depth_m):
        return []
    if str(cfg.method).strip().lower() == "hough":
        return _detect_hough(rgb, depth_m, intr, cfg, ref_depth_m)
    return _detect_contour(rgb, depth_m, intr, cfg, ref_depth_m)


def select_hole(
    detections: list[HoleDetection],
    intr: CameraIntrinsics,
    *,
    expected_uv: tuple[float, float] | None = None,
    max_center_dist_px: float | None = None,
    prefer_paired: bool = True,
) -> HoleDetection | None:
    """Pick the target opening.

    Prefer the CAD-paired sockets when any exist (they are the two real holes). Then, with
    ``expected_uv`` (the perception socket projected into the image) pick the nearest to it and reject
    anything beyond ``max_center_dist_px``; without perception, pick the one nearest the image centre
    (you are hovering over the target socket, so it images near the principal point).
    """
    if not detections:
        return None
    pool = detections
    if prefer_paired and any(d.paired for d in detections):
        pool = [d for d in detections if d.paired]

    if expected_uv is not None:
        eu, ev = expected_uv
        scored = sorted(pool, key=lambda d: math.hypot(d.u - eu, d.v - ev))
        best = scored[0]
        if max_center_dist_px is not None and math.hypot(best.u - eu, best.v - ev) > max_center_dist_px:
            return None
        return best
    return min(pool, key=lambda d: math.hypot(d.u - intr.cx, d.v - intr.cy))


# ---- debug rendering --------------------------------------------------------------------------

def render_debug(
    rgb: np.ndarray,
    detections: list[HoleDetection],
    chosen: HoleDetection | None,
    *,
    expected_uv: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Annotated BGR image: part mask tint, all circles (yellow), paired (cyan), chosen (green),
    perception estimate (red +)."""
    if cv2 is None:
        raise RuntimeError("OpenCV required for render_debug")
    img = rgb.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        tint = np.zeros_like(img)
        tint[mask > 0] = (60, 0, 0)
        img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)
    for d in detections:
        col = (255, 220, 0) if d.paired else (0, 220, 220)  # cyan if paired, else yellow
        cv2.circle(img, (int(d.u), int(d.v)), int(d.radius_px), col, 1)
        cv2.circle(img, (int(d.u), int(d.v)), 2, col, -1)
    if expected_uv is not None:
        eu, ev = int(expected_uv[0]), int(expected_uv[1])
        cv2.drawMarker(img, (eu, ev), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    if chosen is not None:
        cv2.circle(img, (int(chosen.u), int(chosen.v)), int(chosen.radius_px), (0, 255, 0), 2)
        cv2.circle(img, (int(chosen.u), int(chosen.v)), 3, (0, 255, 0), -1)
        cv2.putText(
            img, f"z={chosen.depth_m*1000:.0f}mm circ={chosen.circularity:.2f} r_err={chosen.radius_err_frac*100:.0f}%",
            (int(chosen.u) + 8, int(chosen.v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return img


def depth_colormap(depth_m: np.ndarray, cfg: HoleDetectorConfig) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV required for depth_colormap")
    return cv2.applyColorMap(_depth_to_u8(depth_m, cfg), cv2.COLORMAP_JET)
