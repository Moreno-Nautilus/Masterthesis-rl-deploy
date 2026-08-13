"""Unit tests for the pure (ROS-free) hole detector + geometry helpers."""

import numpy as np
import pytest

from rl_deploy_inference import hole_detector as hd

INTR = hd.CameraIntrinsics(fx=430.0, fy=430.0, cx=320.0, cy=240.0)


def test_deproject_project_roundtrip():
    p = hd.deproject(400.0, 300.0, 0.2, INTR)
    u, v = hd.project_point_cam(p, INTR)
    assert u == pytest.approx(400.0)
    assert v == pytest.approx(300.0)
    # centre pixel deprojects to a pure +z ray
    c = hd.deproject(INTR.cx, INTR.cy, 0.5, INTR)
    assert c[0] == pytest.approx(0.0)
    assert c[1] == pytest.approx(0.0)
    assert c[2] == pytest.approx(0.5)


def test_predicted_radius_scales_with_geometry():
    # 14 mm circle, closer => larger apparent radius; radius is linear in diameter, inverse in depth.
    r_near = hd.predicted_radius_px(0.014, 0.15, INTR)
    r_far = hd.predicted_radius_px(0.014, 0.30, INTR)
    assert r_near == pytest.approx(2.0 * r_far, rel=1e-6)
    assert hd.predicted_radius_px(0.028, 0.15, INTR) == pytest.approx(2.0 * r_near, rel=1e-6)


def test_cam_to_base_roundtrip_with_identity_flange():
    R_ee_cam = np.eye(3)
    t_ee_cam = np.array([0.01, -0.02, 0.03])
    base_ee_pos = np.array([0.4, 0.1, 0.5])
    base_ee_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity
    p_cam = np.array([0.02, -0.01, 0.2])
    p_base = hd.cam_point_to_base(p_cam, R_ee_cam, t_ee_cam, base_ee_pos, base_ee_quat)
    back = hd.base_point_to_cam(p_base, R_ee_cam, t_ee_cam, base_ee_pos, base_ee_quat)
    assert np.allclose(back, p_cam, atol=1e-9)
    # identity rotations => p_base = p_cam + t_ee_cam + base_ee_pos
    assert np.allclose(p_base, p_cam + t_ee_cam + base_ee_pos, atol=1e-9)


def test_cam_to_base_roundtrip_with_rotated_flange():
    rng = np.random.default_rng(0)
    R_ee_cam = hd.rotmat_from_quat_xyzw(rng.normal(size=4))
    t_ee_cam = rng.normal(size=3) * 0.05
    base_ee_pos = rng.normal(size=3) * 0.3
    base_ee_quat = rng.normal(size=4)
    p_cam = np.array([0.03, -0.02, 0.25])
    p_base = hd.cam_point_to_base(p_cam, R_ee_cam, t_ee_cam, base_ee_pos, base_ee_quat)
    back = hd.base_point_to_cam(p_base, R_ee_cam, t_ee_cam, base_ee_pos, base_ee_quat)
    assert np.allclose(back, p_cam, atol=1e-9)


def test_sample_opening_depth_ignores_invalid_and_takes_median():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[40:60, 40:60] = 0.2       # a valid patch
    depth[49:51, 49:51] = 0.0       # hole centre has no return
    cfg = hd.HoleDetectorConfig()
    z = hd.sample_opening_depth(depth, 50, 50, 10, cfg)
    assert z == pytest.approx(0.2)
    # all-invalid -> NaN
    assert np.isnan(hd.sample_opening_depth(np.zeros((10, 10), np.float32), 5, 5, 3, cfg))


def _synthetic_scene(radius_px: int, depth_m: float):
    cv2 = pytest.importorskip("cv2")
    h, w = 240, 320
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)      # bright top face
    cv2.circle(rgb, (160, 120), radius_px, (20, 20, 20), -1)  # dark hole
    depth = np.full((h, w), depth_m, dtype=np.float32)  # flat top face at depth
    return rgb, depth


def test_detect_holes_finds_synthetic_circle():
    pytest.importorskip("cv2")
    intr = hd.CameraIntrinsics(fx=600.0, fy=600.0, cx=160.0, cy=120.0)
    depth_m = 0.20
    r = int(round(hd.predicted_radius_px(0.014, depth_m, intr)))  # geometric radius for a 14 mm hole
    rgb, depth = _synthetic_scene(r, depth_m)
    cfg = hd.HoleDetectorConfig(method="hough", hough_param2=15.0, radius_tol_frac=0.6)
    dets = hd.detect_holes(rgb, depth, intr, cfg, ref_depth_m=depth_m)
    assert dets, "expected at least one detected circle"
    best = dets[0]
    assert best.u == pytest.approx(160.0, abs=4.0)
    assert best.v == pytest.approx(120.0, abs=4.0)
    assert best.depth_m == pytest.approx(depth_m, abs=1e-3)
    # deprojected 3D centre lands ~on the optical axis at the right depth
    assert best.point_cam[2] == pytest.approx(depth_m, abs=1e-3)


def test_select_hole_prefers_nearest_to_expected():
    dets = [
        hd.HoleDetection(100, 100, 12, 0.2, np.zeros(3), 0.1),
        hd.HoleDetection(200, 120, 12, 0.2, np.zeros(3), 0.05),
    ]
    chosen = hd.select_hole(dets, INTR, expected_uv=(198, 121), max_center_dist_px=20)
    assert chosen is dets[1]
    # outside the gate -> None
    assert hd.select_hole(dets, INTR, expected_uv=(0, 0), max_center_dist_px=20) is None
    # no expectation -> nearest the image centre (cx=320, cy=240): dets[1] at (200,120) is closer
    assert hd.select_hole(dets, INTR, expected_uv=None) is dets[1]


def _two_socket_scene(intr, depth_m, spacing_m, add_fins=True):
    """Bright coloured plate with two dark round holes ``spacing_m`` apart, optionally cooling fins."""
    cv2 = pytest.importorskip("cv2")
    h, w = 480, 848
    # Vividly-coloured plate (high saturation) on a gray background so the saturation mask bites.
    rgb = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)          # gray table
    cv2.rectangle(rgb, (250, 120), (600, 380), (10, 180, 200), -1)  # teal plate (high S)
    r = int(round(hd.predicted_radius_px(0.014, depth_m, intr)))
    # place two holes symmetric about the plate centre, spacing_m apart in metres -> px via fx
    dpx = int(round(spacing_m * intr.fx / depth_m))
    cxp = (250 + 600) // 2
    cyp = (120 + 380) // 2
    centers = [(cxp - dpx // 2, cyp), (cxp + dpx // 2, cyp)]
    if add_fins:
        for x in range(300, 560, 12):  # elongated dark fin slots on the plate
            cv2.rectangle(rgb, (x, 150), (x + 4, 350), (5, 5, 5), -1)
    for (cx, cy) in centers:
        cv2.circle(rgb, (cx, cy), int(r * 1.8), (10, 180, 200), -1)  # blue margin isolates the socket
        cv2.circle(rgb, (cx, cy), r, (5, 5, 5), -1)                  # dark round hole (blind socket)
    depth = np.full((h, w), depth_m, dtype=np.float32)
    return rgb, depth, centers


def test_contour_detector_finds_both_sockets_and_rejects_fins():
    pytest.importorskip("cv2")
    intr = hd.CameraIntrinsics(fx=436.0, fy=436.0, cx=424.0, cy=240.0)
    depth_m, spacing = 0.12, 0.060
    rgb, depth, centers = _two_socket_scene(intr, depth_m, spacing, add_fins=True)
    cfg = hd.HoleDetectorConfig(method="contour")
    dets = hd.detect_holes(rgb, depth, intr, cfg, ref_depth_m=depth_m)
    # exactly the two round holes survive shape + surround filters (fins rejected)
    assert len(dets) == 2, [d.as_dict() for d in dets]
    found = sorted((round(d.u), round(d.v)) for d in dets)
    want = sorted(centers)
    for (fu, fv), (wu, wv) in zip(found, want):
        assert abs(fu - wu) <= 4 and abs(fv - wv) <= 4
    # CAD 60 mm pairing flags both
    assert all(d.paired for d in dets)
    # measured 3D spacing matches the CAD spacing
    sep = np.linalg.norm(dets[0].point_cam - dets[1].point_cam)
    assert sep == pytest.approx(spacing, abs=0.006)


def test_contour_selects_socket_nearest_image_centre_without_perception():
    pytest.importorskip("cv2")
    intr = hd.CameraIntrinsics(fx=436.0, fy=436.0, cx=424.0, cy=240.0)
    rgb, depth, _ = _two_socket_scene(intr, 0.12, 0.060, add_fins=True)
    cfg = hd.HoleDetectorConfig(method="contour")
    dets = hd.detect_holes(rgb, depth, intr, cfg, ref_depth_m=0.12)
    chosen = hd.select_hole(dets, intr)  # no expected_uv
    assert chosen is not None
    # nearest-centre tie-breaker picks a real (paired) socket, never a stray
    assert chosen.paired


def test_surround_fraction_high_when_embedded_low_at_edge():
    cv2 = pytest.importorskip("cv2")
    h, w = 200, 200
    rgb = np.full((h, w, 3), (20, 20, 20), dtype=np.uint8)          # gray background
    cv2.rectangle(rgb, (0, 0), (100, 200), (10, 180, 200), -1)      # coloured part fills the LEFT half
    cfg = hd.HoleDetectorConfig()
    # a hole deep inside the part -> ring is all part
    assert hd._surround_part_fraction(rgb, 50, 100, 12, cfg) > 0.9
    # a hole straddling the part edge (x=100) -> half the ring is background
    assert hd._surround_part_fraction(rgb, 100, 100, 12, cfg) < 0.75


def test_mark_socket_pair_flags_the_60mm_pair_only():
    cfg = hd.HoleDetectorConfig()
    dets = [
        hd.HoleDetection(10, 10, 12, 0.2, np.array([0.0, 0.0, 0.20]), 0.0),
        hd.HoleDetection(20, 10, 12, 0.2, np.array([0.060, 0.0, 0.20]), 0.0),  # 60 mm from #0
        hd.HoleDetection(30, 10, 12, 0.2, np.array([0.30, 0.0, 0.20]), 0.0),   # far outlier
    ]
    pair = hd.mark_socket_pair(dets, cfg)
    assert pair == (0, 1)
    assert dets[0].paired and dets[1].paired and not dets[2].paired


def test_load_extrinsics_reads_realsense_entry(tmp_path):
    import yaml

    p = tmp_path / "ext.yaml"
    p.write_text(yaml.safe_dump({
        "realsense_2": {"R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "t": [0.1, 0.2, 0.3]},
    }))
    R, t = hd.load_cam_flange_extrinsics(str(p), "realsense_2")
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, [0.1, 0.2, 0.3])
    with pytest.raises(KeyError):
        hd.load_cam_flange_extrinsics(str(p), "nope")
