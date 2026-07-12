"""Multi-camera geometry, trust-boundary, uncertainty, and workload contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from manwe.multicam import (
    Camera,
    CameraRig,
    Detection2D,
    Detection3D,
    correlate_and_triangulate,
    reprojection_error,
    to_measurements,
    triangulate_dlt,
    triangulate_midpoint,
)


def _rig() -> list[Camera]:
    return [
        Camera.from_lookat([0, 0, 100], [0, 0, 0], fov_deg=60, name="cam0"),
        Camera.from_lookat([100, 0, 20], [0, 0, 0], fov_deg=60, name="cam1"),
        Camera.from_lookat([-80, 60, 40], [0, 0, 0], fov_deg=60, name="cam2"),
    ]


def _detection(
    camera_index: int,
    pixel,
    class_label: str | None = None,
    confidence: float = 1.0,
    timestamp: float | None = None,
    camera_id: str | None = None,
    *,
    pixel_std_px: float = 0.75,
    timestamp_std_s: float = 0.0,
) -> Detection2D:
    return Detection2D(
        camera_index,
        pixel,
        class_label,
        confidence,
        timestamp,
        camera_id,
        pixels_undistorted=True,
        pixel_std_px=pixel_std_px,
        timestamp_std_s=timestamp_std_s,
    )


def _correlate(cameras, detections, **kwargs):
    kwargs.setdefault("max_speed_mps", 100.0)
    return correlate_and_triangulate(cameras, detections, **kwargs)


def test_project_backproject_consistency():
    camera = _rig()[0]
    point = np.array([5.0, 3.0, 2.0])
    pixel = camera.project(point)
    origin, direction = camera.backproject_ray(pixel)
    to_point = point - origin
    to_point /= np.linalg.norm(to_point)
    assert np.dot(to_point, direction) > 0.999


def test_camera_calibration_is_defensively_copied_and_irreversibly_read_only():
    intrinsics = np.diag([100.0, 100.0, 1.0])
    rotation = np.eye(3)
    translation = np.zeros(3)
    camera = Camera(intrinsics, rotation, translation)

    intrinsics[0, 0] = 200.0
    rotation[0, 0] = -1.0
    translation[0] = 100.0
    assert camera.K[0, 0] == 100.0
    assert np.array_equal(camera.R, np.eye(3))
    assert np.array_equal(camera.t, np.zeros(3))
    assert not camera.K.flags.writeable
    assert not camera.R.flags.writeable
    assert not camera.t.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        camera.K[0, 0] = 1.0
    with pytest.raises(ValueError):
        camera.K.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        camera.name = "mutated"  # type: ignore[misc]


def test_dlt_recovers_point_with_finite_reprojection_error():
    cameras = _rig()
    point = np.array([4.0, -6.0, 3.0])
    pixels = [camera.project(point) for camera in cameras]
    recovered = triangulate_dlt(cameras, pixels)
    assert np.linalg.norm(recovered - point) < 1e-3
    assert reprojection_error(cameras, pixels, recovered) < 1e-3


def test_two_view_triangulation():
    cameras = _rig()[:2]
    point = np.array([2.0, 8.0, -5.0])
    pixels = [camera.project(point) for camera in cameras]
    assert np.linalg.norm(triangulate_dlt(cameras, pixels) - point) < 1e-3


def test_cross_camera_correlation_recovers_two_targets_with_covariance():
    cameras = _rig()
    truth = {"drone": np.array([5.0, 3.0, 2.0]), "bird": np.array([-6.0, 4.0, -3.0])}
    detections = []
    for camera_index, camera in enumerate(cameras):
        for label, point in truth.items():
            detections.append(
                _detection(
                    camera_index,
                    camera.project(point),
                    class_label=label,
                    confidence=0.9,
                )
            )
    fused = _correlate(
        cameras,
        detections,
        max_ray_gap=8.0,
        max_reprojection=2.0,
    )
    assert len(fused) == 2
    by_label = {detection.class_label: detection for detection in fused}
    for label, point in truth.items():
        detection = by_label[label]
        assert np.linalg.norm(detection.position - point) < 0.5
        assert detection.camera_indices == (0, 1, 2)
        assert detection.reprojection_error < 2.0
        assert detection.position_covariance.shape == (3, 3)
        assert np.linalg.eigvalsh(detection.position_covariance)[0] >= -1e-10
        assert not detection.position_covariance.flags.writeable


def test_legacy_untimestamped_batch_uses_full_skew_uncertainty():
    cameras = _rig()[:2]
    target = np.array([3.0, -2.0, 1.0])
    detections = [_detection(index, camera.project(target)) for index, camera in enumerate(cameras)]
    fused = _correlate(cameras, detections, max_time_skew=0.04, max_speed_mps=25.0)
    assert len(fused) == 1
    assert fused[0].timestamp is None
    assert fused[0].timestamp_reference == "external_batch_reference"
    assert fused[0].time_uncertainty_s == 0.04
    with pytest.raises(ValueError, match="timestamp is required"):
        to_measurements(fused)
    measurements = to_measurements(fused, timestamp=50.0)
    assert measurements[0].timestamp == 50.0


def test_camera_rejects_invalid_calibration_dimensions_fov_and_extremes():
    with pytest.raises(ValueError, match="finite"):
        Camera(np.diag([np.nan, 1.0, 1.0]), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="focal"):
        Camera(np.diag([-1.0, 1.0, 1.0]), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="orthonormal"):
        Camera(np.eye(3), np.diag([1.0, 1.0, 2.0]), np.zeros(3))
    huge_rotation = np.eye(3)
    huge_rotation[0, 0] = 1e308
    with pytest.raises(ValueError, match="supported geometry magnitude"):
        Camera(np.eye(3), huge_rotation, np.zeros(3))
    huge_intrinsics = np.eye(3)
    huge_intrinsics[0, 0] = 1e10
    with pytest.raises(ValueError, match="intrinsic magnitude"):
        Camera(huge_intrinsics, np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="determinant"):
        Camera(np.eye(3), np.diag([1.0, 1.0, -1.0]), np.zeros(3))
    with pytest.raises(ValueError, match="both be zero"):
        Camera(np.eye(3), np.eye(3), np.zeros(3), width=640, height=0)
    with pytest.raises(ValueError, match="non-negative"):
        Camera(np.eye(3), np.eye(3), np.zeros(3), width=-1, height=0)
    with pytest.raises(ValueError, match="image dimension"):
        Camera.from_lookat([0, 0, 1], [0, 0, 0], width=100_001)
    with pytest.raises(ValueError, match="fov_deg"):
        Camera.from_lookat([0, 0, 1], [0, 0, 0], fov_deg=0.5)
    with pytest.raises(ValueError, match="fov_deg"):
        Camera.from_lookat([0, 0, 1], [0, 0, 0], fov_deg=171.0)


def test_camera_rejects_degenerate_lookat_and_points_behind_it():
    with pytest.raises(ValueError, match="distinct"):
        Camera.from_lookat([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="parallel"):
        Camera.from_lookat([0, 0, 1], [0, 0, 0], up=[0, 0, 1])
    camera = Camera.from_lookat([0, 0, 0], [0, 0, 1])
    with pytest.raises(ValueError, match="front"):
        camera.project([0, 0, -1])


def test_camera_and_rig_schema_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        Camera.from_dict(
            {
                "K": np.eye(3),
                "R": np.eye(3),
                "t": np.zeros(3),
                "distortion": [0.1],
            }
        )
    with pytest.raises(ValueError, match="unknown keys"):
        Camera.from_dict(
            {
                "position": [0, 0, 1],
                "look_at": [0, 0, 0],
                "fov": 60,
            }
        )
    with pytest.raises(ValueError, match="unknown keys"):
        CameraRig.from_dict({"schema_version": 1, "cameras": [], "max_speed_mps": 1, "typo": 1})


def test_committed_rig_example_loads_and_binds_all_gates():
    pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "configs" / "multicam" / "rig.example.yaml"
    rig = CameraRig.from_yaml(path)
    assert len(rig.cameras) == 3
    assert [camera.name for camera in rig.cameras] == ["cam0", "cam1", "cam2"]
    assert rig.max_ray_gap_m == 8.0
    assert rig.max_reprojection_px == 12.0
    assert rig.max_time_skew_s == 0.05
    assert rig.min_ray_angle_deg == 1.0
    assert rig.max_range_m == 100_000.0
    assert rig.max_speed_mps == 100.0
    assert rig.max_cameras == 16
    assert rig.max_detections == 4096
    assert rig.max_candidate_pairs == 1_000_000
    assert rig.max_hypotheses == 100_000
    assert rig.max_association_states == 1_000_000

    target = np.array([2.0, 3.0, 4.0])
    detections = [
        _detection(
            index,
            camera.project(target),
            timestamp=10.0 + index * 0.001,
            camera_id=camera.name,
        )
        for index, camera in enumerate(rig.cameras)
    ]
    fused = rig.correlate(detections)
    assert len(fused) == 1
    assert fused[0].timestamp == 10.002
    assert fused[0].timestamp_reference == "latest_capture"

    with pytest.raises(ValueError, match="unique"):
        CameraRig((rig.cameras[0], rig.cameras[0]), max_speed_mps=100.0)
    with pytest.raises(ValueError, match="max_time_skew"):
        CameraRig(rig.cameras[:2], max_speed_mps=100.0, max_time_skew_s=-0.1)


def test_rig_yaml_rejects_ambiguous_and_symlinked_inputs(tmp_path):
    pytest.importorskip("yaml")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: 1\nschema_version: 1\ncameras: []\nmax_speed_mps: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        CameraRig.from_yaml(duplicate)

    anchored = tmp_path / "anchored.yaml"
    anchored.write_text("cameras: &c []\nmax_speed_mps: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aliases, anchors"):
        CameraRig.from_yaml(anchored)

    linked = tmp_path / "linked.yaml"
    linked.symlink_to(duplicate)
    with pytest.raises(ValueError, match="symbolic link"):
        CameraRig.from_yaml(linked)


def test_detection_requires_undistorted_pixels_and_localization_uncertainty():
    with pytest.raises(TypeError, match="pixels_undistorted"):
        Detection2D(0, [1, 2])
    with pytest.raises(TypeError, match="pixel_std_px"):
        Detection2D(0, [1, 2], pixels_undistorted=True)
    with pytest.raises(ValueError, match="explicitly True"):
        Detection2D(
            0,
            [1, 2],
            pixels_undistorted=False,
            pixel_std_px=1.0,
        )
    with pytest.raises(ValueError, match="pixel_std_px"):
        Detection2D(
            0,
            [1, 2],
            pixels_undistorted=True,
            pixel_std_px=0.0,
        )


def test_detection_validation_identity_image_bounds_and_immutability():
    with pytest.raises(ValueError, match="confidence"):
        _detection(0, [1, 2], confidence=1.01)
    with pytest.raises(ValueError, match="finite"):
        _detection(0, [np.nan, 2])
    with pytest.raises(ValueError, match="camera_index"):
        _detection(-1, [1, 2])
    with pytest.raises(ValueError, match="camera_id"):
        _detection(0, [1, 2], camera_id="")
    with pytest.raises(ValueError, match="timestamp_std_s must be zero"):
        _detection(0, [1, 2], timestamp_std_s=0.01)
    with pytest.raises(ValueError, match="camera_id"):
        _detection(0, [1, 2], camera_id="x" * 257)
    with pytest.raises(ValueError, match="class_label"):
        _detection(0, [1, 2], class_label="x" * 257)
    with pytest.raises(ValueError, match="camera_id"):
        _detection(0, [1, 2], camera_id="camera\nspoof")
    with pytest.raises(ValueError, match="supported magnitude"):
        _detection(0, [1, 2], timestamp=float(1 << 33))
    with pytest.raises(ValueError, match="camera_index"):
        _detection(64, [1, 2])
    with pytest.raises(ValueError, match="256-byte"):
        Camera.from_lookat([0, 0, 1], [0, 0, 0], name="x" * 257)

    source = np.array([1.0, 2.0])
    detection = _detection(0, source)
    source[0] = 99.0
    assert np.array_equal(detection.pixel, [1.0, 2.0])
    with pytest.raises(ValueError, match="read-only"):
        detection.pixel[0] = 10.0

    cameras = _rig()[:2]
    target = np.zeros(3)
    identities = [
        _detection(0, cameras[0].project(target), camera_id="wrong"),
        _detection(1, cameras[1].project(target), camera_id="cam1"),
    ]
    with pytest.raises(ValueError, match="does not match"):
        _correlate(cameras, identities)

    outside = [
        _detection(0, [cameras[0].width, 10]),
        _detection(1, cameras[1].project(target)),
    ]
    with pytest.raises(ValueError, match="outside"):
        _correlate(cameras, outside)


def test_capture_timestamp_uncertainty_is_enforced_and_propagated():
    cameras = _rig()[:2]
    target = np.array([2.0, 3.0, 4.0])
    mixed = [
        _detection(0, cameras[0].project(target), timestamp=10.0),
        _detection(1, cameras[1].project(target)),
    ]
    with pytest.raises(ValueError, match="cannot be mixed"):
        _correlate(cameras, mixed)

    stale = [
        _detection(0, cameras[0].project(target), timestamp=10.0, camera_id="cam0"),
        _detection(1, cameras[1].project(target), timestamp=10.2, camera_id="cam1"),
    ]
    with pytest.raises(ValueError, match="max_time_skew"):
        _correlate(cameras, stale, max_time_skew=0.05)

    synchronized = [
        _detection(
            0,
            cameras[0].project(target),
            timestamp=10.0,
            camera_id="cam0",
            timestamp_std_s=0.001,
        ),
        _detection(
            1,
            cameras[1].project(target),
            timestamp=10.01,
            camera_id="cam1",
            timestamp_std_s=0.001,
        ),
    ]
    fused = _correlate(cameras, synchronized, max_time_skew=0.05, max_speed_mps=20.0)
    assert len(fused) == 1
    assert fused[0].timestamp == 10.01
    assert fused[0].time_uncertainty_s == pytest.approx(0.012)
    assert fused[0].camera_ids == ("cam0", "cam1")
    measurements = to_measurements(fused)
    assert measurements[0].timestamp == 10.01
    assert np.allclose(measurements[0].covariance, fused[0].position_covariance)

    later = to_measurements(fused, timestamp=10.02)[0]
    expected_added_variance = (20.0 * 0.01) ** 2
    assert np.allclose(
        later.covariance - measurements[0].covariance,
        np.eye(3) * expected_added_variance,
    )


def test_pixel_and_time_uncertainty_increase_output_covariance():
    cameras = _rig()[:2]
    target = np.array([2.0, 3.0, 4.0])

    def fuse(pixel_std: float, second_timestamp: float):
        detections = [
            _detection(
                index,
                camera.project(target),
                timestamp=second_timestamp if index else 20.0,
                pixel_std_px=pixel_std,
            )
            for index, camera in enumerate(cameras)
        ]
        return _correlate(
            cameras,
            detections,
            max_time_skew=0.1,
            max_speed_mps=50.0,
        )[0]

    low_pixel = fuse(0.25, 20.0)
    high_pixel = fuse(2.0, 20.0)
    skewed = fuse(0.25, 20.02)
    assert np.trace(high_pixel.position_covariance) > np.trace(low_pixel.position_covariance)
    assert np.all(np.diag(skewed.position_covariance) > np.diag(low_pixel.position_covariance))


def test_uncertainty_and_work_inputs_fail_closed_when_absent_or_exceeded():
    cameras = _rig()
    target = np.zeros(3)
    detections = [_detection(index, camera.project(target)) for index, camera in enumerate(cameras)]
    with pytest.raises(ValueError, match="max_speed_mps is required"):
        correlate_and_triangulate(cameras, detections)
    with pytest.raises(ValueError, match="camera count"):
        _correlate(cameras, detections, max_cameras=2)
    with pytest.raises(ValueError, match="detection count"):
        _correlate(cameras, detections, max_detections=2)
    with pytest.raises(ValueError, match="candidate pair count"):
        _correlate(cameras, detections, max_candidate_pairs=2)
    with pytest.raises(ValueError, match="max_hypotheses"):
        _correlate(cameras, detections, max_hypotheses=3)
    with pytest.raises(ValueError, match="max_association_states"):
        _correlate(cameras, detections, max_association_states=1)
    with pytest.raises(ValueError, match="camera count"):
        CameraRig(cameras, max_speed_mps=100.0, max_cameras=2)

    with pytest.raises(ValueError, match="at most 64"):
        Detection3D(
            position=np.zeros(3),
            class_label=None,
            confidence=1.0,
            camera_indices=tuple(range(65)),
            reprojection_error=0.0,
            position_covariance=np.eye(3),
        )
    with pytest.raises(ValueError, match="camera_ids must contain at most 64"):
        Detection3D(
            position=np.zeros(3),
            class_label=None,
            confidence=1.0,
            camera_indices=(0, 1),
            camera_ids=("camera",) * 65,
            reprojection_error=0.0,
            position_covariance=np.eye(3),
        )
    bounded_detection = Detection3D(
        position=np.zeros(3),
        class_label=None,
        confidence=1.0,
        camera_indices=(0, 1),
        reprojection_error=0.0,
        position_covariance=np.eye(3),
    )
    with pytest.raises(ValueError, match="at most 100000"):
        to_measurements([bounded_detection] * 100_001, timestamp=0.0)


def test_two_camera_assignment_maximizes_cardinality_before_ray_gap():
    intrinsics = np.diag([100.0, 100.0, 1.0])
    cameras = [
        Camera(intrinsics, np.eye(3), np.zeros(3), name="left"),
        Camera(intrinsics, np.eye(3), [-1.0, 0.0, 0.0], name="right"),
    ]
    # Candidate gaps form the adversarial graph
    #   A-X (smallest), A-Y, B-X; B-Y is gated out.
    # A greedy first edge returns one target. Exact bipartite assignment must
    # choose A-Y and B-X and recover two.
    detections = [
        _detection(0, [0.0, 0.0], camera_id="left"),
        _detection(0, [0.0, 20.0], camera_id="left"),
        _detection(1, [-10.0, 9.0], camera_id="right"),
        _detection(1, [-10.0, -10.0], camera_id="right"),
    ]
    fused = _correlate(
        cameras,
        detections,
        max_ray_gap=0.8,
        max_reprojection=6.0,
    )
    reversed_fused = _correlate(
        cameras,
        list(reversed(detections)),
        max_ray_gap=0.8,
        max_reprojection=6.0,
    )

    assert len(fused) == 2
    assert all(detection.camera_indices == (0, 1) for detection in fused)
    assert np.allclose(
        [detection.position for detection in fused],
        [detection.position for detection in reversed_fused],
    )


def test_two_camera_assignment_minimizes_total_gap_at_equal_cardinality():
    intrinsics = np.diag([100.0, 100.0, 1.0])
    cameras = [
        Camera(intrinsics, np.eye(3), np.zeros(3), name="left"),
        Camera(intrinsics, np.eye(3), [-1.0, 0.0, 0.0], name="right"),
    ]
    detections = [
        _detection(0, [0.0, 0.0], camera_id="left"),
        _detection(0, [0.0, 20.0], camera_id="left"),
        _detection(1, [-10.0, 1.0], camera_id="right"),
        _detection(1, [-10.0, 19.0], camera_id="right"),
    ]

    fused = _correlate(
        cameras,
        detections,
        max_ray_gap=1.0,
        max_reprojection=10.0,
    )

    assert len(fused) == 2
    # Both perfect matchings are admissible. The low-gap diagonal matching
    # triangulates near y=0.05 and y=1.95; the crossed matching is near y=1.
    assert np.allclose(
        sorted(float(detection.position[1]) for detection in fused),
        [0.049996, 1.94986],
        atol=1e-3,
    )


def test_invalid_multi_view_bridge_cannot_discard_valid_pair_hypothesis():
    cameras = _rig()
    first_target = np.zeros(3)
    # This second target lies on cam1's ray to first_target. AB and BC are
    # individually exact, but AC is outside the ray-gap gate. A connected-
    # component merge would create one invalid three-view fit and lose AB.
    second_target = cameras[1].center + 0.8 * (first_target - cameras[1].center)
    detections = [
        _detection(0, cameras[0].project(first_target), camera_id="cam0"),
        _detection(1, cameras[1].project(first_target), camera_id="cam1"),
        _detection(2, cameras[2].project(second_target), camera_id="cam2"),
    ]

    fused = _correlate(cameras, detections, max_ray_gap=8.0, max_reprojection=12.0)
    reversed_fused = _correlate(
        cameras,
        list(reversed(detections)),
        max_ray_gap=8.0,
        max_reprojection=12.0,
    )

    assert len(fused) == 1
    assert fused[0].camera_ids == ("cam0", "cam1")
    assert np.linalg.norm(fused[0].position - first_target) < 1e-6
    assert np.allclose(fused[0].position, reversed_fused[0].position)


def test_near_parallel_rays_are_rejected_before_dlt_or_association():
    intrinsics = np.diag([1000.0, 1000.0, 1.0])
    cameras = [
        Camera(intrinsics, np.eye(3), np.zeros(3), name="near0"),
        Camera(intrinsics, np.eye(3), [-0.01, 0.0, 0.0], name="near1"),
    ]
    far_point = np.array([0.0, 0.0, 10_000.0])
    pixels = [camera.project(far_point) for camera in cameras]
    with pytest.raises(ValueError, match="parallax"):
        triangulate_dlt(
            cameras,
            pixels,
            min_ray_angle_deg=1.0,
            max_range_m=20_000.0,
        )
    with pytest.raises(ValueError, match="parallax"):
        triangulate_midpoint(
            *cameras[0].backproject_ray(pixels[0]),
            *cameras[1].backproject_ray(pixels[1]),
            require_forward=True,
            min_ray_angle_deg=1.0,
        )
    detections = [
        _detection(index, pixel, camera_id=camera.name)
        for index, (camera, pixel) in enumerate(zip(cameras, pixels))
    ]
    assert (
        _correlate(
            cameras,
            detections,
            min_ray_angle_deg=1.0,
            max_range_m=20_000.0,
        )
        == []
    )


def test_positive_ray_cheirality_and_range_constraints():
    with pytest.raises(ValueError, match="behind"):
        triangulate_midpoint(
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            require_forward=True,
        )

    intrinsics = np.diag([100.0, 100.0, 1.0])
    cameras = [
        Camera(intrinsics, np.eye(3), np.zeros(3)),
        Camera(intrinsics, np.eye(3), [-1.0, 0.0, 0.0]),
    ]
    behind = np.array([0.0, 0.0, -5.0, 1.0])
    pixels = []
    for camera in cameras:
        homogeneous = camera.P @ behind
        pixels.append(homogeneous[:2] / homogeneous[2])
    with pytest.raises(ValueError, match="behind"):
        triangulate_dlt(cameras, pixels)

    far = np.array([0.0, 0.0, 500.0])
    far_pixels = [camera.project(far) for camera in cameras]
    with pytest.raises(ValueError, match="max_range"):
        triangulate_dlt(cameras, far_pixels, min_ray_angle_deg=0.1, max_range_m=100.0)


def test_correlation_is_deterministic_under_input_reversal():
    cameras = _rig()
    targets = [("bird", [-4.0, 2.0, 1.0]), ("drone", [5.0, -3.0, 2.0])]
    detections = [
        _detection(
            camera_index,
            camera.project(target),
            label,
            0.9,
            timestamp=20.0 + camera_index * 0.001,
            camera_id=camera.name,
        )
        for camera_index, camera in enumerate(cameras)
        for label, target in targets
    ]
    forward = _correlate(cameras, detections)
    reverse = _correlate(cameras, list(reversed(detections)))
    assert [detection.class_label for detection in forward] == [
        detection.class_label for detection in reverse
    ]
    assert np.allclose(
        [detection.position for detection in forward],
        [detection.position for detection in reverse],
    )
    assert np.allclose(
        [detection.position_covariance for detection in forward],
        [detection.position_covariance for detection in reverse],
    )
