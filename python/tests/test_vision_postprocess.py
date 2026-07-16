"""Vision postprocessing + crebain mapping (torch-free)."""

import numpy as np
import pytest

from manwe.common.device import Device
from manwe.vision import (
    Detection,
    Detector,
    crebain_class_map,
    letterbox_params,
    nms,
    resolve_ultralytics_device,
    results_to_detections,
    scale_boxes,
    xywh2xyxy,
)


def test_letterbox_params_centre_pad():
    ratio, (pw, ph) = letterbox_params((480, 640), new_size=640)
    assert abs(ratio - 1.0) < 1e-9  # 640 wide fits exactly
    assert abs(ph - 80.0) < 1e-6  # (640-480)/2 vertical pad
    assert abs(pw) < 1e-6


@pytest.mark.parametrize(
    ("orig_hw", "expected_ratio", "expected_pad"),
    [
        ((1, 256), 2.5, (0.0, 319.0)),
        ((2, 256), 2.5, (0.0, 317.0)),
        ((256, 1), 2.5, (319.0, 0.0)),
        # 1 * 5 / 2 == 2.5 exercises Python/Ultralytics ties-to-even rounding.
        ((2, 1), 2.5, (1.0, 0.0)),
    ],
)
def test_letterbox_params_match_actual_ultralytics_left_top_padding(
    orig_hw: tuple[int, int],
    expected_ratio: float,
    expected_pad: tuple[float, float],
):
    ratio, pad = letterbox_params(orig_hw, new_size=640 if max(orig_hw) == 256 else 5)
    assert ratio == expected_ratio
    assert pad == expected_pad


def test_xywh_and_scale_roundtrip():
    xywh = np.array([[100.0, 100.0, 40.0, 20.0]])
    xyxy = xywh2xyxy(xywh)
    assert np.allclose(xyxy[0], [80, 90, 120, 110])
    # scale back from a letterboxed 640 space to a 1280x960 original
    ratio, pad = letterbox_params((960, 1280), 640)
    orig = scale_boxes(xyxy, ratio, pad, (960, 1280))
    # ratio=0.5, pad=(0,80): inverse-letterbox maps [80,90,120,110] → [160,20,240,60]
    assert np.allclose(orig[0], [160.0, 20.0, 240.0, 60.0])


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep and 2 in keep  # the near-duplicate (idx 1) is dropped
    assert 1 not in keep


def test_crebain_class_map_from_coco_and_native():
    coco = crebain_class_map({4: "airplane", 14: "bird", 2: "car"})
    assert coco == {4: 2, 14: 1}  # airplane→aircraft(2), bird→bird(1); car dropped
    native = crebain_class_map({0: "drone", 1: "helicopter"})
    assert native == {0: 0, 1: 3}
    normalized = crebain_class_map({0: "  DRONE\t", 4: " AirPlane "})
    assert normalized == {0: 0, 4: 2}


def test_results_to_detections_drops_unmapped():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    confs = np.array([0.9, 0.8])
    cls = np.array([0, 1])  # 0→"drone", 1→"car" (dropped)
    dets = results_to_detections(boxes, confs, cls, {0: "drone", 1: "car"})
    assert len(dets) == 1
    assert dets[0].crebain_class == "drone"
    assert dets[0].class_index == 0


def test_detection_multicam_bridge_requires_explicit_geometry_uncertainty():
    detection = Detection(np.array([10.0, 20.0, 30.0, 40.0]), 0.9, "drone", 0)
    bridged = detection.to_detection2d(
        2,
        pixels_undistorted=True,
        pixel_std_px=1.5,
        timestamp=12.0,
        camera_id="cam2",
        timestamp_std_s=0.002,
    )
    assert np.array_equal(bridged.pixel, [20.0, 30.0])
    assert bridged.pixel_std_px == 1.5
    assert bridged.timestamp == 12.0
    assert bridged.camera_id == "cam2"
    with pytest.raises(ValueError, match="explicitly True"):
        detection.to_detection2d(2, pixels_undistorted=False, pixel_std_px=1.5)
    with pytest.raises(ValueError, match="read-only"):
        detection.bbox[0] = 0.0


def test_postprocess_rejects_malformed_or_nonfinite_outputs():
    with pytest.raises(ValueError, match="match the box count"):
        nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([]))
    with pytest.raises(ValueError, match="probabilities"):
        nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([np.nan]))
    with pytest.raises(ValueError, match="equal lengths"):
        results_to_detections(
            np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([]), np.array([0]), {0: "drone"}
        )
    with pytest.raises(ValueError, match="integer indices"):
        results_to_detections(
            np.array([[0.0, 0.0, 1.0, 1.0]]),
            np.array([0.9]),
            np.array([0.5]),
            {0: "drone"},
        )
    with pytest.raises(ValueError, match=r"\[0, 4096\)"):
        results_to_detections(
            np.array([[0.0, 0.0, 1.0, 1.0]]),
            np.array([0.9]),
            np.array([2**63], dtype=np.uint64),
            {0: "drone"},
        )

    maximum = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="coordinate magnitude"):
        nms(
            np.array([[-maximum, -maximum, maximum, maximum]]),
            np.array([0.9]),
        )
    with pytest.raises(ValueError, match="representable float64 range"):
        scale_boxes(
            np.array([[1.0, 1.0, 2.0, 2.0]]),
            np.nextafter(0.0, 1.0),
            (0.0, 0.0),
            (10, 10),
        )


def test_postprocess_rejects_coercive_and_unrepresentable_numeric_inputs():
    huge = 10**1000
    valid_boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
    for boxes in (
        [["0", "0", "1", "1"]],
        [[False, False, True, True]],
        np.ones((1, 4), dtype=complex),
        [[0, 0, huge, 1]],
    ):
        with pytest.raises(ValueError, match="real numeric array"):
            nms(boxes, np.array([0.5]))

    for scores in (["0.5"], [True], np.array([0.5 + 0.0j]), [huge]):
        with pytest.raises(ValueError, match="real numeric one-dimensional"):
            nms(valid_boxes, scores)

    for threshold in ("0.5", True, huge, np.array([0.5])):
        with pytest.raises(ValueError, match="iou_threshold"):
            nms(valid_boxes, np.array([0.5]), threshold)

    for ratio in ("1", True, huge):
        with pytest.raises(ValueError, match="ratio"):
            scale_boxes(valid_boxes, ratio, (0.0, 0.0), (2, 2))
    for pad in (("0", "0"), (False, False), (huge, 0)):
        with pytest.raises(ValueError, match="pad"):
            scale_boxes(valid_boxes, 1.0, pad, (2, 2))

    with pytest.raises(ValueError, match="real numeric array"):
        xywh2xyxy([["1", "1", "1", "1"]])


def test_detection_boundaries_reject_coercive_numeric_inputs_before_backend_use():
    huge = 10**1000
    for bbox in (
        ["0", "0", "1", "1"],
        [False, False, True, True],
        np.ones(4, dtype=complex),
        [0, 0, huge, 1],
    ):
        with pytest.raises(ValueError, match="real numeric"):
            Detection(bbox, 0.5, "drone", 0)
    for confidence in ("0.5", True, 0.5 + 0.0j, huge):
        with pytest.raises(ValueError, match="confidence"):
            Detection([0, 0, 1, 1], confidence, "drone", 0)

    for boxes in (
        [["0", "0", "1", "1"]],
        [[False, False, True, True]],
        np.ones((1, 4), dtype=complex),
        [[0, 0, huge, 1]],
    ):
        with pytest.raises(ValueError, match="detector boxes"):
            results_to_detections(boxes, [0.5], [0], {0: "drone"})
    for confidences in (["0.5"], [True], np.array([0.5 + 0.0j]), [huge]):
        with pytest.raises(ValueError, match="detector confidences"):
            results_to_detections([[0, 0, 1, 1]], confidences, [0], {0: "drone"})

    # Threshold validation precedes path resolution and optional backend import.
    for field, value in (("conf", "0.5"), ("conf", huge), ("iou", True), ("iou", huge)):
        kwargs = {field: value}
        with pytest.raises(ValueError, match=field):
            Detector("missing.pt", expected_sha256="0" * 64, **kwargs)


def test_postprocess_threshold_boundaries_are_exact_and_ties_are_stable():
    boxes = np.array([[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    # Standard NMS suppresses only overlaps strictly above the threshold.
    assert nms(boxes, np.array([0.9, 0.8]), iou_threshold=0.5) == [0, 1]
    separate = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 3.0, 1.0]])
    assert nms(separate, np.array([0.5, 0.5])) == [0, 1]

    overlapping = np.array([[1.0, 0.0, 11.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    forward = nms(overlapping, np.array([0.5, 0.5]), iou_threshold=0.5)
    reverse = nms(overlapping[::-1], np.array([0.5, 0.5]), iou_threshold=0.5)
    assert np.array_equal(overlapping[forward], overlapping[::-1][reverse])

    tiny = np.array([[0.0, 0.0, 1e-200, 1e-200], [0.0, 0.0, 1e-200, 1e-200]])
    assert len(nms(tiny, np.array([0.9, 0.8]), iou_threshold=0.5)) == 1
    elongated = np.array([[0.0, 0.0, 1e153, 1e-171], [0.0, 0.0, 1e153, 1e-171]])
    assert nms(elongated, np.array([0.9, 0.8]), iou_threshold=0.5) == [0]
    with pytest.raises(ValueError, match="too small"):
        xywh2xyxy(np.array([[1e15, 1e15, 1e-10, 1e-10]]))


def test_nms_exact_boundary_is_scale_invariant():
    # Intersection=32 and union=160, so IoU is exactly 1/5. The former
    # normalized-area quotient rounded this one ULP above float(0.2), suppressing
    # the second box despite the strict-above threshold contract.
    boxes = np.array([[3.0, 0.0, 10.0, 8.0], [-1.0, -5.0, 7.0, 12.0]])
    scores = np.array([0.9, 0.8])
    for scale in (np.ldexp(1.0, -600), 1.0, np.ldexp(1.0, 500)):
        scaled = boxes * scale
        assert nms(scaled, scores, iou_threshold=0.2) == [0, 1]
        assert nms(scaled, scores, iou_threshold=0.19) == [0]
        assert nms(scaled, scores, iou_threshold=0.0) == [0]


def test_resolve_ultralytics_device():
    assert resolve_ultralytics_device(Device("cuda", index=1)) == "1"
    assert resolve_ultralytics_device(Device("mps")) == "mps"
    assert resolve_ultralytics_device(Device("cpu")) == "cpu"
