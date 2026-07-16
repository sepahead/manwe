"""Detection metrics: IoU, AP, mAP, and AP-small filtering."""

import numpy as np
import pytest

from manwe.eval.detection import (
    Detections,
    GroundTruth,
    average_precision,
    iou_matrix,
    mean_average_precision,
)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], float)
    ious = iou_matrix(a, b)
    assert abs(ious[0, 0] - 1.0) < 1e-9
    assert ious[0, 1] == 0.0


def test_average_precision_perfect_and_empty():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    assert average_precision(gt, np.array([0.9, 0.8]), gt, 0.5) == 1.0
    # predictions but no ground truth → AP 0
    assert average_precision(gt, np.array([0.9, 0.8]), np.empty((0, 4)), 0.5) == 0.0
    # no predictions, has GT → AP 0
    assert average_precision(np.empty((0, 4)), np.array([]), gt, 0.5) == 0.0


def test_map_perfect_predictions():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]], float)
    labels = np.array([0, 1, 2])
    preds = [Detections(boxes, np.array([0.9, 0.85, 0.8]), labels)]
    gts = [GroundTruth(boxes, labels)]
    out = mean_average_precision(preds, gts, num_classes=5)
    assert abs(out["mAP"] - 1.0) < 1e-9


def test_map_drops_on_missing_detection():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]], float)
    labels = np.array([0, 1, 2])
    gts = [GroundTruth(boxes, labels)]
    # drop the class-2 detection → its AP is 0 → mAP = 2/3
    preds = [Detections(boxes[:2], np.array([0.9, 0.85]), labels[:2])]
    out = mean_average_precision(preds, gts, num_classes=5)
    assert abs(out["mAP"] - (2.0 / 3.0)) < 1e-6


def test_ap_small_only_counts_small_boxes():
    small = np.array([[0, 0, 10, 10]], float)  # area 100 < 32²
    large = np.array([[0, 0, 200, 200]], float)  # area 40000 > 96²
    boxes = np.vstack([small, large])
    labels = np.array([0, 0])
    preds = [Detections(boxes, np.array([0.9, 0.9]), labels)]
    gts = [GroundTruth(boxes, labels)]
    out_small = mean_average_precision(preds, gts, num_classes=1, area="small")
    # only the small box is measured, and it is perfectly detected
    assert abs(out_small["mAP"] - 1.0) < 1e-9


def test_map_rejects_unbounded_class_by_box_work():
    count = 5_000
    x = np.arange(count, dtype=float) * 2.0
    boxes = np.column_stack((x, np.zeros(count), x + 1.0, np.ones(count)))
    labels = np.zeros(count, dtype=int)
    predictions = [Detections(boxes, np.ones(count), labels, image_id="frame")]
    truth = [GroundTruth(np.empty((0, 4)), np.empty(0, dtype=int), image_id="frame")]
    with pytest.raises(ValueError, match="class-by-box work"):
        mean_average_precision(predictions, truth, num_classes=4096)


def test_metric_input_caps_and_shape_checks_precede_float_widening(monkeypatch):
    from manwe.eval import detection as detection_module

    oversized_boxes = np.broadcast_to(
        np.array([0, 0, 1, 1], dtype=np.int8),
        (detection_module._MAX_BOXES_PER_FRAME + 1, 4),
    )
    malformed_boxes = np.broadcast_to(np.array(1, dtype=np.int8), (1, 4_000_000))
    mismatched_scores = np.broadcast_to(np.array(1, dtype=np.int8), (4_000_000,))
    forbidden = (oversized_boxes, malformed_boxes, mismatched_scores)
    real_float_array = detection_module._float_array

    def guarded_float_array(raw, error_message):
        if any(
            raw.shape == rejected.shape and np.shares_memory(raw, rejected)
            for rejected in forbidden
        ):
            pytest.fail("evaluation input was widened before raw admission")
        return real_float_array(raw, error_message)

    monkeypatch.setattr(detection_module, "_float_array", guarded_float_array)

    with pytest.raises(ValueError, match="box safety limit"):
        GroundTruth(oversized_boxes, np.zeros(len(oversized_boxes), dtype=np.int8))
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        GroundTruth(malformed_boxes, np.zeros(1, dtype=np.int8))
    with pytest.raises(ValueError, match="length"):
        Detections(
            np.array([[0, 0, 1, 1]], dtype=np.int8),
            mismatched_scores,
            np.array([0], dtype=np.int8),
        )


def test_metric_inputs_reject_object_coercion_before_float_conversion():
    class Coercive:
        calls = 0

        def __float__(self):
            type(self).calls += 1
            return 1.0

    with pytest.raises(ValueError, match="real numeric"):
        Detections(
            np.array([Coercive(), Coercive(), Coercive(), Coercive()], dtype=object),
            np.array([0.5]),
            np.array([0]),
        )
    assert Coercive.calls == 0
