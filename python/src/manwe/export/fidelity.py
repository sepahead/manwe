"""Export-fidelity gate: an exported model must match its PyTorch reference.

Gate every backend on the AP delta versus the FP32 PyTorch reference before any
performance claim or hand-off to crebain. The default tolerance is an absolute
0.005 AP (0.5 percentage points); callers may set a stricter small-object policy
for quantized exports. Small-object AP is always checked separately.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field

import numpy as np

from ..eval.detection import Detections, GroundTruth, iou_matrix, mean_average_precision
from ..fusion.association import GATE_INF, linear_assignment


@dataclass
class FidelityReport:
    """Threshold-specific export comparison.

    The ``*_map`` field names are retained for API compatibility. ``metric_name``
    and ``iou_threshold`` state their precise meaning (``mAP50`` by default); this
    report does not claim COCO mAP@[.50:.95].
    """

    ref_map: float
    exp_map: float
    delta_map: float
    ref_map_small: float
    exp_map_small: float
    delta_map_small: float
    tolerance: float
    small_tolerance: float
    passed: bool
    metric_name: str = "mAP50"
    small_metric_name: str = "mAP50-small"
    iou_threshold: float = 0.5
    frame_count: int = 0
    num_classes: int = 0
    evaluated_classes: int = 0
    evaluated_small_classes: int = 0
    class_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    small_class_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    ref_precision: float = 0.0
    exp_precision: float = 0.0
    delta_precision: float = 0.0
    ref_recall: float = 0.0
    exp_recall: float = 0.0
    delta_recall: float = 0.0
    ref_fppi: float = 0.0
    exp_fppi: float = 0.0
    delta_fppi: float = 0.0
    operating_tolerance: float = 0.0
    fppi_tolerance: float = 0.0
    agreement_iou: float = 0.95
    max_score_delta: float = 0.05
    observed_max_score_delta: float = 0.0
    matched_detections: int = 0
    missing_detections: int = 0
    extra_detections: int = 0
    required_classes: list[int] = field(default_factory=list)
    required_small_classes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        # Add explicit threshold-named aliases while preserving the historical
        # ref_map/exp_map/delta_map keys consumed by existing callers.
        metric_suffix = self.metric_name.removeprefix("mAP").lower().replace("-", "_")
        small_suffix = self.small_metric_name.removeprefix("mAP").lower().replace("-", "_")
        payload[f"ref_map{metric_suffix}"] = self.ref_map
        payload[f"exp_map{metric_suffix}"] = self.exp_map
        payload[f"delta_map{metric_suffix}"] = self.delta_map
        payload[f"ref_map{small_suffix}"] = self.ref_map_small
        payload[f"exp_map{small_suffix}"] = self.exp_map_small
        payload[f"delta_map{small_suffix}"] = self.delta_map_small
        return payload


def _validate_tolerance(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return tolerance


def _per_class_metrics(reference, exported, num_classes: int) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for class_index in range(num_classes):
        key = f"AP/{class_index}"
        if key not in reference:
            continue
        ref_ap = float(reference[key])
        exp_ap = float(exported[key])
        metrics[class_index] = {
            "reference": ref_ap,
            "exported": exp_ap,
            "drop": ref_ap - exp_ap,
        }
    return metrics


def _validate_probability(value: float, name: str, *, include_zero: bool = False) -> float:
    number = _validate_tolerance(value, name)
    lower_ok = number >= 0.0 if include_zero else number > 0.0
    if not lower_ok or number > 1.0:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return number


def _validate_required_classes(
    values: Collection[int] | None, num_classes: int, name: str
) -> list[int]:
    if values is None:
        return []
    if len(values) > num_classes:
        raise ValueError(f"{name} must contain at most {num_classes} class indices")
    result: set[int] = set()
    for value in values:
        if type(value) is not int or not 0 <= value < num_classes:
            raise ValueError(f"{name} must contain unique class indices in [0, {num_classes})")
        if value in result:
            raise ValueError(f"{name} must not contain duplicate class indices")
        result.add(value)
    if not result:
        raise ValueError(f"{name} must not be empty when provided")
    return sorted(result)


def _optimal_pairs(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    a_scores: np.ndarray | None = None,
    b_scores: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Maximize admissible matches, then IoU, then score similarity."""
    if len(a) == 0 or len(b) == 0:
        return []
    if (a_scores is None) != (b_scores is None):
        raise ValueError("both score arrays must be provided together")
    overlaps = iou_matrix(a, b)
    admissible = (overlaps >= threshold) | np.isclose(overlaps, threshold, rtol=0.0, atol=1e-12)
    geometry_cost = 1.0 - overlaps
    if a_scores is not None and b_scores is not None:
        geometry_cost = geometry_cost + np.abs(a_scores[:, None] - b_scores[None, :]) * 1e-12
    cost = np.full(overlaps.shape, GATE_INF)
    cost[admissible] = geometry_cost[admissible]
    return linear_assignment(cost)


def _require_aligned_image_ids(
    reference: list[Detections], exported: list[Detections], ground_truth: list[GroundTruth]
) -> None:
    sequences = {
        "reference": [frame.image_id for frame in reference],
        "exported": [frame.image_id for frame in exported],
        "ground_truth": [frame.image_id for frame in ground_truth],
    }
    for name, values in sequences.items():
        if any(value is None for value in values):
            raise ValueError(f"fidelity evaluation requires image_id on every {name} frame")
        if len(set(values)) != len(values):
            raise ValueError(f"fidelity evaluation requires unique {name} image_id values")
    if (
        sequences["reference"] != sequences["exported"]
        or sequences["reference"] != sequences["ground_truth"]
    ):
        raise ValueError("fidelity image_id sequences are not aligned")


def _within_limit(value: float, limit: float) -> bool:
    return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)


def _operating_metrics(
    predictions: list[Detections], ground_truth: list[GroundTruth], iou_thr: float
) -> tuple[float, float, float]:
    """Precision, recall and false positives per image for post-threshold detections."""
    total_predictions = 0
    total_ground_truth = 0
    true_positives = 0
    labels: set[int] = set()
    for prediction, truth in zip(predictions, ground_truth, strict=True):
        labels.update(int(value) for value in prediction.labels)
        labels.update(int(value) for value in truth.labels)
    for prediction, truth in zip(predictions, ground_truth, strict=True):
        total_predictions += len(prediction.boxes)
        total_ground_truth += len(truth.boxes)
        for class_index in labels:
            predicted = prediction.boxes[prediction.labels == class_index]
            actual = truth.boxes[truth.labels == class_index]
            true_positives += len(_optimal_pairs(predicted, actual, iou_thr))
    false_positives = total_predictions - true_positives
    precision = true_positives / total_predictions if total_predictions else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth else 0.0
    return precision, recall, false_positives / len(predictions)


def _prediction_agreement(
    reference: list[Detections], exported: list[Detections], agreement_iou: float
) -> tuple[int, int, int, float]:
    """Compare postprocessed outputs directly, class by class and frame by frame."""
    matched = 0
    missing = 0
    extra = 0
    score_deltas: list[float] = []
    for ref_frame, exp_frame in zip(reference, exported, strict=True):
        labels = sorted(set(ref_frame.labels.tolist()) | set(exp_frame.labels.tolist()))
        for class_index in labels:
            ref_mask = ref_frame.labels == class_index
            exp_mask = exp_frame.labels == class_index
            ref_boxes = ref_frame.boxes[ref_mask]
            exp_boxes = exp_frame.boxes[exp_mask]
            ref_scores = ref_frame.scores[ref_mask]
            exp_scores = exp_frame.scores[exp_mask]
            pairs = _optimal_pairs(ref_boxes, exp_boxes, agreement_iou, ref_scores, exp_scores)
            matched += len(pairs)
            missing += len(ref_boxes) - len(pairs)
            extra += len(exp_boxes) - len(pairs)
            score_deltas.extend(abs(float(ref_scores[i]) - float(exp_scores[j])) for i, j in pairs)
    return matched, missing, extra, max(score_deltas, default=0.0)


def fidelity_report(
    reference: list[Detections],
    exported: list[Detections],
    ground_truth: list[GroundTruth],
    num_classes: int,
    iou_thr: float = 0.5,
    tolerance: float = 0.005,
    small_tolerance: float | None = None,
    *,
    operating_tolerance: float | None = None,
    fppi_tolerance: float = 0.0,
    agreement_iou: float = 0.95,
    max_score_delta: float = 0.05,
    required_classes: Collection[int] | None = None,
    required_small_classes: Collection[int] | None = None,
) -> FidelityReport:
    """Compare an exported model's detections to the reference on shared frames.

    ``tolerance`` gates the overall mAP drop; ``small_tolerance`` (defaults to
    ``tolerance``) gates the AP-small drop. Both are absolute AP differences in
    ``[0, 1]`` (``0.005`` means 0.5 percentage points). ``passed`` also requires every
    measured class to stay within the corresponding tolerance, so gains in one
    class cannot hide a regression in another. Inputs are the detections emitted
    at the deployed confidence threshold. The gate therefore also compares
    precision, recall, false positives per image, and direct box/score agreement;
    AP alone does not penalize false positives after full recall.
    """
    if type(num_classes) is not int or not 1 <= num_classes <= 4096:
        raise ValueError("num_classes must be an integer in [1, 4096]")
    counts = {
        "reference": len(reference),
        "exported": len(exported),
        "ground_truth": len(ground_truth),
    }
    if 0 in counts.values():
        raise ValueError(f"fidelity evaluation requires nonempty frame lists, got {counts}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"fidelity frame count mismatch: {counts}")
    _require_aligned_image_ids(reference, exported, ground_truth)

    tolerance = _validate_probability(tolerance, "tolerance", include_zero=True)
    small_tol = (
        tolerance
        if small_tolerance is None
        else _validate_probability(small_tolerance, "small_tolerance", include_zero=True)
    )
    operating_tol = (
        tolerance
        if operating_tolerance is None
        else _validate_probability(operating_tolerance, "operating_tolerance", include_zero=True)
    )
    fppi_tol = _validate_tolerance(fppi_tolerance, "fppi_tolerance")
    agreement_threshold = _validate_probability(agreement_iou, "agreement_iou")
    score_tolerance = _validate_probability(max_score_delta, "max_score_delta", include_zero=True)
    required = _validate_required_classes(required_classes, num_classes, "required_classes")
    required_small = _validate_required_classes(
        required_small_classes, num_classes, "required_small_classes"
    )

    ref_metrics = mean_average_precision(reference, ground_truth, num_classes, iou_thr)
    exp_metrics = mean_average_precision(exported, ground_truth, num_classes, iou_thr)
    ref_small_metrics = mean_average_precision(
        reference, ground_truth, num_classes, iou_thr, area="small"
    )
    exp_small_metrics = mean_average_precision(
        exported, ground_truth, num_classes, iou_thr, area="small"
    )
    evaluated_classes = ref_metrics.evaluated_classes
    evaluated_small_classes = ref_small_metrics.evaluated_classes
    if evaluated_classes == 0:
        raise ValueError("fidelity evaluation has no in-range ground-truth classes")
    if evaluated_small_classes == 0:
        raise ValueError("fidelity evaluation has no small-object ground-truth coverage")
    if exp_metrics.evaluated_classes != evaluated_classes:
        raise ValueError("reference/export metric class coverage differs")
    if exp_small_metrics.evaluated_classes != evaluated_small_classes:
        raise ValueError("reference/export small-object class coverage differs")

    ref = float(ref_metrics["mAP"])
    exp = float(exp_metrics["mAP"])
    ref_s = float(ref_small_metrics["mAP"])
    exp_s = float(exp_small_metrics["mAP"])
    d, d_s = ref - exp, ref_s - exp_s
    class_metrics = _per_class_metrics(ref_metrics, exp_metrics, num_classes)
    small_class_metrics = _per_class_metrics(ref_small_metrics, exp_small_metrics, num_classes)
    missing_required = [value for value in required if value not in class_metrics]
    missing_required_small = [value for value in required_small if value not in small_class_metrics]
    if missing_required:
        raise ValueError(f"required classes have no ground-truth coverage: {missing_required}")
    if missing_required_small:
        raise ValueError(
            f"required small classes have no small-object ground-truth coverage: "
            f"{missing_required_small}"
        )
    class_gate_passed = all(
        _within_limit(values["drop"], tolerance) for values in class_metrics.values()
    )
    small_class_gate_passed = all(
        _within_limit(values["drop"], small_tol) for values in small_class_metrics.values()
    )
    ref_precision, ref_recall, ref_fppi = _operating_metrics(reference, ground_truth, iou_thr)
    exp_precision, exp_recall, exp_fppi = _operating_metrics(exported, ground_truth, iou_thr)
    delta_precision = ref_precision - exp_precision
    delta_recall = ref_recall - exp_recall
    delta_fppi = exp_fppi - ref_fppi
    matched, missing, extra, observed_score_delta = _prediction_agreement(
        reference, exported, agreement_threshold
    )
    operating_gate_passed = (
        _within_limit(delta_precision, operating_tol)
        and _within_limit(delta_recall, operating_tol)
        and _within_limit(delta_fppi, fppi_tol)
    )
    agreement_gate_passed = (
        missing == 0 and extra == 0 and _within_limit(observed_score_delta, score_tolerance)
    )
    return FidelityReport(
        ref_map=ref,
        exp_map=exp,
        delta_map=d,
        ref_map_small=ref_s,
        exp_map_small=exp_s,
        delta_map_small=d_s,
        tolerance=tolerance,
        small_tolerance=small_tol,
        passed=(
            _within_limit(d, tolerance)
            and _within_limit(d_s, small_tol)
            and class_gate_passed
            and small_class_gate_passed
            and operating_gate_passed
            and agreement_gate_passed
        ),
        metric_name=ref_metrics.metric_name,
        small_metric_name=ref_small_metrics.metric_name,
        iou_threshold=ref_metrics.iou_threshold,
        frame_count=counts["ground_truth"],
        num_classes=num_classes,
        evaluated_classes=evaluated_classes,
        evaluated_small_classes=evaluated_small_classes,
        class_metrics=class_metrics,
        small_class_metrics=small_class_metrics,
        ref_precision=ref_precision,
        exp_precision=exp_precision,
        delta_precision=delta_precision,
        ref_recall=ref_recall,
        exp_recall=exp_recall,
        delta_recall=delta_recall,
        ref_fppi=ref_fppi,
        exp_fppi=exp_fppi,
        delta_fppi=delta_fppi,
        operating_tolerance=operating_tol,
        fppi_tolerance=fppi_tol,
        agreement_iou=agreement_threshold,
        max_score_delta=score_tolerance,
        observed_max_score_delta=observed_score_delta,
        matched_detections=matched,
        missing_detections=missing,
        extra_detections=extra,
        required_classes=required,
        required_small_classes=required_small,
    )


__all__ = ["FidelityReport", "fidelity_report"]
