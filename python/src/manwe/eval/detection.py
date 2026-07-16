"""Detection metrics — IoU, average precision, and AP-small (numpy).

A dependency-free mAP for quick iteration and for the export-fidelity harness.
For canonical COCO numbers use ``pycocotools`` (in the ``vision`` extra); this is
a faithful all-point AP with a *simplified* small/medium/large split (it filters
by area rather than using COCO's ignore logic) — good enough to compare runs and
gate export drift, and documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fusion.association import GATE_INF, linear_assignment

# Keeping coordinates inside this envelope guarantees that subtracting any two
# accepted coordinates, multiplying box side lengths, and adding two box areas
# remain representable in float64. Real image coordinates are many orders of
# magnitude smaller; rejecting values outside the envelope is preferable to
# silently returning NaN IoUs for finite inputs.
_MAX_COORDINATE_MAGNITUDE = np.sqrt(np.finfo(np.float64).max) / 4.0
_MAX_BOXES_PER_FRAME = 100_000
_MAX_IOU_PAIRS = 4_000_000
_MAX_CLASSES = 4096
_MAX_EVALUATION_FRAMES = 10_000
_MAX_TOTAL_EVALUATION_BOXES = 1_000_000
_MAX_CLASS_MASK_WORK = 20_000_000

# COCO area ranges (pixels²).
AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32**2),
    "medium": (32**2, 96**2),
    "large": (96**2, float("inf")),
}


class DetectionMetricResult(dict[str, float]):
    """Backward-compatible numeric metric mapping with explicit metadata attributes."""

    def __init__(
        self,
        values: dict[str, float],
        *,
        metric_name: str,
        iou_threshold: float,
        area: str,
        frame_count: int,
        evaluated_classes: int,
    ) -> None:
        super().__init__(values)
        self.metric_name = metric_name
        self.iou_threshold = iou_threshold
        self.area = area
        self.frame_count = frame_count
        self.evaluated_classes = evaluated_classes


def _raw_real_array(value: object, error_message: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(error_message)
    return raw


def _float_array(raw: np.ndarray, error_message: str) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc


def _validated_boxes(boxes: np.ndarray, name: str) -> np.ndarray:
    error_message = f"{name} must be a real numeric array with shape (N, 4)"
    raw = _raw_real_array(boxes, error_message)
    if raw.shape == (0,):
        raw = raw.reshape(0, 4)
    elif raw.shape == (4,):
        raw = raw.reshape(1, 4)
    if raw.ndim != 2 or raw.shape[1:] != (4,):
        raise ValueError(f"{name} must have shape (N, 4), got {raw.shape}")
    if len(raw) > _MAX_BOXES_PER_FRAME:
        raise ValueError(f"{name} exceeds the {_MAX_BOXES_PER_FRAME}-box safety limit")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if np.any(np.abs(raw) > _MAX_COORDINATE_MAGNITUDE):
        raise ValueError(
            f"{name} coordinate magnitude exceeds the float64 geometry limit "
            f"{_MAX_COORDINATE_MAGNITUDE:g}"
        )
    array = _float_array(raw, error_message)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if len(array) and (np.any(array[:, 2] <= array[:, 0]) or np.any(array[:, 3] <= array[:, 1])):
        raise ValueError(f"{name} must contain positive-area xyxy boxes")
    return array


def _validated_scores(scores: np.ndarray, expected: int, name: str) -> np.ndarray:
    error_message = f"{name} must be a real numeric one-dimensional array"
    raw = _raw_real_array(scores, error_message)
    if raw.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {raw.shape}")
    if len(raw) != expected:
        raise ValueError(f"{name} length {len(raw)} does not match box count {expected}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite scores")
    array = _float_array(raw, error_message)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite scores")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _validated_ranking_scores(scores: np.ndarray, expected: int, name: str) -> np.ndarray:
    """Validate finite sortable AP scores without requiring probability units."""
    error_message = f"{name} must be a real numeric one-dimensional array"
    raw = _raw_real_array(scores, error_message)
    if raw.ndim != 1 or len(raw) != expected:
        raise ValueError(f"{name} must have shape ({expected},), got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite scores")
    array = _float_array(raw, error_message)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite scores")
    return array


def _validated_labels(labels: np.ndarray, expected: int, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {array.shape}")
    if len(array) != expected:
        raise ValueError(f"{name} length {len(array)} does not match box count {expected}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain numeric integer class indices")
    if np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
            raise ValueError(f"{name} must contain nonnegative class indices")
        if np.any(array >= _MAX_CLASSES):
            raise ValueError(f"{name} class indices must be smaller than {_MAX_CLASSES}")
        return array.astype(np.int64)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite class indices")
        if np.any(array != np.floor(array)):
            raise ValueError(f"{name} must contain integer class indices")
        if np.any((array < 0) | (array >= _MAX_CLASSES)):
            raise ValueError(f"{name} class indices must be in the range [0, {_MAX_CLASSES})")
        return array.astype(np.int64)
    raise ValueError(f"{name} must contain numeric integer class indices")


def _validated_iou_threshold(iou_thr: float) -> float:
    if isinstance(iou_thr, bool):
        raise ValueError("iou_thr must be a finite number in (0, 1]")
    try:
        threshold = float(iou_thr)
    except (TypeError, ValueError) as exc:
        raise ValueError("iou_thr must be a finite number in (0, 1]") from exc
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("iou_thr must be a finite number in (0, 1]")
    return threshold


def _metric_name(iou_thr: float, area: str = "all") -> str:
    percentage = iou_thr * 100.0
    suffix = (
        str(int(round(percentage)))
        if np.isclose(percentage, round(percentage))
        else f"{percentage:g}"
    )
    name = f"mAP{suffix}"
    return name if area == "all" else f"{name}-{area}"


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) source-image pixel xyxy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)
    image_id: str | int | None = None

    def __post_init__(self) -> None:
        self.boxes = _validated_boxes(self.boxes, "detections.boxes")
        self.scores = _validated_scores(self.scores, len(self.boxes), "detections.scores")
        self.labels = _validated_labels(self.labels, len(self.boxes), "detections.labels")
        self.image_id = _validated_image_id(self.image_id, "detections.image_id")


@dataclass
class GroundTruth:
    boxes: np.ndarray  # (M, 4) source-image pixel xyxy
    labels: np.ndarray  # (M,)
    image_id: str | int | None = None

    def __post_init__(self) -> None:
        self.boxes = _validated_boxes(self.boxes, "ground_truth.boxes")
        self.labels = _validated_labels(self.labels, len(self.boxes), "ground_truth.labels")
        self.image_id = _validated_image_id(self.image_id, "ground_truth.image_id")


def _validated_image_id(value: str | int | None, name: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{name} must be a nonempty string, integer, or None")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _validate_aligned_image_ids(preds: list[Detections], gts: list[GroundTruth]) -> None:
    pred_ids = [item.image_id for item in preds]
    gt_ids = [item.image_id for item in gts]
    if not any(value is not None for value in pred_ids + gt_ids):
        return
    if any(value is None for value in pred_ids + gt_ids):
        raise ValueError("image_id must be present on every prediction and ground-truth frame")
    if len(set(pred_ids)) != len(pred_ids) or len(set(gt_ids)) != len(gt_ids):
        raise ValueError("image_id values must be unique within each frame sequence")
    if pred_ids != gt_ids:
        raise ValueError("prediction and ground-truth image_id sequences are not aligned")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes → ``(len(a), len(b))``."""
    a = _validated_boxes(a, "a")
    b = _validated_boxes(b, "b")
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    if len(a) * len(b) > _MAX_IOU_PAIRS:
        raise ValueError(f"pairwise IoU exceeds the {_MAX_IOU_PAIRS}-pair safety limit")
    width_a = a[:, 2] - a[:, 0]
    height_a = a[:, 3] - a[:, 1]
    width_b = b[:, 2] - b[:, 0]
    height_b = b[:, 3] - b[:, 1]
    x_scale = np.maximum(width_a[:, None], width_b[None, :])
    y_scale = np.maximum(height_a[:, None], height_b[None, :])
    area_a = (width_a[:, None] / x_scale) * (height_a[:, None] / y_scale)
    area_b = (width_b[None, :] / x_scale) * (height_b[None, :] / y_scale)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = ((x2 - x1).clip(0) / x_scale) * ((y2 - y1).clip(0) / y_scale)
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    boxes = _validated_boxes(boxes, "boxes")
    return (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)


def _filter_area(boxes: np.ndarray, lo: float, hi: float, scores: np.ndarray | None = None):
    """Keep boxes whose area is in ``[lo, hi)``; filter ``scores`` in lockstep."""
    boxes = _validated_boxes(boxes, "boxes")
    areas = _box_areas(boxes)
    mask = (areas >= lo) & (areas < hi)
    if scores is None:
        return boxes[mask]
    validated_scores = _validated_scores(scores, len(boxes), "scores")
    return boxes[mask], validated_scores[mask]


def _all_point_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _canonical_box_order(boxes: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    """Order boxes independent of their input permutation.

    Score is the primary key when supplied; coordinates provide a deterministic
    convention for exact score ties. Identical boxes are interchangeable.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    coordinate_keys = (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
    keys = coordinate_keys if scores is None else (*coordinate_keys, -scores)
    return np.lexsort(keys)


def _match_frame(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonically ranked scores and binary matches for one frame."""
    order = _canonical_box_order(pred_boxes, pred_scores)
    ranked_scores = pred_scores[order]
    true_positives = np.zeros(len(order), dtype=float)
    if len(gt_boxes) == 0:
        return ranked_scores, true_positives

    canonical_gt = gt_boxes[_canonical_box_order(gt_boxes)]
    ious = iou_matrix(pred_boxes[order], canonical_gt)
    matched: set[int] = set()
    group_start = 0
    group_ends = np.flatnonzero(
        np.concatenate([ranked_scores[1:] != ranked_scores[:-1], np.array([True])])
    )
    for group_end in group_ends:
        available = [index for index in range(len(canonical_gt)) if index not in matched]
        if available:
            overlaps = ious[group_start : group_end + 1, available]
            admissible = overlaps >= iou_thr
            costs = np.where(admissible, 1.0 - overlaps, GATE_INF)
            for local_prediction, local_gt in linear_assignment(costs):
                true_positives[group_start + local_prediction] = 1.0
                matched.add(available[local_gt])
        group_start = int(group_end) + 1
    return ranked_scores, true_positives


def _ap_at_score_thresholds(scores: np.ndarray, tp: np.ndarray, total_gt: int) -> float:
    """Compute AP after admitting each complete equal-score group.

    A confidence threshold cannot distinguish predictions with identical
    scores. Evaluating only at the end of each tie group makes AP invariant to
    frame and input ordering instead of assigning an arbitrary TP/FP order.
    """
    if len(scores) == 0 or total_gt == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked_tp = tp[order]
    group_ends = np.flatnonzero(
        np.concatenate([ranked_scores[1:] != ranked_scores[:-1], np.array([True])])
    )
    tp_cum = np.cumsum(ranked_tp)[group_ends]
    predictions_cum = group_ends + 1
    recall = tp_cum / total_gt
    precision = tp_cum / predictions_cum
    return _all_point_ap(recall, precision)


def average_precision(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
) -> float:
    """All-point AP for a single class in one image.

    Use :func:`mean_average_precision` for multiple frames; concatenating frames
    here would permit boxes from different images to match each other.
    """
    pred_boxes = _validated_boxes(pred_boxes, "pred_boxes")
    pred_scores = _validated_ranking_scores(pred_scores, len(pred_boxes), "pred_scores")
    gt_boxes = _validated_boxes(gt_boxes, "gt_boxes")
    iou_thr = _validated_iou_threshold(iou_thr)
    n_gt = len(gt_boxes)
    if len(pred_boxes) == 0:
        return 0.0 if n_gt > 0 else 1.0
    scores, tp = _match_frame(pred_boxes, pred_scores, gt_boxes, iou_thr)
    return _ap_at_score_thresholds(scores, tp, n_gt)


def mean_average_precision(
    preds: list[Detections],
    gts: list[GroundTruth],
    num_classes: int,
    iou_thr: float = 0.5,
    area: str = "all",
) -> DetectionMetricResult:
    """mAP across images and classes, optionally restricted to an ``area`` range.

    The numeric mapping remains ``{"mAP": ..., "AP/<class>": ...}`` for
    compatibility. Metadata attributes such as ``metric_name == "mAP50"`` and
    ``iou_threshold == 0.5`` make clear that this is macro AP at one IoU
    threshold, not COCO mAP@[.50:.95]. Non-"all" ranges filter boxes by area (a
    simplified small-object view — see module docstring).
    """
    if type(num_classes) is not int or not 1 <= num_classes <= _MAX_CLASSES:
        raise ValueError(f"num_classes must be an integer in [1, {_MAX_CLASSES}]")
    iou_thr = _validated_iou_threshold(iou_thr)
    if not isinstance(area, str) or area not in AREA_RANGES:
        raise ValueError(f"unknown area {area!r}; expected one of {tuple(AREA_RANGES)}")
    if len(preds) != len(gts):
        raise ValueError(
            f"prediction/ground-truth frame count mismatch: {len(preds)} != {len(gts)}"
        )
    if not preds:
        raise ValueError("at least one prediction/ground-truth frame is required")
    if len(preds) > _MAX_EVALUATION_FRAMES:
        raise ValueError(f"evaluation exceeds the {_MAX_EVALUATION_FRAMES}-frame safety limit")
    validated_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for frame_index, (det, gt) in enumerate(zip(preds, gts, strict=True)):
        if not isinstance(det, Detections):
            raise ValueError(f"preds[{frame_index}] must be a Detections instance")
        if not isinstance(gt, GroundTruth):
            raise ValueError(f"gts[{frame_index}] must be a GroundTruth instance")
    _validate_aligned_image_ids(preds, gts)
    for frame_index, (det, gt) in enumerate(zip(preds, gts, strict=True)):
        db = _validated_boxes(det.boxes, f"preds[{frame_index}].boxes")
        dsc = _validated_scores(det.scores, len(db), f"preds[{frame_index}].scores")
        dl = _validated_labels(det.labels, len(db), f"preds[{frame_index}].labels")
        gb = _validated_boxes(gt.boxes, f"gts[{frame_index}].boxes")
        gl = _validated_labels(gt.labels, len(gb), f"gts[{frame_index}].labels")
        if np.any(dl >= num_classes):
            raise ValueError(
                f"preds[{frame_index}].labels contains an index outside [0, {num_classes})"
            )
        if np.any(gl >= num_classes):
            raise ValueError(
                f"gts[{frame_index}].labels contains an index outside [0, {num_classes})"
            )
        validated_frames.append((db, dsc, dl, gb, gl))

    total_boxes = sum(len(frame[0]) + len(frame[3]) for frame in validated_frames)
    if total_boxes > _MAX_TOTAL_EVALUATION_BOXES:
        raise ValueError(f"evaluation exceeds the {_MAX_TOTAL_EVALUATION_BOXES}-box safety limit")
    if total_boxes * num_classes > _MAX_CLASS_MASK_WORK:
        raise ValueError("evaluation exceeds the bounded class-by-box work limit")

    lo, hi = AREA_RANGES[area]
    per_class: dict[int, float] = {}
    for c in range(num_classes):
        scored: list[tuple[float, int]] = []  # (score, is_true_positive) across all images
        total_gt = 0
        for det_boxes, det_scores, det_labels, gt_boxes, gt_labels in validated_frames:
            dm = det_labels == c
            db, dsc = det_boxes[dm], det_scores[dm]
            gbx = gt_boxes[gt_labels == c]
            if area != "all":
                db, dsc = _filter_area(db, lo, hi, dsc)
                gbx = _filter_area(gbx, lo, hi)
            total_gt += len(gbx)
            if len(db) == 0:
                continue
            frame_scores, frame_tp = _match_frame(db, dsc, gbx, iou_thr)
            scored.extend(
                (float(score), int(is_tp))
                for score, is_tp in zip(frame_scores, frame_tp, strict=True)
            )
        if total_gt == 0:
            continue  # no ground truth for this class in this area range → not measured
        if not scored:
            per_class[c] = 0.0
            continue
        scores = np.array([item[0] for item in scored], dtype=float)
        tp = np.array([item[1] for item in scored], dtype=float)
        per_class[c] = _ap_at_score_thresholds(scores, tp, total_gt)
    mAP = float(np.mean(list(per_class.values()))) if per_class else 0.0
    metric_name = _metric_name(iou_thr, area)
    out = {"mAP": mAP}
    out.update({f"AP/{c}": value for c, value in per_class.items()})
    return DetectionMetricResult(
        out,
        metric_name=metric_name,
        iou_threshold=iou_thr,
        area=area,
        frame_count=len(preds),
        evaluated_classes=len(per_class),
    )


__all__ = [
    "AREA_RANGES",
    "DetectionMetricResult",
    "Detections",
    "GroundTruth",
    "iou_matrix",
    "average_precision",
    "mean_average_precision",
]
