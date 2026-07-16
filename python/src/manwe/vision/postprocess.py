"""Detection postprocessing — letterbox, NMS, coordinate scaling, class remap.

Pure-numpy so it is testable without torch and reusable by every backend
(training, ONNX/CoreML inference, the Rust bridge validation). The letterbox and
scaling conventions match Ultralytics so exported models stay contract-faithful.
"""

from __future__ import annotations

import numpy as np

from ..common.contracts import CREBAIN_CLASSES, coco_to_crebain, crebain_class_index

_MAX_COORDINATE_MAGNITUDE = np.sqrt(np.finfo(np.float64).max) / 4.0
_MAX_EXACT_IMAGE_DIMENSION = 2**53
_MAX_BOX_ARRAY_BOXES = 100_000
_MAX_NMS_BOXES = 4096
_MAX_NMS_PAIR_WORK = 10_000_000
_MAX_MODEL_CLASSES = 4096
# The intersection-relative IoU predicate below performs two divisions and two
# products per area ratio, then one addition, subtraction, and final threshold
# product. Sixteen float64 epsilons conservatively cover the accumulated forward
# error at the exact suppression boundary; ambiguous cases are retained because
# NMS is specified to suppress only overlaps strictly above the threshold.
_NMS_BOUNDARY_ERROR = 16.0 * np.finfo(np.float64).eps


def _positive_size(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > _MAX_EXACT_IMAGE_DIMENSION:
        raise ValueError(
            f"{name} exceeds the largest image dimension represented exactly in float64"
        )
    return value


def _require_representable_coordinates(value: np.ndarray, name: str) -> None:
    if np.any(np.abs(value) > _MAX_COORDINATE_MAGNITUDE):
        raise ValueError(
            f"{name} coordinate magnitude exceeds the float64 geometry limit "
            f"{_MAX_COORDINATE_MAGNITUDE:g}"
        )


def _raw_real_numeric_array(value: object, error_message: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(error_message)
    return raw


def _float_numeric_array(raw: np.ndarray, error_message: str) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc


def _real_numeric_scalar(value: object, error_message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(error_message)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc


def _round_ratio_ties_to_even(numerator: int, denominator: int) -> int:
    """Round a nonnegative rational exactly using Python's ties-to-even rule."""
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (twice_remainder == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


def _boxes(
    value: np.ndarray,
    name: str,
    *,
    require_positive_area: bool = True,
    max_boxes: int = _MAX_BOX_ARRAY_BOXES,
    limit_message: str | None = None,
) -> np.ndarray:
    error_message = f"{name} must be a real numeric array ending in four coordinates"
    raw = _raw_real_numeric_array(
        value,
        error_message,
    )
    if raw.shape == (4,):
        raw = raw.reshape(1, 4)
    if raw.ndim != 2 or raw.shape[1:] != (4,):
        raise ValueError(f"{name} must have shape (N, 4), got {raw.shape}")
    if len(raw) > max_boxes:
        raise ValueError(limit_message or f"{name} exceeds the {max_boxes}-box safety limit")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite coordinates")
    _require_representable_coordinates(raw, name)
    boxes = _float_numeric_array(raw, error_message)
    if not np.all(np.isfinite(boxes)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if require_positive_area and (
        np.any(boxes[..., 2] <= boxes[..., 0]) or np.any(boxes[..., 3] <= boxes[..., 1])
    ):
        raise ValueError(f"{name} must contain positive-area xyxy boxes")
    return boxes


def letterbox_params(
    orig_hw: tuple[int, int], new_size: int = 640
) -> tuple[float, tuple[float, float]]:
    """Return ``(ratio, (pad_w, pad_h))`` to fit ``orig_hw`` into a square canvas.

    Matches Ultralytics letterboxing: single scale, centred padding.
    """
    if not isinstance(orig_hw, tuple) or len(orig_hw) != 2:
        raise ValueError("orig_hw must be a (height, width) tuple")
    h = _positive_size(orig_hw[0], "height")
    w = _positive_size(orig_hw[1], "width")
    new_size = _positive_size(new_size, "new_size")
    if h >= w:
        r = new_size / h
        new_w = _round_ratio_ties_to_even(w * new_size, h)
        new_h = new_size
    else:
        r = new_size / w
        new_w = new_size
        new_h = _round_ratio_ties_to_even(h * new_size, w)

    # Ultralytics distributes odd padding asymmetrically. Its inverse transform
    # subtracts the actual left/top integer border, not half the total padding.
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    return r, (float(pad_left), float(pad_top))


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h]`` boxes to ``[x1, y1, x2, y2]``."""
    error_message = "boxes must be a real numeric array ending in four coordinates"
    raw = _raw_real_numeric_array(
        boxes,
        error_message,
    )
    if raw.shape == (4,):
        raw = raw.reshape(1, 4)
    if raw.ndim < 2 or raw.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (..., 4), got {raw.shape}")
    box_count = raw.size // 4
    if box_count > _MAX_BOX_ARRAY_BOXES:
        raise ValueError(f"boxes exceed the {_MAX_BOX_ARRAY_BOXES}-box safety limit")
    if not np.all(np.isfinite(raw)):
        raise ValueError("boxes must contain only finite values")
    _require_representable_coordinates(raw, "boxes")
    boxes = _float_numeric_array(raw, error_message)
    if not np.all(np.isfinite(boxes)):
        raise ValueError("boxes must contain only finite values")
    if np.any(boxes[..., 2:] <= 0.0):
        raise ValueError("xywh widths and heights must be positive")
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    _require_representable_coordinates(out, "converted boxes")
    if np.any(out[..., 2] <= out[..., 0]) or np.any(out[..., 3] <= out[..., 1]):
        raise ValueError("xywh boxes are too small to remain positive-area in float64")
    return out


def scale_boxes(
    boxes_xyxy: np.ndarray, ratio: float, pad: tuple[float, float], orig_hw: tuple[int, int]
) -> np.ndarray:
    """Map boxes from letterboxed space back to original image pixels."""
    boxes = _boxes(boxes_xyxy, "boxes_xyxy").copy()
    ratio = _real_numeric_scalar(ratio, "ratio must be finite and positive")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be finite and positive")
    if not isinstance(pad, tuple) or len(pad) != 2:
        raise ValueError("pad must be a finite (pad_w, pad_h) tuple")
    pad_error = "pad must be a finite (pad_w, pad_h) tuple"
    raw_pad = _raw_real_numeric_array(pad, pad_error)
    if raw_pad.shape != (2,) or not np.all(np.isfinite(raw_pad)):
        raise ValueError("pad must be a finite (pad_w, pad_h) tuple")
    _require_representable_coordinates(raw_pad, "pad")
    pad_values = _float_numeric_array(raw_pad, pad_error)
    if not np.all(np.isfinite(pad_values)):
        raise ValueError("pad must be a finite (pad_w, pad_h) tuple")
    pad_w, pad_h = pad_values
    boxes[..., [0, 2]] -= pad_w
    boxes[..., [1, 3]] -= pad_h
    with np.errstate(over="ignore", invalid="ignore"):
        boxes[..., :4] /= ratio
    if not np.all(np.isfinite(boxes)):
        raise ValueError("inverse letterbox coordinates exceed the representable float64 range")
    if not isinstance(orig_hw, tuple) or len(orig_hw) != 2:
        raise ValueError("orig_hw must be a (height, width) tuple")
    h = _positive_size(orig_hw[0], "height")
    w = _positive_size(orig_hw[1], "width")
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    if len(boxes) and (np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])):
        raise ValueError("clipping produced a zero-area box; discard out-of-frame boxes first")
    return boxes


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    *,
    labels: np.ndarray | None = None,
) -> list[int]:
    """Greedy NMS, optionally class-aware, returning score-descending indices."""
    work_limit_message = "NMS input exceeds the bounded quadratic-work limit"
    boxes = _boxes(
        boxes_xyxy,
        "boxes_xyxy",
        max_boxes=_MAX_NMS_BOXES,
        limit_message=work_limit_message,
    )
    score_error = "scores must be a real numeric one-dimensional array"
    raw_scores = _raw_real_numeric_array(scores, score_error)
    if raw_scores.ndim != 1 or len(raw_scores) != len(boxes):
        raise ValueError("scores must have shape (N,) and match the box count")
    if not np.all(np.isfinite(raw_scores)) or np.any((raw_scores < 0.0) | (raw_scores > 1.0)):
        raise ValueError("scores must contain finite probabilities in [0, 1]")
    scores = _float_numeric_array(raw_scores, score_error)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain finite probabilities in [0, 1]")
    validated_labels: np.ndarray | None = None
    if labels is not None:
        validated_labels = np.asarray(labels)
        if validated_labels.ndim != 1 or len(validated_labels) != len(boxes):
            raise ValueError("labels must have shape (N,) and match the box count")
        if np.issubdtype(validated_labels.dtype, np.bool_) or not np.issubdtype(
            validated_labels.dtype, np.integer
        ):
            raise ValueError("labels must contain integer class indices")
        if np.any(validated_labels < 0):
            raise ValueError("labels must contain nonnegative class indices")
    iou_threshold = _real_numeric_scalar(iou_threshold, "iou_threshold must be in [0, 1]")
    if not np.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if len(boxes) == 0:
        return []
    if len(boxes) > _MAX_NMS_BOXES or len(boxes) * (len(boxes) - 1) // 2 > _MAX_NMS_PAIR_WORK:
        raise ValueError(work_limit_message)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    original_indices = np.arange(len(boxes))
    tie_keys: tuple[np.ndarray, ...] = (original_indices, y2, x2, y1, x1)
    if validated_labels is not None:
        tie_keys = (*tie_keys, validated_labels)
    order = np.lexsort((*tie_keys, -scores))
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        width_i = x2[i] - x1[i]
        height_i = y2[i] - y1[i]
        width_rest = x2[rest] - x1[rest]
        height_rest = y2[rest] - y1[rest]
        inter_width = (xx2 - xx1).clip(0)
        inter_height = (yy2 - yy1).clip(0)
        overlaps = (inter_width > 0.0) & (inter_height > 0.0)
        keep_mask = ~overlaps
        if iou_threshold > 0.0 and np.any(overlaps):
            # For a positive intersection I, IoU <= t exactly when
            #
            #   t * (area_i / I + area_rest / I - 1) >= 1.
            #
            # Factoring each area ratio by the intersection dimensions keeps
            # every factor >= 1. This avoids both area underflow for tiny boxes
            # and area overflow for huge or elongated boxes. An infinite ratio
            # means the positive IoU rounded below float64 range and is therefore
            # safely below every positive threshold.
            overlap_width = inter_width[overlaps]
            overlap_height = inter_height[overlaps]
            with np.errstate(over="ignore", invalid="ignore"):
                relative_union = (
                    (width_i / overlap_width) * (height_i / overlap_height)
                    + (width_rest[overlaps] / overlap_width)
                    * (height_rest[overlaps] / overlap_height)
                    - 1.0
                )
                scaled_threshold = iou_threshold * relative_union
            keep_mask[overlaps] = scaled_threshold >= 1.0 - _NMS_BOUNDARY_ERROR
        if validated_labels is not None:
            keep_mask |= validated_labels[rest] != validated_labels[i]
        order = rest[keep_mask]
    return keep


def crebain_class_map(model_names: dict[int, str]) -> dict[int, int]:
    """Map a model's own class ids to crebain class indices.

    If a model is already trained on the crebain taxonomy, its names map straight
    through. A COCO model routes through :func:`coco_to_crebain`. Names with no
    aerial counterpart are dropped (absent from the returned dict).
    """
    if not isinstance(model_names, dict) or not model_names:
        raise ValueError("model_names must be a nonempty integer-keyed class mapping")
    if len(model_names) > _MAX_MODEL_CLASSES:
        raise ValueError(f"model_names exceeds the {_MAX_MODEL_CLASSES}-class safety limit")
    if any(type(index) is not int or not 0 <= index < _MAX_MODEL_CLASSES for index in model_names):
        raise ValueError(f"model class ids must be integers in [0, {_MAX_MODEL_CLASSES})")
    out: dict[int, int] = {}
    normalized_names: list[str] = []
    for mid, name in model_names.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not name.strip().isprintable()
            or len(name.strip().encode("utf-8")) > 256
        ):
            raise ValueError(f"model class name for id {mid} must be a bounded printable string")
        normalized_name = name.strip()
        normalized_names.append(normalized_name)
        name_l = normalized_name.lower()
        if name_l in CREBAIN_CLASSES:
            out[mid] = crebain_class_index(name_l)
            continue
        mapped = coco_to_crebain(name_l)
        if mapped is not None:
            out[mid] = crebain_class_index(mapped)
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("model class names must be unique")
    return out


__all__ = [
    "letterbox_params",
    "xywh2xyxy",
    "scale_boxes",
    "nms",
    "crebain_class_map",
]
