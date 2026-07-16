"""Run a detector and map its output into the crebain class taxonomy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.artifacts import ArtifactSnapshot, require_pickle_acknowledgement
from ..common.contracts import CREBAIN_CLASSES
from ..common.deps import require
from ..common.ultralytics import harden_ultralytics_runtime, verify_ultralytics_policy
from .input import prepare_single_image
from .postprocess import (
    _float_numeric_array,
    _raw_real_numeric_array,
    _real_numeric_scalar,
    crebain_class_map,
)

_MAX_PIXEL_MAGNITUDE = 1e9
_MAX_DETECTIONS = 100_000
_MAX_MODEL_CLASSES = 4096


def _ultralytics_bgr_input(prepared_image: object) -> np.ndarray:
    """Convert Manwe's owned RGB image into Ultralytics' NumPy BGR convention."""
    try:
        rgb = (
            prepared_image
            if isinstance(prepared_image, np.ndarray)
            else np.asarray(prepared_image, dtype=np.uint8)
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - internal contract guard
        raise ValueError("prepared image must be a uint8 RGB image") from exc
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("prepared image must be uint8 RGB with shape (height, width, 3)")
    # Ultralytics interprets NumPy inputs as OpenCV-style BGR, while it interprets
    # Pillow inputs as RGB. Always pass one owned, contiguous representation so an
    # RGB ndarray and the same pixels loaded from disk have identical semantics.
    return np.array(rgb[..., ::-1], dtype=np.uint8, order="C", copy=True)


@dataclass(frozen=True, slots=True)
class Detection:
    """An image-space detection already mapped to a crebain class."""

    bbox: np.ndarray  # [x1, y1, x2, y2] pixels
    confidence: float
    crebain_class: str  # one of CREBAIN_CLASSES
    class_index: int

    def __post_init__(self) -> None:
        bbox_error = "bbox must contain four real numeric xyxy coordinates"
        raw_bbox = _raw_real_numeric_array(
            self.bbox,
            bbox_error,
        )
        if raw_bbox.shape != (4,) or not np.all(np.isfinite(raw_bbox)):
            raise ValueError("bbox must contain four finite xyxy coordinates")
        if np.any(np.abs(raw_bbox) > _MAX_PIXEL_MAGNITUDE):
            raise ValueError(f"bbox coordinates must not exceed {_MAX_PIXEL_MAGNITUDE:g} pixels")
        bbox = _float_numeric_array(raw_bbox, bbox_error)
        if not np.all(np.isfinite(bbox)):
            raise ValueError("bbox must contain four finite xyxy coordinates")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox must have positive area")
        confidence = _real_numeric_scalar(
            self.confidence,
            "confidence must be a finite probability in [0, 1]",
        )
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite probability in [0, 1]")
        if self.crebain_class not in CREBAIN_CLASSES:
            raise ValueError(f"unknown crebain class {self.crebain_class!r}")
        if type(self.class_index) is not int or not 0 <= self.class_index < len(CREBAIN_CLASSES):
            raise ValueError("class_index is outside the crebain taxonomy")
        if CREBAIN_CLASSES[self.class_index] != self.crebain_class:
            raise ValueError("class_index and crebain_class disagree")
        immutable_bbox = np.frombuffer(bbox.tobytes(order="C"), dtype=bbox.dtype)
        object.__setattr__(self, "bbox", immutable_bbox)
        object.__setattr__(self, "confidence", confidence)

    def to_detection2d(
        self,
        camera_index: int,
        *,
        pixels_undistorted: bool,
        pixel_std_px: float,
        timestamp: float | None = None,
        camera_id: str | None = None,
        timestamp_std_s: float = 0.0,
    ):
        """Create a multicamera detection without inventing calibration or timing facts."""
        from ..multicam.tracking import Detection2D

        cx = float(self.bbox[0] + (self.bbox[2] - self.bbox[0]) / 2.0)
        cy = float(self.bbox[1] + (self.bbox[3] - self.bbox[1]) / 2.0)
        return Detection2D(
            camera_index,
            np.array([cx, cy]),
            self.crebain_class,
            self.confidence,
            timestamp,
            camera_id,
            pixels_undistorted=pixels_undistorted,
            pixel_std_px=pixel_std_px,
            timestamp_std_s=timestamp_std_s,
        )


def results_to_detections(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    model_names: dict[int, str],
) -> list[Detection]:
    """Convert raw detector output to crebain-mapped :class:`Detection` objects.

    Detections whose model class has no crebain counterpart are dropped. Pure —
    unit-tested without torch.
    """
    remap = crebain_class_map(model_names)
    box_error = "detector boxes must be real numeric arrays"
    score_error = "detector confidences must be real numeric arrays"
    raw_boxes = _raw_real_numeric_array(boxes_xyxy, box_error)
    raw_scores = _raw_real_numeric_array(confidences, score_error)
    try:
        raw_ids = np.asarray(class_ids)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("detector class_ids must be a numeric array") from exc
    if raw_boxes.shape == (0,):
        raw_boxes = raw_boxes.reshape(0, 4)
    elif raw_boxes.shape == (4,):
        raw_boxes = raw_boxes.reshape(1, 4)
    if raw_boxes.ndim != 2 or raw_boxes.shape[1:] != (4,):
        raise ValueError(f"boxes_xyxy must have shape (N, 4), got {raw_boxes.shape}")
    if len(raw_boxes) > _MAX_DETECTIONS:
        raise ValueError(f"detector output exceeds the {_MAX_DETECTIONS}-detection safety limit")
    if raw_scores.ndim != 1 or raw_ids.ndim != 1:
        raise ValueError("confidences and class_ids must be one-dimensional")
    if len(raw_scores) != len(raw_boxes) or len(raw_ids) != len(raw_boxes):
        raise ValueError("boxes, confidences, and class_ids must have equal lengths")
    if not np.all(np.isfinite(raw_boxes)) or (
        len(raw_boxes)
        and (
            np.any(raw_boxes[:, 2] <= raw_boxes[:, 0]) or np.any(raw_boxes[:, 3] <= raw_boxes[:, 1])
        )
    ):
        raise ValueError("boxes_xyxy must contain finite positive-area boxes")
    if np.any(np.abs(raw_boxes) > _MAX_PIXEL_MAGNITUDE):
        raise ValueError(f"box coordinates must not exceed {_MAX_PIXEL_MAGNITUDE:g} pixels")
    if not np.all(np.isfinite(raw_scores)) or np.any((raw_scores < 0.0) | (raw_scores > 1.0)):
        raise ValueError("confidences must contain finite probabilities in [0, 1]")
    boxes = _float_numeric_array(raw_boxes, box_error)
    scores = _float_numeric_array(raw_scores, score_error)
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
        raise ValueError("detector outputs must remain finite in float64")
    if np.issubdtype(raw_ids.dtype, np.bool_):
        raise ValueError("class_ids must contain finite nonnegative integer indices")
    if np.issubdtype(raw_ids.dtype, np.integer):
        if (np.issubdtype(raw_ids.dtype, np.signedinteger) and np.any(raw_ids < 0)) or np.any(
            raw_ids >= _MAX_MODEL_CLASSES
        ):
            raise ValueError(f"class_ids must contain integer indices in [0, {_MAX_MODEL_CLASSES})")
        ids = raw_ids.astype(np.int64)
    elif np.issubdtype(raw_ids.dtype, np.floating):
        if (
            not np.all(np.isfinite(raw_ids))
            or np.any(raw_ids != np.floor(raw_ids))
            or np.any(raw_ids < 0)
            or np.any(raw_ids >= _MAX_MODEL_CLASSES)
        ):
            raise ValueError(f"class_ids must contain integer indices in [0, {_MAX_MODEL_CLASSES})")
        ids = raw_ids.astype(np.int64)
    else:
        raise ValueError("class_ids must contain finite nonnegative integer indices")
    out: list[Detection] = []
    for box, conf, cid in zip(boxes, scores, ids, strict=True):
        if cid not in remap:
            continue
        idx = remap[cid]
        out.append(Detection(box.copy(), float(conf), CREBAIN_CLASSES[idx], idx))
    return out


class Detector:
    """Thin Ultralytics inference wrapper returning crebain-mapped detections."""

    def __init__(
        self,
        weights: str,
        *,
        expected_sha256: str,
        allow_pickle_checkpoint: bool = False,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        if not isinstance(weights, str) or not weights.strip():
            raise TypeError("weights must be a nonempty local artifact path")
        if type(allow_pickle_checkpoint) is not bool:
            raise TypeError("allow_pickle_checkpoint must be a boolean")
        conf = _real_numeric_scalar(conf, "conf must be a finite probability in [0, 1]")
        if not np.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be a finite probability in [0, 1]")
        iou = _real_numeric_scalar(iou, "iou must be finite and in (0, 1]")
        if not np.isfinite(iou) or not 0.0 < iou <= 1.0:
            raise ValueError("iou must be finite and in (0, 1]")
        from ..common.device import resolve_device

        resolved_device = resolve_device(device)
        snapshot = ArtifactSnapshot(
            weights,
            expected_sha256,
            allowed_suffixes={".pt", ".onnx", ".engine", ".mlpackage", ".mlmodelc"},
        )
        try:
            require_pickle_acknowledgement(snapshot.path, allow_pickle_checkpoint)
            harden_ultralytics_runtime()
            ultralytics = require("ultralytics", "vision")
            verify_ultralytics_policy()
            self.model = ultralytics.YOLO(str(snapshot.path))
            if getattr(self.model, "task", None) != "detect":
                raise ValueError(
                    f"checkpoint task must be 'detect', got {getattr(self.model, 'task', None)!r}"
                )
            names = getattr(self.model, "names", None)
            if not isinstance(names, dict):
                raise ValueError("checkpoint must expose an integer-keyed class-name mapping")
            if set(names) != set(range(len(names))):
                raise ValueError("checkpoint class-name mapping must be contiguous from zero")
            mapped_names = crebain_class_map(names)
            if not mapped_names:
                raise ValueError(
                    "checkpoint class table has no mapping into the candidate taxonomy"
                )
            self.device = resolved_device
        except BaseException:
            snapshot.close()
            raise
        self._artifact_snapshot = snapshot
        self._closed = False
        self.conf = float(conf)
        self.iou = float(iou)

    def close(self) -> None:
        if not self._closed:
            self._artifact_snapshot.close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("detector is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def detect(self, image) -> list[Detection]:
        if self._closed:
            raise RuntimeError("detector is closed")
        from .train import resolve_ultralytics_device

        prepared_image = prepare_single_image(image)
        backend_image = _ultralytics_bgr_input(prepared_image)
        results = self.model.predict(
            backend_image,
            conf=self.conf,
            iou=self.iou,
            device=resolve_ultralytics_device(self.device),
            verbose=False,
        )
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise ValueError("detect expects exactly one input image and one result")
        res = results[0]
        boxes = res.boxes
        return results_to_detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            self.model.names,
        )


__all__ = ["Detection", "Detector", "results_to_detections"]
