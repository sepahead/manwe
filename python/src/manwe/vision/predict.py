"""Run a detector and map its output into Manwe's airspace taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common.artifacts import (
    ArtifactSnapshot,
    normalize_sha256,
    require_pickle_acknowledgement,
)
from ..common.contracts import AIRSPACE_CLASSES
from ..common.deps import require
from ..common.ultralytics import harden_ultralytics_runtime, verify_ultralytics_policy
from .input import prepare_single_image
from .postprocess import (
    _float_numeric_array,
    _raw_real_numeric_array,
    _real_numeric_scalar,
    airspace_class_map,
)

_MAX_PIXEL_MAGNITUDE = 1e9
_MAX_DETECTIONS = 100_000
# Preserve the vetted Ultralytics inference default explicitly. This bounds
# device-to-host copies before NumPy can observe or allocate for backend output.
_MAX_BACKEND_DETECTIONS = 300
_MAX_MODEL_CLASSES = 4096


def _checkpoint_class_names(model: object) -> tuple[str, ...]:
    """Validate mutable backend metadata and return one immutable class table."""
    task = getattr(model, "task", None)
    if type(task) is not str or task != "detect":
        raise ValueError(f"checkpoint task must be 'detect', got {task!r}")
    names = getattr(model, "names", None)
    if type(names) is not dict:
        raise ValueError("checkpoint must expose an integer-keyed class-name mapping")
    # Validate independent key/name/count bounds before constructing a contiguous
    # sequence. Iterating an exact dict with exact admitted keys/values cannot run
    # third-party coercion or comparison hooks.
    mapped_names = airspace_class_map(names)
    if set(names) != set(range(len(names))):
        raise ValueError("checkpoint class-name mapping must be contiguous from zero")
    if not mapped_names:
        raise ValueError("checkpoint class table has no mapping into the candidate taxonomy")
    return tuple(names[index] for index in range(len(names)))


def _names_dict(names: tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


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


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Detection:
    """An image-space detection already mapped to a Manwe airspace class."""

    bbox: np.ndarray  # [x1, y1, x2, y2] pixels
    confidence: float
    airspace_class: str  # one of AIRSPACE_CLASSES
    class_index: int

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        airspace_class: str | None = None,
        class_index: int | None = None,
        *,
        crebain_class: str | None = None,
    ) -> None:
        """Create a detection, accepting the pre-2.0 class keyword as an alias."""
        if airspace_class is not None and type(airspace_class) is not str:
            raise TypeError("airspace_class must be a string")
        if crebain_class is not None and type(crebain_class) is not str:
            raise TypeError("crebain_class must be a string")
        if airspace_class is None:
            airspace_class = crebain_class
        elif crebain_class is not None and crebain_class != airspace_class:
            raise ValueError("airspace_class and crebain_class aliases disagree")
        if airspace_class is None:
            raise TypeError("airspace_class is required")
        if class_index is None:
            raise TypeError("class_index is required")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "airspace_class", airspace_class)
        object.__setattr__(self, "class_index", class_index)
        self.__post_init__()

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
        if type(self.airspace_class) is not str or self.airspace_class not in AIRSPACE_CLASSES:
            raise ValueError(f"unknown Manwe airspace class {self.airspace_class!r}")
        if type(self.class_index) is not int or not 0 <= self.class_index < len(AIRSPACE_CLASSES):
            raise ValueError("class_index is outside the Manwe airspace taxonomy")
        if AIRSPACE_CLASSES[self.class_index] != self.airspace_class:
            raise ValueError("class_index and airspace_class disagree")
        immutable_bbox = np.frombuffer(bbox.tobytes(order="C"), dtype=bbox.dtype)
        object.__setattr__(self, "bbox", immutable_bbox)
        object.__setattr__(self, "confidence", confidence)

    @property
    def crebain_class(self) -> str:
        """Pre-2.0 compatibility alias for :attr:`airspace_class`."""
        return self.airspace_class

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
            self.airspace_class,
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
    *,
    image_hw: tuple[int, int] | None = None,
) -> list[Detection]:
    """Convert raw detector output to Manwe-mapped :class:`Detection` objects.

    Detections whose model class has no airspace counterpart are dropped. When
    ``image_hw`` is supplied, every backend box must remain inside that exact
    ``(height, width)`` image; malformed output is rejected rather than clipped.
    Pure — unit-tested without torch.
    """
    remap = airspace_class_map(model_names)
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
    if image_hw is not None:
        if (
            type(image_hw) is not tuple
            or len(image_hw) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in image_hw)
        ):
            raise ValueError("image_hw must be a (positive height, positive width) tuple")
        image_height, image_width = image_hw
        if len(raw_boxes) and (
            np.any(raw_boxes[:, 0] < 0)
            or np.any(raw_boxes[:, 1] < 0)
            or np.any(raw_boxes[:, 2] > image_width)
            or np.any(raw_boxes[:, 3] > image_height)
        ):
            raise ValueError("detector boxes must remain inside the source image bounds")
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
        if int(cid) not in model_names:
            raise ValueError(
                f"detector class id {int(cid)} is outside the declared model class table"
            )
        if cid not in remap:
            continue
        idx = remap[cid]
        out.append(Detection(box.copy(), float(conf), AIRSPACE_CLASSES[idx], idx))
    return out


def _backend_tensor_shape(value: object, name: str) -> tuple[int, ...]:
    """Read a backend tensor's cheap shape metadata without materializing it."""
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"detector backend {name} must expose shape metadata")
    try:
        dimensions = tuple(shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"detector backend {name} has invalid shape metadata") from exc
    normalized: list[int] = []
    for dimension in dimensions:
        if (
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer))
            or int(dimension) < 0
        ):
            raise ValueError(f"detector backend {name} has invalid shape metadata")
        normalized.append(int(dimension))
    return tuple(normalized)


def _bounded_backend_tensors(boxes: Any) -> tuple[Any, Any, Any]:
    """Preflight every Ultralytics result tensor before the first CPU copy."""
    try:
        xyxy = boxes.xyxy
        confidences = boxes.conf
        class_ids = boxes.cls
    except AttributeError as exc:
        raise ValueError("detector backend boxes output is incomplete") from exc
    xyxy_shape = _backend_tensor_shape(xyxy, "boxes.xyxy")
    confidence_shape = _backend_tensor_shape(confidences, "boxes.conf")
    class_shape = _backend_tensor_shape(class_ids, "boxes.cls")
    if len(xyxy_shape) != 2 or xyxy_shape[1:] != (4,):
        raise ValueError(f"detector backend boxes.xyxy must have shape (N, 4), got {xyxy_shape}")
    count = xyxy_shape[0]
    if count > _MAX_BACKEND_DETECTIONS:
        raise ValueError(
            "detector backend output exceeds the "
            f"{_MAX_BACKEND_DETECTIONS}-detection device-copy safety limit"
        )
    if confidence_shape != (count,) or class_shape != (count,):
        raise ValueError("detector backend boxes, confidences, and class_ids must align")
    return xyxy, confidences, class_ids


class Detector:
    """Thin Ultralytics inference wrapper returning Manwe-mapped detections."""

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
        if type(weights) is not str or not weights.strip():
            raise TypeError("weights must be a nonempty local artifact path")
        if type(allow_pickle_checkpoint) is not bool:
            raise TypeError("allow_pickle_checkpoint must be a boolean")
        expected_sha256 = normalize_sha256(expected_sha256)
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
            expect_directory=Path(weights).suffix.lower() in {".mlpackage", ".mlmodelc"},
        )
        try:
            require_pickle_acknowledgement(snapshot.path, allow_pickle_checkpoint)
            harden_ultralytics_runtime()
            ultralytics = require("ultralytics", "vision")
            verify_ultralytics_policy()
            self.model = ultralytics.YOLO(str(snapshot.path))
            self._model_names = _checkpoint_class_names(self.model)
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
        if type(self._closed) is not bool or self._closed:
            raise RuntimeError("detector is closed")
        from ..common.device import Device
        from .train import resolve_ultralytics_device

        model = self.model
        model_names = _checkpoint_class_names(model)
        canonical_names = self._model_names
        if (
            type(canonical_names) is not tuple
            or any(type(name) is not str for name in canonical_names)
            or model_names != canonical_names
        ):
            raise RuntimeError("checkpoint class-name mapping changed after model load")
        conf = _real_numeric_scalar(self.conf, "conf must be a finite probability in [0, 1]")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be a finite probability in [0, 1]")
        iou = _real_numeric_scalar(self.iou, "iou must be finite and in (0, 1]")
        if not 0.0 < iou <= 1.0:
            raise ValueError("iou must be finite and in (0, 1]")
        if type(self.device) is not Device:
            raise RuntimeError("detector device changed after model load")
        device = self.device
        prepared_image = prepare_single_image(image)
        backend_image = _ultralytics_bgr_input(prepared_image)
        results = model.predict(
            backend_image,
            conf=conf,
            iou=iou,
            max_det=_MAX_BACKEND_DETECTIONS,
            device=resolve_ultralytics_device(device),
            verbose=False,
        )
        if self.model is not model:
            raise RuntimeError("detector model changed during inference")
        if (
            _checkpoint_class_names(model) != model_names
            or self._model_names is not canonical_names
        ):
            raise RuntimeError("checkpoint class-name mapping changed during inference")
        try:
            final_conf = _real_numeric_scalar(
                self.conf, "conf must be a finite probability in [0, 1]"
            )
            final_iou = _real_numeric_scalar(self.iou, "iou must be finite and in (0, 1]")
        except ValueError as exc:
            raise RuntimeError("detector configuration changed during inference") from exc
        if self.device is not device or final_conf != conf or final_iou != iou:
            raise RuntimeError("detector configuration changed during inference")
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise ValueError("detect expects exactly one input image and one result")
        res = results[0]
        try:
            boxes = res.boxes
        except AttributeError as exc:
            raise ValueError("detector backend result must expose boxes") from exc
        xyxy, confidences, class_ids = _bounded_backend_tensors(boxes)
        return results_to_detections(
            xyxy.cpu().numpy(),
            confidences.cpu().numpy(),
            class_ids.cpu().numpy(),
            _names_dict(model_names),
            image_hw=(int(backend_image.shape[0]), int(backend_image.shape[1])),
        )


__all__ = ["Detection", "Detector", "results_to_detections"]
