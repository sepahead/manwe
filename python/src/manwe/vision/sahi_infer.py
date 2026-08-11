"""Sliced inference (SAHI) for tiny aerial objects.

Drones occupy a handful of pixels in wide-area imagery; running the detector on
overlapping high-resolution slices and merging recovers small targets a single
640² forward pass misses. Thin wrapper over the SAHI library (``vision`` extra).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common.artifacts import (
    ArtifactSnapshot,
    normalize_sha256,
    require_pickle_acknowledgement,
)
from ..common.deps import require
from ..common.numeric import finite_float64_scalar
from ..common.ultralytics import harden_ultralytics_runtime, verify_ultralytics_policy
from .input import _validate_dimensions, prepare_single_image
from .postprocess import airspace_class_map
from .predict import Detection, results_to_detections

_MAX_SLICES = 256
_MAX_OBJECT_PREDICTIONS = 100_000


@dataclass(frozen=True, slots=True)
class SliceConfig:
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    conf: float = 0.25

    def __post_init__(self) -> None:
        for name in ("slice_height", "slice_width"):
            value = getattr(self, name)
            if type(value) is not int or not 32 <= value <= 8192:
                raise ValueError(f"{name} must be an integer in [32, 8192]")
        for name in ("overlap_height_ratio", "overlap_width_ratio"):
            try:
                value = finite_float64_scalar(getattr(self, name), name)
            except ValueError:
                raise ValueError(f"{name} must be in [0, 1)") from None
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        try:
            confidence = finite_float64_scalar(self.conf, "conf")
        except ValueError:
            raise ValueError("conf must be in [0, 1]") from None
        if not 0 <= confidence <= 1:
            raise ValueError("conf must be in [0, 1]")
        object.__setattr__(self, "conf", confidence)


def _slice_plan(height: int, width: int, cfg: SliceConfig) -> tuple[int, int]:
    step_height = max(1, int(cfg.slice_height * (1.0 - cfg.overlap_height_ratio)))
    step_width = max(1, int(cfg.slice_width * (1.0 - cfg.overlap_width_ratio)))
    rows = 1 + max(0, math.ceil((height - cfg.slice_height) / step_height))
    columns = 1 + max(0, math.ceil((width - cfg.slice_width) / step_width))
    if rows * columns > _MAX_SLICES:
        raise ValueError(f"slice plan exceeds the {_MAX_SLICES}-slice safety limit")
    return rows, columns


def _sahi_class_names(model: object) -> dict[int, str]:
    """Admit the pinned SAHI adapter's complete mutable class table."""
    underlying = getattr(model, "model", None)
    task = getattr(underlying, "task", None)
    if type(task) is not str or task != "detect":
        raise ValueError(f"checkpoint task must be 'detect', got {task!r}")
    mapping = getattr(model, "category_mapping", None)
    if type(mapping) is not dict or not mapping:
        raise ValueError("SAHI model must expose a nonempty category mapping")
    if len(mapping) > 4096:
        raise ValueError("SAHI category mapping exceeds the 4096-class safety limit")
    if any(type(key) is not str or type(value) is not str for key, value in mapping.items()):
        raise ValueError("SAHI category mapping must contain built-in string keys and names")
    expected_keys = {str(index) for index in range(len(mapping))}
    if set(mapping) != expected_keys:
        raise ValueError("SAHI category mapping must be contiguous from zero")
    names = {index: mapping[str(index)] for index in range(len(mapping))}
    if not airspace_class_map(names):
        raise ValueError("checkpoint class table has no mapping into the candidate taxonomy")
    return names


def _sahi_predictions_to_detections(
    predictions: object,
    model_names: dict[int, str],
    image_hw: tuple[int, int],
) -> list[Detection]:
    """Convert one bounded SAHI result without leaking backend-owned objects."""
    if type(predictions) is not list or len(predictions) > _MAX_OBJECT_PREDICTIONS:
        raise ValueError(f"SAHI result must contain at most {_MAX_OBJECT_PREDICTIONS} predictions")
    if not predictions:
        return []

    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for index, prediction in enumerate(predictions):
        bbox = getattr(prediction, "bbox", None)
        to_xyxy = getattr(bbox, "to_xyxy", None)
        category = getattr(prediction, "category", None)
        score = getattr(prediction, "score", None)
        if not callable(to_xyxy) or category is None or score is None:
            raise ValueError(f"SAHI prediction {index} is incomplete")
        coordinates = to_xyxy()
        if type(coordinates) not in {list, tuple} or len(coordinates) != 4:
            raise ValueError(f"SAHI prediction {index} bbox must contain four coordinates")
        try:
            admitted_coordinates = [
                finite_float64_scalar(value, f"SAHI prediction {index} bbox")
                for value in coordinates
            ]
            confidence = finite_float64_scalar(
                getattr(score, "value", None), f"SAHI prediction {index} score"
            )
        except ValueError as exc:
            raise ValueError(f"SAHI prediction {index} contains invalid numeric output") from exc
        category_id = getattr(category, "id", None)
        category_name = getattr(category, "name", None)
        if type(category_id) is not int or category_id not in model_names:
            raise ValueError(f"SAHI prediction {index} has an undeclared category id")
        if type(category_name) is not str or category_name != model_names[category_id]:
            raise ValueError(f"SAHI prediction {index} category metadata disagrees with the model")
        boxes.append(admitted_coordinates)
        scores.append(confidence)
        class_ids.append(category_id)

    return results_to_detections(
        np.asarray(boxes, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(class_ids, dtype=np.int64),
        model_names,
        image_hw=image_hw,
    )


def sliced_predict(
    weights: str,
    image: Any,
    cfg: SliceConfig | None = None,
    device: str = "auto",
    *,
    expected_sha256: str,
    allow_pickle_checkpoint: bool = False,
) -> list[Detection]:
    """Run SAHI sliced prediction and return immutable Manwe detections.

    ``image`` may be one bounded RGB uint8 array or a local still-image path. Path
    inputs are decoded to RGB before SAHI receives them. Backend objects are
    validated and converted before the verified model snapshot is released.
    """
    if type(weights) is not str or not weights.strip():
        raise TypeError("weights must be a nonempty local artifact path")
    if type(device) is not str:
        raise TypeError("device must be a string")
    if type(allow_pickle_checkpoint) is not bool:
        raise TypeError("allow_pickle_checkpoint must be a boolean")
    expected_sha256 = normalize_sha256(expected_sha256)
    if cfg is None:
        cfg = SliceConfig()
    elif type(cfg) is not SliceConfig:
        raise TypeError("cfg must be a SliceConfig or None")
    numpy_plan_checked = False
    if (
        type(image) is np.ndarray
        and image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
    ):
        _validate_dimensions(int(image.shape[0]), int(image.shape[1]))
        _slice_plan(int(image.shape[0]), int(image.shape[1]), cfg)
        numpy_plan_checked = True
    prepared_image = prepare_single_image(image)
    if type(prepared_image) is np.ndarray:
        image_height, image_width = prepared_image.shape[:2]
    else:
        image_width, image_height = prepared_image.size
    if not numpy_plan_checked:
        _slice_plan(int(image_height), int(image_width), cfg)
    from ..common.device import resolve_device

    dev = resolve_device(device).torch_device
    snapshot = ArtifactSnapshot(
        weights,
        expected_sha256,
        allowed_suffixes={".pt", ".onnx", ".engine", ".mlpackage", ".mlmodelc"},
        expect_directory=Path(weights).suffix.lower() in {".mlpackage", ".mlmodelc"},
    )
    with snapshot:
        require_pickle_acknowledgement(snapshot.path, allow_pickle_checkpoint)
        harden_ultralytics_runtime()
        sahi_models = require("sahi.models.ultralytics", "vision")
        verify_ultralytics_policy()
        from sahi.predict import get_sliced_prediction

        # SAHI expects a torch-style device string ("cuda:0"/"mps"/"cpu"), not the
        # Ultralytics-CLI form ("0"), so use Device.torch_device here.
        model = sahi_models.UltralyticsDetectionModel(
            model_path=str(snapshot.path),
            confidence_threshold=cfg.conf,
            device=dev,
            task="detect",
        )
        model_names = _sahi_class_names(model)
        result = get_sliced_prediction(
            prepared_image,
            model,
            slice_height=cfg.slice_height,
            slice_width=cfg.slice_width,
            overlap_height_ratio=cfg.overlap_height_ratio,
            overlap_width_ratio=cfg.overlap_width_ratio,
        )
        if _sahi_class_names(model) != model_names:
            raise RuntimeError("SAHI model task or category mapping changed during inference")
        return _sahi_predictions_to_detections(
            getattr(result, "object_prediction_list", None),
            model_names,
            (int(image_height), int(image_width)),
        )


__all__ = ["SliceConfig", "sliced_predict"]
