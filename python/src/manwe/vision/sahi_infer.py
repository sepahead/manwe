"""Sliced inference (SAHI) for tiny aerial objects.

Drones occupy a handful of pixels in wide-area imagery; running the detector on
overlapping high-resolution slices and merging recovers small targets a single
640² forward pass misses. Thin wrapper over the SAHI library (``vision`` extra).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..common.artifacts import ArtifactSnapshot, require_pickle_acknowledgement
from ..common.deps import require
from ..common.ultralytics import harden_ultralytics_runtime, verify_ultralytics_policy
from .input import prepare_single_image

_MAX_SLICES = 10_000
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
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if (
            isinstance(self.conf, bool)
            or not isinstance(self.conf, (int, float))
            or not 0 <= self.conf <= 1
        ):
            raise ValueError("conf must be in [0, 1]")


def sliced_predict(
    weights: str,
    image,
    cfg: SliceConfig | None = None,
    device: str = "auto",
    *,
    expected_sha256: str,
    allow_pickle_checkpoint: bool = False,
):
    """Run SAHI sliced prediction and return its object-prediction list.

    ``image`` may be one bounded RGB uint8 array or a local still-image path. Path
    inputs are decoded to RGB before SAHI receives them. Returns SAHI ``ObjectPrediction``
    objects; convert to crebain detections with
    :func:`manwe.vision.predict.results_to_detections` via their ``category``/bbox.
    """
    if cfg is None:
        cfg = SliceConfig()
    elif not isinstance(cfg, SliceConfig):
        raise TypeError("cfg must be a SliceConfig or None")
    prepared_image = prepare_single_image(image)
    if isinstance(prepared_image, np.ndarray):
        image_height, image_width = prepared_image.shape[:2]
    else:
        image_width, image_height = prepared_image.size
    step_height = max(1, int(cfg.slice_height * (1.0 - cfg.overlap_height_ratio)))
    step_width = max(1, int(cfg.slice_width * (1.0 - cfg.overlap_width_ratio)))
    rows = 1 + max(0, math.ceil((image_height - cfg.slice_height) / step_height))
    columns = 1 + max(0, math.ceil((image_width - cfg.slice_width) / step_width))
    if rows * columns > _MAX_SLICES:
        raise ValueError(f"slice plan exceeds the {_MAX_SLICES}-slice safety limit")
    from ..common.device import resolve_device

    dev = resolve_device(device).torch_device
    if type(allow_pickle_checkpoint) is not bool:
        raise TypeError("allow_pickle_checkpoint must be a boolean")
    snapshot = ArtifactSnapshot(
        weights,
        expected_sha256,
        allowed_suffixes={".pt", ".onnx", ".engine", ".mlpackage", ".mlmodelc"},
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
            model_path=str(snapshot.path), confidence_threshold=cfg.conf, device=dev
        )
        result = get_sliced_prediction(
            prepared_image,
            model,
            slice_height=cfg.slice_height,
            slice_width=cfg.slice_width,
            overlap_height_ratio=cfg.overlap_height_ratio,
            overlap_width_ratio=cfg.overlap_width_ratio,
        )
        predictions = result.object_prediction_list
        if not isinstance(predictions, list) or len(predictions) > _MAX_OBJECT_PREDICTIONS:
            raise ValueError(
                f"SAHI result must contain at most {_MAX_OBJECT_PREDICTIONS} predictions"
            )
        return predictions


__all__ = ["SliceConfig", "sliced_predict"]
