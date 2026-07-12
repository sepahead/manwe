"""Vision pillar: detection, from-scratch architecture training, and sliced inference."""

from __future__ import annotations

from .models import (
    DEFAULT_ACCURACY,
    DEFAULT_DETECTOR,
    MODEL_ZOO,
    ModelSpec,
    build_model,
    list_models,
)
from .postprocess import crebain_class_map, letterbox_params, nms, scale_boxes, xywh2xyxy
from .predict import Detection, Detector, results_to_detections
from .sahi_infer import SliceConfig, sliced_predict
from .train import VisionTrainConfig, resolve_ultralytics_device, train

__all__ = [
    "DEFAULT_DETECTOR",
    "DEFAULT_ACCURACY",
    "ModelSpec",
    "MODEL_ZOO",
    "build_model",
    "list_models",
    "letterbox_params",
    "xywh2xyxy",
    "scale_boxes",
    "nms",
    "crebain_class_map",
    "Detection",
    "Detector",
    "results_to_detections",
    "SliceConfig",
    "sliced_predict",
    "VisionTrainConfig",
    "train",
    "resolve_ultralytics_device",
]
