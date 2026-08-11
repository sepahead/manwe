"""Shared Manwe contracts, device selection, seeding, and logging."""

from __future__ import annotations

from .contracts import (
    AIRSPACE_CLASSES,
    CREBAIN_CLASSES,
    AirspaceClass,
    CrebainClass,
    DetectionOutputSpec,
    ImageInputSpec,
    ModelContract,
    RuntimeSpec,
    TensorSpec,
    airspace_class_index,
    coco_to_airspace,
    coco_to_crebain,
    crebain_class_index,
    detection_runtime,
)
from .device import Device, DeviceKind, describe_hardware, resolve_device
from .logging import configure_logging, get_logger
from .seed import seed_everything

__all__ = [
    "AIRSPACE_CLASSES",
    "AirspaceClass",
    "airspace_class_index",
    "coco_to_airspace",
    "CREBAIN_CLASSES",
    "CrebainClass",
    "ModelContract",
    "TensorSpec",
    "ImageInputSpec",
    "DetectionOutputSpec",
    "RuntimeSpec",
    "detection_runtime",
    "coco_to_crebain",
    "crebain_class_index",
    "Device",
    "DeviceKind",
    "describe_hardware",
    "resolve_device",
    "configure_logging",
    "get_logger",
    "seed_everything",
]
