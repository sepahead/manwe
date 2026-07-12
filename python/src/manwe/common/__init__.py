"""Shared plumbing: crebain contracts, device selection, seeding, logging."""

from __future__ import annotations

from .contracts import (
    CREBAIN_CLASSES,
    CrebainClass,
    ModelContract,
    TensorSpec,
    coco_to_crebain,
    crebain_class_index,
)
from .device import Device, DeviceKind, describe_hardware, resolve_device
from .logging import configure_logging, get_logger
from .seed import seed_everything

__all__ = [
    "CREBAIN_CLASSES",
    "CrebainClass",
    "ModelContract",
    "TensorSpec",
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
