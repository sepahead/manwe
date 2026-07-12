"""Multi-camera calibration, triangulation and cross-camera correlation."""

from __future__ import annotations

from .camera import Camera, CameraRig
from .tracking import (
    Detection2D,
    Detection3D,
    correlate_and_triangulate,
    to_measurements,
)
from .triangulation import (
    reprojection_error,
    triangulate_dlt,
    triangulate_midpoint,
    triangulation_covariance,
)

__all__ = [
    "Camera",
    "CameraRig",
    "triangulate_dlt",
    "triangulate_midpoint",
    "reprojection_error",
    "triangulation_covariance",
    "Detection2D",
    "Detection3D",
    "correlate_and_triangulate",
    "to_measurements",
]
