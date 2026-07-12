"""Evaluation: detection AP metrics and a bounded latency microbenchmark."""

from __future__ import annotations

from .benchmark import BenchmarkResult, benchmark, resolve_sync
from .detection import (
    AREA_RANGES,
    Detections,
    GroundTruth,
    average_precision,
    iou_matrix,
    mean_average_precision,
)

__all__ = [
    "Detections",
    "GroundTruth",
    "iou_matrix",
    "average_precision",
    "mean_average_precision",
    "AREA_RANGES",
    "BenchmarkResult",
    "benchmark",
    "resolve_sync",
]
