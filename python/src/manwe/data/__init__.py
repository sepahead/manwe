"""Datasets: registry of real corpora + an offline synthetic generator."""

from __future__ import annotations

from .datasets import (
    DATASETS,
    DatasetSpec,
    access_instructions,
    get_dataset,
    list_datasets,
)
from .synthetic import make_vision_smoke, write_png

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "list_datasets",
    "get_dataset",
    "access_instructions",
    "make_vision_smoke",
    "write_png",
]
