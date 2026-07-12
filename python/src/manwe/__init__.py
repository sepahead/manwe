"""Manwe — an independent perception research and validation workbench.

Manwe develops candidate models, numerical references, and raw artifacts that
may eventually be adapted to downstream systems. No reviewed consumer currently
ingests these APIs or artifacts directly. The package is organised as four
capability pillars plus shared plumbing:

    manwe.vision    detection / from-scratch training for aerial objects (drone, bird,
                    aircraft, helicopter) + sliced small-object inference
    manwe.audio     acoustic detection, microphone-array direction-of-arrival,
                    and conversion to local fusion measurements
    manwe.multicam  camera calibration, triangulation and cross-camera association
    manwe.fusion    multi-target tracking (KF / EKF / UKF / PF / IMM), data
                    association, track lifecycle, and tracking metrics
    manwe.export    raw ONNX / CoreML / TensorRT conversion, explicit receipts,
                    candidate manifests, and fidelity measurement
    manwe.common    device selection (Metal / CUDA / CPU), seeding, logging,
                    and candidate integration contracts

The *core* (fusion, geometry, DOA, metrics, contracts) depends only on numpy and
runs without ML runtimes on the supported Linux/macOS hosts. The heavy pillars
lazily import their optional dependencies (torch, ultralytics, coremltools, ...)
and raise a clear error listing the extra to install if they are missing.
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]


def __getattr__(name: str):
    # Lazy submodule access so `import manwe` never pulls in heavy optional deps.
    import importlib

    if name in {"vision", "audio", "multicam", "fusion", "export", "common", "data", "eval"}:
        return importlib.import_module(f"manwe.{name}")
    raise AttributeError(f"module 'manwe' has no attribute {name!r}")
