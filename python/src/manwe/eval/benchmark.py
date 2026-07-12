"""Bounded latency microbenchmarking for a caller-supplied operation.

The 2026 survey flags the old ``metal-yolo-tests`` numbers as needing a rigorous
protocol: discard warm-up, run many timed iterations, synchronise the device, and
report P50/P95/P99 (not just mean FPS) with hardware recorded on every result.
This utility implements those basic mechanics for any callable on CUDA, MPS, or
CPU. It is not an MLPerf implementation or a cross-system comparison harness.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..common.device import describe_hardware

MAX_BENCHMARK_CALLS = 1_000_000


@dataclass
class BenchmarkResult:
    name: str
    iters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    fps: float
    hardware: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iters": self.iters,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_ms": self.std_ms,
            "fps": self.fps,
            "hardware": self.hardware,
        }


def benchmark(
    fn: Callable[[], object],
    name: str = "op",
    warmup: int = 15,
    iters: int = 100,
    sync: Callable[[], None] | None = None,
) -> BenchmarkResult:
    """Time ``fn`` over ``iters`` runs after ``warmup`` discarded runs.

    ``sync`` is called after each run to flush async device work (e.g.
    ``torch.cuda.synchronize`` or ``torch.mps.synchronize``); pass ``None`` on CPU.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a nonempty string")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if type(iters) is not int or iters <= 0:
        raise ValueError("iters must be a positive integer")
    if warmup + iters > MAX_BENCHMARK_CALLS:
        raise ValueError(f"warmup + iters exceeds the {MAX_BENCHMARK_CALLS}-call safety limit")
    if sync is not None and not callable(sync):
        raise TypeError("sync must be callable or None")

    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    times = np.empty(iters)
    for i in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter_ns()
        fn()
        if sync is not None:
            sync()
        times[i] = (time.perf_counter_ns() - t0) / 1_000_000.0

    mean = float(times.mean())
    return BenchmarkResult(
        name=name,
        iters=iters,
        mean_ms=mean,
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        std_ms=float(times.std()),
        fps=1000.0 / mean if mean > 0 else float("inf"),
        hardware=describe_hardware(),
    )


def resolve_sync(device_kind: str) -> Callable[[], None] | None:
    """Return the required device-sync callable, failing on unavailable accelerators."""
    if device_kind not in {"cpu", "cuda", "mps"}:
        raise ValueError("device_kind must be 'cpu', 'cuda', or 'mps'")
    if device_kind == "cpu":
        return None
    try:
        import torch
    except ImportError:
        raise RuntimeError(f"cannot synchronize {device_kind}: torch is not installed") from None
    if device_kind == "cuda" and torch.cuda.is_available():
        return torch.cuda.synchronize
    if device_kind == "mps" and getattr(torch, "mps", None) is not None:
        backend = getattr(torch.backends, "mps", None)
        if backend is not None and backend.is_available():
            return torch.mps.synchronize
    raise RuntimeError(f"cannot synchronize unavailable {device_kind} backend")


__all__ = ["BenchmarkResult", "benchmark", "resolve_sync"]
