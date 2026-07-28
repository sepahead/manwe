"""Small, callback-free numeric admission helpers for public scalar boundaries."""

from __future__ import annotations

import math
from typing import cast

import numpy as np

_MAX_EXACT_FLOAT64_INTEGER = 2**53
_FLOAT64_DTYPE = np.dtype(np.float64)
_NUMPY_REAL_SCALAR_TYPES = frozenset(
    np.dtype(dtype).type
    for dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    )
)


def finite_float64_scalar(value: object, name: str) -> float:
    """Return one finite float64 scalar without invoking user coercion hooks.

    Built-in numeric scalars, exact NumPy scalar types, and exact zero-dimensional
    ``ndarray`` values are accepted. Numeric subclasses are deliberately rejected:
    calling ``float(value)`` or ``np.asarray(value)`` on them can execute arbitrary
    ``__float__``/``__int__`` code. Integer values outside float64's consecutive
    exact range and wider floating values that would narrow are rejected rather
    than silently changing a threshold or configuration value.
    """
    value_type = type(value)
    if value_type in {int, float}:
        integer_value = cast(int, value) if value_type is int else 0
        if value_type is int and not (
            -_MAX_EXACT_FLOAT64_INTEGER <= integer_value <= _MAX_EXACT_FLOAT64_INTEGER
        ):
            raise ValueError(f"{name} must be exactly representable as a finite float64 scalar")
    elif value_type is np.ndarray:
        array_value = cast(np.ndarray, value)
        if array_value.ndim != 0:
            raise ValueError(f"{name} must be a finite real scalar")
    elif value_type not in _NUMPY_REAL_SCALAR_TYPES:
        raise ValueError(f"{name} must be a finite real scalar")

    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    if raw.dtype.kind in "iu":
        integer = int(raw.item())
        if not -_MAX_EXACT_FLOAT64_INTEGER <= integer <= _MAX_EXACT_FLOAT64_INTEGER:
            raise ValueError(f"{name} must be exactly representable as a finite float64 scalar")
    if not bool(np.isfinite(raw).item()):
        raise ValueError(f"{name} must be a finite real scalar")

    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=np.float64)
    result = converted.item()
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if raw.dtype.kind == "f" and raw.dtype.itemsize > _FLOAT64_DTYPE.itemsize:
        restored = np.asarray(converted, dtype=raw.dtype)
        if bool((restored != raw).item()):
            raise ValueError(f"{name} loses precision when converted to float64")
    return result


def validate_exact_float64_integers(array: np.ndarray, name: str) -> None:
    """Reject integer domains whose distinct values can collapse in float64."""
    if array.size == 0 or array.dtype.kind not in "iu":
        return
    minimum = int(np.min(array))
    maximum = int(np.max(array))
    if minimum < -_MAX_EXACT_FLOAT64_INTEGER or maximum > _MAX_EXACT_FLOAT64_INTEGER:
        raise ValueError(
            f"{name} contains integers outside the consecutive exact float64 range "
            f"[-{_MAX_EXACT_FLOAT64_INTEGER}, {_MAX_EXACT_FLOAT64_INTEGER}]"
        )


def widen_to_float64(
    array: np.ndarray, name: str, *, validate_exact_integers: bool = False
) -> np.ndarray:
    """Widen an already shape-bounded array without silent finite-value narrowing.

    When ``validate_exact_integers`` is set, integer inputs are first rejected if
    any value falls outside float64's consecutive-integer range, so distinct
    integers cannot collapse onto a shared float.
    """
    if validate_exact_integers:
        validate_exact_float64_integers(array, name)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(array, dtype=np.float64)
    if array.dtype.kind == "f":
        finite_before = np.isfinite(array)
        if np.any(finite_before & ~np.isfinite(converted)):
            raise ValueError(f"{name} contains values outside the finite float64 range")
        if array.dtype.itemsize > _FLOAT64_DTYPE.itemsize:
            restored = np.asarray(converted, dtype=array.dtype)
            if np.any(finite_before & (restored != array)):
                raise ValueError(f"{name} loses numeric precision when converted to float64")
    return converted


__all__ = [
    "finite_float64_scalar",
    "validate_exact_float64_integers",
    "widen_to_float64",
]
