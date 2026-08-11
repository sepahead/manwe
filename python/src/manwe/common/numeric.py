"""Callback-free scalar and covariance admission helpers."""

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


def validated_psd_covariance(
    covariance: np.ndarray,
    name: str,
    *,
    require_nonzero: bool = False,
) -> np.ndarray:
    """Validate/repair one finite float64 covariance in correlation coordinates.

    Positive semidefiniteness is invariant under diagonal congruence. Testing the
    correlation matrix therefore prevents a large variance in one coordinate
    from hiding a material defect in a much smaller principal block. Eigenvalues
    outside a roundoff band are accepted or rejected directly; an unresolved
    near-zero block is moved a tiny distance into the positive interior so an
    eigensolver rounding it just above zero cannot return an exactly indefinite
    binary matrix. Diagonal variances are preserved exactly.
    """
    if type(covariance) is not np.ndarray or covariance.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be an admitted float64 ndarray")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    size = covariance.shape[0]
    if size == 0 or size > 64:
        raise ValueError(f"{name} dimension must be in [1, 64]")
    if type(require_nonzero) is not bool:
        raise TypeError("require_nonzero must be a boolean")
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} must contain only finite values")

    result = covariance.copy()
    diagonal = np.diag(result).copy()
    if np.any(diagonal < 0.0):
        raise ValueError(f"{name} must be positive semidefinite")
    positive = diagonal > 0.0
    zero = ~positive
    if np.any(zero) and (np.any(result[zero, :] != 0.0) or np.any(result[:, zero] != 0.0)):
        raise ValueError(
            f"{name} must be positive semidefinite; zero variances require zero covariance"
        )
    if require_nonzero and not np.any(positive):
        raise ValueError(f"{name} must contain non-zero uncertainty")
    if not np.any(positive):
        return result

    indices = np.flatnonzero(positive)
    deviations = np.sqrt(diagonal[positive])
    smaller = np.minimum.outer(deviations, deviations)
    larger = np.maximum.outer(deviations, deviations)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        block = (result[np.ix_(indices, indices)] / smaller) / larger
    if not np.isfinite(block).all():
        raise ValueError(f"{name} must be positive semidefinite")

    tolerance = 100.0 * np.finfo(np.float64).eps * size
    if not np.allclose(block, block.T, rtol=tolerance, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    block = 0.5 * block + 0.5 * block.T
    block_diagonal = np.diag(block).copy()
    if np.any(block_diagonal <= 0.0) or not np.isfinite(block_diagonal).all():
        raise ValueError(f"{name} must be positive semidefinite")
    block_smaller = np.minimum.outer(block_diagonal, block_diagonal)
    block_larger = np.maximum.outer(block_diagonal, block_diagonal)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        block_scale = np.sqrt(block_smaller / block_larger) * block_larger
        block = block / block_scale
    block = 0.5 * block + 0.5 * block.T
    np.fill_diagonal(block, 1.0)
    if not np.isfinite(block).all() or np.any(np.abs(block) > 1.0 + tolerance):
        raise ValueError(f"{name} must be positive semidefinite")

    try:
        values, vectors = np.linalg.eigh(block)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} eigendecomposition did not converge") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError(f"{name} eigendecomposition is non-finite")
    if float(values[0]) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    result = 0.5 * result + 0.5 * result.T
    np.fill_diagonal(result, diagonal)
    if values[0] > tolerance:
        _validate_covariance_spectral_range(result, name)
        return result

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        block = vectors @ np.diag(np.maximum(values, tolerance)) @ vectors.T
    block = 0.5 * block + 0.5 * block.T
    repaired_diagonal = np.diag(block).copy()
    if np.any(repaired_diagonal <= 0.0) or not np.isfinite(block).all():
        raise ValueError(f"{name} covariance repair is non-finite")
    repaired_smaller = np.minimum.outer(repaired_diagonal, repaired_diagonal)
    repaired_larger = np.maximum.outer(repaired_diagonal, repaired_diagonal)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        repaired_scale = np.sqrt(repaired_smaller / repaired_larger) * repaired_larger
        block = block / repaired_scale
    block = np.clip(0.5 * block + 0.5 * block.T, -1.0, 1.0)
    np.fill_diagonal(block, 1.0)
    try:
        repaired_values = np.linalg.eigvalsh(block)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance repair could not be verified") from exc
    if not np.isfinite(repaired_values).all() or float(repaired_values[0]) <= 0.0:
        raise ValueError(f"{name} covariance repair is not positive semidefinite")

    block_mantissa, block_exponent = np.frexp(block)
    deviation_mantissa, deviation_exponent = np.frexp(deviations)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mantissa = (
            block_mantissa * deviation_mantissa[:, np.newaxis] * deviation_mantissa[np.newaxis, :]
        )
        exponent = (
            block_exponent + deviation_exponent[:, np.newaxis] + deviation_exponent[np.newaxis, :]
        )
        repaired_block = np.ldexp(mantissa, exponent)
    if not np.isfinite(repaired_block).all():
        raise ValueError(f"{name} covariance repair exceeds the finite numeric range")
    repaired_block = 0.5 * repaired_block + 0.5 * repaired_block.T
    np.fill_diagonal(repaired_block, diagonal[positive])
    result[np.ix_(indices, indices)] = repaired_block
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        returned_block = (result[np.ix_(indices, indices)] / smaller) / larger
    returned_block = 0.5 * returned_block + 0.5 * returned_block.T
    if not np.isfinite(returned_block).all() or np.any(np.abs(returned_block) > 1.0 + tolerance):
        raise ValueError(f"{name} covariance repair is not positive semidefinite")
    try:
        returned_values = np.linalg.eigvalsh(returned_block)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance repair could not be verified") from exc
    if not np.isfinite(returned_values).all() or float(returned_values[0]) <= 0.0:
        raise ValueError(f"{name} covariance repair is not positive semidefinite")
    _validate_covariance_spectral_range(result, name)
    return result


def _validate_covariance_spectral_range(covariance: np.ndarray, name: str) -> None:
    """Reject a finite matrix whose largest eigenvalue is not finite in float64."""
    scale = float(np.max(np.abs(covariance)))
    if scale == 0.0:
        return
    normalized = covariance / scale
    try:
        values = np.linalg.eigvalsh(normalized)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} eigendecomposition did not converge") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} eigendecomposition is non-finite")
    largest = float(values[-1])
    if largest > 1.0 and scale > np.finfo(np.float64).max / largest:
        raise ValueError(f"{name} spectral magnitude exceeds the finite numeric range")


__all__ = [
    "finite_float64_scalar",
    "validate_exact_float64_integers",
    "validated_psd_covariance",
    "widen_to_float64",
]
