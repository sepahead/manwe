"""Recursive Bayesian filters for a 6-state constant-velocity target.

This is an independent numerical reference using a six-state
``x = [px, py, pz, vx, vy, vz]`` convention (metres and m/s in a common world
frame). It shares estimator vocabulary with downstream systems, but parity
requires explicit adapters and intermediate-state fixtures.

Filters
-------
- :class:`KalmanFilter`          linear Cartesian CV (Joseph-form covariance)
- :class:`ExtendedKalmanFilter`  adds a polar (radar) measurement update
- :class:`UnscentedKalmanFilter` derivative-free, scaled sigma points
- :class:`ParticleFilter`        Monte-Carlo posterior, systematic resampling
- :class:`IMMEstimator`          bank of models mixed by a Markov chain

All are pure-numpy.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from inspect import getattr_static
from itertools import islice
from types import (
    ClassMethodDescriptorType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import Any, Protocol, cast

import numpy as np

POS_DIM = 3
STATE_DIM = 6
MAX_FILTER_DIMENSION = 64
MAX_FILTER_PARTICLES = 100_000
MAX_WRAP_ANGLE_CELLS = 1_000_000
MAX_WRAP_ANGLE_MAGNITUDE = 1_000_000.0
MIN_POLAR_HORIZONTAL_RANGE = 1e-6
_MAX_EXACT_FLOAT64_INTEGER = 2**53
_FLOAT64_DTYPE = np.dtype(np.float64)
_REAL_NUMERIC_KINDS = frozenset("iuf")
_NUMPY_INTEGER_SCALAR_TYPES = frozenset(
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
    )
)
_NUMPY_FLOAT_SCALAR_TYPES = frozenset(
    np.dtype(dtype).type
    for dtype in (
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    )
)
_SUPPORTED_INTEGER_SCALAR_TYPES = frozenset({int} | _NUMPY_INTEGER_SCALAR_TYPES)
_SUPPORTED_REAL_SCALAR_TYPES = frozenset(
    {float} | _SUPPORTED_INTEGER_SCALAR_TYPES | _NUMPY_FLOAT_SCALAR_TYPES
)
_SUPPORTED_BIT_GENERATOR_TYPES = frozenset(
    {
        np.random.MT19937,
        np.random.PCG64,
        np.random.PCG64DXSM,
        np.random.Philox,
        np.random.SFC64,
    }
)
_MAX_RAW_SEQUENCE_CELLS = MAX_FILTER_PARTICLES * (2 * MAX_FILTER_DIMENSION + 1)


# ---------------------------------------------------------------------------
# Motion / measurement models
# ---------------------------------------------------------------------------
def cv_transition(dt: float, dim: int = POS_DIM) -> np.ndarray:
    """Constant-velocity state-transition matrix ``F`` for a ``2*dim`` state."""
    dim = _validate_dim(dim)
    dt = _validate_dt(dt)
    n = 2 * dim
    F = np.eye(n)
    F[:dim, dim:] = dt * np.eye(dim)
    return F


def cv_process_noise(dt: float, sigma_a: float, dim: int = POS_DIM) -> np.ndarray:
    """Discrete white-noise-acceleration process covariance ``Q``.

    ``sigma_a`` is the standard deviation of the un-modelled acceleration (m/s²).
    One acceleration draw is defined per ``predict`` call. This is a discrete,
    event-indexed model, not continuous white acceleration: callers must not
    introduce numerical substeps and expect an equivalent covariance.
    """
    dim = _validate_dim(dim)
    dt = _validate_dt(dt)
    sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
    if dt == 0 or sigma_a == 0:
        return np.zeros((2 * dim, 2 * dim))
    try:
        with np.errstate(over="raise", invalid="raise"):
            # Form the process-noise gain first. This is algebraically identical
            # to sigma_a**2 times the usual dt powers, but avoids needless
            # intermediate overflow when a very large dt and tiny sigma cancel.
            velocity_std = np.float64(sigma_a) * dt
            position_std = 0.5 * velocity_std * dt
            I = np.eye(dim)
            Q = np.zeros((2 * dim, 2 * dim))
            Q[:dim, :dim] = np.square(position_std) * I
            Q[:dim, dim:] = (position_std * velocity_std) * I
            Q[dim:, :dim] = Q[:dim, dim:]
            Q[dim:, dim:] = np.square(velocity_std) * I
    except FloatingPointError as exc:
        raise ValueError("dt and sigma_a produce a non-finite process covariance") from exc
    if not np.isfinite(Q).all():
        raise ValueError("dt and sigma_a produce a non-finite process covariance")
    return Q


def position_measurement_matrix(dim: int = POS_DIM) -> np.ndarray:
    """Measurement matrix ``H = [I | 0]`` mapping state → observed position."""
    dim = _validate_dim(dim)
    H = np.zeros((dim, 2 * dim))
    H[:, :dim] = np.eye(dim)
    return H


def _symmetrize(P: np.ndarray) -> np.ndarray:
    # Halve before adding so unequal near-float-max pairs cannot overflow, but
    # preserve equal minimum-subnormal entries that halving would erase.
    transpose = P.T
    with np.errstate(under="ignore"):
        averaged = 0.5 * P + 0.5 * transpose
    return np.where(transpose == P, P, averaged)


def _is_supported_real_scalar(value: object) -> bool:
    return type(value) in _SUPPORTED_REAL_SCALAR_TYPES


def _sequence_matches_shape(
    value: object,
    shape: tuple[int, ...],
    active_containers: set[int],
) -> bool:
    """Inspect container structure without invoking scalar conversion hooks."""
    if isinstance(value, np.ndarray):
        return type(value) is np.ndarray and value.shape == shape
    if isinstance(value, (list, tuple)):
        if type(value) not in (list, tuple) or not shape or len(value) != shape[0]:
            return False
        identity = id(value)
        if identity in active_containers:
            return False
        active_containers.add(identity)
        try:
            return all(
                _sequence_matches_shape(item, shape[1:], active_containers) for item in value
            )
        finally:
            active_containers.remove(identity)
    return not shape


def _preflight_numeric_sequence(
    value: list | tuple,
    name: str,
    *,
    allowed_shapes: tuple[tuple[int, ...], ...] | None,
    maximum_cells: int,
) -> None:
    """Bound and prove a built-in numeric container before NumPy materializes it."""
    if allowed_shapes is not None and not any(
        _sequence_matches_shape(value, shape, set()) for shape in allowed_shapes
    ):
        expected = " or ".join(str(shape) for shape in allowed_shapes)
        raise ValueError(f"{name} must have shape {expected}")

    pending: list[tuple[object, bool]] = [(value, False)]
    active_containers: set[int] = set()
    cells = 0
    while pending:
        current, exiting = pending.pop()
        if isinstance(current, (list, tuple)):
            if type(current) not in (list, tuple):
                raise ValueError(f"{name} must use built-in list/tuple containers")
            identity = id(current)
            if exiting:
                active_containers.remove(identity)
                continue
            if identity in active_containers:
                raise ValueError(f"{name} must not contain cyclic containers")
            cells += len(current)
            if cells > maximum_cells:
                raise ValueError(f"{name} exceeds the {maximum_cells}-value safety limit")
            active_containers.add(identity)
            pending.append((current, True))
            pending.extend((item, False) for item in reversed(current))
            continue
        if isinstance(current, np.ndarray):
            if type(current) is not np.ndarray:
                raise ValueError(f"{name} must not contain ndarray subclasses")
            if current.dtype.kind not in _REAL_NUMERIC_KINDS:
                raise ValueError(f"{name} must contain real numeric values")
            cells += current.size
            if cells > maximum_cells:
                raise ValueError(f"{name} exceeds the {maximum_cells}-value safety limit")

    pending_values: list[object] = [value]
    while pending_values:
        current = pending_values.pop()
        if type(current) in (list, tuple):
            pending_values.extend(cast(list[object] | tuple[object, ...], current))
            continue
        if type(current) is np.ndarray:
            continue
        if not _is_supported_real_scalar(current):
            raise ValueError(f"{name} must contain real numeric values")


def _raw_real_array(
    value: object,
    name: str,
    *,
    allowed_shapes: tuple[tuple[int, ...], ...] | None = None,
    maximum_cells: int = _MAX_RAW_SEQUENCE_CELLS,
) -> np.ndarray:
    """Admit real primitives without executing user-defined coercion hooks."""
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray:
            raise ValueError(f"{name} must not be an ndarray subclass")
        if allowed_shapes is not None and value.shape not in allowed_shapes:
            expected = " or ".join(str(shape) for shape in allowed_shapes)
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    elif isinstance(value, (list, tuple)):
        _preflight_numeric_sequence(
            value,
            name,
            allowed_shapes=allowed_shapes,
            maximum_cells=maximum_cells,
        )
    elif not _is_supported_real_scalar(value):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if raw.dtype.kind not in _REAL_NUMERIC_KINDS:
        raise ValueError(f"{name} must contain real numeric values")
    return raw


def _float_array(raw: np.ndarray, name: str) -> np.ndarray:
    if raw.dtype.kind == "u" and np.any(raw > _MAX_EXACT_FLOAT64_INTEGER):
        raise ValueError(f"{name} integers must be exactly representable in float64")
    if raw.dtype.kind == "i" and (
        np.any(raw < -_MAX_EXACT_FLOAT64_INTEGER) or np.any(raw > _MAX_EXACT_FLOAT64_INTEGER)
    ):
        raise ValueError(f"{name} integers must be exactly representable in float64")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if raw.dtype.kind == "f":
        finite_before = np.isfinite(raw)
        if np.any(finite_before & ~np.isfinite(converted)):
            raise ValueError(f"{name} contains values outside the finite float64 range")
        if raw.dtype.itemsize > _FLOAT64_DTYPE.itemsize:
            restored = np.asarray(converted, dtype=raw.dtype)
            if np.any(finite_before & (restored != raw)):
                raise ValueError(f"{name} loses numeric precision when converted to float64")
    return converted


def _float_scalar(value: object, name: str) -> float:
    """Admit one real scalar without invoking object-element coercion."""
    raw = _raw_real_array(value, name)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a real numeric scalar")
    return float(_float_array(raw, name))


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap finite angle(s) to ``[-pi, pi)``."""
    raw = _raw_real_array(a, "angle", maximum_cells=MAX_WRAP_ANGLE_CELLS)
    if raw.size > MAX_WRAP_ANGLE_CELLS:
        raise ValueError(f"angle exceeds the {MAX_WRAP_ANGLE_CELLS}-value safety limit")
    if not np.isfinite(raw).all():
        raise ValueError("angle must contain only finite values")
    if np.any(np.abs(raw) > MAX_WRAP_ANGLE_MAGNITUDE):
        raise ValueError(
            f"angle magnitude exceeds the reliable canonicalization limit "
            f"{MAX_WRAP_ANGLE_MAGNITUDE:g}"
        )
    array = _float_array(raw, "angle")
    if not np.isfinite(array).all():
        raise ValueError("angle must contain only finite values")
    wrapped = (array + np.pi) % (2 * np.pi) - np.pi
    return float(wrapped) if array.ndim == 0 else wrapped


def _validate_dim(dim: object) -> int:
    if type(dim) not in _SUPPORTED_INTEGER_SCALAR_TYPES:
        raise ValueError(f"dim must be an integer in [1, {MAX_FILTER_DIMENSION}]")
    result = int(cast(int, dim))
    if not 1 <= result <= MAX_FILTER_DIMENSION:
        raise ValueError(f"dim must be an integer in [1, {MAX_FILTER_DIMENSION}]")
    return result


def _validate_particle_count(value: object) -> int:
    if type(value) not in _SUPPORTED_INTEGER_SCALAR_TYPES:
        raise ValueError("n_particles must be an integer of at least 2")
    result = int(cast(int, value))
    if not 2 <= result <= MAX_FILTER_PARTICLES:
        raise ValueError(f"n_particles must be an integer in [2, {MAX_FILTER_PARTICLES}]")
    return result


def _validate_rng(value: object) -> np.random.Generator:
    if type(value) is not np.random.Generator:
        raise TypeError("rng must be a numpy.random.Generator")
    generator = cast(np.random.Generator, value)
    if type(generator.bit_generator) not in _SUPPORTED_BIT_GENERATOR_TYPES:
        raise TypeError("rng must use a built-in numpy BitGenerator")
    return generator


def _validate_finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = _float_scalar(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _validate_nonnegative_scalar(value: object, name: str) -> float:
    value = _validate_finite_scalar(value, name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_positive_scalar(value: object, name: str) -> float:
    value = _validate_nonnegative_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_dt(dt: float) -> float:
    return _validate_nonnegative_scalar(dt, "dt")


def _as_vector(value: object, length: int, name: str) -> np.ndarray:
    raw = _raw_real_array(value, name, allowed_shapes=((length,),))
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    array = _float_array(raw, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _correlation_coordinates(
    covariance: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent a covariance in per-coordinate standard-deviation units.

    Positive-semidefiniteness is invariant under this diagonal congruence.
    Unlike one global scale, it cannot let an unrelated large variance hide a
    material negative eigenvalue in a smaller principal block.
    """
    diagonal = np.diag(covariance)
    standard_deviations = np.sqrt(diagonal)
    positive = standard_deviations > 0
    zero = ~positive
    if np.any(zero) and (np.any(covariance[zero, :] != 0.0) or np.any(covariance[:, zero] != 0.0)):
        raise ValueError(
            f"{name} must be positive semidefinite; zero variances require zero covariance"
        )

    correlation = np.zeros_like(covariance)
    if np.any(positive):
        indices = np.flatnonzero(positive)
        deviations = standard_deviations[positive]
        smaller = np.minimum.outer(deviations, deviations)
        larger = np.maximum.outer(deviations, deviations)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            block = (covariance[np.ix_(indices, indices)] / smaller) / larger
        if not np.isfinite(block).all():
            raise ValueError(f"{name} must be positive semidefinite")
        correlation[np.ix_(indices, indices)] = block
    return correlation, standard_deviations, positive


def _covariance_from_correlation(
    correlation: np.ndarray,
    standard_deviations: np.ndarray,
    name: str,
) -> np.ndarray:
    """Rescale correlation coordinates without forming overflow-prone products."""
    correlation_mantissa, correlation_exponent = np.frexp(correlation)
    deviation_mantissa, deviation_exponent = np.frexp(standard_deviations)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mantissa = correlation_mantissa * deviation_mantissa[:, None] * deviation_mantissa[None, :]
        exponent = correlation_exponent + deviation_exponent[:, None] + deviation_exponent[None, :]
        covariance = np.ldexp(mantissa, exponent)
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} could not be repaired to a finite covariance")
    return _symmetrize(covariance)


def _as_covariance(
    value: object, size: int, name: str, *, positive_definite: bool = False
) -> np.ndarray:
    raw = _raw_real_array(value, name, allowed_shapes=((size, size),))
    if raw.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    array = _float_array(raw, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diag(array) < 0):
        raise ValueError(
            f"{name} must be positive semidefinite; diagonal variances must be nonnegative"
        )
    correlation, standard_deviations, positive = _correlation_coordinates(array, name)
    tolerance = 100.0 * np.finfo(float).eps * size
    if np.any(np.abs(correlation) > 1.0 + tolerance):
        raise ValueError(f"{name} must be positive semidefinite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        symmetric = np.allclose(
            correlation,
            correlation.T,
            rtol=tolerance,
            atol=tolerance,
        )
    if not symmetric:
        raise ValueError(f"{name} must be symmetric")
    array = _symmetrize(array)
    if np.any(positive):
        indices = np.flatnonzero(positive)
        correlation = _symmetrize(correlation)
        block = correlation[np.ix_(indices, indices)]
        block_diagonal = np.diag(block)
        if np.any(block_diagonal <= 0) or not np.isfinite(block_diagonal).all():
            raise ValueError(f"{name} must be positive semidefinite")
        smaller_diagonal = np.minimum.outer(block_diagonal, block_diagonal)
        larger_diagonal = np.maximum.outer(block_diagonal, block_diagonal)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            block_scale = np.sqrt(smaller_diagonal / larger_diagonal) * larger_diagonal
            block = block / block_scale
        block = _symmetrize(block)
        np.fill_diagonal(block, 1.0)
        if np.any(np.abs(block) > 1.0 + tolerance):
            raise ValueError(f"{name} must be positive semidefinite")
        try:
            values, vectors = np.linalg.eigh(block)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} could not be decomposed") from exc
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValueError(f"{name} decomposition is not finite")
        if float(values[0]) < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite")
        if positive_definite and float(values[0]) <= tolerance:
            raise ValueError(f"{name} must be positive definite")
        if values[0] < 0:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                repaired = _symmetrize(vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T)
            repaired_diagonal = np.diag(repaired)
            if np.any(repaired_diagonal <= 0) or not np.isfinite(repaired).all():
                raise ValueError(f"{name} could not be repaired to a finite covariance")
            repaired_smaller = np.minimum.outer(repaired_diagonal, repaired_diagonal)
            repaired_larger = np.maximum.outer(repaired_diagonal, repaired_diagonal)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                repaired_scale = np.sqrt(repaired_smaller / repaired_larger) * repaired_larger
                repaired = repaired / repaired_scale
            repaired = _symmetrize(repaired)
            np.fill_diagonal(repaired, 1.0)
            if np.any(np.abs(repaired) > 1.0 + tolerance):
                raise ValueError(f"{name} could not be repaired to a finite covariance")
            repaired = np.clip(repaired, -1.0, 1.0)
            repaired_block = _covariance_from_correlation(
                repaired,
                standard_deviations[positive],
                name,
            )
            np.fill_diagonal(repaired_block, np.diag(array)[positive])
            array[np.ix_(indices, indices)] = repaired_block
            array = _symmetrize(array)
            repaired_correlation, _, repaired_positive = _correlation_coordinates(array, name)
            repaired_indices = np.flatnonzero(repaired_positive)
            try:
                repaired_values = np.linalg.eigvalsh(
                    _symmetrize(repaired_correlation[np.ix_(repaired_indices, repaired_indices)])
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name} repair could not be verified") from exc
            if not np.isfinite(repaired_values).all() or float(repaired_values[0]) < -tolerance:
                raise ValueError(f"{name} could not be repaired to a finite covariance")
    if positive_definite:
        _cholesky(array, name)
    return array.copy()


def _cholesky(matrix: np.ndarray, name: str) -> np.ndarray:
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise ValueError(f"{name} must be positive definite")
    standard_deviations = np.sqrt(diagonal)
    smaller = np.minimum.outer(standard_deviations, standard_deviations)
    larger = np.maximum.outer(standard_deviations, standard_deviations)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        correlation = _symmetrize((matrix / smaller) / larger)
    if not np.isfinite(correlation).all():
        raise ValueError(f"{name} must be positive definite")
    try:
        correlation_factor = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        factor = standard_deviations[:, None] * correlation_factor
    if not np.isfinite(factor).all():
        raise FloatingPointError(f"{name} Cholesky factor is not finite")
    if np.any(np.diag(factor) <= 0):
        raise ValueError(f"{name} must be numerically positive definite")
    return factor


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray, name: str) -> np.ndarray:
    factor = _cholesky(matrix, name)
    solution = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    if not np.isfinite(solution).all():
        raise FloatingPointError(f"{name} solve is not finite")
    return solution


def _finite_difference(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
    if not np.isfinite(difference).all():
        raise FloatingPointError(f"{name} is not finite")
    return difference


def _gaussian_log_likelihood(y: np.ndarray, covariance: np.ndarray) -> float:
    factor = _cholesky(covariance, "innovation covariance")
    whitened = np.linalg.solve(factor, y)
    log_det = 2.0 * float(np.log(np.diag(factor)).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic = float(whitened @ whitened)
        result = float(-0.5 * (len(y) * np.log(2.0 * np.pi) + log_det + quadratic))
    if not np.isfinite(result):
        raise FloatingPointError("Gaussian log likelihood is not finite")
    return result


def _squared_mahalanobis(y: np.ndarray, covariance: np.ndarray) -> float:
    solution = _solve_spd(covariance, y, "innovation covariance")
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(y @ solution)
    if not np.isfinite(distance) or distance < -1e-10:
        raise FloatingPointError("gating distance is not finite and nonnegative")
    return max(distance, 0.0)


def _likelihood_from_log(log_likelihood: float) -> float:
    if not np.isfinite(log_likelihood):
        raise FloatingPointError("log likelihood must be finite")
    lower = float(np.log(np.finfo(float).tiny))
    upper = float(np.log(np.finfo(float).max))
    return float(np.exp(np.clip(log_likelihood, lower, upper)))


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _sample_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    """Draw from a possibly singular Gaussian without backend matmul warnings."""
    mean = _as_vector(mean, len(mean), f"{name} mean")
    covariance = _as_covariance(covariance, len(mean), f"{name} covariance")
    correlation, standard_deviations, _ = _correlation_coordinates(covariance, name)
    try:
        values, vectors = np.linalg.eigh(_symmetrize(correlation))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance could not be factored") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError(f"{name} covariance factor is not finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        root = standard_deviations[:, None] * (vectors * np.sqrt(np.maximum(values, 0.0))[None, :])
    if not np.isfinite(root).all():
        raise ValueError(f"{name} covariance factor is not finite")
    standard = rng.standard_normal((size, len(mean)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        samples = np.einsum("sj,ij->si", standard, root) + mean
    if not np.isfinite(samples).all():
        raise FloatingPointError(f"{name} samples are not finite")
    return samples


@dataclass
class GaussianState:
    """A Gaussian belief ``N(x, P)``."""

    x: np.ndarray
    P: np.ndarray

    def __post_init__(self) -> None:
        if type(self.x) is np.ndarray:
            if self.x.ndim != 1 or len(self.x) == 0:
                raise ValueError(f"x must be a non-empty 1-D vector, got shape {self.x.shape}")
            if len(self.x) > 2 * MAX_FILTER_DIMENSION:
                raise ValueError(
                    f"x must contain at most {2 * MAX_FILTER_DIMENSION} state coordinates"
                )
        elif type(self.x) in (list, tuple):
            if len(self.x) == 0:
                raise ValueError("x must be a non-empty 1-D vector")
            if len(self.x) > 2 * MAX_FILTER_DIMENSION:
                raise ValueError(
                    f"x must contain at most {2 * MAX_FILTER_DIMENSION} state coordinates"
                )
        raw_x = _raw_real_array(
            self.x,
            "x",
            maximum_cells=2 * MAX_FILTER_DIMENSION,
        )
        if raw_x.ndim != 1 or len(raw_x) == 0:
            raise ValueError(f"x must be a non-empty 1-D vector, got shape {raw_x.shape}")
        if len(raw_x) > 2 * MAX_FILTER_DIMENSION:
            raise ValueError(f"x must contain at most {2 * MAX_FILTER_DIMENSION} state coordinates")
        self.x = _as_vector(raw_x, len(raw_x), "x")
        self.P = _as_covariance(self.P, len(raw_x), "P")

    def copy(self) -> GaussianState:
        return GaussianState(self.x.copy(), self.P.copy())

    @property
    def position(self) -> np.ndarray:
        return self.x[: len(self.x) // 2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[len(self.x) // 2 :]


def _canonical_filter_state(state: object, dim: int) -> tuple[np.ndarray, np.ndarray]:
    dim = _validate_dim(dim)
    if type(state) is not GaussianState:
        raise TypeError("state must be a GaussianState")
    size = 2 * dim
    return (
        _as_vector(state.x, size, "state.x"),
        _as_covariance(state.P, size, "state.P"),
    )


def _validate_filter_state(state: object, dim: int) -> tuple[np.ndarray, np.ndarray]:
    x, P = _canonical_filter_state(state, dim)
    typed_state = cast(GaussianState, state)
    typed_state.x, typed_state.P = x, P
    return typed_state.x, typed_state.P


def _validate_measurement_matrix(value: object, dim: int) -> np.ndarray:
    expected = position_measurement_matrix(dim)
    raw = _raw_real_array(value, "H", allowed_shapes=(expected.shape,))
    if raw.shape != expected.shape:
        raise ValueError(f"H must have shape {expected.shape}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("H must contain only finite values")
    matrix = _float_array(raw, "H")
    if not np.array_equal(matrix, expected):
        raise ValueError("H must retain the Cartesian position projection")
    return matrix.copy()


def _measurement_inputs(
    z: np.ndarray, R: np.ndarray, dim: int, *, covariance_positive_definite: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _as_vector(z, dim, "z"),
        _as_covariance(
            R,
            dim,
            "R",
            positive_definite=covariance_positive_definite,
        ),
    )


def _polar_measurement_inputs(
    z: np.ndarray, R: np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    z, R = _measurement_inputs(z, R, dim)
    if z[0] <= MIN_POLAR_HORIZONTAL_RANGE:
        raise ValueError(f"polar range must be > {MIN_POLAR_HORIZONTAL_RANGE:g} m")
    if abs(z[1]) > 1_000_000.0:
        raise ValueError("polar azimuth magnitude is too large to canonicalize reliably")
    z[1] = wrap_angle(z[1])
    if not -np.pi / 2.0 <= z[2] <= np.pi / 2.0:
        raise ValueError("polar elevation must be in [-pi/2, pi/2]")
    if z[0] * abs(float(np.cos(z[2]))) <= MIN_POLAR_HORIZONTAL_RANGE:
        raise ValueError("polar azimuth is singular on the sensor's vertical axis")
    return z, R


# ---------------------------------------------------------------------------
# Kalman filter (linear, Cartesian)
# ---------------------------------------------------------------------------
class KalmanFilter:
    """Linear Kalman filter on the CV model with Joseph-stabilised covariance."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, sigma_a: float = 3.0, dim: int = POS_DIM):
        self.dim = _validate_dim(dim)
        self.sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        self.state = GaussianState(
            _as_vector(x0, 2 * self.dim, "x0"),
            _as_covariance(P0, 2 * self.dim, "P0"),
        )
        self.H = position_measurement_matrix(self.dim)
        self._last_likelihood = 1.0
        self._last_log_likelihood = 0.0

    def _runtime(
        self, *, canonicalize_state: bool = True
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        dim = _validate_dim(self.dim)
        _validate_nonnegative_scalar(self.sigma_a, "sigma_a")
        x, P = _canonical_filter_state(self.state, dim)
        H = _validate_measurement_matrix(self.H, dim)
        if canonicalize_state:
            state = cast(GaussianState, self.state)
            state.x, state.P = x, P
        return dim, x, P, H

    # -- prediction ------------------------------------------------------
    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        dim, x, P, _ = self._runtime()
        if dt == 0:
            return
        F = cv_transition(dt, dim)
        Q = cv_process_noise(dt, self.sigma_a, dim)
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_x = F @ x
            raw_predicted_P = F @ P @ F.T + Q
        if not np.isfinite(predicted_x).all():
            raise FloatingPointError("predicted state is not finite")
        predicted_P = _as_covariance(raw_predicted_P, 2 * dim, "predicted P")
        self.state.x, self.state.P = predicted_x, predicted_P

    # -- Cartesian update ------------------------------------------------
    def innovation(self, z: np.ndarray) -> np.ndarray:
        dim, x, _, H = self._runtime()
        z = _as_vector(z, dim, "z")
        return _finite_difference(z, H @ x, "innovation")

    def innovation_covariance(self, R: np.ndarray) -> np.ndarray:
        dim, _, P, H = self._runtime()
        R = _as_covariance(R, dim, "R")
        return _as_covariance(
            H @ P @ H.T + R,
            dim,
            "innovation covariance",
            positive_definite=True,
        )

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        """Squared Mahalanobis distance of measurement ``z`` from the prediction."""
        y = self.innovation(z)
        S = self.innovation_covariance(R)
        return _squared_mahalanobis(y, S)

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        dim, x, P, H = self._runtime()
        z, R = _measurement_inputs(z, R, dim)
        y = _finite_difference(z, H @ x, "innovation")
        S = _as_covariance(
            H @ P @ H.T + R,
            dim,
            "innovation covariance",
            positive_definite=True,
        )
        PHt = P @ H.T
        K = _solve_spd(S, PHt.T, "innovation covariance").T
        I = np.eye(P.shape[0])
        A = I - K @ H
        with np.errstate(over="ignore", invalid="ignore"):
            updated_x = x + K @ y
            raw_updated_P = A @ P @ A.T + K @ R @ K.T
        if not np.isfinite(updated_x).all():
            raise FloatingPointError("updated state is not finite")
        updated_P = _as_covariance(raw_updated_P, 2 * dim, "updated P")
        log_likelihood = _gaussian_log_likelihood(y, S)
        likelihood = _likelihood_from_log(log_likelihood)
        self.state.x, self.state.P = updated_x, updated_P
        self._last_log_likelihood = log_likelihood
        self._last_likelihood = likelihood

    @property
    def likelihood(self) -> float:
        return self._last_likelihood

    @property
    def log_likelihood(self) -> float:
        return self._last_log_likelihood


# ---------------------------------------------------------------------------
# Extended Kalman filter (adds polar / radar update)
# ---------------------------------------------------------------------------
class ExtendedKalmanFilter(KalmanFilter):
    """KF plus a polar measurement update for native radar geometry.

    ``update_polar`` consumes ``z = [range, azimuth, elevation]`` relative to a
    ``sensor_origin``; the angular error is modelled in polar space (a diagonal
    Cartesian covariance would misrepresent it).
    """

    def update_polar(
        self, z: np.ndarray, R: np.ndarray, sensor_origin: np.ndarray | None = None
    ) -> None:
        dim, x, P, _ = self._runtime()
        if dim != 3:
            raise ValueError("polar radar updates require dim=3")
        z, R = _polar_measurement_inputs(z, R, dim)
        s = (
            np.zeros(dim)
            if sensor_origin is None
            else _as_vector(sensor_origin, dim, "sensor_origin")
        )
        h, Hj = self._polar_h_and_jacobian(s)
        y = _finite_difference(z, h, "polar innovation")
        y[1] = wrap_angle(y[1])  # azimuth
        y[2] = wrap_angle(y[2])  # elevation
        S = _as_covariance(
            Hj @ P @ Hj.T + R,
            dim,
            "innovation covariance",
            positive_definite=True,
        )
        PHt = P @ Hj.T
        K = _solve_spd(S, PHt.T, "innovation covariance").T
        I = np.eye(P.shape[0])
        A = I - K @ Hj
        with np.errstate(over="ignore", invalid="ignore"):
            updated_x = x + K @ y
            raw_updated_P = A @ P @ A.T + K @ R @ K.T
        if not np.isfinite(updated_x).all():
            raise FloatingPointError("updated state is not finite")
        updated_P = _as_covariance(raw_updated_P, 2 * dim, "updated P")
        log_likelihood = _gaussian_log_likelihood(y, S)
        likelihood = _likelihood_from_log(log_likelihood)
        self.state.x, self.state.P = updated_x, updated_P
        self._last_log_likelihood = log_likelihood
        self._last_likelihood = likelihood

    def _polar_h_and_jacobian(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dim, x, _, _ = self._runtime()
        s = _as_vector(s, dim, "sensor_origin")
        p = x[:dim]
        with np.errstate(over="ignore", invalid="ignore"):
            relative = p - s
        if not np.isfinite(relative).all():
            raise FloatingPointError("relative radar position is not finite")
        dx, dy, dz = relative
        # hypot avoids overflow for large but finite coordinates. Rewriting the
        # Jacobian as products of unit-vector components avoids squaring them.
        rho = float(np.hypot(dx, dy))
        r = float(np.hypot(rho, dz))
        if r <= MIN_POLAR_HORIZONTAL_RANGE or rho <= MIN_POLAR_HORIZONTAL_RANGE:
            raise ValueError("predicted polar geometry is singular at the sensor origin or axis")
        h = np.array([r, np.arctan2(dy, dx), np.arctan2(dz, rho)])
        H = np.zeros((3, 2 * dim))
        # d range
        H[0, 0], H[0, 1], H[0, 2] = dx / r, dy / r, dz / r
        # d azimuth
        H[1, 0], H[1, 1] = -(dy / rho) / rho, (dx / rho) / rho
        # d elevation
        H[2, 0] = -(dx / rho) * (dz / r) / r
        H[2, 1] = -(dy / rho) * (dz / r) / r
        H[2, 2] = (rho / r) / r
        if not np.isfinite(h).all() or not np.isfinite(H).all():
            raise FloatingPointError("polar measurement geometry is not finite")
        return h, H


# ---------------------------------------------------------------------------
# Unscented Kalman filter
# ---------------------------------------------------------------------------
class UnscentedKalmanFilter:
    """Scaled unscented KF (Van der Merwe sigma points), Cartesian position update."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        sigma_a: float = 3.0,
        dim: int = POS_DIM,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dim = _validate_dim(dim)
        self.n = 2 * self.dim
        self.sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        self.state = GaussianState(
            _as_vector(x0, self.n, "x0"),
            _as_covariance(P0, self.n, "P0"),
        )
        self.H = position_measurement_matrix(self.dim)
        self.alpha = _validate_positive_scalar(alpha, "alpha")
        self.beta = _validate_nonnegative_scalar(beta, "beta")
        self.kappa = _validate_finite_scalar(kappa, "kappa")
        if self.n + self.kappa <= 0:
            raise ValueError("kappa must satisfy n + kappa > 0")
        try:
            with np.errstate(over="raise", under="ignore", invalid="raise"):
                self._sigma_scale = float(np.square(np.float64(self.alpha)) * (self.n + self.kappa))
        except FloatingPointError as exc:
            raise ValueError("UKF parameters produce a non-finite sigma-point scale") from exc
        if not np.isfinite(self._sigma_scale) or self._sigma_scale <= 0:
            raise ValueError("UKF scaling must satisfy n + lambda > 0")
        self.lambda_ = self._sigma_scale - self.n
        self._wm, self._wc = self._weights()
        self._last_likelihood = 1.0
        self._last_log_likelihood = 0.0

    def _weights(self) -> tuple[np.ndarray, np.ndarray]:
        n, lam, scale = self.n, self.lambda_, self._sigma_scale
        wm = np.full(2 * n + 1, 1.0 / (2 * scale))
        wc = wm.copy()
        wm[0] = lam / scale
        wc[0] = lam / scale + (1 - self.alpha**2 + self.beta)
        if not np.isfinite(wm).all() or not np.isfinite(wc).all():
            raise ValueError("UKF parameters produce non-finite sigma-point weights")
        cancellation_error = np.finfo(float).eps * len(wm) * float(np.max(np.abs(wm)))
        if cancellation_error > 1e-6 or not np.isclose(wm.sum(), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("UKF parameters produce numerically unstable sigma-point weights")
        return wm, wc

    def _runtime(
        self,
        *,
        canonicalize_state: bool = True,
    ) -> tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        dim = _validate_dim(self.dim)
        _validate_nonnegative_scalar(self.sigma_a, "sigma_a")
        if type(self.n) not in _SUPPORTED_INTEGER_SCALAR_TYPES:
            raise ValueError("UKF state dimension was corrupted")
        n = int(self.n)
        if n != 2 * dim:
            raise ValueError("UKF state dimension was corrupted")
        x, P = _canonical_filter_state(self.state, dim)
        H = _validate_measurement_matrix(self.H, dim)
        alpha = _validate_positive_scalar(self.alpha, "alpha")
        beta = _validate_nonnegative_scalar(self.beta, "beta")
        kappa = _validate_finite_scalar(self.kappa, "kappa")
        if n + kappa <= 0:
            raise ValueError("kappa must satisfy n + kappa > 0")
        try:
            with np.errstate(over="raise", under="ignore", invalid="raise"):
                expected_scale = float(np.square(np.float64(alpha)) * (n + kappa))
        except FloatingPointError as exc:
            raise ValueError("UKF parameters produce a non-finite sigma-point scale") from exc
        if not np.isfinite(expected_scale) or expected_scale <= 0:
            raise ValueError("UKF scaling must satisfy n + lambda > 0")
        sigma_scale = _validate_positive_scalar(self._sigma_scale, "_sigma_scale")
        lambda_value = _validate_finite_scalar(self.lambda_, "lambda_")
        if sigma_scale != expected_scale or lambda_value != expected_scale - n:
            raise ValueError("UKF derived scaling parameters were corrupted")

        expected_wm = np.full(2 * n + 1, 1.0 / (2 * expected_scale))
        expected_wc = expected_wm.copy()
        expected_wm[0] = (expected_scale - n) / expected_scale
        expected_wc[0] = expected_wm[0] + (1 - alpha**2 + beta)
        if not np.isfinite(expected_wm).all() or not np.isfinite(expected_wc).all():
            raise ValueError("UKF parameters produce non-finite sigma-point weights")
        cancellation_error = (
            np.finfo(float).eps * len(expected_wm) * float(np.max(np.abs(expected_wm)))
        )
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = float(expected_wm.sum())
        if cancellation_error > 1e-6 or not np.isclose(
            weight_sum,
            1.0,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("UKF parameters produce numerically unstable sigma-point weights")
        wm = _as_vector(self._wm, 2 * n + 1, "_wm")
        wc = _as_vector(self._wc, 2 * n + 1, "_wc")
        if not np.array_equal(wm, expected_wm) or not np.array_equal(wc, expected_wc):
            raise ValueError("UKF sigma-point weights were corrupted")
        if canonicalize_state:
            state = cast(GaussianState, self.state)
            state.x, state.P = x, P
        return dim, n, x, P, H, wm, wc, sigma_scale

    def _sigma_points(self) -> np.ndarray:
        _, n, x, P, _, _, _, sigma_scale = self._runtime()
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = sigma_scale * P
        if not np.isfinite(scaled).all():
            raise FloatingPointError("scaled sigma-point covariance is not finite")
        try:
            U = _cholesky(scaled, "scaled sigma-point covariance")
        except ValueError:
            # A covariance may legitimately be singular. An eigen square root
            # represents that PSD distribution exactly instead of injecting
            # arbitrary jitter into the posterior.
            correlation, standard_deviations, _ = _correlation_coordinates(
                scaled,
                "scaled sigma-point covariance",
            )
            values, vectors = np.linalg.eigh(_symmetrize(correlation))
            if float(values[0]) < -100.0 * np.finfo(float).eps * n:
                raise ValueError(
                    "scaled sigma-point covariance must be positive semidefinite"
                ) from None
            with np.errstate(over="ignore", invalid="ignore"):
                U = standard_deviations[:, None] * (
                    vectors * np.sqrt(np.maximum(values, 0.0))[None, :]
                )
        if not np.isfinite(U).all():
            raise FloatingPointError("sigma points are not finite")
        pts = np.zeros((2 * n + 1, n))
        pts[0] = x
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n):
                pts[1 + i] = x + U[:, i]
                pts[1 + n + i] = x - U[:, i]
        if not np.isfinite(pts).all():
            raise FloatingPointError("sigma points are not finite")
        return pts

    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        dim, n, _, _, _, wm, wc, _ = self._runtime()
        if dt == 0:
            return
        F = cv_transition(dt, dim)
        Q = cv_process_noise(dt, self.sigma_a, dim)
        pts = self._sigma_points()
        with np.errstate(over="ignore", invalid="ignore"):
            prop = pts @ F.T
            x = wm @ prop
        if not np.isfinite(x).all():
            raise FloatingPointError("predicted state is not finite")
        P = Q.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(prop.shape[0]):
                d = prop[i] - x
                P += wc[i] * np.outer(d, d)
        updated_P = _as_covariance(P, n, "predicted P")
        self.state.x, self.state.P = x, updated_P

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        dim, n, x_prior, P_prior, H, wm, wc, _ = self._runtime()
        z, R = _measurement_inputs(z, R, dim)
        pts = self._sigma_points()
        with np.errstate(over="ignore", invalid="ignore"):
            zpts = pts @ H.T
            zhat = wm @ zpts
        if not np.isfinite(zhat).all():
            raise FloatingPointError("predicted measurement is not finite")
        S = R.copy()
        Pxz = np.zeros((n, dim))
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(zpts.shape[0]):
                dz = zpts[i] - zhat
                dx = pts[i] - self.state.x
                S += wc[i] * np.outer(dz, dz)
                Pxz += wc[i] * np.outer(dx, dz)
        S = _as_covariance(
            S,
            dim,
            "innovation covariance",
            positive_definite=True,
        )
        K = _solve_spd(S, Pxz.T, "innovation covariance").T
        y = _finite_difference(z, zhat, "innovation")
        with np.errstate(over="ignore", invalid="ignore"):
            updated_x = x_prior + K @ y
            raw_updated_P = P_prior - K @ S @ K.T
        if not np.isfinite(updated_x).all():
            raise FloatingPointError("updated state is not finite")
        updated_P = _as_covariance(raw_updated_P, n, "updated P")
        log_likelihood = _gaussian_log_likelihood(y, S)
        likelihood = _likelihood_from_log(log_likelihood)
        self.state.x, self.state.P = updated_x, updated_P
        self._last_log_likelihood = log_likelihood
        self._last_likelihood = likelihood

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        dim, _, x, P, H, _, _, _ = self._runtime()
        z, R = _measurement_inputs(z, R, dim)
        y = _finite_difference(z, H @ x, "innovation")
        S = _as_covariance(
            H @ P @ H.T + R,
            dim,
            "innovation covariance",
            positive_definite=True,
        )
        return _squared_mahalanobis(y, S)

    @property
    def likelihood(self) -> float:
        return self._last_likelihood

    @property
    def log_likelihood(self) -> float:
        return self._last_log_likelihood


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------
class ParticleFilter:
    """Bootstrap particle filter for non-Gaussian / multi-modal posteriors."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        sigma_a: float = 3.0,
        dim: int = POS_DIM,
        n_particles: int = 512,
        rng: np.random.Generator | None = None,
    ):
        self.dim = _validate_dim(dim)
        self.sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        self.H = position_measurement_matrix(self.dim)
        self.n_particles = _validate_particle_count(n_particles)
        if rng is not None and type(rng) is not np.random.Generator:
            raise TypeError("rng must be a numpy.random.Generator or None")
        self.rng = rng if rng is not None else np.random.default_rng()
        _validate_rng(self.rng)
        x0 = _as_vector(x0, 2 * self.dim, "x0")
        P0 = _as_covariance(P0, 2 * self.dim, "P0")
        self.particles: np.ndarray = _sample_gaussian(
            self.rng,
            x0,
            P0,
            self.n_particles,
            "initial particle",
        )
        self.weights: np.ndarray = np.full(self.n_particles, 1.0 / self.n_particles)
        self._last_likelihood = 1.0
        self._last_log_likelihood = 0.0
        self._validate_population()

    def _runtime(self) -> tuple[int, int, np.ndarray, np.random.Generator]:
        dim = _validate_dim(self.dim)
        _validate_nonnegative_scalar(self.sigma_a, "sigma_a")
        n_particles = _validate_particle_count(self.n_particles)
        H = _validate_measurement_matrix(self.H, dim)
        return dim, n_particles, H, _validate_rng(self.rng)

    def _validate_population(self) -> tuple[np.ndarray, np.ndarray]:
        dim, n_particles, _, _ = self._runtime()
        expected = (n_particles, 2 * dim)
        raw_particles = _raw_real_array(
            self.particles,
            "particles",
            allowed_shapes=(expected,),
        )
        raw_weights = _raw_real_array(
            self.weights,
            "weights",
            allowed_shapes=((n_particles,),),
        )
        if raw_particles.shape != expected:
            raise ValueError(f"particles must have shape {expected}, got {raw_particles.shape}")
        if raw_weights.shape != (n_particles,):
            raise ValueError(f"weights must have shape ({n_particles},), got {raw_weights.shape}")
        if not np.isfinite(raw_particles).all():
            raise ValueError("particles must contain only finite values")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
            raise ValueError("weights must be finite and nonnegative")
        particles = _float_array(raw_particles, "particles")
        weights = _float_array(raw_weights, "weights")
        if not np.isfinite(particles).all():
            raise ValueError("particles must contain only finite values")
        if not np.isfinite(weights).all():
            raise ValueError("weights must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(weights.sum())
        if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("weights must sum to 1")
        self.particles = particles
        self.weights = weights
        return particles, weights

    @property
    def state(self) -> GaussianState:
        particles, weights = self._validate_population()
        dim = _validate_dim(self.dim)
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.einsum("n,nd->d", weights, particles)
        if not np.isfinite(x).all():
            raise FloatingPointError("particle mean is not finite")
        d = _finite_difference(particles, x, "particle deviations")
        with np.errstate(over="ignore", invalid="ignore"):
            P = np.einsum("n,ni,nj->ij", weights, d, d)
        return GaussianState(x, _as_covariance(P, 2 * dim, "particle covariance"))

    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        dim, n_particles, _, rng = self._runtime()
        particles, _ = self._validate_population()
        if dt == 0:
            return
        F = cv_transition(dt, dim)
        Q = cv_process_noise(dt, self.sigma_a, dim)
        rng_state = copy.deepcopy(rng.bit_generator.state)
        try:
            noise = _sample_gaussian(
                rng,
                np.zeros(2 * dim),
                Q,
                n_particles,
                "process-noise",
            )
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = np.einsum("ni,ji->nj", particles, F) + noise
            if not np.isfinite(predicted).all():
                raise FloatingPointError("predicted particles are not finite")
            self.particles = predicted
        except BaseException:
            rng.bit_generator.state = rng_state
            raise

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        dim, n_particles, H, _ = self._runtime()
        z, R = _measurement_inputs(z, R, dim, covariance_positive_definite=True)
        particles, weights = self._validate_population()
        factor = _cholesky(R, "R")
        with np.errstate(over="ignore", invalid="ignore"):
            pred = np.einsum("ni,ji->nj", particles, H)
        if not np.isfinite(pred).all():
            raise FloatingPointError("predicted particle measurements are not finite")
        d = _finite_difference(pred, z, "particle innovations")
        whitened = np.linalg.solve(factor, d.T)
        with np.errstate(over="ignore", invalid="ignore"):
            maha = np.sum(whitened**2, axis=0)
        log_det = 2.0 * float(np.log(np.diag(factor)).sum())
        log_measurement = -0.5 * (dim * np.log(2.0 * np.pi) + log_det + maha)
        log_prior = np.full(n_particles, -np.inf)
        positive = weights > 0
        log_prior[positive] = np.log(weights[positive])
        log_posterior = log_prior + log_measurement
        log_evidence = _logsumexp(log_posterior)
        if not np.isfinite(log_evidence):
            raise FloatingPointError("particle update has zero or non-finite evidence")
        updated_weights = np.exp(log_posterior - log_evidence)
        if not np.isfinite(updated_weights).all() or updated_weights.sum() <= 0:
            raise FloatingPointError("particle weights are not finite and positive")
        self.weights = updated_weights / updated_weights.sum()
        self._last_log_likelihood = log_evidence
        self._last_likelihood = _likelihood_from_log(log_evidence)
        if self._ess() < len(self.particles) / 2:
            self._resample()

    def _ess(self) -> float:
        _, weights = self._validate_population()
        return 1.0 / np.sum(weights**2)

    def _resample(self) -> None:
        _, n, _, rng = self._runtime()
        particles, weights = self._validate_population()
        positions = (rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        idx = np.clip(idx, 0, n - 1)
        self.particles = particles[idx].copy()
        self.weights = np.full(n, 1.0 / n)

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        dim, _, H, _ = self._runtime()
        z, R = _measurement_inputs(z, R, dim)
        st = self.state
        y = _finite_difference(z, H @ st.x, "innovation")
        S = _as_covariance(
            H @ st.P @ H.T + R,
            dim,
            "innovation covariance",
            positive_definite=True,
        )
        return _squared_mahalanobis(y, S)

    @property
    def likelihood(self) -> float:
        return self._last_likelihood

    @property
    def log_likelihood(self) -> float:
        return self._last_log_likelihood


# ---------------------------------------------------------------------------
# Interacting Multiple Model estimator
# ---------------------------------------------------------------------------
_MAX_IMM_MODELS = 32
_MAX_IMM_TRANSACTION_NODES = 1_000_000
_MAX_IMM_TRANSACTION_ARRAY_CELLS = 2_000_000
_MAX_IMM_TRANSACTION_ARRAY_BYTES = 128 * 1024 * 1024
_OBJECT_CLASS_DESCRIPTOR: Any = object.__dict__["__class__"]


class _IMMModel(Protocol):
    state: GaussianState

    def predict(self, dt: float) -> None: ...


@dataclass(frozen=True)
class _IMMArrayUndo:
    array: np.ndarray
    value: np.ndarray
    dtype: np.dtype
    shape: tuple[int, ...]
    writeable: bool


@dataclass(frozen=True)
class _IMMGeneratorUndo:
    generator: np.random.Generator
    state: dict


@dataclass(frozen=True)
class _IMMObjectUndo:
    instance: object
    instance_type: type
    namespace: dict[str, object]
    namespace_descriptor: Any


@dataclass(frozen=True)
class _IMMGraphJournal:
    objects: tuple[_IMMObjectUndo, ...]
    dicts: tuple[tuple[dict, tuple[tuple[object, object], ...]], ...]
    lists: tuple[tuple[list, tuple[object, ...]], ...]
    sets: tuple[tuple[set, frozenset[object]], ...]
    deques: tuple[tuple[deque, tuple[object, ...]], ...]
    arrays: tuple[_IMMArrayUndo, ...]
    generators: tuple[_IMMGeneratorUndo, ...]


@dataclass(frozen=True)
class _IMMTransaction:
    journal: _IMMGraphJournal
    owner_namespace: dict[str, object]
    owner_type: type
    models_container: list | tuple
    models: tuple[_IMMModel, ...]
    model_namespaces: tuple[dict[str, object], ...]
    model_types: tuple[type, ...]
    transition: np.ndarray
    mode_probs: np.ndarray
    cbar: np.ndarray


@dataclass(frozen=True)
class _IMMRuntime:
    models: tuple[_IMMModel, ...]
    states: tuple[tuple[GaussianState, np.ndarray, np.ndarray], ...]
    dim: int
    transition: np.ndarray
    mode_probs: np.ndarray
    cbar: np.ndarray


@dataclass(frozen=True)
class _IMMSeal:
    owner_type: type
    owner_namespace: dict[str, object]
    owner_keys: frozenset[str]
    models_container: list | tuple
    models: tuple[_IMMModel, ...]
    model_types: tuple[type, ...]
    dim_type: type
    dim: object
    transition: np.ndarray
    transition_values: np.ndarray
    mode_probs: np.ndarray
    mode_prob_values: np.ndarray
    cbar: np.ndarray
    cbar_values: np.ndarray


def _class_has_hidden_slots(instance_type: type) -> bool:
    for base in type.__getattribute__(instance_type, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        raw_slots = namespace.get("__slots__", ())
        if type(raw_slots) is str:
            slots = (raw_slots,)
        elif type(raw_slots) is tuple and all(type(slot) is str for slot in raw_slots):
            slots = raw_slots
        else:
            raise TypeError("IMM model slot metadata must be an exact string tuple")
        if any(slot not in ("__dict__", "__weakref__") for slot in slots):
            return True
    return False


def _static_class_attribute(instance_type: type, name: str) -> object | None:
    for base in type.__getattribute__(instance_type, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return None


def _validate_imm_class_surface(instance_type: type, name: str) -> None:
    safe_descriptor_types = (
        FunctionType,
        property,
        staticmethod,
        classmethod,
        GetSetDescriptorType,
        MemberDescriptorType,
        MethodDescriptorType,
        WrapperDescriptorType,
        ClassMethodDescriptorType,
    )
    for base in type.__getattribute__(instance_type, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        for attribute, value in namespace.items():
            if type(value) in safe_descriptor_types:
                continue
            if _static_class_attribute(type(value), "__get__") is not None:
                raise TypeError(f"{name}.{attribute} uses an unsupported custom descriptor")


def _ordinary_instance_namespace(
    instance: object,
    name: str,
) -> tuple[dict[str, object], Any]:
    if _class_has_hidden_slots(type(instance)):
        raise TypeError(f"{name} must not contain hidden slot state")
    _validate_imm_class_surface(type(instance), name)
    if _static_class_attribute(type(instance), "__del__") is not None:
        raise TypeError(f"{name} must not define a finalizer")
    descriptor = _static_class_attribute(type(instance), "__dict__")
    if type(descriptor) is not GetSetDescriptorType:
        raise TypeError(f"{name} must expose an ordinary instance namespace")
    try:
        namespace = descriptor.__get__(instance, type(instance))
    except BaseException as exc:
        raise TypeError(f"{name} must expose an ordinary instance namespace") from exc
    if type(namespace) is not dict:
        raise TypeError(f"{name} must expose an ordinary instance namespace")
    if any(type(key) is not str for key in namespace):
        raise TypeError(f"{name} namespace keys must be exact strings")
    return cast(dict[str, object], namespace), descriptor


def _safe_imm_dict_key(value: object) -> bool:
    pending = [value]
    nodes = 0
    while pending:
        current = pending.pop()
        nodes += 1
        if nodes > _MAX_IMM_TRANSACTION_NODES:
            return False
        current_type = type(current)
        if current_type in _SUPPORTED_REAL_SCALAR_TYPES or current_type in (
            type(None),
            bool,
            np.bool_,
            str,
            bytes,
        ):
            continue
        if current_type is tuple:
            pending.extend(cast(tuple[object, ...], current))
            continue
        if current_type is frozenset:
            pending.extend(cast(frozenset[object], current))
            continue
        return False
    return True


def _copy_inert_rng_state(value: object) -> object:
    value_type = type(value)
    if value_type in _SUPPORTED_REAL_SCALAR_TYPES or value_type in (
        type(None),
        bool,
        np.bool_,
        str,
        bytes,
    ):
        return value
    if value_type is np.ndarray:
        array = cast(np.ndarray, value)
        if array.dtype.hasobject:
            raise TypeError("IMM RNG state must not contain object arrays")
        return array.copy(order="K")
    if value_type is list:
        return [_copy_inert_rng_state(item) for item in cast(list[object], value)]
    if value_type is tuple:
        return tuple(_copy_inert_rng_state(item) for item in cast(tuple[object, ...], value))
    if value_type is dict:
        result: dict[object, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if not _safe_imm_dict_key(key):
                raise TypeError("IMM RNG state contains an unsafe dictionary key")
            result[key] = _copy_inert_rng_state(item)
        return result
    raise TypeError(f"IMM RNG state has unsafe type {value_type.__name__}")


def _inert_rng_states_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is np.ndarray:
        left_array = cast(np.ndarray, left)
        right_array = cast(np.ndarray, right)
        return (
            left_array.dtype == right_array.dtype
            and left_array.shape == right_array.shape
            and np.array_equal(left_array.view(np.uint8), right_array.view(np.uint8))
        )
    if type(left) in (list, tuple):
        left_values = cast(list[object] | tuple[object, ...], left)
        right_values = cast(list[object] | tuple[object, ...], right)
        return len(left_values) == len(right_values) and all(
            _inert_rng_states_equal(a, b) for a, b in zip(left_values, right_values)
        )
    if type(left) is dict:
        left_dict = cast(dict[object, object], left)
        right_dict = cast(dict[object, object], right)
        return left_dict.keys() == right_dict.keys() and all(
            _inert_rng_states_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    return bool(left == right)


def _snapshot_imm_graph(instances: tuple[object, ...]) -> _IMMGraphJournal:
    object_undos: list[_IMMObjectUndo] = []
    dict_undos: list[tuple[dict, tuple[tuple[object, object], ...]]] = []
    list_undos: list[tuple[list, tuple[object, ...]]] = []
    set_undos: list[tuple[set, frozenset[object]]] = []
    deque_undos: list[tuple[deque, tuple[object, ...]]] = []
    array_undos: list[_IMMArrayUndo] = []
    generator_undos: list[_IMMGeneratorUndo] = []
    pending: list[tuple[object, str]] = []
    preserved_ids = {id(instance) for instance in instances}
    for index, instance in enumerate(instances):
        namespace, descriptor = _ordinary_instance_namespace(
            instance,
            "IMM estimator" if index == 0 else f"models[{index - 1}]",
        )
        object_undos.append(_IMMObjectUndo(instance, type(instance), namespace, descriptor))
        pending.append((namespace, "IMM transaction state"))

    seen: set[int] = set(preserved_ids)
    nodes = 0
    array_cells = 0
    array_bytes = 0
    while pending:
        current, current_name = pending.pop()
        current_type = type(current)
        if current_type in _SUPPORTED_REAL_SCALAR_TYPES or current_type in (
            type(None),
            bool,
            np.bool_,
            str,
            bytes,
            type,
        ):
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        nodes += 1
        if nodes > _MAX_IMM_TRANSACTION_NODES:
            raise ValueError("IMM transaction graph exceeds the node safety limit")

        if current_type is np.ndarray:
            array = cast(np.ndarray, current)
            if array.dtype.hasobject or array.dtype.kind not in _REAL_NUMERIC_KINDS:
                raise TypeError(f"{current_name} must contain only real numeric arrays")
            if array.base is not None or not array.flags.owndata:
                raise TypeError(f"{current_name} arrays must own their storage")
            if not (array.flags.c_contiguous or array.flags.f_contiguous):
                raise TypeError(f"{current_name} arrays must be contiguous")
            array_cells += array.size
            array_bytes += array.nbytes
            if (
                array_cells > _MAX_IMM_TRANSACTION_ARRAY_CELLS
                or array_bytes > _MAX_IMM_TRANSACTION_ARRAY_BYTES
            ):
                raise ValueError("IMM transaction arrays exceed the safety limit")
            array_undos.append(
                _IMMArrayUndo(
                    array=array,
                    value=array.copy(order="K"),
                    dtype=array.dtype,
                    shape=array.shape,
                    writeable=bool(array.flags.writeable),
                )
            )
            continue
        if current_type is np.random.Generator:
            generator = _validate_rng(current)
            state = _copy_inert_rng_state(dict(generator.bit_generator.state))
            generator_undos.append(_IMMGeneratorUndo(generator, cast(dict, state)))
            continue
        if current_type is GaussianState:
            namespace, descriptor = _ordinary_instance_namespace(
                current,
                f"{current_name} GaussianState",
            )
            object_undos.append(_IMMObjectUndo(current, GaussianState, namespace, descriptor))
            pending.append((namespace, f"{current_name} GaussianState namespace"))
            continue
        if current_type is dict:
            mapping = cast(dict[object, object], current)
            if nodes + len(mapping) > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            items = tuple(mapping.items())
            nodes += len(items)
            if nodes > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            if any(not _safe_imm_dict_key(key) for key, _ in items):
                raise TypeError(f"{current_name} contains an unsafe dictionary key")
            dict_undos.append((mapping, items))
            pending.extend((value, f"{current_name}[{key!r}]") for key, value in items)
            pending.extend((key, f"{current_name} key") for key, _ in items)
            continue
        if current_type is list:
            list_source = cast(list[object], current)
            if nodes + len(list_source) > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            list_items = tuple(list_source)
            nodes += len(list_items)
            if nodes > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            list_undos.append((cast(list, current), list_items))
            pending.extend((value, f"{current_name}[]") for value in list_items)
            continue
        if current_type is tuple:
            if nodes + len(cast(tuple[object, ...], current)) > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            nodes += len(cast(tuple[object, ...], current))
            if nodes > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            pending.extend(
                (value, f"{current_name}[]") for value in cast(tuple[object, ...], current)
            )
            continue
        if current_type is set:
            set_source = cast(set[object], current)
            if nodes + len(set_source) > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            set_items = frozenset(set_source)
            nodes += len(set_items)
            if nodes > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            if any(not _safe_imm_dict_key(value) for value in set_items):
                raise TypeError(f"{current_name} contains an unsafe set value")
            set_undos.append((cast(set, current), set_items))
            pending.extend((value, f"{current_name} set value") for value in set_items)
            continue
        if current_type is frozenset:
            frozen_items = cast(frozenset[object], current)
            if nodes + len(frozen_items) > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            nodes += len(frozen_items)
            if nodes > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            if any(not _safe_imm_dict_key(value) for value in frozen_items):
                raise TypeError(f"{current_name} contains an unsafe frozenset value")
            pending.extend((value, f"{current_name} frozenset value") for value in frozen_items)
            continue
        if current_type is deque:
            deque_source = cast(deque[object], current)
            if nodes + len(deque_source) > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            deque_items = tuple(deque_source)
            nodes += len(deque_items)
            if nodes > _MAX_IMM_TRANSACTION_NODES:
                raise ValueError("IMM transaction graph exceeds the node safety limit")
            deque_undos.append((cast(deque, current), deque_items))
            pending.extend((value, f"{current_name}[]") for value in deque_items)
            continue
        raise TypeError(
            f"{current_name} has unsafe type {current_type.__name__} for IMM transactions"
        )

    return _IMMGraphJournal(
        objects=tuple(object_undos),
        dicts=tuple(dict_undos),
        lists=tuple(list_undos),
        sets=tuple(set_undos),
        deques=tuple(deque_undos),
        arrays=tuple(array_undos),
        generators=tuple(generator_undos),
    )


def _imm_graph_is_unchanged(journal: _IMMGraphJournal) -> bool:
    for object_undo in journal.objects:
        if type(object_undo.instance) is not object_undo.instance_type:
            return False
        try:
            namespace = object_undo.namespace_descriptor.__get__(
                object_undo.instance,
                object_undo.instance_type,
            )
        except BaseException:
            return False
        if namespace is not object_undo.namespace:
            return False
    for mapping, dict_items in journal.dicts:
        current_items = tuple(mapping.items())
        if len(current_items) != len(dict_items) or any(
            left_key is not right_key or left_value is not right_value
            for (left_key, left_value), (right_key, right_value) in zip(
                current_items,
                dict_items,
            )
        ):
            return False
    for list_value, list_items in journal.lists:
        if len(list_value) != len(list_items) or any(
            a is not b for a, b in zip(list_value, list_items)
        ):
            return False
    for set_value, set_items in journal.sets:
        if len(set_value) != len(set_items) or {id(value) for value in set_value} != {
            id(value) for value in set_items
        }:
            return False
    for deque_value, deque_items in journal.deques:
        if len(deque_value) != len(deque_items) or any(
            a is not b for a, b in zip(deque_value, deque_items)
        ):
            return False
    for array_undo in journal.arrays:
        array = array_undo.array
        if (
            array.dtype != array_undo.dtype
            or array.shape != array_undo.shape
            or bool(array.flags.writeable) != array_undo.writeable
            or array.tobytes(order="A") != array_undo.value.tobytes(order="A")
        ):
            return False
    for generator_undo in journal.generators:
        current_state = _copy_inert_rng_state(dict(generator_undo.generator.bit_generator.state))
        if not _inert_rng_states_equal(current_state, generator_undo.state):
            return False
    return True


def _restore_imm_graph(journal: _IMMGraphJournal) -> None:
    quarantine: list[object] = []
    for mapping, _ in journal.dicts:
        quarantine.extend(mapping.keys())
        quarantine.extend(mapping.values())
    for list_value, _ in journal.lists:
        quarantine.extend(list_value)
    for set_value, _ in journal.sets:
        quarantine.extend(set_value)
    for deque_value, _ in journal.deques:
        quarantine.extend(deque_value)

    def restore_all() -> None:
        for object_undo in journal.objects:
            if type(object_undo.instance) is not object_undo.instance_type:
                _OBJECT_CLASS_DESCRIPTOR.__set__(
                    object_undo.instance,
                    object_undo.instance_type,
                )
        for object_undo in journal.objects:
            object_undo.namespace_descriptor.__set__(
                object_undo.instance,
                object_undo.namespace,
            )
        for mapping, dict_items in journal.dicts:
            mapping.clear()
            mapping.update(dict_items)
        for list_value, list_items in journal.lists:
            list_value[:] = list_items
        for set_value, set_items in journal.sets:
            set_value.clear()
            set_value.update(set_items)
        for deque_value, deque_items in journal.deques:
            deque_value.clear()
            deque_value.extend(deque_items)
        # Restore mutable value objects last so any callback-introduced finalizer
        # triggered by the structural cleanup cannot leave them corrupted.
        for array_undo in journal.arrays:
            array = array_undo.array
            if not array.flags.writeable:
                array.setflags(write=True)
            if array.dtype != array_undo.dtype:
                array.dtype = array_undo.dtype  # type: ignore[misc]
            if array.shape != array_undo.shape:
                array.resize(array_undo.shape, refcheck=False)
            np.copyto(array, array_undo.value, casting="no")
            if not array_undo.writeable:
                array.setflags(write=False)
        for generator_undo in journal.generators:
            state = _copy_inert_rng_state(generator_undo.state)
            generator_undo.generator.bit_generator.state = cast(dict, state)

    restore_all()
    quarantine.clear()
    # Reapply once after callback-introduced finalizers have run. Initial graph
    # values with finalizers are rejected, so every pre-existing item is still
    # strongly held by the journal and cannot execute here.
    restore_all()


class IMMEstimator:
    """IMM over a bank of KF-like models mixed by a Markov transition matrix.

    Each model must expose ``.state`` (a :class:`GaussianState`), ``predict(dt)``,
    ``update(z, R)`` and ``.likelihood``. The default bank is two constant-
    velocity Kalman filters — a quiescent one and a high-manoeuvre one — a
    standard maneuver-adaptive configuration. A coordinated-turn model can
    approximate the same model-family choice used elsewhere, but parity still
    requires full configuration and intermediate-state fixtures.

    ``transition`` is a discrete probability matrix applied once per non-zero
    ``predict`` call; it is scan/event indexed rather than a continuous-time
    transition rate. Changing the caller's prediction cadence therefore changes
    the mode prior by design.

    Model callbacks execute inside an in-place undo journal. To keep that
    journal callback-free and failure-atomic, model instance state is restricted
    to exact inert scalars, exact built-in containers, owning numeric ndarrays,
    :class:`GaussianState`, and NumPy generators backed by built-in bit
    generators. Hidden slots, custom payload objects/copy protocols, object
    arrays, and hostile namespace descriptors are rejected before a callback.
    Rollback covers this admitted object graph, including caller-held aliases;
    process-global state, class dictionaries, files, threads, and other external
    side effects of arbitrary model code are outside the in-process guarantee.
    """

    def __init__(
        self,
        models: list,
        transition: np.ndarray | None = None,
        mode_probs: np.ndarray | None = None,
    ):
        if isinstance(models, (str, bytes)):
            raise TypeError("models must be a non-empty sequence of filter models")
        try:
            self.models = list(islice(iter(models), _MAX_IMM_MODELS + 1))
        except TypeError as exc:
            raise TypeError("models must be a non-empty sequence of filter models") from exc
        m = len(self.models)
        if m == 0:
            raise ValueError("models must not be empty")
        if m > _MAX_IMM_MODELS:
            raise ValueError(f"models exceeds the {_MAX_IMM_MODELS}-mode safety limit")
        if len({id(model) for model in self.models}) != m:
            raise ValueError("models must contain distinct filter objects")
        for index, model in enumerate(self.models):
            if getattr_static(model, "predict", None) is None:
                raise TypeError(f"models[{index}] must provide predict(dt)")
            if getattr_static(model, "update", None) is None:
                raise TypeError(f"models[{index}] must provide update(z, R)")
            if getattr_static(model, "state", None) is None:
                raise TypeError(f"models[{index}] must expose Gaussian state")
            if getattr_static(model, "likelihood", None) is None:
                raise TypeError(f"models[{index}] must expose likelihood")

        construction_journal = _snapshot_imm_graph((self, *self.models))
        try:
            dimensions: list[int] = []
            for index, model in enumerate(self.models):
                model_dim, _, _, _ = IMMEstimator._validate_model(model, index)
                # Mixing replaces each model's Gaussian prior. Read-only state
                # properties (for example ParticleFilter.state) are incompatible.
                state_descriptor = getattr_static(type(model), "state", None)
                if isinstance(state_descriptor, property) and state_descriptor.fset is None:
                    raise TypeError(f"models[{index}].state must be assignable")
                IMMEstimator._model_log_likelihood(model, index)
                dimensions.append(model_dim)
            if not _imm_graph_is_unchanged(construction_journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            if len(set(dimensions)) != 1:
                raise ValueError("all IMM models must have the same dimension")
            self.dim = dimensions[0]

            if transition is None:
                if m == 1:
                    transition = np.ones((1, 1))
                else:
                    transition = np.full((m, m), 0.05 / (m - 1))
                    np.fill_diagonal(transition, 0.95)
            self.transition = IMMEstimator._validate_transition(transition, m)
            probabilities = np.full(m, 1.0 / m) if mode_probs is None else mode_probs
            self.mode_probs = IMMEstimator._validate_probabilities(probabilities, m, "mode_probs")
            # Predicted mode probabilities; set here so update() before the first
            # predict() degrades to the prior instead of raising AttributeError.
            self._cbar = self.mode_probs.copy()
        except BaseException:
            _restore_imm_graph(construction_journal)
            raise

    @staticmethod
    def _validate_model(
        model,
        index: int,
    ) -> tuple[int, GaussianState, np.ndarray, np.ndarray]:
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"models[{index}] must provide predict(dt)")
        if not callable(getattr(model, "update", None)):
            raise TypeError(f"models[{index}] must provide update(z, R)")
        if getattr_static(model, "likelihood", None) is None:
            raise TypeError(f"models[{index}] must expose likelihood")
        model_dim = _validate_dim(getattr(model, "dim", None))
        state = getattr(model, "state", None)
        if type(model) in (KalmanFilter, ExtendedKalmanFilter):
            runtime_dim, x, P, _ = KalmanFilter._runtime(
                model,
                canonicalize_state=False,
            )
            if runtime_dim != model_dim:
                raise ValueError(f"models[{index}] dimension was corrupted")
        elif type(model) is UnscentedKalmanFilter:
            runtime_dim, _, x, P, _, _, _, _ = UnscentedKalmanFilter._runtime(
                model,
                canonicalize_state=False,
            )
            if runtime_dim != model_dim:
                raise ValueError(f"models[{index}] dimension was corrupted")
        else:
            x, P = _canonical_filter_state(state, model_dim)
        return model_dim, cast(GaussianState, state), x, P

    @staticmethod
    def _model_log_likelihood(model, index: int) -> float:
        missing = object()
        if getattr_static(model, "log_likelihood", missing) is not missing:
            raw_value = model.log_likelihood
            try:
                value = _float_scalar(raw_value, f"models[{index}] log likelihood")
            except ValueError as exc:
                raise FloatingPointError(
                    f"models[{index}] produced an invalid log likelihood"
                ) from exc
            if np.isnan(value) or np.isposinf(value):
                raise FloatingPointError(f"models[{index}] produced an invalid log likelihood")
            return value

        raw_value = model.likelihood
        try:
            likelihood = _float_scalar(raw_value, f"models[{index}] likelihood")
        except ValueError as exc:
            raise FloatingPointError(f"models[{index}] produced an invalid likelihood") from exc
        if not np.isfinite(likelihood) or likelihood < 0:
            raise FloatingPointError(f"models[{index}] produced an invalid likelihood")
        return -np.inf if likelihood == 0 else float(np.log(likelihood))

    @staticmethod
    def _validate_transition(transition: np.ndarray, size: int) -> np.ndarray:
        raw = _raw_real_array(
            transition,
            "transition",
            allowed_shapes=((size, size),),
        )
        if raw.shape != (size, size):
            raise ValueError(f"transition must have shape ({size}, {size}), got {raw.shape}")
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("transition must contain finite nonnegative probabilities")
        matrix = _float_array(raw, "transition")
        if not np.isfinite(matrix).all():
            raise ValueError("transition must contain finite nonnegative probabilities")
        with np.errstate(over="ignore", invalid="ignore"):
            row_sums = matrix.sum(axis=1)
        if not np.isfinite(row_sums).all() or not np.allclose(
            row_sums,
            1.0,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("each transition row must sum to 1")
        return matrix.copy()

    @staticmethod
    def _validate_probabilities(values: np.ndarray, size: int, name: str) -> np.ndarray:
        raw = _raw_real_array(values, name, allowed_shapes=((size,),))
        if raw.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {raw.shape}")
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError(f"{name} must contain finite nonnegative probabilities")
        probabilities = _float_array(raw, name)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name} must contain finite nonnegative probabilities")
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(probabilities.sum())
        if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must sum to 1")
        return probabilities.copy()

    def _validate_runtime(self) -> _IMMRuntime:
        if type(self.models) not in (list, tuple) or len(self.models) == 0:
            raise ValueError("models must remain a non-empty sequence")
        size = len(self.models)
        if size > _MAX_IMM_MODELS:
            raise ValueError(f"models exceeds the {_MAX_IMM_MODELS}-mode safety limit")
        if len({id(model) for model in self.models}) != size:
            raise ValueError("models must retain distinct filter objects")
        configured_dim = _validate_dim(self.dim)
        states: list[tuple[GaussianState, np.ndarray, np.ndarray]] = []
        for index, model in enumerate(self.models):
            model_dim, state, x, P = IMMEstimator._validate_model(model, index)
            if model_dim != configured_dim:
                raise ValueError("all IMM models must retain the configured dimension")
            states.append((state, x, P))
        transition = IMMEstimator._validate_transition(self.transition, size)
        mode_probs = IMMEstimator._validate_probabilities(
            self.mode_probs,
            size,
            "mode_probs",
        )
        cbar = IMMEstimator._validate_probabilities(
            self._cbar,
            size,
            "predicted mode probabilities",
        )
        return _IMMRuntime(
            models=tuple(cast(list[_IMMModel] | tuple[_IMMModel, ...], self.models)),
            states=tuple(states),
            dim=configured_dim,
            transition=transition,
            mode_probs=mode_probs,
            cbar=cbar,
        )

    @staticmethod
    def _commit_runtime(self: IMMEstimator, runtime: _IMMRuntime) -> None:
        for state, x, P in runtime.states:
            state_namespace, _ = _ordinary_instance_namespace(state, "IMM model state")
            state_namespace["x"] = x
            state_namespace["P"] = P
        owner_namespace, _ = _ordinary_instance_namespace(self, "IMM estimator")
        owner_namespace["transition"] = runtime.transition
        owner_namespace["mode_probs"] = runtime.mode_probs
        owner_namespace["_cbar"] = runtime.cbar

    def _snapshot_transaction(self) -> _IMMTransaction:
        owner_namespace, _ = _ordinary_instance_namespace(self, "IMM estimator")
        models_container = owner_namespace.get("models")
        if type(models_container) not in (list, tuple) or not models_container:
            raise ValueError("models must remain a non-empty sequence")
        models_container = cast("list[Any] | tuple[Any, ...]", models_container)
        if len(models_container) > _MAX_IMM_MODELS:
            raise ValueError(f"models exceeds the {_MAX_IMM_MODELS}-mode safety limit")
        models = tuple(models_container)
        if len({id(model) for model in models}) != len(models):
            raise ValueError("models must retain distinct filter objects")
        journal = _snapshot_imm_graph((self, *models))
        model_namespaces = tuple(
            cast(dict[str, object], undo.namespace) for undo in journal.objects[1 : 1 + len(models)]
        )
        return _IMMTransaction(
            journal=journal,
            owner_namespace=owner_namespace,
            owner_type=type(self),
            models_container=models_container,
            models=cast(tuple[_IMMModel, ...], models),
            model_namespaces=model_namespaces,
            model_types=tuple(type(model) for model in models),
            transition=cast(np.ndarray, owner_namespace.get("transition")),
            mode_probs=cast(np.ndarray, owner_namespace.get("mode_probs")),
            cbar=cast(np.ndarray, owner_namespace.get("_cbar")),
        )

    @staticmethod
    def _restore_transaction(self: IMMEstimator, transaction: _IMMTransaction) -> None:
        _restore_imm_graph(transaction.journal)

    @staticmethod
    def _capture_seal(self: IMMEstimator, transaction: _IMMTransaction) -> _IMMSeal:
        if type(self) is not transaction.owner_type or any(
            type(model) is not model_type
            for model, model_type in zip(transaction.models, transaction.model_types)
        ):
            raise ValueError("IMM object types changed during a model callback")
        owner_namespace = transaction.journal.objects[0].namespace_descriptor.__get__(
            self,
            transaction.owner_type,
        )
        models_container = owner_namespace.get("models")
        transition = owner_namespace.get("transition")
        mode_probs = owner_namespace.get("mode_probs")
        cbar = owner_namespace.get("_cbar")
        if (
            type(owner_namespace) is not dict
            or type(models_container) not in (list, tuple)
            or type(transition) is not np.ndarray
            or type(mode_probs) is not np.ndarray
            or type(cbar) is not np.ndarray
        ):
            raise ValueError("IMM configuration changed during a model callback")
        return _IMMSeal(
            owner_type=transaction.owner_type,
            owner_namespace=owner_namespace,
            owner_keys=frozenset(owner_namespace),
            models_container=models_container,
            models=transaction.models,
            model_types=transaction.model_types,
            dim_type=type(owner_namespace.get("dim")),
            dim=owner_namespace.get("dim"),
            transition=transition,
            transition_values=transition.copy(),
            mode_probs=mode_probs,
            mode_prob_values=mode_probs.copy(),
            cbar=cbar,
            cbar_values=cbar.copy(),
        )

    @staticmethod
    def _require_seal(
        self: IMMEstimator,
        transaction: _IMMTransaction,
        seal: _IMMSeal,
    ) -> None:
        if type(self) is not seal.owner_type or any(
            type(model) is not model_type
            for model, model_type in zip(transaction.models, seal.model_types)
        ):
            raise ValueError("IMM configuration changed during a model callback")
        try:
            current_namespace = transaction.journal.objects[0].namespace_descriptor.__get__(
                self,
                seal.owner_type,
            )
        except BaseException as exc:
            raise ValueError("IMM configuration changed during a model callback") from exc
        models_container = current_namespace.get("models")
        if (
            current_namespace is not seal.owner_namespace
            or frozenset(current_namespace) != seal.owner_keys
            or models_container is not seal.models_container
            or len(models_container) != len(seal.models)
            or any(
                current is not original for current, original in zip(models_container, seal.models)
            )
            or type(current_namespace.get("dim")) is not seal.dim_type
            or current_namespace.get("dim") != seal.dim
            or current_namespace.get("transition") is not seal.transition
            or current_namespace.get("mode_probs") is not seal.mode_probs
            or current_namespace.get("_cbar") is not seal.cbar
            or not np.array_equal(seal.transition, seal.transition_values)
            or not np.array_equal(seal.mode_probs, seal.mode_prob_values)
            or not np.array_equal(seal.cbar, seal.cbar_values)
        ):
            raise ValueError("IMM configuration changed during a model callback")

    @classmethod
    def default_cv_bank(
        cls,
        x0,
        P0,
        dim: int = POS_DIM,
        sigma_a: float = 1.0,
    ) -> IMMEstimator:
        # Use EKF models so the bank honours the radar polar contract (they behave
        # as a linear KF for Cartesian updates and add update_polar for radar). The
        # configured acceleration scale is the quiet model; the maneuver model has
        # ten times that standard deviation while retaining the historical 1:10
        # default ratio.
        quiet_sigma = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        if quiet_sigma > np.finfo(float).max / 10.0:
            raise ValueError("sigma_a is too large for the IMM maneuver model")
        maneuver_sigma = quiet_sigma * 10.0
        quiet = ExtendedKalmanFilter(x0, P0, sigma_a=quiet_sigma, dim=dim)
        maneuver = ExtendedKalmanFilter(
            quiet.state.x.copy(),
            quiet.state.P.copy(),
            sigma_a=maneuver_sigma,
            dim=dim,
        )
        return cls([quiet, maneuver])

    # -- mixing → predict → update → combine -----------------------------
    @staticmethod
    def _mix_runtime(runtime: _IMMRuntime) -> tuple[list[GaussianState], np.ndarray]:
        m = len(runtime.models)
        cbar = runtime.transition.T @ runtime.mode_probs  # predicted mode probs
        if not np.isfinite(cbar).all() or np.any(cbar < 0):
            raise FloatingPointError("predicted IMM mode probabilities are invalid")
        mixed: list[GaussianState] = []
        for j in range(m):
            if cbar[j] == 0:
                _, x, P = runtime.states[j]
                mixed.append(GaussianState(x.copy(), P.copy()))
                continue
            mu_ij = runtime.transition[:, j] * runtime.mode_probs / cbar[j]
            mixing_total = float(mu_ij.sum())
            if not np.isfinite(mu_ij).all() or mixing_total <= 0:
                raise FloatingPointError("IMM mixing probabilities are invalid")
            mu_ij /= mixing_total
            with np.errstate(over="ignore", invalid="ignore"):
                x0 = sum(
                    (mu_ij[i] * runtime.states[i][1] for i in range(m)),
                    start=np.zeros_like(runtime.states[j][1]),
                )
            if not np.isfinite(x0).all():
                raise FloatingPointError("mixed IMM state is not finite")
            P0 = np.zeros_like(runtime.states[j][2])
            with np.errstate(over="ignore", invalid="ignore"):
                for i in range(m):
                    d = runtime.states[i][1] - x0
                    P0 += mu_ij[i] * (runtime.states[i][2] + np.outer(d, d))
            mixed.append(GaussianState(x0, _as_covariance(P0, 2 * runtime.dim, "mixed P")))
        return mixed, IMMEstimator._validate_probabilities(
            cbar,
            m,
            "predicted mode probabilities",
        )

    @staticmethod
    def _validate_without_side_effects(
        self: IMMEstimator,
        transaction: _IMMTransaction,
    ) -> _IMMRuntime:
        validation_journal = _snapshot_imm_graph((self, *transaction.models))
        runtime = IMMEstimator._validate_runtime(self)
        if not _imm_graph_is_unchanged(validation_journal):
            raise ValueError("IMM model validation must not mutate transaction state")
        return runtime

    @staticmethod
    def _restore_original_exception(
        self: IMMEstimator,
        transaction: _IMMTransaction,
        exc: BaseException,
    ) -> None:
        try:
            IMMEstimator._restore_transaction(self, transaction)
        except BaseException as rollback_exc:
            try:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(f"IMM rollback also failed: {rollback_exc!r}")
            except BaseException:
                pass

    def predict(self, dt: float) -> None:
        trusted_type = IMMEstimator
        snapshot_transaction = trusted_type._snapshot_transaction
        validate_runtime = trusted_type._validate_runtime
        commit_runtime = trusted_type._commit_runtime
        capture_seal = trusted_type._capture_seal
        require_seal = trusted_type._require_seal
        mix_runtime = trusted_type._mix_runtime
        validate_model = trusted_type._validate_model
        snapshot_graph = _snapshot_imm_graph
        graph_unchanged = _imm_graph_is_unchanged
        namespace_reader = _ordinary_instance_namespace
        restore_graph = _restore_imm_graph
        dt = _validate_dt(dt)
        transaction = snapshot_transaction(self)
        try:
            runtime = validate_runtime(self)
            if not graph_unchanged(transaction.journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            commit_runtime(self, runtime)
            seal = capture_seal(self, transaction)
            if dt == 0:
                return
            mixed, cbar = mix_runtime(runtime)
            transaction.owner_namespace["_cbar"] = cbar
            seal = capture_seal(self, transaction)
            for index, (model, mstate) in enumerate(zip(transaction.models, mixed)):
                model.state = mstate.copy()
                model.predict(dt)
                require_seal(self, transaction, seal)
                validation_journal = snapshot_graph((self, *transaction.models))
                _, state, x, P = validate_model(model, index)
                if not graph_unchanged(validation_journal):
                    raise ValueError("IMM model validation must not mutate transaction state")
                require_seal(self, transaction, seal)
                state_namespace, _ = namespace_reader(state, "IMM model state")
                state_namespace["x"] = x
                state_namespace["P"] = P
            # The transition step changes the mode prior even before a measurement
            # arrives. Keeping mode_probs at the previous posterior would make gating
            # and the combined predicted state use stale probabilities.
            transaction.owner_namespace["mode_probs"] = cbar.copy()
            seal = capture_seal(self, transaction)
            final_journal = snapshot_graph((self, *transaction.models))
            validate_runtime(self)
            if not graph_unchanged(final_journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            require_seal(self, transaction, seal)
        except BaseException as exc:
            try:
                restore_graph(transaction.journal)
            except BaseException as rollback_exc:
                try:
                    add_note = getattr(exc, "add_note", None)
                    if callable(add_note):
                        add_note(f"IMM rollback also failed: {rollback_exc!r}")
                except BaseException:
                    pass
            raise

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        owner_namespace, _ = _ordinary_instance_namespace(self, "IMM estimator")
        dim = _validate_dim(owner_namespace.get("dim"))
        z, R = _measurement_inputs(z, R, dim)
        IMMEstimator._update_each(self, lambda m: m.update(z, R))

    def update_polar(self, z: np.ndarray, R: np.ndarray, sensor_origin=None) -> None:
        """Polar (radar) update across the bank; requires EKF-style models."""
        owner_namespace, _ = _ordinary_instance_namespace(self, "IMM estimator")
        dim = _validate_dim(owner_namespace.get("dim"))
        if dim != 3:
            raise ValueError("polar radar updates require dim=3")
        z, R = _polar_measurement_inputs(z, R, dim)
        origin = None if sensor_origin is None else _as_vector(sensor_origin, dim, "sensor_origin")
        IMMEstimator._update_each(
            self,
            lambda m: m.update_polar(z, R, origin),
            required_method="update_polar",
        )

    def _update_each(self, do_update, *, required_method: str | None = None) -> None:
        trusted_type = IMMEstimator
        snapshot_transaction = trusted_type._snapshot_transaction
        validate_runtime = trusted_type._validate_runtime
        commit_runtime = trusted_type._commit_runtime
        capture_seal = trusted_type._capture_seal
        require_seal = trusted_type._require_seal
        validate_model = trusted_type._validate_model
        model_log_likelihood = trusted_type._model_log_likelihood
        validate_probabilities = trusted_type._validate_probabilities
        snapshot_graph = _snapshot_imm_graph
        graph_unchanged = _imm_graph_is_unchanged
        namespace_reader = _ordinary_instance_namespace
        restore_graph = _restore_imm_graph
        transaction = snapshot_transaction(self)
        try:
            runtime = validate_runtime(self)
            if not graph_unchanged(transaction.journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            commit_runtime(self, runtime)
            size = len(runtime.models)
            seal = capture_seal(self, transaction)
            if required_method is not None:
                validation_journal = snapshot_graph((self, *transaction.models))
                if any(
                    not callable(getattr(model, required_method, None))
                    for model in transaction.models
                ):
                    raise TypeError(
                        f"every IMM model must provide {required_method} for this update"
                    )
                if not graph_unchanged(validation_journal):
                    raise ValueError("IMM model validation must not mutate transaction state")
                require_seal(self, transaction, seal)
            log_likelihoods = np.empty(size)
            for k, model in enumerate(transaction.models):
                do_update(model)
                require_seal(self, transaction, seal)
                validation_journal = snapshot_graph((self, *transaction.models))
                _, state, x, P = validate_model(model, k)
                if not graph_unchanged(validation_journal):
                    raise ValueError("IMM model validation must not mutate transaction state")
                require_seal(self, transaction, seal)
                state_namespace, _ = namespace_reader(state, "IMM model state")
                state_namespace["x"] = x
                state_namespace["P"] = P
                likelihood_journal = snapshot_graph((self, *transaction.models))
                log_likelihoods[k] = model_log_likelihood(model, k)
                if not graph_unchanged(likelihood_journal):
                    raise ValueError("IMM likelihood access must not mutate transaction state")
                require_seal(self, transaction, seal)
            # A track may receive one update from each modality in the same cycle.
            # Accumulate that evidence from the current posterior rather than reusing
            # the pre-update cbar prior and silently discarding earlier modalities.
            prior_logs = np.full(size, -np.inf)
            positive = seal.mode_probs > 0
            prior_logs[positive] = np.log(seal.mode_probs[positive])
            posterior_logs = prior_logs + log_likelihoods
            normalizer = _logsumexp(posterior_logs)
            if not np.isfinite(normalizer):
                raise FloatingPointError("IMM mode posterior has zero or non-finite evidence")
            posterior = np.exp(posterior_logs - normalizer)
            mode_probs = validate_probabilities(posterior, size, "mode_probs")
            transaction.owner_namespace["mode_probs"] = mode_probs
            transaction.owner_namespace["_cbar"] = mode_probs.copy()
            seal = capture_seal(self, transaction)
            final_journal = snapshot_graph((self, *transaction.models))
            validate_runtime(self)
            if not graph_unchanged(final_journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            require_seal(self, transaction, seal)
        except BaseException as exc:
            try:
                restore_graph(transaction.journal)
            except BaseException as rollback_exc:
                try:
                    add_note = getattr(exc, "add_note", None)
                    if callable(add_note):
                        add_note(f"IMM rollback also failed: {rollback_exc!r}")
                except BaseException:
                    pass
            raise

    @property
    def state(self) -> GaussianState:
        trusted_type = IMMEstimator
        snapshot_transaction = trusted_type._snapshot_transaction
        validate_runtime = trusted_type._validate_runtime
        graph_unchanged = _imm_graph_is_unchanged
        restore_graph = _restore_imm_graph
        transaction = snapshot_transaction(self)
        try:
            runtime = validate_runtime(self)
            if not graph_unchanged(transaction.journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            with np.errstate(over="ignore", invalid="ignore"):
                x = sum(
                    (
                        mu * state_x
                        for mu, (_, state_x, _) in zip(runtime.mode_probs, runtime.states)
                    ),
                    start=np.zeros_like(runtime.states[0][1]),
                )
            if not np.isfinite(x).all():
                raise FloatingPointError("combined IMM state is not finite")
            P = np.zeros_like(runtime.states[0][2])
            with np.errstate(over="ignore", invalid="ignore"):
                for mu, (_, state_x, state_P) in zip(runtime.mode_probs, runtime.states):
                    d = state_x - x
                    P += mu * (state_P + np.outer(d, d))
            return GaussianState(x, _as_covariance(P, 2 * runtime.dim, "combined P"))
        except BaseException as exc:
            try:
                restore_graph(transaction.journal)
            except BaseException as rollback_exc:
                try:
                    add_note = getattr(exc, "add_note", None)
                    if callable(add_note):
                        add_note(f"IMM rollback also failed: {rollback_exc!r}")
                except BaseException:
                    pass
            raise

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        trusted_type = IMMEstimator
        snapshot_transaction = trusted_type._snapshot_transaction
        validate_runtime = trusted_type._validate_runtime
        graph_unchanged = _imm_graph_is_unchanged
        restore_graph = _restore_imm_graph
        transaction = snapshot_transaction(self)
        try:
            runtime = validate_runtime(self)
            if not graph_unchanged(transaction.journal):
                raise ValueError("IMM model validation must not mutate transaction state")
            z, R = _measurement_inputs(z, R, runtime.dim)
            with np.errstate(over="ignore", invalid="ignore"):
                x = sum(
                    (
                        mu * state_x
                        for mu, (_, state_x, _) in zip(runtime.mode_probs, runtime.states)
                    ),
                    start=np.zeros_like(runtime.states[0][1]),
                )
            P = np.zeros_like(runtime.states[0][2])
            with np.errstate(over="ignore", invalid="ignore"):
                for mu, (_, state_x, state_P) in zip(runtime.mode_probs, runtime.states):
                    d = state_x - x
                    P += mu * (state_P + np.outer(d, d))
            state = GaussianState(x, _as_covariance(P, 2 * runtime.dim, "combined P"))
            H = position_measurement_matrix(runtime.dim)
            y = _finite_difference(z, H @ state.x, "innovation")
            S = _as_covariance(
                H @ state.P @ H.T + R,
                runtime.dim,
                "innovation covariance",
                positive_definite=True,
            )
            return _squared_mahalanobis(y, S)
        except BaseException as exc:
            try:
                restore_graph(transaction.journal)
            except BaseException as rollback_exc:
                try:
                    add_note = getattr(exc, "add_note", None)
                    if callable(add_note):
                        add_note(f"IMM rollback also failed: {rollback_exc!r}")
                except BaseException:
                    pass
            raise


FILTERS = {
    "kalman": KalmanFilter,
    "ekf": ExtendedKalmanFilter,
    "ukf": UnscentedKalmanFilter,
    "particle": ParticleFilter,
    "imm": IMMEstimator,
}

__all__ = [
    "POS_DIM",
    "STATE_DIM",
    "MIN_POLAR_HORIZONTAL_RANGE",
    "GaussianState",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "IMMEstimator",
    "cv_transition",
    "cv_process_noise",
    "position_measurement_matrix",
    "wrap_angle",
    "FILTERS",
]
