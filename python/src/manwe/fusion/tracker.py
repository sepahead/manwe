"""Independent multi-sensor, multi-target tracking reference implementation.

One :meth:`MultiSensorTracker.step` runs a full cycle: **predict** every track to
now → **associate** measurements by a Mahalanobis gate → **update** matched tracks
→ **initiate** tentative tracks from the rest → **lifecycle** (sliding-window
M-of-N confirmation, coasting, deletion). Its local measurement convention uses
polar radar observations end-to-end with an EKF and Cartesian points for other
modalities; downstream schemas require an explicit adapter.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from itertools import combinations, islice
from types import FunctionType
from typing import cast

import numpy as np

from .association import CHI2_99, GATE_INF, linear_assignment
from .filters import (
    FILTERS,
    MIN_POLAR_HORIZONTAL_RANGE,
    POS_DIM,
    GaussianState,
    IMMEstimator,
    ParticleFilter,
)

Modality = str  # "visual" | "thermal" | "acoustic" | "radar" | "lidar" | "rf"
CARTESIAN_MODALITIES = {"visual", "thermal", "acoustic", "lidar", "rf"}
VALID_MODALITIES = frozenset(CARTESIAN_MODALITIES | {"radar"})
_TRACKER_FILTER_TYPES = {
    name: FILTERS[name] for name in ("kalman", "ekf", "ukf", "particle", "imm")
}
_TRACK_NAMESPACE_FIELDS = frozenset(
    {
        "id",
        "filt",
        "cfg",
        "class_label",
        "hits",
        "consecutive_misses",
        "age",
        "ever_confirmed",
        "state",
        "state_timestamp",
        "last_measurement_timestamp",
        "updated_this_cycle",
    }
)
_GAUSSIAN_STATE_NAMESPACE_FIELDS = frozenset({"x", "P"})
_LINEAR_FILTER_NAMESPACE_FIELDS = frozenset(
    {
        "dim",
        "sigma_a",
        "state",
        "H",
        "_last_likelihood",
        "_last_log_likelihood",
    }
)
_TRACKER_FILTER_NAMESPACE_FIELDS = {
    _TRACKER_FILTER_TYPES["kalman"]: _LINEAR_FILTER_NAMESPACE_FIELDS,
    _TRACKER_FILTER_TYPES["ekf"]: _LINEAR_FILTER_NAMESPACE_FIELDS,
    _TRACKER_FILTER_TYPES["ukf"]: frozenset(
        {
            "dim",
            "n",
            "sigma_a",
            "state",
            "H",
            "alpha",
            "beta",
            "kappa",
            "_sigma_scale",
            "lambda_",
            "_wm",
            "_wc",
            "_last_likelihood",
            "_last_log_likelihood",
        }
    ),
    _TRACKER_FILTER_TYPES["particle"]: frozenset(
        {
            "dim",
            "sigma_a",
            "H",
            "n_particles",
            "rng",
            "particles",
            "weights",
            "_last_likelihood",
            "_last_log_likelihood",
        }
    ),
    _TRACKER_FILTER_TYPES["imm"]: frozenset({"models", "dim", "transition", "mode_probs", "_cbar"}),
}
TIMESTAMP_ATOL = 1e-6
MAX_ABSOLUTE_TIMESTAMP = 4_000_000_000.0
MAX_LIFECYCLE_WINDOW = 4096
MAX_TRACKS = 10_000
MAX_SUBSTEPS = 10_000
MAX_MEASUREMENTS = 10_000
MAX_PARTICLES = 100_000
MAX_ASSIGNMENT_WORK = 100_000_000
MAX_PARTICLE_POPULATION = 2_000_000
MAX_CLASS_LABEL_BYTES = 256
MAX_SENSOR_ID_BYTES = 128
_MAX_EXACT_FLOAT64_INTEGER = 1 << 53
_MAX_IMM_MODES = 32
_MAX_RAW_SEQUENCE_CELLS = 30_000_000
_REAL_NUMERIC_KINDS = frozenset("iuf")
_FLOAT64_DTYPE = np.dtype(np.float64)
_VALID_TRACK_STATES = frozenset({"tentative", "confirmed", "coasting", "lost"})
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


def _is_supported_real_scalar(value: object) -> bool:
    return type(value) in _SUPPORTED_REAL_SCALAR_TYPES


def _sequence_matches_shape(
    value: object,
    shape: tuple[int, ...],
    active_containers: set[int],
) -> bool:
    """Inspect only container structure, never scalar conversion hooks."""
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
    allowed_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """Reject unsafe structure before inspecting leaves or asking NumPy to allocate."""
    if allowed_shapes is not None and not any(
        _sequence_matches_shape(value, shape, set()) for shape in allowed_shapes
    ):
        expected = " or ".join(str(shape) for shape in allowed_shapes)
        raise ValueError(f"{name} must have shape {expected}")

    # First prove that the complete container graph is acyclic and bounded.
    # This phase intentionally does not inspect scalar leaves: a huge nested
    # shape or cycle must lose before a coercive element can be observed.
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
            if cells > _MAX_RAW_SEQUENCE_CELLS:
                raise ValueError(f"{name} exceeds the {_MAX_RAW_SEQUENCE_CELLS}-value safety limit")
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
            if cells > _MAX_RAW_SEQUENCE_CELLS:
                raise ValueError(f"{name} exceeds the {_MAX_RAW_SEQUENCE_CELLS}-value safety limit")
            continue

    # Container safety is now established, so scalar inspection cannot be
    # delayed behind an unbounded traversal or trapped in a cycle.
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
) -> np.ndarray:
    """Admit an array container without widening or coercing object elements."""
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray:
            raise ValueError(f"{name} must not be an ndarray subclass")
        if allowed_shapes is not None and value.shape not in allowed_shapes:
            expected = " or ".join(str(shape) for shape in allowed_shapes)
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    elif isinstance(value, (list, tuple)):
        _preflight_numeric_sequence(value, name, allowed_shapes=allowed_shapes)
    elif not _is_supported_real_scalar(value):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.dtype.kind not in _REAL_NUMERIC_KINDS:
        raise ValueError(f"{name} must contain real numeric values")
    return array


def _validate_exact_float64_integers(array: np.ndarray, name: str) -> None:
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


def _float64_array(array: np.ndarray, name: str) -> np.ndarray:
    """Widen an already shape-bounded array without silent finite narrowing."""
    _validate_exact_float64_integers(array, name)
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


def _finite_float64_scalar(value: object, name: str) -> float:
    """Validate one real scalar without invoking ``__float__`` on object values."""
    if not _is_supported_real_scalar(value):
        raise ValueError(f"{name} must be a finite scalar number")
    raw = _raw_real_array(value, name)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar number")
    result = float(_float64_array(raw, name))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar number")
    return result


def _as_finite_vector(value: object, name: str) -> np.ndarray:
    raw = _raw_real_array(value, name, allowed_shapes=((POS_DIM,),))
    if raw.shape != (POS_DIM,):
        raise ValueError(f"{name} must have shape ({POS_DIM},), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    array = _float64_array(raw, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_covariance(value: object, name: str = "covariance") -> np.ndarray:
    raw = _raw_real_array(
        value,
        name,
        allowed_shapes=((POS_DIM,), (POS_DIM, POS_DIM)),
    )
    if raw.shape not in ((POS_DIM,), (POS_DIM, POS_DIM)):
        raise ValueError(
            f"{name} must have shape ({POS_DIM},) or ({POS_DIM}, {POS_DIM}), got {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    array = _float64_array(raw, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if array.shape == (POS_DIM,):
        array = np.diag(array)
    if np.any(np.diag(array) < 0):
        raise ValueError(f"{name} diagonal variances must be nonnegative")
    normalizer = max(1.0, float(np.max(np.abs(array))))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = array / normalizer
        symmetric = np.allclose(normalized, normalized.T, rtol=1e-10, atol=1e-10)
    if not symmetric:
        raise ValueError(f"{name} must be symmetric")
    array = 0.5 * array + 0.5 * array.T
    try:
        values, vectors = np.linalg.eigh(0.5 * normalized + 0.5 * normalized.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} could not be decomposed") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError(f"{name} decomposition must remain finite")
    if float(values[0]) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    if values[0] < 0:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            repaired = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
            array = (0.5 * repaired + 0.5 * repaired.T) * normalizer
        if not np.isfinite(array).all():
            raise ValueError(f"{name} repair must remain finite")
    return array.copy()


def _as_state_vector(value: object, name: str = "state.x") -> np.ndarray:
    expected = (2 * POS_DIM,)
    raw = _raw_real_array(value, name, allowed_shapes=(expected,))
    if raw.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    array = _float64_array(raw, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_state_covariance(value: object, name: str = "state.P") -> np.ndarray:
    size = 2 * POS_DIM
    expected = (size, size)
    raw = _raw_real_array(value, name, allowed_shapes=(expected,))
    if raw.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    array = _float64_array(raw, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diag(array) < 0):
        raise ValueError(f"{name} diagonal variances must be nonnegative")
    normalizer = max(1.0, float(np.max(np.abs(array))))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = array / normalizer
        symmetric = np.allclose(normalized, normalized.T, rtol=1e-10, atol=1e-10)
    if not symmetric:
        raise ValueError(f"{name} must be symmetric")
    array = 0.5 * array + 0.5 * array.T
    try:
        values = np.linalg.eigvalsh(0.5 * normalized + 0.5 * normalized.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} could not be decomposed") from exc
    if not np.isfinite(values).all() or values[0] < 0:
        raise ValueError(f"{name} must be finite and positive semidefinite")
    return array.copy()


def _as_probability_array(value: object, size: int, name: str) -> np.ndarray:
    raw = _raw_real_array(value, name, allowed_shapes=((size,),))
    if raw.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {raw.shape}")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    probabilities = _float64_array(raw, name)
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(probabilities.sum())
    if not math.isfinite(total) or not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must sum to 1")
    return probabilities.copy()


def _validated_filter_state(filt: object) -> tuple[np.ndarray, np.ndarray]:
    if type(filt) is ParticleFilter:
        n_particles = _positive_int(
            getattr(filt, "n_particles", None),
            "particle count",
            maximum=MAX_PARTICLES,
        )
        if n_particles < 2 or getattr(filt, "dim", None) != POS_DIM:
            raise ValueError("particle filter must retain its configured three-dimensional shape")
        expected = (n_particles, 2 * POS_DIM)
        raw_particles = _raw_real_array(
            getattr(filt, "particles", None),
            "particles",
            allowed_shapes=(expected,),
        )
        raw_weights = _raw_real_array(
            getattr(filt, "weights", None),
            "weights",
            allowed_shapes=((n_particles,),),
        )
        if raw_particles.shape != expected:
            raise ValueError(f"particles must have shape {expected}, got {raw_particles.shape}")
        if not np.isfinite(raw_particles).all():
            raise ValueError("particles must contain only finite values")
        particles = _float64_array(raw_particles, "particles")
        weights = _as_probability_array(raw_weights, n_particles, "weights")
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.einsum("n,nd->d", weights, particles)
            deviations = particles - x
            covariance = np.einsum("n,ni,nj->ij", weights, deviations, deviations)
        return _as_state_vector(x), _as_state_covariance(covariance)

    if type(filt) is IMMEstimator:
        models = getattr(filt, "models", None)
        if not isinstance(models, list) or not 1 <= len(models) <= _MAX_IMM_MODES:
            raise ValueError("IMM models must remain a bounded non-empty list")
        if len({id(model) for model in models}) != len(models):
            raise ValueError("IMM models must remain distinct")
        probabilities = _as_probability_array(
            getattr(filt, "mode_probs", None),
            len(models),
            "IMM mode probabilities",
        )
        model_states = [_validated_filter_state(model) for model in models]
        with np.errstate(over="ignore", invalid="ignore"):
            x = sum(
                (
                    probability * state_x
                    for probability, (state_x, _) in zip(
                        probabilities,
                        model_states,
                        strict=True,
                    )
                ),
                start=np.zeros(2 * POS_DIM),
            )
            covariance = np.zeros((2 * POS_DIM, 2 * POS_DIM))
            for probability, (state_x, state_covariance) in zip(
                probabilities,
                model_states,
                strict=True,
            ):
                difference = state_x - x
                covariance += probability * (state_covariance + np.outer(difference, difference))
        return _as_state_vector(x), _as_state_covariance(covariance)

    state = getattr(filt, "state", None)
    if state is None:
        raise TypeError("track filter must expose Gaussian state")
    x = _as_state_vector(getattr(state, "x", None))
    covariance = _as_state_covariance(getattr(state, "P", None))
    return x, covariance


def _validate_filter_measurement_matrix(filt: object, name: str) -> None:
    expected_shape = (POS_DIM, 2 * POS_DIM)
    raw = _raw_real_array(
        getattr(filt, "H", None),
        f"{name}.H",
        allowed_shapes=(expected_shape,),
    )
    if raw.shape != expected_shape:
        raise ValueError(f"{name}.H must have shape {expected_shape}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name}.H must contain only finite values")
    matrix = _float64_array(raw, f"{name}.H")
    expected = np.zeros(expected_shape)
    expected[:, :POS_DIM] = np.eye(POS_DIM)
    if not np.array_equal(matrix, expected):
        raise ValueError(f"{name}.H must retain the Cartesian position projection")


def _reject_namespace_ndarray_subclasses(owner: object, name: str) -> None:
    try:
        namespace = vars(owner)
    except TypeError as exc:
        raise TypeError(f"{name} must expose ordinary instance state") from exc
    for attribute, value in namespace.items():
        if isinstance(value, np.ndarray) and type(value) is not np.ndarray:
            raise ValueError(f"{name}.{attribute} must not be an ndarray subclass")


def _validate_exact_namespace(
    owner: object,
    name: str,
    expected_fields: frozenset[str],
    *,
    optional_function_fields: frozenset[str] = frozenset(),
) -> None:
    """Reject injected instance fields before transactional deepcopy."""
    try:
        namespace = vars(owner)
    except TypeError as exc:
        raise TypeError(f"{name} must expose ordinary instance state") from exc
    actual_fields = frozenset(namespace)
    if not expected_fields <= actual_fields or not actual_fields <= (
        expected_fields | optional_function_fields
    ):
        raise ValueError(f"{name} instance namespace was corrupted")
    for attribute in actual_fields - expected_fields:
        if type(namespace[attribute]) is not FunctionType:
            raise TypeError(f"{name}.{attribute} must remain an ordinary function override")


def _positive_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if type(value) not in _SUPPORTED_INTEGER_SCALAR_TYPES:
        raise ValueError(f"{name} must be a positive integer")
    integer_value = cast(int, value)
    if integer_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and integer_value > maximum:
        raise ValueError(f"{name} must not exceed the supported maximum {maximum}")
    return int(integer_value)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) not in _SUPPORTED_INTEGER_SCALAR_TYPES:
        raise ValueError(f"{name} must be a nonnegative integer")
    integer_value = cast(int, value)
    if integer_value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(integer_value)


def _finite_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        converted = _finite_float64_scalar(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number: {exc}") from exc
    if positive and converted <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and converted < 0:
        raise ValueError(f"{name} must be >= 0")
    return converted


def _bounded_optional_text(
    value: object,
    name: str,
    *,
    maximum_bytes: int,
) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{name} must be a bounded printable non-empty string or None")
    # Bound the temporary allocation performed by strip() while retaining the
    # established contract that modest surrounding whitespace is accepted.
    if len(value) > 4 * maximum_bytes:
        raise ValueError(f"{name} must be a bounded printable non-empty string or None")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum_bytes
        or not normalized.isprintable()
        or len(normalized.encode("utf-8")) > maximum_bytes
    ):
        raise ValueError(f"{name} must be a bounded printable non-empty string or None")
    return normalized


def _optional_timestamp(value: object, name: str) -> float | None:
    if value is None:
        return None
    timestamp = _finite_number(value, name)
    if abs(timestamp) > MAX_ABSOLUTE_TIMESTAMP:
        raise ValueError(f"{name} magnitude must not exceed {MAX_ABSOLUTE_TIMESTAMP:g} seconds")
    return timestamp


def _probability_list(value: object, name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a bounded probability sequence or None")
    if not 1 <= len(value) <= _MAX_IMM_MODES:
        raise ValueError(f"{name} must contain between 1 and {_MAX_IMM_MODES} probabilities")
    return _as_probability_array(value, len(value), name).tolist()


# ---------------------------------------------------------------------------
# Measurement + coordinate contract
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    """A single sensor measurement.

    ``position`` frame is selected by Manwe's local ``modality`` convention:
      * ``radar`` → polar ``[range_m, azimuth_rad, elevation_rad]``
      * everything else → Cartesian ``[x, y, z]`` metres
    ``covariance`` is a length-3 diagonal (or 3×3) in the matching units.
    ``velocity`` (Cartesian) only seeds a new track's initial velocity.
    """

    modality: Modality
    position: np.ndarray
    covariance: np.ndarray
    timestamp: float
    velocity: np.ndarray | None = None
    sensor_origin: np.ndarray = field(default_factory=lambda: np.zeros(POS_DIM))
    class_label: str | None = None
    sensor_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.modality) is not str or self.modality not in VALID_MODALITIES:
            raise ValueError(
                f"unknown modality {self.modality!r}; expected one of {sorted(VALID_MODALITIES)}"
            )
        self.position = _as_finite_vector(self.position, "position")
        self.covariance = _as_covariance(self.covariance)
        self.sensor_origin = _as_finite_vector(self.sensor_origin, "sensor_origin")
        self.timestamp = _finite_number(self.timestamp, "timestamp")
        if abs(self.timestamp) > MAX_ABSOLUTE_TIMESTAMP:
            raise ValueError(
                f"timestamp magnitude must not exceed {MAX_ABSOLUTE_TIMESTAMP:g} seconds"
            )
        if self.velocity is not None:
            self.velocity = _as_finite_vector(self.velocity, "velocity")
        if self.modality == "radar":
            if self.position[0] <= MIN_POLAR_HORIZONTAL_RANGE:
                raise ValueError(f"radar range must be > {MIN_POLAR_HORIZONTAL_RANGE:g} m")
            if abs(self.position[1]) > 1_000_000.0:
                raise ValueError("radar azimuth magnitude is too large to canonicalize reliably")
            azimuth = math.remainder(float(self.position[1]), 2.0 * math.pi)
            self.position[1] = -math.pi if azimuth == math.pi else azimuth
            if not -math.pi / 2 <= self.position[2] <= math.pi / 2:
                raise ValueError("radar elevation must be in [-pi/2, pi/2]")
            if (
                self.position[0] * abs(math.cos(float(self.position[2])))
                <= MIN_POLAR_HORIZONTAL_RANGE
            ):
                raise ValueError("radar azimuth is singular on the sensor's vertical axis")
        self.class_label = _bounded_optional_text(
            self.class_label,
            "class_label",
            maximum_bytes=MAX_CLASS_LABEL_BYTES,
        )
        self.sensor_id = _bounded_optional_text(
            self.sensor_id,
            "sensor_id",
            maximum_bytes=MAX_SENSOR_ID_BYTES,
        )


def radar_polar_to_cartesian(pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    pos = _as_finite_vector(pos, "polar position")
    origin = _as_finite_vector(origin, "sensor origin")
    return _radar_polar_to_cartesian_validated(pos, origin)


def _radar_polar_to_cartesian_validated(
    pos: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    r, az, el = pos
    if abs(az) > 1_000_000.0 or abs(el) > 1_000_000.0:
        raise ValueError("polar angle magnitude is too large to convert reliably")
    ce = np.cos(el)
    with np.errstate(over="ignore", invalid="ignore"):
        result = origin + np.array([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)])
    if not np.isfinite(result).all():
        raise ValueError("polar position conversion must remain finite")
    return result


def _radar_polar_jacobian(pos: np.ndarray) -> np.ndarray:
    pos = _as_finite_vector(pos, "polar position")
    return _radar_polar_jacobian_validated(pos)


def _radar_polar_jacobian_validated(pos: np.ndarray) -> np.ndarray:
    r, az, el = pos
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    with np.errstate(over="ignore", invalid="ignore"):
        jacobian = np.array(
            [
                [ce * ca, -r * ce * sa, -r * se * ca],
                [ce * sa, r * ce * ca, -r * se * sa],
                [se, 0.0, r * ce],
            ]
        )
    if not np.isfinite(jacobian).all():
        raise ValueError("polar covariance Jacobian must remain finite")
    return jacobian


def measurement_cartesian(m: Measurement) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position_xyz, covariance_3x3)`` in the common world frame.

    Radar is converted polar→Cartesian with a first-order covariance transform so
    it can be *gated* alongside the other modalities; the EKF still updates in
    polar space to model the true angular error.
    """
    if type(m) is not Measurement:
        raise TypeError("m must be a Measurement")
    # Measurement is intentionally mutable for compatibility. Revalidate a
    # private copy so post-construction mutation cannot bypass this public
    # projection boundary or expose aliases to the caller.
    validated = Measurement(
        modality=m.modality,
        position=m.position,
        covariance=m.covariance,
        timestamp=m.timestamp,
        velocity=m.velocity,
        sensor_origin=m.sensor_origin,
        class_label=m.class_label,
        sensor_id=m.sensor_id,
    )
    return _measurement_cartesian_validated(validated)


def _measurement_cartesian_validated(
    validated: Measurement,
) -> tuple[np.ndarray, np.ndarray]:
    if validated.modality == "radar":
        p = _radar_polar_to_cartesian_validated(
            validated.position,
            validated.sensor_origin,
        )
        J = _radar_polar_jacobian_validated(validated.position)
        with np.errstate(over="ignore", invalid="ignore"):
            C = J @ validated.covariance @ J.T
        if not np.isfinite(C).all():
            raise ValueError("Cartesian radar covariance projection must remain finite")
        C = 0.5 * C + 0.5 * C.T
        return p, C
    return validated.position.copy(), validated.covariance.copy()


# ---------------------------------------------------------------------------
# Track + config
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Validated local tuning and per-cycle resource limits."""

    filter: str = "ekf"  # kalman | ekf | ukf | particle | imm
    sigma_a: float = 3.0  # process-noise accel std (m/s²)
    gate_chi2: float = CHI2_99[3]  # 3-DOF position gate at 0.99
    confirm_hits: int = 3  # M
    confirm_window: int = 5  # N  (M-of-N)
    coast_after_misses: int = 2  # consecutive misses → Coasting
    max_missed_in_window: int = 5  # misses within window → Lost (must be ≤ N)
    max_position_cov_volume: float = 1e9  # det(P_pos), in m^6, → Lost
    max_tracks: int = 200
    init_vel_var: float = 100.0
    init_merge_dist: float = 15.0  # cluster simultaneous new measurements within this radius
    # Gap-budget quantum.  It bounds accepted clock gaps together with
    # ``max_substeps`` but must not partition the discrete per-cycle model:
    # acceleration noise and an IMM Markov transition are defined once per call
    # to ``step``, not once per numerical implementation detail. Consequently,
    # callers changing the actual step cadence are changing the event-indexed
    # stochastic model; this knob only admits or rejects a given clock gap.
    max_dt: float = 1.0
    max_prediction_gap: float = 60.0
    max_substeps: int = 120
    max_measurements: int = 1024
    n_particles: int = 512

    def __post_init__(self) -> None:
        valid = {"kalman", "ekf", "ukf", "particle", "imm"}
        if type(self.filter) is not str or self.filter not in valid:
            raise ValueError(f"unknown filter {self.filter!r}; expected one of {sorted(valid)}")
        object.__setattr__(
            self,
            "sigma_a",
            _finite_number(self.sigma_a, "sigma_a", nonnegative=True),
        )
        object.__setattr__(
            self,
            "gate_chi2",
            _finite_number(self.gate_chi2, "gate_chi2", positive=True),
        )
        for name, maximum in (
            ("confirm_hits", MAX_LIFECYCLE_WINDOW),
            ("confirm_window", MAX_LIFECYCLE_WINDOW),
            ("coast_after_misses", MAX_LIFECYCLE_WINDOW),
            ("max_missed_in_window", MAX_LIFECYCLE_WINDOW),
            ("max_tracks", MAX_TRACKS),
            ("max_substeps", MAX_SUBSTEPS),
            ("max_measurements", MAX_MEASUREMENTS),
            ("n_particles", MAX_PARTICLES),
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=maximum),
            )
        for name, positive, nonnegative in (
            ("max_position_cov_volume", True, False),
            ("init_vel_var", True, False),
            ("init_merge_dist", False, True),
            ("max_dt", True, False),
            ("max_prediction_gap", True, False),
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(
                    getattr(self, name),
                    name,
                    positive=positive,
                    nonnegative=nonnegative,
                ),
            )
        if not math.isfinite(self.max_dt * self.max_substeps):
            raise ValueError("max_dt and max_substeps must produce a finite prediction-gap budget")
        if self.filter == "particle" and self.n_particles < 2:
            raise ValueError("particle tracking requires at least 2 particles")
        if self.confirm_hits > self.confirm_window:
            raise ValueError("confirm_hits must be ≤ confirm_window")
        if self.coast_after_misses > self.confirm_window:
            raise ValueError("coast_after_misses must be ≤ confirm_window")
        if self.max_missed_in_window > self.confirm_window:
            raise ValueError("max_missed_in_window must be ≤ confirm_window")
        smaller = min(self.max_tracks, self.max_measurements)
        larger = max(self.max_tracks, self.max_measurements)
        if smaller * smaller * larger > MAX_ASSIGNMENT_WORK:
            raise ValueError(
                "max_tracks and max_measurements exceed the assignment-work safety limit"
            )
        if (
            self.filter == "particle"
            and self.max_tracks * self.n_particles > MAX_PARTICLE_POPULATION
        ):
            raise ValueError(
                "max_tracks and n_particles exceed the aggregate particle safety limit"
            )


def _validate_tracker_config_runtime(config: object) -> TrackerConfig:
    """Revalidate the frozen configuration against reflective corruption."""
    if type(config) is not TrackerConfig:
        raise TypeError("tracker configuration must remain a TrackerConfig")
    original_values = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(TrackerConfig)
    }
    try:
        validated = TrackerConfig(**original_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("tracker configuration was corrupted") from exc
    for name, original in original_values.items():
        canonical = getattr(validated, name)
        if type(original) is not type(canonical) or original != canonical:
            raise ValueError("tracker configuration must retain canonical scalar values")
    return validated


@dataclass
class TrackOutput:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    state: str
    age: int
    class_label: str | None
    state_timestamp: float | None
    last_measurement_timestamp: float | None
    updated_this_cycle: bool
    mode_probs: list[float] | None = None

    def __post_init__(self) -> None:
        self.id = _positive_int(self.id, "track output id")
        self.position = _as_finite_vector(self.position, "track output position")
        self.velocity = _as_finite_vector(self.velocity, "track output velocity")
        self.covariance = _as_covariance(self.covariance, "track output covariance")
        if type(self.state) is not str or self.state not in _VALID_TRACK_STATES:
            raise ValueError(f"track output state must be one of {sorted(_VALID_TRACK_STATES)}")
        self.age = _nonnegative_int(self.age, "track output age")
        self.class_label = _bounded_optional_text(
            self.class_label,
            "track output class_label",
            maximum_bytes=MAX_CLASS_LABEL_BYTES,
        )
        self.state_timestamp = _optional_timestamp(
            self.state_timestamp,
            "track output state_timestamp",
        )
        self.last_measurement_timestamp = _optional_timestamp(
            self.last_measurement_timestamp,
            "track output last_measurement_timestamp",
        )
        if (
            self.state_timestamp is not None
            and self.last_measurement_timestamp is not None
            and self.last_measurement_timestamp > self.state_timestamp
        ):
            raise ValueError("last measurement timestamp must not exceed state timestamp")
        if not isinstance(self.updated_this_cycle, (bool, np.bool_)):
            raise ValueError("updated_this_cycle must be boolean")
        self.updated_this_cycle = bool(self.updated_this_cycle)
        self.mode_probs = _probability_list(self.mode_probs, "track output mode_probs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "covariance": self.covariance.tolist(),  # position-block 3×3 uncertainty
            "state": self.state,
            "age": self.age,
            "class": self.class_label,
            "state_timestamp": self.state_timestamp,
            "last_measurement_timestamp": self.last_measurement_timestamp,
            "updated_this_cycle": self.updated_this_cycle,
            "mode_probs": self.mode_probs,
        }


class Track:
    def __init__(self, track_id: int, filt, cfg: TrackerConfig, class_label: str | None):
        if type(cfg) is not TrackerConfig:
            raise TypeError("cfg must be a TrackerConfig")
        if not callable(getattr(filt, "gating_distance", None)):
            raise TypeError("filt must provide gating_distance(z, R)")
        if not callable(getattr(filt, "predict", None)):
            raise TypeError("filt must provide predict(dt)")
        if not callable(getattr(filt, "update", None)):
            raise TypeError("filt must provide update(z, R)")
        _validated_filter_state(filt)
        self.id = _positive_int(track_id, "track id")
        self.filt = filt
        self.cfg = cfg
        self.class_label = _bounded_optional_text(
            class_label,
            "track class_label",
            maximum_bytes=MAX_CLASS_LABEL_BYTES,
        )
        self.hits: deque[int] = deque(maxlen=cfg.confirm_window)
        self.consecutive_misses = 0
        self.age = 0
        self.ever_confirmed = False
        self.state = "tentative"
        self.state_timestamp: float | None = None
        self.last_measurement_timestamp: float | None = None
        self.updated_this_cycle = False

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        """Delegate gating to the underlying filter (association works on tracks)."""
        return self.filt.gating_distance(z, R)

    def record(self, hit: bool, timestamp: float | None = None) -> None:
        if not isinstance(hit, (bool, np.bool_)):
            raise ValueError("track hit must be boolean")
        hit = bool(hit)
        if timestamp is not None:
            self.state_timestamp = _optional_timestamp(timestamp, "track state timestamp")
        self.updated_this_cycle = hit
        if hit and timestamp is not None:
            self.last_measurement_timestamp = self.state_timestamp
        self.hits.append(1 if hit else 0)
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1
        self.age += 1
        self._recompute_state()

    def _recompute_state(self) -> None:
        n_hits = sum(self.hits)
        misses_in_window = len(self.hits) - n_hits
        _, covariance = _validated_filter_state(self.filt)
        pos_cov = covariance[:POS_DIM, :POS_DIM]
        determinant_sign, log_cov_volume = np.linalg.slogdet(pos_cov)
        if not np.isfinite(determinant_sign) or determinant_sign < 0 or np.isnan(log_cov_volume):
            raise FloatingPointError("position covariance has an invalid determinant")
        covariance_limit_exceeded = determinant_sign > 0 and log_cov_volume > math.log(
            self.cfg.max_position_cov_volume
        )
        if n_hits >= self.cfg.confirm_hits:
            self.ever_confirmed = True
        if misses_in_window >= self.cfg.max_missed_in_window or covariance_limit_exceeded:
            self.state = "lost"
        elif self.consecutive_misses >= self.cfg.coast_after_misses:
            self.state = "coasting"
        elif self.ever_confirmed:
            self.state = "confirmed"
        else:
            self.state = "tentative"

    def output(self) -> TrackOutput:
        state_x, state_covariance = _validated_filter_state(self.filt)
        mode_probs = None
        if isinstance(self.filt, IMMEstimator):
            mode_probs = self.filt.mode_probs.tolist()
        return TrackOutput(
            id=self.id,
            position=state_x[:POS_DIM],
            velocity=state_x[POS_DIM : 2 * POS_DIM],
            covariance=state_covariance[:POS_DIM, :POS_DIM],
            state=self.state,
            age=self.age,
            class_label=self.class_label,
            state_timestamp=self.state_timestamp,
            last_measurement_timestamp=self.last_measurement_timestamp,
            updated_this_cycle=self.updated_this_cycle,
            mode_probs=mode_probs,
        )


def _preflight_snapshot_graph(value: object, name: str = "tracker state") -> None:
    """Prove that deepcopy cannot reach user-defined object hooks."""
    safe_atomic_types = _SUPPORTED_REAL_SCALAR_TYPES | frozenset({type(None), bool, str, np.bool_})
    safe_state_types = frozenset(
        {
            Track,
            GaussianState,
            *_TRACKER_FILTER_NAMESPACE_FIELDS,
        }
    )
    pending: list[tuple[object, str]] = [(value, name)]
    seen: set[int] = set()
    while pending:
        current, current_name = pending.pop()
        current_type = type(current)
        if current_type in safe_atomic_types:
            continue
        if current_type in (TrackerConfig, FunctionType, np.random.Generator):
            continue
        if current_type is np.ndarray:
            array = cast(np.ndarray, current)
            if array.dtype.kind not in _REAL_NUMERIC_KINDS:
                raise TypeError(f"{current_name} must contain only real numeric arrays")
            continue

        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if current_type in (list, tuple, deque):
            container = cast(
                list[object] | tuple[object, ...] | deque[object],
                current,
            )
            pending.extend(
                (item, f"{current_name}[{index}]") for index, item in enumerate(container)
            )
            continue
        if current_type in safe_state_types:
            pending.extend(
                (item, f"{current_name}.{attribute}") for attribute, item in vars(current).items()
            )
            continue
        raise TypeError(
            f"{current_name} has unsafe type {current_type.__name__} for transactional copying"
        )


class MultiSensorTracker:
    """Recursive multi-target tracker over heterogeneous sensor measurements."""

    def __init__(self, config: TrackerConfig | None = None, rng: np.random.Generator | None = None):
        if config is not None and type(config) is not TrackerConfig:
            raise TypeError("config must be a TrackerConfig or None")
        if rng is not None and type(rng) is not np.random.Generator:
            raise TypeError("rng must be a numpy.random.Generator or None")
        self.cfg = config if config is not None else TrackerConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_t: float | None = None

    def _validate_runtime_state(self) -> None:
        """Validate mutable tracker state before it is copied or consumed."""
        _validate_tracker_config_runtime(self.cfg)
        if type(self.rng) is not np.random.Generator:
            raise TypeError("tracker RNG must remain a numpy.random.Generator")
        if type(self.tracks) is not list:
            raise TypeError("tracker tracks must remain a list")
        if len(self.tracks) > self.cfg.max_tracks:
            raise ValueError("tracker state exceeds the configured track limit")
        next_id = _positive_int(self._next_id, "next track id")
        last_t = _optional_timestamp(self._last_t, "last cycle timestamp")
        if last_t is None and self.tracks:
            raise ValueError("tracker cannot contain tracks before its first committed cycle")

        expected_filter_type = _TRACKER_FILTER_TYPES[self.cfg.filter]
        track_ids: set[int] = set()
        filter_ids: set[int] = set()
        for index, track in enumerate(self.tracks):
            if type(track) is not Track:
                raise TypeError(f"tracks[{index}] must be a Track")
            _validate_exact_namespace(
                track,
                f"tracks[{index}]",
                _TRACK_NAMESPACE_FIELDS,
                optional_function_fields=frozenset({"gating_distance"}),
            )
            _reject_namespace_ndarray_subclasses(track, f"tracks[{index}]")
            if track.cfg is not self.cfg:
                raise ValueError(f"tracks[{index}] must retain the tracker configuration")
            track_id = _positive_int(track.id, f"tracks[{index}].id")
            if track_id in track_ids:
                raise ValueError("track ids must remain unique")
            track_ids.add(track_id)
            if type(track.filt) is not expected_filter_type:
                raise TypeError(
                    f"tracks[{index}] filter must remain a {expected_filter_type.__name__}"
                )
            _validate_exact_namespace(
                track.filt,
                f"tracks[{index}] filter",
                _TRACKER_FILTER_NAMESPACE_FIELDS[type(track.filt)],
            )
            if id(track.filt) in filter_ids:
                raise ValueError("tracks must not share mutable filter instances")
            filter_ids.add(id(track.filt))
            _reject_namespace_ndarray_subclasses(
                track.filt,
                f"tracks[{index}] filter",
            )
            filter_dim = getattr(track.filt, "dim", None)
            if type(filter_dim) is not int or filter_dim != POS_DIM:
                raise ValueError(f"tracks[{index}] filter dimension must remain {POS_DIM}")
            if type(track.filt) not in (ParticleFilter, IMMEstimator):
                filter_state = getattr(track.filt, "state", None)
                if type(filter_state) is not GaussianState:
                    raise TypeError(f"tracks[{index}] filter state must remain a GaussianState")
                _validate_exact_namespace(
                    filter_state,
                    f"tracks[{index}] filter state",
                    _GAUSSIAN_STATE_NAMESPACE_FIELDS,
                )
                _reject_namespace_ndarray_subclasses(
                    filter_state,
                    f"tracks[{index}] filter state",
                )
            if type(track.filt) is not IMMEstimator:
                _validate_filter_measurement_matrix(
                    track.filt,
                    f"tracks[{index}] filter",
                )
                sigma_a = _finite_number(
                    getattr(track.filt, "sigma_a", None),
                    f"tracks[{index}] filter sigma_a",
                    nonnegative=True,
                )
                if sigma_a != self.cfg.sigma_a:
                    raise ValueError(f"tracks[{index}] filter sigma_a was corrupted")
            particle_filter = cast(ParticleFilter, track.filt)
            if type(track.filt) is ParticleFilter and (
                type(particle_filter.n_particles) is not int
                or particle_filter.n_particles != self.cfg.n_particles
                or type(particle_filter.rng) is not np.random.Generator
            ):
                raise ValueError(f"tracks[{index}] particle filter configuration was corrupted")
            if type(track.filt) is IMMEstimator:
                if (
                    type(track.filt.models) is not list
                    or len(track.filt.models) != 2
                    or any(
                        type(model) is not _TRACKER_FILTER_TYPES["ekf"]
                        for model in track.filt.models
                    )
                ):
                    raise ValueError(f"tracks[{index}] IMM model bank was corrupted")
                expected_sigmas = (self.cfg.sigma_a, self.cfg.sigma_a * 10.0)
                for model_index, (model, expected_sigma) in enumerate(
                    zip(track.filt.models, expected_sigmas, strict=True)
                ):
                    _validate_exact_namespace(
                        model,
                        f"tracks[{index}] IMM model {model_index}",
                        _TRACKER_FILTER_NAMESPACE_FIELDS[type(model)],
                    )
                    _reject_namespace_ndarray_subclasses(
                        model,
                        f"tracks[{index}] IMM model {model_index}",
                    )
                    if type(model.state) is not GaussianState:
                        raise TypeError(
                            f"tracks[{index}] IMM model {model_index} state was corrupted"
                        )
                    _validate_exact_namespace(
                        model.state,
                        f"tracks[{index}] IMM model {model_index} state",
                        _GAUSSIAN_STATE_NAMESPACE_FIELDS,
                    )
                    _reject_namespace_ndarray_subclasses(
                        model.state,
                        f"tracks[{index}] IMM model {model_index} state",
                    )
                    if type(model.dim) is not int or model.dim != POS_DIM:
                        raise ValueError(
                            f"tracks[{index}] IMM model {model_index} dimension was corrupted"
                        )
                    _validate_filter_measurement_matrix(
                        model,
                        f"tracks[{index}] IMM model {model_index}",
                    )
                    sigma_a = _finite_number(
                        getattr(model, "sigma_a", None),
                        f"tracks[{index}] IMM model {model_index} sigma_a",
                        nonnegative=True,
                    )
                    if sigma_a != expected_sigma:
                        raise ValueError(f"tracks[{index}] IMM sigma_a was corrupted")
                transition = _raw_real_array(
                    track.filt.transition,
                    "IMM transition",
                    allowed_shapes=((2, 2),),
                )
                if transition.shape != (2, 2):
                    raise ValueError("IMM transition must retain shape (2, 2)")
                if not np.isfinite(transition).all() or np.any(transition < 0):
                    raise ValueError("IMM transition must contain finite nonnegative values")
                transition64 = _float64_array(transition, "IMM transition")
                with np.errstate(over="ignore", invalid="ignore"):
                    transition_totals = transition64.sum(axis=1)
                if not np.allclose(
                    transition_totals,
                    1.0,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError("IMM transition rows must sum to 1")
                _as_probability_array(track.filt._cbar, 2, "IMM predicted probabilities")

            _, covariance = _validated_filter_state(track.filt)
            if type(track.hits) is not deque or track.hits.maxlen != self.cfg.confirm_window:
                raise ValueError(f"tracks[{index}] lifecycle window was corrupted")
            if any(type(hit) is not int or hit not in (0, 1) for hit in track.hits):
                raise ValueError(f"tracks[{index}] lifecycle history must contain only 0/1")
            age = _nonnegative_int(track.age, f"tracks[{index}].age")
            if age == 0 or len(track.hits) != min(age, self.cfg.confirm_window):
                raise ValueError(f"tracks[{index}] lifecycle age/history invariant was corrupted")
            consecutive_misses = _nonnegative_int(
                track.consecutive_misses,
                f"tracks[{index}].consecutive_misses",
            )
            trailing_misses = 0
            for hit in reversed(track.hits):
                if hit:
                    break
                trailing_misses += 1
            if consecutive_misses != trailing_misses:
                raise ValueError(f"tracks[{index}] consecutive-miss invariant was corrupted")
            if not isinstance(track.ever_confirmed, (bool, np.bool_)):
                raise ValueError(f"tracks[{index}].ever_confirmed must be boolean")
            if not isinstance(track.updated_this_cycle, (bool, np.bool_)):
                raise ValueError(f"tracks[{index}].updated_this_cycle must be boolean")
            if bool(track.updated_this_cycle) != bool(track.hits[-1]):
                raise ValueError(f"tracks[{index}] update/history invariant was corrupted")
            if type(track.state) is not str or track.state not in _VALID_TRACK_STATES:
                raise ValueError(f"tracks[{index}] has an invalid lifecycle state")
            normalized_label = _bounded_optional_text(
                track.class_label,
                f"tracks[{index}].class_label",
                maximum_bytes=MAX_CLASS_LABEL_BYTES,
            )
            if normalized_label != track.class_label:
                raise ValueError(f"tracks[{index}] class label must remain normalized")
            state_timestamp = _optional_timestamp(
                track.state_timestamp,
                f"tracks[{index}].state_timestamp",
            )
            measurement_timestamp = _optional_timestamp(
                track.last_measurement_timestamp,
                f"tracks[{index}].last_measurement_timestamp",
            )
            if (
                state_timestamp is None
                or measurement_timestamp is None
                or measurement_timestamp > state_timestamp
                or (last_t is not None and abs(state_timestamp - last_t) > TIMESTAMP_ATOL)
                or (
                    bool(track.updated_this_cycle)
                    and abs(measurement_timestamp - state_timestamp) > TIMESTAMP_ATOL
                )
            ):
                raise ValueError(f"tracks[{index}] timestamp invariant was corrupted")

            n_hits = sum(track.hits)
            misses_in_window = len(track.hits) - n_hits
            determinant_sign, log_cov_volume = np.linalg.slogdet(covariance[:POS_DIM, :POS_DIM])
            if (
                not np.isfinite(determinant_sign)
                or determinant_sign < 0
                or np.isnan(log_cov_volume)
            ):
                raise ValueError(f"tracks[{index}] covariance determinant is invalid")
            covariance_limit_exceeded = determinant_sign > 0 and log_cov_volume > math.log(
                self.cfg.max_position_cov_volume
            )
            if n_hits >= self.cfg.confirm_hits and not bool(track.ever_confirmed):
                raise ValueError(f"tracks[{index}] confirmation history was corrupted")
            if bool(track.ever_confirmed) and age < self.cfg.confirm_hits:
                raise ValueError(f"tracks[{index}] confirmation age was corrupted")
            if misses_in_window >= self.cfg.max_missed_in_window or covariance_limit_exceeded:
                expected_state = "lost"
            elif consecutive_misses >= self.cfg.coast_after_misses:
                expected_state = "coasting"
            elif bool(track.ever_confirmed):
                expected_state = "confirmed"
            else:
                expected_state = "tentative"
            if track.state != expected_state or track.state == "lost":
                raise ValueError(f"tracks[{index}] lifecycle state is inconsistent")

        if track_ids and next_id <= max(track_ids):
            raise ValueError("next track id must exceed every committed track id")

    # -- filter factory --------------------------------------------------
    def _make_filter(self, x0: np.ndarray, P0: np.ndarray):
        f = self.cfg.filter
        if f == "imm":
            return IMMEstimator.default_cv_bank(x0, P0, sigma_a=self.cfg.sigma_a)
        if f == "particle":
            # A filter owns its stochastic stream.  Drawing only the child seed
            # from the tracker stream prevents a later, unrelated track birth
            # from changing existing tracks' future process-noise samples.
            seed_words = self.rng.integers(0, 1 << 32, size=4, dtype=np.uint32)
            filter_rng = np.random.default_rng(seed_words)
            return ParticleFilter(
                x0,
                P0,
                sigma_a=self.cfg.sigma_a,
                n_particles=self.cfg.n_particles,
                rng=filter_rng,
            )
        cls = _TRACKER_FILTER_TYPES[f]
        return cls(x0, P0, sigma_a=self.cfg.sigma_a)

    def _spawn(
        self,
        m: Measurement,
        pos_xyz: np.ndarray,
        cov_xyz: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        if len(self.tracks) >= self.cfg.max_tracks:
            return
        x0 = np.zeros(2 * POS_DIM)
        x0[:POS_DIM] = pos_xyz
        initial_velocity = m.velocity if velocity is None else velocity
        if initial_velocity is not None:
            x0[POS_DIM : 2 * POS_DIM] = initial_velocity
        P0 = np.zeros((2 * POS_DIM, 2 * POS_DIM))
        P0[:POS_DIM, :POS_DIM] = self._stabilize_covariance(cov_xyz)
        P0[POS_DIM:, POS_DIM:] = self.cfg.init_vel_var * np.eye(POS_DIM)
        track = Track(self._next_id, self._make_filter(x0, P0), self.cfg, m.class_label)
        track.record(hit=True, timestamp=m.timestamp)
        self.tracks.append(track)
        self._next_id += 1

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Return a numerically positive-definite version of a validated PSD matrix."""
        covariance = _as_covariance(covariance)
        normalizer = max(1.0, float(np.max(np.abs(covariance))))
        with np.errstate(under="ignore"):
            normalized = covariance / normalizer
        values, vectors = np.linalg.eigh(normalized)
        floor = max(
            1e-12 / normalizer,
            100.0 * np.finfo(float).eps,
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            stabilized = (vectors @ np.diag(np.maximum(values, floor)) @ vectors.T) * normalizer
        stabilized = 0.5 * stabilized + 0.5 * stabilized.T
        if not np.isfinite(stabilized).all():
            raise ValueError("stabilized covariance must remain finite")
        return stabilized

    def _cluster_estimate(
        self,
        group: list[int],
        measurements: list[Measurement],
        positions: np.ndarray,
        covariances: np.ndarray,
    ) -> tuple[Measurement, np.ndarray, np.ndarray, np.ndarray | None]:
        """Fuse a new-track cluster using information weighting when covariances are PD."""
        group_positions = positions[group]
        group_covariances = covariances[group]
        identity = np.eye(POS_DIM)
        try:
            precisions = []
            with np.errstate(over="ignore", invalid="ignore"):
                for covariance in group_covariances:
                    # Cholesky distinguishes invertible covariances from valid but
                    # singular PSD inputs, for which information fusion is undefined.
                    np.linalg.cholesky(covariance)
                    precisions.append(np.linalg.solve(covariance, identity))
                total_precision = np.sum(precisions, axis=0)
                weighted_position = sum(
                    (
                        precision @ position
                        for precision, position in zip(
                            precisions,
                            group_positions,
                            strict=True,
                        )
                    ),
                    start=np.zeros(POS_DIM),
                )
                fused_covariance = np.linalg.solve(total_precision, identity)
                fused_position = fused_covariance @ weighted_position
        except np.linalg.LinAlgError:
            # The covariance of an independent arithmetic mean is sum(C_i)/n^2.
            # This preserves actual uncertainty for singular PSD measurements
            # without pretending their pseudo-inverse is ordinary information.
            with np.errstate(over="ignore", invalid="ignore"):
                fused_position = np.mean(group_positions, axis=0)
                fused_covariance = np.sum(group_covariances, axis=0) / len(group) ** 2
        if not np.isfinite(fused_position).all() or not np.isfinite(fused_covariance).all():
            raise FloatingPointError("new-track cluster estimate must remain finite")
        fused_covariance = _as_covariance(fused_covariance, "fused covariance")

        velocities: list[np.ndarray] = []
        for index in group:
            velocity = measurements[index].velocity
            if velocity is not None:
                velocities.append(velocity)
        if velocities:
            with np.errstate(over="ignore", invalid="ignore"):
                fused_velocity = np.mean(velocities, axis=0)
            if not np.isfinite(fused_velocity).all():
                raise FloatingPointError("new-track velocity estimate must remain finite")
        else:
            fused_velocity = None
        representative_index = next(
            (index for index in group if measurements[index].class_label is not None),
            group[0],
        )
        return (
            measurements[representative_index],
            fused_position,
            fused_covariance,
            fused_velocity,
        )

    @staticmethod
    def _labels_compatible(left: str | None, right: str | None) -> bool:
        return left is None or right is None or left == right

    @staticmethod
    def _sensor_key(measurement: Measurement) -> tuple[str, str]:
        """Identify an independent source; legacy measurements group by modality."""
        return measurement.modality, measurement.sensor_id or ""

    def _cluster_unmatched(
        self,
        idxs: list[int],
        measurements: list[Measurement],
        positions: np.ndarray,
    ) -> list[list[int]]:
        """Merge only unambiguous, fully compatible components of unmatched hits.

        A chain such as ``A—B—C`` with ``A`` outside ``C``'s merge radius has
        two equally plausible pairings.  Selecting one greedily makes track
        births depend on producer order or sensor names, so every member of an
        accepted component must be pairwise compatible.  Ambiguous components
        are conservatively returned as singleton births.
        """

        def can_merge(left: int, right: int) -> bool:
            return (
                self._labels_compatible(
                    measurements[left].class_label,
                    measurements[right].class_label,
                )
                and self._sensor_key(measurements[left]) != self._sensor_key(measurements[right])
                and math.dist(positions[left], positions[right]) <= self.cfg.init_merge_dist
            )

        def order_key(index: int) -> tuple:
            measurement = measurements[index]
            velocity = () if measurement.velocity is None else tuple(measurement.velocity)
            return (
                tuple(positions[index]),
                tuple(measurement.covariance.ravel()),
                velocity,
                measurement.modality,
                measurement.class_label or "",
            )

        ordered = sorted(idxs, key=order_key)
        parent = {index: index for index in ordered}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left, right in combinations(ordered, 2):
            if can_merge(left, right):
                union(left, right)

        components: dict[int, list[int]] = {}
        for index in ordered:
            components.setdefault(find(index), []).append(index)

        groups: list[list[int]] = []
        for component in components.values():
            if all(can_merge(left, right) for left, right in combinations(component, 2)):
                groups.append(component)
            else:
                groups.extend([index] for index in component)
        return groups

    def _update_track(
        self, track: Track, m: Measurement, pos_xyz: np.ndarray, cov_xyz: np.ndarray
    ) -> None:
        # Radar stays polar end-to-end for any filter that supports it (EKF, and
        # an IMM bank of EKFs). Other filters fall back to the Cartesian update.
        if m.modality == "radar" and hasattr(track.filt, "update_polar"):
            track.filt.update_polar(m.position, m.covariance, m.sensor_origin)
        else:
            track.filt.update(pos_xyz, cov_xyz)
        if track.class_label is None and m.class_label is not None:
            track.class_label = m.class_label

    def _gating_cost(
        self,
        track: Track,
        position: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        raw_distance = track.gating_distance(position, covariance)
        try:
            distance = _finite_float64_scalar(raw_distance, "gating distance")
        except ValueError as exc:
            raise ValueError(f"track returned invalid gating distance {raw_distance!r}") from exc
        if distance < 0:
            raise ValueError(f"track returned invalid gating distance {distance!r}")
        return distance if distance <= self.cfg.gate_chi2 else GATE_INF

    def _provisional_track_labels(
        self,
        measurements: list[Measurement],
        positions: np.ndarray,
        covariances: np.ndarray,
    ) -> list[str | None]:
        """Choose one concrete class for each unclassified prior, if observable.

        This anchor assignment prevents an unclassified track from accepting
        contradictory concrete classes from different modalities in one cycle.
        It is provisional: the track is labelled only if a final matched update
        actually carries that class.
        """
        labels = [track.class_label for track in self.tracks]
        unknown_tracks = [index for index, label in enumerate(labels) if label is None]
        concrete_measurements = [
            index
            for index, measurement in enumerate(measurements)
            if measurement.class_label is not None
        ]
        if not unknown_tracks or not concrete_measurements:
            return labels

        cost = np.full((len(unknown_tracks), len(concrete_measurements)), GATE_INF)
        for local_track, track_index in enumerate(unknown_tracks):
            for local_measurement, measurement_index in enumerate(concrete_measurements):
                cost[local_track, local_measurement] = self._gating_cost(
                    self.tracks[track_index],
                    positions[measurement_index],
                    covariances[measurement_index],
                )
        for local_track, local_measurement in linear_assignment(cost):
            track_index = unknown_tracks[local_track]
            measurement_index = concrete_measurements[local_measurement]
            labels[track_index] = measurements[measurement_index].class_label
        return labels

    def _associate_modality(
        self,
        group: list[int],
        measurements: list[Measurement],
        positions: np.ndarray,
        covariances: np.ndarray,
        track_labels: list[str | None],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Class-gated global assignment for one modality against a frozen prior."""
        if not self.tracks or not group:
            return [], group.copy()
        cost = np.full((len(self.tracks), len(group)), GATE_INF)
        for track_index, track in enumerate(self.tracks):
            for local_index, measurement_index in enumerate(group):
                if not self._labels_compatible(
                    track_labels[track_index], measurements[measurement_index].class_label
                ):
                    continue
                cost[track_index, local_index] = self._gating_cost(
                    track,
                    positions[measurement_index],
                    covariances[measurement_index],
                )
        pairs = linear_assignment(cost)
        matched_local = {local_index for _, local_index in pairs}
        matches = [(track_index, group[local_index]) for track_index, local_index in pairs]
        unmatched = [
            measurement_index
            for local_index, measurement_index in enumerate(group)
            if local_index not in matched_local
        ]
        return matches, unmatched

    def _materialize_measurements(self, measurements: Iterable[Measurement]) -> list[Measurement]:
        if isinstance(measurements, (str, bytes)):
            raise TypeError("measurements must be an iterable of Measurement instances")
        try:
            iterator = iter(measurements)
        except TypeError as exc:
            raise TypeError("measurements must be an iterable of Measurement instances") from exc
        values = list(islice(iterator, self.cfg.max_measurements + 1))
        if len(values) > self.cfg.max_measurements:
            raise ValueError(
                f"measurement count exceeds configured maximum {self.cfg.max_measurements}"
            )
        copied: list[Measurement] = []
        for index, measurement in enumerate(values):
            if type(measurement) is not Measurement:
                raise TypeError(f"measurements[{index}] must be a Measurement instance")
            copied.append(
                Measurement(
                    modality=measurement.modality,
                    position=measurement.position,
                    covariance=measurement.covariance,
                    timestamp=measurement.timestamp,
                    velocity=measurement.velocity,
                    sensor_origin=measurement.sensor_origin,
                    class_label=measurement.class_label,
                    sensor_id=measurement.sensor_id,
                )
            )
        return copied

    def _snapshot(
        self,
    ) -> tuple[
        TrackerConfig,
        np.random.Generator,
        list[Track],
        int,
        float | None,
        dict,
    ]:
        self._validate_runtime_state()
        _preflight_snapshot_graph(self.tracks)
        canonical_config = _validate_tracker_config_runtime(self.cfg)
        return (
            canonical_config,
            self.rng,
            copy.deepcopy(self.tracks),
            self._next_id,
            self._last_t,
            copy.deepcopy(dict(self.rng.bit_generator.state)),
        )

    def _restore(
        self,
        snapshot: tuple[
            TrackerConfig,
            np.random.Generator,
            list[Track],
            int,
            float | None,
            dict,
        ],
    ) -> None:
        self.cfg, self.rng, tracks, self._next_id, self._last_t, rng_state = snapshot
        self.rng.bit_generator.state = rng_state
        self.tracks = tracks
        for track in self.tracks:
            track.cfg = self.cfg

    @staticmethod
    def _measurement_order_key(index: int, measurement: Measurement) -> tuple:
        """Stable order independent of producer list ordering for non-identical hits."""
        velocity = () if measurement.velocity is None else tuple(measurement.velocity)
        return (
            measurement.timestamp,
            tuple(measurement.position),
            tuple(measurement.covariance.ravel()),
            tuple(measurement.sensor_origin),
            velocity,
            measurement.class_label or "",
            measurement.sensor_id or "",
            index,
        )

    # -- one cycle -------------------------------------------------------
    def step(self, measurements: Iterable[Measurement], timestamp: float) -> list[TrackOutput]:
        timestamp = _finite_number(timestamp, "cycle timestamp")
        if abs(timestamp) > MAX_ABSOLUTE_TIMESTAMP:
            raise ValueError(
                f"cycle timestamp magnitude must not exceed {MAX_ABSOLUTE_TIMESTAMP:g} seconds"
            )
        # Iteration is a callback surface: snapshot before invoking __iter__ or
        # __next__ so callback-side mutations are transactional too.
        snapshot = self._snapshot()
        try:
            materialized = self._materialize_measurements(measurements)
            measurements = materialized
            for measurement in measurements:
                if abs(measurement.timestamp - timestamp) > TIMESTAMP_ATOL:
                    raise ValueError(
                        f"measurement timestamp {measurement.timestamp} differs from cycle "
                        f"timestamp {timestamp} by more than {TIMESTAMP_ATOL} s"
                    )
            if self._last_t is not None and timestamp <= self._last_t:
                raise ValueError(
                    "cycle timestamp must be strictly increasing "
                    "(monotonic without duplicates): "
                    f"{timestamp} <= previous {self._last_t}"
                )
            dt = 0.0 if self._last_t is None else timestamp - self._last_t
            if dt > self.cfg.max_prediction_gap:
                raise ValueError(
                    f"prediction gap {dt} exceeds configured maximum {self.cfg.max_prediction_gap}"
                )
            max_integrated_gap = self.cfg.max_dt * self.cfg.max_substeps
            if dt > max_integrated_gap:
                raise ValueError(
                    f"prediction gap {dt} exceeds the configured gap budget "
                    f"{self.cfg.max_dt} * {self.cfg.max_substeps}"
                )

            # Project and validate every copied measurement before filter/lifecycle
            # state is touched. Polar conversion can expose overflow hidden by
            # finite inputs.
            cart = [_measurement_cartesian_validated(m) for m in measurements]
            positions = np.array([c[0] for c in cart]) if cart else np.empty((0, POS_DIM))
            covariances = (
                np.array([c[1] for c in cart]) if cart else np.empty((0, POS_DIM, POS_DIM))
            )
            if not np.isfinite(positions).all() or not np.isfinite(covariances).all():
                raise ValueError("Cartesian measurement projection must remain finite")

            # 1. PREDICT
            if dt:
                # Advance the discrete, event-indexed model once per cycle.
                # Repeating predict(dt / n) changes the current discrete-
                # acceleration covariance (Q is not a semigroup) and repeats an
                # IMM's Markov transition n times.  ``max_dt`` is therefore an
                # admission/safety budget only, never a hidden model parameter.
                for track in self.tracks:
                    track.filt.predict(dt)

            # 2. PLAN ALL ASSOCIATIONS FROM THE SAME PREDICTED PRIOR. A track may
            # accept one independent update per modality, but incompatible classes
            # are forbidden before their numerical gate is evaluated.
            by_source: dict[tuple[str, str], list[int]] = {}
            for index, measurement in enumerate(measurements):
                by_source.setdefault(self._sensor_key(measurement), []).append(index)
            track_labels = self._provisional_track_labels(measurements, positions, covariances)
            planned_updates: list[tuple[int, int]] = []
            unmatched_m: list[int] = []
            for source_key in sorted(by_source):
                group = sorted(
                    by_source[source_key],
                    key=lambda index: self._measurement_order_key(index, measurements[index]),
                )
                matches, unmatched = self._associate_modality(
                    group,
                    measurements,
                    positions,
                    covariances,
                    track_labels,
                )
                planned_updates.extend(matches)
                unmatched_m.extend(unmatched)

            # 3. APPLY the already-frozen plan in deterministic modality/track order.
            planned_updates.sort(
                key=lambda pair: (
                    measurements[pair[1]].modality,
                    measurements[pair[1]].sensor_id or "",
                    pair[0],
                    self._measurement_order_key(pair[1], measurements[pair[1]]),
                )
            )
            matched_tracks: set[int] = set()
            for track_index, measurement_index in planned_updates:
                self._update_track(
                    self.tracks[track_index],
                    measurements[measurement_index],
                    positions[measurement_index],
                    covariances[measurement_index],
                )
                matched_tracks.add(track_index)

            # 4. LIFECYCLE RECORD — one hit/miss opportunity per complete cycle.
            for track_index, track in enumerate(self.tracks):
                track.record(hit=track_index in matched_tracks, timestamp=timestamp)

            # 5. INITIATE — class-compatible simultaneous hits may seed one track.
            for group in self._cluster_unmatched(unmatched_m, measurements, positions):
                rep, initial_position, initial_covariance, initial_velocity = (
                    self._cluster_estimate(group, measurements, positions, covariances)
                )
                self._spawn(rep, initial_position, initial_covariance, initial_velocity)

            # 6. COMMIT lifecycle and timestamp only after the complete cycle succeeds.
            self.tracks = [track for track in self.tracks if track.state != "lost"]
            self._last_t = timestamp
            self._validate_runtime_state()
            return [
                track.output()
                for track in self.tracks
                if track.ever_confirmed and track.state in ("confirmed", "coasting")
            ]
        except BaseException:
            self._restore(snapshot)
            raise

    def all_outputs(self) -> list[TrackOutput]:
        self._validate_runtime_state()
        return [t.output() for t in self.tracks]


__all__ = [
    "Measurement",
    "TrackerConfig",
    "TrackOutput",
    "Track",
    "MultiSensorTracker",
    "measurement_cartesian",
    "radar_polar_to_cartesian",
    "CARTESIAN_MODALITIES",
]
