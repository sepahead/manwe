"""Measurement-to-track association: Mahalanobis gating + exact assignment.

The assignment solver is a pure-numpy implementation of the shortest augmenting
path form of the Hungarian algorithm.  Association and set metrics therefore have
the same globally optimal semantics whether or not scipy happens to be installed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np

# Chi-square 0.99 quantiles by degrees of freedom (gating thresholds).
CHI2_99 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 6: 16.812}

GATE_INF = float("inf")
MAX_ASSIGNMENT_CELLS = 4_000_000
MAX_ASSIGNMENT_WORK = 100_000_000
# Products vanish when either association side is empty, so cardinalities need
# an independent bound before unmatched-index lists or per-item validation run.
MAX_ASSOCIATION_ITEMS = 100_000
# Bound both input widening and the diagonal-to-dense covariance expansion.
MAX_ASSOCIATION_ARRAY_CELLS = 4_000_000
# Batched eigendecomposition costs O(N D^3), independently of assignment work.
MAX_COVARIANCE_VALIDATION_WORK = 100_000_000
MAX_ASSOCIATION_DIMENSION = 64
_MAX_EXACT_FLOAT64_INTEGER = 1 << 53
_REAL_NUMERIC_KINDS = frozenset("biuf")
_FLOAT64_DTYPE = np.dtype(np.float64)


def _raw_real_array(value: object, name: str) -> np.ndarray:
    """Admit an array container without invoking element coercion."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in _REAL_NUMERIC_KINDS:
        raise ValueError(f"{name} must be a real numeric array")
    return array


def _validate_exact_float64_integers(array: np.ndarray, name: str) -> None:
    """Ensure integer-to-float64 conversion is injective over accepted values."""
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
    """Convert an already-bounded array without silent finite-value narrowing."""
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


def _float64_scalar(value: object, name: str) -> float:
    """Convert a real scalar without invoking ``__float__`` on object values."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric scalar") from exc
    if array.ndim != 0 or array.dtype.kind not in _REAL_NUMERIC_KINDS:
        raise ValueError(f"{name} must be a real numeric scalar")
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must be a real numeric scalar")
    _validate_exact_float64_integers(array, name)
    result = float(array)
    if array.dtype.kind == "f" and np.isfinite(array) and not math.isfinite(result):
        raise ValueError(f"{name} is outside the finite float64 range")
    if (
        array.dtype.kind == "f"
        and array.dtype.itemsize > _FLOAT64_DTYPE.itemsize
        and np.isfinite(array)
        and np.asarray(result, dtype=array.dtype) != array
    ):
        raise ValueError(f"{name} loses numeric precision when converted to float64")
    return result


def _assignment_work(shape: tuple[int, int]) -> int:
    smaller, larger = sorted(shape)
    return smaller * smaller * larger


def _small_exact_assignment(cost: np.ndarray, admissible: np.ndarray) -> list[tuple[int, int]]:
    """Exact rational fallback for tiny matrices with extreme dynamic range."""
    rows, columns = cost.shape
    if max(rows, columns) > 8:
        raise ValueError("assignment cost dynamic range exceeds reliable float64 resolution")
    maximum = min(rows, columns)
    for cardinality in range(maximum, -1, -1):
        best_pairs: tuple[tuple[int, int], ...] | None = None
        best_cost: Fraction | None = None
        for selected_rows in combinations(range(rows), cardinality):
            for selected_columns in permutations(range(columns), cardinality):
                pairs = tuple(zip(selected_rows, selected_columns))
                if any(not admissible[row, column] for row, column in pairs):
                    continue
                total = sum(
                    (Fraction.from_float(float(cost[row, column])) for row, column in pairs),
                    start=Fraction(0),
                )
                if (
                    best_cost is None
                    or total < best_cost
                    or (total == best_cost and best_pairs is not None and pairs < best_pairs)
                ):
                    best_cost = total
                    best_pairs = pairs
        if best_pairs is not None:
            return list(best_pairs)
    return []


def linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return a min-cost one-to-one assignment as ``(row, col)`` pairs.

    This is an exact rectangular Hungarian assignment. Entries greater than or
    equal to :data:`GATE_INF` (positive infinity) represent forbidden pairs;
    rows/columns with no admissible partner are left unassigned.
    """
    raw_cost = _raw_real_array(cost, "cost")
    if raw_cost.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {raw_cost.shape}")
    if raw_cost.size == 0:
        return []
    rows, columns = raw_cost.shape
    if (
        raw_cost.size > MAX_ASSIGNMENT_CELLS
        or _assignment_work((rows, columns)) > MAX_ASSIGNMENT_WORK
    ):
        raise ValueError("cost matrix exceeds the bounded assignment-work limit")
    _validate_exact_float64_integers(raw_cost, "cost")
    cost = _float64_array(raw_cost, "cost")
    if np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("cost must not contain NaN or negative infinity")

    admissible = np.isfinite(cost) & (cost < GATE_INF)
    if not admissible.any():
        return []
    finite_costs = cost[admissible]
    minimum = float(np.min(finite_costs))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = cost - minimum
    finite_shifted = shifted[admissible]
    if not np.isfinite(finite_shifted).all():
        return _small_exact_assignment(cost, admissible)
    # Subtracting a large negative minimum can merge adjacent, individually
    # exact float64 costs (for example ``2**53 - 1`` and ``2**53``). Once that
    # happens the normalized Hungarian objective is no longer the caller's
    # objective, so use the exact bounded fallback or fail closed for a matrix
    # too large for it.
    if np.unique(finite_costs).size != np.unique(finite_shifted).size:
        return _small_exact_assignment(cost, admissible)
    scale = float(np.max(finite_shifted))
    normalized = shifted if scale == 0.0 else shifted / scale
    finite_normalized = normalized[admissible]
    positive = finite_shifted > 0.0
    lost_distinction = np.unique(finite_shifted).size != np.unique(finite_normalized).size
    below_solver_resolution = bool(
        np.any(positive & (finite_normalized <= np.finfo(float).eps * max(1, min(rows, columns))))
    )
    if lost_distinction or below_solver_resolution:
        return _small_exact_assignment(cost, admissible)
    # Make one forbidden edge costlier than every possible all-admissible
    # assignment. This maximises admissible cardinality before minimising cost,
    # even when legitimate costs happen to be close to the public sentinel.
    penalty = (float(finite_normalized.max()) + 1.0) * (min(cost.shape) + 1)
    work = np.where(admissible, normalized, penalty)
    pairs = _hungarian(work)
    return [(i, j) for i, j in pairs if admissible[i, j]]


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Exact rectangular assignment for a finite, non-empty cost matrix.

    The implementation follows the O(n^2 m) shortest-augmenting-path algorithm
    for ``n <= m``. Rectangular matrices with more rows are transposed and mapped
    back to the caller's coordinates.
    """
    transposed = cost.shape[0] > cost.shape[1]
    work = cost.T if transposed else cost
    n, m = work.shape

    # 1-based arrays match the standard formulation: p[j] is the row currently
    # assigned to column j, and column zero is the augmenting-path sentinel.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = work[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                # Strict comparison gives deterministic lowest-column tie breaks.
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] == 0:
            continue
        row, col = int(p[j] - 1), j - 1
        pairs.append((col, row) if transposed else (row, col))
    return sorted(pairs)


def gated_cost_matrix(
    tracks: list,
    measurements: Sequence[object],
    positions: np.ndarray,
    covariances: np.ndarray,
    gate_chi2: float = CHI2_99[3],
) -> np.ndarray:
    """Build a squared-Mahalanobis cost matrix, ``GATE_INF`` outside the gate.

    ``tracks[i]`` must expose ``gating_distance(z, R)``. ``positions[j]`` /
    ``covariances[j]`` are the Cartesian position and its diagonal covariance for
    measurement ``j``.
    """
    if not isinstance(tracks, list):
        raise TypeError("tracks must be a list")
    if not isinstance(measurements, Sequence):
        raise TypeError("measurements must be a sequence")
    n_t, n_m = len(tracks), len(measurements)
    _validate_association_counts(n_t, n_m)
    gating_methods = _gating_methods(tracks)
    gate_chi2 = _float64_scalar(gate_chi2, "gate_chi2")
    if not math.isfinite(gate_chi2) or gate_chi2 <= 0:
        raise ValueError("gate_chi2 must be finite and positive")
    raw_positions = _raw_real_array(positions, "positions")
    raw_covariances = _raw_real_array(covariances, "covariances")
    positions, covariances = _validated_association_arrays(
        raw_positions,
        raw_covariances,
        n_m,
    )
    return _gated_cost_matrix(gating_methods, positions, covariances, gate_chi2)


def _validate_association_counts(n_t: int, n_m: int) -> None:
    if n_t > MAX_ASSOCIATION_ITEMS:
        raise ValueError(f"tracks exceed the {MAX_ASSOCIATION_ITEMS}-item safety limit")
    if n_m > MAX_ASSOCIATION_ITEMS:
        raise ValueError(f"measurements exceed the {MAX_ASSOCIATION_ITEMS}-item safety limit")
    if n_t * n_m > MAX_ASSIGNMENT_CELLS or _assignment_work((n_t, n_m)) > MAX_ASSIGNMENT_WORK:
        raise ValueError("tracks and measurements exceed the bounded assignment-work limit")


def _gating_methods(tracks: list) -> list:
    methods = []
    for index, track in enumerate(tracks):
        method = getattr(track, "gating_distance", None)
        if not callable(method):
            raise TypeError(f"tracks[{index}] must expose a callable gating_distance")
        methods.append(method)
    return methods


def _validated_association_arrays(
    raw_positions: np.ndarray,
    raw_covariances: np.ndarray,
    n_m: int,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_positions.ndim != 2 or raw_positions.shape[0] != n_m or raw_positions.shape[1] == 0:
        raise ValueError(f"positions must have shape ({n_m}, D), got {raw_positions.shape}")
    dimension = raw_positions.shape[1]
    if dimension > MAX_ASSOCIATION_DIMENSION:
        raise ValueError(
            f"association dimension exceeds the {MAX_ASSOCIATION_DIMENSION}-coordinate limit"
        )
    if raw_positions.size > MAX_ASSOCIATION_ARRAY_CELLS:
        raise ValueError(
            f"positions exceed the {MAX_ASSOCIATION_ARRAY_CELLS}-coordinate safety limit"
        )

    diagonal_shape = (n_m, dimension)
    full_shape = (n_m, dimension, dimension)
    if raw_covariances.shape not in (diagonal_shape, full_shape):
        raise ValueError(
            f"covariances must have shape {diagonal_shape} or {full_shape}, "
            f"got {raw_covariances.shape}"
        )
    dense_covariance_cells = n_m * dimension * dimension
    if (
        raw_covariances.size > MAX_ASSOCIATION_ARRAY_CELLS
        or dense_covariance_cells > MAX_ASSOCIATION_ARRAY_CELLS
    ):
        raise ValueError(
            f"covariances exceed the {MAX_ASSOCIATION_ARRAY_CELLS}-coordinate safety limit"
        )
    if n_m * dimension**3 > MAX_COVARIANCE_VALIDATION_WORK:
        raise ValueError("covariances exceed the bounded validation-work limit")

    _validate_exact_float64_integers(raw_positions, "positions")
    _validate_exact_float64_integers(raw_covariances, "covariances")
    if not np.all(np.isfinite(raw_positions)) or not np.all(np.isfinite(raw_covariances)):
        raise ValueError("positions and covariances must contain only finite values")

    positions = _float64_array(raw_positions, "positions")
    covariance_values = _float64_array(raw_covariances, "covariances")
    if raw_covariances.shape == diagonal_shape:
        covariances = np.zeros(full_shape, dtype=np.float64)
        diagonal = np.arange(dimension)
        covariances[:, diagonal, diagonal] = covariance_values
    else:
        covariances = covariance_values

    for index, covariance in enumerate(covariances):
        if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
            raise ValueError(f"covariances[{index}] must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError(f"covariances[{index}] must be positive semidefinite")
    return positions, covariances


def _gated_cost_matrix(
    gating_methods: list,
    positions: np.ndarray,
    covariances: np.ndarray,
    gate_chi2: float,
) -> np.ndarray:
    n_t, n_m = len(gating_methods), len(positions)
    cost = np.full((n_t, n_m), GATE_INF)
    for i, gating_distance in enumerate(gating_methods):
        for j in range(n_m):
            R = covariances[j]
            raw_distance = gating_distance(positions[j], R)
            try:
                d2 = _float64_scalar(raw_distance, f"tracks[{i}] gating distance")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"tracks[{i}] returned invalid gating distance") from exc
            if not math.isfinite(d2) or d2 < 0:
                raise ValueError(f"tracks[{i}] returned invalid gating distance {d2!r}")
            cost[i, j] = d2 if d2 <= gate_chi2 else GATE_INF
    return cost


def associate_per_measurement(
    tracks: list,
    positions: np.ndarray,
    covariances: np.ndarray,
    gate_chi2: float = CHI2_99[3],
) -> tuple[dict[int, list[int]], list[int], list[int]]:
    """Compatibility wrapper returning an exact one-to-one gated assignment.

    Each value list contains at most one measurement. New code should generally
    use :func:`associate`, whose return type makes the one-to-one invariant clear.
    """
    matches, unmatched_t, unmatched_m = associate(tracks, positions, covariances, gate_chi2)
    n_t = len(tracks)
    assign: dict[int, list[int]] = {i: [] for i in range(n_t)}
    for i, j in matches:
        assign[i].append(j)
    return assign, unmatched_t, unmatched_m


def associate(
    tracks: list,
    positions: np.ndarray,
    covariances: np.ndarray,
    gate_chi2: float = CHI2_99[3],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Gate + assign. Returns ``(matches, unmatched_track_idx, unmatched_meas_idx)``."""
    if not isinstance(tracks, list):
        raise TypeError("tracks must be a list")
    n_t = len(tracks)
    if n_t > MAX_ASSOCIATION_ITEMS:
        raise ValueError(f"tracks exceed the {MAX_ASSOCIATION_ITEMS}-item safety limit")
    gate_chi2 = _float64_scalar(gate_chi2, "gate_chi2")
    if not math.isfinite(gate_chi2) or gate_chi2 <= 0:
        raise ValueError("gate_chi2 must be finite and positive")
    raw_positions = _raw_real_array(positions, "positions")
    raw_covariances = _raw_real_array(covariances, "covariances")
    if raw_positions.ndim != 2:
        raise ValueError(f"positions must have shape (N, D), got {raw_positions.shape}")
    n_m = len(raw_positions)
    _validate_association_counts(n_t, n_m)
    gating_methods = _gating_methods(tracks)
    position_array, covariance_array = _validated_association_arrays(
        raw_positions,
        raw_covariances,
        n_m,
    )
    # Validate every boundary even when one side is empty.
    cost = _gated_cost_matrix(gating_methods, position_array, covariance_array, gate_chi2)
    if n_t == 0 or n_m == 0:
        return [], list(range(n_t)), list(range(n_m))
    matches = linear_assignment(cost)
    matched_t = {i for i, _ in matches}
    matched_m = {j for _, j in matches}
    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_m = [j for j in range(n_m) if j not in matched_m]
    return matches, unmatched_t, unmatched_m


__all__ = [
    "CHI2_99",
    "GATE_INF",
    "linear_assignment",
    "gated_cost_matrix",
    "associate",
    "associate_per_measurement",
]
