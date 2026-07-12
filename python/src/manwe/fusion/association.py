"""Measurement-to-track association: Mahalanobis gating + exact assignment.

The assignment solver is a pure-numpy implementation of the shortest augmenting
path form of the Hungarian algorithm.  Association and set metrics therefore have
the same globally optimal semantics whether or not scipy happens to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np

# Chi-square 0.99 quantiles by degrees of freedom (gating thresholds).
CHI2_99 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 6: 16.812}

GATE_INF = float("inf")
MAX_ASSIGNMENT_CELLS = 4_000_000
MAX_ASSIGNMENT_WORK = 100_000_000
MAX_ASSOCIATION_DIMENSION = 64


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
    cost = np.asarray(cost, float)
    if cost.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {cost.shape}")
    if cost.size == 0:
        return []
    rows, columns = cost.shape
    if cost.size > MAX_ASSIGNMENT_CELLS or _assignment_work((rows, columns)) > MAX_ASSIGNMENT_WORK:
        raise ValueError("cost matrix exceeds the bounded assignment-work limit")
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
    try:
        positions = np.asarray(positions, dtype=float)
        covariances = np.asarray(covariances, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions and covariances must be numeric arrays") from exc
    n_t, n_m = len(tracks), len(measurements)
    if n_t * n_m > MAX_ASSIGNMENT_CELLS or _assignment_work((n_t, n_m)) > MAX_ASSIGNMENT_WORK:
        raise ValueError("tracks and measurements exceed the bounded assignment-work limit")
    if positions.ndim != 2 or positions.shape[0] != n_m or positions.shape[1] == 0:
        raise ValueError(f"positions must have shape ({n_m}, D), got {positions.shape}")
    dimension = positions.shape[1]
    if dimension > MAX_ASSOCIATION_DIMENSION:
        raise ValueError(
            f"association dimension exceeds the {MAX_ASSOCIATION_DIMENSION}-coordinate limit"
        )
    if covariances.shape == (n_m, dimension):
        if n_m == 0:
            covariances = np.empty((0, dimension, dimension), dtype=float)
        else:
            covariances = covariances[:, :, None] * np.eye(dimension)[None, :, :]
    if covariances.shape != (n_m, dimension, dimension):
        raise ValueError(
            f"covariances must have shape ({n_m}, {dimension}) or "
            f"({n_m}, {dimension}, {dimension}), got {covariances.shape}"
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(covariances)):
        raise ValueError("positions and covariances must contain only finite values")
    if isinstance(gate_chi2, bool) or not np.isfinite(gate_chi2) or gate_chi2 <= 0:
        raise ValueError("gate_chi2 must be finite and positive")
    for index, covariance in enumerate(covariances):
        if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
            raise ValueError(f"covariances[{index}] must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError(f"covariances[{index}] must be positive semidefinite")
    cost = np.full((n_t, n_m), GATE_INF)
    for i, track in enumerate(tracks):
        if not hasattr(track, "gating_distance") or not callable(track.gating_distance):
            raise TypeError(f"tracks[{i}] must expose a callable gating_distance")
        for j in range(n_m):
            R = covariances[j]
            raw_distance = track.gating_distance(positions[j], R)
            try:
                d2 = float(raw_distance)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"tracks[{i}] returned invalid gating distance {raw_distance!r}"
                ) from exc
            if isinstance(raw_distance, bool) or not np.isfinite(d2) or d2 < 0:
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
    try:
        position_array = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a numeric array") from exc
    if position_array.ndim != 2:
        raise ValueError(f"positions must have shape (N, D), got {position_array.shape}")
    n_t = len(tracks)
    n_m = len(position_array)
    # Validate every boundary even when one side is empty.
    cost = gated_cost_matrix(tracks, range(n_m), position_array, covariances, gate_chi2)
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
