"""Multi-target tracking metrics: OSPA and GOSPA.

These score a set of *estimated* target positions against *ground-truth*
positions for one frame, penalising both localisation error and cardinality
error. They are the standard way to compare fusion filters on the synthetic
scenarios in :mod:`manwe.fusion.scenarios`.

References: Schuhmacher et al. 2008 (OSPA); Rahmathullah et al. 2017 (GOSPA).
"""

from __future__ import annotations

import math

import numpy as np

from .association import linear_assignment

_MAX_COORDINATE_MAGNITUDE = np.sqrt(np.finfo(np.float64).max) / 4.0
_LOG_FLOAT64_MAX = math.log(np.finfo(np.float64).max)
_MAX_POINTS = 4096
_MAX_POINT_DIMENSION = 64
_MAX_POINT_CELLS = 262_144
_MAX_PAIR_CELLS = 4_000_000
_MAX_PAIR_COMPONENTS = 12_000_000
_MAX_ASSIGNMENT_WORK = 50_000_000


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    # ``hypot`` uses scaling internally, unlike sqrt(sum(diff**2)), so valid
    # large coordinate differences do not overflow while forming the norm.
    return np.hypot.reduce(np.abs(diff), axis=2)


def _point_set(value: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric point set") from exc
    if array.shape == (0,):
        return np.empty((0, 0))
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D), got {array.shape}")
    if array.shape[1] > _MAX_POINT_DIMENSION:
        raise ValueError(f"{name} exceeds the {_MAX_POINT_DIMENSION}-coordinate dimension limit")
    if len(array) > _MAX_POINTS:
        raise ValueError(f"{name} exceeds the {_MAX_POINTS}-point safety limit")
    if array.size > _MAX_POINT_CELLS:
        raise ValueError(f"{name} exceeds the {_MAX_POINT_CELLS}-coordinate safety limit")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if np.any(np.abs(array) > _MAX_COORDINATE_MAGNITUDE):
        raise ValueError(
            f"{name} coordinate magnitude exceeds the float64 metric limit "
            f"{_MAX_COORDINATE_MAGNITUDE:g}"
        )
    return array


def _validate_sets(truth: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth_set = _point_set(truth, "truth")
    estimate_set = _point_set(est, "est")
    if truth_set.shape[1] and estimate_set.shape[1] and truth_set.shape[1] != estimate_set.shape[1]:
        raise ValueError("truth and est points must have the same dimensionality")
    if len(truth_set) * len(estimate_set) > _MAX_PAIR_CELLS:
        raise ValueError(f"metric exceeds the {_MAX_PAIR_CELLS}-pair safety limit")
    dimension = max(truth_set.shape[1], estimate_set.shape[1])
    if len(truth_set) * len(estimate_set) * dimension > _MAX_PAIR_COMPONENTS:
        raise ValueError(f"metric exceeds the {_MAX_PAIR_COMPONENTS}-pair-coordinate safety limit")
    return truth_set, estimate_set


def _stable_nonnegative_power(values: np.ndarray, p: float, name: str) -> np.ndarray:
    """Raise bounded nonnegative values to ``p`` without silent rank loss."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        powered = values**p
    positive = values > 0.0
    if np.any(positive & (powered == 0.0)):
        raise ValueError(f"{name} underflows at this metric order")
    if not np.isfinite(powered).all():
        raise ValueError(f"{name} overflows at this metric order")
    if np.unique(values[positive]).size != np.unique(powered[positive]).size:
        raise ValueError(f"{name} loses numeric resolution at this metric order")
    return powered


def _scaled_assignment_power(values: np.ndarray, p: float, scale: float, name: str) -> np.ndarray:
    """Raise costs after a common scale without silently merging positive values."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        normalized = values / scale
    positive = values > 0.0
    if np.any(positive & (normalized == 0.0)):
        raise ValueError(f"{name} underflows while scaling assignment costs")
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} overflows while scaling assignment costs")
    if np.unique(values[positive]).size != np.unique(normalized[positive]).size:
        raise ValueError(f"{name} loses numeric resolution while scaling assignment costs")
    return _stable_nonnegative_power(normalized, p, name)


def _gospa_assignment_costs(
    distances: np.ndarray,
    c: float,
    p: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Represent pair and dummy costs at one common, lossless float64 scale.

    Raw costs preserve subnormal distances when their powers are representable.
    The cutoff and largest admissible distance provide fallbacks for cases where
    raw powers overflow. If no common scale preserves every positive distinction,
    fail closed instead of optimizing a rounded objective.
    """
    positive = distances[distances > 0.0]
    candidates = [1.0, c]
    if positive.size:
        candidates.append(float(np.max(positive)))

    first_error: ValueError | None = None
    attempted: set[float] = set()
    for scale in candidates:
        if scale in attempted or not math.isfinite(scale) or scale <= 0.0:
            continue
        attempted.add(scale)
        try:
            pair_costs = _scaled_assignment_power(distances, p, scale, "GOSPA assignment")
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                dummy_base = c / scale
                dummy_cost = float(dummy_base**p / alpha)
            if not math.isfinite(dummy_cost) or dummy_cost <= 0.0:
                raise ValueError("GOSPA dummy assignment cost is not representable")
        except OverflowError:
            if first_error is None:
                first_error = ValueError("GOSPA dummy assignment cost is not representable")
            continue
        except ValueError as exc:
            if first_error is None:
                first_error = exc
            continue
        return pair_costs, dummy_cost

    if first_error is not None:
        raise first_error
    raise ValueError("GOSPA assignment costs are not representable")


def _metric_parameters(c: float, p: float) -> tuple[float, float]:
    if isinstance(c, bool) or not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")
    if isinstance(p, bool) or not np.isfinite(p) or p < 1:
        raise ValueError("p must be finite and at least 1")
    c, p = float(c), float(p)
    try:
        scale = c**p
    except OverflowError as exc:
        raise ValueError("c**p must be finite") from exc
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("c**p must be finite and representable")
    return c, p


def _stable_p_norm(values: np.ndarray, p: float, name: str) -> float:
    """Compute a finite p-norm without raising values to ``p`` at full scale."""
    if values.size == 0:
        return 0.0
    scale = float(np.max(values))
    if scale == 0.0:
        return 0.0
    # Underflow of a term that is negligible relative to the largest norm term
    # is harmless; assignment costs use the stricter rank-preserving helper.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized_sum = float(np.sum((values / scale) ** p))
    factor = normalized_sum ** (1.0 / p)
    if math.log(scale) + math.log(factor) > _LOG_FLOAT64_MAX:
        raise ValueError(f"{name} exceeds the representable float64 range")
    result = scale * factor
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds the representable float64 range")
    return result


def _scaled_fraction_root(
    scale: float, numerator: int, denominator: float, p: float, name: str
) -> float:
    """Return ``scale * (numerator/denominator)**(1/p)`` in log space."""
    if numerator == 0:
        return 0.0
    log_result = math.log(scale) + (math.log(numerator) - math.log(denominator)) / p
    if log_result > _LOG_FLOAT64_MAX:
        raise ValueError(f"{name} exceeds the representable float64 range")
    result = math.exp(log_result)
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds the representable float64 range")
    return result


def ospa(truth: np.ndarray, est: np.ndarray, c: float = 20.0, p: float = 2.0) -> dict[str, float]:
    """Optimal Sub-Pattern Assignment distance between two point sets.

    Returns the total distance plus its localisation and cardinality components.
    ``c`` is the cutoff (max per-object penalty); ``p`` the order.
    """
    c, p = _metric_parameters(c, p)
    truth, est = _validate_sets(truth, est)
    m, n = len(truth), len(est)
    if m == 0 and n == 0:
        return {"ospa": 0.0, "localization": 0.0, "cardinality": 0.0}
    if m == 0 or n == 0:
        return {"ospa": c, "localization": 0.0, "cardinality": c}

    # Order so that m <= n.
    if m > n:
        truth, est, m, n = est, truth, n, m
    if m * m * n > _MAX_ASSIGNMENT_WORK:
        raise ValueError("OSPA exceeds the assignment-work safety limit")
    D = np.minimum(_pairwise(truth, est), c)
    # Keep representable subnormal distances intact. Infinity, rather than a
    # large finite sentinel, denotes forbidden assignment edges, so raw powered
    # costs cannot collide with the gate representation.
    pairs = linear_assignment(_stable_nonnegative_power(D, p, "OSPA assignment"))
    matched_distances = np.array([D[i, j] for i, j in pairs], dtype=float)
    normalizer = n ** (1.0 / p)
    loc = _stable_p_norm(matched_distances, p, "OSPA localization") / normalizer
    card = c * ((n - m) / n) ** (1.0 / p)
    total = min(_stable_p_norm(np.array([loc, card]), p, "OSPA"), c)
    return {"ospa": float(total), "localization": float(loc), "cardinality": float(card)}


def gospa(
    truth: np.ndarray,
    est: np.ndarray,
    c: float = 20.0,
    p: float = 2.0,
    alpha: float = 2.0,
) -> dict[str, float]:
    """Generalised OSPA (unnormalised).

    With ``alpha=2`` it decomposes into localisation error on assigned pairs
    plus a ``c^p/alpha`` penalty per missed/false target — the variant that
    rewards correct cardinality.  For other ``alpha`` values the returned
    components follow Definition 1 of Rahmathullah et al.: every point in the
    smaller set is assigned through the cut-off metric, and ``cardinality`` is
    the remaining set-size penalty.  The partial-assignment decomposition is
    equivalent to that definition only when ``alpha=2``.
    """
    c, p = _metric_parameters(c, p)
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0.0 < alpha <= 2.0:
        raise ValueError("alpha must be finite and in (0, 2]")
    alpha = float(alpha)
    truth, est = _validate_sets(truth, est)
    m, n = len(truth), len(est)
    if m == 0 and n == 0:
        return {"gospa": 0.0, "localization": 0.0, "cardinality": 0.0}
    if m == 0 or n == 0:
        card = _scaled_fraction_root(c, max(m, n), alpha, p, "GOSPA cardinality")
        return {
            "gospa": card,
            "localization": 0.0,
            "cardinality": card,
        }

    if alpha != 2.0:
        # Definition 1 assigns every member of the smaller set using the
        # cut-off distance.  In particular, equal-cardinality sets farther than
        # ``c`` remain exactly ``c`` apart for every alpha; representing them as
        # two unassigned points would incorrectly introduce alpha dependence.
        if m > n:
            truth, est, m, n = est, truth, n, m
        if m * m * n > _MAX_ASSIGNMENT_WORK:
            raise ValueError("GOSPA exceeds the assignment-work safety limit")
        distances = np.minimum(_pairwise(truth, est), c)
        costs = _stable_nonnegative_power(distances, p, "GOSPA assignment")
        assignment = linear_assignment(costs)
        matched_distances = np.array([distances[i, j] for i, j in assignment], dtype=float)
        loc = _stable_p_norm(matched_distances, p, "GOSPA localization")
        card = _scaled_fraction_root(c, n - m, alpha, p, "GOSPA cardinality")
        total = _stable_p_norm(np.array([loc, card]), p, "GOSPA")
        return {
            "gospa": total,
            "localization": loc,
            "cardinality": card,
        }

    assignment_size = m + n
    if assignment_size**3 > _MAX_ASSIGNMENT_WORK:
        raise ValueError("GOSPA exceeds the assignment-work safety limit")
    D = _pairwise(truth, est)
    positive = D > 0.0
    log_cost = np.full_like(D, -np.inf)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_cost[positive] = math.log(alpha) + p * (np.log(D[positive]) - math.log(c))
    admissible = (D == 0.0) | (log_cost < math.log(2.0))
    admissible_distances = D[admissible]
    pair_costs, dummy_cost = _gospa_assignment_costs(admissible_distances, c, p, alpha)
    represented_pair_cost = np.full_like(D, np.inf)
    represented_pair_cost[admissible] = pair_costs

    # Square augmentation permits every truth/estimate to select its own dummy
    # at the common-scale equivalent of c^p/alpha. The lower-right zero block
    # completes unused dummy assignments. Unlike cardinality-first gating, this
    # minimizes localization plus missed/false penalties jointly.
    augmented = np.full((assignment_size, assignment_size), np.inf)
    augmented[:m, :n] = represented_pair_cost
    augmented[np.arange(m), n + np.arange(m)] = dummy_cost
    augmented[m + np.arange(n), np.arange(n)] = dummy_cost
    augmented[m:, n:] = 0.0
    assignment = linear_assignment(augmented)
    pairs = [(i, j) for i, j in assignment if i < m and j < n and admissible[i, j]]
    matched_distances = np.array([D[i, j] for i, j in pairs], dtype=float)
    loc = _stable_p_norm(matched_distances, p, "GOSPA localization")
    n_assigned = len(pairs)
    card = _scaled_fraction_root(
        c,
        m + n - 2 * n_assigned,
        alpha,
        p,
        "GOSPA cardinality",
    )
    total = _stable_p_norm(np.array([loc, card]), p, "GOSPA")
    return {
        "gospa": total,
        "localization": loc,
        "cardinality": card,
    }


def rmse(truth: np.ndarray, est: np.ndarray) -> float:
    """Position RMSE for equal-length, index-aligned trajectories."""
    truth = _point_set(truth, "truth")
    est = _point_set(est, "est")
    if truth.shape != est.shape:
        raise ValueError(f"truth and est must have equal shapes, got {truth.shape} and {est.shape}")
    if truth.size == 0:
        raise ValueError("truth and est must not be empty")
    differences = np.abs(truth - est)
    point_distances = np.hypot.reduce(differences, axis=-1)
    return _stable_p_norm(np.ravel(point_distances), 2.0, "RMSE") / math.sqrt(point_distances.size)


__all__ = ["ospa", "gospa", "rmse"]
