"""Bounded cross-camera association with propagated position uncertainty.

The association contract is fail-closed: camera identities, capture timestamps,
undistorted-pixel acknowledgements, uncertainty inputs, workload limits, forward
range, and ray conditioning are checked before a 3D detection is emitted.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from .camera import Camera
from .triangulation import (
    reprojection_error,
    triangulate_dlt,
    triangulate_midpoint,
    triangulation_covariance,
)

_MAX_PIXEL_MAGNITUDE = 1e9
_MAX_POSITION_MAGNITUDE = 1e12
# Below 2**33, adjacent float64 timestamps remain less than one microsecond
# apart and signed int64 microsecond conversion remains representable.
_MAX_TIMESTAMP_MAGNITUDE = float((1 << 33) - 1)
_MAX_CAMERAS = 64
_MAX_DETECTIONS = 100_000
_MAX_CANDIDATE_PAIRS = 10_000_000
_MAX_HYPOTHESES = 1_000_000
_MAX_ASSOCIATION_STATES = 10_000_000
_MAX_TIME_SKEW_S = 60.0
_MAX_SPEED_MPS = 1_000_000.0
_MAX_IDENTITY_BYTES = 256


def _finite_scalar(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be >= 0")
    if maximum is not None and abs(result) > maximum:
        raise ValueError(f"{name} exceeds the supported magnitude {maximum:g}")
    return result


def _bounded_positive_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _finite_vector(value: Any, name: str, size: int, *, maximum: float) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values, got {vector.size}")
    vector = vector.reshape(size).copy()
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.abs(vector) > maximum):
        raise ValueError(f"{name} exceeds the supported magnitude")
    return vector


def _readonly_array(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=float)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _validated_covariance(value: Any, name: str) -> np.ndarray:
    try:
        covariance = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if covariance.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} must contain only finite values")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-10 * scale):
        raise ValueError(f"{name} must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    if float(values[0]) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    if values[0] < 0.0:
        covariance = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
        covariance = 0.5 * (covariance + covariance.T)
    if float(np.trace(covariance)) <= 0.0:
        raise ValueError(f"{name} must contain non-zero uncertainty")
    return _readonly_array(covariance)


def _bounded_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} must not contain control characters")
    result = value.strip()
    try:
        encoded = result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8 text") from exc
    if len(encoded) > _MAX_IDENTITY_BYTES:
        raise ValueError(f"{name} exceeds the {_MAX_IDENTITY_BYTES}-byte limit")
    return result


def _class_label(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return _bounded_identity(value, "class_label")
    except ValueError as exc:
        raise ValueError("class_label must be a bounded non-empty string or None") from exc


@dataclass(frozen=True, slots=True)
class Detection2D:
    """One already-undistorted image-space detection.

    ``pixels_undistorted=True`` is a required producer acknowledgement because
    this pinhole-only module has no distortion coefficients. ``pixel_std_px`` is
    a required one-sigma localization uncertainty. ``timestamp`` is the image
    exposure/capture time (never inference completion time), and
    ``timestamp_std_s`` describes its clock/capture uncertainty.

    The first four fields retain the original positional order. Trust-boundary
    acknowledgements are required keyword-only arguments.
    """

    camera_index: int
    pixel: np.ndarray
    class_label: str | None = None
    confidence: float = 1.0
    timestamp: float | None = None
    camera_id: str | None = None
    pixels_undistorted: bool = field(kw_only=True)
    pixel_std_px: float = field(kw_only=True)
    timestamp_std_s: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        if (
            isinstance(self.camera_index, bool)
            or not isinstance(self.camera_index, (int, np.integer))
            or self.camera_index < 0
            or self.camera_index >= _MAX_CAMERAS
        ):
            raise ValueError(f"camera_index must be an integer in [0, {_MAX_CAMERAS})")
        if self.pixels_undistorted is not True:
            raise ValueError(
                "pixels_undistorted must be explicitly True; distort pixels before this boundary"
            )
        pixel = _finite_vector(self.pixel, "pixel", 2, maximum=_MAX_PIXEL_MAGNITUDE)
        confidence = _finite_scalar(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        timestamp = self.timestamp
        if timestamp is not None:
            timestamp = _finite_scalar(timestamp, "timestamp", maximum=_MAX_TIMESTAMP_MAGNITUDE)
        timestamp_std = _finite_scalar(
            self.timestamp_std_s,
            "timestamp_std_s",
            nonnegative=True,
            maximum=_MAX_TIME_SKEW_S,
        )
        if timestamp is None and timestamp_std != 0.0:
            raise ValueError("timestamp_std_s must be zero when timestamp is absent")
        camera_id = self.camera_id
        if camera_id is not None:
            try:
                camera_id = _bounded_identity(camera_id, "camera_id")
            except ValueError as exc:
                raise ValueError("camera_id must be a bounded non-empty string or None") from exc

        object.__setattr__(self, "camera_index", int(self.camera_index))
        object.__setattr__(self, "pixel", _readonly_array(pixel))
        object.__setattr__(self, "class_label", _class_label(self.class_label))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(
            self,
            "pixel_std_px",
            _finite_scalar(
                self.pixel_std_px,
                "pixel_std_px",
                positive=True,
                maximum=_MAX_PIXEL_MAGNITUDE,
            ),
        )
        object.__setattr__(self, "timestamp_std_s", timestamp_std)


@dataclass(frozen=True, slots=True)
class Detection3D:
    """A triangulated world-space detection at a declared time reference.

    When ``timestamp`` is present, it is the latest contributing capture time and
    the covariance includes motion uncertainty from earlier captures to that
    reference. When absent, the capture time is unspecified and a caller-supplied
    cycle timestamp is required by :func:`to_measurements`; the full configured
    batch skew is already included in ``position_covariance``.
    """

    position: np.ndarray
    class_label: str | None
    confidence: float
    camera_indices: Sequence[int]
    reprojection_error: float
    position_covariance: np.ndarray
    timestamp: float | None = None
    camera_ids: Sequence[str] = field(default_factory=tuple)
    time_uncertainty_s: float = 0.0
    motion_speed_bound_mps: float = 0.0

    def __post_init__(self) -> None:
        position = _finite_vector(self.position, "position", 3, maximum=_MAX_POSITION_MAGNITUDE)
        confidence = _finite_scalar(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.camera_indices, (list, tuple)) or len(self.camera_indices) < 2:
            raise ValueError("camera_indices must contain at least two non-negative integers")
        if len(self.camera_indices) > _MAX_CAMERAS:
            raise ValueError(f"camera_indices must contain at most {_MAX_CAMERAS} entries")
        camera_indices: list[int] = []
        for index in self.camera_indices:
            if (
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                or index < 0
                or index >= _MAX_CAMERAS
            ):
                raise ValueError(f"camera_indices must contain integers in [0, {_MAX_CAMERAS})")
            camera_indices.append(int(index))
        if len(set(camera_indices)) != len(camera_indices):
            raise ValueError("camera_indices must be unique")
        if not isinstance(self.camera_ids, (list, tuple)):
            raise ValueError("camera_ids must be a sequence")
        if len(self.camera_ids) > _MAX_CAMERAS:
            raise ValueError(f"camera_ids must contain at most {_MAX_CAMERAS} entries")
        camera_ids = tuple(self.camera_ids)
        if camera_ids:
            if len(camera_ids) != len(camera_indices):
                raise ValueError("camera_ids must be empty or align with camera_indices")
            try:
                camera_ids = tuple(
                    _bounded_identity(value, f"camera_ids[{index}]")
                    for index, value in enumerate(camera_ids)
                )
            except ValueError as exc:
                raise ValueError("camera_ids must contain bounded non-empty strings") from exc
            if len(set(camera_ids)) != len(camera_ids):
                raise ValueError("camera_ids must be unique")
        timestamp = self.timestamp
        if timestamp is not None:
            timestamp = _finite_scalar(timestamp, "timestamp", maximum=_MAX_TIMESTAMP_MAGNITUDE)

        object.__setattr__(self, "position", _readonly_array(position))
        object.__setattr__(self, "class_label", _class_label(self.class_label))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "camera_indices", tuple(camera_indices))
        object.__setattr__(
            self,
            "reprojection_error",
            _finite_scalar(self.reprojection_error, "reprojection_error", nonnegative=True),
        )
        object.__setattr__(
            self,
            "position_covariance",
            _validated_covariance(self.position_covariance, "position_covariance"),
        )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(
            self,
            "time_uncertainty_s",
            _finite_scalar(
                self.time_uncertainty_s,
                "time_uncertainty_s",
                nonnegative=True,
                maximum=_MAX_TIME_SKEW_S,
            ),
        )
        object.__setattr__(
            self,
            "motion_speed_bound_mps",
            _finite_scalar(
                self.motion_speed_bound_mps,
                "motion_speed_bound_mps",
                nonnegative=True,
                maximum=_MAX_SPEED_MPS,
            ),
        )

    @property
    def timestamp_reference(self) -> str:
        return "latest_capture" if self.timestamp is not None else "external_batch_reference"


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    members: tuple[int, ...]
    mask: int
    association_cost: float
    key: tuple[tuple[Any, ...], ...]
    detection: Detection3D

    @property
    def cardinality(self) -> int:
        return len(self.members)

    @property
    def support(self) -> int:
        return self.cardinality * (self.cardinality - 1) // 2


@dataclass(slots=True)
class _AssociationBudget:
    max_hypotheses: int
    max_states: int
    hypothesis_evaluations: int = 0
    state_transitions: int = 0

    def hypothesis(self) -> None:
        self.hypothesis_evaluations += 1
        if self.hypothesis_evaluations > self.max_hypotheses:
            raise ValueError("multi-camera hypothesis count exceeds max_hypotheses")

    def transition(self, amount: int = 1) -> None:
        self.state_transitions += amount
        if self.state_transitions > self.max_states:
            raise ValueError("multi-camera association exceeds max_association_states")


def _classes_compatible(first: str | None, second: str | None) -> bool:
    return first is None or second is None or first == second


def _resolve_camera_ids(cameras: list[Camera], detections: list[Detection2D]) -> list[str]:
    configured_names: dict[str, int] = {}
    for index, camera in enumerate(cameras):
        if not isinstance(camera, Camera):
            raise TypeError("cameras must contain only Camera instances")
        if camera.name:
            previous = configured_names.setdefault(camera.name, index)
            if previous != index:
                raise ValueError(f"camera name {camera.name!r} is not unique")

    resolved: list[str] = []
    identity_to_index: dict[str, int] = {}
    index_to_identity: dict[int, str] = {}
    for detection in detections:
        if not isinstance(detection, Detection2D):
            raise TypeError("detections must contain only Detection2D instances")
        index = detection.camera_index
        if index >= len(cameras):
            raise ValueError(
                f"camera_index {index} is out of range for {len(cameras)} configured cameras"
            )
        camera = cameras[index]
        configured = camera.name
        if detection.camera_id is not None and configured and detection.camera_id != configured:
            raise ValueError(
                f"camera_id {detection.camera_id!r} does not match camera name {configured!r}"
            )
        if camera.width and not (
            0.0 <= detection.pixel[0] < camera.width and 0.0 <= detection.pixel[1] < camera.height
        ):
            raise ValueError("detection pixel lies outside the configured image dimensions")
        identity = detection.camera_id or configured or f"camera:{index}"
        previous_index = identity_to_index.setdefault(identity, index)
        if previous_index != index:
            raise ValueError(f"camera identity {identity!r} maps to multiple camera indices")
        previous_identity = index_to_identity.setdefault(index, identity)
        if previous_identity != identity:
            raise ValueError(f"camera_index {index} maps to multiple camera identities")
        resolved.append(identity)
    return resolved


def _detection_key(detection: Detection2D, camera_id: str) -> tuple[Any, ...]:
    return (
        camera_id,
        float(detection.pixel[0]),
        float(detection.pixel[1]),
        detection.class_label or "",
        -detection.confidence,
        float("-inf") if detection.timestamp is None else detection.timestamp,
        detection.pixel_std_px,
        detection.timestamp_std_s,
    )


def _select_class(members: list[int], detections: list[Detection2D]) -> str | None:
    votes: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    for index in members:
        detection = detections[index]
        if detection.class_label is None:
            continue
        count, confidence = votes[detection.class_label]
        votes[detection.class_label] = (count + 1, confidence + detection.confidence)
    if not votes:
        return None
    return min(votes, key=lambda label: (-votes[label][0], -votes[label][1], label))


def _cluster_time_reference(
    chosen: list[int],
    detections: list[Detection2D],
    *,
    stamped: bool,
    max_time_skew: float,
) -> tuple[float | None, float]:
    if not stamped:
        return None, max_time_skew
    timestamped: list[tuple[Detection2D, float]] = []
    for index in chosen:
        detection = detections[index]
        if detection.timestamp is None:  # defensive: caller established all-or-none
            raise ValueError("timestamped cluster unexpectedly contains an unstamped detection")
        timestamped.append((detection, detection.timestamp))
    timestamps = [timestamp for _, timestamp in timestamped]
    reference = max(timestamps)
    reference_std = max(
        detection.timestamp_std_s for detection, value in timestamped if value == reference
    )
    uncertainty = max(
        reference - value + reference_std + detection.timestamp_std_s
        for detection, value in timestamped
    )
    return reference, uncertainty


def _build_hypothesis(
    members: tuple[int, ...],
    *,
    cameras: list[Camera],
    detections: list[Detection2D],
    camera_ids: list[str],
    keys: list[tuple[Any, ...]],
    canonical_ranks: dict[int, int],
    edge_gaps: dict[tuple[int, int], float],
    stamped: bool,
    max_reprojection: float,
    max_time_skew: float,
    min_ray_angle_deg: float,
    max_range_m: float,
    max_speed: float,
    max_cameras: int,
) -> _Hypothesis | None:
    """Validate a complete association hypothesis before it can consume detections."""

    cluster_cameras = [cameras[detections[index].camera_index] for index in members]
    pixels = [detections[index].pixel for index in members]
    try:
        point = triangulate_dlt(
            cluster_cameras,
            pixels,
            min_ray_angle_deg=min_ray_angle_deg,
            max_range_m=max_range_m,
            max_cameras=max_cameras,
        )
        error = reprojection_error(cluster_cameras, pixels, point, max_cameras=max_cameras)
        if error > max_reprojection:
            return None
        covariance = triangulation_covariance(
            cluster_cameras,
            pixels,
            [detections[index].pixel_std_px for index in members],
            min_ray_angle_deg=min_ray_angle_deg,
            max_range_m=max_range_m,
            max_cameras=max_cameras,
        )
    except ValueError:
        return None

    timestamp, time_uncertainty = _cluster_time_reference(
        list(members),
        detections,
        stamped=stamped,
        max_time_skew=max_time_skew,
    )
    if time_uncertainty > max_time_skew + 1e-12:
        return None
    motion_std = max_speed * time_uncertainty
    covariance = covariance + np.eye(3) * motion_std**2
    if not np.isfinite(covariance).all():
        return None

    pair_costs = [
        edge_gaps[(min(first, second), max(first, second))]
        for first, second in combinations(members, 2)
    ]
    association_cost = math.fsum(pair_costs)
    if not np.isfinite(association_cost) or association_cost < 0.0:
        return None
    detection = Detection3D(
        position=point,
        class_label=_select_class(list(members), detections),
        confidence=math.fsum(detections[index].confidence for index in members) / len(members),
        camera_indices=[detections[index].camera_index for index in members],
        reprojection_error=error,
        position_covariance=covariance,
        timestamp=timestamp,
        camera_ids=[camera_ids[index] for index in members],
        time_uncertainty_s=time_uncertainty,
        motion_speed_bound_mps=max_speed,
    )
    return _Hypothesis(
        members=members,
        mask=sum(1 << canonical_ranks[index] for index in members),
        association_cost=association_cost,
        key=tuple(keys[index] for index in members),
        detection=detection,
    )


def _enumerate_multi_camera_hypotheses(
    order: list[int],
    edge_gaps: dict[tuple[int, int], float],
    *,
    budget: _AssociationBudget,
    build: Callable[[tuple[int, ...]], _Hypothesis | None],
) -> list[_Hypothesis]:
    """Enumerate every pairwise-consistent clique under explicit work budgets."""

    adjacency: dict[int, set[int]] = {index: set() for index in order}
    for first, second in edge_gaps:
        adjacency[first].add(second)
        adjacency[second].add(first)
    positions = {index: position for position, index in enumerate(order)}
    hypotheses: list[_Hypothesis] = []

    def extend(clique: tuple[int, ...], candidates: tuple[int, ...]) -> None:
        for offset, candidate in enumerate(candidates):
            budget.transition()
            members = (*clique, candidate)
            budget.hypothesis()
            hypothesis = build(members)
            if hypothesis is not None:
                hypotheses.append(hypothesis)

            later = candidates[offset + 1 :]
            budget.transition(len(later))
            next_candidates = tuple(other for other in later if other in adjacency[candidate])
            if next_candidates:
                extend(members, next_candidates)

    for position, first in enumerate(order):
        candidates = tuple(
            second
            for second in order[position + 1 :]
            if second in adjacency[first] and positions[second] > position
        )
        budget.transition(len(order) - position - 1)
        if candidates:
            extend((first,), candidates)
    return hypotheses


def _select_two_camera_hypotheses(
    hypotheses: list[_Hypothesis],
    detections: list[Detection2D],
    camera_ids: list[str],
    keys: list[tuple[Any, ...]],
    budget: _AssociationBudget,
) -> list[_Hypothesis]:
    """Return an exact max-cardinality, then min-ray-gap bipartite assignment."""

    from ..fusion.association import linear_assignment

    identities = sorted(set(camera_ids))
    if len(identities) != 2 or not hypotheses:
        return []
    first_identity, second_identity = identities
    rows = sorted(
        (index for index in range(len(detections)) if camera_ids[index] == first_identity),
        key=lambda index: keys[index],
    )
    columns = sorted(
        (index for index in range(len(detections)) if camera_ids[index] == second_identity),
        key=lambda index: keys[index],
    )
    budget.transition(len(rows) * len(columns))
    row_for = {index: row for row, index in enumerate(rows)}
    column_for = {index: column for column, index in enumerate(columns)}
    cost = np.full((len(rows), len(columns)), np.inf)
    lookup: dict[tuple[int, int], _Hypothesis] = {}
    for hypothesis in hypotheses:
        first, second = hypothesis.members
        if camera_ids[first] == second_identity:
            first, second = second, first
        row, column = row_for[first], column_for[second]
        cost[row, column] = hypothesis.association_cost
        lookup[(row, column)] = hypothesis
    return sorted(
        (lookup[pair] for pair in linear_assignment(cost)),
        key=lambda hypothesis: hypothesis.key,
    )


def _select_multi_camera_hypotheses(
    hypotheses: list[_Hypothesis], budget: _AssociationBudget
) -> list[_Hypothesis]:
    """Solve bounded set packing exactly with deterministic lexicographic objectives.

    The objective maximizes covered detections, then the number of internal
    cross-camera links (favoring a supported N-view fit over arbitrary pair
    fragmentation), then minimizes total pairwise ray-gap cost. Search aborts
    rather than approximating when the caller's state budget is exhausted.
    """

    if not hypotheses:
        return []
    ordered = sorted(hypotheses, key=lambda hypothesis: hypothesis.key)
    all_indices = tuple(range(len(ordered)))
    stack: list[tuple[int, tuple[int, ...], int, int, tuple[int, ...]]] = [
        (0, (), 0, 0, all_indices)
    ]
    best_indices: tuple[int, ...] = ()
    best_cardinality = -1
    best_support = -1
    best_cost = math.inf
    best_key: tuple[tuple[tuple[Any, ...], ...], ...] | None = None

    while stack:
        budget.transition()
        resolved, selected, cardinality, support, available = stack.pop()
        available_mask = 0
        for index in available:
            budget.transition()
            available_mask |= ordered[index].mask
        if not available:
            candidate_indices = tuple(sorted(selected))
            candidate_cost = math.fsum(
                ordered[index].association_cost for index in candidate_indices
            )
            candidate_key = tuple(ordered[index].key for index in candidate_indices)
            if cardinality > best_cardinality or (
                cardinality == best_cardinality
                and (
                    support > best_support
                    or (
                        support == best_support
                        and (
                            candidate_cost < best_cost
                            or (
                                candidate_cost == best_cost
                                and (best_key is None or candidate_key < best_key)
                            )
                        )
                    )
                )
            ):
                best_indices = candidate_indices
                best_cardinality = cardinality
                best_support = support
                best_cost = candidate_cost
                best_key = candidate_key
            continue

        remaining_count = available_mask.bit_count()
        if cardinality + remaining_count < best_cardinality:
            continue
        if (
            cardinality + remaining_count == best_cardinality
            and support + remaining_count * (remaining_count - 1) // 2 < best_support
        ):
            continue

        pivot = available_mask & -available_mask
        containing: list[int] = []
        without_pivot: list[int] = []
        for index in available:
            budget.transition()
            if ordered[index].mask & pivot:
                containing.append(index)
            else:
                without_pivot.append(index)
        # The skip branch is pushed first so the strongest select branch is
        # explored first and establishes useful exact branch-and-bound limits.
        stack.append((resolved | pivot, selected, cardinality, support, tuple(without_pivot)))
        choices = sorted(
            containing,
            key=lambda index: (
                -ordered[index].cardinality,
                -ordered[index].support,
                ordered[index].association_cost,
                ordered[index].key,
            ),
        )
        for index in reversed(choices):
            hypothesis = ordered[index]
            compatible: list[int] = []
            for other in available:
                budget.transition()
                if other != index and not (ordered[other].mask & hypothesis.mask):
                    compatible.append(other)
            stack.append(
                (
                    resolved | hypothesis.mask,
                    (*selected, index),
                    cardinality + hypothesis.cardinality,
                    support + hypothesis.support,
                    tuple(compatible),
                )
            )

    return [ordered[index] for index in best_indices]


def correlate_and_triangulate(
    cameras: Sequence[Camera],
    detections: Sequence[Detection2D],
    max_ray_gap: float = 8.0,
    max_reprojection: float = 12.0,
    max_time_skew: float = 0.05,
    *,
    min_ray_angle_deg: float = 1.0,
    max_range_m: float = 100_000.0,
    max_speed_mps: float | None = None,
    max_cameras: int = 16,
    max_detections: int = 4096,
    max_candidate_pairs: int = 1_000_000,
    max_hypotheses: int = 100_000,
    max_association_states: int = 1_000_000,
) -> list[Detection3D]:
    """Correlate a bounded synchronized batch and emit uncertainty-aware 3D points.

    Timestamped output is referenced to the latest contributing image capture.
    Untimestamped legacy batches use the full configured ``max_time_skew`` as
    temporal uncertainty and leave the output timestamp unset. ``max_speed_mps``
    is mandatory so time skew can be converted into spatial covariance.
    """

    if not isinstance(cameras, (list, tuple)) or not cameras:
        raise ValueError("cameras must be a non-empty sequence")
    if not isinstance(detections, (list, tuple)):
        raise ValueError("detections must be a sequence")
    max_cameras = _bounded_positive_integer(max_cameras, "max_cameras", _MAX_CAMERAS)
    max_detections = _bounded_positive_integer(max_detections, "max_detections", _MAX_DETECTIONS)
    max_candidate_pairs = _bounded_positive_integer(
        max_candidate_pairs, "max_candidate_pairs", _MAX_CANDIDATE_PAIRS
    )
    max_hypotheses = _bounded_positive_integer(max_hypotheses, "max_hypotheses", _MAX_HYPOTHESES)
    max_association_states = _bounded_positive_integer(
        max_association_states,
        "max_association_states",
        _MAX_ASSOCIATION_STATES,
    )
    if len(cameras) < 2:
        raise ValueError("at least two cameras are required")
    if len(cameras) > max_cameras:
        raise ValueError("camera count exceeds max_cameras")
    if len(detections) > max_detections:
        raise ValueError("detection count exceeds max_detections")
    pair_count = len(detections) * (len(detections) - 1) // 2
    if pair_count > max_candidate_pairs:
        raise ValueError("candidate pair count exceeds max_candidate_pairs")
    camera_values = list(cameras)
    detection_values = list(detections)

    max_ray_gap = _finite_scalar(max_ray_gap, "max_ray_gap", positive=True)
    max_reprojection = _finite_scalar(max_reprojection, "max_reprojection", positive=True)
    max_time_skew = _finite_scalar(
        max_time_skew,
        "max_time_skew",
        nonnegative=True,
        maximum=_MAX_TIME_SKEW_S,
    )
    min_ray_angle_deg = _finite_scalar(min_ray_angle_deg, "min_ray_angle_deg", positive=True)
    if min_ray_angle_deg >= 90.0:
        raise ValueError("min_ray_angle_deg must be in the open interval (0, 90)")
    max_range_m = _finite_scalar(max_range_m, "max_range_m", positive=True)
    if max_speed_mps is None:
        raise ValueError("max_speed_mps is required for temporal uncertainty propagation")
    max_speed = _finite_scalar(
        max_speed_mps,
        "max_speed_mps",
        nonnegative=True,
        maximum=_MAX_SPEED_MPS,
    )
    camera_ids = _resolve_camera_ids(camera_values, detection_values)

    stamped_flags = [detection.timestamp is not None for detection in detection_values]
    if any(stamped_flags) and not all(stamped_flags):
        raise ValueError("timestamped and untimestamped detections cannot be mixed")
    stamped = bool(stamped_flags and all(stamped_flags))
    if stamped:
        lower = min(
            float(detection.timestamp) - detection.timestamp_std_s
            for detection in detection_values
            if detection.timestamp is not None
        )
        upper = max(
            float(detection.timestamp) + detection.timestamp_std_s
            for detection in detection_values
            if detection.timestamp is not None
        )
        if upper - lower > max_time_skew:
            raise ValueError("detection capture timestamps and uncertainty exceed max_time_skew")

    count = len(detection_values)
    if count < 2:
        return []

    rays = [
        camera_values[detection.camera_index].backproject_ray(detection.pixel)
        for detection in detection_values
    ]
    keys = [
        _detection_key(detection, camera_ids[index])
        for index, detection in enumerate(detection_values)
    ]
    order = sorted(range(count), key=lambda index: keys[index])
    canonical_ranks = {index: rank for rank, index in enumerate(order)}
    edge_gaps: dict[tuple[int, int], float] = {}
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            if camera_ids[first_index] == camera_ids[second_index]:
                continue
            if not _classes_compatible(
                detection_values[first_index].class_label,
                detection_values[second_index].class_label,
            ):
                continue
            first, second = (
                (first_index, second_index)
                if keys[first_index] <= keys[second_index]
                else (second_index, first_index)
            )
            try:
                _, gap = triangulate_midpoint(
                    rays[first][0],
                    rays[first][1],
                    rays[second][0],
                    rays[second][1],
                    require_forward=True,
                    min_ray_angle_deg=min_ray_angle_deg,
                    max_range_m=max_range_m,
                )
            except ValueError:
                continue
            if gap <= max_ray_gap:
                edge_gaps[(min(first, second), max(first, second))] = gap

    if not edge_gaps:
        return []
    budget = _AssociationBudget(max_hypotheses, max_association_states)

    def build(members: tuple[int, ...]) -> _Hypothesis | None:
        return _build_hypothesis(
            members,
            cameras=camera_values,
            detections=detection_values,
            camera_ids=camera_ids,
            keys=keys,
            canonical_ranks=canonical_ranks,
            edge_gaps=edge_gaps,
            stamped=stamped,
            max_reprojection=max_reprojection,
            max_time_skew=max_time_skew,
            min_ray_angle_deg=min_ray_angle_deg,
            max_range_m=max_range_m,
            max_speed=max_speed,
            max_cameras=max_cameras,
        )

    if len(set(camera_ids)) == 2:
        hypotheses: list[_Hypothesis] = []
        for pair in sorted(
            edge_gaps,
            key=lambda edge: tuple(
                keys[index] for index in sorted(edge, key=lambda value: canonical_ranks[value])
            ),
        ):
            budget.hypothesis()
            members = tuple(sorted(pair, key=lambda index: canonical_ranks[index]))
            hypothesis = build(members)
            if hypothesis is not None:
                hypotheses.append(hypothesis)
        selected = _select_two_camera_hypotheses(
            hypotheses,
            detection_values,
            camera_ids,
            keys,
            budget,
        )
    else:
        hypotheses = _enumerate_multi_camera_hypotheses(
            order,
            edge_gaps,
            budget=budget,
            build=build,
        )
        selected = _select_multi_camera_hypotheses(hypotheses, budget)

    output = [hypothesis.detection for hypothesis in selected]
    return sorted(
        output,
        key=lambda detection: (
            detection.class_label or "",
            tuple(detection.position),
            tuple(detection.camera_ids),
        ),
    )


def to_measurements(detections: Sequence[Detection3D], timestamp: float | None = None) -> list[Any]:
    """Convert uncertainty-aware 3D detections into visual fusion measurements.

    Per-detection covariance is preserved. An explicit cycle timestamp may move
    the reference away from ``latest_capture``; the covariance is then inflated
    by the stored speed bound and absolute time offset. For an untimestamped
    detection, the explicit timestamp is the external batch reference and is
    required.
    """

    from manwe.fusion.tracker import Measurement

    if not isinstance(detections, (list, tuple)):
        raise TypeError("detections must contain only Detection3D instances")
    if len(detections) > _MAX_DETECTIONS:
        raise ValueError(f"detections must contain at most {_MAX_DETECTIONS} entries")
    if any(not isinstance(detection, Detection3D) for detection in detections):
        raise TypeError("detections must contain only Detection3D instances")
    if timestamp is not None:
        timestamp = _finite_scalar(timestamp, "timestamp", maximum=_MAX_TIMESTAMP_MAGNITUDE)
    measurements = []
    for detection in detections:
        measurement_timestamp = timestamp if timestamp is not None else detection.timestamp
        if measurement_timestamp is None:
            raise ValueError("timestamp is required when a detection has no capture timestamp")
        covariance = np.array(detection.position_covariance, copy=True)
        if timestamp is not None and detection.timestamp is not None:
            offset = abs(timestamp - detection.timestamp)
            if offset > _MAX_TIME_SKEW_S:
                raise ValueError("timestamp override exceeds the supported temporal offset")
            motion_std = detection.motion_speed_bound_mps * offset
            covariance += np.eye(3) * motion_std**2
        measurements.append(
            Measurement(
                "visual",
                detection.position,
                covariance,
                measurement_timestamp,
                class_label=detection.class_label,
            )
        )
    return measurements


__all__ = ["Detection2D", "Detection3D", "correlate_and_triangulate", "to_measurements"]
