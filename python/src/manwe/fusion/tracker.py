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
from dataclasses import dataclass, field
from itertools import combinations, islice

import numpy as np

from .association import CHI2_99, GATE_INF, linear_assignment
from .filters import FILTERS, POS_DIM, IMMEstimator, ParticleFilter

Modality = str  # "visual" | "thermal" | "acoustic" | "radar" | "lidar" | "rf"
CARTESIAN_MODALITIES = {"visual", "thermal", "acoustic", "lidar", "rf"}
VALID_MODALITIES = CARTESIAN_MODALITIES | {"radar"}
TIMESTAMP_ATOL = 1e-6
MAX_ABSOLUTE_TIMESTAMP = 4_000_000_000.0
MAX_LIFECYCLE_WINDOW = 4096
MAX_TRACKS = 10_000
MAX_SUBSTEPS = 10_000
MAX_MEASUREMENTS = 10_000
MAX_PARTICLES = 100_000
MAX_ASSIGNMENT_WORK = 100_000_000
MAX_PREDICTION_WORK = 10_000_000
MAX_PARTICLE_POPULATION = 2_000_000
MAX_CLASS_LABEL_BYTES = 256


def _as_finite_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, float)
    if array.shape != (POS_DIM,):
        raise ValueError(f"{name} must have shape ({POS_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_covariance(value, name: str = "covariance") -> np.ndarray:
    array = np.asarray(value, float)
    if array.shape == (POS_DIM,):
        array = np.diag(array)
    elif array.shape != (POS_DIM, POS_DIM):
        raise ValueError(
            f"{name} must have shape ({POS_DIM},) or ({POS_DIM}, {POS_DIM}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diag(array) < 0):
        raise ValueError(f"{name} diagonal variances must be nonnegative")
    scale = max(1.0, float(np.max(np.abs(array))))
    tolerance = 1e-10 * scale
    if not np.allclose(array, array.T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    array = 0.5 * (array + array.T)
    values, vectors = np.linalg.eigh(array)
    if float(values[0]) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if values[0] < 0:
        array = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
        array = 0.5 * (array + array.T)
    return array


def _positive_int(value, name: str, *, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed the supported maximum {maximum}")


def _finite_number(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0")


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
        if not isinstance(self.modality, str) or self.modality not in VALID_MODALITIES:
            raise ValueError(
                f"unknown modality {self.modality!r}; expected one of {sorted(VALID_MODALITIES)}"
            )
        self.position = _as_finite_vector(self.position, "position")
        self.covariance = _as_covariance(self.covariance)
        self.sensor_origin = _as_finite_vector(self.sensor_origin, "sensor_origin")
        _finite_number(self.timestamp, "timestamp")
        self.timestamp = float(self.timestamp)
        if abs(self.timestamp) > MAX_ABSOLUTE_TIMESTAMP:
            raise ValueError(
                f"timestamp magnitude must not exceed {MAX_ABSOLUTE_TIMESTAMP:g} seconds"
            )
        if self.velocity is not None:
            self.velocity = _as_finite_vector(self.velocity, "velocity")
        if self.modality == "radar":
            if self.position[0] < 0:
                raise ValueError("radar range must be >= 0")
            if abs(self.position[1]) > 1_000_000.0:
                raise ValueError("radar azimuth magnitude is too large to canonicalize reliably")
            azimuth = math.remainder(float(self.position[1]), 2.0 * math.pi)
            self.position[1] = -math.pi if azimuth == math.pi else azimuth
            if not -math.pi / 2 <= self.position[2] <= math.pi / 2:
                raise ValueError("radar elevation must be in [-pi/2, pi/2]")
        if self.class_label is not None:
            if not isinstance(self.class_label, str):
                raise ValueError("class_label must be a bounded printable non-empty string or None")
            normalized_label = self.class_label.strip()
            if (
                not normalized_label
                or not normalized_label.isprintable()
                or len(normalized_label.encode("utf-8")) > MAX_CLASS_LABEL_BYTES
            ):
                raise ValueError("class_label must be a bounded printable non-empty string or None")
            self.class_label = normalized_label
        if self.sensor_id is not None:
            if (
                not isinstance(self.sensor_id, str)
                or not self.sensor_id.strip()
                or not self.sensor_id.strip().isprintable()
                or len(self.sensor_id.strip().encode("utf-8")) > 128
            ):
                raise ValueError("sensor_id must be a bounded printable non-empty string or None")
            self.sensor_id = self.sensor_id.strip()


def radar_polar_to_cartesian(pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    r, az, el = pos
    ce = np.cos(el)
    return origin + np.array([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)])


def _radar_polar_jacobian(pos: np.ndarray) -> np.ndarray:
    r, az, el = pos
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    return np.array(
        [
            [ce * ca, -r * ce * sa, -r * se * ca],
            [ce * sa, r * ce * ca, -r * se * sa],
            [se, 0.0, r * ce],
        ]
    )


def measurement_cartesian(m: Measurement) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position_xyz, covariance_3x3)`` in the common world frame.

    Radar is converted polar→Cartesian with a first-order covariance transform so
    it can be *gated* alongside the other modalities; the EKF still updates in
    polar space to model the true angular error.
    """
    if m.modality == "radar":
        p = radar_polar_to_cartesian(m.position, m.sensor_origin)
        J = _radar_polar_jacobian(m.position)
        C = J @ m.covariance @ J.T
        return p, C
    return m.position, m.covariance


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
    max_dt: float = 1.0
    max_prediction_gap: float = 60.0
    max_substeps: int = 120
    max_measurements: int = 1024
    n_particles: int = 512

    def __post_init__(self) -> None:
        valid = {"kalman", "ekf", "ukf", "particle", "imm"}
        if self.filter not in valid:
            raise ValueError(f"unknown filter {self.filter!r}; expected one of {sorted(valid)}")
        _finite_number(self.sigma_a, "sigma_a", nonnegative=True)
        _finite_number(self.gate_chi2, "gate_chi2", positive=True)
        _positive_int(self.confirm_hits, "confirm_hits", maximum=MAX_LIFECYCLE_WINDOW)
        _positive_int(self.confirm_window, "confirm_window", maximum=MAX_LIFECYCLE_WINDOW)
        _positive_int(
            self.coast_after_misses,
            "coast_after_misses",
            maximum=MAX_LIFECYCLE_WINDOW,
        )
        _positive_int(
            self.max_missed_in_window,
            "max_missed_in_window",
            maximum=MAX_LIFECYCLE_WINDOW,
        )
        _positive_int(self.max_tracks, "max_tracks", maximum=MAX_TRACKS)
        _positive_int(self.max_substeps, "max_substeps", maximum=MAX_SUBSTEPS)
        _positive_int(self.max_measurements, "max_measurements", maximum=MAX_MEASUREMENTS)
        _positive_int(self.n_particles, "n_particles", maximum=MAX_PARTICLES)
        _finite_number(self.max_position_cov_volume, "max_position_cov_volume", positive=True)
        _finite_number(self.init_vel_var, "init_vel_var", positive=True)
        _finite_number(self.init_merge_dist, "init_merge_dist", nonnegative=True)
        _finite_number(self.max_dt, "max_dt", positive=True)
        _finite_number(self.max_prediction_gap, "max_prediction_gap", positive=True)
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
        if self.max_tracks * self.max_substeps > MAX_PREDICTION_WORK:
            raise ValueError("max_tracks and max_substeps exceed the prediction-work safety limit")
        if (
            self.filter == "particle"
            and self.max_tracks * self.n_particles > MAX_PARTICLE_POPULATION
        ):
            raise ValueError(
                "max_tracks and n_particles exceed the aggregate particle safety limit"
            )


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
        self.id = track_id
        self.filt = filt
        self.cfg = cfg
        self.class_label = class_label
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
        if timestamp is not None:
            _finite_number(timestamp, "track state timestamp")
            self.state_timestamp = float(timestamp)
        self.updated_this_cycle = hit
        if hit and timestamp is not None:
            self.last_measurement_timestamp = float(timestamp)
        self.hits.append(1 if hit else 0)
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1
        self.age += 1
        self._recompute_state()

    def _recompute_state(self) -> None:
        n_hits = sum(self.hits)
        misses_in_window = len(self.hits) - n_hits
        pos_cov = self.filt.state.P[:POS_DIM, :POS_DIM]
        determinant_sign, log_cov_volume = np.linalg.slogdet(pos_cov)
        if determinant_sign < 0 or np.isnan(log_cov_volume):
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
        st = self.filt.state
        mode_probs = None
        if isinstance(self.filt, IMMEstimator):
            mode_probs = self.filt.mode_probs.tolist()
        return TrackOutput(
            id=self.id,
            position=st.x[:POS_DIM].copy(),
            velocity=st.x[POS_DIM : 2 * POS_DIM].copy(),
            covariance=st.P[:POS_DIM, :POS_DIM].copy(),
            state=self.state,
            age=self.age,
            class_label=self.class_label,
            state_timestamp=self.state_timestamp,
            last_measurement_timestamp=self.last_measurement_timestamp,
            updated_this_cycle=self.updated_this_cycle,
            mode_probs=mode_probs,
        )


class MultiSensorTracker:
    """Recursive multi-target tracker over heterogeneous sensor measurements."""

    def __init__(self, config: TrackerConfig | None = None, rng: np.random.Generator | None = None):
        if config is not None and not isinstance(config, TrackerConfig):
            raise TypeError("config must be a TrackerConfig or None")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None")
        self.cfg = config if config is not None else TrackerConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_t: float | None = None

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
        cls = FILTERS[f]
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
        covariance = 0.5 * (np.asarray(covariance, float) + np.asarray(covariance, float).T)
        values, vectors = np.linalg.eigh(covariance)
        scale = max(1.0, float(np.max(np.abs(values))))
        floor = max(1e-12, 100.0 * np.finfo(float).eps * scale)
        stabilized = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T
        return 0.5 * (stabilized + stabilized.T)

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
            for covariance in group_covariances:
                # Cholesky distinguishes invertible covariances from valid but
                # singular PSD inputs, for which information fusion is undefined.
                np.linalg.cholesky(covariance)
                precisions.append(np.linalg.solve(covariance, identity))
            total_precision = np.sum(precisions, axis=0)
            weighted_position = sum(
                precision @ position for precision, position in zip(precisions, group_positions)
            )
            fused_covariance = np.linalg.solve(total_precision, identity)
            fused_position = fused_covariance @ weighted_position
        except np.linalg.LinAlgError:
            # The covariance of an independent arithmetic mean is sum(C_i)/n^2.
            # This preserves actual uncertainty for singular PSD measurements
            # without pretending their pseudo-inverse is ordinary information.
            fused_position = np.mean(group_positions, axis=0)
            fused_covariance = np.sum(group_covariances, axis=0) / len(group) ** 2

        velocities: list[np.ndarray] = []
        for index in group:
            velocity = measurements[index].velocity
            if velocity is not None:
                velocities.append(velocity)
        fused_velocity = np.mean(velocities, axis=0) if velocities else None
        representative_index = next(
            (index for index in group if measurements[index].class_label is not None),
            group[0],
        )
        return (
            measurements[representative_index],
            fused_position,
            0.5 * (fused_covariance + fused_covariance.T),
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
        if isinstance(raw_distance, (bool, np.bool_)):
            raise ValueError("track returned an invalid boolean gating distance")
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"track returned invalid gating distance {raw_distance!r}") from exc
        if not np.isfinite(distance) or distance < 0:
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
            if not isinstance(measurement, Measurement):
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

    def _snapshot(self) -> tuple[list[Track], int, float | None, dict]:
        return (
            copy.deepcopy(self.tracks),
            self._next_id,
            self._last_t,
            copy.deepcopy(dict(self.rng.bit_generator.state)),
        )

    def _restore(self, snapshot: tuple[list[Track], int, float | None, dict]) -> None:
        tracks, self._next_id, self._last_t, rng_state = snapshot
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
        _finite_number(timestamp, "cycle timestamp")
        timestamp = float(timestamp)
        if abs(timestamp) > MAX_ABSOLUTE_TIMESTAMP:
            raise ValueError(
                f"cycle timestamp magnitude must not exceed {MAX_ABSOLUTE_TIMESTAMP:g} seconds"
            )
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
        max_integrated_gap = float(self.cfg.max_dt) * self.cfg.max_substeps
        if dt > max_integrated_gap:
            raise ValueError(
                f"prediction requires more than {self.cfg.max_substeps} substeps at "
                f"max_dt={self.cfg.max_dt}"
            )
        n_steps = 0 if dt == 0 else max(1, math.ceil(dt / self.cfg.max_dt))
        if n_steps > self.cfg.max_substeps:
            raise ValueError(
                f"prediction requires {n_steps} substeps, exceeding configured maximum "
                f"{self.cfg.max_substeps}"
            )

        # Project and validate every copied measurement before mutable tracker state
        # is touched. Polar conversion can expose overflow hidden by finite inputs.
        cart = [measurement_cartesian(m) for m in measurements]
        positions = np.array([c[0] for c in cart]) if cart else np.empty((0, POS_DIM))
        covariances = np.array([c[1] for c in cart]) if cart else np.empty((0, POS_DIM, POS_DIM))
        if not np.isfinite(positions).all() or not np.isfinite(covariances).all():
            raise ValueError("Cartesian measurement projection must remain finite")

        snapshot = self._snapshot()
        try:
            # 1. PREDICT
            if n_steps:
                substep = dt / n_steps
                for _ in range(n_steps):
                    for track in self.tracks:
                        track.filt.predict(substep)

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
            return [
                track.output()
                for track in self.tracks
                if track.ever_confirmed and track.state in ("confirmed", "coasting")
            ]
        except BaseException:
            self._restore(snapshot)
            raise

    def all_outputs(self) -> list[TrackOutput]:
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
