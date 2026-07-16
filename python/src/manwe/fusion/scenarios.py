"""Synthetic multi-modal scenarios for exercising and scoring the tracker.

Generates ground-truth target trajectories and the noisy, cluttered, partially-
detected multi-sensor measurement streams a fusion engine has to reconstruct —
the reproducible substrate for comparing KF/EKF/UKF/PF/IMM without field data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from .metrics import _MAX_COORDINATE_MAGNITUDE as _MAX_TRUTH_COORDINATE_MAGNITUDE
from .metrics import _MAX_POINTS as _MAX_SCORE_TARGETS
from .metrics import ospa
from .tracker import (
    _SUPPORTED_INTEGER_SCALAR_TYPES,
    MAX_ABSOLUTE_TIMESTAMP,
    TIMESTAMP_ATOL,
    Measurement,
    MultiSensorTracker,
    TrackOutput,
    _as_finite_vector,
    _finite_number,
    _float64_array,
    _measurement_cartesian_validated,
    _raw_real_array,
)

# Per-modality measurement noise (std). Cartesian entries are metres; radar is
# [range_m, azimuth_rad, elevation_rad].
MODALITY_NOISE = {
    "visual": np.array([2.0, 2.0, 3.0]),
    "thermal": np.array([2.5, 2.5, 3.5]),
    "lidar": np.array([0.5, 0.5, 0.5]),
    "acoustic": np.array([8.0, 8.0, 12.0]),  # poor range, ok bearing
    "radar": np.array([3.0, 0.02, 0.02]),
    "rf": np.array([15.0, 15.0, 20.0]),
}
_EXPECTED_MODALITIES = frozenset({"visual", "thermal", "lidar", "acoustic", "radar", "rf"})

_MAX_SCENARIO_FRAMES = 1_000_000
_MAX_TARGET_FRAMES = 10_000_000
_MAX_FRAME_MEASUREMENTS = 10_000
_MAX_SCENARIO_MEASUREMENTS = 10_000_000
_MAX_SCENARIO_TARGETS = 10_000
_MAX_SEED = (1 << 128) - 1


@dataclass
class Scenario:
    times: np.ndarray  # (T,)
    truth: list[np.ndarray]  # per target, (T, 3); rows are NaN before/after life
    frames: list[list[Measurement]]

    def truth_at(self, k: int) -> np.ndarray:
        """Active ground-truth positions at frame ``k`` (shape ``(n_active, 3)``)."""
        if type(k) not in _SUPPORTED_INTEGER_SCALAR_TYPES:
            raise TypeError("truth frame index must be an integer")
        k = int(k)
        if k < 0:
            raise IndexError("truth frame index must be nonnegative")
        if type(self.truth) is not list:
            raise TypeError("scenario truth must be a list")
        if len(self.truth) > _MAX_SCENARIO_TARGETS:
            raise ValueError("scenario truth exceeds the bounded target limit")
        total_target_frames = 0
        points: list[np.ndarray] = []
        for index, trajectory in enumerate(self.truth):
            if type(trajectory) in (list, tuple):
                if len(trajectory) > _MAX_TARGET_FRAMES - total_target_frames:
                    raise ValueError("scenario truth exceeds the bounded target-frame limit")
                raw = _raw_real_array(
                    trajectory,
                    f"truth trajectory {index}",
                    allowed_shapes=((len(trajectory), 3),),
                )
            else:
                raw = _raw_real_array(trajectory, f"truth trajectory {index}")
            if raw.ndim != 2 or raw.shape[1:] != (3,):
                raise ValueError(
                    f"truth trajectory {index} must have shape (T, 3), got {raw.shape}"
                )
            total_target_frames += raw.shape[0]
            if total_target_frames > _MAX_TARGET_FRAMES:
                raise ValueError("scenario truth exceeds the bounded target-frame limit")
            if k >= raw.shape[0]:
                raise IndexError(f"truth frame index {k} is out of range")
            row = raw[k]
            inactive = np.isnan(row)
            if np.any(np.isinf(row)) or (np.any(inactive) and not np.all(inactive)):
                raise ValueError("truth rows must be either fully finite or fully NaN")
            if not np.all(inactive):
                points.append(_float64_array(row, f"truth trajectory {index} row").copy())
        return np.stack(points) if points else np.empty((0, 3))


def _validated_times(value: object) -> np.ndarray:
    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        if len(sequence) > _MAX_SCENARIO_FRAMES:
            raise ValueError("scenario exceeds the bounded frame limit")
    raw = _raw_real_array(value, "scenario times")
    if raw.ndim != 1:
        raise ValueError("scenario times must be a finite one-dimensional array")
    if raw.size > _MAX_SCENARIO_FRAMES:
        raise ValueError("scenario exceeds the bounded frame limit")
    if not np.isfinite(raw).all():
        raise ValueError("scenario times must be a finite one-dimensional array")
    times = _float64_array(raw, "scenario times").copy()
    if np.any(np.abs(times) > MAX_ABSOLUTE_TIMESTAMP):
        raise ValueError(
            f"scenario time magnitude must not exceed {MAX_ABSOLUTE_TIMESTAMP:g} seconds"
        )
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("scenario times must be strictly increasing")
    return times


def _validated_modality_noise() -> dict[str, np.ndarray]:
    if (
        type(MODALITY_NOISE) is not dict
        or len(MODALITY_NOISE) != len(_EXPECTED_MODALITIES)
        or set(MODALITY_NOISE) != _EXPECTED_MODALITIES
    ):
        raise ValueError("MODALITY_NOISE must retain the supported modality keys")
    validated: dict[str, np.ndarray] = {}
    for modality in sorted(_EXPECTED_MODALITIES):
        noise = _as_finite_vector(
            MODALITY_NOISE[modality],
            f"MODALITY_NOISE[{modality!r}]",
        )
        if np.any(noise < 0):
            raise ValueError("modality noise standard deviations must be nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            variances = np.square(noise)
        if not np.isfinite(variances).all():
            raise ValueError("modality noise variances must remain finite")
        validated[modality] = noise
    return validated


def _validated_truth(
    truth: object,
    frame_count: int,
) -> list[np.ndarray]:
    if type(truth) is not list:
        raise TypeError("scenario truth must be a list")
    if len(truth) > _MAX_SCENARIO_TARGETS:
        raise ValueError("scenario truth exceeds the bounded target limit")
    if len(truth) * frame_count > _MAX_TARGET_FRAMES:
        raise ValueError("scenario truth exceeds the bounded target-frame limit")
    if not truth:
        return []

    raw_trajectories: list[tuple[int, np.ndarray]] = []
    active_counts = np.zeros(frame_count, dtype=np.uint16)
    for index, trajectory in enumerate(truth):
        expected = (frame_count, 3)
        raw = _raw_real_array(
            trajectory,
            f"truth trajectory {index}",
            allowed_shapes=(expected,),
        )
        if raw.shape != expected:
            raise ValueError(
                f"truth trajectory {index} must have shape {expected}, got {raw.shape}"
            )
        inactive_any = np.isnan(raw).any(axis=1)
        inactive_all = np.isnan(raw).all(axis=1)
        if np.any(np.isinf(raw)) or np.any(inactive_any != inactive_all):
            raise ValueError("truth rows must be either fully finite or fully NaN while inactive")
        active_counts += ~inactive_all
        raw_trajectories.append((index, raw))
    if active_counts.size and int(active_counts.max()) > _MAX_SCORE_TARGETS:
        raise ValueError(f"scenario exceeds the {_MAX_SCORE_TARGETS}-active-target scoring limit")
    validated: list[np.ndarray] = []
    for index, raw in raw_trajectories:
        values = _float64_array(raw, f"truth trajectory {index}").copy()
        if np.any(
            np.logical_or(
                values > _MAX_TRUTH_COORDINATE_MAGNITUDE,
                values < -_MAX_TRUTH_COORDINATE_MAGNITUDE,
            )
        ):
            raise ValueError(
                "truth coordinate magnitude exceeds the float64 metric limit "
                f"{_MAX_TRUTH_COORDINATE_MAGNITUDE:g}"
            )
        validated.append(values)
    return validated


def _truth_at_validated(truth: list[np.ndarray], index: int) -> np.ndarray:
    points = [trajectory[index] for trajectory in truth if not np.isnan(trajectory[index, 0])]
    return np.stack(points) if points else np.empty((0, 3))


def _constant_acceleration_step(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one interval whose sampled acceleration is held constant."""
    position = _as_finite_vector(position, "position")
    velocity = _as_finite_vector(velocity, "velocity")
    acceleration = _as_finite_vector(acceleration, "acceleration")
    dt = _finite_number(dt, "dt", nonnegative=True)
    return _constant_acceleration_step_unchecked(position, velocity, acceleration, dt)


def _constant_acceleration_step_unchecked(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        velocity_delta = acceleration * dt
        next_position = position + velocity * dt + 0.5 * velocity_delta * dt
        next_velocity = velocity + velocity_delta
    if not np.isfinite(next_position).all() or not np.isfinite(next_velocity).all():
        raise FloatingPointError("constant-acceleration step must remain finite")
    return next_position, next_velocity


def make_scenario(
    n_targets: int = 3,
    duration: float = 20.0,
    dt: float = 0.5,
    modalities: tuple[str, ...] = ("visual", "radar", "acoustic"),
    p_detect: float = 0.9,
    clutter_rate: float = 1.0,
    area: float = 200.0,
    sensor_origin: np.ndarray | None = None,
    seed: int = 0,
) -> Scenario:
    """Build a reproducible multi-target, multi-sensor scenario.

    Targets follow constant-velocity motion with light acceleration noise. Each
    active sensor detects each target with probability ``p_detect`` and adds
    ``clutter_rate`` Poisson false alarms per frame (as Cartesian "visual" hits).
    """
    if type(n_targets) is not int or not 0 <= n_targets <= _MAX_SCENARIO_TARGETS:
        raise ValueError(f"n_targets must be an integer in [0, {_MAX_SCENARIO_TARGETS}]")
    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_SEED}]")
    duration = _finite_number(duration, "duration", positive=True)
    dt = _finite_number(dt, "dt", positive=True)
    clutter_rate = _finite_number(clutter_rate, "clutter_rate", nonnegative=True)
    area = _finite_number(area, "area", positive=True)
    if area > 2.0 * _MAX_TRUTH_COORDINATE_MAGNITUDE:
        raise ValueError("area exceeds the float64 scoring-coordinate limit")
    p_detect = _finite_number(p_detect, "p_detect")
    if not 0.0 <= p_detect <= 1.0:
        raise ValueError("p_detect must be finite and in [0, 1]")
    if type(modalities) is not tuple or not modalities:
        raise ValueError("modalities must be a nonempty tuple")
    if len(modalities) > len(_EXPECTED_MODALITIES):
        raise ValueError(
            f"modalities must contain at most {len(_EXPECTED_MODALITIES)} unique entries"
        )
    if any(type(modality) is not str for modality in modalities):
        raise ValueError("modalities must contain only modality names")
    if len(set(modalities)) != len(modalities):
        raise ValueError("modalities must not contain duplicates")
    unknown = [modality for modality in modalities if modality not in _EXPECTED_MODALITIES]
    if unknown:
        raise ValueError(f"unknown modalities {unknown}; expected {sorted(_EXPECTED_MODALITIES)}")
    noise_table = _validated_modality_noise()
    ratio = duration / dt
    if not math.isfinite(ratio):
        raise ValueError("scenario is too large; duration / dt must be finite")
    nearest = round(ratio)
    ratio_tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    last_frame = nearest if abs(ratio - nearest) <= ratio_tolerance else math.floor(ratio)
    frame_count = last_frame + 1
    if frame_count > _MAX_SCENARIO_FRAMES or n_targets * frame_count > _MAX_TARGET_FRAMES:
        raise ValueError("scenario is too large; reduce duration, target count, or time resolution")
    maximum_true_per_frame = n_targets * len(modalities)
    maximum_true_total = maximum_true_per_frame * frame_count
    expected_clutter_total = clutter_rate * frame_count
    if (
        maximum_true_per_frame > _MAX_FRAME_MEASUREMENTS
        or maximum_true_total > _MAX_SCENARIO_MEASUREMENTS
        or not math.isfinite(expected_clutter_total)
        or clutter_rate > _MAX_FRAME_MEASUREMENTS
        or maximum_true_total + expected_clutter_total > _MAX_SCENARIO_MEASUREMENTS
    ):
        raise ValueError("scenario exceeds the bounded measurement-work limit")

    origin: np.ndarray
    if sensor_origin is None:
        origin = np.zeros(3)
    else:
        try:
            origin = _as_finite_vector(sensor_origin, "sensor_origin")
        except ValueError as exc:
            raise ValueError("sensor_origin must be a finite three-vector") from exc
    # Construct by integer frame index. An absolute epsilon in the stop value can
    # create thousands of extra frames when dt is tiny and can bypass the bound
    # calculated above.
    with np.errstate(over="raise", invalid="raise"):
        try:
            times = np.arange(frame_count, dtype=float) * dt
        except FloatingPointError as exc:
            raise ValueError("scenario timestamps exceed the representable float64 range") from exc
    if (
        not np.all(np.isfinite(times))
        or np.any(np.abs(times) > MAX_ABSOLUTE_TIMESTAMP)
        or (len(times) > 1 and np.any(np.diff(times) <= 0))
    ):
        raise ValueError("scenario timestamps must be finite and strictly increasing")
    T = len(times)
    rng = np.random.default_rng(seed)

    # --- ground-truth trajectories -------------------------------------
    truth: list[np.ndarray] = []
    for _ in range(n_targets):
        pos = rng.uniform(-area / 2, area / 2, size=3)
        pos[2] = abs(pos[2]) + 20.0  # keep targets above ground
        vel = rng.uniform(-8.0, 8.0, size=3)
        traj = np.full((T, 3), np.nan)
        # each target is born at frame 0..T//4 and lives to the end (simple)
        birth = int(rng.integers(0, max(T // 4, 1)))
        p = pos.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                for k in range(T):
                    if k < birth:
                        continue
                    if k > birth:
                        accel = rng.normal(0, 0.5, size=3)
                        p, vel = _constant_acceleration_step_unchecked(
                            p,
                            vel,
                            accel,
                            dt,
                        )
                    if not np.all(np.isfinite(p)) or np.any(
                        np.abs(p) > _MAX_TRUTH_COORDINATE_MAGNITUDE
                    ):
                        raise FloatingPointError
                    traj[k] = p
        except FloatingPointError as exc:
            raise ValueError(
                "scenario trajectory exceeds the finite float64 scoring-coordinate range"
            ) from exc
        truth.append(traj)

    # --- measurements ---------------------------------------------------
    frames: list[list[Measurement]] = []
    total_measurements = 0
    for k, t in enumerate(times):
        frame: list[Measurement] = []
        for traj in truth:
            p = traj[k]
            if np.isnan(p[0]):
                continue
            for modality in modalities:
                if rng.random() > p_detect:
                    continue
                frame.append(
                    _sample_measurement(
                        modality,
                        p,
                        origin,
                        float(t),
                        rng,
                        noise_table[modality],
                    )
                )
        # clutter
        n_clutter = int(rng.poisson(clutter_rate))
        if (
            len(frame) + n_clutter > _MAX_FRAME_MEASUREMENTS
            or total_measurements + len(frame) + n_clutter > _MAX_SCENARIO_MEASUREMENTS
        ):
            raise ValueError("sampled scenario exceeds the bounded measurement-work limit")
        for _ in range(n_clutter):
            fp = rng.uniform(-area / 2, area / 2, size=3)
            fp[2] = abs(fp[2]) + 10.0
            noise = noise_table["visual"]
            frame.append(Measurement("visual", fp, noise**2, float(t)))
        total_measurements += len(frame)
        frames.append(frame)

    return Scenario(times=times, truth=truth, frames=frames)


def _sample_measurement(
    modality: str,
    p: np.ndarray,
    origin: np.ndarray,
    t: float,
    rng: np.random.Generator,
    noise: np.ndarray,
) -> Measurement:
    if modality == "radar":
        with np.errstate(over="ignore", invalid="ignore"):
            d = p - origin
        if not np.isfinite(d).all():
            raise ValueError("radar relative position must remain finite")
        rho = np.hypot(d[0], d[1])
        polar = np.array(
            [
                np.hypot.reduce(np.abs(d)),
                np.arctan2(d[1], d[0]),
                np.arctan2(d[2], rho),
            ]
        )
        with np.errstate(over="ignore", invalid="ignore"):
            polar = polar + rng.normal(0, noise)
        if not np.isfinite(polar).all():
            raise ValueError("sampled radar measurement must remain finite")
        polar[0] = abs(polar[0])
        # Keep the spherical representation canonical after adding angular noise.
        polar[2] = (polar[2] + np.pi) % (2.0 * np.pi) - np.pi
        if polar[2] > np.pi / 2:
            polar[2] = np.pi - polar[2]
            polar[1] += np.pi
        elif polar[2] < -np.pi / 2:
            polar[2] = -np.pi - polar[2]
            polar[1] += np.pi
        polar[1] = (polar[1] + np.pi) % (2.0 * np.pi) - np.pi
        return Measurement("radar", polar, noise**2, float(t), sensor_origin=origin)
    with np.errstate(over="ignore", invalid="ignore"):
        meas = p + rng.normal(0, noise)
    if not np.isfinite(meas).all():
        raise ValueError("sampled Cartesian measurement must remain finite")
    return Measurement(modality, meas, noise**2, float(t))


def score_tracker(
    tracker: MultiSensorTracker, scenario: Scenario, c: float = 20.0, p: float = 2.0
) -> dict[str, float]:
    """Run ``tracker`` over ``scenario`` and return mean OSPA (+ components).

    Skips a short warm-up (first 3 frames) so filters can lock on before scoring.
    """
    if type(tracker) is not MultiSensorTracker:
        raise TypeError("tracker must be a MultiSensorTracker")
    if type(scenario) is not Scenario:
        raise TypeError("scenario must be a Scenario")
    c = _finite_number(c, "c", positive=True)
    p = _finite_number(p, "p")
    if p < 1:
        raise ValueError("c must be positive and p must be at least 1")
    # Exercise the complete metric parameter contract before the tracker can be
    # mutated. In particular, this rejects finite-looking ``c, p`` pairs whose
    # power is not representable.
    ospa(np.empty((0, 3)), np.empty((0, 3)), c=c, p=p)
    tracker._validate_runtime_state()
    times = _validated_times(scenario.times)
    warmup = 3
    if type(scenario.frames) is not list:
        raise TypeError("scenario frames must be a list")
    if len(times) != len(scenario.frames):
        raise ValueError("scenario times and frames must have the same length")
    total_measurements = 0
    for frame_index, frame in enumerate(scenario.frames):
        if type(frame) is not list:
            raise ValueError(
                f"scenario frame {frame_index} must be a list of Measurement instances"
            )
        if len(frame) > min(_MAX_FRAME_MEASUREMENTS, tracker.cfg.max_measurements):
            raise ValueError(f"scenario frame {frame_index} exceeds the bounded measurement limit")
        total_measurements += len(frame)
        if total_measurements > _MAX_SCENARIO_MEASUREMENTS:
            raise ValueError("scenario exceeds the bounded measurement-work limit")
        if any(type(measurement) is not Measurement for measurement in frame):
            raise ValueError(
                f"scenario frame {frame_index} must contain only Measurement instances"
            )
    if len(times) <= warmup:
        raise ValueError(
            f"scenario needs more than {warmup} frames so at least one frame is scored"
        )
    validated_truth = _validated_truth(scenario.truth, len(times))

    previous_time = tracker._last_t
    for k, frame in enumerate(scenario.frames):
        timestamp = float(times[k])
        if previous_time is not None:
            gap = timestamp - previous_time
            if gap <= 0:
                raise ValueError(
                    "scenario timestamps must be strictly later than the tracker's current timestamp"
                )
            if gap > tracker.cfg.max_prediction_gap:
                raise ValueError("scenario prediction gap exceeds the tracker configuration")
            if gap > tracker.cfg.max_dt * tracker.cfg.max_substeps:
                raise ValueError("scenario prediction exceeds the tracker gap budget")
        previous_time = timestamp
        for measurement in frame:
            validated_measurement = Measurement(
                modality=measurement.modality,
                position=measurement.position,
                covariance=measurement.covariance,
                timestamp=measurement.timestamp,
                velocity=measurement.velocity,
                sensor_origin=measurement.sensor_origin,
                class_label=measurement.class_label,
                sensor_id=measurement.sensor_id,
            )
            if abs(validated_measurement.timestamp - timestamp) > TIMESTAMP_ATOL:
                raise ValueError(f"scenario frame {k} contains a measurement timestamp mismatch")
            position, covariance = _measurement_cartesian_validated(validated_measurement)
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(covariance)):
                raise ValueError(f"scenario frame {k} has a non-finite Cartesian projection")

    totals: dict[str, list[float]] = {"ospa": [], "localization": [], "cardinality": []}
    tracker_snapshot = tracker._snapshot()
    try:
        for k, (t, frame) in enumerate(zip(times, scenario.frames, strict=True)):
            outputs = tracker.step(frame, float(t))
            if type(outputs) is not list or len(outputs) > tracker.cfg.max_tracks:
                raise TypeError("tracker.step must return a bounded list of TrackOutput")
            if any(type(output) is not TrackOutput for output in outputs):
                raise TypeError("tracker.step must return only TrackOutput instances")
            if k < warmup:
                continue
            est = (
                np.stack(
                    [
                        _as_finite_vector(output.position, "track output position")
                        for output in outputs
                    ]
                )
                if outputs
                else np.empty((0, 3))
            )
            d = ospa(_truth_at_validated(validated_truth, k), est, c=c, p=p)
            for key in totals:
                totals[key].append(d[key])
    except BaseException:
        tracker._restore(tracker_snapshot)
        raise
    return {key: float(np.mean(values)) for key, values in totals.items()}


__all__ = ["Scenario", "make_scenario", "score_tracker", "MODALITY_NOISE"]
