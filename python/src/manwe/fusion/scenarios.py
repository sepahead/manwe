"""Synthetic multi-modal scenarios for exercising and scoring the tracker.

Generates ground-truth target trajectories and the noisy, cluttered, partially-
detected multi-sensor measurement streams a fusion engine has to reconstruct —
the reproducible substrate for comparing KF/EKF/UKF/PF/IMM without field data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .metrics import ospa
from .tracker import TIMESTAMP_ATOL, Measurement, MultiSensorTracker, measurement_cartesian

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

_MAX_SCENARIO_FRAMES = 1_000_000
_MAX_TARGET_FRAMES = 10_000_000
_MAX_FRAME_MEASUREMENTS = 10_000
_MAX_SCENARIO_MEASUREMENTS = 10_000_000


@dataclass
class Scenario:
    times: np.ndarray  # (T,)
    truth: list[np.ndarray]  # per target, (T, 3); rows are NaN before/after life
    frames: list[list[Measurement]]

    def truth_at(self, k: int) -> np.ndarray:
        """Active ground-truth positions at frame ``k`` (shape ``(n_active, 3)``)."""
        pts = [t[k] for t in self.truth if not np.isnan(t[k, 0])]
        return np.array(pts) if pts else np.empty((0, 3))


def _constant_acceleration_step(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one interval whose sampled acceleration is held constant."""
    next_position = position + velocity * dt + 0.5 * acceleration * dt * dt
    next_velocity = velocity + acceleration * dt
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
    if type(n_targets) is not int or n_targets < 0:
        raise ValueError("n_targets must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, name, allow_zero in (
        (duration, "duration", False),
        (dt, "dt", False),
        (clutter_rate, "clutter_rate", True),
        (area, "area", False),
    ):
        if (
            isinstance(value, bool)
            or not np.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
    if isinstance(p_detect, bool) or not np.isfinite(p_detect) or not 0.0 <= p_detect <= 1.0:
        raise ValueError("p_detect must be finite and in [0, 1]")
    if not isinstance(modalities, tuple) or not modalities:
        raise ValueError("modalities must be a nonempty tuple")
    if len(set(modalities)) != len(modalities):
        raise ValueError("modalities must not contain duplicates")
    unknown = [modality for modality in modalities if modality not in MODALITY_NOISE]
    if unknown:
        raise ValueError(f"unknown modalities {unknown}; expected {sorted(MODALITY_NOISE)}")
    ratio = float(duration) / float(dt)
    if not np.isfinite(ratio):
        raise ValueError("scenario is too large; duration / dt must be finite")
    nearest = round(ratio)
    ratio_tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    last_frame = nearest if abs(ratio - nearest) <= ratio_tolerance else math.floor(ratio)
    frame_count = last_frame + 1
    if frame_count > _MAX_SCENARIO_FRAMES or n_targets * frame_count > _MAX_TARGET_FRAMES:
        raise ValueError("scenario is too large; reduce duration, target count, or time resolution")
    maximum_true_per_frame = n_targets * len(modalities)
    maximum_true_total = maximum_true_per_frame * frame_count
    expected_clutter_total = float(clutter_rate) * frame_count
    if (
        maximum_true_per_frame > _MAX_FRAME_MEASUREMENTS
        or maximum_true_total > _MAX_SCENARIO_MEASUREMENTS
        or not math.isfinite(expected_clutter_total)
        or clutter_rate > _MAX_FRAME_MEASUREMENTS
        or maximum_true_total + expected_clutter_total > _MAX_SCENARIO_MEASUREMENTS
    ):
        raise ValueError("scenario exceeds the bounded measurement-work limit")

    rng = np.random.default_rng(seed)
    origin: np.ndarray
    if sensor_origin is None:
        origin = np.zeros(3)
    else:
        try:
            origin = np.asarray(sensor_origin, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("sensor_origin must be a finite three-vector") from exc
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("sensor_origin must be a finite three-vector")
    # Construct by integer frame index. An absolute epsilon in the stop value can
    # create thousands of extra frames when dt is tiny and can bypass the bound
    # calculated above.
    with np.errstate(over="raise", invalid="raise"):
        try:
            times = np.arange(frame_count, dtype=float) * float(dt)
        except FloatingPointError as exc:
            raise ValueError("scenario timestamps exceed the representable float64 range") from exc
    if not np.all(np.isfinite(times)) or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("scenario timestamps must be finite and strictly increasing")
    T = len(times)

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
                        p, vel = _constant_acceleration_step(p, vel, accel, dt)
                    if not np.all(np.isfinite(p)):
                        raise FloatingPointError
                    traj[k] = p
        except FloatingPointError as exc:
            raise ValueError("scenario trajectory exceeds the representable float64 range") from exc
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
                frame.append(_sample_measurement(modality, p, origin, float(t), rng))
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
            noise = MODALITY_NOISE["visual"]
            frame.append(Measurement("visual", fp, noise**2, float(t)))
        total_measurements += len(frame)
        frames.append(frame)

    return Scenario(times=times, truth=truth, frames=frames)


def _sample_measurement(
    modality: str, p: np.ndarray, origin: np.ndarray, t: float, rng: np.random.Generator
) -> Measurement:
    noise = MODALITY_NOISE[modality]
    if modality == "radar":
        d = p - origin
        rho = np.hypot(d[0], d[1])
        polar = np.array([np.linalg.norm(d), np.arctan2(d[1], d[0]), np.arctan2(d[2], rho)])
        polar = polar + rng.normal(0, noise)
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
    meas = p + rng.normal(0, noise)
    return Measurement(modality, meas, noise**2, float(t))


def score_tracker(
    tracker: MultiSensorTracker, scenario: Scenario, c: float = 20.0, p: float = 2.0
) -> dict[str, float]:
    """Run ``tracker`` over ``scenario`` and return mean OSPA (+ components).

    Skips a short warm-up (first 3 frames) so filters can lock on before scoring.
    """
    if not isinstance(tracker, MultiSensorTracker):
        raise TypeError("tracker must be a MultiSensorTracker")
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if (
        isinstance(c, bool)
        or not np.isfinite(c)
        or c <= 0
        or isinstance(p, bool)
        or not np.isfinite(p)
        or p < 1
    ):
        raise ValueError("c must be positive and p must be at least 1")
    c = float(c)
    p = float(p)
    # Exercise the complete metric parameter contract before the tracker can be
    # mutated. In particular, this rejects finite-looking ``c, p`` pairs whose
    # power is not representable.
    ospa(np.empty((0, 3)), np.empty((0, 3)), c=c, p=p)
    times = np.asarray(scenario.times, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("scenario times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("scenario times must be strictly increasing")
    warmup = 3
    if len(scenario.times) != len(scenario.frames):
        raise ValueError("scenario times and frames must have the same length")
    if len(times) > _MAX_SCENARIO_FRAMES:
        raise ValueError("scenario exceeds the bounded frame limit")
    for index, trajectory in enumerate(scenario.truth):
        values = np.asarray(trajectory, dtype=float)
        if values.shape != (len(times), 3):
            raise ValueError(
                f"truth trajectory {index} must have shape ({len(times)}, 3), got {values.shape}"
            )
        inactive_any = np.isnan(values).any(axis=1)
        inactive_all = np.isnan(values).all(axis=1)
        if np.any(np.isinf(values)) or np.any(inactive_any != inactive_all):
            raise ValueError("truth rows must be either fully finite or fully NaN while inactive")
    if len(scenario.times) <= warmup:
        raise ValueError(
            f"scenario needs more than {warmup} frames so at least one frame is scored"
        )
    total_measurements = 0
    previous_time = tracker._last_t
    for k, frame in enumerate(scenario.frames):
        if not isinstance(frame, list) or any(not isinstance(item, Measurement) for item in frame):
            raise ValueError(f"scenario frame {k} must be a list of Measurement instances")
        if len(frame) > min(_MAX_FRAME_MEASUREMENTS, tracker.cfg.max_measurements):
            raise ValueError(f"scenario frame {k} exceeds the bounded measurement limit")
        total_measurements += len(frame)
        if total_measurements > _MAX_SCENARIO_MEASUREMENTS:
            raise ValueError("scenario exceeds the bounded measurement-work limit")
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
            if abs(measurement.timestamp - timestamp) > TIMESTAMP_ATOL:
                raise ValueError(f"scenario frame {k} contains a measurement timestamp mismatch")
            position, covariance = measurement_cartesian(measurement)
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(covariance)):
                raise ValueError(f"scenario frame {k} has a non-finite Cartesian projection")
        if k >= warmup:
            ospa(scenario.truth_at(k), np.empty((0, 3)), c=c, p=p)

    totals: dict[str, list[float]] = {"ospa": [], "localization": [], "cardinality": []}
    tracker_snapshot = tracker._snapshot()
    try:
        for k, (t, frame) in enumerate(zip(times, scenario.frames, strict=True)):
            outputs = tracker.step(frame, float(t))
            if k < warmup:
                continue
            est = np.array([o.position for o in outputs]) if outputs else np.empty((0, 3))
            d = ospa(scenario.truth_at(k), est, c=c, p=p)
            for key in totals:
                totals[key].append(d[key])
    except BaseException:
        tracker._restore(tracker_snapshot)
        raise
    return {key: float(np.mean(values)) for key, values in totals.items()}


__all__ = ["Scenario", "make_scenario", "score_tracker", "MODALITY_NOISE"]
