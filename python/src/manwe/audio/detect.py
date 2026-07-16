"""Acoustic detections and the Cartesian fusion bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .doa import SPEED_OF_SOUND, _signal_matrix, srp_peak_prominence, srp_phat
from .features import sound_pressure_level_db

MAX_CLASS_LABEL_BYTES = 256
MAX_ABSOLUTE_AZIMUTH = 1_000_000.0


def _finite_scalar(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.float64(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a finite float") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be representable as a finite float")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if raw.size != 3:
        raise ValueError(f"{name} must contain three values")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(raw, dtype=float)
    vector = vector.reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _canonical_azimuth(value: Any) -> float:
    azimuth = _finite_scalar(value, "azimuth")
    if abs(azimuth) > MAX_ABSOLUTE_AZIMUTH:
        raise ValueError("azimuth magnitude is too large to canonicalize reliably")
    return float(np.arctan2(np.sin(azimuth), np.cos(azimuth)))


def _rotation(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.eye(3)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sensor_rotation must be a real numeric matrix") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError("sensor_rotation must be a real numeric matrix")
    if raw.shape != (3, 3):
        raise ValueError("sensor_rotation must be a finite 3x3 matrix")
    if not np.isfinite(raw).all():
        raise ValueError("sensor_rotation must be a finite 3x3 matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        rotation = np.asarray(raw, dtype=float)
    if not np.isfinite(rotation).all():
        raise ValueError("sensor_rotation must be a finite 3x3 matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        gram = np.einsum("ik,jk->ij", rotation, rotation)
    if not np.isfinite(gram).all() or not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-7):
        raise ValueError("sensor_rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-7):
        raise ValueError("sensor_rotation must have determinant +1")
    return rotation


def _bounded_class_label(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("class_label must be a bounded printable non-empty string or None")
    normalized = value.strip()
    if (
        not normalized
        or not normalized.isprintable()
        or len(normalized.encode("utf-8")) > MAX_CLASS_LABEL_BYTES
    ):
        raise ValueError("class_label must be a bounded printable non-empty string or None")
    return normalized


@dataclass
class AcousticDetection:
    """A validated dominant acoustic source in the microphone-array frame.

    ``range_observed`` is an explicit fusion-boundary attestation. Direction
    finding from one array does not observe range, even when a nominal range is
    useful for display. Callers may set the flag only after supplying an
    independent range observation.
    """

    azimuth: float
    elevation: float
    range_estimate: float
    spl_db: float
    timestamp: float = 0.0
    confidence: float = 1.0
    class_label: str | None = None
    range_observed: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        self.azimuth = _canonical_azimuth(self.azimuth)
        self.elevation = _finite_scalar(self.elevation, "elevation")
        if not -np.pi / 2.0 <= self.elevation <= np.pi / 2.0:
            raise ValueError("elevation must lie in [-pi/2, pi/2]")
        self.range_estimate = _finite_scalar(
            self.range_estimate, "range_estimate", nonnegative=True
        )
        # dB SPL is relative to 20 µPa and may legitimately be negative for
        # signals below that reference pressure.
        self.spl_db = _finite_scalar(self.spl_db, "spl_db")
        self.timestamp = _finite_scalar(self.timestamp, "timestamp")
        self.confidence = _finite_scalar(self.confidence, "confidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.class_label = _bounded_class_label(self.class_label)
        if type(self.range_observed) is not bool:
            raise ValueError("range_observed must be a boolean")

    def direction(self) -> np.ndarray:
        azimuth = _canonical_azimuth(self.azimuth)
        elevation = _finite_scalar(self.elevation, "elevation")
        if not -np.pi / 2.0 <= elevation <= np.pi / 2.0:
            raise ValueError("elevation must lie in [-pi/2, pi/2]")
        cos_elevation = np.cos(elevation)
        return np.array(
            [
                cos_elevation * np.cos(azimuth),
                cos_elevation * np.sin(azimuth),
                np.sin(elevation),
            ]
        )

    def to_measurement(
        self,
        sensor_origin: np.ndarray | None = None,
        angle_std: float = 0.05,
        range_std: float = 30.0,
        sensor_rotation: np.ndarray | None = None,
    ):
        """Convert to an acoustic fusion measurement in the world frame.

        ``sensor_rotation`` maps array-frame vectors into the world frame. Both
        the position and anisotropic covariance are rotated; omission preserves
        the original identity-orientation behavior. ``range_observed=True`` is
        required because a single array's nominal range is not a Cartesian
        position observation.
        """
        from manwe.fusion.tracker import Measurement

        if type(self.range_observed) is not bool:
            raise ValueError("range_observed must be a boolean")
        if not self.range_observed:
            raise ValueError(
                "Cartesian fusion requires an independently observed range; "
                "a single-array nominal range is not an observation"
            )
        origin = (
            np.zeros(3) if sensor_origin is None else _finite_vector(sensor_origin, "sensor_origin")
        )
        rotation = _rotation(sensor_rotation)
        angle_std = _finite_scalar(angle_std, "angle_std", positive=True)
        range_std = _finite_scalar(range_std, "range_std", positive=True)
        distance = _finite_scalar(self.range_estimate, "range_estimate", nonnegative=True)
        azimuth = _canonical_azimuth(self.azimuth)
        elevation = _finite_scalar(self.elevation, "elevation")
        if not -np.pi / 2.0 <= elevation <= np.pi / 2.0:
            raise ValueError("elevation must lie in [-pi/2, pi/2]")
        _finite_scalar(self.spl_db, "spl_db")
        timestamp = _finite_scalar(self.timestamp, "timestamp")
        confidence = _finite_scalar(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        class_label = _bounded_class_label(self.class_label)
        with np.errstate(over="ignore", invalid="ignore"):
            cos_elevation = np.cos(elevation)
            direction = np.array(
                [
                    cos_elevation * np.cos(azimuth),
                    cos_elevation * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
            local_position = distance * direction
            position = origin + np.einsum("ij,j->i", rotation, local_position)
        if not np.isfinite(position).all():
            raise FloatingPointError("acoustic position is not finite")

        cos_elevation = np.cos(elevation)
        sin_elevation = np.sin(elevation)
        cos_azimuth = np.cos(azimuth)
        sin_azimuth = np.sin(azimuth)
        jacobian_array = np.array(
            [
                [
                    cos_elevation * cos_azimuth,
                    -distance * cos_elevation * sin_azimuth,
                    -distance * sin_elevation * cos_azimuth,
                ],
                [
                    cos_elevation * sin_azimuth,
                    distance * cos_elevation * cos_azimuth,
                    -distance * sin_elevation * sin_azimuth,
                ],
                [sin_elevation, 0.0, distance * cos_elevation],
            ]
        )
        with np.errstate(over="ignore", invalid="ignore"):
            jacobian_world = np.einsum("ij,jk->ik", rotation, jacobian_array)
            scaled_jacobian = jacobian_world * np.array([range_std, angle_std, angle_std])
            covariance = np.einsum("ik,jk->ij", scaled_jacobian, scaled_jacobian)
        if not np.isfinite(jacobian_world).all() or not np.isfinite(covariance).all():
            raise FloatingPointError("acoustic covariance is not finite")
        with np.errstate(under="ignore", invalid="ignore"):
            contribution_power = np.einsum("ik,ik->k", scaled_jacobian, scaled_jacobian)
        observable = np.any(jacobian_world != 0.0, axis=0)
        if np.any(observable & (contribution_power <= 0.0)):
            raise FloatingPointError("acoustic covariance contribution underflowed to zero")
        covariance = 0.5 * covariance + 0.5 * covariance.T
        if not np.any(np.diag(covariance) > 0):
            raise FloatingPointError("acoustic covariance underflowed to zero")
        return Measurement(
            "acoustic",
            position,
            covariance,
            timestamp,
            class_label=class_label,
        )


def detect_from_array(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: float,
    timestamp: float = 0.0,
    nominal_range: float = 100.0,
    class_label: str | None = None,
    az_grid: np.ndarray | None = None,
    el_grid: np.ndarray | None = None,
    min_rms: float = 1e-8,
    min_peak_prominence: float | None = 4.0,
) -> AcousticDetection:
    """Estimate one quality-gated dominant source with SRP-PHAT.

    A single array cannot infer range, so ``nominal_range`` is retained only as
    an unobserved display prior. The result therefore cannot be converted to a
    Cartesian fusion measurement until a caller supplies an independent range
    and explicitly sets ``range_observed=True``. The returned confidence is
    derived from SRP peak prominence rather than defaulting every grid maximum
    to full confidence.
    """
    azimuth, elevation, power = srp_phat(
        signals,
        mic_positions,
        fs,
        az_grid=az_grid,
        el_grid=el_grid,
        min_rms=min_rms,
        min_peak_prominence=min_peak_prominence,
    )
    prominence = srp_peak_prominence(power)
    confidence = float(np.clip(prominence / 10.0, 0.0, 1.0))
    signal_array = _signal_matrix(signals)
    # Channel ordering is arbitrary, and SRP can succeed when the first channel
    # is silent. Use the highest RMS level across valid microphones.
    spl = max(sound_pressure_level_db(channel) for channel in signal_array)
    return AcousticDetection(
        azimuth=azimuth,
        elevation=elevation,
        range_estimate=nominal_range,
        spl_db=spl,
        timestamp=timestamp,
        confidence=confidence,
        class_label=class_label,
        range_observed=False,
    )


__all__ = ["AcousticDetection", "detect_from_array", "SPEED_OF_SOUND"]
