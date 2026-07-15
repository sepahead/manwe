"""Validated microphone-array GCC-PHAT and SRP-PHAT direction finding.

Directions point from the array towards a far-field source. Azimuth is measured
in the xy plane and elevation is signed in ``[-pi/2, pi/2]`` by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at approximately 20 degrees C
MAX_SIGNAL_SAMPLES = 1_000_000
MAX_INTERPOLATED_FFT_POINTS = 16_777_216
MAX_MICROPHONES = 64
MAX_SIGNAL_CELLS = 8_000_000
MAX_SEARCH_CELLS = 100_000
MAX_SRP_FREQUENCY_WORK = 100_000_000


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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real_array(value: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def _signal(value: np.ndarray, name: str) -> np.ndarray:
    signal = _real_array(value, name)
    if signal.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if signal.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if signal.size > MAX_SIGNAL_SAMPLES:
        raise ValueError(f"{name} exceeds the {MAX_SIGNAL_SAMPLES}-sample safety limit")
    if not np.isfinite(signal).all():
        raise ValueError(f"{name} must contain only finite samples")
    return signal


def _rms(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return 0.0
    return peak * float(np.sqrt(np.mean((signal / peak) ** 2)))


def _stable_norm(vector: np.ndarray) -> float:
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        return 0.0
    normalized = vector / scale
    result = scale * float(np.sqrt(np.sum(normalized**2)))
    return result if np.isfinite(result) else float("inf")


def _search_grid(value: np.ndarray, name: str) -> np.ndarray:
    grid = _real_array(value, name)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(grid).all():
        raise ValueError(f"{name} must contain only finite values")
    return grid


def _frequency_bins(n_fft: int, fs: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        bin_spacing = fs / n_fft
    if not np.isfinite(bin_spacing) or bin_spacing <= 0:
        raise ValueError("fs is too small or large for a resolvable FFT frequency grid")
    with np.errstate(over="ignore", invalid="ignore"):
        frequencies = np.arange(n_fft // 2 + 1, dtype=float) * bin_spacing
    if not np.isfinite(frequencies).all():
        raise ValueError("FFT frequency grid is not finite")
    return frequencies


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: float,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, np.ndarray]:
    """Estimate the delay of ``sig`` relative to ``refsig`` in seconds."""
    sig = _signal(sig, "sig")
    refsig = _signal(refsig, "refsig")
    fs = _finite_scalar(fs, "fs", positive=True)
    interp = _positive_int(interp, "interp")
    if max_tau is not None:
        max_tau = _finite_scalar(max_tau, "max_tau", positive=True)
    if _rms(sig) <= 1e-12 or _rms(refsig) <= 1e-12:
        raise ValueError("GCC-PHAT requires non-silent signals")
    sig = sig / np.max(np.abs(sig))
    refsig = refsig / np.max(np.abs(refsig))

    n = sig.size + refsig.size
    interpolated_size = interp * n
    if interpolated_size > min(np.iinfo(np.intp).max, MAX_INTERPOLATED_FFT_POINTS):
        raise ValueError(
            f"interp and signal sizes exceed the {MAX_INTERPOLATED_FFT_POINTS}-point FFT limit"
        )
    interpolated_fs = float(interp) * fs
    if not np.isfinite(interpolated_fs) or interpolated_fs <= 0:
        raise ValueError("interp and fs must produce a finite interpolation rate")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        spectrum = np.fft.rfft(sig, n=n)
        reference_spectrum = np.fft.rfft(refsig, n=n)
        cross_spectrum = spectrum * np.conj(reference_spectrum)
        normalized_cross = cross_spectrum / np.maximum(np.abs(cross_spectrum), np.finfo(float).eps)
        cc = np.fft.irfft(normalized_cross, n=interpolated_size)
    if not np.isfinite(cc).all():
        raise FloatingPointError("GCC-PHAT correlation is not finite")
    max_shift = interpolated_size // 2
    if max_tau is not None:
        if max_tau >= max_shift / interpolated_fs:
            requested_shift = max_shift
        else:
            requested_shift = int(interpolated_fs * max_tau)
        if requested_shift == 0:
            return 0.0, cc[:1].copy()
        max_shift = requested_shift
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    delay = shift / interpolated_fs
    if not np.isfinite(delay):
        raise FloatingPointError("GCC-PHAT delay is not finite")
    return delay, cc


def _direction(azimuth: float, elevation: float) -> np.ndarray:
    cos_elevation = np.cos(elevation)
    return np.array(
        [
            cos_elevation * np.cos(azimuth),
            cos_elevation * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def _validate_search_observability(
    baselines: dict[tuple[int, int], np.ndarray],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
) -> None:
    """Require array steering to distinguish the requested direction space.

    Let ``B`` contain the microphone baselines and let ``V`` be the linear span
    of all differences between requested unit directions. Two candidates
    ``u`` and ``v`` have the same ideal far-field delays exactly when
    ``B @ (u - v) == 0``. Requiring ``B`` to have full column rank on ``V`` is
    therefore a sufficient injectivity condition for the whole requested
    direction space, not merely for the sampled peak that happened to win.
    """
    directions = np.array(
        [
            _direction(float(azimuth), float(elevation))
            for elevation in el_grid
            for azimuth in az_grid
        ]
    )
    differences = directions - directions[0]
    direction_scale = float(np.max(np.abs(differences)))
    if direction_scale == 0.0:
        return
    normalized_differences = differences / direction_scale
    _, singular_values, right_vectors = np.linalg.svd(
        normalized_differences,
        full_matrices=False,
    )
    search_rank = int(np.count_nonzero(singular_values > 1e-8))
    search_basis = right_vectors[:search_rank].T

    geometry = np.stack(tuple(baselines.values()))
    geometry_scale = float(np.max(np.abs(geometry)))
    normalized_geometry = geometry / geometry_scale
    observed_rank = int(np.linalg.matrix_rank(normalized_geometry @ search_basis, tol=1e-8))
    if observed_rank < search_rank:
        raise ValueError(
            "microphone geometry cannot uniquely observe the requested direction search"
        )


def srp_peak_prominence(power: np.ndarray) -> float:
    """Return a robust, dimensionless prominence score for an SRP map."""
    power = _real_array(power, "power")
    if power.size == 0 or not np.isfinite(power).all():
        raise ValueError("power must be a non-empty finite array")
    scale = float(np.max(np.abs(power)))
    if scale == 0:
        return 0.0
    normalized = power / scale
    spread = float(np.std(normalized))
    if spread <= np.finfo(float).eps:
        return 0.0
    prominence = float((np.max(normalized) - np.median(normalized)) / spread)
    if not np.isfinite(prominence):
        raise FloatingPointError("SRP peak prominence is not finite")
    return prominence


def srp_phat(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: float,
    az_grid: np.ndarray | None = None,
    el_grid: np.ndarray | None = None,
    c: float = SPEED_OF_SOUND,
    min_rms: float = 1e-8,
    min_peak_prominence: float | None = 4.0,
) -> tuple[float, float, np.ndarray]:
    """Estimate a dominant source over a configurable signed-elevation grid.

    At least two energetic channels are required. The default prominence gate
    rejects silence and incoherent noise rather than emitting a confident but
    arbitrary grid cell; set ``min_peak_prominence=None`` only when a caller has
    its own quality gate.
    """
    signals = _real_array(signals, "signals")
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n_mics, n_samples)")
    n_mics, n_samples = signals.shape
    if n_mics < 2 or n_samples < 2:
        raise ValueError("signals require at least two microphones and two samples")
    if n_mics > MAX_MICROPHONES or n_samples > MAX_SIGNAL_SAMPLES:
        raise ValueError("signals exceed the microphone or sample safety limit")
    if signals.size > MAX_SIGNAL_CELLS:
        raise ValueError(f"signals exceed the {MAX_SIGNAL_CELLS}-value safety limit")
    if not np.isfinite(signals).all():
        raise ValueError("signals must contain only finite samples")

    microphones = _real_array(mic_positions, "mic_positions")
    if microphones.shape != (n_mics, 3):
        raise ValueError(f"mic_positions must have shape ({n_mics}, 3)")
    if not np.isfinite(microphones).all():
        raise ValueError("mic_positions must contain only finite values")
    fs = _finite_scalar(fs, "fs", positive=True)
    c = _finite_scalar(c, "c", positive=True)
    if fs > np.finfo(float).max / np.pi:
        raise ValueError("fs is outside the numerically representable range")
    min_rms = _finite_scalar(min_rms, "min_rms", nonnegative=True)
    if min_peak_prominence is not None:
        min_peak_prominence = _finite_scalar(
            min_peak_prominence, "min_peak_prominence", nonnegative=True
        )
    channel_peaks = np.max(np.abs(signals), axis=1)
    normalized_signals = np.zeros_like(signals)
    np.divide(
        signals,
        channel_peaks[:, None],
        out=normalized_signals,
        where=channel_peaks[:, None] > 0.0,
    )
    channel_rms = channel_peaks * np.sqrt(np.mean(normalized_signals**2, axis=1))
    if np.count_nonzero(channel_rms > min_rms) < 2:
        raise ValueError("SRP-PHAT requires at least two non-silent microphone channels")

    pairs = [(first, second) for first in range(n_mics) for second in range(first + 1, n_mics)]
    baselines: dict[tuple[int, int], np.ndarray] = {}
    for first, second in pairs:
        with np.errstate(over="ignore", invalid="ignore"):
            baseline = microphones[first] - microphones[second]
        if not np.isfinite(baseline).all():
            raise ValueError("microphone baselines must be numerically representable")
        if _stable_norm(baseline) <= 1e-9:
            raise ValueError("microphone positions must be distinct")
        baselines[(first, second)] = baseline

    if az_grid is None:
        az_grid = np.linspace(-np.pi, np.pi, 73)[:-1]
    if el_grid is None:
        el_grid = np.linspace(-np.pi / 2.0, np.pi / 2.0, 37)
    az_grid = _search_grid(az_grid, "az_grid")
    el_grid = _search_grid(el_grid, "el_grid")
    if np.any(np.abs(az_grid) > 2.0 * np.pi):
        raise ValueError("az_grid values must lie in [-2pi, 2pi]")
    if np.any(np.abs(el_grid) > np.pi / 2.0 + 1e-12):
        raise ValueError("el_grid values must lie in [-pi/2, pi/2]")
    grid_cells = int(az_grid.size) * int(el_grid.size)
    if grid_cells > MAX_SEARCH_CELLS:
        raise ValueError(f"search grid exceeds the {MAX_SEARCH_CELLS}-cell safety limit")
    _validate_search_observability(baselines, az_grid, el_grid)

    nfft = 1 << (2 * n_samples - 1).bit_length()
    frequency_bins = nfft // 2 + 1
    work = len(pairs) * grid_cells * frequency_bins
    if work > MAX_SRP_FREQUENCY_WORK:
        raise ValueError(
            f"SRP search exceeds the {MAX_SRP_FREQUENCY_WORK}-frequency-work safety limit"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        spectra = np.fft.rfft(normalized_signals, n=nfft, axis=1)
    frequencies = _frequency_bins(nfft, fs)
    if not np.isfinite(spectra).all() or not np.isfinite(frequencies).all():
        raise FloatingPointError("SRP-PHAT spectra or frequencies are not finite")
    cross: dict[tuple[int, int], np.ndarray] = {}
    for first, second in pairs:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cross_spectrum = spectra[first] * np.conj(spectra[second])
            normalized_cross = cross_spectrum / np.maximum(
                np.abs(cross_spectrum), np.finfo(float).eps
            )
        if not np.isfinite(normalized_cross).all():
            raise FloatingPointError("SRP-PHAT cross spectrum is not finite")
        cross[(first, second)] = normalized_cross

    power = np.zeros((el_grid.size, az_grid.size))
    two_pi_f = 2.0 * np.pi * frequencies
    for elevation_index, elevation in enumerate(el_grid):
        for azimuth_index, azimuth in enumerate(az_grid):
            direction = _direction(float(azimuth), float(elevation))
            accumulated_power = 0.0
            for first, second in pairs:
                with np.errstate(over="ignore", invalid="ignore"):
                    delay = -float(np.einsum("i,i->", baselines[(first, second)], direction)) / c
                    phase = two_pi_f * delay
                if not np.isfinite(delay) or not np.isfinite(phase).all():
                    raise FloatingPointError("SRP-PHAT steering delay is not finite")
                with np.errstate(over="ignore", invalid="ignore"):
                    steered = cross[(first, second)] * np.exp(1j * phase)
                if not np.isfinite(steered).all():
                    raise FloatingPointError("SRP-PHAT steered spectrum is not finite")
                accumulated_power += float(np.real(np.sum(steered)))
            if not np.isfinite(accumulated_power):
                raise FloatingPointError("SRP-PHAT power is not finite")
            power[elevation_index, azimuth_index] = accumulated_power

    prominence = srp_peak_prominence(power)
    if min_peak_prominence is not None and prominence < min_peak_prominence:
        raise ValueError(
            f"SRP-PHAT peak prominence {prominence:.3g} is below "
            f"the required {min_peak_prominence:.3g}"
        )
    peak_index = np.unravel_index(int(np.argmax(power)), power.shape)
    elevation_index, azimuth_index = int(peak_index[0]), int(peak_index[1])
    return (
        float(az_grid[azimuth_index]),
        float(el_grid[elevation_index]),
        power,
    )


def synth_plane_wave(
    source: np.ndarray,
    mic_positions: np.ndarray,
    fs: float,
    duration: float = 0.25,
    c: float = SPEED_OF_SOUND,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a broadband far-field plane wave for tests and demos."""
    direction = _real_array(source, "source")
    if direction.size != 3:
        raise ValueError("source must contain three values")
    direction = direction.reshape(3)
    if not np.isfinite(direction).all():
        raise ValueError("source must be a finite, non-zero direction")
    direction_scale = float(np.max(np.abs(direction)))
    if direction_scale == 0:
        raise ValueError("source must be a finite, non-zero direction")
    scaled_direction = direction / direction_scale
    scaled_norm = float(np.sqrt(np.sum(scaled_direction**2)))
    if direction_scale <= 1e-12 / scaled_norm:
        raise ValueError("source must be a finite, non-zero direction")
    direction = scaled_direction / scaled_norm

    microphones = _real_array(mic_positions, "mic_positions")
    if microphones.ndim != 2 or microphones.shape[1:] != (3,) or microphones.shape[0] == 0:
        raise ValueError("mic_positions must have shape (n_mics, 3)")
    if microphones.shape[0] > MAX_MICROPHONES:
        raise ValueError(f"mic_positions exceeds the {MAX_MICROPHONES}-microphone safety limit")
    if not np.isfinite(microphones).all():
        raise ValueError("mic_positions must contain only finite values")
    fs = _finite_scalar(fs, "fs", positive=True)
    duration = _finite_scalar(duration, "duration", positive=True)
    c = _finite_scalar(c, "c", positive=True)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    with np.errstate(over="ignore", invalid="ignore"):
        sample_count = duration * fs
    if not np.isfinite(sample_count) or sample_count > MAX_SIGNAL_SAMPLES:
        raise ValueError(f"duration and fs exceed the {MAX_SIGNAL_SAMPLES}-sample safety limit")
    n_samples = int(sample_count)
    if n_samples < 2:
        raise ValueError("duration and fs must produce at least two samples")
    if len(microphones) * n_samples > MAX_SIGNAL_CELLS:
        raise ValueError(f"synthesized array exceeds the {MAX_SIGNAL_CELLS}-value safety limit")
    if fs > np.finfo(float).max / np.pi:
        raise ValueError("fs is outside the numerically representable range")
    rng = np.random.default_rng(int(seed))
    base = rng.standard_normal(n_samples)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        spectrum = np.fft.rfft(base)
    frequencies = _frequency_bins(n_samples, fs)
    if not np.isfinite(frequencies).all() or not np.isfinite(spectrum).all():
        raise FloatingPointError("plane-wave spectrum is not finite")
    output = np.zeros((len(microphones), n_samples))
    for index, position in enumerate(microphones):
        with np.errstate(over="ignore", invalid="ignore"):
            delay = -float(np.einsum("i,i->", position, direction)) / c
            phase = -2.0 * np.pi * frequencies * delay
        if not np.isfinite(delay) or not np.isfinite(phase).all():
            raise FloatingPointError("plane-wave steering delay is not finite")
        with np.errstate(over="ignore", invalid="ignore"):
            output[index] = np.fft.irfft(
                spectrum * np.exp(1j * phase),
                n=n_samples,
            )
    if not np.isfinite(output).all():
        raise FloatingPointError("synthesized plane wave is not finite")
    return output


__all__ = [
    "gcc_phat",
    "srp_phat",
    "srp_peak_prominence",
    "synth_plane_wave",
    "SPEED_OF_SOUND",
]
