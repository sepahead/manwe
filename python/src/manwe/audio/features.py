"""Dependency-light STFT, log-mel, and sound-pressure features."""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_SIGNAL_SAMPLES = 10_000_000
MAX_FFT_POINTS = 1_048_576
MAX_SPECTRUM_CELLS = 16_000_000
MAX_MEL_BANDS = 4096
MAX_MEL_MULTIPLY_WORK = 100_000_000


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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


def _finite_signal(signal: np.ndarray) -> np.ndarray:
    array = _real_array(signal, "signal")
    if array.ndim != 1:
        raise ValueError("signal must be a one-dimensional array")
    if array.size == 0:
        raise ValueError("signal must contain at least one sample")
    if array.size > MAX_SIGNAL_SAMPLES:
        raise ValueError(f"signal exceeds the {MAX_SIGNAL_SAMPLES}-sample safety limit")
    if not np.isfinite(array).all():
        raise ValueError("signal must contain only finite samples")
    return array


def stft(signal: np.ndarray, n_fft: int = 1024, hop: int = 256, window: str = "hann") -> np.ndarray:
    """Short-time Fourier transform as ``(n_freq, n_frames)``.

    A non-empty signal shorter than ``n_fft`` is zero-padded to one complete
    frame. Empty and non-finite inputs are rejected explicitly.
    """
    signal = _finite_signal(signal)
    n_fft = _positive_int(n_fft, "n_fft")
    hop = _positive_int(hop, "hop")
    if n_fft > MAX_FFT_POINTS:
        raise ValueError(f"n_fft exceeds the {MAX_FFT_POINTS}-point safety limit")
    win = _window(window, n_fft)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    n_frames = 1 + (signal.size - n_fft) // hop
    if n_frames * (n_fft // 2 + 1) > MAX_SPECTRUM_CELLS:
        raise ValueError(f"STFT output exceeds the {MAX_SPECTRUM_CELLS}-value safety limit")
    with np.errstate(over="ignore", invalid="ignore"):
        frames = np.stack(
            [signal[index * hop : index * hop + n_fft] * win for index in range(n_frames)],
            axis=1,
        )
        spectrum = np.fft.rfft(frames, axis=0)
    if not np.isfinite(spectrum).all():
        raise ValueError("signal magnitude overflows the STFT")
    return spectrum


def _window(kind: str, n: int) -> np.ndarray:
    if not isinstance(kind, str):
        raise ValueError("window must be a string")
    if kind == "hann":
        return np.hanning(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind in {"boxcar", "rect", "rectangular"}:
        return np.ones(n)
    raise ValueError("window must be one of: hann, hamming, boxcar, rect, rectangular")


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(over="ignore", invalid="ignore"):
        return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(over="ignore", invalid="ignore"):
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_fft: int, n_mels: int = 64, fmin: float = 20.0, fmax: float | None = None
) -> np.ndarray:
    """Triangular mel filterbank ``(n_mels, n_fft//2 + 1)``."""
    sample_rate = _finite_scalar(sr, "sr", positive=True)
    n_fft = _positive_int(n_fft, "n_fft")
    n_mels = _positive_int(n_mels, "n_mels")
    if n_fft > MAX_FFT_POINTS:
        raise ValueError(f"n_fft exceeds the {MAX_FFT_POINTS}-point safety limit")
    if n_mels > MAX_MEL_BANDS:
        raise ValueError(f"n_mels exceeds the {MAX_MEL_BANDS}-band safety limit")
    fmin = _finite_scalar(fmin, "fmin", nonnegative=True)
    fmax = sample_rate / 2.0 if fmax is None else _finite_scalar(fmax, "fmax", positive=True)
    if fmin >= fmax:
        raise ValueError("fmin must be less than fmax")
    if fmax > sample_rate / 2.0:
        raise ValueError("fmax must not exceed the Nyquist frequency")

    n_freqs = n_fft // 2 + 1
    if n_mels * n_freqs > MAX_SPECTRUM_CELLS:
        raise ValueError(f"mel filterbank exceeds the {MAX_SPECTRUM_CELLS}-value safety limit")
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = np.asarray(_mel_to_hz(mel_points), dtype=float)
    if not np.isfinite(hz_points).all() or np.any(np.diff(hz_points) <= 0):
        raise ValueError("frequency bounds do not produce resolvable mel bands")
    filterbank = np.zeros((n_mels, n_freqs))
    for mel_index in range(1, n_mels + 1):
        low, centre, high = (
            hz_points[mel_index - 1],
            hz_points[mel_index],
            hz_points[mel_index + 1],
        )
        left = (fft_freqs - low) / (centre - low)
        right = (high - fft_freqs) / (high - centre)
        filterbank[mel_index - 1] = np.clip(np.minimum(left, right), 0.0, None)
    if not np.isfinite(filterbank).all():
        raise ValueError("mel filterbank is not finite")
    if np.any(np.max(filterbank, axis=1) <= 0):
        raise ValueError("n_fft is too small to resolve every requested mel band")
    return filterbank


def log_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: float | None = None,
    top_db: float = 80.0,
) -> np.ndarray:
    """Log-mel spectrogram in dB, floored ``top_db`` below its peak."""
    top_db = _finite_scalar(top_db, "top_db", positive=True)
    spectrum = stft(signal, n_fft, hop)
    n_mels = _positive_int(n_mels, "n_mels")
    if n_mels > MAX_MEL_BANDS:
        raise ValueError(f"n_mels exceeds the {MAX_MEL_BANDS}-band safety limit")
    n_frequencies, n_frames = spectrum.shape
    if n_mels * n_frames > MAX_SPECTRUM_CELLS:
        raise ValueError(f"log-mel output exceeds the {MAX_SPECTRUM_CELLS}-value safety limit")
    if n_mels * n_frequencies * n_frames > MAX_MEL_MULTIPLY_WORK:
        raise ValueError(
            f"log-mel projection exceeds the {MAX_MEL_MULTIPLY_WORK}-operation safety limit"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.abs(spectrum) ** 2
    if not np.isfinite(power).all():
        raise ValueError("signal magnitude overflows the power spectrum")
    filterbank = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    with np.errstate(over="ignore", invalid="ignore"):
        mel = np.einsum("mf,ft->mt", filterbank, power)
    if not np.isfinite(mel).all() or np.any(mel < 0):
        raise ValueError("mel power spectrum is not finite and nonnegative")
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
    result = np.maximum(log_mel, log_mel.max() - top_db)
    if not np.isfinite(result).all():
        raise FloatingPointError("log-mel spectrogram is not finite")
    return result


def sound_pressure_level_db(signal: np.ndarray, ref: float = 2e-5) -> float:
    """Approximate SPL from RMS; absolute values require calibrated microphones."""
    signal = _finite_signal(signal)
    ref = _finite_scalar(ref, "ref", positive=True)
    peak = float(np.max(np.abs(signal)))
    rms = 0.0 if peak == 0.0 else peak * float(np.sqrt(np.mean((signal / peak) ** 2)))
    rms = max(rms, np.finfo(float).tiny)
    result = float(20.0 * (np.log10(rms) - np.log10(ref)))
    if not np.isfinite(result):
        raise FloatingPointError("sound-pressure level is not finite")
    return result


__all__ = [
    "stft",
    "mel_filterbank",
    "log_mel_spectrogram",
    "sound_pressure_level_db",
]
