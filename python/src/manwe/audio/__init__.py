"""Acoustic detection: features, microphone-array DOA, and the fusion bridge."""

from __future__ import annotations

from .detect import AcousticDetection, detect_from_array
from .doa import SPEED_OF_SOUND, gcc_phat, srp_peak_prominence, srp_phat, synth_plane_wave
from .features import log_mel_spectrogram, mel_filterbank, sound_pressure_level_db, stft

__all__ = [
    "log_mel_spectrogram",
    "mel_filterbank",
    "stft",
    "sound_pressure_level_db",
    "gcc_phat",
    "srp_phat",
    "srp_peak_prominence",
    "synth_plane_wave",
    "SPEED_OF_SOUND",
    "AcousticDetection",
    "detect_from_array",
]
