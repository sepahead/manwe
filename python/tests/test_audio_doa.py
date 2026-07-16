"""Audio: GCC-PHAT delay estimation, SRP-PHAT DOA, features, fusion bridge."""

import warnings

import numpy as np
import pytest

from manwe.audio import (
    AcousticDetection,
    detect_from_array,
    gcc_phat,
    log_mel_spectrogram,
    mel_filterbank,
    sound_pressure_level_db,
    srp_peak_prominence,
    srp_phat,
    stft,
    synth_plane_wave,
)


def test_gcc_phat_recovers_known_delay():
    rng = np.random.default_rng(0)
    fs = 16000
    ref = rng.standard_normal(4000)
    shift = 17  # samples
    sig = np.concatenate([np.zeros(shift), ref])[: len(ref)]  # ref delayed by `shift`
    tau, _ = gcc_phat(sig, ref, fs, max_tau=0.01)
    assert abs(abs(tau * fs) - shift) <= 1.5, f"estimated {tau * fs:.1f} samples vs {shift}"


def test_srp_phat_localises_source():
    fs = 16000
    # non-coplanar (tetrahedral) array so elevation is observable
    a = 0.1
    mics = np.array([[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]])
    az_true, el_true = np.radians(40.0), np.radians(30.0)
    u = np.array(
        [
            np.cos(el_true) * np.cos(az_true),
            np.cos(el_true) * np.sin(az_true),
            np.sin(el_true),
        ]
    )
    signals = synth_plane_wave(u, mics, fs, duration=0.15, seed=1)
    az_grid = np.linspace(-np.pi, np.pi, 61)[:-1]
    el_grid = np.linspace(0.0, np.pi / 2, 16)
    az_est, el_est, power = srp_phat(signals, mics, fs, az_grid=az_grid, el_grid=el_grid)
    az_err = np.degrees(abs(np.arctan2(np.sin(az_est - az_true), np.cos(az_est - az_true))))
    el_err = np.degrees(abs(el_est - el_true))
    assert az_err < 12.0, f"azimuth error {az_err:.1f}°"
    assert el_err < 15.0, f"elevation error {el_err:.1f}°"
    assert power.shape == (len(el_grid), len(az_grid))


def test_log_mel_shape():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(16000)
    lm = log_mel_spectrogram(sig, sr=16000, n_fft=1024, hop=256, n_mels=64)
    assert lm.shape[0] == 64
    assert lm.shape[1] > 50


def test_acoustic_detection_to_measurement_bridge():
    det = AcousticDetection(
        azimuth=0.0,
        elevation=0.0,
        range_estimate=100.0,
        spl_db=70.0,
        range_observed=True,
    )
    m = det.to_measurement(sensor_origin=np.zeros(3))
    assert m.modality == "acoustic"
    # az=el=0, range=100 → straight down +x
    assert np.allclose(m.position, [100.0, 0.0, 0.0], atol=1e-6)
    # covariance must be positive-definite and anisotropic (poor range along x)
    assert np.all(np.linalg.eigvalsh(m.covariance) > 0)


def _tetrahedral_mics() -> np.ndarray:
    extent = 0.1
    return np.array(
        [
            [extent, extent, extent],
            [extent, -extent, -extent],
            [-extent, extent, -extent],
            [-extent, -extent, extent],
        ]
    )


def test_stft_pads_short_input_and_rejects_invalid_input():
    spectrum = stft(np.ones(8), n_fft=16, hop=4)
    assert spectrum.shape == (9, 1)
    with pytest.raises(ValueError, match="at least one"):
        stft(np.array([]), n_fft=16)
    with pytest.raises(ValueError, match="finite"):
        stft(np.array([0.0, np.nan]), n_fft=16)
    with pytest.raises(ValueError, match="window"):
        stft(np.ones(16), n_fft=16, window="mystery")


def test_srp_rejects_silence_and_incoherent_noise():
    microphones = _tetrahedral_mics()
    with pytest.raises(ValueError, match="non-silent"):
        srp_phat(np.zeros((4, 1024)), microphones, fs=16000)

    noise = np.random.default_rng(0).standard_normal((4, 2400))
    with pytest.raises(ValueError, match="prominence"):
        srp_phat(noise, microphones, fs=16000)
    with pytest.raises(ValueError):
        detect_from_array(np.zeros((4, 1024)), microphones, fs=16000)


def test_default_srp_grid_supports_negative_elevation():
    fs = 16000
    microphones = _tetrahedral_mics()
    azimuth_true = np.radians(25.0)
    elevation_true = np.radians(-30.0)
    direction = np.array(
        [
            np.cos(elevation_true) * np.cos(azimuth_true),
            np.cos(elevation_true) * np.sin(azimuth_true),
            np.sin(elevation_true),
        ]
    )
    signals = synth_plane_wave(direction, microphones, fs, duration=0.15, seed=8)
    azimuth, elevation, _ = srp_phat(signals, microphones, fs)
    azimuth_error = np.degrees(
        abs(np.arctan2(np.sin(azimuth - azimuth_true), np.cos(azimuth - azimuth_true)))
    )
    assert azimuth_error < 12.0
    assert abs(np.degrees(elevation - elevation_true)) < 12.0


def test_acoustic_detection_validation():
    with pytest.raises(ValueError, match="elevation"):
        AcousticDetection(0.0, np.pi, 10.0, 60.0)
    with pytest.raises(ValueError, match="range_estimate"):
        AcousticDetection(0.0, 0.0, -1.0, 60.0)
    assert AcousticDetection(0.0, 0.0, 10.0, -1.0).spl_db == -1.0
    with pytest.raises(ValueError, match="spl_db"):
        AcousticDetection(0.0, 0.0, 10.0, np.nan)
    with pytest.raises(ValueError, match="confidence"):
        AcousticDetection(0.0, 0.0, 10.0, 60.0, confidence=np.nan)
    with pytest.raises(ValueError, match="range_observed"):
        AcousticDetection(0.0, 0.0, 10.0, 60.0, range_observed=1)
    with pytest.raises(ValueError, match="class_label"):
        AcousticDetection(0.0, 0.0, 10.0, 60.0, class_label="")
    for class_label in ("line\nbreak", "x" * 257, "é" * 129):
        with pytest.raises(ValueError, match="bounded printable"):
            AcousticDetection(0.0, 0.0, 10.0, 60.0, class_label=class_label)
    assert (
        AcousticDetection(0.0, 0.0, 10.0, 60.0, class_label="  " + "x" * 256 + "  ").class_label
        == "x" * 256
    )


def test_acoustic_bridge_rotates_position_and_covariance():
    detection = AcousticDetection(0.0, 0.0, 10.0, 60.0, range_observed=True)
    origin = np.array([1.0, 2.0, 3.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    array_measurement = detection.to_measurement()
    world_measurement = detection.to_measurement(
        sensor_origin=origin,
        sensor_rotation=rotation,
    )
    assert np.allclose(world_measurement.position, [1.0, 12.0, 3.0])
    assert np.allclose(
        world_measurement.covariance,
        rotation @ array_measurement.covariance @ rotation.T,
    )
    with pytest.raises(ValueError, match="determinant"):
        detection.to_measurement(sensor_rotation=np.diag([1.0, 1.0, -1.0]))


def test_log_mel_happy_path_is_runtime_warning_free():
    signal = np.random.default_rng(20).standard_normal(4096)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = log_mel_spectrogram(signal, sr=16000, n_fft=512, hop=128, n_mels=32)
    assert result.shape[0] == 32
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 8)),
        np.ones(8, dtype=bool),
        np.ones(8, dtype=complex),
        np.array(["1", "2"]),
    ],
)
def test_audio_features_require_one_dimensional_real_signals(signal):
    with pytest.raises(ValueError):
        stft(signal, n_fft=8)
    with pytest.raises(ValueError):
        sound_pressure_level_db(signal)


def test_audio_features_reject_overflow_without_runtime_warnings():
    signal = np.full(32, np.finfo(float).max)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="overflows"):
            stft(signal, n_fft=32, window="boxcar")


def test_sound_pressure_level_uses_stable_log_domain_ratio():
    result = sound_pressure_level_db(
        np.array([np.finfo(float).max]),
        ref=np.nextafter(0.0, 1.0),
    )
    assert np.isfinite(result)
    assert result > 0


def test_mel_filterbank_rejects_degenerate_resolution_and_bad_bounds():
    filterbank = mel_filterbank(16000, 1024, n_mels=64)
    assert np.isfinite(filterbank).all()
    assert np.all(filterbank >= 0)
    assert np.all(np.max(filterbank, axis=1) > 0)

    with pytest.raises(ValueError, match="too small"):
        mel_filterbank(16000, 8, n_mels=64)
    with pytest.raises(ValueError, match="less than"):
        mel_filterbank(16000, 1024, fmin=8000.0)
    with pytest.raises(ValueError, match="Nyquist"):
        mel_filterbank(16000, 1024, fmax=9000.0)
    with pytest.raises(ValueError):
        mel_filterbank(np.nan, 1024)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fs": 0.0}, "fs"),
        ({"fs": np.nan}, "fs"),
        ({"fs": np.finfo(float).max}, "interpolation rate"),
        ({"fs": 16000.0, "interp": 0}, "interp"),
        ({"fs": 16000.0, "interp": True}, "interp"),
        ({"fs": 16000.0, "max_tau": 0.0}, "max_tau"),
    ],
)
def test_gcc_phat_rejects_invalid_numeric_parameters(kwargs, message):
    signal = np.random.default_rng(21).standard_normal(32)
    with pytest.raises(ValueError, match=message):
        gcc_phat(signal, signal, **kwargs)


@pytest.mark.parametrize(
    "signal",
    [
        np.ones((2, 8)),
        np.ones(8, dtype=bool),
        np.ones(8, dtype=complex),
        np.full(8, np.nan),
    ],
)
def test_gcc_phat_rejects_invalid_signal_arrays(signal):
    with pytest.raises(ValueError):
        gcc_phat(signal, np.ones(8), fs=16000)


def test_srp_peak_prominence_is_stable_for_extreme_finite_power():
    maximum = np.finfo(float).max
    prominence = srp_peak_prominence(np.array([-maximum, 0.0, maximum]))
    assert np.isfinite(prominence)
    assert prominence > 0
    with pytest.raises(ValueError, match="real numeric"):
        srp_peak_prominence(np.ones(3, dtype=complex))


def test_srp_phat_rejects_invalid_geometry_and_frequency_ranges():
    signals = np.ones((2, 64))
    microphones = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    with pytest.raises(ValueError, match="distinct"):
        srp_phat(signals, np.zeros((2, 3)), fs=16000, min_peak_prominence=None)
    unrepresentable = np.array([[np.finfo(float).max, 0.0, 0.0], [-np.finfo(float).max, 0.0, 0.0]])
    with pytest.raises(ValueError, match="representable"):
        srp_phat(signals, unrepresentable, fs=16000, min_peak_prominence=None)
    with pytest.raises(ValueError, match="representable range"):
        srp_phat(
            signals,
            microphones,
            fs=np.finfo(float).max,
            min_peak_prominence=None,
        )
    with pytest.raises(ValueError, match="el_grid"):
        srp_phat(
            signals,
            microphones,
            fs=16000,
            el_grid=np.array([np.pi]),
            min_peak_prominence=None,
        )
    with pytest.raises(ValueError, match="real numeric"):
        srp_phat(
            signals.astype(complex),
            microphones,
            fs=16000,
            min_peak_prominence=None,
        )


def test_signed_elevation_rejects_planar_microphone_geometry():
    planar = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    signals = np.random.default_rng(33).standard_normal((3, 64))
    with pytest.raises(ValueError, match="cannot uniquely observe"):
        srp_phat(
            signals,
            planar,
            fs=16000,
            el_grid=np.array([-0.5, 0.5]),
            min_peak_prominence=None,
        )


def test_srp_observability_excludes_subthreshold_microphones():
    fs = 16000
    microphones = _tetrahedral_mics()
    signals = synth_plane_wave(
        np.array([1.0, 0.0, 0.0]),
        microphones,
        fs,
        duration=0.05,
        seed=20260716,
    )
    # Only the first baseline is energetic. The other two channels are nonzero,
    # but PHAT normalization must not promote their sub-threshold signals into
    # geometric evidence. That active line cannot identify a full 3-D direction.
    signals[2:] *= 1e-20
    active_baseline = microphones[0] - microphones[1]
    assert active_baseline @ np.array([1.0, 0.0, 0.0]) == active_baseline @ np.array(
        [-1.0, 0.0, 0.0]
    )

    with pytest.raises(ValueError, match="cannot uniquely observe"):
        srp_phat(
            signals,
            microphones,
            fs,
            min_rms=1e-8,
            min_peak_prominence=None,
        )


def test_srp_rejects_a_prominent_but_unobservable_collinear_array_peak():
    fs = 16000
    microphones = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    direction = np.array([0.0, 1.0, 0.0])
    signals = synth_plane_wave(direction, microphones, fs, duration=0.25, seed=20260715)
    azimuths = np.deg2rad(np.arange(-180.0, 180.0, 1.0))

    # For a baseline on x, delay is proportional to cos(azimuth), so the
    # steering signatures at +90 and -90 degrees are exactly identical. Before
    # the observability gate, both peaks had prominence 5.51 and passed the
    # default threshold of 4 despite the unresolved front/back ambiguity.
    baseline = microphones[0] - microphones[1]
    positive = np.array([0.0, 1.0, 0.0])
    negative = np.array([0.0, -1.0, 0.0])
    assert float(baseline @ positive) == float(baseline @ negative)
    with pytest.raises(ValueError, match="cannot uniquely observe"):
        srp_phat(
            signals,
            microphones,
            fs,
            az_grid=azimuths,
            el_grid=np.array([0.0]),
        )


def test_srp_allows_a_line_array_when_the_grid_encodes_a_two_direction_prior():
    fs = 16000
    microphones = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    signals = synth_plane_wave(
        np.array([1.0, 0.0, 0.0]),
        microphones,
        fs,
        duration=0.05,
        seed=20260715,
    )
    azimuth, elevation, _ = srp_phat(
        signals,
        microphones,
        fs,
        az_grid=np.array([0.0, np.pi / 2.0]),
        el_grid=np.array([0.0]),
        min_peak_prominence=None,
    )
    assert azimuth == 0.0
    assert elevation == 0.0


def test_synth_plane_wave_handles_large_direction_and_rejects_overflow():
    microphones = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    maximum = np.finfo(float).max
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        signal = synth_plane_wave(
            np.array([maximum, maximum, maximum]),
            microphones,
            fs=100.0,
            duration=0.02,
        )
    assert signal.shape == (2, 2)
    assert np.isfinite(signal).all()

    with pytest.raises(ValueError, match="sample safety limit"):
        synth_plane_wave(np.ones(3), microphones, fs=maximum, duration=maximum)
    with pytest.raises(ValueError, match="real numeric"):
        synth_plane_wave(np.ones(3, dtype=complex), microphones, fs=100.0)
    with pytest.raises(ValueError, match="non-zero"):
        synth_plane_wave(np.zeros(3), microphones, fs=100.0)


def test_acoustic_bridge_rejects_nonfinite_position_and_covariance_math():
    detection = AcousticDetection(
        0.0,
        0.0,
        np.finfo(float).max,
        60.0,
        range_observed=True,
    )
    with pytest.raises(FloatingPointError, match="position"):
        detection.to_measurement(sensor_origin=np.array([np.finfo(float).max, 0.0, 0.0]))

    detection = AcousticDetection(0.0, 0.0, 10.0, 60.0, range_observed=True)
    with pytest.raises(FloatingPointError, match="covariance"):
        detection.to_measurement(range_std=np.finfo(float).max)
    with pytest.raises(ValueError, match="real numeric"):
        detection.to_measurement(sensor_rotation=np.eye(3, dtype=complex))


def test_huge_python_integers_are_rejected_as_values_not_type_errors():
    huge = 10**1000
    signal = np.ones(16)
    with pytest.raises(ValueError, match="finite float"):
        gcc_phat(signal, signal, fs=huge)
    with pytest.raises(ValueError, match="finite float"):
        AcousticDetection(0.0, 0.0, huge, 60.0)
    with pytest.raises(ValueError, match="finite float"):
        sound_pressure_level_db(signal, ref=huge)


def test_detect_from_array_supports_negative_spl_and_silent_first_channel():
    fs = 16000
    # Leave a full-rank tetrahedral sub-array after silencing the first channel.
    microphones = np.vstack((np.zeros((1, 3)), _tetrahedral_mics()))
    signals = synth_plane_wave(np.array([1.0, 0.0, 0.0]), microphones, fs, duration=0.05, seed=30)
    signals *= 1e-6
    signals[0] = 0.0
    detection = detect_from_array(
        signals,
        microphones,
        fs,
        min_rms=1e-9,
        min_peak_prominence=None,
    )
    assert np.isfinite(detection.spl_db)
    assert detection.spl_db < 0
    assert detection.range_observed is False
    with pytest.raises(ValueError, match="independently observed range"):
        detection.to_measurement()


def test_gcc_phat_honours_subsample_max_tau():
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    delayed = np.array([0.0, 1.0, 0.0, 0.0])
    max_tau = 0.01
    delay, correlation = gcc_phat(delayed, reference, fs=1.0, max_tau=max_tau, interp=1)
    assert abs(delay) <= max_tau
    assert delay == 0.0
    assert correlation.shape == (1,)


def test_fft_frequency_grid_preserves_subnormal_but_resolvable_bins():
    from manwe.audio.doa import _frequency_bins

    frequencies = _frequency_bins(4, 1e-308)
    assert frequencies[0] == 0.0
    assert frequencies[1] > 0.0
    assert np.all(np.diff(frequencies) > 0.0)
    with pytest.raises(ValueError, match="resolvable"):
        _frequency_bins(4, np.nextafter(0.0, 1.0))


def test_acoustic_covariance_rejects_positive_std_underflow():
    detection = AcousticDetection(0.0, 0.0, 10.0, 60.0, range_observed=True)
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(FloatingPointError, match="underflowed"):
        detection.to_measurement(angle_std=tiny, range_std=tiny)
    with pytest.raises(FloatingPointError, match="underflowed"):
        detection.to_measurement(range_std=tiny)


def test_mutated_acoustic_detection_is_revalidated_at_public_boundaries():
    detection = AcousticDetection(0.0, 0.0, 10.0, 60.0, range_observed=True)
    detection.azimuth = np.inf
    with pytest.raises(ValueError, match="azimuth"):
        detection.direction()
    with pytest.raises(ValueError, match="azimuth"):
        detection.to_measurement()

    detection.azimuth = 0.0
    detection.class_label = " "
    with pytest.raises(ValueError, match="class_label"):
        detection.to_measurement()

    detection.class_label = "x" * 257
    with pytest.raises(ValueError, match="bounded printable"):
        detection.to_measurement()

    detection.class_label = None
    detection.range_observed = 1
    with pytest.raises(ValueError, match="range_observed"):
        detection.to_measurement()


def test_acoustic_azimuth_rejects_unreliable_float_canonicalization():
    # At 1e16, adjacent finite floats are already two radians apart. Producing a
    # plausible-looking unit direction would hide information lost before entry.
    assert np.isfinite(AcousticDetection(1_000_000.0, 0.0, 10.0, 60.0).azimuth)
    with pytest.raises(ValueError, match="canonicalize reliably"):
        AcousticDetection(1e16, 0.0, 10.0, 60.0)

    detection = AcousticDetection(0.0, 0.0, 10.0, 60.0, range_observed=True)
    detection.azimuth = 1e16
    with pytest.raises(ValueError, match="canonicalize reliably"):
        detection.direction()
    with pytest.raises(ValueError, match="canonicalize reliably"):
        detection.to_measurement()


def test_single_array_nominal_range_cannot_be_repeated_as_an_observation():
    fs = 16000
    microphones = _tetrahedral_mics()
    signals = synth_plane_wave(
        np.array([1.0, 0.0, 0.0]),
        microphones,
        fs,
        duration=0.05,
        seed=20260715,
    )
    detections = [
        detect_from_array(
            signals,
            microphones,
            fs,
            nominal_range=100.0,
            min_peak_prominence=None,
        )
        for _ in range(2)
    ]
    for detection in detections:
        assert detection.range_estimate == 100.0
        assert detection.range_observed is False
        with pytest.raises(ValueError, match="single-array nominal range"):
            detection.to_measurement()


def test_synth_plane_wave_rejects_negative_seed_explicitly():
    with pytest.raises(ValueError, match="nonnegative integer"):
        synth_plane_wave(
            np.ones(3),
            np.array([[0.0, 0.0, 0.0]]),
            fs=100.0,
            duration=0.02,
            seed=-1,
        )


def test_audio_work_budgets_fail_before_large_allocations():
    signal = np.ones(8)
    with pytest.raises(ValueError, match="FFT limit"):
        gcc_phat(signal, signal, fs=16000, interp=2_000_000)
    with pytest.raises(ValueError, match="n_fft exceeds"):
        stft(signal, n_fft=1_048_577)
    with pytest.raises(ValueError, match="n_mels exceeds"):
        mel_filterbank(16000, 1024, n_mels=4097)
    with pytest.raises(ValueError, match="log-mel output"):
        log_mel_spectrogram(np.ones(4001), n_fft=2, hop=1, n_mels=4096)
    with pytest.raises(ValueError, match="log-mel projection"):
        log_mel_spectrogram(np.ones(1), n_fft=65536, hop=1, n_mels=4096)

    signals = np.ones((65, 2))
    microphones = np.column_stack((np.arange(65), np.zeros(65), np.zeros(65)))
    with pytest.raises(ValueError, match="microphone or sample safety limit"):
        srp_phat(signals, microphones, fs=16000, min_peak_prominence=None)

    with pytest.raises(ValueError, match="sample safety limit"):
        synth_plane_wave(
            np.array([1.0, 0.0, 0.0]),
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            fs=16000,
            duration=100.0,
        )


def test_audio_input_caps_precede_float_widening(monkeypatch):
    from manwe.audio import doa as doa_module
    from manwe.audio import features as features_module

    oversized_gcc_signal = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (doa_module.MAX_SIGNAL_SAMPLES + 1,),
    )
    oversized_signal_matrix = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (9, 888_889),
    )
    oversized_grid = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (doa_module.MAX_SEARCH_CELLS + 1,),
    )
    oversized_source = np.broadcast_to(np.array(1, dtype=np.int8), (4,))
    oversized_microphones = np.broadcast_to(np.array(1, dtype=np.int8), (65, 3))
    oversized_feature_signal = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (features_module.MAX_SIGNAL_SAMPLES + 1,),
    )
    forbidden_doa_inputs = (
        oversized_gcc_signal,
        oversized_signal_matrix,
        oversized_grid,
        oversized_source,
        oversized_microphones,
    )
    real_doa_float_array = doa_module._float_array
    real_feature_float_array = features_module._float_array

    def guarded_doa_float_array(raw, name):
        if any(
            raw.shape == forbidden.shape and np.shares_memory(raw, forbidden)
            for forbidden in forbidden_doa_inputs
        ):
            pytest.fail(f"{name} was widened before its raw shape/size limit")
        return real_doa_float_array(raw, name)

    def guarded_feature_float_array(raw, name):
        if raw.shape == oversized_feature_signal.shape and np.shares_memory(
            raw, oversized_feature_signal
        ):
            pytest.fail(f"{name} was widened before its raw shape/size limit")
        return real_feature_float_array(raw, name)

    monkeypatch.setattr(doa_module, "_float_array", guarded_doa_float_array)
    monkeypatch.setattr(features_module, "_float_array", guarded_feature_float_array)

    with pytest.raises(ValueError, match="sample safety limit"):
        gcc_phat(oversized_gcc_signal, np.ones(2), fs=16000)
    with pytest.raises(ValueError, match="value safety limit"):
        srp_phat(
            oversized_signal_matrix,
            np.zeros((9, 3)),
            fs=16000,
            min_peak_prominence=None,
        )
    with pytest.raises(ValueError, match="value safety limit"):
        doa_module._search_grid(oversized_grid, "az_grid")
    with pytest.raises(ValueError, match="value safety limit"):
        srp_peak_prominence(oversized_grid)
    with pytest.raises(ValueError, match="three values"):
        synth_plane_wave(
            oversized_source,
            np.zeros((2, 3)),
            fs=100,
            duration=0.02,
        )
    with pytest.raises(ValueError, match="microphone safety limit"):
        synth_plane_wave(
            np.ones(3),
            oversized_microphones,
            fs=100,
            duration=0.02,
        )
    with pytest.raises(ValueError, match="sample safety limit"):
        stft(oversized_feature_signal, n_fft=8)
