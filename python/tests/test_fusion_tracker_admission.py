"""Adversarial admission and transaction tests for tracker/scenario boundaries."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable

import numpy as np
import pytest

import manwe.fusion.scenarios as scenarios_module
import manwe.fusion.tracker as tracker_module
from manwe.fusion.scenarios import (
    Scenario,
    _constant_acceleration_step,
    make_scenario,
    score_tracker,
)
from manwe.fusion.tracker import (
    Measurement,
    MultiSensorTracker,
    TrackerConfig,
    TrackOutput,
    measurement_cartesian,
    radar_polar_to_cartesian,
)


def _zero_stride(
    *,
    shape: tuple[int, ...],
    dtype: np.dtype | type = np.int64,
    value: int | float = 1,
) -> np.ndarray:
    base = np.array([value], dtype=dtype)
    return np.lib.stride_tricks.as_strided(
        base,
        shape=shape,
        strides=(0,) * len(shape),
        writeable=False,
    )


class _Coercive:
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("object element coercion must not run")


@pytest.fixture(autouse=True)
def _reset_coercion_counter():
    _Coercive.calls = 0
    yield
    assert _Coercive.calls == 0


@pytest.mark.parametrize(
    ("field", "expected_name"),
    [
        ("position", "position"),
        ("covariance", "covariance"),
        ("sensor_origin", "sensor_origin"),
        ("velocity", "velocity"),
    ],
)
def test_measurement_rejects_wrong_shape_before_float64_widening(
    monkeypatch,
    field,
    expected_name,
):
    calls: list[str] = []
    original: Callable = tracker_module._float64_array

    def recording_float64(array, name):
        calls.append(name)
        return original(array, name)

    monkeypatch.setattr(tracker_module, "_float64_array", recording_float64)
    values = {
        "modality": "visual",
        "position": np.zeros(3),
        "covariance": np.eye(3),
        "timestamp": 0.0,
        "sensor_origin": np.zeros(3),
        "velocity": np.zeros(3),
    }
    values[field] = _zero_stride(shape=(2_000_000,), dtype=np.int64)

    with pytest.raises(ValueError, match="shape"):
        Measurement(**values)

    assert expected_name not in calls


def test_radar_helpers_and_postconstruction_projection_admit_shape_before_cast(
    monkeypatch,
):
    calls: list[str] = []
    original: Callable = tracker_module._float64_array

    def recording_float64(array, name):
        calls.append(name)
        return original(array, name)

    monkeypatch.setattr(tracker_module, "_float64_array", recording_float64)
    oversized = _zero_stride(shape=(2_000_000,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        radar_polar_to_cartesian(oversized, np.zeros(3))
    assert "polar position" not in calls

    measurement = Measurement("visual", [1.0, 2.0, 3.0], np.eye(3), 0.0)
    measurement.position = oversized
    calls.clear()
    with pytest.raises(ValueError, match="shape"):
        measurement_cartesian(measurement)
    assert "position" not in calls


def test_cartesian_projection_returns_owned_arrays():
    measurement = Measurement("visual", [1.0, 2.0, 3.0], np.eye(3), 0.0)
    position, covariance = measurement_cartesian(measurement)

    position[0] = 99.0
    covariance[0, 0] = 99.0

    assert measurement.position[0] == 1.0
    assert measurement.covariance[0, 0] == 1.0


@pytest.mark.parametrize(
    "constructor",
    [
        lambda value: Measurement(
            "visual",
            np.full(3, value, dtype=object),
            np.eye(3),
            0.0,
        ),
        lambda value: TrackerConfig(sigma_a=value),
        lambda value: make_scenario(duration=value),
        lambda value: score_tracker(
            MultiSensorTracker(),
            Scenario(np.arange(4.0), [], [[], [], [], []]),
            c=value,
        ),
    ],
)
def test_object_numeric_inputs_are_rejected_without_float_coercion(constructor):
    with pytest.raises(ValueError):
        constructor(_Coercive())


def test_python_and_numpy_numeric_subclass_hooks_are_rejected_before_coercion():
    calls = 0

    class HookedFloat(float):
        def __float__(self):
            nonlocal calls
            calls += 1
            return super().__float__()

    class HookedInt(int):
        def __int__(self):
            nonlocal calls
            calls += 1
            return super().__int__()

    class HookedNumpyInt(np.int64):
        def __int__(self):
            nonlocal calls
            calls += 1
            return super().__int__()

    with pytest.raises(ValueError, match="real numeric"):
        Measurement(
            "visual",
            [HookedFloat(1.0), 2.0, 3.0],
            np.eye(3),
            0.0,
        )
    with pytest.raises(ValueError):
        TrackerConfig(sigma_a=HookedFloat(1.0))
    with pytest.raises(ValueError):
        TrackerConfig(confirm_hits=HookedInt(1))
    with pytest.raises(ValueError):
        TrackerConfig(confirm_hits=HookedNumpyInt(1))
    assert calls == 0


def test_fixed_shape_nested_sequence_is_rejected_before_leaf_inspection(monkeypatch):
    oversized_wrong_shape = [[_Coercive()] * 1_000_000]

    def forbidden_scalar_inspection(_value):
        raise AssertionError("wrong container shape must be rejected before leaf inspection")

    monkeypatch.setattr(
        tracker_module,
        "_is_supported_real_scalar",
        forbidden_scalar_inspection,
    )
    with pytest.raises(ValueError, match="shape"):
        tracker_module._as_finite_vector(oversized_wrong_shape, "position")


def test_cyclic_numeric_sequence_is_rejected_before_numpy_traversal(monkeypatch):
    cyclic: list[object] = []
    cyclic.append(cyclic)

    def forbidden_asarray(*_args, **_kwargs):
        raise AssertionError("cyclic containers must be rejected before np.asarray")

    monkeypatch.setattr(tracker_module.np, "asarray", forbidden_asarray)
    with pytest.raises(ValueError, match="cyclic"):
        tracker_module._raw_real_array(cyclic, "cyclic values")


def test_gating_callback_object_result_is_not_coerced_and_cycle_rolls_back():
    tracker = MultiSensorTracker(TrackerConfig(confirm_hits=1))
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    before = tracker.tracks[0].filt.state.copy()
    before_age = tracker.tracks[0].age
    tracker.tracks[0].gating_distance = lambda *_: _Coercive()

    with pytest.raises(ValueError, match="invalid gating distance"):
        tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 1.0)], 1.0)

    assert tracker._last_t == 0.0
    assert tracker.tracks[0].age == before_age
    np.testing.assert_array_equal(tracker.tracks[0].filt.state.x, before.x)
    np.testing.assert_array_equal(tracker.tracks[0].filt.state.P, before.P)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Measurement(
            "visual",
            np.array([2**53 + 1, 0, 0], dtype=np.uint64),
            np.eye(3),
            0.0,
        ),
        lambda: Measurement(
            "visual",
            np.zeros(3),
            np.array([2**53 + 1, 1, 1], dtype=np.uint64),
            0.0,
        ),
        lambda: TrackerConfig(gate_chi2=np.uint64(2**53 + 1)),
    ],
)
def test_float64_integer_admission_prevents_distinct_value_collapse(factory):
    with pytest.raises(ValueError, match="exact|consecutive"):
        factory()


def test_score_rejects_large_integer_times_and_truth_before_semantic_collapse():
    frames = [[], [], [], []]
    large_times = np.arange(2**53, 2**53 + 4, dtype=np.uint64)
    with pytest.raises(ValueError, match="exact|consecutive"):
        score_tracker(MultiSensorTracker(), Scenario(large_times, [], frames))

    truth = [
        np.full((4, 3), np.uint64(2**53 + 1), dtype=np.uint64),
    ]
    with pytest.raises(ValueError, match="precision|exact|consecutive"):
        score_tracker(
            MultiSensorTracker(),
            Scenario(np.arange(4.0), truth, frames),
        )


def test_extended_precision_narrowing_is_explicitly_rejected():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform long double has no precision beyond float64")
    precise = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    position = np.array([precise, 0.0, 0.0], dtype=np.longdouble)

    with pytest.raises(ValueError, match="precision"):
        Measurement("visual", position, np.eye(3), 0.0)


def test_probability_overflow_and_near_max_covariance_fail_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="sum to 1"):
            TrackOutput(
                id=1,
                position=np.zeros(3),
                velocity=np.zeros(3),
                covariance=np.eye(3),
                state="confirmed",
                age=1,
                class_label=None,
                state_timestamp=1.0,
                last_measurement_timestamp=1.0,
                updated_this_cycle=True,
                mode_probs=[np.finfo(float).max, np.finfo(float).max],
            )

        near_max_psd = np.full((3, 3), np.finfo(float).max / 3.0)
        measurement = Measurement(
            "visual",
            np.zeros(3),
            near_max_psd,
            0.0,
        )
        stabilized = MultiSensorTracker._stabilize_covariance(measurement.covariance)

    assert np.isfinite(measurement.covariance).all()
    assert np.isfinite(stabilized).all()
    assert np.all(np.linalg.eigvalsh(stabilized) > 0)


def test_oversized_scenario_times_fail_before_float64_widening(monkeypatch):
    calls: list[str] = []
    original: Callable = scenarios_module._float64_array

    def recording_float64(array, name):
        calls.append(name)
        return original(array, name)

    monkeypatch.setattr(scenarios_module, "_float64_array", recording_float64)
    times = _zero_stride(
        shape=(scenarios_module._MAX_SCENARIO_FRAMES + 1,),
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="frame limit"):
        score_tracker(MultiSensorTracker(), Scenario(times, [], []))

    assert "scenario times" not in calls


def test_truth_shape_and_origin_shape_fail_before_bulk_widening(monkeypatch):
    scenario_calls: list[str] = []
    tracker_calls: list[str] = []
    original_scenario: Callable = scenarios_module._float64_array
    original_tracker: Callable = tracker_module._float64_array

    def record_scenario(array, name):
        scenario_calls.append(name)
        return original_scenario(array, name)

    def record_tracker(array, name):
        tracker_calls.append(name)
        return original_tracker(array, name)

    monkeypatch.setattr(scenarios_module, "_float64_array", record_scenario)
    monkeypatch.setattr(tracker_module, "_float64_array", record_tracker)
    malformed_truth = _zero_stride(shape=(2_000_000, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        score_tracker(
            MultiSensorTracker(),
            Scenario(np.arange(4.0), [malformed_truth], [[], [], [], []]),
        )
    assert not any(name.startswith("truth trajectory") for name in scenario_calls)

    oversized_origin = _zero_stride(shape=(2_000_000,), dtype=np.int64)
    with pytest.raises(ValueError, match="sensor_origin"):
        make_scenario(sensor_origin=oversized_origin)
    assert "sensor_origin" not in tracker_calls


def test_scenario_cardinality_and_seed_limits_precede_rng_creation(monkeypatch):
    def forbidden_rng(*_args, **_kwargs):
        raise AssertionError("RNG construction must follow scalar/cardinality admission")

    monkeypatch.setattr(scenarios_module.np.random, "default_rng", forbidden_rng)
    with pytest.raises(ValueError, match="n_targets"):
        make_scenario(n_targets=scenarios_module._MAX_SCENARIO_TARGETS + 1)
    with pytest.raises(ValueError, match="seed"):
        make_scenario(seed=scenarios_module._MAX_SEED + 1)
    with pytest.raises(ValueError, match="at most"):
        make_scenario(modalities=("visual",) * (len(scenarios_module.MODALITY_NOISE) + 1))


def test_mutated_noise_table_is_bounded_before_widening(monkeypatch):
    calls: list[str] = []
    original: Callable = tracker_module._float64_array

    def recording_float64(array, name):
        calls.append(name)
        return original(array, name)

    monkeypatch.setattr(tracker_module, "_float64_array", recording_float64)
    monkeypatch.setitem(
        scenarios_module.MODALITY_NOISE,
        "visual",
        _zero_stride(shape=(2_000_000,), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="shape"):
        make_scenario(n_targets=0, duration=0.5, modalities=("visual",))

    assert "MODALITY_NOISE['visual']" not in calls


def test_constant_acceleration_admits_shape_before_cast_and_fails_closed_on_overflow(
    monkeypatch,
):
    calls: list[str] = []
    original: Callable = tracker_module._float64_array

    def recording_float64(array, name):
        calls.append(name)
        return original(array, name)

    monkeypatch.setattr(tracker_module, "_float64_array", recording_float64)
    with pytest.raises(ValueError, match="shape"):
        _constant_acceleration_step(
            _zero_stride(shape=(2_000_000,), dtype=np.float32),
            np.zeros(3),
            np.zeros(3),
            1.0,
        )
    assert "position" not in calls

    with pytest.raises(FloatingPointError, match="remain finite"):
        _constant_acceleration_step(
            np.full(3, np.finfo(float).max),
            np.full(3, np.finfo(float).max),
            np.ones(3),
            2.0,
        )


def test_corrupted_runtime_shape_is_rejected_before_deepcopy(monkeypatch):
    tracker = MultiSensorTracker()
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    tracker.tracks[0].filt.state.x = _zero_stride(
        shape=(2_000_000,),
        dtype=np.float32,
    )

    def forbidden_deepcopy(_value):
        raise AssertionError("invalid state must be rejected before deepcopy")

    monkeypatch.setattr(tracker_module.copy, "deepcopy", forbidden_deepcopy)
    with pytest.raises(ValueError, match="state.x must have shape"):
        tracker.step([], 1.0)


def test_internal_ndarray_subclass_is_rejected_before_its_deepcopy_hook():
    deepcopy_calls = 0

    class HookedArray(np.ndarray):
        def __deepcopy__(self, _memo):
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            raise AssertionError("ndarray subclass deepcopy hook must not run")

    tracker = MultiSensorTracker()
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    tracker.tracks[0].filt.state.x = tracker.tracks[0].filt.state.x.view(HookedArray)

    with pytest.raises(ValueError, match="ndarray subclass"):
        tracker.step([], 1.0)

    assert deepcopy_calls == 0


def test_injected_snapshot_hook_is_rejected_before_deepcopy():
    deepcopy_calls = 0

    class HookedState:
        def __deepcopy__(self, _memo):
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            raise AssertionError("injected deepcopy hook must not run")

    tracker = MultiSensorTracker()
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    tracker.tracks[0].injected = HookedState()

    with pytest.raises(ValueError, match="namespace was corrupted"):
        tracker.step([], 1.0)

    assert deepcopy_calls == 0


def test_hook_in_expected_filter_field_is_rejected_before_deepcopy():
    deepcopy_calls = 0

    class HookedState:
        def __deepcopy__(self, _memo):
            nonlocal deepcopy_calls
            deepcopy_calls += 1
            raise AssertionError("injected deepcopy hook must not run")

    tracker = MultiSensorTracker()
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    tracker.tracks[0].filt._last_likelihood = HookedState()

    with pytest.raises(TypeError, match="unsafe type"):
        tracker.step([], 1.0)

    assert deepcopy_calls == 0


def test_state_covariance_rejects_even_tiny_negative_eigenvalues():
    covariance = np.eye(6)
    covariance[0, 1] = covariance[1, 0] = 1.0 + 5e-11

    with pytest.raises(ValueError, match="positive semidefinite"):
        tracker_module._as_state_covariance(covariance)


def test_slotted_runtime_config_is_revalidated_and_reflective_corruption_rejected():
    tracker = MultiSensorTracker()
    tracker.step([], 0.0)
    object.__setattr__(tracker.cfg, "max_tracks", _Coercive())

    with pytest.raises(ValueError, match="configuration was corrupted"):
        tracker.step([], 1.0)


def test_iterable_callback_mutations_are_rolled_back():
    tracker = MultiSensorTracker(TrackerConfig(confirm_hits=1))
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    original_cfg = tracker.cfg
    expected_cfg = TrackerConfig(confirm_hits=1)
    original_rng = tracker.rng
    original_rng_state = copy.deepcopy(tracker.rng.bit_generator.state)
    original_state = tracker.tracks[0].filt.state.copy()
    original_age = tracker.tracks[0].age

    def mutating_iterable():
        object.__setattr__(tracker.cfg, "max_tracks", 1)
        tracker.rng = np.random.default_rng(999)
        tracker.tracks.clear()
        yield object()

    with pytest.raises(TypeError, match=r"measurements\[0\]"):
        tracker.step(mutating_iterable(), 1.0)

    assert tracker.cfg == expected_cfg
    assert tracker.cfg is not original_cfg
    assert tracker.rng is original_rng
    assert tracker.rng.bit_generator.state == original_rng_state
    assert tracker._last_t == 0.0
    assert tracker.tracks[0].age == original_age
    np.testing.assert_array_equal(tracker.tracks[0].filt.state.x, original_state.x)
    np.testing.assert_array_equal(tracker.tracks[0].filt.state.P, original_state.P)


def test_committed_track_state_timestamp_must_match_last_cycle():
    tracker = MultiSensorTracker(TrackerConfig(confirm_hits=1))
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    tracker.step([], 1.0)
    tracker.tracks[0].state_timestamp = 0.0

    with pytest.raises(ValueError, match="timestamp invariant"):
        tracker.all_outputs()


def test_lifecycle_corruption_is_rejected_before_copy_or_output():
    tracker = MultiSensorTracker(TrackerConfig(confirm_hits=1))
    tracker.step([Measurement("visual", [0, 0, 0], np.eye(3), 0.0)], 0.0)
    tracker.tracks[0].consecutive_misses = 1

    with pytest.raises(ValueError, match="consecutive-miss"):
        tracker.all_outputs()


def test_track_output_rejects_unbounded_or_inconsistent_state_before_copy():
    with pytest.raises(ValueError, match="shape"):
        TrackOutput(
            id=1,
            position=_zero_stride(shape=(2_000_000,), dtype=np.float32),
            velocity=np.zeros(3),
            covariance=np.eye(3),
            state="confirmed",
            age=1,
            class_label=None,
            state_timestamp=1.0,
            last_measurement_timestamp=1.0,
            updated_this_cycle=True,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        TrackOutput(
            id=1,
            position=np.zeros(3),
            velocity=np.zeros(3),
            covariance=np.eye(3),
            state="confirmed",
            age=1,
            class_label=None,
            state_timestamp=1.0,
            last_measurement_timestamp=2.0,
            updated_this_cycle=True,
        )


def test_truth_at_rejects_partial_nan_and_object_rows_without_coercion():
    scenario = Scenario(
        times=np.arange(1.0),
        truth=[np.array([[np.nan, 0.0, np.nan]])],
        frames=[[]],
    )
    with pytest.raises(ValueError, match="fully finite or fully NaN"):
        scenario.truth_at(0)

    object_truth = np.full((1, 3), _Coercive(), dtype=object)
    with pytest.raises(ValueError, match="real numeric"):
        Scenario(np.arange(1.0), [object_truth], [[]]).truth_at(0)


def test_score_rejects_invalid_callback_output_and_restores_tracker(monkeypatch):
    scenario = make_scenario(
        n_targets=0,
        duration=2.0,
        dt=0.5,
        modalities=("visual",),
        clutter_rate=0.0,
    )
    tracker = MultiSensorTracker()
    original_step = tracker.step

    def invalid_step(frame, timestamp):
        original_step(frame, timestamp)
        return [object()]

    monkeypatch.setattr(tracker, "step", invalid_step)
    with pytest.raises(TypeError, match="TrackOutput"):
        score_tracker(tracker, scenario)

    assert tracker.tracks == []
    assert tracker._next_id == 1
    assert tracker._last_t is None
