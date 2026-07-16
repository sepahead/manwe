"""Multi-sensor tracker: coordinate contract, M-of-N lifecycle, multi-target OSPA."""

import copy

import numpy as np
import pytest

from manwe.fusion.association import associate
from manwe.fusion.scenarios import _constant_acceleration_step, make_scenario, score_tracker
from manwe.fusion.tracker import (
    Measurement,
    MultiSensorTracker,
    Track,
    TrackerConfig,
    measurement_cartesian,
    radar_polar_to_cartesian,
)


def test_radar_polar_is_cartesianised_correctly():
    # target at (100, 0, 0): range 100, azimuth 0, elevation 0.
    m = Measurement("radar", [100.0, 0.0, 0.0], [9.0, 4e-4, 4e-4], 0.0)
    p, cov = measurement_cartesian(m)
    assert np.allclose(p, [100.0, 0.0, 0.0], atol=1e-6)
    assert cov.shape == (3, 3)
    # round-trip helper
    assert np.allclose(
        radar_polar_to_cartesian(np.array([100.0, 0.0, 0.0]), np.zeros(3)), [100.0, 0.0, 0.0]
    )


def test_lidar_is_not_routed_through_polar():
    # A lidar centroid is Cartesian and must pass through untouched.
    m = Measurement("lidar", [12.0, -3.0, 44.0], [0.25, 0.25, 0.25], 0.0)
    p, cov = measurement_cartesian(m)
    assert np.allclose(p, [12.0, -3.0, 44.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": [1.0, 2.0]},
        {"position": [1.0, 2.0, np.nan]},
        {"covariance": [1.0, 2.0]},
        {"covariance": [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        {"covariance": [1.0, -1.0, 1.0]},
        {"covariance": [-1e6, 1e20, 1e20]},
        {"timestamp": np.nan},
        {"sensor_origin": [0.0, 0.0]},
        {"velocity": [1.0, 2.0]},
    ],
)
def test_measurement_rejects_malformed_numeric_contract(kwargs):
    values = {
        "modality": "visual",
        "position": [1.0, 2.0, 3.0],
        "covariance": [1.0, 1.0, 1.0],
        "timestamp": 0.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        Measurement(**values)


@pytest.mark.parametrize("radar_range", [-1.0, 0.0, 1e-7])
def test_measurement_rejects_singular_radar_range(radar_range):
    with pytest.raises(ValueError, match="range"):
        Measurement("radar", [radar_range, 0.0, 0.0], [1.0, 1e-3, 1e-3], 0.0)


@pytest.mark.parametrize("elevation", [-np.pi / 2, np.pi / 2])
def test_measurement_rejects_radar_azimuth_singularity(elevation):
    with pytest.raises(ValueError, match="singular"):
        Measurement("radar", [1.0, 0.0, elevation], [1.0, 1e-3, 1e-3], 0.0)


@pytest.mark.parametrize("elevation", [-np.pi / 2 - 1e-12, np.pi / 2 + 1e-12, 1e300])
def test_measurement_rejects_noncanonical_radar_elevation(elevation):
    with pytest.raises(ValueError, match="elevation"):
        Measurement("radar", [1.0, 0.0, elevation], [1.0, 1e-3, 1e-3], 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_a": -1.0},
        {"gate_chi2": 0.0},
        {"confirm_hits": 0},
        {"confirm_window": 0},
        {"confirm_hits": 6},
        {"coast_after_misses": 6},
        {"max_missed_in_window": 6},
        {"max_position_cov_volume": 0.0},
        {"max_tracks": 0},
        {"init_vel_var": 0.0},
        {"init_merge_dist": -1.0},
        {"max_dt": 0.0},
        {"max_prediction_gap": 0.0},
        {"max_substeps": 0},
        {"max_measurements": 0},
        {"n_particles": 0},
        {"confirm_window": 4097},
        {"max_tracks": 10_001},
        {"max_substeps": 10_001},
        {"max_measurements": 10_001},
        {"n_particles": 100_001},
        {"max_tracks": 1000, "max_measurements": 1000},
        {
            "filter": "particle",
            "max_tracks": 100,
            "max_measurements": 1,
            "n_particles": 20_001,
        },
        {"filter": "particle", "n_particles": 1},
        {"max_measurements": 10**1000},
    ],
)
def test_tracker_config_rejects_invalid_invariants(kwargs):
    with pytest.raises(ValueError):
        TrackerConfig(**kwargs)


def test_tracker_constructor_rejects_invalid_config_and_rng():
    with pytest.raises(TypeError, match="config"):
        MultiSensorTracker(config={})
    with pytest.raises(TypeError, match="rng"):
        MultiSensorTracker(rng="seed")
    config = TrackerConfig()
    with pytest.raises(AttributeError):
        config.max_measurements = 10**1000  # type: ignore[misc]


def test_association_is_global_one_to_one_within_a_modality():
    class PointTrack:
        def __init__(self, x):
            self.x = x

        def gating_distance(self, z, _R):
            return float((z[0] - self.x) ** 2)

    tracks = [PointTrack(0.0), PointTrack(10.0)]
    positions = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    covariances = np.repeat(np.eye(3)[None], 2, axis=0)
    matches, unmatched_tracks, unmatched_measurements = associate(
        tracks, positions, covariances, gate_chi2=100.0
    )
    assert matches == [(0, 0), (1, 1)]
    assert unmatched_tracks == []
    assert unmatched_measurements == []


def test_association_does_not_confuse_large_valid_costs_with_forbidden_edges():
    class LargeCostTrack:
        def gating_distance(self, _z, _covariance):
            return 2.0e12

    matches, unmatched_tracks, unmatched_measurements = associate(
        [LargeCostTrack()],
        np.array([[0.0, 0.0, 0.0]]),
        np.eye(3)[None],
        gate_chi2=3.0e12,
    )
    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_measurements == []


def test_track_initialization_uses_information_weighted_measurement_covariance():
    tracker = MultiSensorTracker(TrackerConfig(init_merge_dist=15.0))
    tracker.step(
        [
            Measurement("visual", [0.0, 0.0, 0.0], [100.0, 100.0, 100.0], 0.0),
            Measurement("lidar", [10.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0),
        ],
        0.0,
    )
    assert len(tracker.tracks) == 1
    state = tracker.tracks[0].filt.state
    assert np.allclose(state.position, [1000.0 / 101.0, 0.0, 0.0])
    assert np.allclose(state.P[:3, :3], np.eye(3) * (100.0 / 101.0))


def test_single_measurement_covariance_seeds_track():
    tracker = MultiSensorTracker()
    covariance = np.diag([4.0, 9.0, 16.0])
    tracker.step([Measurement("visual", [1.0, 2.0, 3.0], covariance, 0.0)], 0.0)
    assert np.allclose(tracker.tracks[0].filt.state.P[:3, :3], covariance)


def test_birth_clustering_never_merges_two_hits_from_the_same_modality():
    tracker = MultiSensorTracker(TrackerConfig(init_merge_dist=15.0))
    tracker.step(
        [
            Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone"),
            Measurement("visual", [10.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone"),
        ],
        0.0,
    )
    assert len(tracker.tracks) == 2


def test_birth_clustering_is_invariant_to_sensor_names_across_an_ambiguous_chain():
    def births(sensor_ids, order=(0, 1, 2)):
        tracker = MultiSensorTracker(
            TrackerConfig(confirm_hits=1, init_merge_dist=1.1, sigma_a=0.0)
        )
        measurements = [
            Measurement(
                "visual",
                [position, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                0.0,
                sensor_id=sensor_id,
            )
            for position, sensor_id in zip((0.0, 1.0, 2.0), sensor_ids)
        ]
        tracker.step(
            [measurements[index] for index in order],
            0.0,
        )
        return sorted(tuple(track.filt.state.position) for track in tracker.tracks)

    alphabetical = births(("a", "b", "c"))
    renamed = births(("b", "a", "c"))
    reordered = births(("c", "a", "b"), order=(2, 0, 1))
    assert (
        alphabetical
        == renamed
        == reordered
        == [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ]
    )


def test_birth_clustering_merges_a_complete_cross_sensor_component():
    tracker = MultiSensorTracker(TrackerConfig(confirm_hits=1, init_merge_dist=1.1))
    tracker.step(
        [
            Measurement(
                "visual",
                [position, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                0.0,
                sensor_id=sensor_id,
            )
            for position, sensor_id in zip((0.0, 0.5, 1.0), ("left", "middle", "right"))
        ],
        0.0,
    )
    assert len(tracker.tracks) == 1
    assert np.allclose(tracker.tracks[0].filt.state.position, [0.5, 0.0, 0.0])


def test_generator_measurements_are_materialized_once_and_revalidated():
    tracker = MultiSensorTracker()
    measurement = Measurement("visual", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.0)

    tracker.step((value for value in [measurement]), 0.0)

    assert len(tracker.tracks) == 1
    assert np.array_equal(tracker.tracks[0].filt.state.position, [1.0, 2.0, 3.0])

    invalid = Measurement("visual", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 1.0)
    invalid.covariance[0, 0] = -1.0
    with pytest.raises(ValueError, match="variance"):
        tracker.step(iter([invalid]), 1.0)
    assert tracker._last_t == 0.0


def test_measurement_count_limit_is_checked_before_state_mutation():
    tracker = MultiSensorTracker(TrackerConfig(max_measurements=1))
    frame = (
        Measurement("visual", [float(index), 0.0, 0.0], [1.0, 1.0, 1.0], 0.0) for index in range(2)
    )

    with pytest.raises(ValueError, match="measurement count"):
        tracker.step(frame, 0.0)

    assert tracker.tracks == []
    assert tracker._last_t is None
    assert tracker._next_id == 1


def test_class_aware_birth_and_association_never_merge_or_rewrite_classes():
    tracker = MultiSensorTracker(TrackerConfig(init_merge_dist=15.0, max_position_cov_volume=1e12))
    tracker.step(
        [
            Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone"),
            Measurement("visual", [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="bird"),
        ],
        0.0,
    )
    assert sorted(track.class_label for track in tracker.tracks) == ["bird", "drone"]

    drone = next(track for track in tracker.tracks if track.class_label == "drone")
    drone_position = drone.filt.state.position.copy()
    tracker.step(
        [Measurement("visual", drone_position, [1.0, 1.0, 1.0], 1.0, class_label="helicopter")],
        1.0,
    )

    assert sorted(track.class_label for track in tracker.tracks) == ["bird", "drone", "helicopter"]


def test_unclassified_track_cannot_accept_conflicting_cross_modality_classes():
    tracker = MultiSensorTracker(TrackerConfig(filter="kalman", sigma_a=0.0))
    tracker.step([Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)], 0.0)

    tracker.step(
        [
            Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, class_label="bird"),
            Measurement("lidar", [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, class_label="drone"),
        ],
        1.0,
    )

    assert sorted(track.class_label for track in tracker.tracks) == ["bird", "drone"]


def test_all_modalities_associate_against_the_same_predicted_prior():
    tracker = MultiSensorTracker(
        TrackerConfig(filter="kalman", sigma_a=0.0, gate_chi2=10.0, init_merge_dist=0.0)
    )
    tracker.step(
        [Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone")],
        0.0,
    )

    tracker.step(
        [
            Measurement("acoustic", [4.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1e-9, class_label="drone"),
            Measurement("visual", [5.5, 0.0, 0.0], [1.0, 1.0, 1.0], 1e-9, class_label="drone"),
        ],
        1e-9,
    )

    # Acoustic is in gate from the prior; visual is only in gate after that update.
    # A frozen association plan therefore leaves visual unmatched and births track 2.
    assert len(tracker.tracks) == 2
    assert np.isclose(tracker.tracks[0].filt.state.position[0], 2.0)
    assert np.isclose(tracker.tracks[1].filt.state.position[0], 5.5)


def test_imm_uses_configured_process_noise_scale():
    tracker = MultiSensorTracker(TrackerConfig(filter="imm", sigma_a=2.5))
    tracker.step([Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)], 0.0)

    assert [model.sigma_a for model in tracker.tracks[0].filt.models] == [2.5, 25.0]


def test_covariance_volume_lifecycle_is_log_domain_overflow_safe():
    tracker = MultiSensorTracker(
        TrackerConfig(
            confirm_hits=1,
            confirm_window=10,
            coast_after_misses=10,
            max_missed_in_window=10,
            max_position_cov_volume=1e300,
        )
    )
    tracker.step([Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)], 0.0)
    tracker.tracks[0].filt.state.P[:3, :3] = np.eye(3) * 1e200

    tracker.tracks[0].record(False)

    assert tracker.tracks[0].state == "lost"


def test_track_output_carries_cycle_and_measurement_freshness():
    tracker = MultiSensorTracker(
        TrackerConfig(confirm_hits=1, coast_after_misses=2, max_missed_in_window=5)
    )
    first = tracker.step([Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 4.0)], 4.0)[0]
    assert first.state_timestamp == 4.0
    assert first.last_measurement_timestamp == 4.0
    assert first.updated_this_cycle is True

    coast = tracker.step([], 5.0)[0]
    assert coast.state_timestamp == 5.0
    assert coast.last_measurement_timestamp == 4.0
    assert coast.updated_this_cycle is False
    assert coast.to_dict()["updated_this_cycle"] is False


def test_default_covariance_limit_allows_the_configured_first_miss():
    tracker = MultiSensorTracker(TrackerConfig(confirm_hits=1))
    tracker.step([Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)], 0.0)
    outputs = tracker.step([], tracker.cfg.max_dt)
    assert len(outputs) == 1
    assert outputs[0].state == "confirmed"


def test_radar_angles_are_canonical_and_extreme_values_fail_closed():
    wrapped = Measurement("radar", [10.0, 3.0 * np.pi, 0.0], [1.0, 0.1, 0.1], 0.0)
    assert wrapped.position[1] == -np.pi
    with pytest.raises(ValueError, match="elevation"):
        Measurement("radar", [10.0, 0.0, np.pi], [1.0, 0.1, 0.1], 0.0)
    with pytest.raises(ValueError, match="azimuth magnitude"):
        Measurement("radar", [10.0, 1e300, 0.0], [1.0, 0.1, 0.1], 0.0)


def test_tracker_integrates_long_gap_with_bounded_admission_and_rejects_time_reversal():
    tracker = MultiSensorTracker(
        TrackerConfig(
            confirm_hits=1,
            confirm_window=20,
            coast_after_misses=20,
            max_missed_in_window=20,
            max_position_cov_volume=1e100,
            max_dt=1.0,
        )
    )
    tracker.step([Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)], 0.0)
    tracker.tracks[0].filt.state.x[3] = 10.0
    tracker.step([], 10.0)
    assert abs(tracker.tracks[0].filt.state.position[0] - 100.0) < 1e-12
    with pytest.raises(ValueError, match="monotonic"):
        tracker.step([], 9.0)
    assert tracker._last_t == 10.0


@pytest.mark.parametrize("filter_name", ["kalman", "ekf", "ukf", "particle", "imm"])
def test_prediction_is_invariant_to_internal_gap_budget_partition(filter_name):
    """A safety-budget knob must not alter the discrete per-cycle filter model."""

    common = dict(
        filter=filter_name,
        sigma_a=1.5,
        confirm_hits=1,
        confirm_window=10,
        coast_after_misses=10,
        max_missed_in_window=10,
        max_position_cov_volume=1e100,
        max_prediction_gap=10.0,
        max_substeps=20,
        n_particles=128,
    )
    whole = MultiSensorTracker(TrackerConfig(max_dt=10.0, **common))
    partitioned_budget = MultiSensorTracker(TrackerConfig(max_dt=0.25, **common))
    measurement = Measurement(
        "visual",
        [3.0, -2.0, 7.0],
        [1.0, 2.0, 3.0],
        0.0,
        velocity=[4.0, -1.0, 0.5],
    )

    whole.step([measurement], 0.0)
    partitioned_budget.step([measurement], 0.0)
    whole.step([], 4.0)
    partitioned_budget.step([], 4.0)

    left = whole.tracks[0].filt.state
    right = partitioned_budget.tracks[0].filt.state
    np.testing.assert_array_equal(left.x, right.x)
    np.testing.assert_array_equal(left.P, right.P)
    if filter_name == "imm":
        np.testing.assert_array_equal(
            whole.tracks[0].filt.mode_probs,
            partitioned_budget.tracks[0].filt.mode_probs,
        )
        np.testing.assert_array_equal(
            whole.tracks[0].filt._cbar,
            partitioned_budget.tracks[0].filt._cbar,
        )
        for left_model, right_model in zip(
            whole.tracks[0].filt.models,
            partitioned_budget.tracks[0].filt.models,
            strict=True,
        ):
            np.testing.assert_array_equal(left_model.state.x, right_model.state.x)
            np.testing.assert_array_equal(left_model.state.P, right_model.state.P)
    if filter_name == "particle":
        np.testing.assert_array_equal(
            whole.tracks[0].filt.particles,
            partitioned_budget.tracks[0].filt.particles,
        )
        np.testing.assert_array_equal(
            whole.tracks[0].filt.weights,
            partitioned_budget.tracks[0].filt.weights,
        )
        assert (
            whole.tracks[0].filt.rng.bit_generator.state
            == partitioned_budget.tracks[0].filt.rng.bit_generator.state
        )


def test_tracker_rejects_same_timestamp_replays_without_advancing_lifecycle():
    tracker = MultiSensorTracker(
        TrackerConfig(
            confirm_hits=2,
            confirm_window=3,
            coast_after_misses=1,
            max_missed_in_window=3,
        )
    )
    measurement = Measurement(
        "visual",
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        1.0,
        class_label="drone",
    )
    tracker.step([measurement], 1.0)
    before = tracker.tracks[0]
    before_state = (
        before.age,
        list(before.hits),
        before.consecutive_misses,
        before.state,
        before.filt.state.x.copy(),
        before.filt.state.P.copy(),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.step([measurement], 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.step([], 1.0)

    after = tracker.tracks[0]
    assert after.age == before_state[0]
    assert list(after.hits) == before_state[1]
    assert after.consecutive_misses == before_state[2]
    assert after.state == before_state[3] == "tentative"
    assert np.array_equal(after.filt.state.x, before_state[4])
    assert np.array_equal(after.filt.state.P, before_state[5])
    assert tracker._last_t == 1.0


@pytest.mark.parametrize("class_label", ["line\nbreak", "x" * 257, "é" * 129])
def test_measurement_rejects_unbounded_or_nonprintable_class_labels(class_label):
    with pytest.raises(ValueError, match="bounded printable"):
        Measurement("visual", [0.0, 0.0, 0.0], np.eye(3), 0.0, class_label=class_label)

    accepted = Measurement(
        "visual",
        [0.0, 0.0, 0.0],
        np.eye(3),
        0.0,
        class_label="  " + "x" * 256 + "  ",
    )
    assert accepted.class_label == "x" * 256


@pytest.mark.parametrize(
    ("config", "timestamp", "message"),
    [
        (TrackerConfig(max_prediction_gap=1.0), 2.0, "prediction gap"),
        (
            TrackerConfig(max_prediction_gap=10.0, max_dt=1.0, max_substeps=2),
            3.0,
            "gap budget",
        ),
    ],
)
def test_prediction_gap_admission_bounds_reject_before_mutation(config, timestamp, message):
    tracker = MultiSensorTracker(config)
    tracker.step(
        [Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone")],
        0.0,
    )
    before_x = tracker.tracks[0].filt.state.x.copy()
    before_P = tracker.tracks[0].filt.state.P.copy()
    before_age = tracker.tracks[0].age

    with pytest.raises(ValueError, match=message):
        tracker.step([], timestamp)

    assert tracker._last_t == 0.0
    assert tracker.tracks[0].age == before_age
    assert np.array_equal(tracker.tracks[0].filt.state.x, before_x)
    assert np.array_equal(tracker.tracks[0].filt.state.P, before_P)


def test_prediction_gap_budget_accepts_its_floating_point_product_boundary():
    config = TrackerConfig(
        confirm_hits=1,
        confirm_window=10,
        coast_after_misses=10,
        max_missed_in_window=10,
        max_dt=0.1,
        max_substeps=3,
        max_prediction_gap=1.0,
    )
    tracker = MultiSensorTracker(config)
    tracker.step(
        [Measurement("visual", [0.0, 0.0, 0.0], np.eye(3), 0.0)],
        0.0,
    )

    boundary = config.max_dt * config.max_substeps
    tracker.step([], boundary)

    assert tracker._last_t == boundary


def test_failed_cycle_rolls_back_filter_lifecycle_ids_and_time():
    tracker = MultiSensorTracker()
    tracker.step(
        [Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone")],
        0.0,
    )
    track = tracker.tracks[0]
    before = {
        "x": track.filt.state.x.copy(),
        "P": track.filt.state.P.copy(),
        "age": track.age,
        "hits": list(track.hits),
        "misses": track.consecutive_misses,
        "state": track.state,
        "next_id": tracker._next_id,
        "last_t": tracker._last_t,
    }

    with pytest.raises((FloatingPointError, ValueError), match="gating|finite"):
        tracker.step(
            [
                Measurement("acoustic", [0.1, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, class_label="drone"),
                Measurement("visual", [1e308, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, class_label="drone"),
            ],
            1.0,
        )

    restored = tracker.tracks[0]
    assert np.array_equal(restored.filt.state.x, before["x"])
    assert np.array_equal(restored.filt.state.P, before["P"])
    assert restored.age == before["age"]
    assert list(restored.hits) == before["hits"]
    assert restored.consecutive_misses == before["misses"]
    assert restored.state == before["state"]
    assert tracker._next_id == before["next_id"]
    assert tracker._last_t == before["last_t"]


def test_failed_particle_cycle_restores_population_and_independent_rng():
    tracker = MultiSensorTracker(
        TrackerConfig(filter="particle", n_particles=32), rng=np.random.default_rng(42)
    )
    tracker.step(
        [Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, class_label="drone")],
        0.0,
    )
    particle_filter = tracker.tracks[0].filt
    before_particles = particle_filter.particles.copy()
    before_weights = particle_filter.weights.copy()
    before_tracker_rng = copy.deepcopy(tracker.rng.bit_generator.state)
    before_filter_rng = copy.deepcopy(particle_filter.rng.bit_generator.state)

    with pytest.raises((FloatingPointError, ValueError), match="gating|finite"):
        tracker.step(
            [Measurement("visual", [1e308, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, class_label="drone")],
            1.0,
        )

    restored_filter = tracker.tracks[0].filt
    assert np.array_equal(restored_filter.particles, before_particles)
    assert np.array_equal(restored_filter.weights, before_weights)
    assert tracker.rng.bit_generator.state == before_tracker_rng
    assert restored_filter.rng.bit_generator.state == before_filter_rng
    assert restored_filter.rng is not tracker.rng
    assert tracker._last_t == 0.0


def test_later_unrelated_birth_does_not_change_existing_particle_stream():
    def run(*, add_unrelated_birth):
        tracker = MultiSensorTracker(
            TrackerConfig(
                filter="particle",
                n_particles=32,
                confirm_hits=1,
                init_merge_dist=0.0,
            ),
            rng=np.random.default_rng(123),
        )
        tracker.step(
            [Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)],
            0.0,
        )
        initial = tracker.tracks[0].filt.particles.copy()
        births = (
            [Measurement("visual", [10_000.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.5)]
            if add_unrelated_birth
            else []
        )
        tracker.step(births, 0.5)
        tracker.step([], 1.0)
        return initial, tracker.tracks[0].filt.particles.copy()

    baseline_initial, baseline_predicted = run(add_unrelated_birth=False)
    extra_initial, extra_predicted = run(add_unrelated_birth=True)
    assert np.array_equal(baseline_initial, extra_initial)
    assert np.array_equal(baseline_predicted, extra_predicted)


def test_failure_after_births_rolls_back_tracks_ids_time_and_rng(monkeypatch):
    tracker = MultiSensorTracker(
        TrackerConfig(filter="particle", n_particles=16, confirm_hits=1),
        rng=np.random.default_rng(7),
    )
    before_rng = copy.deepcopy(tracker.rng.bit_generator.state)

    def fail_output(_track):
        raise RuntimeError("synthetic output failure")

    monkeypatch.setattr(Track, "output", fail_output)
    with pytest.raises(RuntimeError, match="synthetic output failure"):
        tracker.step(
            [Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0)],
            0.0,
        )

    assert tracker.tracks == []
    assert tracker._next_id == 1
    assert tracker._last_t is None
    assert tracker.rng.bit_generator.state == before_rng


def test_tracker_rejects_measurement_from_a_different_cycle():
    tracker = MultiSensorTracker()
    measurement = Measurement("visual", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.01)
    with pytest.raises(ValueError, match="differs from cycle"):
        tracker.step([measurement], 0.0)
    assert tracker._last_t is None


def test_modality_update_order_is_deterministic():
    def run(reverse):
        tracker = MultiSensorTracker(TrackerConfig(filter="ekf"))
        tracker.step([Measurement("visual", [100.0, 0.0, 0.0], [4.0, 4.0, 4.0], 0.0)], 0.0)
        frame = [
            Measurement("visual", [102.0, 0.5, 0.0], [4.0, 4.0, 4.0], 1.0),
            Measurement("radar", [102.0, 0.0, 0.0], [9.0, 1e-4, 1e-4], 1.0),
        ]
        tracker.step(list(reversed(frame)) if reverse else frame, 1.0)
        return tracker.tracks[0].filt.state

    forward = run(False)
    reverse = run(True)
    assert np.allclose(forward.x, reverse.x)
    assert np.allclose(forward.P, reverse.P)


def test_mofn_lifecycle_transitions():
    cfg = TrackerConfig(
        filter="kalman",
        confirm_hits=3,
        confirm_window=5,
        coast_after_misses=2,
        max_missed_in_window=5,
    )
    tr = MultiSensorTracker(cfg)
    pos = np.array([10.0, 5.0, 50.0])

    # 4 consecutive detections → confirmed
    for k in range(4):
        tr.step([Measurement("visual", pos, [4.0, 4.0, 9.0], float(k))], float(k))
    assert tr.tracks[0].state == "confirmed"

    # 2 consecutive misses → coasting
    tr.step([], 4.0)
    tr.step([], 5.0)
    assert tr.tracks[0].state == "coasting"

    # keep missing → eventually Lost and removed from the table
    for k in range(6, 12):
        tr.step([], float(k))
        if not tr.tracks:
            break
    assert tr.tracks == [], "track should be deleted after sustained misses"


def test_single_target_multimodal_track():
    rng = np.random.default_rng(0)
    tr = MultiSensorTracker(TrackerConfig(filter="ekf"))
    truth = np.array([0.0, 0.0, 60.0])
    vel = np.array([5.0, 0.0, 0.0])
    confirmed_seen = False
    for k in range(12):
        t = k * 0.5
        p = truth + vel * t
        frame = [
            Measurement("visual", p + rng.normal(0, [2, 2, 3]), [4.0, 4.0, 9.0], t),
        ]
        outs = tr.step(frame, t)
        confirmed_seen = confirmed_seen or any(o.state == "confirmed" for o in outs)
    assert confirmed_seen
    # final estimate near truth
    final = tr.all_outputs()[0]
    assert np.linalg.norm(final.position - (truth + vel * (11 * 0.5))) < 8.0


def test_multitarget_scenario_ospa_reasonable():
    scenario = make_scenario(
        n_targets=3,
        duration=20.0,
        dt=0.5,
        modalities=("visual", "radar"),
        p_detect=0.9,
        clutter_rate=0.5,
        seed=7,
    )
    tr = MultiSensorTracker(TrackerConfig(filter="ekf"))
    score = score_tracker(tr, scenario, c=20.0, p=2.0)
    # With per-measurement association + initiation clustering the tracker recovers
    # both position (low localization) and target count (low cardinality).
    assert score["ospa"] < 8.0, f"OSPA {score['ospa']:.2f} too high (tracker not locking on)"
    assert score["cardinality"] < 6.0, f"cardinality {score['cardinality']:.2f}: wrong track count"


def test_short_scenario_cannot_report_a_perfect_empty_score():
    scenario = make_scenario(duration=1.0, dt=0.5, modalities=("visual",), clutter_rate=0.0)
    with pytest.raises(ValueError, match="at least one frame is scored"):
        score_tracker(MultiSensorTracker(), scenario)


def test_scenario_motion_uses_the_exact_constant_acceleration_step():
    position, velocity = _constant_acceleration_step(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([0.5, -1.0, 2.0]),
        2.0,
    )
    assert np.array_equal(position, [10.0, 10.0, 19.0])
    assert np.array_equal(velocity, [5.0, 3.0, 10.0])


def test_scenario_generation_rejects_invalid_boundaries():
    with pytest.raises(ValueError, match="dt"):
        make_scenario(dt=0.0)
    with pytest.raises(ValueError, match="p_detect"):
        make_scenario(p_detect=1.1)
    with pytest.raises(ValueError, match="unknown modalities"):
        make_scenario(modalities=("sonar",))
    with pytest.raises(ValueError, match="duplicates"):
        make_scenario(modalities=("visual", "visual"))
    with pytest.raises(ValueError, match="sensor_origin"):
        make_scenario(sensor_origin=np.array([0.0, np.nan, 0.0]))


def test_tiny_scenario_dt_uses_exact_bounded_frame_indices():
    scenario = make_scenario(
        n_targets=0,
        duration=1e-9,
        dt=1e-12,
        modalities=("visual",),
        clutter_rate=0.0,
    )

    assert len(scenario.times) == 1001
    assert len(scenario.frames) == 1001
    assert scenario.times[0] == 0.0
    assert np.isclose(scenario.times[-1], 1e-9, rtol=0.0, atol=1e-24)
    assert np.all(np.diff(scenario.times) > 0)


def test_score_tracker_rolls_back_if_a_later_frame_fails(monkeypatch):
    scenario = make_scenario(
        n_targets=1,
        duration=3.0,
        dt=0.5,
        modalities=("visual",),
        clutter_rate=0.0,
        seed=19,
    )
    tracker = MultiSensorTracker()
    original_step = tracker.step
    calls = 0

    def fail_late(frame, timestamp):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected late scoring failure")
        return original_step(frame, timestamp)

    monkeypatch.setattr(tracker, "step", fail_late)
    with pytest.raises(RuntimeError, match="late scoring failure"):
        score_tracker(tracker, scenario)
    assert tracker.tracks == []
    assert tracker._next_id == 1
    assert tracker._last_t is None


def test_association_rejects_bad_covariance_and_gate_results():
    class Track:
        def gating_distance(self, _position, _covariance):
            return np.nan

    position = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        associate([Track()], position, np.array([np.diag([1.0, -1.0, 1.0])]))
    with pytest.raises(ValueError, match="invalid gating distance"):
        associate([Track()], position, np.array([np.eye(3)]))
