"""OSPA/GOSPA metrics and the audio/radar → fusion bridges."""

from itertools import permutations

import numpy as np
import pytest

from manwe.audio import detect_from_array, synth_plane_wave
from manwe.eval.benchmark import benchmark
from manwe.fusion import gospa, ospa, rmse
from manwe.fusion.association import associate, linear_assignment
from manwe.fusion.tracker import Measurement, measurement_cartesian


def test_ospa_zero_for_identical_sets():
    x = np.array([[0.0, 0, 0], [10, 10, 10]])
    assert ospa(x, x, c=20)["ospa"] == 0.0
    assert gospa(x, x, c=20)["gospa"] == 0.0


def test_ospa_all_cardinality_when_one_side_empty():
    x = np.array([[0.0, 0, 0]])
    d = ospa(x, np.empty((0, 3)), c=20)
    assert d["ospa"] == 20.0 and d["cardinality"] == 20.0 and d["localization"] == 0.0


def test_gospa_penalises_missed_target():
    truth = np.array([[0.0, 0, 0], [50, 0, 0]])
    est = np.array([[0.1, 0, 0]])  # second target missed
    d = gospa(truth, est, c=20, p=2.0, alpha=2.0)
    assert d["gospa"] > 0
    assert d["cardinality"] > d["localization"]  # dominated by the miss


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_gospa_can_leave_expensive_admissible_pairs_unassigned(p):
    truth = np.array([[0.0], [9.0]])
    estimate = np.array([[0.0], [-9.0]])
    result = gospa(truth, estimate, c=10.0, p=p, alpha=2.0)
    assert result["localization"] == 0.0
    assert result["cardinality"] == pytest.approx(10.0)
    assert result["gospa"] == pytest.approx(10.0)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
def test_gospa_general_alpha_applies_the_cutoff_to_equal_cardinality_sets(alpha):
    result = gospa(
        np.array([[0.0]]),
        np.array([[20.0]]),
        c=10.0,
        p=2.0,
        alpha=alpha,
    )
    assert result == pytest.approx({"gospa": 10.0, "localization": 10.0, "cardinality": 0.0})


def test_gospa_alpha_one_equals_unnormalized_ospa_on_bounded_point_sets():
    rng = np.random.default_rng(20260715)
    for p in (1.0, 2.0, 3.0):
        for truth_count in range(5):
            for estimate_count in range(5):
                truth = rng.normal(size=(truth_count, 3))
                estimate = rng.normal(size=(estimate_count, 3))
                normalized = ospa(truth, estimate, c=2.5, p=p)["ospa"]
                expected = normalized * max(truth_count, estimate_count, 1) ** (1.0 / p)
                assert gospa(truth, estimate, c=2.5, p=p, alpha=1.0)["gospa"] == pytest.approx(
                    expected
                )


def _gospa_definition_one(truth, estimate, *, c, p, alpha):
    if len(truth) > len(estimate):
        truth, estimate = estimate, truth
    smaller, larger = len(truth), len(estimate)
    assignment_cost = min(
        sum(
            min(float(np.linalg.norm(truth[index] - estimate[target])), c) ** p
            for index, target in enumerate(assignment)
        )
        for assignment in permutations(range(larger), smaller)
    )
    return (assignment_cost + c**p * (larger - smaller) / alpha) ** (1.0 / p)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_gospa_matches_primary_definition_on_small_bounded_sets(alpha, p):
    rng = np.random.default_rng(20260715)
    for truth_count in range(4):
        for estimate_count in range(4):
            truth = rng.normal(scale=8.0, size=(truth_count, 2))
            estimate = rng.normal(scale=8.0, size=(estimate_count, 2))
            expected = _gospa_definition_one(truth, estimate, c=2.5, p=p, alpha=alpha)
            assert gospa(truth, estimate, c=2.5, p=p, alpha=alpha)["gospa"] == pytest.approx(
                expected
            )


def test_numpy_assignment_is_globally_optimal_not_greedy():
    cost = np.array(
        [
            [16.741327761993155, 75.01494722940399],
            [39.84970452258166, 347.99881726624284],
        ]
    )
    assert linear_assignment(cost) == [(0, 1), (1, 0)]


def test_assignment_preserves_tiny_cost_differences_beside_huge_edges():
    cost = np.array([[0.0, 1e300, np.inf], [np.inf, 1e-100, 0.0], [np.inf, 0.0, 1e-100]])
    assert linear_assignment(cost) == [(0, 0), (1, 2), (2, 1)]


def test_assignment_preserves_subnormal_differences_beside_large_edges():
    for large in (1e4, 1e11):
        cost = np.array(
            [
                [0.0 if large == 1e4 else large, large if large == 1e4 else np.inf, np.inf],
                [np.inf, 4e-320 if large == 1e4 else 1e-320, 3e-320 if large == 1e4 else 0.0],
                [np.inf, 3e-320 if large == 1e4 else 0.0, 4e-320 if large == 1e4 else 1e-320],
            ]
        )
        assert linear_assignment(cost) == [(0, 0), (1, 2), (2, 1)]


def test_empty_association_accepts_diagonal_and_full_covariance_shapes():
    positions = np.empty((0, 3))
    assert associate([], positions, np.empty((0, 3))) == ([], [], [])
    assert associate([], positions, np.empty((0, 3, 3))) == ([], [], [])


def test_ospa_uses_exact_assignment_without_scipy():
    truth = np.array(
        [
            [1.288256422411882, -0.310021153074993, 0.0],
            [7.97647530496969, -8.279751278779948, 0.0],
        ]
    )
    estimate = np.array(
        [
            [3.9230890068178548, -3.4403542047079494, 0.0],
            [-6.4918050002983785, 3.495972999345578, 0.0],
        ]
    )
    assert abs(ospa(truth, estimate, c=20.0, p=2.0)["ospa"] - 7.578411830719734) < 1e-12


def test_set_metrics_reject_malformed_nonfinite_and_invalid_parameters():
    point = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="same dimensionality"):
        ospa(point, np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite coordinates"):
        ospa(point, np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="c must be"):
        ospa(point, point, c=0.0)
    with pytest.raises(ValueError, match="p must be"):
        gospa(point, point, p=0.5)
    with pytest.raises(ValueError, match="alpha"):
        gospa(point, point, alpha=3.0)
    with pytest.raises(ValueError, match="equal shapes"):
        rmse(point, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="must not be empty"):
        rmse(np.empty((0, 3)), np.empty((0, 3)))
    for malformed in (np.empty((2, 0)), np.empty((1, 0)), np.empty((0, 0, 2))):
        with pytest.raises(ValueError, match="shape"):
            ospa(malformed, np.empty((0, 3)))
    with pytest.raises(ValueError, match="dimension limit"):
        ospa(np.empty((0, 65)), np.empty((0, 65)))
    with pytest.raises(ValueError, match="underflows"):
        gospa(np.array([[0.0]]), np.array([[1e-200]]), c=1.0, p=200.0)


def test_gospa_large_order_ignores_safely_forbidden_far_pairs():
    result = gospa(
        np.array([[0.0], [1.0]]),
        np.array([[100.0], [200.0]]),
        c=1.0,
        p=200.0,
    )
    assert result["gospa"] == pytest.approx(2.0 ** (1.0 / 200.0))


def test_set_metrics_preserve_representable_subnormal_assignment_costs():
    smallest = np.nextafter(0.0, 1.0)
    single_truth = np.array([[0.0]])
    single_estimate = np.array([[smallest]])
    single = ospa(single_truth, single_estimate, c=1e308, p=1.0)
    assert single["ospa"] == smallest
    assert single["localization"] == smallest

    truth = np.array([[0.0], [2.0 * smallest]])
    estimate = truth[::-1].copy()
    assert ospa(truth, estimate, c=1e308, p=1.0)["ospa"] == 0.0
    assert gospa(truth, estimate, c=1e308, p=1.0)["gospa"] == 0.0


def test_set_metrics_handle_large_representable_coordinates_without_overflow():
    truth = np.array([[-1e150, 0.0]])
    estimate = np.array([[1e150, 0.0]])
    assert ospa(truth, estimate, c=20.0, p=2.0)["ospa"] == pytest.approx(20.0)
    assert gospa(truth, estimate, c=20.0, p=2.0)["gospa"] == pytest.approx(20.0)
    assert rmse(truth, estimate) == pytest.approx(2e150)

    maximum = np.finfo(np.float64).max
    extreme = np.array([[maximum, 0.0]])
    for metric in (ospa, gospa, rmse):
        with pytest.raises(ValueError, match="coordinate magnitude"):
            metric(extreme, -extreme)


def test_benchmark_validates_iteration_contract_and_runs_sync():
    calls = {"fn": 0, "sync": 0}

    def operation():
        calls["fn"] += 1

    def synchronize():
        calls["sync"] += 1

    result = benchmark(operation, warmup=2, iters=3, sync=synchronize)
    assert result.iters == 3 and calls == {"fn": 5, "sync": 8}
    with pytest.raises(ValueError, match="iters"):
        benchmark(operation, iters=0)
    with pytest.raises(ValueError, match="warmup"):
        benchmark(operation, warmup=-1)
    with pytest.raises(ValueError, match="call safety limit"):
        benchmark(operation, warmup=1, iters=1_000_000)


def test_radar_covariance_transform_is_psd():
    # 45° azimuth, some elevation, range 150 → cartesian cov must be PSD
    m = Measurement("radar", [150.0, np.pi / 4, 0.2], [9.0, 1e-4, 1e-4], 0.0)
    pos, cov = measurement_cartesian(m)
    assert np.linalg.norm(pos) > 0
    eig = np.linalg.eigvalsh(cov)
    assert np.all(eig > 0), f"radar cartesian covariance not PSD: {eig}"


def test_acoustic_detect_from_array_end_to_end():
    fs = 16000
    a = 0.1
    mics = np.array([[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]])
    az_true = np.radians(-70.0)
    el_true = np.radians(20.0)
    u = np.array(
        [
            np.cos(el_true) * np.cos(az_true),
            np.cos(el_true) * np.sin(az_true),
            np.sin(el_true),
        ]
    )
    signals = synth_plane_wave(u, mics, fs, duration=0.15, seed=3)
    det = detect_from_array(signals, mics, fs, class_label="drone")
    az_err = np.degrees(
        abs(np.arctan2(np.sin(det.azimuth - az_true), np.cos(det.azimuth - az_true)))
    )
    assert az_err < 15.0, f"azimuth error {az_err:.1f}°"
    # bridges into a Cartesian fusion measurement
    m = det.to_measurement(sensor_origin=np.zeros(3))
    assert m.modality == "acoustic" and m.class_label == "drone"
