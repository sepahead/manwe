"""OSPA/GOSPA metrics and the audio/radar → fusion bridges."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

import manwe.fusion.association as association_module
import manwe.fusion.metrics as metrics_module
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


def test_fusion_numeric_boundaries_reject_object_and_complex_without_coercion():
    float_calls = 0

    class FloatBomb:
        def __float__(self):
            nonlocal float_calls
            float_calls += 1
            raise AssertionError("object element coercion must not run")

    bomb = FloatBomb()
    object_matrix = np.array([[bomb]], dtype=object)
    complex_matrix = np.array([[1.0 + 0.0j]])

    for matrix in (object_matrix, complex_matrix):
        with pytest.raises(ValueError, match="real numeric"):
            linear_assignment(matrix)
        with pytest.raises(ValueError, match="real numeric"):
            ospa(matrix, np.zeros((1, 1)))
        with pytest.raises(ValueError, match="real numeric"):
            ospa(np.zeros((1, 1)), matrix)
        with pytest.raises(ValueError, match="real numeric"):
            associate([], matrix, np.ones((1, 1)))
        with pytest.raises(ValueError, match="real numeric"):
            associate([], np.zeros((1, 1)), matrix)

    with pytest.raises(ValueError, match="c must be"):
        ospa(np.zeros((1, 1)), np.zeros((1, 1)), c=bomb)  # type: ignore[arg-type]

    class Track:
        def gating_distance(self, _position, _covariance):
            return bomb

    with pytest.raises(ValueError, match="invalid gating distance"):
        associate([Track()], np.zeros((1, 1)), np.ones((1, 1)))
    assert float_calls == 0


def test_assignment_and_association_preflight_broadcast_views_before_widening(monkeypatch):
    def unexpected_widening(_array, _name):
        raise AssertionError("float64 widening ran before admission")

    monkeypatch.setattr(association_module, "_float64_array", unexpected_widening)

    oversized_cost = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (1, association_module.MAX_ASSIGNMENT_CELLS + 1),
    )
    assert oversized_cost.strides == (0, 0)
    with pytest.raises(ValueError, match="assignment-work"):
        linear_assignment(oversized_cost)

    oversized_positions = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (association_module.MAX_ASSOCIATION_ITEMS + 1, 40),
    )
    oversized_covariances = np.broadcast_to(
        np.array(1, dtype=np.int8),
        oversized_positions.shape,
    )
    with pytest.raises(ValueError, match="measurements exceed"):
        associate([], oversized_positions, oversized_covariances)

    positions = np.broadcast_to(np.array(0, dtype=np.int8), (1000, 64))
    diagonal_covariances = np.broadcast_to(np.array(1, dtype=np.int8), (1000, 64))
    with pytest.raises(ValueError, match="covariances exceed"):
        associate([], positions, diagonal_covariances)

    too_many_tracks = [None] * (association_module.MAX_ASSOCIATION_ITEMS + 1)
    with pytest.raises(ValueError, match="tracks exceed"):
        associate(too_many_tracks, np.empty((0, 1)), np.empty((0, 1)))


def test_set_metrics_preflight_all_broadcast_inputs_and_work_before_widening(monkeypatch):
    def unexpected_widening(_array, _name):
        raise AssertionError("float64 widening ran before aggregate admission")

    monkeypatch.setattr(metrics_module, "_float64_array", unexpected_widening)

    valid_truth = np.array([[0]], dtype=np.int8)
    oversized_estimate = np.broadcast_to(
        np.array(0, dtype=np.int8),
        (metrics_module._MAX_POINTS + 1, metrics_module._MAX_POINT_DIMENSION),
    )
    assert oversized_estimate.strides == (0, 0)
    with pytest.raises(ValueError, match="point safety limit"):
        ospa(valid_truth, oversized_estimate)

    rectangular = np.broadcast_to(np.array(0, dtype=np.int8), (400, 1))
    with pytest.raises(ValueError, match="assignment-work"):
        ospa(rectangular, rectangular)

    augmented = np.broadcast_to(np.array(0, dtype=np.int8), (200, 1))
    with pytest.raises(ValueError, match="assignment-work"):
        gospa(augmented, augmented, alpha=2.0)


def test_exact_integer_admission_prevents_uint64_objective_collapse():
    exact_limit = 2**53
    # The exact off-diagonal objective is one smaller. Legacy eager float64
    # conversion merged the two uint64 costs and selected the diagonal tie.
    counterexample = np.array(
        [[exact_limit + 1, exact_limit], [0, 0]],
        dtype=np.uint64,
    )
    with pytest.raises(ValueError, match="consecutive exact float64 range"):
        linear_assignment(counterexample)

    accepted_boundary = np.array(
        [[exact_limit, exact_limit - 1], [0, 0]],
        dtype=np.uint64,
    )
    assert linear_assignment(accepted_boundary) == [(0, 1), (1, 0)]

    # Every input remains individually exact here, but subtracting the negative
    # minimum in float64 merges the two first-row costs at 2**54. The solver
    # must detect that preprocessing loss and preserve the exact -1 objective.
    signed_boundary = np.array(
        [[exact_limit, exact_limit - 1], [-exact_limit, -exact_limit]],
        dtype=np.int64,
    )
    assert linear_assignment(signed_boundary) == [(0, 1), (1, 0)]

    truth = np.array([[exact_limit]], dtype=np.uint64)
    estimate = np.array([[exact_limit + 1]], dtype=np.uint64)
    # These points differ by exactly one, but eager float64 conversion made
    # them identical and previously returned zero OSPA/GOSPA/RMSE.
    for metric in (ospa, gospa, rmse):
        with pytest.raises(ValueError, match="consecutive exact float64 range"):
            metric(truth, estimate)
    assert rmse(truth, np.array([[exact_limit - 1]], dtype=np.uint64)) == 1.0

    with pytest.raises(ValueError, match="consecutive exact float64 range"):
        associate([], estimate, np.ones((1, 1), dtype=np.uint8))

    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        extended = np.longdouble(str(exact_limit + 1))
        extended_cost = np.array([[extended, exact_limit], [0, 0]], dtype=np.longdouble)
        with pytest.raises(ValueError, match="loses numeric precision"):
            linear_assignment(extended_cost)
        with pytest.raises(ValueError, match="loses numeric precision"):
            rmse(
                np.array([[np.longdouble(exact_limit)]], dtype=np.longdouble),
                np.array([[extended]], dtype=np.longdouble),
            )


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
    with pytest.raises(ValueError, match="independently observed range"):
        det.to_measurement(sensor_origin=np.zeros(3))

    # An independent ranging sensor makes the Cartesian bridge well-defined.
    det = replace(det, range_estimate=120.0, range_observed=True)
    m = det.to_measurement(sensor_origin=np.zeros(3))
    assert m.modality == "acoustic" and m.class_label == "drone"
    assert np.linalg.norm(m.position) == pytest.approx(120.0)
