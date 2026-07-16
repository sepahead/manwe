"""Filter bank: every filter should lock onto a constant-velocity target."""

import copy
import warnings

import numpy as np
import pytest

from manwe.fusion.filters import (
    ExtendedKalmanFilter,
    GaussianState,
    IMMEstimator,
    KalmanFilter,
    ParticleFilter,
    UnscentedKalmanFilter,
    cv_process_noise,
    cv_transition,
    position_measurement_matrix,
    wrap_angle,
)


def _cv_truth(steps=40, dt=0.5, x0=(0.0, 0.0, 60.0), v=(10.0, 2.0, 0.0)):
    x0 = np.array(x0)
    v = np.array(v)
    return np.array([x0 + v * (dt * k) for k in range(steps)]), np.array(v), dt


def _init_state(first_meas):
    x0 = np.zeros(6)
    x0[:3] = first_meas
    P0 = np.diag([25.0, 25.0, 25.0, 100.0, 100.0, 100.0])
    return x0, P0


def test_kalman_locks_on_cv_target():
    rng = np.random.default_rng(0)
    truth, v, dt = _cv_truth()
    R = np.diag([4.0, 4.0, 9.0])
    meas0 = truth[0] + rng.normal(0, [2, 2, 3])
    kf = KalmanFilter(*_init_state(meas0), sigma_a=2.0)
    for k in range(1, len(truth)):
        kf.predict(dt)
        z = truth[k] + rng.normal(0, [2, 2, 3])
        kf.update(z, R)
    err = np.linalg.norm(kf.state.position - truth[-1])
    vel_err = np.linalg.norm(kf.state.velocity - v)
    assert err < 5.0, f"position error {err:.2f} m too high"
    assert vel_err < 3.0, f"velocity error {vel_err:.2f} m/s too high"
    # covariance must have shrunk well below the initialisation
    assert np.trace(kf.state.P[:3, :3]) < 30.0


def test_ekf_polar_radar_update():
    rng = np.random.default_rng(1)
    truth, v, dt = _cv_truth(x0=(60.0, 20.0, 40.0), v=(6.0, 1.0, 0.0))
    origin = np.zeros(3)
    noise = np.array([3.0, 0.02, 0.02])
    R = np.diag(noise**2)

    def to_polar(p):
        d = p - origin
        rho = np.hypot(d[0], d[1])
        return np.array([np.linalg.norm(d), np.arctan2(d[1], d[0]), np.arctan2(d[2], rho)])

    z0 = to_polar(truth[0])
    from manwe.fusion.tracker import radar_polar_to_cartesian

    ekf = ExtendedKalmanFilter(*_init_state(radar_polar_to_cartesian(z0, origin)), sigma_a=2.0)
    for k in range(1, len(truth)):
        ekf.predict(dt)
        z = to_polar(truth[k]) + rng.normal(0, noise)
        ekf.update_polar(z, R, origin)
    err = np.linalg.norm(ekf.state.position - truth[-1])
    assert err < 15.0, f"EKF polar position error {err:.2f} m too high"


def test_ukf_locks_on_cv_target():
    rng = np.random.default_rng(2)
    truth, v, dt = _cv_truth()
    R = np.diag([4.0, 4.0, 9.0])
    ukf = UnscentedKalmanFilter(*_init_state(truth[0] + rng.normal(0, [2, 2, 3])), sigma_a=2.0)
    for k in range(1, len(truth)):
        ukf.predict(dt)
        ukf.update(truth[k] + rng.normal(0, [2, 2, 3]), R)
    assert np.linalg.norm(ukf.state.position - truth[-1]) < 6.0


def test_particle_filter_locks_on_cv_target():
    rng = np.random.default_rng(3)
    truth, v, dt = _cv_truth()
    R = np.diag([4.0, 4.0, 9.0])
    pf = ParticleFilter(*_init_state(truth[0]), sigma_a=2.0, n_particles=1000, rng=rng)
    for k in range(1, len(truth)):
        pf.predict(dt)
        pf.update(truth[k] + rng.normal(0, [2, 2, 3]), R)
    assert np.linalg.norm(pf.state.position - truth[-1]) < 8.0


def test_imm_bank_locks_and_probs_normalised():
    rng = np.random.default_rng(4)
    truth, v, dt = _cv_truth()
    R = np.diag([4.0, 4.0, 9.0])
    imm = IMMEstimator.default_cv_bank(*_init_state(truth[0]))
    for k in range(1, len(truth)):
        imm.predict(dt)
        imm.update(truth[k] + rng.normal(0, [2, 2, 3]), R)
    assert np.linalg.norm(imm.state.position - truth[-1]) < 6.0
    assert abs(imm.mode_probs.sum() - 1.0) < 1e-9


def test_imm_accumulates_sequential_measurement_evidence():
    class FixedLikelihoodModel:
        def __init__(self, likelihood):
            self.state = GaussianState(np.zeros(6), np.eye(6))
            self.likelihood = likelihood
            self.dim = 3

        def predict(self, _dt):
            pass

        def update(self, _z, _R):
            pass

    imm = IMMEstimator(
        [FixedLikelihoodModel(0.9), FixedLikelihoodModel(0.1)],
        transition=np.eye(2),
        mode_probs=np.array([0.5, 0.5]),
    )
    imm.update(np.zeros(3), np.eye(3))
    assert np.allclose(imm.mode_probs, [0.9, 0.1])
    imm.update(np.zeros(3), np.eye(3))
    assert np.allclose(imm.mode_probs, [0.9878048780487805, 0.012195121951219514])


@pytest.mark.parametrize("dim", [0, -1, 1.5, True, None])
@pytest.mark.parametrize(
    "builder",
    [
        lambda dim: cv_transition(1.0, dim),
        lambda dim: cv_process_noise(1.0, 1.0, dim),
        position_measurement_matrix,
    ],
    ids=["transition", "process-noise", "measurement"],
)
def test_motion_models_reject_invalid_dimensions(builder, dim):
    with pytest.raises(ValueError):
        builder(dim)


@pytest.mark.parametrize("dt", [-1.0, np.nan, np.inf, -np.inf, True, "1.0"])
def test_motion_models_reject_invalid_time_steps(dt):
    with pytest.raises(ValueError):
        cv_transition(dt)
    with pytest.raises(ValueError):
        cv_process_noise(dt, 1.0)


@pytest.mark.parametrize("sigma_a", [-1.0, np.nan, np.inf, -np.inf, True, "1.0"])
def test_process_noise_rejects_invalid_acceleration_noise(sigma_a):
    with pytest.raises(ValueError):
        cv_process_noise(1.0, sigma_a)


def test_process_noise_rejects_finite_inputs_that_overflow():
    with pytest.raises(ValueError):
        cv_process_noise(1e200, 1.0)
    with pytest.raises(ValueError):
        cv_process_noise(1.0, 1e200)


def test_process_noise_avoids_cancellable_intermediate_overflow():
    Q = cv_process_noise(1e100, 1e-200)
    assert np.isfinite(Q).all()
    assert Q[0, 0] == pytest.approx(0.25)
    assert Q[3, 3] == pytest.approx(1e-200)
    assert np.array_equal(cv_process_noise(1e308, 0.0), np.zeros((6, 6)))


def test_motion_model_outputs_are_finite_and_psd():
    F = cv_transition(0.25, dim=2)
    Q = cv_process_noise(0.25, 2.0, dim=2)
    H = position_measurement_matrix(dim=2)
    assert F.shape == (4, 4)
    assert H.shape == (2, 4)
    assert np.isfinite(Q).all()
    assert np.allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() >= -1e-12
    assert np.array_equal(cv_process_noise(0.0, 2.0, dim=2), np.zeros((4, 4)))


def test_wrap_angle_has_a_finite_explicit_boundary_contract():
    assert isinstance(wrap_angle(np.pi), float)
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert np.allclose(
        wrap_angle(np.array([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi])),
        -np.pi * np.array([1.0, 1.0, 0.0, 1.0, 1.0]),
    )
    for invalid in (np.nan, np.inf, -np.inf, np.array([0.0, np.nan])):
        with pytest.raises(ValueError):
            wrap_angle(invalid)

    with pytest.raises(ValueError, match="canonicalization limit"):
        wrap_angle(1_000_001.0)


def test_gaussian_state_copies_inputs_and_repairs_roundoff_only():
    x = np.arange(6.0)
    P = np.eye(6)
    state = GaussianState(x, P)
    x[:] = -1
    P[:] = 0
    assert np.array_equal(state.x, np.arange(6.0))
    assert np.array_equal(state.P, np.eye(6))

    roundoff_P = np.eye(6)
    roundoff_P[0, 1] = 1.0 + 1e-14
    roundoff_P[1, 0] = 1.0 + 1e-14
    repaired = GaussianState(np.zeros(6), roundoff_P)
    assert np.linalg.eigvalsh(repaired.P).min() >= 0

    negative_variance = np.eye(6)
    negative_variance[-1, -1] = -1e-14
    with pytest.raises(ValueError, match="variance"):
        GaussianState(np.zeros(6), negative_variance)

    two_dimensional = GaussianState(np.arange(4.0), np.eye(4))
    assert np.array_equal(two_dimensional.position, [0.0, 1.0])
    assert np.array_equal(two_dimensional.velocity, [2.0, 3.0])


@pytest.mark.parametrize(
    "x",
    [np.array([]), np.zeros((6, 1)), np.array([0.0, np.nan]), np.array([0.0, np.inf])],
)
def test_gaussian_state_rejects_invalid_vectors(x):
    with pytest.raises(ValueError):
        GaussianState(x, np.eye(x.size or 1))


@pytest.mark.parametrize(
    "P",
    [
        np.eye(5),
        np.full((6, 6), np.nan),
        np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
        np.eye(6) + np.triu(np.ones((6, 6)), 1),
        [["not", "numeric"]],
    ],
)
def test_gaussian_state_rejects_invalid_covariances(P):
    with pytest.raises(ValueError):
        GaussianState(np.zeros(6), P)


@pytest.mark.parametrize(
    ("x", "P"),
    [
        (np.ones(6, dtype=bool), np.eye(6)),
        (np.ones(6, dtype=complex), np.eye(6)),
        (np.zeros(6), np.eye(6, dtype=bool)),
        (np.zeros(6), np.eye(6, dtype=complex)),
    ],
)
def test_gaussian_state_requires_real_numeric_not_boolean_arrays(x, P):
    with pytest.raises(ValueError, match="real numeric"):
        GaussianState(x, P)


def test_large_scale_covariance_checks_do_not_hide_material_errors():
    valid = np.eye(6) * 1e20
    assert np.array_equal(GaussianState(np.zeros(6), valid).P, valid)

    indefinite = valid.copy()
    indefinite[-1, -1] = -1e6
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianState(np.zeros(6), indefinite)

    asymmetric = np.eye(6) * 1e12
    asymmetric[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        GaussianState(np.zeros(6), asymmetric)


_COMMON_FILTER_FACTORIES = [
    lambda x, P, sigma_a=1.0: KalmanFilter(x, P, sigma_a=sigma_a),
    lambda x, P, sigma_a=1.0: ExtendedKalmanFilter(x, P, sigma_a=sigma_a),
    lambda x, P, sigma_a=1.0: UnscentedKalmanFilter(x, P, sigma_a=sigma_a),
    lambda x, P, sigma_a=1.0: ParticleFilter(
        x, P, sigma_a=sigma_a, n_particles=32, rng=np.random.default_rng(100)
    ),
]


@pytest.mark.parametrize("factory", _COMMON_FILTER_FACTORIES, ids=["kf", "ekf", "ukf", "particle"])
@pytest.mark.parametrize("sigma_a", [-1.0, np.nan, np.inf, -np.inf, True])
def test_filter_constructors_reject_invalid_process_noise(factory, sigma_a):
    with pytest.raises(ValueError):
        factory(np.zeros(6), np.eye(6), sigma_a)


@pytest.mark.parametrize("factory", _COMMON_FILTER_FACTORIES, ids=["kf", "ekf", "ukf", "particle"])
def test_filter_constructors_reject_invalid_state_shapes_and_covariance(factory):
    with pytest.raises(ValueError):
        factory(np.zeros(5), np.eye(6))
    with pytest.raises(ValueError):
        factory(np.zeros(6), np.eye(5))
    indefinite = np.eye(6)
    indefinite[-1, -1] = -0.1
    with pytest.raises(ValueError):
        factory(np.zeros(6), indefinite)


@pytest.mark.parametrize("dim", [0, -1, 1.5, True])
def test_filter_constructors_reject_invalid_dimensions(dim):
    with pytest.raises(ValueError):
        KalmanFilter(np.zeros(6), np.eye(6), dim=dim)
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(np.zeros(6), np.eye(6), dim=dim)
    with pytest.raises(ValueError):
        ParticleFilter(np.zeros(6), np.eye(6), dim=dim)


def _predict_factories():
    return [
        KalmanFilter(np.zeros(6), np.eye(6)),
        ExtendedKalmanFilter(np.zeros(6), np.eye(6)),
        UnscentedKalmanFilter(np.zeros(6), np.eye(6)),
        ParticleFilter(np.zeros(6), np.eye(6), n_particles=32, rng=np.random.default_rng(101)),
        IMMEstimator.default_cv_bank(np.zeros(6), np.eye(6)),
    ]


@pytest.mark.parametrize("index", range(5), ids=["kf", "ekf", "ukf", "particle", "imm"])
def test_predict_rejects_bad_dt_without_mutating_state(index):
    estimator = _predict_factories()[index]
    before = estimator.state.copy()
    for dt in (-1.0, np.nan, np.inf, -np.inf, True):
        with pytest.raises(ValueError):
            estimator.predict(dt)
        assert np.array_equal(estimator.state.x, before.x)
        assert np.array_equal(estimator.state.P, before.P)
    estimator.predict(0.0)
    assert np.array_equal(estimator.state.x, before.x)
    assert np.array_equal(estimator.state.P, before.P)


def _update_factories():
    return [
        KalmanFilter(np.zeros(6), np.eye(6)),
        ExtendedKalmanFilter(np.zeros(6), np.eye(6)),
        UnscentedKalmanFilter(np.zeros(6), np.eye(6)),
        ParticleFilter(np.zeros(6), np.eye(6), n_particles=32, rng=np.random.default_rng(102)),
        IMMEstimator.default_cv_bank(np.zeros(6), np.eye(6)),
    ]


@pytest.mark.parametrize("index", range(5), ids=["kf", "ekf", "ukf", "particle", "imm"])
def test_cartesian_updates_reject_bad_measurements_without_mutation(index):
    estimator = _update_factories()[index]
    before = estimator.state.copy()
    asymmetric = np.eye(3)
    asymmetric[0, 1] = 0.25
    indefinite = np.eye(3)
    indefinite[-1, -1] = -1.0
    cases = [
        (np.zeros(2), np.eye(3)),
        (np.zeros((3, 1)), np.eye(3)),
        (np.array([0.0, np.nan, 0.0]), np.eye(3)),
        (np.zeros(3), np.eye(2)),
        (np.zeros(3), asymmetric),
        (np.zeros(3), indefinite),
        (np.zeros(3), np.full((3, 3), np.inf)),
    ]
    for z, R in cases:
        with pytest.raises(ValueError):
            estimator.update(z, R)
        assert np.array_equal(estimator.state.x, before.x)
        assert np.array_equal(estimator.state.P, before.P)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: KalmanFilter(np.zeros(6), np.zeros((6, 6)), sigma_a=0.0),
        lambda: UnscentedKalmanFilter(np.zeros(6), np.zeros((6, 6)), sigma_a=0.0),
        lambda: IMMEstimator([KalmanFilter(np.zeros(6), np.zeros((6, 6)), sigma_a=0.0)]),
    ],
    ids=["kf", "ukf", "imm"],
)
def test_singular_innovation_covariance_is_rejected_transactionally(factory):
    estimator = factory()
    before = estimator.state.copy()
    with pytest.raises(ValueError, match="positive definite"):
        estimator.update(np.zeros(3), np.zeros((3, 3)))
    assert np.array_equal(estimator.state.x, before.x)
    assert np.array_equal(estimator.state.P, before.P)
    with pytest.raises(ValueError, match="positive definite"):
        estimator.gating_distance(np.zeros(3), np.zeros((3, 3)))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: KalmanFilter(np.zeros(6), np.eye(6)),
        lambda: UnscentedKalmanFilter(np.zeros(6), np.eye(6)),
    ],
    ids=["kf", "ukf"],
)
def test_nonfinite_likelihood_math_does_not_partially_apply_update(factory):
    estimator = factory()
    before = estimator.state.copy()
    with pytest.raises(FloatingPointError):
        estimator.update(np.full(3, 1e308), np.eye(3))
    assert np.array_equal(estimator.state.x, before.x)
    assert np.array_equal(estimator.state.P, before.P)
    with pytest.raises(FloatingPointError):
        estimator.gating_distance(np.full(3, 1e308), np.eye(3))


def test_ekf_polar_boundaries_and_origin_singularity_fail_explicitly():
    ekf = ExtendedKalmanFilter(np.zeros(6), np.eye(6))
    R = np.diag([1.0, 0.01, 0.01])
    before = ekf.state.copy()
    with pytest.raises(ValueError, match="range"):
        ekf.update_polar(np.zeros(3), R, np.zeros(3))
    with pytest.raises(ValueError, match="singular"):
        ekf.update_polar(np.array([100.0, 0.0, 0.0]), R, np.zeros(3))
    assert np.array_equal(ekf.state.x, before.x)
    assert np.array_equal(ekf.state.P, before.P)

    invalid_calls = [
        (np.array([-1.0, 0.0, 0.0]), R, None),
        (np.array([1.0, np.nan, 0.0]), R, None),
        (np.zeros(2), R, None),
        (np.zeros(3), R, np.zeros(2)),
        (np.zeros(3), R, np.array([0.0, np.inf, 0.0])),
    ]
    for z, covariance, origin in invalid_calls:
        with pytest.raises(ValueError):
            ekf.update_polar(z, covariance, origin)

    two_dimensional = ExtendedKalmanFilter(np.zeros(4), np.eye(4), dim=2)
    with pytest.raises(ValueError, match="dim=3"):
        two_dimensional.update_polar(np.zeros(3), np.eye(3))


def test_ekf_polar_geometry_avoids_large_coordinate_overflow():
    position = np.array([1e150, -1e150, 1e150])
    x = np.concatenate([position, np.zeros(3)])
    ekf = ExtendedKalmanFilter(x, np.eye(6))
    rho = np.hypot(position[0], position[1])
    z = np.array(
        [
            np.hypot(rho, position[2]),
            np.arctan2(position[1], position[0]),
            np.arctan2(position[2], rho),
        ]
    )
    ekf.update_polar(z, np.eye(3))
    assert np.isfinite(ekf.state.x).all()
    assert np.isfinite(ekf.state.P).all()

    overflowing_origin = np.array([-1e308, 0.0, 0.0])
    overflow_ekf = ExtendedKalmanFilter(np.array([1e308, 0.0, 0.0, 0.0, 0.0, 0.0]), np.eye(6))
    with pytest.raises(FloatingPointError, match="relative radar position"):
        overflow_ekf.update_polar(np.array([1.0, 0.0, 0.0]), np.eye(3), overflowing_origin)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": -1.0},
        {"alpha": np.nan},
        {"alpha": np.inf},
        {"beta": -1.0},
        {"beta": np.nan},
        {"kappa": np.nan},
        {"kappa": np.inf},
        {"kappa": -6.0},
        {"alpha": 1e-12},
    ],
)
def test_ukf_rejects_invalid_or_unstable_scaling_parameters(kwargs):
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(np.zeros(6), np.eye(6), **kwargs)


def test_ukf_accepts_small_but_still_resolvable_scaling():
    ukf = UnscentedKalmanFilter(np.zeros(6), np.eye(6), alpha=1e-4)
    ukf.predict(0.1)
    _assert_valid_gaussian(ukf.state)


def test_ukf_preserves_an_exact_singular_covariance_without_fake_jitter():
    ukf = UnscentedKalmanFilter(np.zeros(6), np.zeros((6, 6)), sigma_a=0.0)
    ukf.predict(1.0)
    assert np.array_equal(ukf.state.P, np.zeros((6, 6)))
    ukf.update(np.zeros(3), np.eye(3))
    assert np.array_equal(ukf.state.x, np.zeros(6))
    assert np.array_equal(ukf.state.P, np.zeros((6, 6)))


@pytest.mark.parametrize("n_particles", [0, -1, 1, 1.5, True, 100_001])
def test_particle_filter_rejects_invalid_particle_counts(n_particles):
    with pytest.raises(ValueError):
        ParticleFilter(np.zeros(6), np.eye(6), n_particles=n_particles)


def test_particle_filter_rejects_invalid_rng_type():
    with pytest.raises(TypeError):
        ParticleFilter(np.zeros(6), np.eye(6), rng=np.random.RandomState(0))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("particles", np.zeros((15, 6))),
        ("particles", np.full((16, 6), np.inf)),
        ("weights", np.full(15, 1.0 / 15.0)),
        ("weights", np.full(16, np.nan)),
        ("weights", np.full(16, 0.01)),
        ("weights", np.array([-0.1] + [1.1 / 15.0] * 15)),
    ],
)
def test_particle_filter_rejects_corrupted_population(attribute, value):
    pf = ParticleFilter(np.zeros(6), np.eye(6), n_particles=16, rng=np.random.default_rng(103))
    setattr(pf, attribute, value)
    with pytest.raises(ValueError):
        _ = pf.state


def test_particle_update_is_stable_in_log_space_and_requires_pd_noise():
    pf = ParticleFilter(
        np.zeros(6), np.zeros((6, 6)), n_particles=32, rng=np.random.default_rng(104)
    )
    original_weights = pf.weights.copy()
    with pytest.raises(ValueError, match="positive definite"):
        pf.update(np.zeros(3), np.zeros((3, 3)))
    assert np.array_equal(pf.weights, original_weights)

    pf.update(np.full(3, 1e6), np.eye(3))
    assert np.isfinite(pf.log_likelihood)
    assert np.isfinite(pf.likelihood) and pf.likelihood > 0
    assert np.isfinite(pf.weights).all()
    assert pf.weights.sum() == pytest.approx(1.0)


def test_failed_particle_prediction_restores_rng_and_population():
    rng = np.random.default_rng(107)
    pf = ParticleFilter(np.zeros(6), np.eye(6), n_particles=16, rng=rng)
    pf.particles[:] = np.finfo(float).max
    before_particles = pf.particles.copy()
    before_rng = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(FloatingPointError, match="predicted particles"):
        pf.predict(1.0)
    assert np.array_equal(pf.particles, before_particles)
    assert rng.bit_generator.state == before_rng


def test_particle_filter_valid_operations_are_runtime_warning_free():
    P0 = np.diag([2.0, 0.5, 4.0, 100.0, 100.0, 100.0])
    P0[0, 1] = P0[1, 0] = 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pf = ParticleFilter(np.zeros(6), P0, n_particles=128, rng=np.random.default_rng(106))
        _ = pf.state
        pf.predict(0.5)
        pf.update(np.ones(3), np.eye(3))
        _ = pf.state


def _assert_valid_gaussian(state):
    assert np.isfinite(state.x).all()
    assert np.isfinite(state.P).all()
    assert np.allclose(state.P, state.P.T, rtol=1e-10, atol=1e-12)
    assert np.linalg.eigvalsh(state.P).min() >= -1e-10


@pytest.mark.parametrize("index", range(5), ids=["kf", "ekf", "ukf", "particle", "imm"])
def test_filter_covariances_remain_psd_over_repeated_cycles(index):
    estimator = _update_factories()[index]
    for step in range(12):
        estimator.predict(0.2)
        estimator.update(np.array([0.1 * step, -0.05 * step, 0.0]), np.eye(3))
        _assert_valid_gaussian(estimator.state)
    if isinstance(estimator, IMMEstimator):
        for model in estimator.models:
            _assert_valid_gaussian(model.state)


class _LikelihoodModel:
    def __init__(
        self,
        likelihood=1.0,
        *,
        dim=3,
        next_likelihood=None,
        fail_predict=False,
        fail_update=False,
    ):
        self.dim = dim
        self.state = GaussianState(np.zeros(2 * dim), np.eye(2 * dim))
        self.likelihood = likelihood
        self.next_likelihood = next_likelihood
        self.fail_predict = fail_predict
        self.fail_update = fail_update

    def predict(self, dt):
        self.state.x[0] += dt
        if self.fail_predict:
            raise RuntimeError("predict failed")

    def update(self, _z, _R):
        self.state.x[0] += 1.0
        if self.next_likelihood is not None:
            self.likelihood = self.next_likelihood
        if self.fail_update:
            raise RuntimeError("update failed")


class _LogLikelihoodModel(_LikelihoodModel):
    def __init__(self, log_likelihood):
        super().__init__(likelihood=1.0)
        self.log_likelihood = log_likelihood


def test_imm_rejects_empty_incomplete_incompatible_or_readonly_models():
    with pytest.raises(ValueError, match="empty"):
        IMMEstimator([])
    with pytest.raises(TypeError, match="sequence"):
        IMMEstimator(None)
    with pytest.raises(TypeError, match="predict"):
        IMMEstimator([object()])
    with pytest.raises(ValueError, match="same dimension"):
        IMMEstimator([_LikelihoodModel(dim=2), _LikelihoodModel(dim=3)])
    duplicate = _LikelihoodModel()
    with pytest.raises(ValueError, match="distinct filter objects"):
        IMMEstimator([duplicate, duplicate], transition=np.eye(2))
    with pytest.raises(ValueError, match="32-mode"):
        IMMEstimator([_LikelihoodModel() for _ in range(33)])

    particle = ParticleFilter(np.zeros(6), np.eye(6), n_particles=8, rng=np.random.default_rng(105))
    with pytest.raises(TypeError, match="assignable"):
        IMMEstimator([particle])


def test_gaussian_state_rejects_oversized_state_before_covariance_decomposition(monkeypatch):
    monkeypatch.setattr(
        np.linalg,
        "eigh",
        lambda *_args, **_kwargs: pytest.fail("oversized covariance must not be decomposed"),
    )
    with pytest.raises(ValueError, match="at most 128"):
        GaussianState(np.zeros(129), np.eye(129))


@pytest.mark.parametrize(
    "transition",
    [
        np.eye(1),
        np.array([[0.9, 0.1], [-0.1, 1.1]]),
        np.array([[0.9, 0.1], [0.2, 0.2]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
        [["bad"]],
    ],
)
def test_imm_rejects_invalid_transition_matrices(transition):
    with pytest.raises(ValueError):
        IMMEstimator([_LikelihoodModel(), _LikelihoodModel()], transition=transition)


@pytest.mark.parametrize(
    "mode_probs",
    [
        np.ones(1),
        np.array([-0.1, 1.1]),
        np.array([0.2, 0.2]),
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        ["bad"],
    ],
)
def test_imm_rejects_invalid_mode_probabilities(mode_probs):
    with pytest.raises(ValueError):
        IMMEstimator([_LikelihoodModel(), _LikelihoodModel()], mode_probs=mode_probs)


@pytest.mark.parametrize("likelihood", [-1.0, np.nan, np.inf, -np.inf, True, "bad"])
def test_imm_rejects_invalid_initial_model_likelihoods(likelihood):
    with pytest.raises(FloatingPointError):
        IMMEstimator([_LikelihoodModel(likelihood=likelihood)])


def test_single_model_imm_has_identity_transition_and_predicts():
    model = _LikelihoodModel()
    imm = IMMEstimator([model])
    assert np.array_equal(imm.transition, np.ones((1, 1)))
    imm.predict(0.5)
    assert np.array_equal(imm.mode_probs, np.ones(1))
    assert imm.state.x[0] == pytest.approx(0.5)


def test_imm_update_failure_rolls_back_every_model_and_probabilities():
    models = [_LikelihoodModel(0.5), _LikelihoodModel(0.5, next_likelihood=np.nan)]
    imm = IMMEstimator(models, transition=np.eye(2))
    before_states = [model.state.copy() for model in imm.models]
    before_probs = imm.mode_probs.copy()
    before_likelihoods = [model.likelihood for model in imm.models]
    with pytest.raises(FloatingPointError, match="invalid likelihood"):
        imm.update(np.zeros(3), np.eye(3))
    assert np.array_equal(imm.mode_probs, before_probs)
    for model, before in zip(imm.models, before_states):
        assert np.array_equal(model.state.x, before.x)
        assert np.array_equal(model.state.P, before.P)
    assert [model.likelihood for model in imm.models] == before_likelihoods


def test_imm_failure_rolls_back_arbitrary_model_fields_and_rng_state():
    class StatefulModel(_LikelihoodModel):
        def __init__(self, *, fail=False):
            super().__init__(fail_update=fail)
            self.update_count = 0
            self.rng = np.random.default_rng(2026)

        def update(self, z, R):
            self.update_count += 1
            self.rng.random()
            super().update(z, R)

    models = [StatefulModel(), StatefulModel(fail=True)]
    imm = IMMEstimator(models, transition=np.eye(2))
    before = [copy.deepcopy(vars(model)) for model in models]
    with pytest.raises(RuntimeError, match="update failed"):
        imm.update(np.zeros(3), np.eye(3))
    for model, expected in zip(models, before):
        assert model.update_count == expected["update_count"]
        assert model.rng.bit_generator.state == expected["rng"].bit_generator.state


def test_direct_ekf_and_imm_polar_updates_share_canonical_angle_contract():
    ekf = ExtendedKalmanFilter(np.zeros(6), np.eye(6))
    imm = IMMEstimator.default_cv_bank(np.zeros(6), np.eye(6))
    for estimator in (ekf, imm):
        with pytest.raises(ValueError, match="elevation"):
            estimator.update_polar(np.array([1.0, 0.0, np.pi]), np.eye(3))
        with pytest.raises(ValueError, match="azimuth magnitude"):
            estimator.update_polar(np.array([1.0, 1e300, 0.0]), np.eye(3))


def test_imm_predict_failure_rolls_back_every_model_and_probabilities():
    models = [_LikelihoodModel(), _LikelihoodModel(fail_predict=True)]
    imm = IMMEstimator(models, transition=np.eye(2), mode_probs=np.array([0.4, 0.6]))
    before_states = [model.state.copy() for model in imm.models]
    before_probs = imm.mode_probs.copy()
    with pytest.raises(RuntimeError, match="predict failed"):
        imm.predict(1.0)
    assert np.array_equal(imm.mode_probs, before_probs)
    for model, before in zip(imm.models, before_states):
        assert np.array_equal(model.state.x, before.x)
        assert np.array_equal(model.state.P, before.P)


def test_imm_rejects_zero_total_evidence_and_rolls_back():
    imm = IMMEstimator([_LikelihoodModel(0.0), _LikelihoodModel(0.0)])
    before_states = [model.state.copy() for model in imm.models]
    with pytest.raises(FloatingPointError, match="zero or non-finite evidence"):
        imm.update(np.zeros(3), np.eye(3))
    for model, before in zip(imm.models, before_states):
        assert np.array_equal(model.state.x, before.x)


def test_imm_log_likelihood_update_does_not_underflow():
    imm = IMMEstimator(
        [_LogLikelihoodModel(-1000.0), _LogLikelihoodModel(-1001.0)],
        transition=np.eye(2),
    )
    imm.update(np.zeros(3), np.eye(3))
    assert np.allclose(imm.mode_probs, [0.7310585786300049, 0.2689414213699951])
    imm.update(np.zeros(3), np.eye(3))
    assert np.allclose(imm.mode_probs, [0.8807970779778823, 0.11920292202211755])


def test_imm_handles_a_zero_probability_destination_mode():
    imm = IMMEstimator(
        [_LikelihoodModel(), _LikelihoodModel()],
        transition=np.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    imm.predict(0.5)
    assert np.array_equal(imm.mode_probs, np.array([1.0, 0.0]))
    _assert_valid_gaussian(imm.state)


def test_imm_revalidates_public_probability_arrays_at_runtime():
    imm = IMMEstimator([_LikelihoodModel(), _LikelihoodModel()])
    imm.mode_probs = np.array([0.2, 0.2])
    with pytest.raises(ValueError, match="sum to 1"):
        _ = imm.state

    imm = IMMEstimator([_LikelihoodModel(), _LikelihoodModel()])
    imm.transition[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        imm.predict(0.1)


def test_filter_shape_and_work_admission_precedes_float_widening(monkeypatch):
    from manwe.fusion import filters as filters_module

    oversized_angles = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (filters_module.MAX_WRAP_ANGLE_CELLS + 1,),
    )
    oversized_state = np.broadcast_to(
        np.array(1, dtype=np.int8),
        (2 * filters_module.MAX_FILTER_DIMENSION + 1,),
    )
    malformed_covariance = np.broadcast_to(np.array(1, dtype=np.int8), (1, 4_000_000))
    malformed_particles = np.broadcast_to(np.array(1, dtype=np.int8), (1, 4_000_000))
    malformed_transition = np.broadcast_to(np.array(1, dtype=np.int8), (1, 4_000_000))
    forbidden = (
        oversized_angles,
        oversized_state,
        malformed_covariance,
        malformed_particles,
        malformed_transition,
    )
    real_float_array = filters_module._float_array

    def guarded_float_array(raw, name):
        if any(
            raw.shape == rejected.shape and np.shares_memory(raw, rejected)
            for rejected in forbidden
        ):
            pytest.fail(f"{name} was widened before raw shape/work admission")
        return real_float_array(raw, name)

    monkeypatch.setattr(filters_module, "_float_array", guarded_float_array)

    with pytest.raises(ValueError, match="value safety limit"):
        wrap_angle(oversized_angles)
    with pytest.raises(ValueError, match="at most 128"):
        GaussianState(oversized_state, np.eye(1))
    with pytest.raises(ValueError, match="must have shape"):
        GaussianState(np.zeros(6), malformed_covariance)

    particle = ParticleFilter(
        np.zeros(6),
        np.eye(6),
        n_particles=4,
        rng=np.random.default_rng(123),
    )
    particle.particles = malformed_particles
    with pytest.raises(ValueError, match="particles must have shape"):
        _ = particle.state

    with pytest.raises(ValueError, match="transition must have shape"):
        IMMEstimator(
            [_LikelihoodModel(), _LikelihoodModel()],
            transition=malformed_transition,
        )


def test_filter_numeric_admission_rejects_coercion_and_lossy_integers():
    class Coercive:
        calls = 0

        def __float__(self):
            type(self).calls += 1
            return 1.0

    with pytest.raises(ValueError, match="real numeric"):
        GaussianState(np.full(6, Coercive(), dtype=object), np.eye(6))
    assert Coercive.calls == 0

    with pytest.raises(ValueError, match="exactly representable"):
        GaussianState(np.array([2**53 + 1], dtype=np.uint64), np.eye(1))

    with pytest.raises(FloatingPointError, match="invalid likelihood"):
        IMMEstimator([_LikelihoodModel(likelihood=Coercive())])
    with pytest.raises(FloatingPointError, match="invalid log likelihood"):
        IMMEstimator([_LogLikelihoodModel(Coercive())])
    assert Coercive.calls == 0


def test_filter_rejects_finite_wide_values_that_overflow_float64():
    wide = np.longdouble
    if np.finfo(wide).max <= np.finfo(np.float64).max:
        pytest.skip("long double has no wider finite range on this platform")
    too_large = np.array([np.finfo(wide).max], dtype=wide)
    with pytest.raises(ValueError, match="finite"):
        GaussianState(too_large, np.eye(1))
    with pytest.raises(ValueError, match="finite"):
        cv_transition(wide(np.finfo(wide).max))


def test_filter_rejects_wider_float_precision_loss_when_platform_exposes_it():
    wide = np.longdouble
    if np.finfo(wide).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("long double has no wider precision on this platform")
    lossy = wide("0.1")
    assert np.asarray(float(lossy), dtype=wide) != lossy
    with pytest.raises(ValueError, match="precision"):
        GaussianState(np.array([lossy], dtype=wide), np.eye(1))
    with pytest.raises(ValueError, match="finite number"):
        cv_transition(lossy)


def test_filter_probability_sum_overflow_fails_as_validation_not_warning():
    particle = ParticleFilter(
        np.zeros(6),
        np.eye(6),
        n_particles=4,
        rng=np.random.default_rng(124),
    )
    particle.weights = np.full(4, np.finfo(float).max)
    with pytest.raises(ValueError, match="sum to 1"):
        _ = particle.state

    huge = np.finfo(float).max
    with pytest.raises(ValueError, match="row must sum to 1"):
        IMMEstimator(
            [_LikelihoodModel(), _LikelihoodModel()],
            transition=np.array([[huge, huge], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="mode_probs must sum to 1"):
        IMMEstimator(
            [_LikelihoodModel(), _LikelihoodModel()],
            mode_probs=np.array([huge, huge]),
        )
