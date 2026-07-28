"""Adversarial admission, covariance, and transaction tests for fusion filters."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from manwe.fusion import filters as filters_module
from manwe.fusion.filters import (
    GaussianState,
    IMMEstimator,
    KalmanFilter,
    ParticleFilter,
    UnscentedKalmanFilter,
    cv_transition,
)


class _CoerciveInt(int):
    calls = 0

    def __ge__(self, _other):
        type(self).calls += 1
        raise AssertionError("integer comparison callback must not run")

    def __int__(self):
        type(self).calls += 1
        raise AssertionError("integer conversion callback must not run")


class _CoerciveFloat(float):
    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("float conversion callback must not run")


class _CoerciveList(list):
    calls = 0

    def __len__(self):
        type(self).calls += 1
        raise AssertionError("list subclass callback must not run")


class _CoerciveArray(np.ndarray):
    calls = 0

    def __new__(cls):
        return np.zeros(1).view(cls)

    def __array__(self, *_args, **_kwargs):
        type(self).calls += 1
        raise AssertionError("array subclass callback must not run")

    def __array_finalize__(self, _source):
        pass


def test_numeric_subclasses_are_rejected_without_callbacks():
    _CoerciveInt.calls = 0
    _CoerciveFloat.calls = 0
    _CoerciveList.calls = 0
    _CoerciveArray.calls = 0

    with pytest.raises(ValueError, match="dim"):
        cv_transition(1.0, _CoerciveInt(3))
    with pytest.raises(ValueError, match="n_particles"):
        ParticleFilter(np.zeros(6), np.eye(6), n_particles=_CoerciveInt(4))
    with pytest.raises(ValueError, match="finite number"):
        cv_transition(_CoerciveFloat(1.0))
    with pytest.raises(ValueError, match="real numeric"):
        GaussianState([_CoerciveFloat(1.0)], np.eye(1))
    with pytest.raises(ValueError, match="built-in"):
        GaussianState(_CoerciveList([1.0]), np.eye(1))
    with pytest.raises(ValueError, match="ndarray subclass"):
        GaussianState(_CoerciveArray(), np.eye(1))

    assert _CoerciveInt.calls == 0
    assert _CoerciveFloat.calls == 0
    assert _CoerciveList.calls == 0
    assert _CoerciveArray.calls == 0


def test_shape_and_cycle_fail_before_coercive_leaves():
    _CoerciveFloat.calls = 0
    with pytest.raises(ValueError, match="shape"):
        GaussianState(np.zeros(2), [[_CoerciveFloat(1.0)]])
    assert _CoerciveFloat.calls == 0

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        GaussianState(cyclic, np.eye(1))


def test_runtime_mutation_cannot_bypass_dimension_or_work_caps(monkeypatch):
    kf = KalmanFilter(np.zeros(6), np.eye(6))
    particle = ParticleFilter(
        np.zeros(6),
        np.eye(6),
        n_particles=2,
        rng=np.random.default_rng(1),
    )
    ukf = UnscentedKalmanFilter(np.zeros(6), np.eye(6))

    kf.dim = filters_module.MAX_FILTER_DIMENSION + 1
    kf.state.x = np.zeros(2 * kf.dim)
    kf.state.P = np.eye(2 * kf.dim)
    monkeypatch.setattr(
        np.linalg,
        "eigh",
        lambda *_args, **_kwargs: pytest.fail("invalid dimension must fail before decomposition"),
    )
    with pytest.raises(ValueError, match="dim"):
        kf.predict(0.0)

    particle.n_particles = filters_module.MAX_FILTER_PARTICLES + 1
    with pytest.raises(ValueError, match="n_particles"):
        _ = particle.state

    ukf.n = 10**9
    real_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape and shape[0] > 100_000:
            pytest.fail(f"invalid UKF state requested an allocation of shape {shape}")
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    with pytest.raises(ValueError, match="state dimension"):
        ukf.predict(0.0)


@pytest.mark.parametrize(
    "estimator",
    [
        KalmanFilter(np.zeros(6), np.eye(6)),
        UnscentedKalmanFilter(np.zeros(6), np.eye(6)),
        ParticleFilter(
            np.zeros(6),
            np.eye(6),
            n_particles=2,
            rng=np.random.default_rng(8),
        ),
    ],
    ids=("kalman", "ukf", "particle"),
)
def test_zero_dt_prediction_still_validates_process_noise(estimator):
    _CoerciveFloat.calls = 0
    estimator.sigma_a = _CoerciveFloat(1.0)
    with pytest.raises(ValueError, match="sigma_a"):
        estimator.predict(0.0)
    assert _CoerciveFloat.calls == 0


def test_ukf_runtime_rechecks_weight_stability_before_zero_dt_return():
    ukf = UnscentedKalmanFilter(np.zeros(6), np.eye(6))
    ukf.alpha = 1e-154
    ukf._sigma_scale = float(np.square(np.float64(ukf.alpha)) * (ukf.n + ukf.kappa))
    ukf.lambda_ = ukf._sigma_scale - ukf.n
    ukf._wm = np.full(2 * ukf.n + 1, 1.0 / (2 * ukf._sigma_scale))
    ukf._wc = ukf._wm.copy()
    ukf._wm[0] = ukf.lambda_ / ukf._sigma_scale
    ukf._wc[0] = ukf._wm[0] + (1 - ukf.alpha**2 + ukf.beta)

    with pytest.raises(ValueError, match="unstable sigma-point weights"):
        ukf.predict(0.0)


def test_runtime_measurement_matrix_is_a_derived_invariant():
    kf = KalmanFilter(np.zeros(6), np.eye(6))
    kf.H = np.broadcast_to(1.0, (1_000_000, 6))
    with pytest.raises(ValueError, match="H must have shape"):
        kf.innovation(np.zeros(3))


def test_failed_runtime_configuration_validation_does_not_canonicalize_state():
    kf = KalmanFilter(np.zeros(6), np.eye(6))
    state_x = [0.0] * 6
    state_P = np.eye(6).tolist()
    kf.state.x = state_x
    kf.state.P = state_P
    kf.H = np.zeros((3, 6))

    with pytest.raises(ValueError, match="Cartesian position projection"):
        kf.predict(0.0)
    assert kf.state.x is state_x
    assert kf.state.P is state_P


def test_particle_constructor_rejects_bit_generator_subclass_before_sampling(monkeypatch):
    class CoercivePCG64(np.random.PCG64):
        calls = 0

        @property
        def state(self):
            type(self).calls += 1
            raise AssertionError("BitGenerator state callback must not run")

        @state.setter
        def state(self, _value):
            type(self).calls += 1
            raise AssertionError("BitGenerator state callback must not run")

    monkeypatch.setattr(
        filters_module,
        "_sample_gaussian",
        lambda *_args, **_kwargs: pytest.fail("sampling must follow RNG admission"),
    )
    generator = np.random.Generator(CoercivePCG64())
    with pytest.raises(TypeError, match="built-in numpy BitGenerator"):
        ParticleFilter(np.zeros(6), np.eye(6), n_particles=2, rng=generator)
    assert CoercivePCG64.calls == 0


def test_covariance_validation_is_scale_invariant_and_preserves_extrema():
    maximum = np.finfo(float).max
    minimum = np.nextafter(0.0, 1.0)
    matrices = (
        np.full((2, 2), maximum),
        np.full((2, 2), minimum),
        np.diag([maximum, minimum]),
        np.array([[maximum, 1e-8], [1e-8, minimum]]),
        np.array([[maximum, minimum], [minimum, minimum]]),
    )
    old_errors = np.seterr(all="raise")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for covariance in matrices:
                state = GaussianState(np.zeros(2), covariance)
                assert np.array_equal(state.P, covariance)
    finally:
        np.seterr(**old_errors)


def test_unrelated_large_variance_cannot_hide_local_indefiniteness():
    covariance = np.diag([1e300, 1.0, 1.0, 1.0])
    covariance[1, 2] = covariance[2, 1] = 2.0
    original = covariance.copy()
    assert np.linalg.eigvalsh(covariance)[0] == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianState(np.zeros(4), covariance)
    assert np.array_equal(covariance, original)


def test_covariance_rejects_zero_variance_correlation_and_extreme_asymmetry():
    minimum = np.nextafter(0.0, 1.0)
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="zero variances"):
        GaussianState(
            np.zeros(2),
            np.array([[0.0, minimum], [minimum, 1.0]]),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="positive semidefinite|symmetric"):
            GaussianState(
                np.zeros(2),
                np.array([[1.0, maximum], [-maximum, 1.0]]),
            )


def test_psd_repair_preserves_rank_and_never_upgrades_pd_required_noise():
    singular = np.ones((6, 6))
    repaired_singular = GaussianState(np.zeros(6), singular).P
    assert np.linalg.matrix_rank(repaired_singular) == 1
    assert np.array_equal(np.diag(repaired_singular), np.ones(6))

    near_indefinite = np.array([[1.0, 1.0 + 1e-14], [1.0 + 1e-14, 1.0]])
    repaired = GaussianState(np.zeros(2), near_indefinite).P
    assert np.array_equal(np.diag(repaired), np.ones(2))
    assert np.linalg.eigvalsh(repaired)[0] >= 0.0
    with pytest.raises(ValueError, match="positive definite"):
        filters_module._as_covariance(
            near_indefinite,
            2,
            "R",
            positive_definite=True,
        )
    with pytest.raises(ValueError, match="positive definite"):
        filters_module._as_covariance(
            singular,
            6,
            "R",
            positive_definite=True,
        )


def test_extreme_singular_covariance_has_a_finite_sampling_factor():
    maximum = np.finfo(float).max
    covariance = np.full((6, 6), maximum)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        particle = ParticleFilter(
            np.zeros(6),
            covariance,
            n_particles=4,
            rng=np.random.default_rng(2),
        )
    assert np.isfinite(particle.particles).all()


def test_imm_constructor_stops_after_cap_plus_one_models():
    class UnboundedModels:
        yielded = 0

        def __iter__(self):
            while True:
                type(self).yielded += 1
                if type(self).yielded > filters_module._MAX_IMM_MODELS + 1:
                    raise AssertionError("constructor consumed beyond cap plus one")
                yield object()

    UnboundedModels.yielded = 0
    with pytest.raises(ValueError, match="32-mode"):
        IMMEstimator(UnboundedModels())
    assert UnboundedModels.yielded == filters_module._MAX_IMM_MODELS + 1


class _OriginalPredictFailure(RuntimeError):
    pass


class _TransactionalModel:
    dim = 3
    likelihood = 1.0

    def __init__(self, *, fail: bool):
        self.state = GaussianState(np.zeros(6), np.eye(6))
        self.metadata = {"payload": "unchanged"}
        self.fail = fail

    def predict(self, dt):
        self.state.x[0] += dt
        if self.fail:
            original_models = self.owner.models
            original_models.reverse()
            self.owner.models = list(original_models)
            object.__setattr__(self, "__dict__", {"corrupted": True})
            raise _OriginalPredictFailure("model predict failed")

    def update(self, _z, _R):
        pass


def test_imm_rollback_preserves_original_exception_and_full_model_values():
    models = [_TransactionalModel(fail=False), _TransactionalModel(fail=True)]
    imm = IMMEstimator(models, transition=np.eye(2), mode_probs=np.array([0.4, 0.6]))
    original_models_container = imm.models
    shared = {"payload": "shared"}
    imm.namespace_alias = vars(imm)
    for model in models:
        model.owner = imm
        model.models_alias = imm.models
        model.namespace_alias = vars(model)
        model.shared = shared
    before_namespaces = [vars(model) for model in models]
    before_keys = [set(vars(model)) for model in models]
    before_states = [model.state.copy() for model in models]
    before_metadata = [model.metadata.copy() for model in models]
    before_failures = [model.fail for model in models]
    before_probabilities = imm.mode_probs.copy()
    before_cbar = imm._cbar.copy()

    with pytest.raises(_OriginalPredictFailure, match="model predict failed"):
        imm.predict(1.0)

    assert np.array_equal(imm.mode_probs, before_probabilities)
    assert np.array_equal(imm._cbar, before_cbar)
    assert imm.models is original_models_container
    assert imm.models == models
    assert imm.namespace_alias is vars(imm)
    for index, model in enumerate(models):
        assert vars(model) is before_namespaces[index]
        assert set(vars(model)) == before_keys[index]
        assert np.array_equal(model.state.x, before_states[index].x)
        assert np.array_equal(model.state.P, before_states[index].P)
        assert model.metadata == before_metadata[index]
        assert model.fail is before_failures[index]
        assert model.owner is imm
        assert model.models_alias is imm.models
        assert model.namespace_alias is vars(model)
    assert models[0].shared is models[1].shared
    assert models[0].shared == shared


def test_successful_model_callback_cannot_mutate_imm_configuration():
    class MutatingModel(_TransactionalModel):
        def update(self, _z, _R):
            self.state.x[0] += 1.0
            self.owner.transition = np.array([[0.0, 1.0], [1.0, 0.0]])
            self.owner.mode_probs = np.array([1.0, 0.0])

    models = [MutatingModel(fail=False), MutatingModel(fail=False)]
    imm = IMMEstimator(models, transition=np.eye(2), mode_probs=np.array([0.4, 0.6]))
    for model in models:
        model.owner = imm
    before_states = [model.state.copy() for model in models]
    before_transition = imm.transition.copy()
    before_probabilities = imm.mode_probs.copy()

    with pytest.raises(ValueError, match="configuration changed"):
        imm.update(np.zeros(3), np.eye(3))

    assert np.array_equal(imm.transition, before_transition)
    assert np.array_equal(imm.mode_probs, before_probabilities)
    for model, before in zip(models, before_states):
        assert np.array_equal(model.state.x, before.x)
        assert np.array_equal(model.state.P, before.P)


def test_imm_snapshot_rejects_non_string_namespace_keys_without_deepcopy():
    class CoerciveKey:
        calls = 0

        def __deepcopy__(self, _memo):
            type(self).calls += 1
            raise AssertionError("non-string key deepcopy callback must not run")

    model = _TransactionalModel(fail=False)
    model.__dict__[CoerciveKey()] = "unsafe"
    with pytest.raises(TypeError, match="exact strings"):
        IMMEstimator([model])
    assert CoerciveKey.calls == 0


def test_imm_namespace_capture_bypasses_hostile_dict_access():
    class HostileNamespaceModel(_TransactionalModel):
        dict_calls = 0

        def __getattribute__(self, name):
            if name == "__dict__":
                type(self).dict_calls += 1
                raise AssertionError("custom __dict__ access must not run")
            return object.__getattribute__(self, name)

    HostileNamespaceModel.dict_calls = 0
    model = HostileNamespaceModel(fail=False)
    imm = IMMEstimator([model])
    imm.predict(1.0)
    assert HostileNamespaceModel.dict_calls == 0


def test_imm_rollback_ignores_hostile_dict_access_installed_by_callback():
    class PatchingModel(_TransactionalModel):
        def predict(self, _dt):
            model_type = type(self)

            def hostile_getattribute(instance, name):
                if name == "__dict__":
                    raise AssertionError("rollback must use its captured namespace")
                return object.__getattribute__(instance, name)

            model_type.__getattribute__ = hostile_getattribute
            object.__setattr__(self, "__dict__", {"corrupted": True})
            raise _OriginalPredictFailure("model predict failed")

    model = PatchingModel(fail=False)
    imm = IMMEstimator([model])
    original_namespace = object.__getattribute__(model, "__dict__")
    original_state = model.state.copy()
    try:
        with pytest.raises(_OriginalPredictFailure, match="model predict failed"):
            imm.predict(1.0)
        assert object.__getattribute__(model, "__dict__") is original_namespace
        assert np.array_equal(model.state.x, original_state.x)
        assert np.array_equal(model.state.P, original_state.P)
    finally:
        del PatchingModel.__getattribute__


def test_failed_imm_construction_preserves_caller_state_identity():
    first = _TransactionalModel(fail=False)
    second = _TransactionalModel(fail=False)
    second.dim = 2
    second.state = GaussianState(np.zeros(4), np.eye(4))
    first_state = first.state
    second_state = second.state

    with pytest.raises(ValueError, match="same dimension"):
        IMMEstimator([first, second])
    assert first.state is first_state
    assert second.state is second_state

    compatible = _TransactionalModel(fail=False)
    compatible_state = compatible.state
    with pytest.raises(ValueError, match="transition"):
        IMMEstimator([compatible], transition=np.zeros((1, 1)))
    assert compatible.state is compatible_state


def test_imm_runtime_validates_cbar_before_model_callbacks():
    class Model(_TransactionalModel):
        calls = 0

        def update(self, _z, _R):
            type(self).calls += 1

    Model.calls = 0
    imm = IMMEstimator([Model(fail=False)])
    imm._cbar = np.array([0.0])
    with pytest.raises(ValueError, match="sum to 1"):
        imm.update(np.zeros(3), np.eye(3))
    assert Model.calls == 0


def test_imm_rejects_unsafe_snapshot_payload_without_invoking_copy_hooks():
    class Poison:
        calls = 0

        def __deepcopy__(self, _memo):
            type(self).calls += 1
            raise AssertionError("custom copy hooks must not run")

    model = _TransactionalModel(fail=False)
    imm = IMMEstimator([model])
    state = model.state
    state_x = state.x
    probabilities = imm.mode_probs
    model.poison = Poison()

    with pytest.raises(TypeError, match="unsafe type"):
        imm.predict(1.0)

    assert Poison.calls == 0
    assert model.state is state
    assert model.state.x is state_x
    assert imm.mode_probs is probabilities


def test_imm_runtime_getter_mutation_is_rolled_back_before_callbacks():
    class GetterModel:
        dim = 3
        likelihood = 1.0

        def __init__(self):
            self._state = GaussianState(np.zeros(6), np.eye(6))
            self.poison = False

        @property
        def state(self):
            if self.poison:
                self._state.x[0] = 888.0
                self.owner.preflight_poison = True
                raise RuntimeError("getter poison")
            return self._state

        @state.setter
        def state(self, value):
            self._state = value

        def predict(self, _dt):
            pass

        def update(self, _z, _R):
            type(self).update_calls += 1

        update_calls = 0

    model = GetterModel()
    imm = IMMEstimator([model])
    model.owner = imm
    model.poison = True
    state = model._state
    state_x = state.x

    with pytest.raises(RuntimeError, match="getter poison"):
        imm.update(np.zeros(3), np.eye(3))

    assert model._state is state
    assert model._state.x is state_x
    assert model._state.x[0] == 0.0
    assert "preflight_poison" not in vars(imm)
    assert GetterModel.update_calls == 0


def test_imm_final_runtime_validation_is_sealed_against_getter_toctou():
    class GetterModel:
        dim = 3
        likelihood = 1.0
        state_reads = 0

        def __init__(self):
            self._state = GaussianState(np.zeros(6), np.eye(6))

        @property
        def state(self):
            type(self).state_reads += 1
            if type(self).state_reads == 4:
                self.owner.final_validation_poison = True
            return self._state

        @state.setter
        def state(self, value):
            self._state = value

        def predict(self, _dt):
            pass

        def update(self, _z, _R):
            pass

    GetterModel.state_reads = 0
    model = GetterModel()
    imm = IMMEstimator([model])
    model.owner = imm

    with pytest.raises(ValueError, match="must not mutate transaction state"):
        imm.update(np.zeros(3), np.eye(3))

    assert "final_validation_poison" not in vars(imm)


def test_imm_restores_model_and_owner_classes_and_bypasses_helper_overrides():
    class BaseModel(_TransactionalModel):
        def __init__(self, *, fail=False):
            super().__init__(fail=False)
            self.callback_fails = fail

    class ChangedModel(BaseModel):
        pass

    class ClassChangingModel(BaseModel):
        def predict(self, _dt):
            self.__class__ = ChangedModel
            self.state.x[0] = 4.0
            if self.callback_fails:
                raise _OriginalPredictFailure("class mutation failed")

    class EvilIMM(IMMEstimator):
        def _require_seal(self, *_args, **_kwargs):
            return None

        def _restore_transaction(self, *_args, **_kwargs):
            return None

    class OwnerChangingModel(BaseModel):
        def predict(self, _dt):
            self.owner._require_seal = lambda *_args, **_kwargs: None
            self.owner._restore_transaction = lambda *_args, **_kwargs: None
            self.owner.__class__ = EvilIMM
            self.state.x[0] = 9.0

    failing = ClassChangingModel(fail=True)
    failing_imm = IMMEstimator([failing])
    with pytest.raises(_OriginalPredictFailure, match="class mutation failed"):
        failing_imm.predict(1.0)
    assert type(failing) is ClassChangingModel
    assert failing.state.x[0] == 0.0

    owner_model = OwnerChangingModel()
    owner_imm = IMMEstimator([owner_model])
    owner_model.owner = owner_imm
    with pytest.raises(ValueError, match="configuration changed"):
        owner_imm.predict(1.0)
    assert type(owner_imm) is IMMEstimator
    assert type(owner_model) is OwnerChangingModel
    assert owner_model.state.x[0] == 0.0
    assert "_require_seal" not in vars(owner_imm)
    assert "_restore_transaction" not in vars(owner_imm)


def test_imm_rejects_hostile_namespace_descriptors_slots_and_array_views():
    class HostileNamespace(_TransactionalModel):
        dict_calls = 0

        @property
        def __dict__(self):
            type(self).dict_calls += 1
            raise AssertionError("hostile namespace descriptor must not run")

    HostileNamespace.dict_calls = 0
    with pytest.raises(TypeError):
        IMMEstimator([HostileNamespace(fail=False)])
    assert HostileNamespace.dict_calls == 0

    class SlottedModel:
        __slots__ = ("secret", "__dict__")
        dim = 3
        likelihood = 1.0

        def __init__(self):
            self.secret = 1
            self.state = GaussianState(np.zeros(6), np.eye(6))

        def predict(self, _dt):
            pass

        def update(self, _z, _R):
            pass

    with pytest.raises(TypeError, match="slot"):
        IMMEstimator([SlottedModel()])

    model = _TransactionalModel(fail=False)
    model.view = np.arange(4.0)[::2]
    with pytest.raises(TypeError, match="own their storage"):
        IMMEstimator([model])


def test_imm_rollback_restores_external_aliases_and_survives_new_finalizers():
    shared = {"value": 0}

    class InjectingFinalizer:
        def __init__(self, model):
            self.model = model

        def __del__(self):
            self.model.injected_by_del = True

    class AliasModel(_TransactionalModel):
        def __init__(self):
            super().__init__(fail=False)
            self.shared = shared

        def update(self, _z, _R):
            self.state.x[0] = 777.0
            self.shared["value"] = 1
            self.payload = InjectingFinalizer(self)
            raise SystemExit("alias failure")

    model = AliasModel()
    imm = IMMEstimator([model])
    state = model.state
    state_x = state.x
    shared_alias = model.shared
    namespace = vars(model)

    with pytest.raises(SystemExit, match="alias failure"):
        imm.update(np.zeros(3), np.eye(3))

    assert vars(model) is namespace
    assert model.state is state
    assert model.state.x is state_x
    assert state_x[0] == 0.0
    assert model.shared is shared_alias
    assert shared_alias == {"value": 0}
    assert "payload" not in vars(model)
    assert "injected_by_del" not in vars(model)
