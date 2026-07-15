"""Recursive Bayesian filters for a 6-state constant-velocity target.

This is an independent numerical reference using a six-state
``x = [px, py, pz, vx, vy, vz]`` convention (metres and m/s in a common world
frame). It shares estimator vocabulary with downstream systems, but parity
requires explicit adapters and intermediate-state fixtures.

Filters
-------
- :class:`KalmanFilter`          linear Cartesian CV (Joseph-form covariance)
- :class:`ExtendedKalmanFilter`  adds a polar (radar) measurement update
- :class:`UnscentedKalmanFilter` derivative-free, scaled sigma points
- :class:`ParticleFilter`        Monte-Carlo posterior, systematic resampling
- :class:`IMMEstimator`          bank of models mixed by a Markov chain

All are pure-numpy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

POS_DIM = 3
STATE_DIM = 6
MAX_FILTER_DIMENSION = 64
MAX_FILTER_PARTICLES = 100_000
MIN_POLAR_HORIZONTAL_RANGE = 1e-6


# ---------------------------------------------------------------------------
# Motion / measurement models
# ---------------------------------------------------------------------------
def cv_transition(dt: float, dim: int = POS_DIM) -> np.ndarray:
    """Constant-velocity state-transition matrix ``F`` for a ``2*dim`` state."""
    dim = _validate_dim(dim)
    dt = _validate_dt(dt)
    n = 2 * dim
    F = np.eye(n)
    F[:dim, dim:] = dt * np.eye(dim)
    return F


def cv_process_noise(dt: float, sigma_a: float, dim: int = POS_DIM) -> np.ndarray:
    """Discrete white-noise-acceleration process covariance ``Q``.

    ``sigma_a`` is the standard deviation of the un-modelled acceleration (m/s²).
    """
    dim = _validate_dim(dim)
    dt = _validate_dt(dt)
    sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
    if dt == 0 or sigma_a == 0:
        return np.zeros((2 * dim, 2 * dim))
    try:
        with np.errstate(over="raise", invalid="raise"):
            # Form the process-noise gain first. This is algebraically identical
            # to sigma_a**2 times the usual dt powers, but avoids needless
            # intermediate overflow when a very large dt and tiny sigma cancel.
            velocity_std = np.float64(sigma_a) * dt
            position_std = 0.5 * velocity_std * dt
            I = np.eye(dim)
            Q = np.zeros((2 * dim, 2 * dim))
            Q[:dim, :dim] = np.square(position_std) * I
            Q[:dim, dim:] = (position_std * velocity_std) * I
            Q[dim:, :dim] = Q[:dim, dim:]
            Q[dim:, dim:] = np.square(velocity_std) * I
    except FloatingPointError as exc:
        raise ValueError("dt and sigma_a produce a non-finite process covariance") from exc
    if not np.isfinite(Q).all():
        raise ValueError("dt and sigma_a produce a non-finite process covariance")
    return Q


def position_measurement_matrix(dim: int = POS_DIM) -> np.ndarray:
    """Measurement matrix ``H = [I | 0]`` mapping state → observed position."""
    dim = _validate_dim(dim)
    H = np.zeros((dim, 2 * dim))
    H[:, :dim] = np.eye(dim)
    return H


def _symmetrize(P: np.ndarray) -> np.ndarray:
    # Halve before adding so symmetric, near-float-max inputs do not overflow.
    return 0.5 * P + 0.5 * P.T


def _as_real_array(value: object, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        return np.asarray(raw, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap finite angle(s) to ``[-pi, pi)``."""
    array = _as_real_array(a, "angle")
    if not np.isfinite(array).all():
        raise ValueError("angle must contain only finite values")
    wrapped = (array + np.pi) % (2 * np.pi) - np.pi
    return float(wrapped) if array.ndim == 0 else wrapped


def _validate_dim(dim: object) -> int:
    if (
        isinstance(dim, bool)
        or not isinstance(dim, (int, np.integer))
        or not 1 <= dim <= MAX_FILTER_DIMENSION
    ):
        raise ValueError(f"dim must be an integer in [1, {MAX_FILTER_DIMENSION}]")
    return int(dim)


def _validate_finite_scalar(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validate_nonnegative_scalar(value: object, name: str) -> float:
    value = _validate_finite_scalar(value, name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_positive_scalar(value: object, name: str) -> float:
    value = _validate_nonnegative_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_dt(dt: float) -> float:
    return _validate_nonnegative_scalar(dt, "dt")


def _as_vector(value: object, length: int, name: str) -> np.ndarray:
    array = _as_real_array(value, name)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_covariance(
    value: object, size: int, name: str, *, positive_definite: bool = False
) -> np.ndarray:
    array = _as_real_array(value, name)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diag(array) < 0):
        raise ValueError(
            f"{name} must be positive semidefinite; diagonal variances must be nonnegative"
        )
    scale = float(np.max(np.abs(array)))
    tolerance = max(
        np.finfo(float).tiny,
        100.0 * np.finfo(float).eps * scale * size,
    )
    relative_tolerance = 100.0 * np.finfo(float).eps * size
    if not np.allclose(array, array.T, rtol=relative_tolerance, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    array = _symmetrize(array)
    try:
        values, vectors = np.linalg.eigh(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} could not be decomposed") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError(f"{name} decomposition is not finite")
    if float(values[0]) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if values[0] < 0:
        array = _symmetrize(vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} could not be repaired to a finite covariance")
    if positive_definite:
        _cholesky(array, name)
    return array.copy()


def _cholesky(matrix: np.ndarray, name: str) -> np.ndarray:
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    if not np.isfinite(factor).all():
        raise FloatingPointError(f"{name} Cholesky factor is not finite")
    return factor


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray, name: str) -> np.ndarray:
    factor = _cholesky(matrix, name)
    solution = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    if not np.isfinite(solution).all():
        raise FloatingPointError(f"{name} solve is not finite")
    return solution


def _finite_difference(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
    if not np.isfinite(difference).all():
        raise FloatingPointError(f"{name} is not finite")
    return difference


def _gaussian_log_likelihood(y: np.ndarray, covariance: np.ndarray) -> float:
    factor = _cholesky(covariance, "innovation covariance")
    whitened = np.linalg.solve(factor, y)
    log_det = 2.0 * float(np.log(np.diag(factor)).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic = float(whitened @ whitened)
        result = float(-0.5 * (len(y) * np.log(2.0 * np.pi) + log_det + quadratic))
    if not np.isfinite(result):
        raise FloatingPointError("Gaussian log likelihood is not finite")
    return result


def _squared_mahalanobis(y: np.ndarray, covariance: np.ndarray) -> float:
    solution = _solve_spd(covariance, y, "innovation covariance")
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(y @ solution)
    if not np.isfinite(distance) or distance < -1e-10:
        raise FloatingPointError("gating distance is not finite and nonnegative")
    return max(distance, 0.0)


def _likelihood_from_log(log_likelihood: float) -> float:
    if not np.isfinite(log_likelihood):
        raise FloatingPointError("log likelihood must be finite")
    lower = float(np.log(np.finfo(float).tiny))
    upper = float(np.log(np.finfo(float).max))
    return float(np.exp(np.clip(log_likelihood, lower, upper)))


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _sample_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    """Draw from a possibly singular Gaussian without backend matmul warnings."""
    mean = _as_vector(mean, len(mean), f"{name} mean")
    covariance = _as_covariance(covariance, len(mean), f"{name} covariance")
    try:
        vectors, values, _ = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} covariance could not be factored") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError(f"{name} covariance factor is not finite")
    root = vectors * np.sqrt(np.maximum(values, 0.0))[None, :]
    standard = rng.standard_normal((size, len(mean)))
    with np.errstate(over="ignore", invalid="ignore"):
        samples = np.einsum("sj,ij->si", standard, root) + mean
    if not np.isfinite(samples).all():
        raise FloatingPointError(f"{name} samples are not finite")
    return samples


@dataclass
class GaussianState:
    """A Gaussian belief ``N(x, P)``."""

    x: np.ndarray
    P: np.ndarray

    def __post_init__(self) -> None:
        x = _as_real_array(self.x, "x")
        if x.ndim != 1 or len(x) == 0:
            raise ValueError(f"x must be a non-empty 1-D vector, got shape {x.shape}")
        if len(x) > 2 * MAX_FILTER_DIMENSION:
            raise ValueError(f"x must contain at most {2 * MAX_FILTER_DIMENSION} state coordinates")
        self.x = _as_vector(x, len(x), "x")
        self.P = _as_covariance(self.P, len(x), "P")

    def copy(self) -> GaussianState:
        return GaussianState(self.x.copy(), self.P.copy())

    @property
    def position(self) -> np.ndarray:
        return self.x[: len(self.x) // 2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[len(self.x) // 2 :]


def _validate_filter_state(state: object, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(state, GaussianState):
        raise TypeError("state must be a GaussianState")
    size = 2 * dim
    state.x = _as_vector(state.x, size, "state.x")
    state.P = _as_covariance(state.P, size, "state.P")
    return state.x, state.P


def _measurement_inputs(
    z: np.ndarray, R: np.ndarray, dim: int, *, covariance_positive_definite: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _as_vector(z, dim, "z"),
        _as_covariance(
            R,
            dim,
            "R",
            positive_definite=covariance_positive_definite,
        ),
    )


def _polar_measurement_inputs(
    z: np.ndarray, R: np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    z, R = _measurement_inputs(z, R, dim)
    if z[0] <= MIN_POLAR_HORIZONTAL_RANGE:
        raise ValueError(f"polar range must be > {MIN_POLAR_HORIZONTAL_RANGE:g} m")
    if abs(z[1]) > 1_000_000.0:
        raise ValueError("polar azimuth magnitude is too large to canonicalize reliably")
    z[1] = wrap_angle(z[1])
    if not -np.pi / 2.0 <= z[2] <= np.pi / 2.0:
        raise ValueError("polar elevation must be in [-pi/2, pi/2]")
    if z[0] * abs(float(np.cos(z[2]))) <= MIN_POLAR_HORIZONTAL_RANGE:
        raise ValueError("polar azimuth is singular on the sensor's vertical axis")
    return z, R


# ---------------------------------------------------------------------------
# Kalman filter (linear, Cartesian)
# ---------------------------------------------------------------------------
class KalmanFilter:
    """Linear Kalman filter on the CV model with Joseph-stabilised covariance."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, sigma_a: float = 3.0, dim: int = POS_DIM):
        self.dim = _validate_dim(dim)
        self.sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        self.state = GaussianState(
            _as_vector(x0, 2 * self.dim, "x0"),
            _as_covariance(P0, 2 * self.dim, "P0"),
        )
        self.H = position_measurement_matrix(self.dim)
        self._last_likelihood = 1.0
        self._last_log_likelihood = 0.0

    # -- prediction ------------------------------------------------------
    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        x, P = _validate_filter_state(self.state, self.dim)
        if dt == 0:
            return
        F = cv_transition(dt, self.dim)
        Q = cv_process_noise(dt, self.sigma_a, self.dim)
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_x = F @ x
            raw_predicted_P = F @ P @ F.T + Q
        if not np.isfinite(predicted_x).all():
            raise FloatingPointError("predicted state is not finite")
        predicted_P = _as_covariance(raw_predicted_P, 2 * self.dim, "predicted P")
        self.state.x, self.state.P = predicted_x, predicted_P

    # -- Cartesian update ------------------------------------------------
    def innovation(self, z: np.ndarray) -> np.ndarray:
        z = _as_vector(z, self.dim, "z")
        x, _ = _validate_filter_state(self.state, self.dim)
        return _finite_difference(z, self.H @ x, "innovation")

    def innovation_covariance(self, R: np.ndarray) -> np.ndarray:
        R = _as_covariance(R, self.dim, "R")
        _, P = _validate_filter_state(self.state, self.dim)
        return _as_covariance(
            self.H @ P @ self.H.T + R,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        """Squared Mahalanobis distance of measurement ``z`` from the prediction."""
        y = self.innovation(z)
        S = self.innovation_covariance(R)
        return _squared_mahalanobis(y, S)

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        z, R = _measurement_inputs(z, R, self.dim)
        x, P = _validate_filter_state(self.state, self.dim)
        y = _finite_difference(z, self.H @ x, "innovation")
        S = _as_covariance(
            self.H @ P @ self.H.T + R,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )
        PHt = P @ self.H.T
        K = _solve_spd(S, PHt.T, "innovation covariance").T
        I = np.eye(P.shape[0])
        A = I - K @ self.H
        with np.errstate(over="ignore", invalid="ignore"):
            updated_x = x + K @ y
            raw_updated_P = A @ P @ A.T + K @ R @ K.T
        if not np.isfinite(updated_x).all():
            raise FloatingPointError("updated state is not finite")
        updated_P = _as_covariance(raw_updated_P, 2 * self.dim, "updated P")
        log_likelihood = _gaussian_log_likelihood(y, S)
        likelihood = _likelihood_from_log(log_likelihood)
        self.state.x, self.state.P = updated_x, updated_P
        self._last_log_likelihood = log_likelihood
        self._last_likelihood = likelihood

    @property
    def likelihood(self) -> float:
        return self._last_likelihood

    @property
    def log_likelihood(self) -> float:
        return self._last_log_likelihood


# ---------------------------------------------------------------------------
# Extended Kalman filter (adds polar / radar update)
# ---------------------------------------------------------------------------
class ExtendedKalmanFilter(KalmanFilter):
    """KF plus a polar measurement update for native radar geometry.

    ``update_polar`` consumes ``z = [range, azimuth, elevation]`` relative to a
    ``sensor_origin``; the angular error is modelled in polar space (a diagonal
    Cartesian covariance would misrepresent it).
    """

    def update_polar(
        self, z: np.ndarray, R: np.ndarray, sensor_origin: np.ndarray | None = None
    ) -> None:
        if self.dim != 3:
            raise ValueError("polar radar updates require dim=3")
        z, R = _polar_measurement_inputs(z, R, self.dim)
        s = (
            np.zeros(self.dim)
            if sensor_origin is None
            else _as_vector(sensor_origin, self.dim, "sensor_origin")
        )
        x, P = _validate_filter_state(self.state, self.dim)
        h, Hj = self._polar_h_and_jacobian(s)
        y = _finite_difference(z, h, "polar innovation")
        y[1] = wrap_angle(y[1])  # azimuth
        y[2] = wrap_angle(y[2])  # elevation
        S = _as_covariance(
            Hj @ P @ Hj.T + R,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )
        PHt = P @ Hj.T
        K = _solve_spd(S, PHt.T, "innovation covariance").T
        I = np.eye(P.shape[0])
        A = I - K @ Hj
        with np.errstate(over="ignore", invalid="ignore"):
            updated_x = x + K @ y
            raw_updated_P = A @ P @ A.T + K @ R @ K.T
        if not np.isfinite(updated_x).all():
            raise FloatingPointError("updated state is not finite")
        updated_P = _as_covariance(raw_updated_P, 2 * self.dim, "updated P")
        log_likelihood = _gaussian_log_likelihood(y, S)
        likelihood = _likelihood_from_log(log_likelihood)
        self.state.x, self.state.P = updated_x, updated_P
        self._last_log_likelihood = log_likelihood
        self._last_likelihood = likelihood

    def _polar_h_and_jacobian(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, _ = _validate_filter_state(self.state, self.dim)
        s = _as_vector(s, self.dim, "sensor_origin")
        p = x[: self.dim]
        with np.errstate(over="ignore", invalid="ignore"):
            relative = p - s
        if not np.isfinite(relative).all():
            raise FloatingPointError("relative radar position is not finite")
        dx, dy, dz = relative
        # hypot avoids overflow for large but finite coordinates. Rewriting the
        # Jacobian as products of unit-vector components avoids squaring them.
        rho = float(np.hypot(dx, dy))
        r = float(np.hypot(rho, dz))
        if r <= MIN_POLAR_HORIZONTAL_RANGE or rho <= MIN_POLAR_HORIZONTAL_RANGE:
            raise ValueError("predicted polar geometry is singular at the sensor origin or axis")
        h = np.array([r, np.arctan2(dy, dx), np.arctan2(dz, rho)])
        H = np.zeros((3, 2 * self.dim))
        # d range
        H[0, 0], H[0, 1], H[0, 2] = dx / r, dy / r, dz / r
        # d azimuth
        H[1, 0], H[1, 1] = -(dy / rho) / rho, (dx / rho) / rho
        # d elevation
        H[2, 0] = -(dx / rho) * (dz / r) / r
        H[2, 1] = -(dy / rho) * (dz / r) / r
        H[2, 2] = (rho / r) / r
        if not np.isfinite(h).all() or not np.isfinite(H).all():
            raise FloatingPointError("polar measurement geometry is not finite")
        return h, H


# ---------------------------------------------------------------------------
# Unscented Kalman filter
# ---------------------------------------------------------------------------
class UnscentedKalmanFilter:
    """Scaled unscented KF (Van der Merwe sigma points), Cartesian position update."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        sigma_a: float = 3.0,
        dim: int = POS_DIM,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dim = _validate_dim(dim)
        self.n = 2 * self.dim
        self.sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        self.state = GaussianState(
            _as_vector(x0, self.n, "x0"),
            _as_covariance(P0, self.n, "P0"),
        )
        self.H = position_measurement_matrix(self.dim)
        self.alpha = _validate_positive_scalar(alpha, "alpha")
        self.beta = _validate_nonnegative_scalar(beta, "beta")
        self.kappa = _validate_finite_scalar(kappa, "kappa")
        if self.n + self.kappa <= 0:
            raise ValueError("kappa must satisfy n + kappa > 0")
        try:
            with np.errstate(over="raise", under="ignore", invalid="raise"):
                self._sigma_scale = float(np.square(np.float64(self.alpha)) * (self.n + self.kappa))
        except FloatingPointError as exc:
            raise ValueError("UKF parameters produce a non-finite sigma-point scale") from exc
        if not np.isfinite(self._sigma_scale) or self._sigma_scale <= 0:
            raise ValueError("UKF scaling must satisfy n + lambda > 0")
        self.lambda_ = self._sigma_scale - self.n
        self._wm, self._wc = self._weights()
        self._last_likelihood = 1.0
        self._last_log_likelihood = 0.0

    def _weights(self) -> tuple[np.ndarray, np.ndarray]:
        n, lam, scale = self.n, self.lambda_, self._sigma_scale
        wm = np.full(2 * n + 1, 1.0 / (2 * scale))
        wc = wm.copy()
        wm[0] = lam / scale
        wc[0] = lam / scale + (1 - self.alpha**2 + self.beta)
        if not np.isfinite(wm).all() or not np.isfinite(wc).all():
            raise ValueError("UKF parameters produce non-finite sigma-point weights")
        cancellation_error = np.finfo(float).eps * len(wm) * float(np.max(np.abs(wm)))
        if cancellation_error > 1e-6 or not np.isclose(wm.sum(), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("UKF parameters produce numerically unstable sigma-point weights")
        return wm, wc

    def _sigma_points(self) -> np.ndarray:
        n = self.n
        x, P = _validate_filter_state(self.state, self.dim)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = self._sigma_scale * P
        if not np.isfinite(scaled).all():
            raise FloatingPointError("scaled sigma-point covariance is not finite")
        try:
            U = np.linalg.cholesky(scaled)
        except np.linalg.LinAlgError:
            # A covariance may legitimately be singular. An eigen square root
            # represents that PSD distribution exactly instead of injecting
            # arbitrary jitter into the posterior.
            values, vectors = np.linalg.eigh(scaled)
            scale = max(1.0, float(np.max(np.abs(scaled))))
            if float(values[0]) < -1e-10 * scale:
                raise ValueError(
                    "scaled sigma-point covariance must be positive semidefinite"
                ) from None
            U = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
        if not np.isfinite(U).all():
            raise FloatingPointError("sigma points are not finite")
        pts = np.zeros((2 * n + 1, n))
        pts[0] = x
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n):
                pts[1 + i] = x + U[:, i]
                pts[1 + n + i] = x - U[:, i]
        if not np.isfinite(pts).all():
            raise FloatingPointError("sigma points are not finite")
        return pts

    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        _validate_filter_state(self.state, self.dim)
        if dt == 0:
            return
        F = cv_transition(dt, self.dim)
        Q = cv_process_noise(dt, self.sigma_a, self.dim)
        pts = self._sigma_points()
        with np.errstate(over="ignore", invalid="ignore"):
            prop = pts @ F.T
            x = self._wm @ prop
        if not np.isfinite(x).all():
            raise FloatingPointError("predicted state is not finite")
        P = Q.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(prop.shape[0]):
                d = prop[i] - x
                P += self._wc[i] * np.outer(d, d)
        updated_P = _as_covariance(P, self.n, "predicted P")
        self.state.x, self.state.P = x, updated_P

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        z, R = _measurement_inputs(z, R, self.dim)
        x_prior, P_prior = _validate_filter_state(self.state, self.dim)
        pts = self._sigma_points()
        with np.errstate(over="ignore", invalid="ignore"):
            zpts = pts @ self.H.T
            zhat = self._wm @ zpts
        if not np.isfinite(zhat).all():
            raise FloatingPointError("predicted measurement is not finite")
        S = R.copy()
        Pxz = np.zeros((self.n, self.dim))
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(zpts.shape[0]):
                dz = zpts[i] - zhat
                dx = pts[i] - self.state.x
                S += self._wc[i] * np.outer(dz, dz)
                Pxz += self._wc[i] * np.outer(dx, dz)
        S = _as_covariance(
            S,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )
        K = _solve_spd(S, Pxz.T, "innovation covariance").T
        y = _finite_difference(z, zhat, "innovation")
        with np.errstate(over="ignore", invalid="ignore"):
            updated_x = x_prior + K @ y
            raw_updated_P = P_prior - K @ S @ K.T
        if not np.isfinite(updated_x).all():
            raise FloatingPointError("updated state is not finite")
        updated_P = _as_covariance(raw_updated_P, self.n, "updated P")
        log_likelihood = _gaussian_log_likelihood(y, S)
        likelihood = _likelihood_from_log(log_likelihood)
        self.state.x, self.state.P = updated_x, updated_P
        self._last_log_likelihood = log_likelihood
        self._last_likelihood = likelihood

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        z, R = _measurement_inputs(z, R, self.dim)
        x, P = _validate_filter_state(self.state, self.dim)
        y = _finite_difference(z, self.H @ x, "innovation")
        S = _as_covariance(
            self.H @ P @ self.H.T + R,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )
        return _squared_mahalanobis(y, S)

    @property
    def likelihood(self) -> float:
        return self._last_likelihood

    @property
    def log_likelihood(self) -> float:
        return self._last_log_likelihood


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------
class ParticleFilter:
    """Bootstrap particle filter for non-Gaussian / multi-modal posteriors."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        sigma_a: float = 3.0,
        dim: int = POS_DIM,
        n_particles: int = 512,
        rng: np.random.Generator | None = None,
    ):
        self.dim = _validate_dim(dim)
        self.sigma_a = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        self.H = position_measurement_matrix(self.dim)
        if isinstance(n_particles, bool) or not isinstance(n_particles, (int, np.integer)):
            raise ValueError("n_particles must be a positive integer")
        self.n_particles = int(n_particles)
        if not 1 <= self.n_particles <= MAX_FILTER_PARTICLES:
            raise ValueError(f"n_particles must be an integer in [1, {MAX_FILTER_PARTICLES}]")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None")
        self.rng = rng if rng is not None else np.random.default_rng()
        x0 = _as_vector(x0, 2 * self.dim, "x0")
        P0 = _as_covariance(P0, 2 * self.dim, "P0")
        self.particles: np.ndarray = _sample_gaussian(
            self.rng,
            x0,
            P0,
            self.n_particles,
            "initial particle",
        )
        self.weights: np.ndarray = np.full(self.n_particles, 1.0 / self.n_particles)
        self._last_likelihood = 1.0
        self._last_log_likelihood = 0.0
        self._validate_population()

    def _validate_population(self) -> tuple[np.ndarray, np.ndarray]:
        expected = (self.n_particles, 2 * self.dim)
        particles = np.asarray(self.particles, float)
        if particles.shape != expected:
            raise ValueError(f"particles must have shape {expected}, got {particles.shape}")
        if not np.isfinite(particles).all():
            raise ValueError("particles must contain only finite values")
        weights = np.asarray(self.weights, float)
        if weights.shape != (self.n_particles,):
            raise ValueError(f"weights must have shape ({self.n_particles},), got {weights.shape}")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("weights must be finite and nonnegative")
        total = float(weights.sum())
        if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("weights must sum to 1")
        self.particles = particles
        self.weights = weights
        return particles, weights

    @property
    def state(self) -> GaussianState:
        particles, weights = self._validate_population()
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.einsum("n,nd->d", weights, particles)
        if not np.isfinite(x).all():
            raise FloatingPointError("particle mean is not finite")
        d = _finite_difference(particles, x, "particle deviations")
        with np.errstate(over="ignore", invalid="ignore"):
            P = np.einsum("n,ni,nj->ij", weights, d, d)
        return GaussianState(x, _as_covariance(P, 2 * self.dim, "particle covariance"))

    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        particles, _ = self._validate_population()
        if dt == 0:
            return
        F = cv_transition(dt, self.dim)
        Q = cv_process_noise(dt, self.sigma_a, self.dim)
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            noise = _sample_gaussian(
                self.rng,
                np.zeros(2 * self.dim),
                Q,
                self.n_particles,
                "process-noise",
            )
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = np.einsum("ni,ji->nj", particles, F) + noise
            if not np.isfinite(predicted).all():
                raise FloatingPointError("predicted particles are not finite")
            self.particles = predicted
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        z, R = _measurement_inputs(z, R, self.dim, covariance_positive_definite=True)
        particles, weights = self._validate_population()
        factor = _cholesky(R, "R")
        with np.errstate(over="ignore", invalid="ignore"):
            pred = np.einsum("ni,ji->nj", particles, self.H)
        if not np.isfinite(pred).all():
            raise FloatingPointError("predicted particle measurements are not finite")
        d = _finite_difference(pred, z, "particle innovations")
        whitened = np.linalg.solve(factor, d.T)
        with np.errstate(over="ignore", invalid="ignore"):
            maha = np.sum(whitened**2, axis=0)
        log_det = 2.0 * float(np.log(np.diag(factor)).sum())
        log_measurement = -0.5 * (self.dim * np.log(2.0 * np.pi) + log_det + maha)
        log_prior = np.full(self.n_particles, -np.inf)
        positive = weights > 0
        log_prior[positive] = np.log(weights[positive])
        log_posterior = log_prior + log_measurement
        log_evidence = _logsumexp(log_posterior)
        if not np.isfinite(log_evidence):
            raise FloatingPointError("particle update has zero or non-finite evidence")
        updated_weights = np.exp(log_posterior - log_evidence)
        if not np.isfinite(updated_weights).all() or updated_weights.sum() <= 0:
            raise FloatingPointError("particle weights are not finite and positive")
        self.weights = updated_weights / updated_weights.sum()
        self._last_log_likelihood = log_evidence
        self._last_likelihood = _likelihood_from_log(log_evidence)
        if self._ess() < len(self.particles) / 2:
            self._resample()

    def _ess(self) -> float:
        _, weights = self._validate_population()
        return 1.0 / np.sum(weights**2)

    def _resample(self) -> None:
        particles, weights = self._validate_population()
        n = self.n_particles
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        idx = np.clip(idx, 0, n - 1)
        self.particles = particles[idx].copy()
        self.weights = np.full(n, 1.0 / n)

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        z, R = _measurement_inputs(z, R, self.dim)
        st = self.state
        y = _finite_difference(z, self.H @ st.x, "innovation")
        S = _as_covariance(
            self.H @ st.P @ self.H.T + R,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )
        return _squared_mahalanobis(y, S)

    @property
    def likelihood(self) -> float:
        return self._last_likelihood

    @property
    def log_likelihood(self) -> float:
        return self._last_log_likelihood


# ---------------------------------------------------------------------------
# Interacting Multiple Model estimator
# ---------------------------------------------------------------------------
_MAX_IMM_MODELS = 32


class IMMEstimator:
    """IMM over a bank of KF-like models mixed by a Markov transition matrix.

    Each model must expose ``.state`` (a :class:`GaussianState`), ``predict(dt)``,
    ``update(z, R)`` and ``.likelihood``. The default bank is two constant-
    velocity Kalman filters — a quiescent one and a high-manoeuvre one — a
    standard maneuver-adaptive configuration. A coordinated-turn model can
    approximate the same model-family choice used elsewhere, but parity still
    requires full configuration and intermediate-state fixtures.
    """

    def __init__(
        self,
        models: list,
        transition: np.ndarray | None = None,
        mode_probs: np.ndarray | None = None,
    ):
        if isinstance(models, (str, bytes)):
            raise TypeError("models must be a non-empty sequence of filter models")
        try:
            self.models = list(models)
        except TypeError as exc:
            raise TypeError("models must be a non-empty sequence of filter models") from exc
        m = len(self.models)
        if m == 0:
            raise ValueError("models must not be empty")
        if m > _MAX_IMM_MODELS:
            raise ValueError(f"models exceeds the {_MAX_IMM_MODELS}-mode safety limit")
        if len({id(model) for model in self.models}) != m:
            raise ValueError("models must contain distinct filter objects")

        dimensions: list[int] = []
        for index, model in enumerate(self.models):
            model_dim, state = self._validate_model(model, index)
            if not hasattr(model, "__dict__"):
                raise TypeError(
                    f"models[{index}] must expose snapshotable instance state for transactions"
                )
            # Mixing replaces each model's Gaussian prior. Read-only state
            # properties (for example ParticleFilter.state) are not compatible
            # with that contract and should fail here, not halfway through predict.
            try:
                model.state = state.copy()
            except (AttributeError, TypeError) as exc:
                raise TypeError(f"models[{index}].state must be assignable") from exc
            self._model_log_likelihood(model, index)
            dimensions.append(model_dim)
        if len(set(dimensions)) != 1:
            raise ValueError("all IMM models must have the same dimension")
        self.dim = dimensions[0]

        if transition is None:
            if m == 1:
                transition = np.ones((1, 1))
            else:
                transition = np.full((m, m), 0.05 / (m - 1))
                np.fill_diagonal(transition, 0.95)
        self.transition = self._validate_transition(transition, m)
        probabilities = np.full(m, 1.0 / m) if mode_probs is None else mode_probs
        self.mode_probs = self._validate_probabilities(probabilities, m, "mode_probs")
        # Predicted mode probabilities; set here so update() before the first
        # predict() degrades to the prior instead of raising AttributeError.
        self._cbar = self.mode_probs.copy()

    @staticmethod
    def _validate_model(model, index: int) -> tuple[int, GaussianState]:
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"models[{index}] must provide predict(dt)")
        if not callable(getattr(model, "update", None)):
            raise TypeError(f"models[{index}] must provide update(z, R)")
        if not hasattr(model, "likelihood"):
            raise TypeError(f"models[{index}] must expose likelihood")
        model_dim = _validate_dim(getattr(model, "dim", None))
        state = getattr(model, "state", None)
        _validate_filter_state(state, model_dim)
        assert isinstance(state, GaussianState)
        return model_dim, state

    @staticmethod
    def _model_log_likelihood(model, index: int) -> float:
        if hasattr(model, "log_likelihood"):
            raw_value = model.log_likelihood
            if isinstance(raw_value, (bool, np.bool_)):
                raise FloatingPointError(f"models[{index}] produced an invalid log likelihood")
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise FloatingPointError(
                    f"models[{index}] produced an invalid log likelihood"
                ) from exc
            if np.isnan(value) or np.isposinf(value):
                raise FloatingPointError(f"models[{index}] produced an invalid log likelihood")
            return value

        raw_value = model.likelihood
        if isinstance(raw_value, (bool, np.bool_)):
            raise FloatingPointError(f"models[{index}] produced an invalid likelihood")
        try:
            likelihood = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FloatingPointError(f"models[{index}] produced an invalid likelihood") from exc
        if not np.isfinite(likelihood) or likelihood < 0:
            raise FloatingPointError(f"models[{index}] produced an invalid likelihood")
        return -np.inf if likelihood == 0 else float(np.log(likelihood))

    @staticmethod
    def _validate_transition(transition: np.ndarray, size: int) -> np.ndarray:
        try:
            matrix = np.asarray(transition, float)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition must be a numeric probability matrix") from exc
        if matrix.shape != (size, size):
            raise ValueError(f"transition must have shape ({size}, {size}), got {matrix.shape}")
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("transition must contain finite nonnegative probabilities")
        if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("each transition row must sum to 1")
        return matrix.copy()

    @staticmethod
    def _validate_probabilities(values: np.ndarray, size: int, name: str) -> np.ndarray:
        try:
            probabilities = np.asarray(values, float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric probability vector") from exc
        if probabilities.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {probabilities.shape}")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(f"{name} must contain finite nonnegative probabilities")
        if not np.isclose(probabilities.sum(), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must sum to 1")
        return probabilities.copy()

    def _validate_runtime(self) -> int:
        if not isinstance(self.models, (list, tuple)) or len(self.models) == 0:
            raise ValueError("models must remain a non-empty sequence")
        size = len(self.models)
        if size > _MAX_IMM_MODELS:
            raise ValueError(f"models exceeds the {_MAX_IMM_MODELS}-mode safety limit")
        if len({id(model) for model in self.models}) != size:
            raise ValueError("models must retain distinct filter objects")
        for index, model in enumerate(self.models):
            model_dim, _ = self._validate_model(model, index)
            if model_dim != self.dim:
                raise ValueError("all IMM models must retain the configured dimension")
        self.transition = self._validate_transition(self.transition, size)
        self.mode_probs = self._validate_probabilities(self.mode_probs, size, "mode_probs")
        return size

    @staticmethod
    def _snapshot_model_fields(models: list) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for index, model in enumerate(models):
            if not hasattr(model, "__dict__"):
                raise TypeError(
                    f"models[{index}] must expose snapshotable instance state for transactions"
                )
            try:
                snapshots.append(copy.deepcopy(vars(model)))
            except BaseException as exc:
                raise TypeError(
                    f"models[{index}] state must be deepcopyable for transactional updates"
                ) from exc
        return snapshots

    @staticmethod
    def _restore_model_fields(models: list, snapshots: list[dict[str, object]]) -> None:
        for model, snapshot in zip(models, snapshots):
            namespace = vars(model)
            namespace.clear()
            namespace.update(copy.deepcopy(snapshot))

    @classmethod
    def default_cv_bank(
        cls,
        x0,
        P0,
        dim: int = POS_DIM,
        sigma_a: float = 1.0,
    ) -> IMMEstimator:
        # Use EKF models so the bank honours the radar polar contract (they behave
        # as a linear KF for Cartesian updates and add update_polar for radar). The
        # configured acceleration scale is the quiet model; the maneuver model has
        # ten times that standard deviation while retaining the historical 1:10
        # default ratio.
        quiet_sigma = _validate_nonnegative_scalar(sigma_a, "sigma_a")
        if quiet_sigma > np.finfo(float).max / 10.0:
            raise ValueError("sigma_a is too large for the IMM maneuver model")
        maneuver_sigma = quiet_sigma * 10.0
        quiet = ExtendedKalmanFilter(x0, P0, sigma_a=quiet_sigma, dim=dim)
        maneuver = ExtendedKalmanFilter(
            x0,
            np.asarray(P0, float).copy(),
            sigma_a=maneuver_sigma,
            dim=dim,
        )
        return cls([quiet, maneuver])

    # -- mixing → predict → update → combine -----------------------------
    def _mix(self) -> list[GaussianState]:
        m = self._validate_runtime()
        cbar = self.transition.T @ self.mode_probs  # predicted mode probs
        if not np.isfinite(cbar).all() or np.any(cbar < 0):
            raise FloatingPointError("predicted IMM mode probabilities are invalid")
        mixed = []
        for j in range(m):
            if cbar[j] == 0:
                mixed.append(self.models[j].state.copy())
                continue
            mu_ij = self.transition[:, j] * self.mode_probs / cbar[j]
            mixing_total = float(mu_ij.sum())
            if not np.isfinite(mu_ij).all() or mixing_total <= 0:
                raise FloatingPointError("IMM mixing probabilities are invalid")
            mu_ij /= mixing_total
            with np.errstate(over="ignore", invalid="ignore"):
                x0 = sum(
                    (mu_ij[i] * self.models[i].state.x for i in range(m)),
                    start=np.zeros_like(self.models[j].state.x),
                )
            if not np.isfinite(x0).all():
                raise FloatingPointError("mixed IMM state is not finite")
            P0 = np.zeros_like(self.models[j].state.P)
            with np.errstate(over="ignore", invalid="ignore"):
                for i in range(m):
                    d = self.models[i].state.x - x0
                    P0 += mu_ij[i] * (self.models[i].state.P + np.outer(d, d))
            mixed.append(GaussianState(x0, _as_covariance(P0, 2 * self.dim, "mixed P")))
        self._cbar = self._validate_probabilities(cbar, m, "predicted mode probabilities")
        return mixed

    def predict(self, dt: float) -> None:
        dt = _validate_dt(dt)
        self._validate_runtime()
        if dt == 0:
            return
        original_models = self._snapshot_model_fields(self.models)
        original_mode_probs = self.mode_probs.copy()
        original_cbar = self._cbar.copy()
        try:
            mixed = self._mix()
            for model, mstate in zip(self.models, mixed):
                model.state = mstate.copy()
                model.predict(dt)
                _validate_filter_state(model.state, self.dim)
            # The transition step changes the mode prior even before a measurement
            # arrives. Keeping mode_probs at the previous posterior would make gating
            # and the combined predicted state use stale probabilities.
            self.mode_probs = self._cbar.copy()
        except BaseException:
            self._restore_model_fields(self.models, original_models)
            self.mode_probs = original_mode_probs
            self._cbar = original_cbar
            raise

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        z, R = _measurement_inputs(z, R, self.dim)
        self._update_each(lambda m: m.update(z, R))

    def update_polar(self, z: np.ndarray, R: np.ndarray, sensor_origin=None) -> None:
        """Polar (radar) update across the bank; requires EKF-style models."""
        if self.dim != 3:
            raise ValueError("polar radar updates require dim=3")
        if any(not callable(getattr(model, "update_polar", None)) for model in self.models):
            raise TypeError("every IMM model must provide update_polar for radar updates")
        z, R = _polar_measurement_inputs(z, R, self.dim)
        origin = (
            None if sensor_origin is None else _as_vector(sensor_origin, self.dim, "sensor_origin")
        )
        self._update_each(lambda m: m.update_polar(z, R, origin))

    def _update_each(self, do_update) -> None:
        size = self._validate_runtime()
        original_models = self._snapshot_model_fields(self.models)
        original_mode_probs = self.mode_probs.copy()
        original_cbar = self._cbar.copy()
        try:
            log_likelihoods = np.empty(size)
            for k, model in enumerate(self.models):
                do_update(model)
                _validate_filter_state(model.state, self.dim)
                log_likelihoods[k] = self._model_log_likelihood(model, k)
            # A track may receive one update from each modality in the same cycle.
            # Accumulate that evidence from the current posterior rather than reusing
            # the pre-update cbar prior and silently discarding earlier modalities.
            prior_logs = np.full(size, -np.inf)
            positive = self.mode_probs > 0
            prior_logs[positive] = np.log(self.mode_probs[positive])
            posterior_logs = prior_logs + log_likelihoods
            normalizer = _logsumexp(posterior_logs)
            if not np.isfinite(normalizer):
                raise FloatingPointError("IMM mode posterior has zero or non-finite evidence")
            posterior = np.exp(posterior_logs - normalizer)
            self.mode_probs = self._validate_probabilities(posterior, size, "mode_probs")
            self._cbar = self.mode_probs.copy()
        except BaseException:
            self._restore_model_fields(self.models, original_models)
            self.mode_probs = original_mode_probs
            self._cbar = original_cbar
            raise

    @property
    def state(self) -> GaussianState:
        self._validate_runtime()
        with np.errstate(over="ignore", invalid="ignore"):
            x = sum(
                (mu * model.state.x for mu, model in zip(self.mode_probs, self.models)),
                start=np.zeros_like(self.models[0].state.x),
            )
        if not np.isfinite(x).all():
            raise FloatingPointError("combined IMM state is not finite")
        P = np.zeros_like(self.models[0].state.P)
        with np.errstate(over="ignore", invalid="ignore"):
            for mu, model in zip(self.mode_probs, self.models):
                d = model.state.x - x
                P += mu * (model.state.P + np.outer(d, d))
        return GaussianState(x, _as_covariance(P, 2 * self.dim, "combined P"))

    def gating_distance(self, z: np.ndarray, R: np.ndarray) -> float:
        z, R = _measurement_inputs(z, R, self.dim)
        st = self.state
        H = position_measurement_matrix(self.dim)
        y = _finite_difference(z, H @ st.x, "innovation")
        S = _as_covariance(
            H @ st.P @ H.T + R,
            self.dim,
            "innovation covariance",
            positive_definite=True,
        )
        return _squared_mahalanobis(y, S)


FILTERS = {
    "kalman": KalmanFilter,
    "ekf": ExtendedKalmanFilter,
    "ukf": UnscentedKalmanFilter,
    "particle": ParticleFilter,
    "imm": IMMEstimator,
}

__all__ = [
    "POS_DIM",
    "STATE_DIM",
    "MIN_POLAR_HORIZONTAL_RANGE",
    "GaussianState",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "IMMEstimator",
    "cv_transition",
    "cv_process_noise",
    "position_measurement_matrix",
    "wrap_angle",
    "FILTERS",
]
