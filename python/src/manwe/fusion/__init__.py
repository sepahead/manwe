"""Independent multi-target tracking and multi-sensor fusion reference.

Public API::

    from manwe.fusion import MultiSensorTracker, TrackerConfig, Measurement
    from manwe.fusion import make_scenario, score_tracker
    from manwe.fusion import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
    from manwe.fusion import ParticleFilter, IMMEstimator
    from manwe.fusion import ospa, gospa
"""

from __future__ import annotations

from .association import associate, linear_assignment
from .filters import (
    ExtendedKalmanFilter,
    GaussianState,
    IMMEstimator,
    KalmanFilter,
    ParticleFilter,
    UnscentedKalmanFilter,
)
from .metrics import gospa, ospa, rmse
from .scenarios import Scenario, make_scenario, score_tracker
from .tracker import (
    Measurement,
    MultiSensorTracker,
    TrackerConfig,
    TrackOutput,
    measurement_cartesian,
    radar_polar_to_cartesian,
)

__all__ = [
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "IMMEstimator",
    "GaussianState",
    "Measurement",
    "MultiSensorTracker",
    "TrackerConfig",
    "TrackOutput",
    "measurement_cartesian",
    "radar_polar_to_cartesian",
    "associate",
    "linear_assignment",
    "make_scenario",
    "score_tracker",
    "Scenario",
    "ospa",
    "gospa",
    "rmse",
]
