"""Configuration files must preserve the types their dataclasses expect."""

from pathlib import Path

import pytest

from manwe.fusion import TrackerConfig


def test_tracker_yaml_scientific_notation_loads_as_number():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "configs" / "fusion" / "tracker.yaml"
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    assert isinstance(values["max_position_cov_volume"], float)
    config = TrackerConfig(**values)
    assert config.max_position_cov_volume == 1.0e9
