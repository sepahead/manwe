"""Regression tests locking in the fixes from the 10-lens review."""

import builtins
import os

import numpy as np
import pytest

from manwe.cli import main
from manwe.fusion.filters import IMMEstimator
from manwe.fusion.tracker import Measurement, measurement_cartesian
from manwe.multicam.triangulation import triangulate_midpoint


def test_core_is_import_clean_without_torch():
    """Invariant #1: a fresh pure-numpy core import must not load torch."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    code = """
import sys
for name in ('manwe.common', 'manwe.fusion', 'manwe.multicam', 'manwe.audio', 'manwe.eval'):
    __import__(name)
raise SystemExit(1 if 'torch' in sys.modules else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or "a core module imported torch"


def test_imm_update_before_predict_does_not_crash():
    """_cbar must be initialised in __init__ (dt==0 frames update without predict)."""
    x0 = np.zeros(6)
    P0 = np.diag([25.0] * 3 + [100.0] * 3)
    imm = IMMEstimator.default_cv_bank(x0, P0)
    imm.update(np.array([1.0, 2.0, 3.0]), np.diag([4.0, 4.0, 9.0]))  # must not raise
    assert abs(imm.mode_probs.sum() - 1.0) < 1e-9


def test_imm_bank_honours_radar_polar():
    """default_cv_bank uses EKFs, so update_polar exists on the bank."""
    x0 = np.zeros(6)
    x0[:3] = [90.0, 1.0, 1.0]
    P0 = np.diag([25.0] * 3 + [100.0] * 3)
    imm = IMMEstimator.default_cv_bank(x0, P0)
    assert hasattr(imm, "update_polar")
    before = imm.state.x.copy()
    imm.update_polar(np.array([100.0, 0.0, 0.0]), np.diag([9.0, 1e-4, 1e-4]), np.zeros(3))
    assert not np.array_equal(imm.state.x, before)


def test_parallel_ray_gap_is_perpendicular_distance():
    # two parallel +x rays offset 3 m in y and 100 m in x → gap must be 3, not 100
    pt, gap = triangulate_midpoint(
        np.zeros(3), np.array([1.0, 0, 0]), np.array([100.0, 3.0, 0]), np.array([1.0, 0, 0])
    )
    assert abs(gap - 3.0) < 1e-6, f"gap {gap} should be perpendicular distance 3.0"


def test_radar_covariance_transform_exact_values():
    # range=100, az=0, el=0 with polar var [9, 4e-4, 4e-4] → cartesian diag [9, 4, 4]
    m = Measurement("radar", [100.0, 0.0, 0.0], [9.0, 4e-4, 4e-4], 0.0)
    _, cov = measurement_cartesian(m)
    assert np.allclose(cov, np.diag([9.0, 4.0, 4.0]), atol=1e-6), cov


def test_unknown_modality_rejected():
    raised = False
    try:
        Measurement("sonar", [1, 2, 3], [1, 1, 1], 0.0)
    except ValueError:
        raised = True
    assert raised


def test_cli_rejects_unknown_filter():
    raised = False
    try:
        main(["fusion-sim", "--filters", "bogus", "--duration", "4"])
    except SystemExit:  # argparse choices → exit(2)
        raised = True
    assert raised


def test_fusion_sim_clutter_over_capacity_is_a_usage_error_not_a_traceback():
    """A --clutter that overruns the tracker's bounded capacity exits cleanly."""
    with pytest.raises(SystemExit) as excinfo:
        main(["fusion-sim", "--targets", "1", "--duration", "2", "--clutter", "8000"])
    assert excinfo.value.code == 2


def test_resolve_device_auto_falls_back_when_torch_fails_to_load(monkeypatch):
    """`manwe doctor` (resolve_device('auto')) must not crash on a broken torch."""
    from manwe.common import device as device_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("simulated broken torch shared library")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device_module.resolve_device("auto").kind == "cpu"
    # An explicit accelerator request still fails closed.
    with pytest.raises(RuntimeError):
        device_module.resolve_device("cuda")


def test_open_regular_nofollow_rejects_a_fifo_without_hanging(tmp_path):
    """O_NONBLOCK means a FIFO is rejected promptly instead of blocking forever."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is unavailable on this platform")
    from manwe.common.config_io import open_regular_nofollow

    fifo = tmp_path / "pipe.yaml"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):  # completing at all proves it did not hang
        open_regular_nofollow(fifo.resolve(), "config")
