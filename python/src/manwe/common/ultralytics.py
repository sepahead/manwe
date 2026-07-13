"""Process-wide safety policy for loading the optional Ultralytics runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys

_NETWORK_INTEGRATIONS = (
    "sync",
    "clearml",
    "comet",
    "dvc",
    "hub",
    "mlflow",
    "neptune",
    "raytune",
    "wandb",
)
_VETTED_ULTRALYTICS_VERSION = "8.4.92"


def _blocked_download(*_args, **_kwargs):
    raise RuntimeError(
        "Ultralytics network downloads are disabled; provide every dependency and artifact "
        "through the locked environment and verified local paths"
    )


def harden_ultralytics_runtime() -> None:
    """Disable runtime package installation and enable restricted checkpoint loading.

    This must run before importing Ultralytics. The already-imported settings are
    also overwritten defensively so another library cannot silently retain the
    unsafe defaults in the same process.
    """
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["ULTRALYTICS_SAFE_LOAD"] = "true"
    os.environ["YOLO_OFFLINE"] = "true"
    if "ultralytics" not in sys.modules:
        return
    ultralytics = importlib.import_module("ultralytics")
    settings = getattr(ultralytics, "SETTINGS", None)
    if isinstance(settings, dict):
        # Bypass SettingsManager.__setitem__ so this process-local safety policy
        # does not rewrite a user's global Ultralytics settings file.
        for key in _NETWORK_INTEGRATIONS:
            if key in settings:
                dict.__setitem__(settings, key, False)
    utils = importlib.import_module("ultralytics.utils")
    vars(utils)["AUTOINSTALL"] = False
    vars(utils)["ONLINE"] = False
    checks = importlib.import_module("ultralytics.utils.checks")
    vars(checks)["AUTOINSTALL"] = False
    vars(checks)["ONLINE"] = False
    downloads = importlib.import_module("ultralytics.utils.downloads")
    vars(downloads)["safe_download"] = _blocked_download
    tasks = importlib.import_module("ultralytics.nn.tasks")
    vars(tasks)["SAFE_LOAD"] = True
    events_module = importlib.import_module("ultralytics.utils.events")
    events = getattr(events_module, "events", None)
    if events is not None:
        events.enabled = False
        events.events = []


def verify_ultralytics_policy() -> None:
    """Assert that an imported Ultralytics process obeys Manwe's safety policy."""
    harden_ultralytics_runtime()
    try:
        installed_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Ultralytics distribution metadata is unavailable") from exc
    if installed_version != _VETTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Ultralytics {installed_version} is not the vetted "
            f"{_VETTED_ULTRALYTICS_VERSION} runtime"
        )
    utils = importlib.import_module("ultralytics.utils")
    checks = importlib.import_module("ultralytics.utils.checks")
    tasks = importlib.import_module("ultralytics.nn.tasks")
    if getattr(utils, "AUTOINSTALL", None) is not False:
        raise RuntimeError("Ultralytics runtime dependency installation could not be disabled")
    if getattr(checks, "AUTOINSTALL", None) is not False:
        raise RuntimeError("Ultralytics checks retained runtime dependency installation")
    if getattr(utils, "ONLINE", None) is not False or getattr(checks, "ONLINE", None) is not False:
        raise RuntimeError("Ultralytics runtime did not enter offline mode")
    if getattr(tasks, "SAFE_LOAD", None) is not True:
        raise RuntimeError("Ultralytics restricted checkpoint loading could not be enabled")
    safe_loader = getattr(tasks, "_SafeLoad", None)
    if safe_loader is None or getattr(safe_loader, "SUPPORTED", False) is not True:
        raise RuntimeError(
            "this Torch/Ultralytics combination cannot enforce restricted checkpoint loading"
        )
    ultralytics = importlib.import_module("ultralytics")
    settings = getattr(ultralytics, "SETTINGS", {})
    enabled = [key for key in _NETWORK_INTEGRATIONS if settings.get(key)]
    if enabled:
        raise RuntimeError(f"Ultralytics network integrations remain enabled: {enabled}")
    events_module = importlib.import_module("ultralytics.utils.events")
    if getattr(getattr(events_module, "events", None), "enabled", True):
        raise RuntimeError("Ultralytics analytics could not be disabled")


__all__ = ["harden_ultralytics_runtime", "verify_ultralytics_policy"]
