"""Process-wide safety policy for loading the optional Ultralytics runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import threading
from contextlib import contextmanager
from urllib import parse

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
_FORMAT_PATCH_LOCK = threading.RLock()
_MAX_FORMAT_IMAGE_DIMENSION = 32_768
_MAX_FORMAT_IMAGE_PIXELS = 32_000_000


def _blocked_download(*_args, **_kwargs):
    raise RuntimeError(
        "Ultralytics network downloads are disabled; provide every dependency and artifact "
        "through the locked environment and verified local paths"
    )


def _offline_is_url(url, check: bool = False) -> bool:
    """Recognize URL syntax without ever probing the network."""
    try:
        result = parse.urlparse(str(url))
    except Exception:
        return False
    syntactically_valid = bool(result.scheme and result.netloc)
    return syntactically_valid and not check


def _ignore_dataset_cache(*_args, **_kwargs):
    """Force the pinned dataset loader down its fresh local-scan path."""
    raise FileNotFoundError("Ultralytics dataset cache loading is disabled")


def _retain_cache_metadata_without_writing(_prefix, _path, payload, version) -> None:
    """Preserve the loader's in-memory cache contract without creating a pickle file."""
    payload["version"] = version


def _deterministic_format_image(formatter, image):
    """Match pinned Format._format_img while making validation BGR conversion total."""
    import numpy as np
    import torch

    if type(image) is not np.ndarray or image.dtype != np.uint8:
        raise ValueError("calibration image must be one uint8 numpy array")
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        raise ValueError("calibration image must have shape (height, width) or (height, width, C)")
    if height <= 0 or width <= 0 or channels not in {1, 3}:
        raise ValueError(
            "calibration image must have positive dimensions and one or three channels"
        )
    if height > _MAX_FORMAT_IMAGE_DIMENSION or width > _MAX_FORMAT_IMAGE_DIMENSION:
        raise ValueError(
            f"calibration image dimensions must not exceed {_MAX_FORMAT_IMAGE_DIMENSION}"
        )
    if height * width > _MAX_FORMAT_IMAGE_PIXELS:
        raise ValueError(
            f"calibration image exceeds the {_MAX_FORMAT_IMAGE_PIXELS}-pixel safety limit"
        )
    if image.ndim == 2:
        image = image[..., None]
    image = image.transpose(2, 0, 1)
    bgr_probability = formatter.bgr
    if bgr_probability != 0:
        raise RuntimeError("deterministic calibration formatting requires Ultralytics bgr=0")
    reverse_channels = image.shape[0] == 3
    image = np.ascontiguousarray(image[::-1] if reverse_channels else image)
    return torch.from_numpy(image)


@contextmanager
def deterministic_ultralytics_validation_format():
    """Temporarily make the pinned validation BGR conversion deterministic."""
    # Format._format_img is process-global. Holding the re-entrant lock for the
    # complete export prevents overlapping contexts from restoring out of order.
    with _FORMAT_PATCH_LOCK:
        augment = importlib.import_module("ultralytics.data.augment")
        format_class = augment.Format
        original = format_class._format_img

        def calibration_format_image(formatter, image):
            # TensorRT may invoke Python calibrator callbacks from a builder
            # thread. bgr=0 is the exact audited validation route and has the
            # same deterministic channel reversal on every calling thread;
            # unrelated nonzero-BGR formatters retain their original behavior.
            if formatter.bgr == 0:
                return _deterministic_format_image(formatter, image)
            return original(formatter, image)

        format_class._format_img = calibration_format_image
        try:
            if (
                augment.Format is not format_class
                or format_class._format_img is not calibration_format_image
            ):
                raise RuntimeError(
                    "Ultralytics validation image formatting could not be made deterministic"
                )
            yield
            if (
                augment.Format is not format_class
                or format_class._format_img is not calibration_format_image
            ):
                raise RuntimeError(
                    "Ultralytics validation image formatting changed during calibration"
                )
        finally:
            format_class._format_img = original


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
    vars(downloads)["is_url"] = _offline_is_url
    data_utils = importlib.import_module("ultralytics.data.utils")
    vars(data_utils)["load_dataset_cache_file"] = _ignore_dataset_cache
    vars(data_utils)["save_dataset_cache_file"] = _retain_cache_metadata_without_writing
    data_dataset = importlib.import_module("ultralytics.data.dataset")
    vars(data_dataset)["load_dataset_cache_file"] = _ignore_dataset_cache
    vars(data_dataset)["save_dataset_cache_file"] = _retain_cache_metadata_without_writing
    patches = importlib.import_module("ultralytics.utils.patches")
    from PIL import Image

    original_image_open = getattr(patches, "_image_open", None)
    if callable(original_image_open):
        Image.open = original_image_open
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
    downloads = importlib.import_module("ultralytics.utils.downloads")
    data_utils = importlib.import_module("ultralytics.data.utils")
    data_dataset = importlib.import_module("ultralytics.data.dataset")
    patches = importlib.import_module("ultralytics.utils.patches")
    from PIL import Image

    tasks = importlib.import_module("ultralytics.nn.tasks")
    if getattr(utils, "AUTOINSTALL", None) is not False:
        raise RuntimeError("Ultralytics runtime dependency installation could not be disabled")
    if getattr(checks, "AUTOINSTALL", None) is not False:
        raise RuntimeError("Ultralytics checks retained runtime dependency installation")
    if getattr(utils, "ONLINE", None) is not False or getattr(checks, "ONLINE", None) is not False:
        raise RuntimeError("Ultralytics runtime did not enter offline mode")
    if (
        getattr(downloads, "safe_download", None) is not _blocked_download
        or getattr(downloads, "is_url", None) is not _offline_is_url
    ):
        raise RuntimeError("Ultralytics download and network-probe functions remain enabled")
    if (
        getattr(data_utils, "load_dataset_cache_file", None) is not _ignore_dataset_cache
        or getattr(data_dataset, "load_dataset_cache_file", None) is not _ignore_dataset_cache
        or getattr(data_utils, "save_dataset_cache_file", None)
        is not _retain_cache_metadata_without_writing
        or getattr(data_dataset, "save_dataset_cache_file", None)
        is not _retain_cache_metadata_without_writing
    ):
        raise RuntimeError("Ultralytics dataset cache I/O could not be disabled")
    original_image_open = getattr(patches, "_image_open", None)
    if not callable(original_image_open) or Image.open is not original_image_open:
        raise RuntimeError("Ultralytics optional-codec Pillow hook could not be disabled")
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


__all__ = [
    "deterministic_ultralytics_validation_format",
    "harden_ultralytics_runtime",
    "verify_ultralytics_policy",
]
