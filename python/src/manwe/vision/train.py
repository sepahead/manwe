"""Config-driven, from-scratch training of aerial-object detector architectures.

Runs the same recipe on Metal (MPS) or CUDA — the device is resolved through
:func:`manwe.common.resolve_device`, never hard-coded. Heavy deps are imported
inside :func:`train`, so this module (and its config) import with numpy alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import MappingProxyType

from ..common.config_io import validate_local_path
from ..common.dataset_manifest import validate_local_detection_manifest
from ..common.device import Device, resolve_device
from ..common.logging import get_logger
from ..common.seed import seed_everything
from .models import DEFAULT_DETECTOR, MODEL_ZOO, build_model

log = get_logger("manwe.vision")

_ULTRALYTICS_SAFE_EXTRA = {
    "amp",
    "box",
    "cls",
    "cos_lr",
    "degrees",
    "deterministic",
    "dfl",
    "erasing",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "multi_scale",
    "optimizer",
    "perspective",
    "plots",
    "rect",
    "save",
    "save_period",
    "scale",
    "shear",
    "single_cls",
    "translate",
    "verbose",
    "warmup_bias_lr",
    "warmup_epochs",
    "warmup_momentum",
    "weight_decay",
}
_RFDETR_SAFE_EXTRA = {
    "checkpoint_interval",
    "ema_decay",
    "gradient_accumulation_steps",
    "lr_component_decay",
    "lr_drop",
    "lr_encoder",
    "num_workers",
    "use_ema",
    "weight_decay",
}

_ULTRALYTICS_BOOL_EXTRA = {
    "amp",
    "cos_lr",
    "deterministic",
    "plots",
    "rect",
    "save",
    "single_cls",
    "verbose",
}


def _extra_bool(extra: Mapping[str, object], key: str) -> None:
    if key in extra and type(extra[key]) is not bool:
        raise ValueError(f"extra.{key} must be a boolean")


def _extra_integer(extra: Mapping[str, object], key: str, minimum: int, maximum: int) -> None:
    if key not in extra:
        return
    value = extra[key]
    if type(value) is not int:
        raise ValueError(f"extra.{key} must be an integer in [{minimum}, {maximum}]")
    assert isinstance(value, int)
    if not minimum <= value <= maximum:
        raise ValueError(f"extra.{key} must be an integer in [{minimum}, {maximum}]")


def _extra_float(
    extra: Mapping[str, object], key: str, minimum: float, maximum: float, *, open_min: bool = False
) -> None:
    if key not in extra:
        return
    value = extra[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"extra.{key} must be a finite number")
    number = float(value)
    valid_minimum = number > minimum if open_min else number >= minimum
    if not math.isfinite(number) or not valid_minimum or number > maximum:
        bracket = "(" if open_min else "["
        raise ValueError(f"extra.{key} must be in {bracket}{minimum}, {maximum}]")


def _validate_extra_values(extra: Mapping[str, object], family: str, epochs: int) -> None:
    if family == "rfdetr":
        _extra_integer(extra, "checkpoint_interval", 1, epochs)
        _extra_integer(extra, "gradient_accumulation_steps", 1, 4096)
        _extra_integer(extra, "lr_drop", 0, 100_000)
        _extra_integer(extra, "num_workers", 0, 256)
        _extra_bool(extra, "use_ema")
        _extra_float(extra, "ema_decay", 0.0, 1.0, open_min=True)
        _extra_float(extra, "lr_component_decay", 0.0, 1.0, open_min=True)
        _extra_float(extra, "lr_encoder", 0.0, 1.0, open_min=True)
        _extra_float(extra, "weight_decay", 0.0, 10.0)
        return

    for key in _ULTRALYTICS_BOOL_EXTRA:
        _extra_bool(extra, key)
    _extra_integer(extra, "save_period", -1, epochs)
    if extra.get("save_period") == 0:
        raise ValueError("extra.save_period must be -1 or a positive epoch interval")
    for key in ("box", "cls", "dfl", "weight_decay"):
        _extra_float(extra, key, 0.0, 100.0)
    for key in ("erasing", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v"):
        _extra_float(extra, key, 0.0, 1.0)
    _extra_float(extra, "degrees", 0.0, 180.0)
    _extra_float(extra, "multi_scale", 0.0, 1.0)
    _extra_float(extra, "perspective", 0.0, 0.001)
    _extra_float(extra, "scale", 0.0, 1.0)
    _extra_float(extra, "shear", 0.0, 180.0)
    _extra_float(extra, "translate", 0.0, 1.0)
    _extra_float(extra, "warmup_bias_lr", 0.0, 10.0)
    _extra_float(extra, "warmup_epochs", 0.0, float(epochs))
    _extra_float(extra, "warmup_momentum", 0.0, 1.0)
    if "optimizer" in extra:
        optimizer = extra["optimizer"]
        if not isinstance(optimizer, str) or optimizer.lower() not in {
            "sgd",
            "adam",
            "adamw",
            "nadam",
            "radam",
            "rmsprop",
            "auto",
        }:
            raise ValueError("extra.optimizer is not a supported optimizer name")


def _reject_ambiguous_opencv_install() -> None:
    installed = []
    for distribution in (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ):
        with suppress(metadata.PackageNotFoundError):
            installed.append((distribution, metadata.version(distribution)))
    if len(installed) != 1:
        versions = ", ".join(f"{name} {version}" for name, version in installed)
        raise RuntimeError(
            "RF-DETR training requires exactly one OpenCV distribution to own the cv2 package; "
            f"found {versions or 'none'}. Create a curated environment with exactly one OpenCV "
            "distribution before training."
        )


@dataclass(frozen=True, slots=True)
class VisionTrainConfig:
    """From-scratch training configuration (see ``configs/vision``)."""

    data: str  # Ultralytics YAML file or RF-DETR COCO dataset directory
    model: str = DEFAULT_DETECTOR
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"  # auto | cuda | mps | cpu | cuda:0
    lr0: float = 0.01
    lrf: float = 0.01
    patience: int = 50
    pretrained: bool = False  # true is rejected until a digest-bound local adapter exists
    seed: int = 1337
    project: str = "runs/vision"
    name: str = "aerial"
    # augmentation (small aerial objects benefit from heavy mosaic + copy-paste)
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    close_mosaic: int = 10
    # optional export after training, e.g. ("onnx", "coreml")
    export_formats: tuple[str, ...] = ()
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in MODEL_ZOO:
            raise ValueError(f"unknown model {self.model!r}; choose from {list(MODEL_ZOO)}")
        if not isinstance(self.data, str) or not self.data.strip():
            raise ValueError("data must be a nonempty dataset path")
        if (
            any(character in self.data for character in "\0\r\n")
            or len(self.data.encode("utf-8")) > 4096
        ):
            raise ValueError("data contains control characters or is too long")
        integer_ranges = {
            "epochs": (1, 100_000),
            "imgsz": (32, 4096),
            "batch": (1, 4096),
        }
        for name, (minimum, maximum) in integer_ranges.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if type(self.patience) is not int or not 0 <= self.patience <= 100_000:
            raise ValueError("patience must be an integer in [0, 100000]")
        for name in ("lr0", "lrf"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.lrf > 1.0:
            raise ValueError("lrf must not exceed 1")
        for name in ("mosaic", "mixup", "copy_paste"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if type(self.close_mosaic) is not int or not 0 <= self.close_mosaic <= 100_000:
            raise ValueError("close_mosaic must be an integer in [0, 100000]")
        if type(self.seed) is not int or not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if type(self.pretrained) is not bool:
            raise TypeError("pretrained must be a boolean")
        if self.pretrained:
            raise ValueError(
                "backend-managed pretrained downloads are disabled; use pretrained=false"
            )
        for name in ("device", "project", "name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
            if (
                any(character in value for character in "\0\r\n")
                or len(value.encode("utf-8")) > 4096
            ):
                raise ValueError(f"{name} contains control characters or is too long")
        if self.name in {".", ".."} or any(separator in self.name for separator in ("/", "\\")):
            raise ValueError("name must be a single output-directory component")
        if not isinstance(self.extra, Mapping):
            raise ValueError("extra must be a mapping")
        try:
            extra = dict(self.extra)
        except (TypeError, ValueError) as exc:
            raise ValueError("extra must be a stable mapping") from exc
        if len(extra) > 128:
            raise ValueError("extra must contain at most 128 fields")
        if any(
            not isinstance(key, str)
            or not key
            or not key.isprintable()
            or len(key.encode("utf-8")) > 128
            for key in extra
        ):
            raise ValueError("extra keys must be printable strings of 1..128 UTF-8 bytes")
        if not isinstance(self.export_formats, (tuple, list)):
            raise ValueError("export_formats must be a sequence")
        export_formats = tuple(self.export_formats)
        if export_formats:
            raise ValueError(
                "post-training export is intentionally separate: run `manwe export` and "
                "then build a contract and execute the fidelity gate"
            )
        object.__setattr__(self, "extra", MappingProxyType(extra))
        object.__setattr__(self, "export_formats", export_formats)
        family = MODEL_ZOO[self.model].family
        _validate_extra_values(extra, family, self.epochs)
        if family == "rfdetr":
            nondefault_ultralytics_fields = [
                name
                for name, default in {
                    "lrf": 0.01,
                    "mosaic": 1.0,
                    "mixup": 0.1,
                    "copy_paste": 0.1,
                    "close_mosaic": 10,
                }.items()
                if getattr(self, name) != default
            ]
            if nondefault_ultralytics_fields:
                raise ValueError(
                    "RF-DETR does not consume these Ultralytics-only fields: "
                    f"{nondefault_ultralytics_fields}"
                )


def resolve_ultralytics_device(device: Device) -> str:
    """Map a manwe :class:`Device` to the string Ultralytics expects."""
    if device.kind == "cuda":
        return str(device.index)
    return device.kind  # "mps" or "cpu"


def train(config: VisionTrainConfig):
    """Train a detector architecture from random initialization.

    Backend-managed downloads and local checkpoint loading are not supported by
    this entry point. The backend result object is returned; raw conversion is a
    separate provenance-bound operation in :mod:`manwe.export`.
    """
    data_path = Path(config.data).expanduser().absolute()

    spec = MODEL_ZOO[config.model]
    if spec.family == "rfdetr":
        try:
            validate_local_path(data_path, "RF-DETR dataset", require_directory=True)
        except ValueError as exc:
            raise ValueError(
                "RF-DETR requires a non-symlinked COCO dataset directory, not a YAML manifest"
            ) from exc
        reserved = {
            "dataset_dir",
            "epochs",
            "batch_size",
            "device",
            "lr",
            "output_dir",
            "resolution",
            "early_stopping",
            "early_stopping_patience",
        }
        overlap = reserved.intersection(config.extra)
        if overlap:
            raise ValueError(f"extra may not override managed RF-DETR fields: {sorted(overlap)}")
        _reject_ambiguous_opencv_install()
        unsupported = sorted(set(config.extra) - _RFDETR_SAFE_EXTRA)
        if unsupported:
            raise ValueError(f"unsupported RF-DETR extra fields: {unsupported}")
    else:
        reserved = {
            "data",
            "epochs",
            "imgsz",
            "batch",
            "device",
            "lr0",
            "lrf",
            "patience",
            "seed",
            "project",
            "name",
            "mosaic",
            "mixup",
            "copy_paste",
            "close_mosaic",
        }
        overlap = reserved.intersection(config.extra)
        if overlap:
            raise ValueError(
                f"extra may not override managed Ultralytics fields: {sorted(overlap)}"
            )
        unsupported = sorted(set(config.extra) - _ULTRALYTICS_SAFE_EXTRA)
        if unsupported:
            raise ValueError(f"unsupported Ultralytics extra fields: {unsupported}")

    manifest_snapshot = None
    try:
        if spec.family != "rfdetr":
            manifest_snapshot = validate_local_detection_manifest(data_path)

        # No global RNG mutation, device probing, dependency import, or checkpoint
        # download happens until the complete family-specific preflight has passed.
        device = resolve_device(config.device)
        seed_everything(config.seed)
        log.info(
            "training %s on %s (data=%s, epochs=%d)",
            config.model,
            device,
            config.data,
            config.epochs,
        )
        model = build_model(config.model, pretrained=config.pretrained)
        if spec.family == "rfdetr":
            return model.train(
                dataset_dir=str(data_path),
                epochs=config.epochs,
                batch_size=config.batch,
                device=device.torch_device,
                lr=config.lr0,
                output_dir=str(Path(config.project) / config.name),
                resolution=config.imgsz,
                early_stopping=True,
                early_stopping_patience=config.patience,
                **config.extra,
            )

        assert manifest_snapshot is not None
        uld = resolve_ultralytics_device(device)
        return model.train(
            data=str(manifest_snapshot.path),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=uld,
            lr0=config.lr0,
            lrf=config.lrf,
            patience=config.patience,
            seed=config.seed,
            project=config.project,
            name=config.name,
            mosaic=config.mosaic,
            mixup=config.mixup,
            copy_paste=config.copy_paste,
            close_mosaic=config.close_mosaic,
            **config.extra,
        )
    finally:
        if manifest_snapshot is not None:
            manifest_snapshot.close()


__all__ = ["VisionTrainConfig", "train", "resolve_ultralytics_device"]
