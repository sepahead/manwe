"""Portable, machine-consumable model and runtime contracts.

Schema 2 binds an exact artifact digest to its tensor interface, image
preprocessing, detector decoding policy, source taxonomy, and optional mapping
into Manwe's five-class airspace taxonomy. Python export tooling emits the
sidecar, and Manwe's native Rust CLI and viewer validate it before loading
model weights. Python's research checkpoint wrappers use their own explicit,
digest-bound admission boundary rather than pretending to implement this native
runtime contract.

The module is intentionally dependency-free (stdlib only). A valid contract
proves internal consistency and, when requested, sibling-artifact identity. It
does not claim accuracy, calibration, rights, or downstream compatibility beyond
the evidence recorded by the producer.
"""

from __future__ import annotations

import html
import json
import math
import os
import pathlib
import re
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, NoReturn, cast

from .artifacts import DEFAULT_MAX_ARTIFACT_BYTES, sha256_artifact_at
from .config_io import open_directory_nofollow, read_bounded_regular_bytes_at
from .fd_io import attach_cleanup_failure

# ---------------------------------------------------------------------------
# Manwe's local candidate taxonomy.
# ---------------------------------------------------------------------------
AirspaceClass = Literal["drone", "bird", "aircraft", "helicopter", "unknown"]

#: Canonical, index-ordered class list. The integer index is the value written
#: into an exported model's class dimension and referenced by its contract.
AIRSPACE_CLASSES: tuple[AirspaceClass, ...] = (
    "drone",
    "bird",
    "aircraft",
    "helicopter",
    "unknown",
)

_CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(AIRSPACE_CLASSES)}


def airspace_class_index(name: str) -> int:
    """Return the canonical index for a Manwe airspace class name.

    >>> airspace_class_index("drone")
    0
    """
    try:
        return _CLASS_INDEX[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"{name!r} is not a Manwe airspace class; expected one of {AIRSPACE_CLASSES}"
        ) from exc


# COCO (80-class) does not contain "drone" or "helicopter". A stock COCO YOLO
# therefore cannot cover the local taxonomy. This fallback maps the COCO names
# that do correspond for bounded producer-side experiments; it is not a consumer
# adapter and does not imply drone or helicopter capability.
_COCO_TO_AIRSPACE: dict[str, AirspaceClass] = {
    "airplane": "aircraft",
    "bird": "bird",
    # Everything else (person, car, ...) is not an airspace object → dropped.
}


def coco_to_airspace(coco_name: str) -> AirspaceClass | None:
    """Map a COCO class name to a Manwe class, or ``None`` if it has no aerial
    counterpart (the caller should drop such detections)."""
    return _COCO_TO_AIRSPACE.get(coco_name)


# Compatibility aliases for the pre-2.0 alpha API.  Manwe owns this taxonomy;
# naming it after one prospective consumer made the producer contract appear to
# promise an integration that does not exist.
CrebainClass = AirspaceClass
CREBAIN_CLASSES = AIRSPACE_CLASSES
crebain_class_index = airspace_class_index
coco_to_crebain = coco_to_airspace


# ---------------------------------------------------------------------------
# Model contract record.
# ---------------------------------------------------------------------------
# MLX ships as a .safetensors *file* but its backend is "mlx"; the file format is
# not a backend. Candle is Manwe's native Rust runtime.
Backend = Literal["onnx", "coreml", "mlx", "tensorrt", "candle"]

MODEL_CONTRACT_SCHEMA_VERSION = "2.0"
MAX_CONTRACT_CLASSES = 4096
MAX_CONTRACT_TENSORS = 64
MAX_TENSOR_RANK = 16
MAX_TENSOR_DIMENSION = 2**31 - 1

# The artifact suffix is part of the validated contract. CoreML artifacts are
# directory bundles; the other backends use regular files.
BACKEND_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "onnx": (".onnx",),
    "coreml": (".mlpackage", ".mlmodelc"),
    "mlx": (".safetensors",),
    "tensorrt": (".engine",),
    "candle": (".safetensors",),
}

MAX_CONTRACT_JSON_BYTES = 1 << 20
MAX_CONTRACT_JSON_NODES = 100_000
MAX_CONTRACT_JSON_DEPTH = 32
MAX_RUNTIME_DETECTIONS = 2_000
MAX_RUNTIME_PREDICTIONS = 2_000_000
MAX_NATIVE_RUNTIME_PREDICTIONS = 100_000
MAX_NATIVE_OUTPUT_ELEMENTS = 16_000_000
MAX_NATIVE_MODEL_BYTES = 1024 * 1024 * 1024
MAX_NATIVE_MODEL_CLASSES = 1_000

COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

RuntimeAdapter = Literal[
    "ultralytics-raw-detect-v1",
    "manwe-candle-yolov8-detect-v1",
    "manwe-candle-yolov8-pose-v1",
]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TENSOR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")
_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

# Contract spellings are deliberately canonical rather than accepting arbitrary
# backend display strings.  Inspectors must translate backend-specific names
# (for example ONNX ``tensor(float)``) into this vocabulary before recording a
# contract. A bounded vocabulary prevents a typo from silently becoming an
# apparent interface promise.
TENSOR_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)
TENSOR_LAYOUTS = frozenset(
    {
        "",
        "CHW",
        "CHW/BGR",
        "CHW/RGB",
        "HWC",
        "HWC/BGR",
        "HWC/RGB",
        "NC",
        "NCHW",
        "NCHW/BGR",
        "NCHW/RGB",
        "NHWC",
        "NHWC/BGR",
        "NHWC/RGB",
    }
)
TENSOR_DIMENSION_SYMBOLS = frozenset(
    {
        "A",
        "B",
        "C",
        "H",
        "N",
        "W",
        "anchors",
        "batch",
        "channels",
        "detections",
        "height",
        "max_det",
        "width",
    }
)


def _has_required_value(value: object) -> bool:
    if type(value) is str:
        return bool(value.strip())
    if type(value) in {list, tuple, dict}:
        return len(value) > 0  # type: ignore[arg-type]
    if type(value) is int:
        return value != 0
    return type(value) in {RuntimeSpec}


def _bounded_printable_utf8(
    value: object,
    *,
    max_bytes: int,
    allow_empty: bool = True,
    require_trimmed: bool = False,
) -> bool:
    """Validate text without leaking surrogate encoding failures to callers."""
    if type(value) is not str:
        return False
    text = cast(str, value)
    if (not allow_empty and not text) or (text and not text.isprintable()):
        return False
    if require_trimmed and text != text.strip():
        return False
    try:
        return len(text.encode("utf-8")) <= max_bytes
    except UnicodeEncodeError:
        return False


def _markdown_cell(value: object) -> str:
    return (
        html.escape(str(value), quote=True)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )


def _strict_json_object(
    value: object,
    name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    data = cast(dict[object, object], value)
    if any(type(key) is not str for key in data):
        raise ValueError(f"{name} field names must be strings")
    keys = cast(set[str], set(data))
    allowed = required | (optional or set())
    if unknown := keys - allowed:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")
    if missing := required - keys:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    return cast(dict[str, object], data)


def _finite_runtime_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return number


def _validate_native_probability_precision(value: float, name: str) -> None:
    """Reject a binary64 probability whose native binary32 value hits an endpoint.

    Native inference necessarily rounds thresholds to binary32. Ordinary interior
    rounding preserves the declared comparison policy, but an interior value that
    rounds to exactly zero or one changes which endpoint cases can pass. Keep that
    qualitative boundary explicit in the producer contract.
    """
    narrowed = struct.unpack("!f", struct.pack("!f", value))[0]
    if (value > 0.0 and narrowed == 0.0) or (value < 1.0 and narrowed == 1.0):
        raise ValueError(f"{name} collapses to a binary32 probability endpoint")


def _runtime_triplet(
    value: object,
    name: str,
    *,
    positive: bool,
) -> tuple[float, float, float]:
    if type(value) not in {list, tuple} or len(cast(Sequence[object], value)) != 3:
        raise ValueError(f"{name} must contain exactly three finite values")
    result = tuple(
        _finite_runtime_number(item, f"{name}[{index}]", positive=positive)
        for index, item in enumerate(cast(Sequence[object], value))
    )
    return cast(tuple[float, float, float], result)


def _runtime_names(
    value: object,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be a list or tuple")
    values = cast(Sequence[object], value)
    if (not values and not allow_empty) or len(values) > 256:
        raise ValueError(f"{name} has an invalid number of entries")
    result: list[str] = []
    for index, item in enumerate(values):
        if not _bounded_printable_utf8(
            item,
            max_bytes=256,
            allow_empty=False,
            require_trimmed=True,
        ):
            raise ValueError(f"{name}[{index}] must be a bounded printable name")
        result.append(cast(str, item))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique names")
    return tuple(result)


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"model contract JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"model contract JSON contains non-finite number {value}")


def _validate_json_graph(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_CONTRACT_JSON_NODES:
            raise ValueError(
                f"model contract JSON exceeds the {MAX_CONTRACT_JSON_NODES}-node safety limit"
            )
        if depth > MAX_CONTRACT_JSON_DEPTH:
            raise ValueError(
                f"model contract JSON exceeds the {MAX_CONTRACT_JSON_DEPTH}-level depth limit"
            )
        if type(current) is dict:
            pending.extend(
                (item, depth + 1) for item in cast(dict[object, object], current).values()
            )
        elif type(current) is list:
            pending.extend((item, depth + 1) for item in cast(list[object], current))
        elif type(current) is float and not math.isfinite(current):
            raise ValueError("model contract JSON contains a non-finite number")
        elif current is not None and type(current) not in {str, int, float, bool}:
            raise ValueError("model contract JSON contains a non-JSON value")


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _close_descriptor(
    descriptor: int,
    label: str,
    *,
    primary: BaseException | None = None,
) -> None:
    try:
        os.close(descriptor)
    except BaseException as cleanup:
        if primary is None:
            raise
        attach_cleanup_failure(primary, cleanup, label)


def _assert_bound_directory_visible(
    path: pathlib.Path,
    directory_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    if _directory_identity(os.fstat(directory_fd)) != expected_identity:
        raise ValueError("model package directory descriptor identity changed")
    visible_fd = open_directory_nofollow(path, "model package directory")
    operation_error: BaseException | None = None
    try:
        if _directory_identity(os.fstat(visible_fd)) != expected_identity:
            raise ValueError("model package directory path was replaced")
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        _close_descriptor(
            visible_fd,
            "visible model package directory cleanup also failed",
            primary=operation_error,
        )


def _verify_artifact_at(
    contract: ModelContract,
    parent_fd: int,
    artifact_name: str,
) -> None:
    suffix = pathlib.Path(artifact_name).suffix.lower()
    max_bytes = (
        MAX_NATIVE_MODEL_BYTES if contract.backend == "candle" else DEFAULT_MAX_ARTIFACT_BYTES
    )
    actual_sha256 = sha256_artifact_at(
        parent_fd,
        artifact_name,
        max_bytes=max_bytes,
        expect_directory=suffix in {".mlpackage", ".mlmodelc"},
    )
    if actual_sha256 != contract.file_sha256:
        raise ValueError(
            "artifact SHA-256 does not match file_sha256: "
            f"expected {contract.file_sha256}, got {actual_sha256}"
        )


@dataclass(slots=True)
class TensorSpec:
    """Description of a single model input or output tensor."""

    name: str
    shape: list[int | str]  # concrete dims or symbolic markers, e.g. ["B", 3, 640, 640]
    dtype: str  # "float32", "float16", "int8", ...
    layout: str = ""  # e.g. "NCHW", channel order "RGB"
    notes: str = ""

    def validation_errors(self, field_name: str) -> list[str]:
        """Return validation errors prefixed with ``field_name``."""
        errors: list[str] = []
        if type(self.name) is not str or not _TENSOR_NAME_RE.fullmatch(self.name):
            errors.append(f"{field_name}.name must be a 1..128 character ASCII tensor identifier")
        if type(self.dtype) is not str or self.dtype not in TENSOR_DTYPES:
            errors.append(f"{field_name}.dtype must be one of {tuple(sorted(TENSOR_DTYPES))}")
        if type(self.shape) is not list or not self.shape:
            errors.append(f"{field_name}.shape must be a nonempty list")
        elif len(self.shape) > MAX_TENSOR_RANK:
            errors.append(f"{field_name}.shape must contain at most {MAX_TENSOR_RANK} dimensions")
        else:
            shape_length = len(self.shape)
            shape_snapshot = self.shape[:shape_length]
            if len(shape_snapshot) != shape_length or len(self.shape) != shape_length:
                errors.append(f"{field_name}.shape changed while it was being validated")
            for index, dim in enumerate(shape_snapshot):
                if type(dim) not in {int, str}:
                    errors.append(
                        f"{field_name}.shape[{index}] must be a positive integer "
                        "or canonical symbolic dimension"
                    )
                elif type(dim) is int and dim <= 0:
                    errors.append(f"{field_name}.shape[{index}] must be positive")
                elif type(dim) is int and dim > MAX_TENSOR_DIMENSION:
                    errors.append(f"{field_name}.shape[{index}] exceeds {MAX_TENSOR_DIMENSION}")
                elif type(dim) is str and dim not in TENSOR_DIMENSION_SYMBOLS:
                    errors.append(
                        f"{field_name}.shape[{index}] must use one of the canonical "
                        f"symbols {tuple(sorted(TENSOR_DIMENSION_SYMBOLS))}"
                    )
        if type(self.layout) is not str or self.layout not in TENSOR_LAYOUTS:
            errors.append(f"{field_name}.layout must be one of {tuple(sorted(TENSOR_LAYOUTS))}")
        if type(self.notes) is not str:
            errors.append(f"{field_name}.notes must be a string")
        for name in ("name", "dtype", "layout", "notes"):
            value = getattr(self, name)
            if type(value) is str and not _bounded_printable_utf8(value, max_bytes=4096):
                errors.append(f"{field_name}.{name} must be bounded printable text")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object, field_name: str = "tensor") -> TensorSpec:
        data = _strict_json_object(
            value,
            field_name,
            required={"name", "shape", "dtype"},
            optional={"layout", "notes"},
        )
        shape = data["shape"]
        if type(shape) is not list:
            raise ValueError(f"{field_name}.shape must be a JSON array")
        tensor = cls(
            name=cast(str, data["name"]),
            shape=cast(list[int | str], shape.copy()),
            dtype=cast(str, data["dtype"]),
            layout=cast(str, data.get("layout", "")),
            notes=cast(str, data.get("notes", "")),
        )
        errors = tensor.validation_errors(field_name)
        if errors:
            raise ValueError("invalid tensor specification: " + "; ".join(errors))
        return tensor


@dataclass(frozen=True, slots=True)
class ImageInputSpec:
    """Exact source-pixel transformation applied before model execution.

    The v1 adapters intentionally expose a small, closed vocabulary.  A new
    preprocessing algorithm needs a new adapter identifier instead of another
    prose spelling that a runtime could interpret differently.
    """

    tensor_name: str
    width: int
    height: int
    dtype: str = "float32"
    layout: str = "NCHW/RGB"
    color_order: str = "RGB"
    resize: str = "letterbox"
    interpolation: str = "bilinear"
    alignment: int = 32
    pad_value: int = 114
    scale: float = 1.0 / 255.0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if type(self.tensor_name) is not str or not _TENSOR_NAME_RE.fullmatch(self.tensor_name):
            raise ValueError("runtime input tensor_name must be a canonical tensor identifier")
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or not 32 <= value <= 4096:
                raise ValueError(f"runtime input {name} must be an integer in [32, 4096]")
        if type(self.dtype) is not str or self.dtype not in {"float16", "float32"}:
            raise ValueError("runtime input dtype must be float16 or float32")
        if (
            type(self.layout) is not str
            or type(self.color_order) is not str
            or self.layout != "NCHW/RGB"
            or self.color_order != "RGB"
        ):
            raise ValueError("runtime input must use the NCHW/RGB layout and RGB channel order")
        if type(self.resize) is not str or self.resize != "letterbox":
            raise ValueError("runtime input resize must be letterbox")
        if type(self.interpolation) is not str or self.interpolation not in {
            "bilinear",
            "catmull-rom",
        }:
            raise ValueError("runtime interpolation must be bilinear or catmull-rom")
        if type(self.alignment) is not int or not 1 <= self.alignment <= 4096:
            raise ValueError("runtime input alignment must be an integer in [1, 4096]")
        if self.width % self.alignment != 0 or self.height % self.alignment != 0:
            raise ValueError("runtime input dimensions must be aligned to runtime input alignment")
        if type(self.pad_value) is not int or not 0 <= self.pad_value <= 255:
            raise ValueError("runtime input pad_value must be an integer in [0, 255]")
        scale = _finite_runtime_number(self.scale, "runtime input scale", positive=True)
        mean = _runtime_triplet(self.mean, "runtime input mean", positive=False)
        std = _runtime_triplet(self.std, "runtime input std", positive=True)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_name": self.tensor_name,
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "layout": self.layout,
            "color_order": self.color_order,
            "resize": self.resize,
            "interpolation": self.interpolation,
            "alignment": self.alignment,
            "pad_value": self.pad_value,
            "scale": self.scale,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, value: object) -> ImageInputSpec:
        data = _strict_json_object(
            value,
            "runtime.input",
            required={
                "tensor_name",
                "width",
                "height",
                "dtype",
                "layout",
                "color_order",
                "resize",
                "interpolation",
                "alignment",
                "pad_value",
                "scale",
                "mean",
                "std",
            },
        )
        return cls(
            tensor_name=cast(str, data["tensor_name"]),
            width=cast(int, data["width"]),
            height=cast(int, data["height"]),
            dtype=cast(str, data["dtype"]),
            layout=cast(str, data["layout"]),
            color_order=cast(str, data["color_order"]),
            resize=cast(str, data["resize"]),
            interpolation=cast(str, data["interpolation"]),
            alignment=cast(int, data["alignment"]),
            pad_value=cast(int, data["pad_value"]),
            scale=cast(float, data["scale"]),
            mean=cast(tuple[float, float, float], data["mean"]),
            std=cast(tuple[float, float, float], data["std"]),
        )


@dataclass(frozen=True, slots=True)
class DetectionOutputSpec:
    """Exact raw detector decoding and deployed postprocessing policy."""

    tensor_name: str
    box_format: str = "cxcywh"
    coordinate_space: str = "input-pixels"
    class_scores: str = "probabilities"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    keypoint_confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if type(self.tensor_name) is not str or not _TENSOR_NAME_RE.fullmatch(self.tensor_name):
            raise ValueError("runtime output tensor_name must be a canonical tensor identifier")
        if type(self.box_format) is not str or self.box_format != "cxcywh":
            raise ValueError("runtime detector box_format must be cxcywh")
        if type(self.coordinate_space) is not str or self.coordinate_space != "input-pixels":
            raise ValueError("runtime detector coordinate_space must be input-pixels")
        if type(self.class_scores) is not str or self.class_scores != "probabilities":
            raise ValueError("runtime detector class_scores must be probabilities")
        confidence = _finite_runtime_number(
            self.confidence_threshold,
            "runtime confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        iou = _finite_runtime_number(
            self.iou_threshold,
            "runtime iou_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            type(self.max_detections) is not int
            or not 1 <= self.max_detections <= MAX_RUNTIME_DETECTIONS
        ):
            raise ValueError(
                f"runtime max_detections must be an integer in [1, {MAX_RUNTIME_DETECTIONS}]"
            )
        keypoint_threshold = self.keypoint_confidence_threshold
        if keypoint_threshold is not None:
            keypoint_threshold = _finite_runtime_number(
                keypoint_threshold,
                "runtime keypoint_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            )
        object.__setattr__(self, "confidence_threshold", confidence)
        object.__setattr__(self, "iou_threshold", iou)
        object.__setattr__(self, "keypoint_confidence_threshold", keypoint_threshold)

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_name": self.tensor_name,
            "box_format": self.box_format,
            "coordinate_space": self.coordinate_space,
            "class_scores": self.class_scores,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold,
        }

    @classmethod
    def from_dict(cls, value: object) -> DetectionOutputSpec:
        data = _strict_json_object(
            value,
            "runtime.output",
            required={
                "tensor_name",
                "box_format",
                "coordinate_space",
                "class_scores",
                "confidence_threshold",
                "iou_threshold",
                "max_detections",
                "keypoint_confidence_threshold",
            },
        )
        return cls(
            tensor_name=cast(str, data["tensor_name"]),
            box_format=cast(str, data["box_format"]),
            coordinate_space=cast(str, data["coordinate_space"]),
            class_scores=cast(str, data["class_scores"]),
            confidence_threshold=cast(float, data["confidence_threshold"]),
            iou_threshold=cast(float, data["iou_threshold"]),
            max_detections=cast(int, data["max_detections"]),
            keypoint_confidence_threshold=cast(float | None, data["keypoint_confidence_threshold"]),
        )


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Machine-consumable adapter contract for one exact tensor interface."""

    adapter: RuntimeAdapter
    input: ImageInputSpec
    output: DetectionOutputSpec
    model_variant: str | None = None
    keypoint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        adapters = {
            "ultralytics-raw-detect-v1",
            "manwe-candle-yolov8-detect-v1",
            "manwe-candle-yolov8-pose-v1",
        }
        if type(self.adapter) is not str or self.adapter not in adapters:
            raise ValueError(f"runtime adapter must be one of {tuple(sorted(adapters))}")
        if type(self.input) is not ImageInputSpec or type(self.output) is not DetectionOutputSpec:
            raise TypeError("runtime input and output must be typed runtime specifications")
        if self.adapter.startswith("manwe-candle-yolov8-"):
            _validate_native_probability_precision(
                self.output.confidence_threshold,
                "runtime confidence_threshold",
            )
            _validate_native_probability_precision(
                self.output.iou_threshold,
                "runtime iou_threshold",
            )
            if self.output.keypoint_confidence_threshold is not None:
                _validate_native_probability_precision(
                    self.output.keypoint_confidence_threshold,
                    "runtime keypoint_confidence_threshold",
                )
            if type(self.model_variant) is not str or self.model_variant not in {
                "n",
                "s",
                "m",
                "l",
                "x",
            }:
                raise ValueError("Candle YOLOv8 runtime requires model_variant n, s, m, l, or x")
            if self.input.interpolation != "catmull-rom":
                raise ValueError("Candle YOLOv8 runtime requires catmull-rom preprocessing")
            if (
                self.input.width != self.input.height
                or self.input.alignment != 32
                or self.input.width % self.input.alignment != 0
            ):
                raise ValueError("Candle YOLOv8 runtime requires a square 32-aligned input")
            if (
                self.input.dtype != "float32"
                or self.input.pad_value != 114
                or self.input.scale != 1.0 / 255.0
                or self.input.mean != (0.0, 0.0, 0.0)
                or self.input.std != (1.0, 1.0, 1.0)
            ):
                raise ValueError(
                    "Candle YOLOv8 runtime preprocessing must match native prepare_image"
                )
        elif self.model_variant is not None:
            raise ValueError("ultralytics raw runtime does not use model_variant")
        names = _runtime_names(self.keypoint_names, "runtime keypoint_names", allow_empty=True)
        if self.adapter.endswith("-pose-v1"):
            if names != COCO_KEYPOINT_NAMES:
                raise ValueError(
                    "Candle YOLOv8 pose runtime requires canonical COCO keypoint order"
                )
            if self.output.keypoint_confidence_threshold is None:
                raise ValueError(
                    "Candle YOLOv8 pose runtime requires a keypoint confidence threshold"
                )
        elif names:
            raise ValueError("detection-only runtime must not declare keypoint_names")
        elif self.output.keypoint_confidence_threshold is not None:
            raise ValueError(
                "detection-only runtime must not declare a keypoint confidence threshold"
            )
        object.__setattr__(self, "keypoint_names", names)

    @property
    def task(self) -> str:
        return "pose" if self.adapter.endswith("-pose-v1") else "detect"

    def validation_errors(
        self,
        *,
        backend: object,
        num_classes: object,
        inputs: object,
        outputs: object,
    ) -> list[str]:
        errors: list[str] = []
        if type(backend) is not str:
            errors.append("runtime requires a valid backend")
        else:
            if self.adapter.startswith("manwe-candle-") and backend != "candle":
                errors.append("Candle runtime adapters require backend='candle'")
            if self.adapter == "ultralytics-raw-detect-v1" and backend == "candle":
                errors.append("the Ultralytics raw adapter cannot execute a Candle artifact")
        if type(num_classes) is not int or not 1 <= num_classes <= MAX_CONTRACT_CLASSES:
            return [*errors, "runtime requires a valid num_classes value"]
        if self.adapter.startswith("manwe-candle-") and num_classes > MAX_NATIVE_MODEL_CLASSES:
            errors.append(f"native runtime supports at most {MAX_NATIVE_MODEL_CLASSES} classes")
        if type(inputs) is not list or len(inputs) != 1 or type(inputs[0]) is not TensorSpec:
            errors.append("runtime requires exactly one typed input tensor")
        else:
            image = cast(TensorSpec, inputs[0])
            expected_shape = [1, 3, self.input.height, self.input.width]
            if image.name != self.input.tensor_name:
                errors.append("runtime input tensor_name does not match inputs[0].name")
            if image.shape != expected_shape:
                errors.append(f"runtime input tensor shape must be {expected_shape}")
            if image.dtype != self.input.dtype or image.layout != self.input.layout:
                errors.append("runtime input dtype/layout does not match inputs[0]")
        if type(outputs) is not list or len(outputs) != 1 or type(outputs[0]) is not TensorSpec:
            errors.append("runtime requires exactly one typed output tensor")
        else:
            prediction = cast(TensorSpec, outputs[0])
            if prediction.name != self.output.tensor_name:
                errors.append("runtime output tensor_name does not match outputs[0].name")
            features = 4 + num_classes + 3 * len(self.keypoint_names)
            prediction_count = (
                prediction.shape[2]
                if type(prediction.shape) is list
                and len(prediction.shape) == 3
                and type(prediction.shape[2]) is int
                else None
            )
            if (
                type(prediction.shape) is not list
                or len(prediction.shape) != 3
                or prediction.shape[0] != 1
                or prediction.shape[1] != features
                or type(prediction.shape[2]) is not int
                or not 1 <= cast(int, prediction.shape[2]) <= MAX_RUNTIME_PREDICTIONS
            ):
                errors.append(
                    "runtime output tensor must have concrete shape "
                    f"[1, {features}, predictions] with 1..{MAX_RUNTIME_PREDICTIONS} predictions"
                )
            elif self.adapter.startswith("manwe-candle-") and (
                cast(int, prediction_count) > MAX_NATIVE_RUNTIME_PREDICTIONS
                or features * cast(int, prediction_count) > MAX_NATIVE_OUTPUT_ELEMENTS
            ):
                errors.append("native runtime output exceeds its bounded element-count limit")
            elif self.adapter.startswith("manwe-candle-"):
                expected_native_predictions = sum(
                    (self.input.height // stride) * (self.input.width // stride)
                    for stride in (8, 16, 32)
                )
                if prediction_count != expected_native_predictions:
                    errors.append(
                        "native YOLOv8 output prediction count must equal its three "
                        f"stride-grid sizes ({expected_native_predictions})"
                    )
            if self.adapter.startswith("manwe-candle-"):
                if prediction.dtype != "float32" or prediction.layout:
                    errors.append("native runtime output must be float32 with an empty layout")
            elif prediction.dtype not in {"float16", "float32"} or prediction.layout:
                errors.append("runtime output must be float16/float32 with an empty layout")
        if self.adapter.endswith("-pose-v1") and num_classes != 1:
            errors.append("native YOLOv8 pose runtime requires exactly one class")
        return errors

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "model_variant": self.model_variant,
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
            "keypoint_names": list(self.keypoint_names),
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeSpec:
        data = _strict_json_object(
            value,
            "runtime",
            required={"adapter", "model_variant", "input", "output", "keypoint_names"},
        )
        keypoints = data["keypoint_names"]
        if type(keypoints) is not list:
            raise ValueError("runtime.keypoint_names must be a JSON array")
        return cls(
            adapter=cast(RuntimeAdapter, data["adapter"]),
            model_variant=cast(str | None, data["model_variant"]),
            input=ImageInputSpec.from_dict(data["input"]),
            output=DetectionOutputSpec.from_dict(data["output"]),
            keypoint_names=cast(tuple[str, ...], tuple(keypoints)),
        )


def detection_runtime(
    *,
    input_tensor: str,
    output_tensor: str,
    image_size: int,
    adapter: RuntimeAdapter = "ultralytics-raw-detect-v1",
    model_variant: str | None = None,
    input_dtype: str = "float32",
    interpolation: str | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    keypoint_names: Sequence[str] = (),
    keypoint_confidence_threshold: float | None = None,
) -> RuntimeSpec:
    """Construct a canonical runtime spec without free-form preprocessing prose."""
    selected_interpolation = "catmull-rom" if adapter.startswith("manwe-candle-") else "bilinear"
    if interpolation is not None:
        selected_interpolation = interpolation
    selected_keypoint_threshold = keypoint_confidence_threshold
    if keypoint_names and selected_keypoint_threshold is None:
        selected_keypoint_threshold = 0.6
    return RuntimeSpec(
        adapter=adapter,
        model_variant=model_variant,
        input=ImageInputSpec(
            tensor_name=input_tensor,
            width=image_size,
            height=image_size,
            dtype=input_dtype,
            interpolation=selected_interpolation,
        ),
        output=DetectionOutputSpec(
            tensor_name=output_tensor,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            keypoint_confidence_threshold=selected_keypoint_threshold,
        ),
        keypoint_names=_runtime_names(
            keypoint_names,
            "runtime keypoint_names",
            allow_empty=True,
        ),
    )


@dataclass(slots=True)
class ModelContract:
    """A self-describing candidate record for a model artifact.

    Fields from the proposed downstream record have a home here so a candidate
    can be checked for internal completeness and serialized beside the artifact.
    """

    # Provenance
    model_name: str
    model_version: str
    source: str  # repo / internal provenance
    rights: str  # redistribution / usage confirmation
    backend: Backend
    file_path: str  # portable artifact basename, resolved beside the contract
    file_sha256: str = ""

    # Tensor contracts
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)

    # Pre/post processing
    preprocess: str = ""  # resize/crop/normalize behaviour
    postprocess: str = ""  # NMS, score threshold, coordinate scaling, max dets

    # Optional mapping into Manwe's local candidate taxonomy. A None value is
    # explicitly unmapped; native structured output still retains source class.
    class_map: dict[int, AirspaceClass | None] = field(default_factory=dict)

    # Validation & benchmark evidence
    validation_data: str = ""  # fixture frames used to verify detections
    benchmark_context: str = ""  # hardware, OS, backend, thresholds, command
    failure_behavior: str = ""  # behaviour on missing/malformed/wrong input

    # Schema metadata is appended after the legacy fields to preserve positional
    # construction compatibility with the original contract dataclass.
    schema_version: str = MODEL_CONTRACT_SCHEMA_VERSION
    num_classes: int = len(AIRSPACE_CLASSES)
    source_classes: list[str] = field(default_factory=list)
    source_sha256: str = ""
    export_options: str = ""
    signature_evidence: str = ""
    runtime: RuntimeSpec | None = None

    def missing_fields(self) -> list[str]:
        """Return proposed candidate-record fields that are empty."""
        required = {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source": self.source,
            "rights": self.rights,
            "backend": self.backend,
            "file_path": self.file_path,
            "file_sha256": self.file_sha256,
            "num_classes": self.num_classes,
            "source_classes": self.source_classes,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "preprocess": self.preprocess,
            "postprocess": self.postprocess,
            "class_map": self.class_map,
            "validation_data": self.validation_data,
            "benchmark_context": self.benchmark_context,
            "failure_behavior": self.failure_behavior,
            "source_sha256": self.source_sha256,
            "export_options": self.export_options,
            "signature_evidence": self.signature_evidence,
            "runtime": self.runtime,
        }
        return [key for key, value in required.items() if not _has_required_value(value)]

    def is_complete(self) -> bool:
        """Return whether the portable record is structurally complete.

        Artifact identity is a separate operation because ``file_path`` is a
        contract-relative filename, never an ambient-current-directory path.
        """
        return not self.validation_errors(check_artifact=False)

    def _class_map_errors(self) -> list[str]:
        errors: list[str] = []
        if type(self.class_map) is not dict:
            return ["class_map must be a dictionary"]
        if not self.class_map:
            return ["class_map must not be empty"]
        if len(self.class_map) > MAX_CONTRACT_CLASSES:
            return [f"class_map must contain at most {MAX_CONTRACT_CLASSES} entries"]

        valid_keys: set[int] = set()
        for idx, name in dict.items(self.class_map):
            if type(idx) is not int:  # bool is deliberately not a class index
                errors.append(f"class_map key {idx!r} must be an integer")
                continue
            if idx < 0:
                errors.append(f"class_map key {idx} must be nonnegative")
                continue
            valid_keys.add(idx)
            if name is not None and (type(name) is not str or name not in AIRSPACE_CLASSES):
                errors.append(
                    f"class_map[{idx}] = {name!r} must be a Manwe airspace class or None (unmapped)"
                )

        if type(self.num_classes) is int and 0 < self.num_classes <= MAX_CONTRACT_CLASSES:
            expected = set(range(self.num_classes))
            missing = sorted(expected - valid_keys)
            extra = sorted(valid_keys - expected)
            if missing:
                errors.append(f"class_map is missing class indices {missing}")
            if extra:
                errors.append(f"class_map has out-of-range class indices {extra}")
        return errors

    def validate_class_map(self) -> None:
        """Validate key types, canonical values, and complete head coverage."""
        errors = self._class_map_errors()
        if errors:
            raise ValueError("invalid class map: " + "; ".join(errors))

    def validation_errors(
        self,
        *,
        check_artifact: bool = False,
        artifact_path: str | pathlib.Path | None = None,
    ) -> list[str]:
        """Return every internal record or explicitly located artifact error.

        ``file_path`` is a package-local basename, never an ambient-working-
        directory authority.  Artifact validation therefore requires the caller
        to supply ``artifact_path`` explicitly or to use :meth:`load`, which
        binds the sidecar and sibling artifact to one retained directory.
        """
        errors = [f"missing required field {name}" for name in self.missing_fields()]

        if type(check_artifact) is not bool:
            errors.append("check_artifact must be a boolean")
        elif check_artifact and artifact_path is None:
            errors.append("artifact_path is required when check_artifact=True")
        elif not check_artifact and artifact_path is not None:
            errors.append("artifact_path requires check_artifact=True")

        string_fields = (
            "schema_version",
            "model_name",
            "model_version",
            "source",
            "rights",
            "backend",
            "file_path",
            "file_sha256",
            "preprocess",
            "postprocess",
            "validation_data",
            "benchmark_context",
            "failure_behavior",
            "source_sha256",
            "export_options",
            "signature_evidence",
        )
        for name in string_fields:
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                errors.append(f"{name} must be a string")
            elif type(value) is str and not _bounded_printable_utf8(
                value,
                max_bytes=4096,
                require_trimmed=True,
            ):
                errors.append(f"{name} must be bounded printable text without outer whitespace")

        if self.schema_version != MODEL_CONTRACT_SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {MODEL_CONTRACT_SCHEMA_VERSION!r}"
            )
        if type(self.backend) is not str or self.backend not in BACKEND_EXTENSIONS:
            errors.append(
                f"unsupported backend {self.backend!r}; expected one of {tuple(BACKEND_EXTENSIONS)}"
            )
        elif type(self.file_path) is str and self.file_path.strip():
            if not _ARTIFACT_NAME_RE.fullmatch(self.file_path):
                errors.append(
                    "file_path must be a 1..255 character portable ASCII artifact filename"
                )
            suffix = pathlib.Path(self.file_path).suffix.lower()
            allowed = BACKEND_EXTENSIONS[self.backend]
            if suffix not in allowed:
                errors.append(
                    f"artifact suffix {suffix or '<none>'!r} is invalid for backend "
                    f"{self.backend!r}; expected one of {allowed}"
                )
        if type(self.num_classes) is not int or not 1 <= self.num_classes <= MAX_CONTRACT_CLASSES:
            errors.append(f"num_classes must be an integer in [1, {MAX_CONTRACT_CLASSES}]")
        if type(self.source_classes) is not list:
            errors.append("source_classes must be a list")
        else:
            if type(self.num_classes) is int and len(self.source_classes) != self.num_classes:
                errors.append("source_classes length must equal num_classes")
            normalized_source_classes: list[str] = []
            if len(self.source_classes) > MAX_CONTRACT_CLASSES:
                errors.append(f"source_classes must contain at most {MAX_CONTRACT_CLASSES} entries")
                bounded_source_classes: tuple[object, ...] = ()
            else:
                bounded_source_classes = tuple(self.source_classes)
            for index, source_name in enumerate(bounded_source_classes):
                if not _bounded_printable_utf8(
                    source_name,
                    max_bytes=256,
                    allow_empty=False,
                ):
                    errors.append(f"source_classes[{index}] must be a bounded printable class name")
                elif cast(str, source_name) != cast(str, source_name).strip():
                    errors.append(f"source_classes[{index}] must not have surrounding whitespace")
                else:
                    normalized_source_classes.append(cast(str, source_name))
            if len(set(normalized_source_classes)) != len(normalized_source_classes):
                errors.append("source_classes must be unique")
        if (
            type(self.file_sha256) is str
            and self.file_sha256
            and (
                not _SHA256_RE.fullmatch(self.file_sha256)
                or self.file_sha256 != self.file_sha256.lower()
            )
        ):
            errors.append("file_sha256 must be 64 lowercase hexadecimal characters")
        if (
            type(self.source_sha256) is str
            and self.source_sha256
            and (
                not _SHA256_RE.fullmatch(self.source_sha256)
                or self.source_sha256 != self.source_sha256.lower()
            )
        ):
            errors.append("source_sha256 must be 64 lowercase hexadecimal characters")

        for collection_name in ("inputs", "outputs"):
            tensors = getattr(self, collection_name)
            if type(tensors) is not list:
                errors.append(f"{collection_name} must be a list")
                continue
            if len(tensors) > MAX_CONTRACT_TENSORS:
                errors.append(
                    f"{collection_name} must contain at most {MAX_CONTRACT_TENSORS} tensors"
                )
                continue
            names: list[str] = []
            for index, tensor in enumerate(tensors):
                field_name = f"{collection_name}[{index}]"
                if type(tensor) is not TensorSpec:
                    errors.append(f"{field_name} must be a TensorSpec")
                    continue
                errors.extend(tensor.validation_errors(field_name))
                if type(tensor.name) is str:
                    names.append(tensor.name)
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                errors.append(
                    f"{collection_name} contains duplicate tensor names {duplicate_names}"
                )

        errors.extend(self._class_map_errors())

        if type(self.runtime) is not RuntimeSpec:
            errors.append("runtime must be a typed machine-consumable RuntimeSpec")
        else:
            errors.extend(
                self.runtime.validation_errors(
                    backend=self.backend,
                    num_classes=self.num_classes,
                    inputs=self.inputs,
                    outputs=self.outputs,
                )
            )

        if check_artifact is True and artifact_path is not None and not errors:
            try:
                self.verify_artifact(pathlib.Path(artifact_path))
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"artifact verification failed: {exc}")
        return errors

    def validate(
        self,
        *,
        check_artifact: bool = False,
        artifact_path: str | pathlib.Path | None = None,
    ) -> None:
        """Raise :class:`ValueError` unless the record is internally valid."""
        errors = self.validation_errors(
            check_artifact=check_artifact,
            artifact_path=artifact_path,
        )
        if errors:
            raise ValueError("invalid model contract: " + "; ".join(errors))

    def verify_artifact(self, artifact_path: str | pathlib.Path) -> pathlib.Path:
        """Verify the sibling artifact's type, filename, and exact digest."""
        self.validate(check_artifact=False)
        artifact = pathlib.Path(os.path.abspath(pathlib.Path(artifact_path).expanduser()))
        if artifact.name != self.file_path:
            raise ValueError(
                f"artifact filename {artifact.name!r} does not match contract file_path "
                f"{self.file_path!r}"
            )
        parent_fd = open_directory_nofollow(artifact.parent, "model package directory")
        operation_error: BaseException | None = None
        try:
            parent_identity = _directory_identity(os.fstat(parent_fd))
            _assert_bound_directory_visible(artifact.parent, parent_fd, parent_identity)
            _verify_artifact_at(self, parent_fd, artifact.name)
            _assert_bound_directory_visible(artifact.parent, parent_fd, parent_identity)
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            _close_descriptor(
                parent_fd,
                "model package directory cleanup also failed",
                primary=operation_error,
            )
        return artifact

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class_map"] = {str(k): self.class_map[k] for k in sorted(self.class_map)}
        d["runtime"] = None if self.runtime is None else self.runtime.to_dict()
        return d

    def to_json(self, *, indent: int = 2) -> str:
        if type(indent) is not int or not 0 <= indent <= 8:
            raise ValueError("JSON indent must be an integer in [0, 8]")
        self.validate(check_artifact=False)
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: object) -> ModelContract:
        """Parse one strict schema-2 contract object without touching the artifact."""
        required = {
            "schema_version",
            "model_name",
            "model_version",
            "source",
            "rights",
            "backend",
            "file_path",
            "file_sha256",
            "num_classes",
            "source_classes",
            "inputs",
            "outputs",
            "preprocess",
            "postprocess",
            "class_map",
            "validation_data",
            "benchmark_context",
            "failure_behavior",
            "source_sha256",
            "export_options",
            "signature_evidence",
            "runtime",
        }
        data = _strict_json_object(value, "model contract", required=required)
        input_values = data["inputs"]
        output_values = data["outputs"]
        source_classes = data["source_classes"]
        class_map_value = data["class_map"]
        if type(input_values) is not list or type(output_values) is not list:
            raise ValueError("model contract inputs and outputs must be JSON arrays")
        if len(input_values) > MAX_CONTRACT_TENSORS or len(output_values) > MAX_CONTRACT_TENSORS:
            raise ValueError(
                f"model contract tensor arrays must contain at most {MAX_CONTRACT_TENSORS} entries"
            )
        if type(source_classes) is not list:
            raise ValueError("model contract source_classes must be a JSON array")
        if len(source_classes) > MAX_CONTRACT_CLASSES:
            raise ValueError(
                f"model contract source_classes must contain at most {MAX_CONTRACT_CLASSES} entries"
            )
        if type(class_map_value) is not dict:
            raise ValueError("model contract class_map must be a JSON object")
        if len(class_map_value) > MAX_CONTRACT_CLASSES:
            raise ValueError(
                f"model contract class_map must contain at most {MAX_CONTRACT_CLASSES} entries"
            )
        class_map: dict[int, AirspaceClass | None] = {}
        for key, mapped in cast(dict[object, object], class_map_value).items():
            if (
                type(key) is not str
                or not key.isascii()
                or not key.isdecimal()
                or (key.startswith("0") and key != "0")
            ):
                raise ValueError("model contract class_map keys must be canonical decimal indices")
            index = int(key)
            if mapped is not None and (type(mapped) is not str or mapped not in AIRSPACE_CLASSES):
                raise ValueError(
                    f"model contract class_map[{key}] must be a Manwe airspace class or null"
                )
            class_map[index] = cast(AirspaceClass | None, mapped)
        contract = cls(
            schema_version=cast(str, data["schema_version"]),
            model_name=cast(str, data["model_name"]),
            model_version=cast(str, data["model_version"]),
            source=cast(str, data["source"]),
            rights=cast(str, data["rights"]),
            backend=cast(Backend, data["backend"]),
            file_path=cast(str, data["file_path"]),
            file_sha256=cast(str, data["file_sha256"]),
            num_classes=cast(int, data["num_classes"]),
            source_classes=cast(list[str], source_classes.copy()),
            inputs=[
                TensorSpec.from_dict(item, f"inputs[{index}]")
                for index, item in enumerate(cast(list[object], input_values))
            ],
            outputs=[
                TensorSpec.from_dict(item, f"outputs[{index}]")
                for index, item in enumerate(cast(list[object], output_values))
            ],
            preprocess=cast(str, data["preprocess"]),
            postprocess=cast(str, data["postprocess"]),
            class_map=class_map,
            validation_data=cast(str, data["validation_data"]),
            benchmark_context=cast(str, data["benchmark_context"]),
            failure_behavior=cast(str, data["failure_behavior"]),
            source_sha256=cast(str, data["source_sha256"]),
            export_options=cast(str, data["export_options"]),
            signature_evidence=cast(str, data["signature_evidence"]),
            runtime=RuntimeSpec.from_dict(data["runtime"]),
        )
        contract.validate(check_artifact=False)
        return contract

    @classmethod
    def from_json(cls, value: str | bytes) -> ModelContract:
        """Parse bounded, duplicate-free JSON into a validated contract."""
        if type(value) is bytes:
            if len(value) > MAX_CONTRACT_JSON_BYTES:
                raise ValueError(
                    f"model contract JSON exceeds {MAX_CONTRACT_JSON_BYTES} encoded bytes"
                )
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("model contract JSON must be UTF-8") from exc
        elif type(value) is str:
            text = value
            try:
                encoded_size = len(text.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ValueError("model contract JSON must be valid UTF-8 text") from exc
            if encoded_size > MAX_CONTRACT_JSON_BYTES:
                raise ValueError(
                    f"model contract JSON exceeds {MAX_CONTRACT_JSON_BYTES} encoded bytes"
                )
        else:
            raise TypeError("model contract JSON must be a string or bytes")
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("model contract contains invalid JSON") from exc
        _validate_json_graph(payload)
        return cls.from_dict(payload)

    @classmethod
    def load(
        cls,
        contract_path: str | pathlib.Path,
        *,
        verify_artifact: bool = True,
    ) -> ModelContract:
        """Read a no-follow sidecar and optionally verify its sibling artifact."""
        if type(verify_artifact) is not bool:
            raise TypeError("verify_artifact must be a boolean")
        path = pathlib.Path(os.path.abspath(pathlib.Path(contract_path).expanduser()))
        parent_fd = open_directory_nofollow(path.parent, "model package directory")
        operation_error: BaseException | None = None
        try:
            parent_identity = _directory_identity(os.fstat(parent_fd))
            _assert_bound_directory_visible(path.parent, parent_fd, parent_identity)
            payload = read_bounded_regular_bytes_at(
                parent_fd,
                path.name,
                MAX_CONTRACT_JSON_BYTES,
                "model contract",
            )
            contract = cls.from_json(payload)
            if verify_artifact:
                _verify_artifact_at(contract, parent_fd, contract.file_path)
            _assert_bound_directory_visible(path.parent, parent_fd, parent_identity)
            return contract
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            _close_descriptor(
                parent_fd,
                "model package directory cleanup also failed",
                primary=operation_error,
            )

    def to_markdown(
        self,
        *,
        check_artifact: bool = False,
        artifact_path: str | pathlib.Path | None = None,
    ) -> str:
        """Render a human-readable candidate contract table."""
        lines = [
            f"# Model Contract — {_markdown_cell(self.model_name)} "
            f"{_markdown_cell(self.model_version)}",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Schema version | {_markdown_cell(self.schema_version)} |",
            f"| Model name/version | {_markdown_cell(self.model_name)} "
            f"{_markdown_cell(self.model_version)} |",
            f"| Source | {_markdown_cell(self.source)} |",
            f"| Rights | {_markdown_cell(self.rights)} |",
            f"| Backend | {_markdown_cell(self.backend)} |",
            f"| File path | {_markdown_cell(self.file_path)} |",
            f"| SHA-256 | {_markdown_cell(self.file_sha256 or 'TODO')} |",
            f"| Source SHA-256 | {_markdown_cell(self.source_sha256 or 'TODO')} |",
            f"| Export options | {_markdown_cell(self.export_options)} |",
            f"| Signature evidence | {_markdown_cell(self.signature_evidence)} |",
            f"| Runtime adapter | {_markdown_cell(self.runtime.adapter if self.runtime else 'TODO')} |",
            f"| Runtime task | {_markdown_cell(self.runtime.task if self.runtime else 'TODO')} |",
            f"| Runtime model variant | "
            f"{_markdown_cell(self.runtime.model_variant if self.runtime else 'TODO')} |",
            f"| Runtime input policy | "
            f"{_markdown_cell(self.runtime.input.to_dict() if self.runtime else 'TODO')} |",
            f"| Runtime output policy | "
            f"{_markdown_cell(self.runtime.output.to_dict() if self.runtime else 'TODO')} |",
            f"| Runtime keypoints | "
            f"{_markdown_cell(self.runtime.keypoint_names if self.runtime else 'TODO')} |",
            f"| Number of classes | {self.num_classes} |",
            f"| Source classes | {_markdown_cell(self.source_classes)} |",
            f"| Preprocess | {_markdown_cell(self.preprocess)} |",
            f"| Postprocess | {_markdown_cell(self.postprocess)} |",
            f"| Validation data | {_markdown_cell(self.validation_data)} |",
            f"| Benchmark context | {_markdown_cell(self.benchmark_context)} |",
            f"| Failure behavior | {_markdown_cell(self.failure_behavior)} |",
            "",
            "## Inputs",
            "",
            "| Name | Shape | Dtype | Layout | Notes |",
            "|------|-------|-------|--------|-------|",
        ]
        for t in self.inputs:
            lines.append(
                f"| {_markdown_cell(t.name)} | {_markdown_cell(t.shape)} | "
                f"{_markdown_cell(t.dtype)} | {_markdown_cell(t.layout)} | "
                f"{_markdown_cell(t.notes)} |"
            )
        lines += [
            "",
            "## Outputs",
            "",
            "| Name | Shape | Dtype | Layout | Notes |",
            "|------|-------|-------|--------|-------|",
        ]
        for t in self.outputs:
            lines.append(
                f"| {_markdown_cell(t.name)} | {_markdown_cell(t.shape)} | "
                f"{_markdown_cell(t.dtype)} | {_markdown_cell(t.layout)} | "
                f"{_markdown_cell(t.notes)} |"
            )
        lines += [
            "",
            "## Class map",
            "",
            "| Index | Manwe airspace class |",
            "|-------|----------------------|",
        ]
        for idx in sorted(self.class_map):
            mapped = self.class_map[idx]
            lines.append(
                f"| {idx} | {_markdown_cell(mapped if mapped is not None else 'UNMAPPED')} |"
            )
        errors = self.validation_errors(
            check_artifact=check_artifact,
            artifact_path=artifact_path,
        )
        if errors:
            lines += ["", f"> ⚠️ Invalid contract — {_markdown_cell('; '.join(errors))}"]
        return "\n".join(lines) + "\n"


__all__ = [
    "AirspaceClass",
    "AIRSPACE_CLASSES",
    "airspace_class_index",
    "coco_to_airspace",
    "CrebainClass",
    "CREBAIN_CLASSES",
    "crebain_class_index",
    "coco_to_crebain",
    "Backend",
    "BACKEND_EXTENSIONS",
    "MODEL_CONTRACT_SCHEMA_VERSION",
    "TensorSpec",
    "ImageInputSpec",
    "DetectionOutputSpec",
    "RuntimeSpec",
    "detection_runtime",
    "COCO_KEYPOINT_NAMES",
    "ModelContract",
]
