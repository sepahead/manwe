"""Candidate model/taxonomy records used to make integration assumptions explicit.

No reviewed consumer currently ingests this record. It captures two facts that a
future adapter must reconcile and verify against the exact downstream revision:

1. The **candidate class taxonomy** — five local labels selected for a future
   consumer adapter. The exact downstream revision must be checked before use.
2. The **model contract record** — provenance/tensor/postprocessing metadata
   proposed for a future trust gate
   (``../crebain/docs/MODEL_CONTRACTS.md``). :class:`ModelContract` mirrors that
   table so an export can emit it automatically.

This module is intentionally dependency-free (stdlib only). Validation proves
record completeness and artifact identity, not downstream compatibility.
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from .artifacts import sha256_artifact

# ---------------------------------------------------------------------------
# Class taxonomy — must match crebain `DetectionClass`.
# ---------------------------------------------------------------------------
CrebainClass = Literal["drone", "bird", "aircraft", "helicopter", "unknown"]

#: Canonical, index-ordered class list. The integer index is the value written
#: into an exported model's class dimension and referenced by its contract.
CREBAIN_CLASSES: tuple[CrebainClass, ...] = (
    "drone",
    "bird",
    "aircraft",
    "helicopter",
    "unknown",
)

_CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CREBAIN_CLASSES)}


def crebain_class_index(name: str) -> int:
    """Return the canonical index for a crebain class name.

    >>> crebain_class_index("drone")
    0
    """
    try:
        return _CLASS_INDEX[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"{name!r} is not a crebain class; expected one of {CREBAIN_CLASSES}"
        ) from exc


# COCO (80-class) does not contain "drone" or "helicopter". A stock COCO YOLO can
# therefore cannot cover the local taxonomy. This fallback maps the COCO names
# that do correspond for bounded producer-side experiments; it is not a consumer
# adapter and does not imply drone or helicopter capability.
_COCO_TO_CREBAIN: dict[str, CrebainClass] = {
    "airplane": "aircraft",
    "bird": "bird",
    # Everything else (person, car, ...) is not an airspace object → dropped.
}


def coco_to_crebain(coco_name: str) -> CrebainClass | None:
    """Map a COCO class name to a crebain class, or ``None`` if it has no aerial
    counterpart (the caller should drop such detections)."""
    return _COCO_TO_CREBAIN.get(coco_name)


# ---------------------------------------------------------------------------
# Model contract record — mirrors crebain/docs/MODEL_CONTRACTS.md.
# ---------------------------------------------------------------------------
# crebain's accepted backends. MLX ships as a .safetensors *file* but its backend
# is "mlx" — the file format is not a backend.
Backend = Literal["onnx", "coreml", "mlx", "tensorrt"]

MODEL_CONTRACT_SCHEMA_VERSION = "1.2"
MAX_CONTRACT_CLASSES = 4096
MAX_CONTRACT_TENSORS = 64
MAX_TENSOR_RANK = 16
MAX_TENSOR_DIMENSION = 2**31 - 1

# The artifact suffix is part of the signed contract. CoreML artifacts are
# directory bundles; the other backends use regular files.
BACKEND_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "onnx": (".onnx",),
    "coreml": (".mlpackage", ".mlmodelc"),
    "mlx": (".safetensors",),
    "tensorrt": (".engine",),
}

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TENSOR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")

# Contract spellings are deliberately canonical rather than accepting arbitrary
# backend display strings.  Inspectors must translate backend-specific names
# (for example ONNX ``tensor(float)``) into this vocabulary before signing a
# contract.  A bounded vocabulary prevents a typo from silently becoming an
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
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


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
            if type(value) is str and (len(value) > 4096 or "\0" in value):
                errors.append(f"{field_name}.{name} is too long or contains NUL")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


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
    file_path: str
    file_sha256: str = ""

    # Tensor contracts
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)

    # Pre/post processing
    preprocess: str = ""  # resize/crop/normalize behaviour
    postprocess: str = ""  # NMS, score threshold, coordinate scaling, max dets

    # Class mapping into crebain DetectionClass values
    class_map: dict[int, CrebainClass | None] = field(default_factory=dict)

    # Validation & benchmark evidence
    validation_data: str = ""  # fixture frames used to verify detections
    benchmark_context: str = ""  # hardware, OS, backend, thresholds, command
    failure_behavior: str = ""  # behaviour on missing/malformed/wrong input

    # Schema metadata is appended after the legacy fields to preserve positional
    # construction compatibility with the original contract dataclass.
    schema_version: str = MODEL_CONTRACT_SCHEMA_VERSION
    num_classes: int = len(CREBAIN_CLASSES)
    source_classes: list[str] = field(default_factory=list)
    source_sha256: str = ""
    export_options: str = ""
    signature_evidence: str = ""

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
        }
        return [key for key, value in required.items() if not _has_required_value(value)]

    def is_complete(self) -> bool:
        """Return whether internal record and artifact validation succeeds."""
        return not self.validation_errors()

    def _class_map_errors(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.class_map, dict):
            return ["class_map must be a dictionary"]
        if not self.class_map:
            return ["class_map must not be empty"]
        if len(self.class_map) > MAX_CONTRACT_CLASSES:
            return [f"class_map must contain at most {MAX_CONTRACT_CLASSES} entries"]

        valid_keys: set[int] = set()
        for idx, name in self.class_map.items():
            if type(idx) is not int:  # bool is deliberately not a class index
                errors.append(f"class_map key {idx!r} must be an integer")
                continue
            if idx < 0:
                errors.append(f"class_map key {idx} must be nonnegative")
                continue
            valid_keys.add(idx)
            if name is not None and (not isinstance(name, str) or name not in CREBAIN_CLASSES):
                errors.append(f"class_map[{idx}] = {name!r} must be a crebain class or None (drop)")

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

    def validation_errors(self, *, check_artifact: bool = True) -> list[str]:
        """Return every internal record or artifact validation error."""
        errors = [f"missing required field {name}" for name in self.missing_fields()]

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
            if value is not None and not isinstance(value, str):
                errors.append(f"{name} must be a string")
            elif isinstance(value, str) and (len(value) > 4096 or "\0" in value):
                errors.append(f"{name} is too long or contains NUL")

        if self.schema_version != MODEL_CONTRACT_SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {MODEL_CONTRACT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.backend, str) or self.backend not in BACKEND_EXTENSIONS:
            errors.append(
                f"unsupported backend {self.backend!r}; expected one of {tuple(BACKEND_EXTENSIONS)}"
            )
        elif isinstance(self.file_path, str) and self.file_path.strip():
            suffix = pathlib.Path(self.file_path).suffix.lower()
            allowed = BACKEND_EXTENSIONS[self.backend]
            if suffix not in allowed:
                errors.append(
                    f"artifact suffix {suffix or '<none>'!r} is invalid for backend "
                    f"{self.backend!r}; expected one of {allowed}"
                )
        if type(self.num_classes) is not int or not 1 <= self.num_classes <= MAX_CONTRACT_CLASSES:
            errors.append(f"num_classes must be an integer in [1, {MAX_CONTRACT_CLASSES}]")
        if not isinstance(self.source_classes, list):
            errors.append("source_classes must be a list")
        else:
            if len(self.source_classes) != self.num_classes:
                errors.append("source_classes length must equal num_classes")
            normalized_source_classes: list[str] = []
            if len(self.source_classes) > MAX_CONTRACT_CLASSES:
                errors.append(f"source_classes must contain at most {MAX_CONTRACT_CLASSES} entries")
                bounded_source_classes: tuple[object, ...] = ()
            else:
                bounded_source_classes = tuple(self.source_classes)
            for index, source_name in enumerate(bounded_source_classes):
                if (
                    not isinstance(source_name, str)
                    or not source_name.strip()
                    or not source_name.isprintable()
                    or len(source_name.encode("utf-8")) > 256
                ):
                    errors.append(f"source_classes[{index}] must be a bounded printable class name")
                elif source_name != source_name.strip():
                    errors.append(f"source_classes[{index}] must not have surrounding whitespace")
                else:
                    normalized_source_classes.append(source_name)
            if len(set(normalized_source_classes)) != len(normalized_source_classes):
                errors.append("source_classes must be unique")
        if (
            isinstance(self.file_sha256, str)
            and self.file_sha256
            and not _SHA256_RE.fullmatch(self.file_sha256)
        ):
            errors.append("file_sha256 must be a 64-character hexadecimal SHA-256 digest")
        if (
            isinstance(self.source_sha256, str)
            and self.source_sha256
            and not _SHA256_RE.fullmatch(self.source_sha256)
        ):
            errors.append("source_sha256 must be a 64-character hexadecimal SHA-256 digest")

        for collection_name in ("inputs", "outputs"):
            tensors = getattr(self, collection_name)
            if not isinstance(tensors, list):
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
                if not isinstance(tensor, TensorSpec):
                    errors.append(f"{field_name} must be a TensorSpec")
                    continue
                errors.extend(tensor.validation_errors(field_name))
                if isinstance(tensor.name, str):
                    names.append(tensor.name)
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                errors.append(
                    f"{collection_name} contains duplicate tensor names {duplicate_names}"
                )

        errors.extend(self._class_map_errors())

        if check_artifact and isinstance(self.file_path, str) and self.file_path.strip():
            artifact = pathlib.Path(self.file_path)
            artifact_is_safe = True
            if artifact.is_symlink():
                errors.append(f"artifact root must not be a symbolic link: {artifact}")
                artifact_is_safe = False
            elif not artifact.exists():
                errors.append(f"artifact does not exist: {artifact}")
                artifact_is_safe = False
            else:
                suffix = artifact.suffix.lower()
                if suffix in {".mlpackage", ".mlmodelc"}:
                    if not artifact.is_dir():
                        errors.append(f"CoreML artifact must be a directory bundle: {artifact}")
                        artifact_is_safe = False
                elif not artifact.is_file():
                    errors.append(f"artifact must be a regular file: {artifact}")
                    artifact_is_safe = False
                elif os.stat(artifact, follow_symlinks=False).st_size == 0:
                    errors.append(f"artifact file is empty: {artifact}")
                    artifact_is_safe = False
            if (
                artifact_is_safe
                and isinstance(self.file_sha256, str)
                and _SHA256_RE.fullmatch(self.file_sha256)
            ):
                try:
                    actual_sha256 = sha256_artifact(artifact)
                except (OSError, ValueError) as exc:
                    errors.append(f"artifact cannot be safely hashed: {exc}")
                else:
                    if actual_sha256 != self.file_sha256.lower():
                        errors.append(
                            "artifact SHA-256 does not match file_sha256: "
                            f"expected {self.file_sha256.lower()}, got {actual_sha256}"
                        )
        return errors

    def validate(self, *, check_artifact: bool = True) -> None:
        """Raise :class:`ValueError` unless the record is internally valid."""
        errors = self.validation_errors(check_artifact=check_artifact)
        if errors:
            raise ValueError("invalid model contract: " + "; ".join(errors))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class_map"] = {str(k): v for k, v in self.class_map.items()}
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_markdown(self, *, check_artifact: bool = True) -> str:
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
        lines += ["", "## Class map", "", "| Index | crebain class |", "|-------|---------------|"]
        for idx in sorted(self.class_map):
            mapped = self.class_map[idx]
            lines.append(f"| {idx} | {_markdown_cell(mapped if mapped is not None else 'DROP')} |")
        errors = self.validation_errors(check_artifact=check_artifact)
        if errors:
            lines += ["", f"> ⚠️ Invalid contract — {_markdown_cell('; '.join(errors))}"]
        return "\n".join(lines) + "\n"


__all__ = [
    "CrebainClass",
    "CREBAIN_CLASSES",
    "crebain_class_index",
    "coco_to_crebain",
    "Backend",
    "BACKEND_EXTENSIONS",
    "MODEL_CONTRACT_SCHEMA_VERSION",
    "TensorSpec",
    "ModelContract",
]
