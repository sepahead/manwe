"""Produce raw backend artifacts from a trained Ultralytics detector.

This converter does not make an artifact consumer-compatible or trusted. Every
result still needs a complete model contract, a consumer-specific adapter, and
the fidelity gate. MLX is retained in the contract vocabulary, but is not a
supported conversion target here.
"""

from __future__ import annotations

import os
import shutil
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..common.artifacts import (
    DEFAULT_MAX_ARTIFACT_ENTRIES,
    ArtifactSnapshot,
    _tree_entries,
    require_pickle_acknowledgement,
    sha256_artifact,
    sha256_artifact_at,
)
from ..common.config_io import open_directory_nofollow
from ..common.dataset_manifest import snapshot_local_calibration_dataset
from ..common.deps import require
from ..common.logging import get_logger
from ..common.ultralytics import harden_ultralytics_runtime, verify_ultralytics_policy

log = get_logger("manwe.export")


@dataclass(frozen=True)
class FormatSpec:
    ultralytics: str | None  # ultralytics export `format=` value, or None
    crebain_backend: str
    ext: str
    notes: str


@dataclass(frozen=True)
class ExportReceipt:
    """Immutable provenance for one raw, not-yet-consumer-validated export."""

    format: str
    artifact_path: str
    artifact_sha256: str
    source_sha256: str
    source_suffix: str
    image_size: int
    precision: str
    embedded_nms: bool
    opset: int | None
    class_count: int
    source_classes: tuple[str, ...]
    end_to_end: bool
    # Retained field name for compatibility; the digest binds both the normalized
    # manifest and the complete bounded dataset-root tree.
    calibration_manifest_sha256: str | None
    tensor_signature_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.format, str) or self.format not in {
            "onnx",
            "tensorrt",
            "coreml",
            "mlx",
        }:
            raise ValueError("format is not a supported contract vocabulary value")
        for name in ("artifact_sha256", "source_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)
            ):
                raise ValueError(f"{name} must be a 64-character hexadecimal digest")
            object.__setattr__(self, name, value.lower())
        if (
            not isinstance(self.artifact_path, str)
            or not self.artifact_path.strip()
            or any(character in self.artifact_path for character in "\0\r\n")
            or len(self.artifact_path) > 4096
        ):
            raise ValueError("artifact_path must be a bounded nonempty path string")
        if (
            not isinstance(self.source_suffix, str)
            or not self.source_suffix.startswith(".")
            or len(self.source_suffix) < 2
            or not self.source_suffix[1:].isalnum()
            or len(self.source_suffix) > 32
        ):
            raise ValueError("source_suffix must be a dotted file suffix")
        object.__setattr__(self, "source_suffix", self.source_suffix.lower())
        if type(self.image_size) is not int or not 32 <= self.image_size <= 4096:
            raise ValueError("image_size must be an integer in [32, 4096]")
        if not isinstance(self.precision, str) or self.precision not in {
            "float32",
            "float16",
            "int8",
        }:
            raise ValueError("precision must be float32, float16, or int8")
        for name in ("embedded_nms", "end_to_end", "tensor_signature_verified"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.embedded_nms and self.end_to_end:
            raise ValueError("embedded_nms and end_to_end cannot both be true")
        requires_opset = self.format in {"onnx", "tensorrt"}
        if requires_opset and (type(self.opset) is not int or not 12 <= self.opset <= 20):
            raise ValueError("ONNX/TensorRT opset must be an integer in [12, 20]")
        if not requires_opset and self.opset is not None:
            raise ValueError("opset must be None for formats that do not use ONNX opsets")
        if type(self.class_count) is not int or not 1 <= self.class_count <= 4096:
            raise ValueError("class_count must be an integer in [1, 4096]")
        if (
            not isinstance(self.source_classes, tuple)
            or len(self.source_classes) != self.class_count
        ):
            raise ValueError("source_classes must be a tuple matching class_count")
        normalized_classes: list[str] = []
        for value in self.source_classes:
            if (
                not isinstance(value, str)
                or not value.strip()
                or not value.strip().isprintable()
                or len(value.strip().encode("utf-8")) > 256
            ):
                raise ValueError("source_classes must contain bounded printable names")
            normalized_classes.append(value.strip())
        if len(set(normalized_classes)) != len(normalized_classes):
            raise ValueError("source_classes must be unique")
        object.__setattr__(self, "source_classes", tuple(normalized_classes))
        if self.calibration_manifest_sha256 is not None:
            value = self.calibration_manifest_sha256
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)
            ):
                raise ValueError("calibration_manifest_sha256 must be a hexadecimal digest or None")
            object.__setattr__(self, "calibration_manifest_sha256", value.lower())
        if self.precision == "int8":
            if self.format != "tensorrt" or self.calibration_manifest_sha256 is None:
                raise ValueError("int8 receipts require TensorRT calibration provenance")
        elif self.calibration_manifest_sha256 is not None:
            raise ValueError("calibration provenance is only valid for int8 receipts")
        expected_suffix = {
            "onnx": ".onnx",
            "tensorrt": ".engine",
            "coreml": ".mlpackage",
            "mlx": ".safetensors",
        }[self.format]
        if Path(self.artifact_path).suffix.lower() != expected_suffix:
            raise ValueError(f"artifact_path suffix must be {expected_suffix!r} for {self.format}")


def _ordered_class_names(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        if any(type(key) is not int for key in value) or set(value) != set(range(len(value))):
            raise ValueError("checkpoint class-name mapping must be contiguous from index zero")
        ordered = tuple(value[index] for index in range(len(value)))
    elif isinstance(value, (list, tuple)):
        ordered = tuple(value)
    else:
        raise ValueError("checkpoint must expose an ordered class-name table")
    if not ordered or len(ordered) > 4096:
        raise ValueError("checkpoint class-name table must contain 1..4096 entries")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not name.strip().isprintable()
        or len(name.strip().encode("utf-8")) > 256
        for name in ordered
    ):
        raise ValueError("checkpoint class names must be bounded printable strings")
    normalized = tuple(name.strip() for name in ordered)
    if len(set(normalized)) != len(normalized):
        raise ValueError("checkpoint class names must be unique")
    return normalized


#: manwe format name → how to produce it + crebain backend + gotchas from research.
EXPORT_FORMATS: dict[str, FormatSpec] = {
    "onnx": FormatSpec("onnx", "onnx", ".onnx", "pin opset, run onnxslim, keep NMS external"),
    "tensorrt": FormatSpec(
        "engine",
        "tensorrt",
        ".engine",
        "NVIDIA-only; ModelOpt explicit INT8 needs ≥1000 drone calib images (never COCO)",
    ),
    "coreml": FormatSpec(
        "coreml",
        "coreml",
        ".mlpackage",
        "FLOAT32 for iOS16 NMS; palettization/INT4 for ANE; compile to .mlmodelc",
    ),
    "mlx": FormatSpec(None, "mlx", ".safetensors", "via yolo-mlx, not the Ultralytics exporter"),
}


@dataclass
class _PreparedDestination:
    path: Path
    parent_fd: int
    parent_identity: tuple[int, int]
    closed: bool = False

    def assert_parent_path(self) -> None:
        if self.closed:
            raise RuntimeError("output parent descriptor is already closed")
        try:
            metadata = os.stat(self.path.parent, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError("output parent was replaced while the export was running") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != self.parent_identity
        ):
            raise RuntimeError("output parent was replaced while the export was running")

    def close(self) -> None:
        if not self.closed:
            os.close(self.parent_fd)
            self.closed = True


def _prepare_destination(path: str, allowed_suffixes: set[str]) -> _PreparedDestination:
    if not isinstance(path, str) or not path.strip():
        raise TypeError("output must be a nonempty destination path")
    destination = Path(path).expanduser().absolute()
    if destination.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"output suffix {destination.suffix or '<none>'!r} is not one of "
            f"{sorted(allowed_suffixes)}"
        )
    parent_fd = open_directory_nofollow(destination.parent, "output parent")
    try:
        metadata = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        parent_metadata = os.fstat(parent_fd)
        return _PreparedDestination(
            destination,
            parent_fd,
            (parent_metadata.st_dev, parent_metadata.st_ino),
        )
    except BaseException:
        os.close(parent_fd)
        raise
    os.close(parent_fd)
    kind = "symbolic link" if stat.S_ISLNK(metadata.st_mode) else "filesystem entry"
    raise FileExistsError(f"output destination is already occupied by a {kind}: {destination}")


def _copy_directory_fd_relative(source: Path, destination_fd: int) -> None:
    """Copy a verified bundle through retained directory descriptors.

    If an attacker renames a published directory while the copy is in progress,
    subsequent writes remain attached to the originally created inode rather than
    being redirected through the replacement path.
    """
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fds: dict[Path, int] = {Path("."): destination_fd}
    owned_fds: list[int] = []
    try:
        for entry, kind, _metadata in _tree_entries(source, DEFAULT_MAX_ARTIFACT_ENTRIES):
            relative = entry.relative_to(source)
            parent_fd = directory_fds[relative.parent]
            if kind == "directory":
                os.mkdir(relative.name, mode=0o700, dir_fd=parent_fd)
                child_fd = os.open(relative.name, directory_flags, dir_fd=parent_fd)
                directory_fds[relative] = child_fd
                owned_fds.append(child_fd)
                continue
            destination_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            source_fd = os.open(entry, source_flags)
            with os.fdopen(source_fd, "rb") as source_handle:
                output_fd = os.open(
                    relative.name,
                    destination_flags,
                    0o600,
                    dir_fd=parent_fd,
                )
                with os.fdopen(output_fd, "wb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle, length=1 << 20)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
                    os.fchmod(destination_handle.fileno(), 0o644)
        for fd in reversed(owned_fds):
            os.fchmod(fd, 0o755)
        for fd in reversed(tuple(directory_fds.values())):
            os.fsync(fd)
    finally:
        for fd in reversed(owned_fds):
            with suppress(OSError):
                os.close(fd)


def _entry_identity_at(parent_fd: int, name: str) -> tuple[int, int] | None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _remove_tree_contents_fd(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child_fd = os.open(name, flags, dir_fd=directory_fd)
            try:
                _remove_tree_contents_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _publish_exclusive(source: Path, destination: Path | _PreparedDestination, digest: str) -> None:
    if isinstance(destination, _PreparedDestination):
        owns_anchor = False
        anchor = destination
    else:
        owns_anchor = True
        anchor = _prepare_destination(str(destination), {destination.suffix.lower()})
    if anchor.closed:
        raise RuntimeError("output parent descriptor is already closed")
    try:
        anchor.assert_parent_path()
        with ArtifactSnapshot(source, digest, allowed_suffixes={source.suffix.lower()}) as snapshot:
            if snapshot.path.suffix.lower() != anchor.path.suffix.lower():
                raise ValueError(
                    "output suffix must match the exact artifact type returned by the exporter"
                )
            created_identity: tuple[int, int] | None = None
            created_is_directory = False
            try:
                if snapshot.path.is_dir():
                    os.mkdir(anchor.path.name, mode=0o700, dir_fd=anchor.parent_fd)
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    root_fd = os.open(anchor.path.name, flags, dir_fd=anchor.parent_fd)
                    try:
                        metadata = os.fstat(root_fd)
                        created_identity = (metadata.st_dev, metadata.st_ino)
                        created_is_directory = True
                        _copy_directory_fd_relative(snapshot.path, root_fd)
                        os.fchmod(root_fd, 0o755)
                        os.fsync(root_fd)
                    finally:
                        os.close(root_fd)
                else:
                    flags = (
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    fd = os.open(anchor.path.name, flags, 0o600, dir_fd=anchor.parent_fd)
                    metadata = os.fstat(fd)
                    created_identity = (metadata.st_dev, metadata.st_ino)
                    with snapshot.path.open("rb") as source_handle, os.fdopen(fd, "wb") as target:
                        shutil.copyfileobj(source_handle, target, length=1 << 20)
                        target.flush()
                        os.fsync(target.fileno())
                        os.fchmod(target.fileno(), 0o644)
                if _entry_identity_at(anchor.parent_fd, anchor.path.name) != created_identity:
                    raise RuntimeError(
                        "output destination was replaced while it was being published"
                    )
                if sha256_artifact_at(anchor.parent_fd, anchor.path.name) != digest:
                    raise RuntimeError("published export digest differs from the verified artifact")
                anchor.assert_parent_path()
                os.fsync(anchor.parent_fd)
            except BaseException:
                if (
                    created_identity is not None
                    and _entry_identity_at(anchor.parent_fd, anchor.path.name) == created_identity
                ):
                    if created_is_directory:
                        flags = (
                            os.O_RDONLY
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0)
                        )
                        root_fd = os.open(anchor.path.name, flags, dir_fd=anchor.parent_fd)
                        try:
                            _remove_tree_contents_fd(root_fd)
                        finally:
                            os.close(root_fd)
                        os.rmdir(anchor.path.name, dir_fd=anchor.parent_fd)
                    else:
                        os.unlink(anchor.path.name, dir_fd=anchor.parent_fd)
                    os.fsync(anchor.parent_fd)
                raise
    finally:
        if owns_anchor:
            anchor.close()


def export_model(
    weights: str,
    formats: list[str],
    *,
    output: str,
    weights_sha256: str,
    allow_pickle_checkpoint: bool = False,
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    data: str | None = None,
    device: str = "auto",
    nms: bool = False,
    opset: int = 17,
) -> ExportReceipt:
    """Export ``weights`` to one supported raw format.

    One target per call makes failure and artifact ownership unambiguous. All
    static arguments are validated before Torch/Ultralytics or the model loads.
    """
    if not isinstance(formats, list) or any(not isinstance(value, str) for value in formats):
        raise TypeError("formats must be a list containing one format name")
    if len(formats) != 1:
        raise ValueError("export exactly one format per call")
    unknown = [f for f in formats if f not in EXPORT_FORMATS]
    if unknown:
        raise ValueError(f"unknown export format(s) {unknown}; choose from {list(EXPORT_FORMATS)}")
    unsupported = [f for f in formats if EXPORT_FORMATS[f].ultralytics is None]
    if unsupported:
        raise NotImplementedError(
            f"conversion is not implemented for {unsupported}; produce those artifacts with "
            "their native toolchain, then build and validate a Manwe model contract"
        )
    if not isinstance(weights, str) or not weights.strip():
        raise TypeError("weights must be a nonempty path string")
    if type(allow_pickle_checkpoint) is not bool:
        raise TypeError("allow_pickle_checkpoint must be a boolean")
    require_pickle_acknowledgement(Path(weights), allow_pickle_checkpoint)
    if type(imgsz) is not int or not 32 <= imgsz <= 4096:
        raise ValueError("imgsz must be an integer in [32, 4096]")
    for value, name in ((half, "half"), (int8, "int8"), (nms, "nms")):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    if half and int8:
        raise ValueError("half and int8 are mutually exclusive export modes")
    if type(opset) is not int or not 12 <= opset <= 20:
        raise ValueError("opset must be an integer in [12, 20]")
    if not isinstance(device, str):
        raise TypeError("device must be a string")

    export_format = formats[0]
    spec = EXPORT_FORMATS[export_format]
    output_suffixes = {spec.ext}
    if int8 and export_format != "tensorrt":
        raise ValueError("INT8 conversion is currently verified only for TensorRT")
    if data is not None and not int8:
        raise ValueError("data is only accepted for TensorRT INT8 calibration")
    calibration_snapshot = None
    if int8:
        if data is None:
            raise ValueError("int8 export requires a calibration dataset manifest")
        calibration_snapshot = snapshot_local_calibration_dataset(data)

    weights_snapshot = None
    destination: _PreparedDestination | None = None
    try:
        destination = _prepare_destination(output, output_suffixes)
        weights_snapshot = ArtifactSnapshot(weights, weights_sha256, allowed_suffixes={".pt"})
        calibration_sha256 = (
            calibration_snapshot.calibration_digest() if calibration_snapshot is not None else None
        )

        from ..common.device import resolve_device
        from ..vision.train import resolve_ultralytics_device

        resolved_device = resolve_device(device)
        if export_format == "tensorrt" and resolved_device.kind != "cuda":
            raise RuntimeError("TensorRT export requires an available CUDA device")
        dev = resolve_ultralytics_device(resolved_device)

        harden_ultralytics_runtime()
        ultralytics = require("ultralytics", "export")
        verify_ultralytics_policy()
        model = ultralytics.YOLO(str(weights_snapshot.path))
        if getattr(model, "task", None) != "detect":
            raise ValueError(
                f"checkpoint task must be 'detect', got {getattr(model, 'task', None)!r}"
            )
        end_to_end = bool(getattr(getattr(model, "model", None), "end2end", False))
        if nms and end_to_end:
            raise ValueError("embedded NMS is incompatible with an end-to-end detector head")
        source_classes = _ordered_class_names(getattr(model, "names", None))
        class_count = len(source_classes)

        kwargs: dict[str, object] = {
            "format": spec.ultralytics,
            "imgsz": imgsz,
            "half": half,
            "device": dev,
            "nms": nms,
        }
        if export_format in {"onnx", "tensorrt"}:
            kwargs["opset"] = opset
        if int8:
            assert calibration_snapshot is not None
            kwargs["int8"] = True
            kwargs["data"] = str(calibration_snapshot.path)

        log.info("exporting verified checkpoint → %s (%s)", export_format, spec.notes)
        produced_value = model.export(**kwargs)
        if not isinstance(produced_value, (str, Path)):
            raise RuntimeError(
                f"{export_format} exporter returned an invalid artifact path {produced_value!r}"
            )
        produced = Path(produced_value).absolute()
        if calibration_snapshot is not None:
            after_calibration_sha256 = calibration_snapshot.calibration_digest()
            if after_calibration_sha256 != calibration_sha256:
                raise RuntimeError("private calibration snapshot changed during export")
            calibration_snapshot.assert_source_unchanged()
        if not produced.is_relative_to(weights_snapshot.path.parent):
            raise RuntimeError("exporter wrote outside its private artifact workspace")
        if produced.suffix.lower() not in output_suffixes:
            raise RuntimeError(
                f"{export_format} exporter returned {produced.suffix!r}, expected one of "
                f"{sorted(output_suffixes)}"
            )
        artifact_sha256 = sha256_artifact(produced)
        _publish_exclusive(produced, destination, artifact_sha256)
        precision = "int8" if int8 else "float16" if half else "float32"
        return ExportReceipt(
            format=export_format,
            artifact_path=str(destination.path),
            artifact_sha256=artifact_sha256,
            source_sha256=weights_snapshot.sha256,
            source_suffix=weights_snapshot.path.suffix.lower(),
            image_size=imgsz,
            precision=precision,
            embedded_nms=nms,
            opset=opset if export_format in {"onnx", "tensorrt"} else None,
            class_count=class_count,
            source_classes=source_classes,
            end_to_end=end_to_end,
            calibration_manifest_sha256=calibration_sha256,
        )
    finally:
        if weights_snapshot is not None:
            weights_snapshot.close()
        if calibration_snapshot is not None:
            calibration_snapshot.close()
        if destination is not None:
            destination.close()


__all__ = ["FormatSpec", "ExportReceipt", "EXPORT_FORMATS", "export_model"]
