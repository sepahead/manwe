"""Produce raw backend artifacts from a trained Ultralytics detector.

This converter does not make an artifact consumer-compatible or trusted. Every
result still needs a complete model contract, a consumer-specific adapter, and
the fidelity gate. MLX is retained in the contract vocabulary, but is not a
supported conversion target here.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..common.artifacts import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_MAX_ARTIFACT_ENTRIES,
    ArtifactSnapshot,
    _copy_directory_fd,
    require_pickle_acknowledgement,
    sha256_artifact,
    sha256_artifact_at,
)
from ..common.config_io import open_directory_nofollow
from ..common.dataset_manifest import snapshot_local_calibration_dataset
from ..common.deps import require
from ..common.fd_io import attach_cleanup_failure, owned_binary_writer
from ..common.logging import get_logger
from ..common.ultralytics import (
    deterministic_ultralytics_validation_format,
    harden_ultralytics_runtime,
    verify_ultralytics_policy,
)

log = get_logger("manwe.export")

_MAX_MODEL_STRIDES = 64
_MAX_MODEL_STRIDE_NODES = 128


def _preflight_tensorrt_int8(device_index: int) -> tuple[str, str | None]:
    """Prove the pinned exporter will take its INT8 rather than silent-FP32 branch."""
    if type(device_index) is not int or device_index < 0:
        raise ValueError("TensorRT device index must be a nonnegative integer")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - export dependency boundary
        raise RuntimeError("TensorRT INT8 export requires installed PyTorch") from exc
    try:
        torch.cuda.set_device(device_index)
    except Exception as exc:
        raise RuntimeError(
            f"TensorRT INT8 export could not select CUDA device {device_index}"
        ) from exc
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - NVIDIA-only dependency boundary
        raise RuntimeError(
            "TensorRT INT8 export requires installed NVIDIA TensorRT Python bindings"
        ) from exc

    version = getattr(trt, "__version__", None)
    if not isinstance(version, str) or not version or not version.isprintable():
        raise RuntimeError("TensorRT must expose a printable version string")
    try:
        major = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError("TensorRT version must start with an integer major version") from exc
    if major < 7:
        raise RuntimeError("TensorRT INT8 export requires TensorRT 7 or newer")
    if major > 11:
        raise RuntimeError(
            "TensorRT versions newer than the audited 7-11 exporter branches are rejected"
        )
    release = version.split("+", 1)[0].split("-", 1)[0].split(".")
    if len(release) > 1:
        try:
            minor = int(release[1])
        except ValueError as exc:
            raise RuntimeError("TensorRT version minor component must be an integer") from exc
        if major == 10 and minor == 2:
            raise RuntimeError("TensorRT 10.2 is rejected by the pinned Ultralytics exporter")

    try:
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        if builder is None:
            raise RuntimeError("TensorRT could not create an engine builder")
        # This intentionally mirrors Ultralytics 8.4.92 utils/export/engine.py.
        # TRT 7-9 expose the capability flag; TRT 10+ removed it and upstream
        # defaults the branch to True.
        platform_has_fast_int8 = getattr(builder, "platform_has_fast_int8", True)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("TensorRT INT8 capability could not be inspected") from exc
    if not platform_has_fast_int8:
        raise RuntimeError(
            "TensorRT reports no fast INT8 capability; Ultralytics 8.4.92 would silently "
            "build FP32 while labeling the request INT8"
        )
    modelopt_version = None
    if major >= 11:
        try:
            from packaging.version import InvalidVersion, Version
        except ImportError as exc:  # pragma: no cover - locked export dependency
            raise RuntimeError("TensorRT 11 INT8 export requires the packaging runtime") from exc
        try:
            modelopt_version = importlib.metadata.version("nvidia-modelopt")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "TensorRT 11 INT8 export requires locally installed nvidia-modelopt>=0.44"
            ) from exc
        if (
            not isinstance(modelopt_version, str)
            or not modelopt_version
            or not modelopt_version.isprintable()
            or len(modelopt_version.encode("utf-8")) > 128
        ):
            raise RuntimeError("ModelOpt must expose a bounded printable version string")
        try:
            parsed_modelopt_version = Version(modelopt_version)
        except InvalidVersion as exc:
            raise RuntimeError("ModelOpt must expose a valid PEP 440 version") from exc
        if parsed_modelopt_version < Version("0.44"):
            raise RuntimeError("TensorRT 11 INT8 export requires nvidia-modelopt>=0.44")
        try:
            importlib.import_module("modelopt.onnx.quantization")
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT 11 INT8 export requires the ModelOpt ONNX quantization runtime"
            ) from exc
    return version, modelopt_version


def _checkpoint_max_stride(value: object) -> int:
    """Return a bounded positive integer stride from backend model metadata."""
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception as exc:
            raise ValueError("checkpoint stride metadata could not be inspected") from exc

    flattened: list[object] = []
    pending = [value]
    visited_nodes = 0
    while pending:
        item = pending.pop()
        visited_nodes += 1
        if visited_nodes > _MAX_MODEL_STRIDE_NODES:
            raise ValueError("checkpoint stride metadata exceeds the bounded structure limit")
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
            continue
        flattened.append(item)
        if len(flattened) > _MAX_MODEL_STRIDES:
            raise ValueError(
                f"checkpoint stride metadata exceeds the {_MAX_MODEL_STRIDES}-value limit"
            )
    if not flattened:
        raise ValueError("checkpoint stride metadata must not be empty")
    strides: list[int] = []
    for item in flattened:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item <= 0
            or not float(item).is_integer()
            or item > 4096
        ):
            raise ValueError("checkpoint strides must be positive integers no greater than 4096")
        strides.append(int(item))
    return max(strides)


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
        if not _is_bounded_path_text(self.artifact_path):
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


def _checkpoint_input_channels(model_graph: object) -> int:
    """Mirror the pinned exporter's ``model.yaml.get('channels', 3)`` input shape."""
    model_yaml = getattr(model_graph, "yaml", None)
    if not isinstance(model_yaml, Mapping):
        raise ValueError("checkpoint model must expose mapping-valued yaml metadata")
    channels = model_yaml.get("channels", 3)
    if type(channels) is not int or channels <= 0:
        raise ValueError("checkpoint input channels must be a positive integer")
    return channels


#: manwe format name → how to produce it + crebain backend + gotchas from research.
EXPORT_FORMATS: dict[str, FormatSpec] = {
    "onnx": FormatSpec("onnx", "onnx", ".onnx", "pin opset, run onnxslim, keep NMS external"),
    "tensorrt": FormatSpec(
        "engine",
        "tensorrt",
        ".engine",
        "NVIDIA-only; INT8 requires ≥1000 unique effective val tensors + domain evidence",
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
            self.closed = True
            os.close(self.parent_fd)


def _is_bounded_path_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not any(character in value for character in "\0\r\n")
        and len(value) <= 4096
    )


def _prepare_destination(path: str, allowed_suffixes: set[str]) -> _PreparedDestination:
    if not _is_bounded_path_text(path):
        raise ValueError("output must be a bounded nonempty destination path string")
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
    source_fd = open_directory_nofollow(source.absolute(), "private export bundle")
    try:
        _copy_directory_fd(
            source_fd,
            destination_fd,
            display=str(source),
            max_bytes=DEFAULT_MAX_ARTIFACT_BYTES,
            max_entries=DEFAULT_MAX_ARTIFACT_ENTRIES,
            destination_directory_mode=0o755,
            destination_file_mode=0o644,
        )
    except BaseException as error:
        try:
            os.close(source_fd)
        except BaseException as cleanup:
            attach_cleanup_failure(error, cleanup, "private export source cleanup failed")
        raise
    os.close(source_fd)


def _copy_regular_snapshot(
    source_handle: BinaryIO,
    destination_handle: BinaryIO,
    *,
    display: Path,
) -> None:
    """Copy exactly one stable verified file without allowing concurrent growth."""
    before = os.fstat(source_handle.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"private export artifact is not a regular file: {display}")
    if before.st_size <= 0 or before.st_size > DEFAULT_MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            f"private export artifact size is outside the bounded publication contract: {display}"
        )
    total = 0
    while True:
        chunk = source_handle.read(min(1 << 20, before.st_size - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > before.st_size:
            raise RuntimeError(f"private export artifact grew while being published: {display}")
        destination_handle.write(chunk)
    destination_handle.flush()
    destination_metadata = os.fstat(destination_handle.fileno())
    after = os.fstat(source_handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        total != before.st_size
        or destination_metadata.st_size != total
        or after_identity != before_identity
    ):
        raise RuntimeError(f"private export artifact changed while being published: {display}")
    os.fsync(destination_handle.fileno())
    if os.fstat(destination_handle.fileno()).st_size != total:
        raise RuntimeError(f"private export artifact changed while being published: {display}")


def _entry_identity_at(parent_fd: int, name: str) -> tuple[int, int] | None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _finalize_published_mode(
    parent_fd: int,
    name: str,
    identity: tuple[int, int],
    *,
    is_directory: bool,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if is_directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    else:
        flags |= getattr(os, "O_NONBLOCK", 0)
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        metadata = os.fstat(fd)
        expected_kind = stat.S_ISDIR if is_directory else stat.S_ISREG
        if not expected_kind(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != identity:
            raise RuntimeError("output destination was replaced before permission finalization")
        os.fchmod(fd, 0o755 if is_directory else 0o644)
        os.fsync(fd)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        raise
    os.close(fd)


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
                    created = os.stat(
                        anchor.path.name,
                        dir_fd=anchor.parent_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(created.st_mode):
                        raise RuntimeError("output destination is not the directory just created")
                    created_identity = (created.st_dev, created.st_ino)
                    created_is_directory = True
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    root_fd = os.open(anchor.path.name, flags, dir_fd=anchor.parent_fd)
                    try:
                        metadata = os.fstat(root_fd)
                        if (
                            not stat.S_ISDIR(metadata.st_mode)
                            or (metadata.st_dev, metadata.st_ino) != created_identity
                        ):
                            raise RuntimeError(
                                "output destination was replaced while it was being opened"
                            )
                        _copy_directory_fd_relative(snapshot.path, root_fd)
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
                    try:
                        metadata = os.fstat(fd)
                        created_identity = (metadata.st_dev, metadata.st_ino)
                        with snapshot.path.open("rb") as source_handle:
                            owned_fd = fd
                            fd = -1
                            target = owned_binary_writer(owned_fd)
                            with target:
                                _copy_regular_snapshot(
                                    source_handle,
                                    target,
                                    display=snapshot.path,
                                )
                    finally:
                        if fd >= 0:
                            os.close(fd)
                if _entry_identity_at(anchor.parent_fd, anchor.path.name) != created_identity:
                    raise RuntimeError(
                        "output destination was replaced while it was being published"
                    )
                if sha256_artifact_at(anchor.parent_fd, anchor.path.name) != digest:
                    raise RuntimeError("published export digest differs from the verified artifact")
                if _entry_identity_at(anchor.parent_fd, anchor.path.name) != created_identity:
                    raise RuntimeError(
                        "output destination was replaced while its digest was being verified"
                    )
                if created_identity is None:
                    raise RuntimeError("output destination identity was not established")
                anchor.assert_parent_path()
                _finalize_published_mode(
                    anchor.parent_fd,
                    anchor.path.name,
                    created_identity,
                    is_directory=created_is_directory,
                )
                anchor.assert_parent_path()
                if _entry_identity_at(anchor.parent_fd, anchor.path.name) != created_identity:
                    raise RuntimeError(
                        "output destination was replaced while the export was being published"
                    )
                os.fsync(anchor.parent_fd)
            except BaseException as error:
                if created_identity is not None and hasattr(error, "add_note"):
                    error.add_note(
                        "automatic output rollback is disabled because POSIX cannot unlink "
                        "a pathname conditionally by inode; inspect the private-mode destination "
                        f"before manual recovery: {anchor.path}"
                    )
                raise
    except BaseException as error:
        if owns_anchor:
            try:
                anchor.close()
            except BaseException as cleanup:
                attach_cleanup_failure(error, cleanup, "output parent cleanup failed")
        raise
    if owns_anchor:
        try:
            anchor.close()
        except OSError as cleanup:
            log.warning("published output but could not close its parent descriptor: %s", cleanup)


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
    if int8 and data is None:
        raise ValueError("int8 export requires a calibration dataset manifest")
    calibration_snapshot = None
    weights_snapshot = None
    destination: _PreparedDestination | None = None
    receipt: ExportReceipt | None = None
    try:
        from ..common.device import resolve_device
        from ..vision.train import resolve_ultralytics_device

        resolved_device = resolve_device(device)
        if export_format == "tensorrt" and resolved_device.kind != "cuda":
            raise RuntimeError("TensorRT export requires an available CUDA device")
        dev = resolve_ultralytics_device(resolved_device)
        if int8:
            assert data is not None
            tensorrt_version, modelopt_version = _preflight_tensorrt_int8(resolved_device.index)
            calibration_snapshot = snapshot_local_calibration_dataset(
                data,
                image_size=imgsz,
                tensorrt_version=tensorrt_version,
                modelopt_version=modelopt_version,
            )

        destination = _prepare_destination(output, output_suffixes)
        weights_snapshot = ArtifactSnapshot(weights, weights_sha256, allowed_suffixes={".pt"})
        calibration_sha256 = (
            calibration_snapshot.sha256 if calibration_snapshot is not None else None
        )

        harden_ultralytics_runtime()
        ultralytics = require("ultralytics", "export")
        verify_ultralytics_policy()
        model = ultralytics.YOLO(str(weights_snapshot.path))
        if getattr(model, "task", None) != "detect":
            raise ValueError(
                f"checkpoint task must be 'detect', got {getattr(model, 'task', None)!r}"
            )
        model_graph = getattr(model, "model", None)
        end_to_end = getattr(model_graph, "end2end", None)
        if type(end_to_end) is not bool:
            raise ValueError("checkpoint must expose a boolean end2end model flag")
        if end_to_end:
            raise ValueError(
                "end-to-end detector exports are not supported until their output graph "
                "can be independently inspected"
            )
        if int8 and _checkpoint_input_channels(model_graph) != 3:
            raise ValueError(
                "TensorRT INT8 calibration is verified only for 3-channel detector checkpoints"
            )
        max_stride = _checkpoint_max_stride(getattr(model_graph, "stride", None))
        if imgsz % max_stride != 0:
            raise ValueError(
                f"imgsz={imgsz} must be divisible by checkpoint max stride {max_stride}; "
                "the exporter otherwise rounds the graph shape and falsifies the receipt"
            )
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
            kwargs["batch"] = 1
            kwargs["fraction"] = 1.0
            kwargs["rect"] = False
            kwargs["dynamic"] = False

        log.info("exporting verified checkpoint → %s (%s)", export_format, spec.notes)
        if int8:
            with deterministic_ultralytics_validation_format():
                produced_value = model.export(**kwargs)
        else:
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
        receipt = ExportReceipt(
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
            end_to_end=False,
            calibration_manifest_sha256=calibration_sha256,
        )
    except BaseException as error:
        for resource, label in (
            (weights_snapshot, "weights snapshot cleanup failed"),
            (calibration_snapshot, "calibration snapshot cleanup failed"),
            (destination, "output parent cleanup failed"),
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as cleanup:
                attach_cleanup_failure(error, cleanup, label)
        raise
    for resource, label in (
        (weights_snapshot, "weights snapshot"),
        (calibration_snapshot, "calibration snapshot"),
        (destination, "output parent descriptor"),
    ):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as cleanup:
            log.warning("export succeeded but %s cleanup failed: %s", label, cleanup)
    if receipt is None:
        raise RuntimeError("export completed without constructing a receipt")
    return receipt


__all__ = ["FormatSpec", "ExportReceipt", "EXPORT_FORMATS", "export_model"]
