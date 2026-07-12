"""Emit a producer-side candidate contract for one exported model artifact."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import stat
from contextlib import suppress
from dataclasses import dataclass

from ..common.artifacts import sha256_artifact, sha256_artifact_at
from ..common.config_io import open_directory_nofollow
from ..common.contracts import (
    CREBAIN_CLASSES,
    Backend,
    CrebainClass,
    ModelContract,
    TensorSpec,
)
from .backends import EXPORT_FORMATS, ExportReceipt


def _path_identity(path: pathlib.Path, *, parent_fd: int | None = None) -> tuple[int, int] | None:
    try:
        metadata = (
            path.lstat()
            if parent_fd is None
            else os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _unlink_if_identity(
    path: pathlib.Path, identity: tuple[int, int], *, parent_fd: int | None = None
) -> None:
    if _path_identity(path, parent_fd=parent_fd) == identity:
        with suppress(FileNotFoundError):
            if parent_fd is None:
                path.unlink()
            else:
                os.unlink(path.name, dir_fd=parent_fd)


@dataclass(frozen=True)
class _SidecarPublication:
    identity: tuple[int, int]
    size: int
    sha256: str


def _assert_directory_path(
    path: pathlib.Path, parent_fd: int, expected_identity: tuple[int, int]
) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("contract parent was replaced during publication") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (
            metadata.st_dev,
            metadata.st_ino,
        )
        != expected_identity
    ):
        raise RuntimeError("contract parent was replaced during publication")


def _write_text_exclusive(
    path: pathlib.Path, value: str, *, parent_fd: int | None = None
) -> _SidecarPublication:
    """Durably create one sidecar without following or replacing any path."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd: int | None = None
    created_identity: tuple[int, int] | None = None
    owns_parent = parent_fd is None
    parent_identity: tuple[int, int] | None = None
    payload = value.encode("utf-8")
    try:
        if parent_fd is None:
            absolute_parent = path.absolute().parent
            parent_fd = open_directory_nofollow(absolute_parent, "contract parent")
            metadata = os.fstat(parent_fd)
            parent_identity = (metadata.st_dev, metadata.st_ino)
        fd = os.open(path.name, flags, 0o644, dir_fd=parent_fd)
        metadata = os.fstat(fd)
        created_identity = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            final_metadata = os.fstat(handle.fileno())
            if final_metadata.st_size != len(payload):
                raise RuntimeError(f"sidecar length changed while it was being written: {path}")
        if _path_identity(path, parent_fd=parent_fd) != created_identity:
            raise RuntimeError(f"sidecar was replaced while it was being written: {path}")
        os.fsync(parent_fd)
        if parent_identity is not None:
            _assert_directory_path(path.absolute().parent, parent_fd, parent_identity)
    except BaseException:
        if fd is not None:
            os.close(fd)
        if created_identity is not None and parent_fd is not None:
            _unlink_if_identity(path, created_identity, parent_fd=parent_fd)
        raise
    finally:
        if owns_parent and parent_fd is not None:
            os.close(parent_fd)
    if created_identity is None:  # pragma: no cover - os.open succeeded on every return path
        raise RuntimeError("sidecar identity was not captured")
    return _SidecarPublication(
        created_identity,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _sidecar_matches(
    path: pathlib.Path,
    publication: _SidecarPublication,
    *,
    parent_fd: int | None = None,
) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    owns_parent = parent_fd is None
    try:
        if parent_fd is None:
            parent_fd = open_directory_nofollow(path.absolute().parent, "contract parent")
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError:
        if owns_parent and parent_fd is not None:
            os.close(parent_fd)
        return False
    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(publication.size + 1)
            after = os.fstat(handle.fileno())
    except OSError:
        return False
    finally:
        if owns_parent and parent_fd is not None:
            os.close(parent_fd)
    identity = (before.st_dev, before.st_ino)
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return (
        identity == publication.identity
        and stable
        and before.st_size == publication.size
        and len(payload) == publication.size
        and hashlib.sha256(payload).hexdigest() == publication.sha256
    )


def sha256_file(path: str | pathlib.Path) -> str:
    """Return the bounded deterministic digest for a local artifact."""
    return sha256_artifact(path)


@dataclass(frozen=True)
class VerifiedArtifactSignature:
    """Backend-inspected tensor and processing facts tied to one artifact digest."""

    artifact_sha256: str
    precision: str
    embedded_nms: bool
    opset: int | None
    source_classes: tuple[str, ...]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    preprocess: str
    postprocess: str
    failure_behavior: str
    evidence: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_sha256, str)
            or len(self.artifact_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in self.artifact_sha256)
        ):
            raise ValueError("signature artifact_sha256 must be a 64-character hex digest")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
        if not isinstance(self.precision, str) or self.precision not in {
            "float32",
            "float16",
            "int8",
        }:
            raise ValueError("signature precision must be float32, float16, or int8")
        if type(self.embedded_nms) is not bool:
            raise TypeError("signature embedded_nms must be a boolean")
        if self.opset is not None and (type(self.opset) is not int or not 12 <= self.opset <= 20):
            raise ValueError("signature opset must be an integer in [12, 20] or None")
        if not isinstance(self.source_classes, tuple) or not self.source_classes:
            raise ValueError("signature source_classes must be a nonempty tuple")
        if len(self.source_classes) > 4096 or any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or not value.isprintable()
            or len(value.encode("utf-8")) > 256
            for value in self.source_classes
        ):
            raise ValueError("signature source_classes must contain bounded printable names")
        if len(set(self.source_classes)) != len(self.source_classes):
            raise ValueError("signature source_classes must be unique")
        if not isinstance(self.inputs, tuple) or not isinstance(self.outputs, tuple):
            raise TypeError("signature inputs and outputs must be tuples")
        if not self.inputs or not self.outputs:
            raise ValueError("signature inputs and outputs must be nonempty")
        if len(self.inputs) > 64 or len(self.outputs) > 64:
            raise ValueError("signature inputs and outputs must each contain at most 64 tensors")
        if any(not isinstance(value, TensorSpec) for value in (*self.inputs, *self.outputs)):
            raise TypeError("signature tensors must be TensorSpec values")
        for name in ("preprocess", "postprocess", "failure_behavior", "evidence"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\0" in value
                or len(value) > 4096
            ):
                raise ValueError(f"signature {name} must be a nonempty string")
        object.__setattr__(self, "inputs", tuple(copy.deepcopy(self.inputs)))
        object.__setattr__(self, "outputs", tuple(copy.deepcopy(self.outputs)))


def build_export_contract(
    *,
    model_name: str,
    model_version: str,
    source: str,
    rights: str,
    receipt: ExportReceipt,
    signature: VerifiedArtifactSignature,
    class_map: dict[int, CrebainClass | None] | None = None,
    validation_data: str = "",
    benchmark_context: str = "",
) -> ModelContract:
    """Build a contract only from one receipt and separately inspected signature."""
    if not isinstance(receipt, ExportReceipt):
        raise TypeError("receipt must be an ExportReceipt")
    if not isinstance(signature, VerifiedArtifactSignature):
        raise TypeError("signature must be a VerifiedArtifactSignature")
    if receipt.format not in EXPORT_FORMATS:
        raise ValueError(f"unknown receipt format {receipt.format!r}")
    if signature.artifact_sha256.lower() != receipt.artifact_sha256.lower():
        raise ValueError("signature evidence belongs to a different artifact digest")
    for field in ("precision", "embedded_nms", "opset"):
        if getattr(signature, field) != getattr(receipt, field):
            raise ValueError(f"signature {field} does not match the export receipt")
    if signature.source_classes != receipt.source_classes:
        raise ValueError("signature source_classes do not match the export receipt")

    backend: Backend = EXPORT_FORMATS[receipt.format].crebain_backend  # type: ignore[assignment]
    artifact = pathlib.Path(receipt.artifact_path)
    if class_map is None:
        if receipt.source_classes != tuple(CREBAIN_CLASSES):
            raise ValueError(
                "class_map is required unless source_classes exactly match the candidate taxonomy"
            )
        cmap: dict[int, CrebainClass | None] = {i: value for i, value in enumerate(CREBAIN_CLASSES)}
    else:
        cmap = class_map.copy()
    export_options = json.dumps(
        {
            "format": receipt.format,
            "image_size": receipt.image_size,
            "precision": receipt.precision,
            "embedded_nms": receipt.embedded_nms,
            "opset": receipt.opset,
            "class_count": receipt.class_count,
            "source_classes": receipt.source_classes,
            "source_suffix": receipt.source_suffix,
            "end_to_end": receipt.end_to_end,
            "calibration_dataset_sha256": receipt.calibration_manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    contract = ModelContract(
        model_name=model_name,
        model_version=model_version,
        source=source,
        rights=rights,
        backend=backend,
        file_path=str(artifact),
        num_classes=receipt.class_count,
        source_classes=list(receipt.source_classes),
        file_sha256=receipt.artifact_sha256.lower(),
        source_sha256=receipt.source_sha256.lower(),
        export_options=export_options,
        signature_evidence=signature.evidence,
        inputs=copy.deepcopy(list(signature.inputs)),
        outputs=copy.deepcopy(list(signature.outputs)),
        preprocess=signature.preprocess,
        postprocess=signature.postprocess,
        class_map=cmap,
        validation_data=validation_data,
        benchmark_context=benchmark_context,
        failure_behavior=signature.failure_behavior,
    )
    # Reject malformed metadata, tensor signatures, class maps, and backend
    # suffixes before performing a potentially large artifact read.
    contract.validate(check_artifact=False)
    if backend == "coreml":
        if not artifact.is_dir():
            raise ValueError("CoreML receipt artifact must be a directory bundle")
    elif not artifact.is_file():
        raise ValueError("receipt artifact must be a regular file")
    actual_sha = sha256_file(artifact)
    if actual_sha != receipt.artifact_sha256.lower():
        raise ValueError("artifact bytes no longer match the export receipt")
    return contract


def save_contract(
    contract: ModelContract, path: str | pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """Validate and write JSON + Markdown sidecars next to the signed artifact."""
    contract.validate(check_artifact=False)
    path = pathlib.Path(path)
    artifact = pathlib.Path(contract.file_path)
    if path.absolute() != artifact.absolute():
        raise ValueError(
            f"contract output path {path} does not identify its signed artifact {artifact}"
        )
    json_path = path.with_suffix(".contract.json")
    md_path = path.with_suffix(".contract.md")
    json_value = contract.to_json()
    md_value = contract.to_markdown(check_artifact=False)
    absolute_parent = path.absolute().parent
    parent_fd = open_directory_nofollow(absolute_parent, "contract parent")
    try:
        parent_metadata = os.fstat(parent_fd)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        _assert_directory_path(absolute_parent, parent_fd, parent_identity)
        actual_sha = sha256_artifact_at(parent_fd, path.absolute().name)
        if actual_sha != contract.file_sha256.lower():
            raise ValueError(
                "artifact SHA-256 no longer matches contract: "
                f"expected {contract.file_sha256.lower()}, got {actual_sha}"
            )
        json_publication = _write_text_exclusive(json_path, json_value, parent_fd=parent_fd)
        try:
            md_publication = _write_text_exclusive(md_path, md_value, parent_fd=parent_fd)
        except BaseException:
            _unlink_if_identity(json_path, json_publication.identity, parent_fd=parent_fd)
            raise
        if not _sidecar_matches(
            json_path, json_publication, parent_fd=parent_fd
        ) or not _sidecar_matches(md_path, md_publication, parent_fd=parent_fd):
            _unlink_if_identity(json_path, json_publication.identity, parent_fd=parent_fd)
            _unlink_if_identity(md_path, md_publication.identity, parent_fd=parent_fd)
            raise RuntimeError("contract sidecars were replaced or modified during publication")
        if sha256_artifact_at(parent_fd, path.absolute().name) != contract.file_sha256.lower():
            _unlink_if_identity(json_path, json_publication.identity, parent_fd=parent_fd)
            _unlink_if_identity(md_path, md_publication.identity, parent_fd=parent_fd)
            raise RuntimeError("artifact changed while contract sidecars were being published")
        try:
            _assert_directory_path(absolute_parent, parent_fd, parent_identity)
        except BaseException:
            _unlink_if_identity(json_path, json_publication.identity, parent_fd=parent_fd)
            _unlink_if_identity(md_path, md_publication.identity, parent_fd=parent_fd)
            raise
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return json_path, md_path


__all__ = [
    "VerifiedArtifactSignature",
    "sha256_file",
    "build_export_contract",
    "save_contract",
]
