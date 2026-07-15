"""Emit a producer-side candidate contract for one exported model artifact."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import secrets
import stat
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

# At the largest receipt image size (4096), a dense P2/P3/P4/P5 pyramid has
# 1,392,640 cells.  Keep a conservative finite margin without accepting the
# generic TensorSpec limit (2^31-1) as a plausible detector output allocation.
_MAX_RAW_DETECT_ANCHORS = 2_000_000
_MAX_STAGE_ATTEMPTS = 128
_STAGE_JSON = "contract.json"
_STAGE_MARKDOWN = "contract.md"


@dataclass(frozen=True)
class _SidecarPublication:
    identity: tuple[int, int]
    size: int
    sha256: str


@dataclass(frozen=True)
class _StageCleanupResult:
    marker_removed: bool
    removal_synced: bool


@dataclass(frozen=True)
class _ContractCommitBoundary:
    parent_path: pathlib.Path
    parent_identity: tuple[int, int]
    artifact_name: str
    artifact_sha256: str
    final_publications: dict[str, _SidecarPublication]


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


def _entry_identity(parent_fd: int, name: str) -> tuple[int, int] | None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _write_staged_text(directory_fd: int, name: str, value: str) -> _SidecarPublication:
    """Durably create and verify one private staged sidecar."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd: int | None = None
    payload = value.encode("utf-8")
    try:
        fd = os.open(name, flags, 0o644, dir_fd=directory_fd)
        metadata = os.fstat(fd)
        created_identity = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            final_metadata = os.fstat(handle.fileno())
            if final_metadata.st_size != len(payload):
                raise RuntimeError(f"staged sidecar length changed while writing {name}")
    finally:
        if fd is not None:
            os.close(fd)
    publication = _SidecarPublication(
        created_identity,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    if not _sidecar_matches(directory_fd, name, publication):
        raise RuntimeError(f"staged sidecar was replaced or modified while writing {name}")
    return publication


def _sidecar_matches(
    directory_fd: int,
    name: str,
    publication: _SidecarPublication,
) -> bool:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError:
        return False
    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return False
            payload = handle.read(publication.size + 1)
            after = os.fstat(handle.fileno())
    except OSError:
        return False
    identity = (before.st_dev, before.st_ino)
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return (
        identity == publication.identity
        and stable
        and before.st_size == publication.size
        and len(payload) == publication.size
        and hashlib.sha256(payload).hexdigest() == publication.sha256
    )


def _stage_is_bound(
    parent_fd: int, stage_fd: int, stage_name: str, expected_identity: tuple[int, int]
) -> bool:
    try:
        opened = os.fstat(stage_fd)
        named = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(opened.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and (opened.st_dev, opened.st_ino) == expected_identity
        and (named.st_dev, named.st_ino) == expected_identity
    )


def _create_private_stage(parent_fd: int) -> tuple[str, int, tuple[int, int]]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(_MAX_STAGE_ATTEMPTS):
        stage_name = f".manwe-contract-{os.getpid()}-{secrets.token_hex(16)}.in-progress"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        stage_fd: int | None = None
        try:
            stage_fd = os.open(stage_name, directory_flags, dir_fd=parent_fd)
            os.fchmod(stage_fd, 0o700)
            metadata = os.fstat(stage_fd)
            identity = (metadata.st_dev, metadata.st_ino)
            if not _stage_is_bound(parent_fd, stage_fd, stage_name, identity):
                raise RuntimeError("contract staging directory was replaced while opening")
            os.fsync(parent_fd)
            return stage_name, stage_fd, identity
        except BaseException:
            if stage_fd is not None:
                os.close(stage_fd)
            # Never check-then-remove a pathname after acquisition failed. The
            # private marker is safer recovery evidence than deleting a replacement.
            raise
    raise RuntimeError("could not reserve a unique contract staging directory")


def _cleanup_private_stage(
    parent_fd: int,
    stage_fd: int,
    stage_name: str,
    stage_identity: tuple[int, int],
    publications: dict[str, _SidecarPublication],
    commit_boundary: _ContractCommitBoundary,
) -> _StageCleanupResult:
    """Best-effort cleanup of the bound private stage, never final pathnames.

    POSIX has no conditional unlink-by-inode primitive. The high-entropy, mode-0700
    stage is therefore cleaned only when every current entry still has the exact
    identity, size, and digest captured during staging.
    """
    try:
        if not _stage_is_bound(parent_fd, stage_fd, stage_name, stage_identity):
            return _StageCleanupResult(False, False)
        names = sorted(os.listdir(stage_fd))
        if names != sorted(publications):
            return _StageCleanupResult(False, False)
        if any(not _sidecar_matches(stage_fd, name, publications[name]) for name in names):
            return _StageCleanupResult(False, False)
        if not _commit_boundary_matches(parent_fd, commit_boundary):
            return _StageCleanupResult(False, False)
        for name in names:
            os.unlink(name, dir_fd=stage_fd)
        os.fsync(stage_fd)
        if not _stage_is_bound(parent_fd, stage_fd, stage_name, stage_identity):
            return _StageCleanupResult(False, False)
        # Marker removal is the commit point. Revalidate final names immediately
        # beforehand so any detected parent/artifact/final swap preserves it.
        if not _commit_boundary_matches(parent_fd, commit_boundary):
            return _StageCleanupResult(False, False)
        os.rmdir(stage_name, dir_fd=parent_fd)
    except OSError:
        return _StageCleanupResult(False, False)
    if not _sync_parent_after_marker_removal(parent_fd):
        return _StageCleanupResult(True, False)
    return _StageCleanupResult(True, True)


def _commit_boundary_matches(parent_fd: int, boundary: _ContractCommitBoundary) -> bool:
    try:
        _assert_directory_path(
            boundary.parent_path,
            parent_fd,
            boundary.parent_identity,
        )
        if sha256_artifact_at(parent_fd, boundary.artifact_name) != boundary.artifact_sha256:
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    finals_match = all(
        _sidecar_matches(parent_fd, name, publication)
        for name, publication in boundary.final_publications.items()
    )
    if not finals_match:
        return False
    # Artifact hashing may be lengthy for a directory bundle. Recheck the cheap
    # pathname binding last so a parent swap during that read cannot commit.
    try:
        _assert_directory_path(
            boundary.parent_path,
            parent_fd,
            boundary.parent_identity,
        )
    except RuntimeError:
        return False
    return True


def _sync_parent_after_marker_removal(parent_fd: int) -> bool:
    try:
        os.fsync(parent_fd)
    except OSError:
        return False
    return True


def _link_staged_sidecar(stage_fd: int, stage_name: str, parent_fd: int, final_name: str) -> None:
    """Publish one hard link atomically without following or replacing paths."""
    os.link(
        stage_name,
        final_name,
        src_dir_fd=stage_fd,
        dst_dir_fd=parent_fd,
        follow_symlinks=False,
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
        copied_inputs = tuple(copy.deepcopy(self.inputs))
        copied_outputs = tuple(copy.deepcopy(self.outputs))
        tensor_errors = _signature_tensor_errors(copied_inputs, copied_outputs)
        if tensor_errors:
            raise ValueError("invalid signature tensor metadata: " + "; ".join(tensor_errors))
        for name in ("preprocess", "postprocess", "failure_behavior", "evidence"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\0" in value
                or len(value) > 4096
            ):
                raise ValueError(f"signature {name} must be a nonempty string")
        object.__setattr__(self, "inputs", copied_inputs)
        object.__setattr__(self, "outputs", copied_outputs)


def _signature_tensor_errors(
    inputs: tuple[TensorSpec, ...], outputs: tuple[TensorSpec, ...]
) -> list[str]:
    """Validate canonical tensor metadata, including names across both interfaces."""
    errors: list[str] = []
    names: list[str] = []
    for collection_name, tensors in (("inputs", inputs), ("outputs", outputs)):
        for index, tensor in enumerate(tensors):
            errors.extend(tensor.validation_errors(f"signature.{collection_name}[{index}]"))
            if isinstance(tensor.name, str):
                names.append(tensor.name)
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        errors.append(f"signature contains duplicate tensor names {duplicate_names}")
    return errors


def _validate_raw_detect_signature(
    receipt: ExportReceipt,
    inputs: tuple[TensorSpec, ...],
    outputs: tuple[TensorSpec, ...],
) -> None:
    """Tie one supported raw Ultralytics detect interface to its export receipt.

    Embedded-NMS and end-to-end heads have backend- and model-specific output
    structures.  Until the receipt records enough facts to validate those
    structures, rejecting them is safer than treating an arbitrary shape as
    verified.
    """
    tensor_errors = _signature_tensor_errors(inputs, outputs)
    if tensor_errors:
        raise ValueError("invalid signature tensor metadata: " + "; ".join(tensor_errors))
    if receipt.embedded_nms:
        raise ValueError("embedded-NMS tensor signatures are not yet supported")
    if receipt.end_to_end:
        raise ValueError("end-to-end detector tensor signatures are not yet supported")
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "raw detect contracts require exactly one image input and one prediction output"
        )

    image = inputs[0]
    if image.layout == "NCHW/RGB":
        expected_image_shape: list[int | str] = [1, 3, receipt.image_size, receipt.image_size]
    else:
        raise ValueError("raw detect image input layout must be NCHW/RGB")
    if image.shape != expected_image_shape:
        raise ValueError(
            f"raw detect image input shape must be {expected_image_shape} for "
            f"image_size={receipt.image_size}, got {image.shape}"
        )
    if image.dtype not in {"float16", "float32", "uint8"}:
        raise ValueError("raw detect image input dtype must be float16, float32, or uint8")

    prediction = outputs[0]
    expected_features = 4 + receipt.class_count
    if len(prediction.shape) != 3:
        raise ValueError("raw detect prediction output must have rank 3 [1, 4+classes, anchors]")
    if prediction.shape[0] != 1 or prediction.shape[1] != expected_features:
        raise ValueError(
            "raw detect prediction output must have shape "
            f"[1, {expected_features}, anchors], got {prediction.shape}"
        )
    anchor_count = prediction.shape[2]
    if type(anchor_count) is not int:
        raise ValueError("raw detect prediction anchor dimension must be a concrete integer")
    if not 1 <= anchor_count <= _MAX_RAW_DETECT_ANCHORS:
        raise ValueError(
            f"raw detect prediction anchor dimension must be in [1, {_MAX_RAW_DETECT_ANCHORS}]"
        )
    if prediction.dtype not in {"float16", "float32"}:
        raise ValueError("raw detect prediction output dtype must be float16 or float32")
    if prediction.layout:
        raise ValueError("raw detect prediction output layout must be empty")


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
    signature_inputs = tuple(copy.deepcopy(signature.inputs))
    signature_outputs = tuple(copy.deepcopy(signature.outputs))
    _validate_raw_detect_signature(receipt, signature_inputs, signature_outputs)

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
        inputs=list(signature_inputs),
        outputs=list(signature_outputs),
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
    """Stage, verify, and no-replace publish JSON + Markdown contract sidecars."""
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
    stage_fd: int | None = None
    stage_name: str | None = None
    stage_identity: tuple[int, int] | None = None
    publications: dict[str, _SidecarPublication] = {}
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
        if _entry_identity(parent_fd, json_path.name) is not None:
            raise FileExistsError(f"contract sidecar already exists: {json_path}")
        if _entry_identity(parent_fd, md_path.name) is not None:
            raise FileExistsError(f"contract sidecar already exists: {md_path}")
        stage_name, stage_fd, stage_identity = _create_private_stage(parent_fd)

        json_publication = _write_staged_text(stage_fd, _STAGE_JSON, json_value)
        publications[_STAGE_JSON] = json_publication
        md_publication = _write_staged_text(stage_fd, _STAGE_MARKDOWN, md_value)
        publications[_STAGE_MARKDOWN] = md_publication
        os.fsync(stage_fd)
        if not _stage_is_bound(parent_fd, stage_fd, stage_name, stage_identity):
            raise RuntimeError("contract staging directory was replaced before publication")
        if not _sidecar_matches(stage_fd, _STAGE_JSON, json_publication) or not _sidecar_matches(
            stage_fd, _STAGE_MARKDOWN, md_publication
        ):
            raise RuntimeError("staged contract sidecars were replaced or modified")
        if sha256_artifact_at(parent_fd, path.absolute().name) != contract.file_sha256.lower():
            raise RuntimeError("artifact changed while contract sidecars were being published")
        _assert_directory_path(absolute_parent, parent_fd, parent_identity)

        _link_staged_sidecar(stage_fd, _STAGE_JSON, parent_fd, json_path.name)
        if not _sidecar_matches(parent_fd, json_path.name, json_publication):
            raise RuntimeError("published JSON sidecar was replaced or modified")
        _assert_directory_path(absolute_parent, parent_fd, parent_identity)
        if not _stage_is_bound(parent_fd, stage_fd, stage_name, stage_identity):
            raise RuntimeError("contract staging directory was replaced during publication")

        _link_staged_sidecar(stage_fd, _STAGE_MARKDOWN, parent_fd, md_path.name)
        if not _sidecar_matches(parent_fd, md_path.name, md_publication):
            raise RuntimeError("published Markdown sidecar was replaced or modified")
        if not _sidecar_matches(stage_fd, _STAGE_JSON, json_publication) or not _sidecar_matches(
            stage_fd, _STAGE_MARKDOWN, md_publication
        ):
            raise RuntimeError("staged contract sidecars changed after publication")
        if not _sidecar_matches(
            parent_fd, json_path.name, json_publication
        ) or not _sidecar_matches(parent_fd, md_path.name, md_publication):
            raise RuntimeError("published contract sidecars changed before commit")
        if sha256_artifact_at(parent_fd, path.absolute().name) != contract.file_sha256.lower():
            raise RuntimeError("artifact changed while contract sidecars were being published")
        _assert_directory_path(absolute_parent, parent_fd, parent_identity)
        os.fsync(parent_fd)
        cleanup = _cleanup_private_stage(
            parent_fd,
            stage_fd,
            stage_name,
            stage_identity,
            publications,
            _ContractCommitBoundary(
                parent_path=absolute_parent,
                parent_identity=parent_identity,
                artifact_name=path.absolute().name,
                artifact_sha256=contract.file_sha256.lower(),
                final_publications={
                    json_path.name: json_publication,
                    md_path.name: md_publication,
                },
            ),
        )
        if not cleanup.marker_removed:
            raise RuntimeError(
                "contract sidecars were published but staging cleanup failed; "
                f"manual recovery is required at {absolute_parent / stage_name}"
            )
        if not cleanup.removal_synced:
            raise RuntimeError(
                "contract sidecars are durable, but marker removal could not be synced; "
                "commit state is indeterminate and the marker may reappear after a crash"
            )
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        os.close(parent_fd)
    return json_path, md_path


__all__ = [
    "VerifiedArtifactSignature",
    "sha256_file",
    "build_export_contract",
    "save_contract",
]
