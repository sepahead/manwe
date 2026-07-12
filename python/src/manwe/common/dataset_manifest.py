"""Strict local-only validation for Ultralytics detection dataset manifests."""

from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence
from itertools import combinations

from .config_io import read_strict_yaml, validate_local_path

_MAX_MANIFEST_BYTES = 1 << 20
_MAX_SPLIT_PATHS = 1024
_MAX_PATH_BYTES = 4096
_MAX_CLASSES = 4096
_MAX_CLASS_NAME_BYTES = 256
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}
_ALLOWED_KEYS = {"path", "train", "val", "test", "names", "nc", "channels"}
_CALIBRATION_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_MIN_CALIBRATION_IMAGES = 1_000
_MAX_DATASET_BYTES = 64 * 1024 * 1024 * 1024
_MAX_DATASET_ENTRIES = 100_000


def _reject_remote_or_archive(value: str, field: str) -> None:
    lowered = value.strip().lower()
    if not lowered or "://" in lowered or lowered.startswith(("data:", "file:")):
        raise ValueError(f"dataset {field} must be a nonempty local filesystem path")
    if pathlib.PurePath(lowered).suffix in _ARCHIVE_SUFFIXES:
        raise ValueError(f"dataset {field} must not reference an archive")
    if any(char in value for char in "\r\n\0"):
        raise ValueError(f"dataset {field} contains a control character")
    if len(value.encode("utf-8")) > _MAX_PATH_BYTES:
        raise ValueError(f"dataset {field} path exceeds {_MAX_PATH_BYTES} UTF-8 bytes")


def _split_values(value: object, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if not value or any(not isinstance(item, str) for item in value):
            raise ValueError(f"dataset {field} must contain nonempty path strings")
        if len(value) > _MAX_SPLIT_PATHS:
            raise ValueError(f"dataset {field} exceeds the {_MAX_SPLIT_PATHS}-path safety limit")
        return list(value)
    raise ValueError(f"dataset {field} must be a path string or nonempty path list")


def _validated_class_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset names must contain strings")
    normalized = value.strip()
    if (
        not normalized
        or not normalized.isprintable()
        or len(normalized.encode("utf-8")) > _MAX_CLASS_NAME_BYTES
    ):
        raise ValueError(
            "dataset names must be nonempty printable strings no longer than "
            f"{_MAX_CLASS_NAME_BYTES} UTF-8 bytes"
        )
    return normalized


class DatasetManifestSnapshot:
    """A private normalized dataset manifest kept alive for one operation."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        import yaml

        self._payload = copy.deepcopy(dict(payload))
        self._temporary = tempfile.TemporaryDirectory(prefix="manwe-dataset-")
        temporary_root = pathlib.Path(self._temporary.name).resolve(strict=True)
        self.path = temporary_root / "dataset.yaml"
        self.root = pathlib.Path(str(self._payload["path"]))
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(self.path, flags, 0o400)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(self._payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            self._temporary.cleanup()
            raise

    def close(self) -> None:
        self._temporary.cleanup()

    def _payload_for_root(self, root: pathlib.Path) -> dict[str, object]:
        payload = copy.deepcopy(self._payload)
        payload["path"] = str(root)
        for field in ("train", "val", "test"):
            if field not in payload:
                continue
            values = _split_values(payload[field], field)
            remapped = [str(root / pathlib.Path(value).relative_to(self.root)) for value in values]
            payload[field] = remapped[0] if isinstance(payload[field], str) else remapped
        return payload

    def _calibration_digests(self) -> tuple[str, str]:
        """Return the normalized-manifest/tree digest and the tree digest."""
        import yaml

        from .artifacts import _tree_entries, sha256_artifact

        entries = _tree_entries(self.root, _MAX_DATASET_ENTRIES)
        image_count = sum(
            kind == "file" and entry.suffix.lower() in _CALIBRATION_IMAGE_SUFFIXES
            for entry, kind, _metadata in entries
        )
        if image_count < _MIN_CALIBRATION_IMAGES:
            raise ValueError(
                "INT8 calibration dataset must contain at least "
                f"{_MIN_CALIBRATION_IMAGES} local still images"
            )
        canonical_payload = self._payload_for_root(pathlib.Path("."))
        manifest_bytes = yaml.safe_dump(canonical_payload, sort_keys=True).encode("utf-8")
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError(f"normalized calibration manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
        content_digest = sha256_artifact(
            self.root,
            max_bytes=_MAX_DATASET_BYTES,
            max_entries=_MAX_DATASET_ENTRIES,
        )
        hasher = hashlib.sha256(b"manwe-calibration-dataset-v2\0")
        hasher.update(hashlib.sha256(manifest_bytes).digest())
        hasher.update(bytes.fromhex(content_digest))
        return hasher.hexdigest(), content_digest

    def calibration_digest(self) -> str:
        """Bind normalized metadata and every bounded dataset-root entry.

        The minimum image count is a policy floor, not a claim that 1,000 images
        are statistically sufficient for every INT8 model.
        """
        digest, _content_digest = self._calibration_digests()
        return digest

    def __enter__(self) -> DatasetManifestSnapshot:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class CalibrationDatasetSnapshot:
    """A bounded, read-only private copy of the exact calibration dataset."""

    def __init__(self, source: DatasetManifestSnapshot) -> None:
        from .artifacts import ArtifactSnapshot

        source_metadata = source.root.lstat()
        source_identity = (source_metadata.st_dev, source_metadata.st_ino)
        expected_digest, source_content_digest = source._calibration_digests()
        artifact_snapshot = None
        manifest_snapshot = None
        try:
            artifact_snapshot = ArtifactSnapshot(
                source.root,
                source_content_digest,
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
            )
            after_copy = source.root.lstat()
            if (after_copy.st_dev, after_copy.st_ino) != source_identity:
                raise RuntimeError("source calibration dataset was replaced while snapshotting")
            manifest_snapshot = DatasetManifestSnapshot(
                source._payload_for_root(artifact_snapshot.path)
            )
            snapshot_digest, snapshot_content_digest = manifest_snapshot._calibration_digests()
            if (
                snapshot_digest != expected_digest
                or snapshot_content_digest != source_content_digest
            ):
                raise RuntimeError("private calibration snapshot differs from its verified source")
        except BaseException:
            if manifest_snapshot is not None:
                manifest_snapshot.close()
            if artifact_snapshot is not None:
                artifact_snapshot.close()
            raise

        self._artifact_snapshot = artifact_snapshot
        self._manifest_snapshot = manifest_snapshot
        self._source_root = source.root
        self._source_identity = source_identity
        self._source_content_digest = source_content_digest
        self.sha256 = expected_digest
        self.root = artifact_snapshot.path
        self.path = manifest_snapshot.path
        self._closed = False

    def calibration_digest(self) -> str:
        """Re-hash the private copy so backend-time mutation fails closed."""
        if self._closed:
            raise RuntimeError("calibration snapshot is closed")
        return self._manifest_snapshot.calibration_digest()

    def assert_source_unchanged(self) -> None:
        """Require the caller-visible source tree to remain the snapshotted tree."""
        if self._closed:
            raise RuntimeError("calibration snapshot is closed")
        from .artifacts import sha256_artifact

        try:
            validate_local_path(
                self._source_root, "source calibration dataset", require_directory=True
            )
            metadata = self._source_root.lstat()
            if (metadata.st_dev, metadata.st_ino) != self._source_identity:
                raise RuntimeError("source calibration dataset root was replaced")
            current_digest = sha256_artifact(
                self._source_root,
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("source calibration dataset changed after snapshotting") from exc
        if current_digest != self._source_content_digest:
            raise RuntimeError("source calibration dataset changed after snapshotting")

    def close(self) -> None:
        if not self._closed:
            try:
                self._manifest_snapshot.close()
            finally:
                self._artifact_snapshot.close()
                self._closed = True

    def __enter__(self) -> CalibrationDatasetSnapshot:
        if self._closed:
            raise RuntimeError("calibration snapshot is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def validate_local_detection_manifest(
    path: str | pathlib.Path,
) -> DatasetManifestSnapshot:
    """Validate local splits and return a private directive-free YAML snapshot."""
    manifest = pathlib.Path(path).expanduser().absolute()
    try:
        payload = read_strict_yaml(manifest, _MAX_MANIFEST_BYTES, "dataset manifest")
    except ImportError as exc:  # pragma: no cover - optional-dependency boundary
        raise RuntimeError("dataset validation requires the config/vision extra") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("dataset manifest must contain a YAML mapping")
    keys = set(payload)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError("dataset manifest keys must be strings")
    if "download" in keys:
        raise ValueError("dataset manifest download directives are forbidden")
    unknown = sorted(keys - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"dataset manifest contains unsupported keys: {unknown}")
    if "train" not in payload or "val" not in payload or "names" not in payload:
        raise ValueError("dataset manifest must define train, val, and names")

    root_value = payload.get("path", ".")
    if not isinstance(root_value, str):
        raise ValueError("dataset path must be a local path string")
    _reject_remote_or_archive(root_value, "path")
    root = pathlib.Path(root_value).expanduser()
    if not root.is_absolute():
        root = manifest.parent / root
    root = root.absolute()
    validate_local_path(root, "dataset root", require_directory=True)
    root = root.resolve(strict=True)

    sanitized: dict[str, object] = {"path": str(root)}
    split_paths: dict[str, list[pathlib.Path]] = {}
    for field in ("train", "val", "test"):
        if payload.get(field) in (None, ""):
            continue
        normalized_paths: list[str] = []
        for raw in _split_values(payload[field], field):
            _reject_remote_or_archive(raw, field)
            candidate = pathlib.Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.absolute()
            if candidate.suffix.lower() == ".txt":
                raise ValueError(
                    f"dataset {field} path-list indirection is not accepted; use explicit "
                    "local directories or files"
                )
            validate_local_path(candidate, f"dataset {field}", require_directory=None)
            candidate = candidate.resolve(strict=True)
            if not candidate.is_relative_to(root):
                raise ValueError(f"dataset {field} must remain inside the declared dataset root")
            normalized_paths.append(str(candidate))
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError(f"dataset {field} must not contain duplicate paths")
        split_paths[field] = [pathlib.Path(value) for value in normalized_paths]
        sanitized[field] = normalized_paths[0] if len(normalized_paths) == 1 else normalized_paths

    for (left_name, left_paths), (right_name, right_paths) in combinations(split_paths.items(), 2):
        for left in left_paths:
            for right in right_paths:
                if left == right or left in right.parents or right in left.parents:
                    raise ValueError(
                        f"dataset {left_name} and {right_name} paths overlap: {left} and {right}"
                    )

    names = payload["names"]
    if isinstance(names, Mapping):
        if not names or any(type(key) is not int for key in names):
            raise ValueError("dataset names mapping must use integer keys and string values")
        if set(names) != set(range(len(names))):
            raise ValueError("dataset names mapping indices must be contiguous from zero")
        sanitized_names: dict[int, str] | list[str] = {
            index: _validated_class_name(names[index]) for index in range(len(names))
        }
    elif isinstance(names, Sequence) and not isinstance(names, (bytes, bytearray, str)):
        if not names:
            raise ValueError("dataset names must contain nonempty strings")
        sanitized_names = [_validated_class_name(value) for value in names]
    else:
        raise ValueError("dataset names must be a list or integer-keyed mapping")
    class_count = len(sanitized_names)
    if class_count > _MAX_CLASSES:
        raise ValueError(f"dataset names exceed the {_MAX_CLASSES}-class safety limit")
    if (
        len(set(sanitized_names.values() if isinstance(sanitized_names, dict) else sanitized_names))
        != class_count
    ):
        raise ValueError("dataset class names must be unique")
    if "nc" in payload and (type(payload["nc"]) is not int or payload["nc"] != class_count):
        raise ValueError("dataset nc must equal the number of names")
    if "channels" in payload and (
        type(payload["channels"]) is not int or payload["channels"] not in {1, 3}
    ):
        raise ValueError("dataset channels must be 1 or 3")
    sanitized["names"] = sanitized_names
    sanitized["nc"] = class_count
    if "channels" in payload:
        sanitized["channels"] = payload["channels"]
    return DatasetManifestSnapshot(sanitized)


def snapshot_local_calibration_dataset(
    path: str | pathlib.Path,
) -> CalibrationDatasetSnapshot:
    """Validate and privately snapshot one bounded local INT8 calibration dataset."""
    validated = validate_local_detection_manifest(path)
    try:
        return CalibrationDatasetSnapshot(validated)
    finally:
        validated.close()


__all__ = [
    "CalibrationDatasetSnapshot",
    "DatasetManifestSnapshot",
    "snapshot_local_calibration_dataset",
    "validate_local_detection_manifest",
]
