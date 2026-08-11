"""Strict local-only validation for Ultralytics detection dataset manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pathlib
import stat
import sys
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import cast

from .artifacts import _DescriptorTreeEntry
from .config_io import (
    load_unambiguous_yaml,
    open_directory_nofollow,
    open_regular_nofollow,
    read_bounded_regular_bytes,
)
from .fd_io import (
    attach_cleanup_failure,
    owned_binary_reader,
    owned_binary_writer,
    owned_text_writer,
)

_MAX_MANIFEST_BYTES = 1 << 20
_MAX_SPLIT_PATHS = 1024
_MAX_PATH_BYTES = 4096
_MAX_CLASSES = 4096
_MAX_CLASS_NAME_BYTES = 256
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}
_ALLOWED_KEYS = {"path", "train", "val", "test", "names", "nc", "channels"}
_CALIBRATION_IMAGE_FORMATS = {
    ".bmp": frozenset({"BMP"}),
    ".jpeg": frozenset({"JPEG"}),
    ".jpg": frozenset({"JPEG"}),
    ".png": frozenset({"PNG"}),
    ".tif": frozenset({"TIFF"}),
    ".tiff": frozenset({"TIFF"}),
    ".webp": frozenset({"WEBP"}),
}
_CALIBRATION_IMAGE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})
_ULTRALYTICS_IMAGE_SUFFIXES = frozenset(
    {
        ".avif",
        ".bmp",
        ".dng",
        ".heic",
        ".heif",
        ".jp2",
        ".jpeg",
        ".jpg",
        ".mpo",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
_MAX_TRAINING_IMAGE_BYTES = 128 * 1024 * 1024
_MAX_TRAINING_IMAGE_DIMENSION = 32_768
_MAX_TRAINING_IMAGE_PIXELS = 64 * 1024 * 1024
_MAX_TRAINING_LABEL_BYTES = 16 * 1024 * 1024
_MAX_TRAINING_LABEL_ROWS = 100_000
_MAX_TRAINING_LABEL_TOKEN_BYTES = 64
# Common YOLO writers round normalized coordinates to six decimal places.
# Independent centre/size rounding can move an exactly edge-touching box by up
# to 0.75e-6, so admit that representation noise while rejecting any material
# out-of-frame annotation.
_TRAINING_LABEL_EDGE_TOLERANCE = 1e-6
_MIN_CALIBRATION_IMAGES = 1_000
_BACKEND_CALIBRATION_IMAGES = 512
_MAX_CALIBRATION_IMAGES = 4_096
_MAX_CALIBRATION_IMAGE_BYTES = 128 * 1024 * 1024
_MAX_CALIBRATION_ENCODED_BYTES = 16 * 1024 * 1024 * 1024
_MAX_CALIBRATION_IMAGE_PIXELS = 32 * 1024 * 1024
_MAX_CALIBRATION_DECODED_PIXELS = 16_000_000_000
_MAX_CALIBRATION_TENSOR_BYTES = 64 * 1024 * 1024 * 1024
_MAX_MODELOPT_CALIBRATION_WORK_BYTES = 8 * 1024 * 1024 * 1024
_MAX_DATASET_BYTES = 64 * 1024 * 1024 * 1024
_MAX_DATASET_ENTRIES = 100_000
_MAX_RFDETR_ANNOTATION_BYTES = 128 * 1024 * 1024
_MAX_RFDETR_JSON_NODES = 2_000_000
_MAX_RFDETR_JSON_DEPTH = 64
_MAX_RFDETR_IMAGES = 1_000_000
_MAX_RFDETR_ANNOTATIONS = 2_000_000
_MAX_RFDETR_IMAGE_DIMENSION = _MAX_TRAINING_IMAGE_DIMENSION
_MAX_RFDETR_IMAGE_PIXELS = _MAX_TRAINING_IMAGE_PIXELS
_MAX_RFDETR_PATH_BYTES = 4096
_RFDETR_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_CALIBRATION_POLICY = (
    b"ultralytics==8.4.92;task=detect;source=validated-val;"
    b"private-splits=train,val,test=same-curated-set;channels=3;"
    b"batch=1;fraction=1.0;rect=false;dynamic=false;augment=false;letterbox=true;"
    b"workers=0;shuffle=true;drop-last=true;uint8-div255;"
    b"backend-count=512;selection=lowest-tensor-sha256;exif-orientation=identity;"
    b"network-url-probes=disabled;dataset-cache-io=disabled;format-bgr0=deterministic;"
    b"optional-codec-hook=disabled;modelopt-peak-factor=10;modelopt-work-limit=8GiB"
)

_Cleanup = tuple[str, Callable[[], None]]


def _release_resources(
    cleanups: Sequence[_Cleanup],
    *,
    primary: BaseException | None = None,
) -> None:
    """Attempt each cleanup once without replacing an earlier failure."""
    failure = primary
    for label, cleanup in cleanups:
        try:
            cleanup()
        except BaseException as cleanup_error:
            if failure is None:
                failure = cleanup_error
            else:
                attach_cleanup_failure(failure, cleanup_error, label)
    if primary is None and failure is not None:
        raise failure.with_traceback(failure.__traceback__)


@dataclass(frozen=True)
class _CalibrationImage:
    relative_path: pathlib.Path
    file_sha256: str
    tensor_sha256: str
    width: int
    height: int
    image_format: str


@dataclass(frozen=True)
class _SplitInventory:
    regular_files: tuple[tuple[pathlib.Path, tuple[int, int]], ...]
    entry_count: int
    byte_count: int


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


def _validated_calibration_image_size(value: int) -> int:
    if type(value) is not int or not 32 <= value <= 4096:
        raise ValueError("calibration image_size must be an integer in [32, 4096]")
    return value


def _calibration_runtime_policy(
    tensorrt_version: str | None,
    modelopt_version: str | None,
) -> bytes:
    if tensorrt_version is None:
        if modelopt_version is not None:
            raise ValueError("ModelOpt version requires a TensorRT runtime")
        return b"tensorrt=unbound-standalone-validation"
    if (
        not isinstance(tensorrt_version, str)
        or not tensorrt_version
        or not tensorrt_version.isprintable()
        or len(tensorrt_version.encode("utf-8")) > 128
    ):
        raise ValueError("TensorRT version must be a bounded printable string")
    try:
        major = int(tensorrt_version.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError("TensorRT version must start with an integer major version") from exc
    if major < 7:
        raise ValueError("TensorRT INT8 export requires TensorRT 7 or newer")
    if major > 11:
        raise ValueError(
            "TensorRT versions newer than the audited 7-11 exporter branches are rejected"
        )
    release = tensorrt_version.split("+", 1)[0].split("-", 1)[0].split(".")
    if len(release) > 1:
        try:
            minor = int(release[1])
        except ValueError as exc:
            raise ValueError("TensorRT version minor component must be an integer") from exc
        if major == 10 and minor == 2:
            raise ValueError("TensorRT 10.2 is rejected by the pinned Ultralytics exporter")
    route = "modelopt-max" if major >= 11 else "legacy-minmax"
    if major >= 11:
        if (
            not isinstance(modelopt_version, str)
            or not modelopt_version
            or not modelopt_version.isprintable()
            or len(modelopt_version.encode("utf-8")) > 128
        ):
            raise ValueError("TensorRT 11 calibration requires a bounded ModelOpt version")
        try:
            from packaging.version import InvalidVersion, Version
        except ImportError as exc:  # pragma: no cover - locked export dependency
            raise RuntimeError("TensorRT 11 calibration requires the packaging runtime") from exc
        try:
            parsed_modelopt_version = Version(modelopt_version)
        except InvalidVersion as exc:
            raise ValueError("ModelOpt must expose a valid PEP 440 version") from exc
        if parsed_modelopt_version < Version("0.44"):
            raise ValueError("TensorRT 11 calibration requires nvidia-modelopt>=0.44")
    elif modelopt_version is not None:
        raise ValueError("ModelOpt version is only valid for the TensorRT 11 calibration route")
    return (
        f"tensorrt={tensorrt_version};route={route};modelopt={modelopt_version or 'none'}"
    ).encode()


def _update_digest_field(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _regular_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _split_entry_identity(metadata: os.stat_result, *, subject: str) -> tuple[int, int]:
    """Require the stable POSIX identity used to prove split disjointness."""
    device = getattr(metadata, "st_dev", None)
    inode = getattr(metadata, "st_ino", None)
    if type(device) is not int or type(inode) is not int or device < 0 or inode <= 0:
        raise ValueError(f"{subject} filesystem does not expose a stable device/inode identity")
    return device, inode


def _read_regular_utf8_at(
    parent_fd: int,
    name: str,
    *,
    limit: int,
    subject: str,
) -> tuple[str, tuple[int, int, int, int, int], str]:
    """Read one stable regular file relative to an authenticated directory."""
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise ValueError(f"{subject} name must be one nonempty path component")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{subject} must be a regular file")
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        expected_identity = _regular_file_identity(metadata)
        if _regular_file_identity(opened) != expected_identity:
            raise ValueError(f"{subject} was replaced while it was being opened")
        owned_fd = fd
        fd = -1
        with owned_binary_reader(owned_fd) as handle:
            if opened.st_size == 0 or opened.st_size > limit:
                raise ValueError(f"{subject} must contain 1..{limit} bytes")
            value = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
        if len(value) != opened.st_size or _regular_file_identity(after) != expected_identity:
            raise ValueError(f"{subject} changed while it was being read")
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _regular_file_identity(current) != expected_identity:
            raise ValueError(f"{subject} was replaced while it was being read")
    except BaseException as primary:
        if fd >= 0:
            cleanup_fd = fd
            fd = -1
            _release_resources(
                ((f"{subject} descriptor cleanup also failed", lambda: os.close(cleanup_fd)),),
                primary=primary,
            )
        raise
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{subject} must be valid UTF-8") from exc
    return text, expected_identity, hashlib.sha256(value).hexdigest()


class _BoundSourceManifest:
    """Manifest bytes plus a retained descriptor for their original parent."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._parent_fd = open_directory_nofollow(path.parent, "dataset manifest parent")
        self._closed = False
        try:
            self._parent_identity = _directory_identity(os.fstat(self._parent_fd))
            self.text, self._identity, self.sha256 = _read_regular_utf8_at(
                self._parent_fd,
                path.name,
                limit=_MAX_MANIFEST_BYTES,
                subject="dataset manifest",
            )
            self.assert_unchanged()
        except BaseException as primary:
            _release_resources(
                (("source dataset manifest binding cleanup also failed", self.close),),
                primary=primary,
            )
            raise

    @property
    def parent_fd(self) -> int:
        if self._closed:
            raise RuntimeError("source dataset manifest binding is closed")
        return self._parent_fd

    def clone(self) -> _BoundSourceManifest:
        if self._closed:
            raise RuntimeError("source dataset manifest binding is closed")
        clone = self.__class__.__new__(self.__class__)
        clone.path = self.path
        clone._parent_fd = os.dup(self._parent_fd)
        clone._parent_identity = self._parent_identity
        clone.text = self.text
        clone._identity = self._identity
        clone.sha256 = self.sha256
        clone._closed = False
        return clone

    def assert_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("source dataset manifest binding is closed")
        if _directory_identity(os.fstat(self._parent_fd)) != self._parent_identity:
            raise ValueError("source dataset manifest parent descriptor changed")
        self._assert_visible_binding()
        text, identity, digest = _read_regular_utf8_at(
            self._parent_fd,
            self.path.name,
            limit=_MAX_MANIFEST_BYTES,
            subject="source dataset manifest",
        )
        if identity != self._identity:
            raise ValueError("source dataset manifest identity changed")
        if digest != self.sha256 or text != self.text:
            raise ValueError("source dataset manifest content changed")
        self._assert_visible_binding()

    def _assert_visible_binding(self) -> None:
        visible_parent_fd = open_directory_nofollow(
            self.path.parent, "source dataset manifest parent"
        )
        try:
            if _directory_identity(os.fstat(visible_parent_fd)) != self._parent_identity:
                raise ValueError("source dataset manifest parent path was replaced")
            visible_manifest = os.stat(
                self.path.name,
                dir_fd=visible_parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(visible_manifest.st_mode)
                or _regular_file_identity(visible_manifest) != self._identity
            ):
                raise ValueError("source dataset manifest path was replaced")
        except BaseException as primary:
            _release_resources(
                (
                    (
                        "visible source dataset manifest parent cleanup also failed",
                        lambda: os.close(visible_parent_fd),
                    ),
                ),
                primary=primary,
            )
            raise
        _release_resources(
            (
                (
                    "visible source dataset manifest parent cleanup failed",
                    lambda: os.close(visible_parent_fd),
                ),
            )
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self._parent_fd)


def _open_relative_directory_nofollow(
    parent_fd: int, relative: pathlib.PurePath, subject: str
) -> int:
    """Open a child directory from an authenticated base without accepting escape components."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{subject} relative path must not escape its authenticated parent")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.dup(parent_fd)
    try:
        for component in relative.parts:
            if component in {"", "."}:
                continue
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"{subject} path does not exist") from exc
            except OSError as exc:
                raise ValueError(
                    f"{subject} path chain contains a symbolic link or special component"
                ) from exc
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
        result = current_fd
        current_fd = -1
        return result
    except BaseException as primary:
        if current_fd >= 0:
            cleanup_fd = current_fd
            current_fd = -1
            _release_resources(
                (
                    (
                        f"{subject} directory descriptor cleanup also failed",
                        lambda: os.close(cleanup_fd),
                    ),
                ),
                primary=primary,
            )
        raise


class _BoundSourceDirectory:
    """Caller-visible directory pinned to one retained root descriptor."""

    def __init__(self, path: pathlib.Path, directory_fd: int) -> None:
        self.path = path
        self._fd = directory_fd
        self._closed = False
        try:
            metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("dataset root must be a directory")
            self._identity = _directory_identity(metadata)
            self.assert_path_unchanged()
        except BaseException as primary:
            _release_resources(
                (("source dataset root binding cleanup also failed", self.close),),
                primary=primary,
            )
            raise

    @classmethod
    def from_manifest_relative(
        cls,
        manifest: _BoundSourceManifest,
        value: pathlib.PurePath,
    ) -> _BoundSourceDirectory:
        directory_fd = _open_relative_directory_nofollow(
            manifest.parent_fd,
            value,
            "dataset root",
        )
        path = pathlib.Path(os.path.abspath(manifest.path.parent / value))
        return cls(path, directory_fd)

    @classmethod
    def from_absolute(cls, path: pathlib.Path) -> _BoundSourceDirectory:
        return cls(path, open_directory_nofollow(path, "dataset root"))

    @property
    def fd(self) -> int:
        if self._closed:
            raise RuntimeError("source dataset root binding is closed")
        return self._fd

    def clone(self) -> _BoundSourceDirectory:
        if self._closed:
            raise RuntimeError("source dataset root binding is closed")
        return self.__class__(self.path, os.dup(self._fd))

    def assert_path_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("source dataset root binding is closed")
        if _directory_identity(os.fstat(self._fd)) != self._identity:
            raise ValueError("source dataset root descriptor changed")
        visible_fd = open_directory_nofollow(self.path, "source dataset root")
        try:
            if _directory_identity(os.fstat(visible_fd)) != self._identity:
                raise ValueError("source dataset root path was replaced")
        except BaseException as primary:
            _release_resources(
                (
                    (
                        "visible source dataset root cleanup also failed",
                        lambda: os.close(visible_fd),
                    ),
                ),
                primary=primary,
            )
            raise
        _release_resources(
            (("visible source dataset root cleanup failed", lambda: os.close(visible_fd)),)
        )

    def digest(self) -> str:
        if self._closed:
            raise RuntimeError("source dataset root binding is closed")
        from .artifacts import sha256_directory_fd

        try:
            return sha256_directory_fd(
                self._fd,
                display=str(self.path),
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
            )
        except ValueError as exc:
            if f"{_MAX_DATASET_BYTES}-byte safety limit" in str(exc):
                raise ValueError(
                    f"dataset exceeds the {_MAX_DATASET_BYTES}-byte safety limit"
                ) from exc
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self._fd)


def _stat_relative_entry(
    root_fd: int,
    relative: pathlib.PurePath,
    *,
    subject: str,
    require_directory: bool | None,
) -> os.stat_result:
    """Validate one root-relative split path with descriptor-relative traversal."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{subject} must remain inside the declared dataset root")
    components = tuple(part for part in relative.parts if part not in {"", "."})
    if not components:
        metadata = os.fstat(root_fd)
    else:
        parent = pathlib.PurePath(*components[:-1])
        parent_fd = _open_relative_directory_nofollow(root_fd, parent, subject)
        try:
            metadata = os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
        except BaseException as primary:
            _release_resources(
                (
                    (
                        f"{subject} parent descriptor cleanup also failed",
                        lambda: os.close(parent_fd),
                    ),
                ),
                primary=primary,
            )
            raise
        _release_resources(
            ((f"{subject} parent descriptor cleanup failed", lambda: os.close(parent_fd)),)
        )
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{subject} must not contain symbolic links")
    if require_directory is True and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{subject} must be a directory")
    if require_directory is False and not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{subject} must be a regular file")
    if require_directory is None and not (
        stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
    ):
        raise ValueError(f"{subject} must be a regular file or directory")
    return metadata


def _inventory_selected_split_path(
    root_fd: int,
    candidate: pathlib.Path,
    relative: pathlib.PurePath,
    metadata: os.stat_result,
    *,
    field: str,
    max_entries: int,
    max_bytes: int,
) -> _SplitInventory:
    """Inventory one selected path through authenticated, bounded descriptors."""
    from .artifacts import _descriptor_tree_entries, _open_descriptor_entry

    subject = f"dataset {field} split {candidate}"
    if max_entries < 1:
        raise ValueError(
            f"selected dataset splits exceed the {_MAX_DATASET_ENTRIES}-entry safety limit"
        )
    expected_identity = _regular_file_identity(metadata)
    if stat.S_ISREG(metadata.st_mode):
        if metadata.st_size > max_bytes:
            raise ValueError(
                f"selected dataset splits exceed the {_MAX_DATASET_BYTES}-byte safety limit"
            )
        try:
            entry_fd, observed = _open_descriptor_entry(
                root_fd,
                relative.as_posix(),
                expect_directory=False,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"{subject} could not be bound safely: {exc}") from exc
        try:
            if _regular_file_identity(observed) != expected_identity:
                raise ValueError(f"{subject} changed while it was being bound")
            file_identity = _split_entry_identity(observed, subject=subject)
        except BaseException as primary:
            _release_resources(
                ((f"{subject} descriptor cleanup also failed", lambda: os.close(entry_fd)),),
                primary=primary,
            )
            raise
        _release_resources(((f"{subject} descriptor cleanup failed", lambda: os.close(entry_fd)),))
        current = _stat_relative_entry(
            root_fd,
            relative,
            subject=subject,
            require_directory=False,
        )
        if _regular_file_identity(current) != expected_identity:
            raise ValueError(f"{subject} changed while it was being inventoried")
        return _SplitInventory(((candidate, file_identity),), 1, metadata.st_size)

    directory_fd = _open_relative_directory_nofollow(root_fd, relative, subject)
    try:
        opened = os.fstat(directory_fd)
        if _regular_file_identity(opened) != expected_identity:
            raise ValueError(f"{subject} changed while it was being bound")
        descendant_budget = max_entries - 1
        inventory_limit = max(descendant_budget, 1)
        try:
            entries = _descriptor_tree_entries(
                directory_fd,
                display=subject,
                max_entries=inventory_limit,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"{subject} inventory failed: {exc}") from exc
        if len(entries) > descendant_budget:
            raise ValueError(
                f"selected dataset splits exceed the {_MAX_DATASET_ENTRIES}-entry safety limit"
            )

        regular_files: list[tuple[pathlib.Path, tuple[int, int]]] = []
        byte_count = 0
        for entry in entries:
            try:
                entry_fd, observed = _open_descriptor_entry(
                    directory_fd,
                    entry.relative,
                    expect_directory=entry.is_directory,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"{subject} entry changed during bounded inventory: {entry.relative}"
                ) from exc
            try:
                if _regular_file_identity(observed) != entry.identity:
                    raise ValueError(
                        f"{subject} entry changed during bounded inventory: {entry.relative}"
                    )
                if not entry.is_directory:
                    byte_count += observed.st_size
                    if byte_count > max_bytes:
                        raise ValueError(
                            "selected dataset splits exceed the "
                            f"{_MAX_DATASET_BYTES}-byte safety limit"
                        )
                    entry_path = candidate / pathlib.PurePosixPath(entry.relative)
                    regular_files.append(
                        (
                            entry_path,
                            _split_entry_identity(observed, subject=str(entry_path)),
                        )
                    )
            except BaseException as primary:
                _release_resources(
                    (
                        (
                            f"{subject} entry descriptor cleanup also failed",
                            partial(os.close, entry_fd),
                        ),
                    ),
                    primary=primary,
                )
                raise
            _release_resources(
                (
                    (
                        f"{subject} entry descriptor cleanup failed",
                        partial(os.close, entry_fd),
                    ),
                )
            )

        try:
            after_entries = _descriptor_tree_entries(
                directory_fd,
                display=subject,
                max_entries=inventory_limit,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"{subject} changed during bounded inventory: {exc}") from exc
        if entries != after_entries:
            raise ValueError(f"{subject} changed during bounded inventory")
        after = os.fstat(directory_fd)
        if _regular_file_identity(after) != expected_identity:
            raise ValueError(f"{subject} changed during bounded inventory")
        current = _stat_relative_entry(
            root_fd,
            relative,
            subject=subject,
            require_directory=True,
        )
        if _regular_file_identity(current) != expected_identity:
            raise ValueError(f"{subject} changed during bounded inventory")
    except BaseException as primary:
        _release_resources(
            (
                (
                    f"{subject} directory descriptor cleanup also failed",
                    lambda: os.close(directory_fd),
                ),
            ),
            primary=primary,
        )
        raise
    _release_resources(
        ((f"{subject} directory descriptor cleanup failed", lambda: os.close(directory_fd)),)
    )
    return _SplitInventory(tuple(regular_files), len(entries) + 1, byte_count)


def _inventory_selected_split_set(
    root_fd: int,
    split_roots: Mapping[
        str,
        Sequence[tuple[pathlib.Path, pathlib.PurePath, os.stat_result]],
    ],
    *,
    reverse: bool,
) -> dict[str, tuple[tuple[pathlib.Path, tuple[int, int]], ...]]:
    """Inventory the complete selected split set within one aggregate budget."""
    remaining_entries = _MAX_DATASET_ENTRIES
    remaining_bytes = _MAX_DATASET_BYTES
    split_files: dict[str, tuple[tuple[pathlib.Path, tuple[int, int]], ...]] = {}
    fields = list(split_roots.items())
    if reverse:
        fields.reverse()
    for field, roots in fields:
        ordered_roots = list(roots)
        if reverse:
            ordered_roots.reverse()
        field_files: list[tuple[pathlib.Path, tuple[int, int]]] = []
        for candidate, root_relative, metadata in ordered_roots:
            inventory = _inventory_selected_split_path(
                root_fd,
                candidate,
                root_relative,
                metadata,
                field=field,
                max_entries=remaining_entries,
                max_bytes=remaining_bytes,
            )
            remaining_entries -= inventory.entry_count
            remaining_bytes -= inventory.byte_count
            field_files.extend(inventory.regular_files)
        # Canonical ordering makes the forward/reverse set observations directly
        # comparable without making declaration order part of the invariant.
        split_files[field] = tuple(sorted(field_files, key=lambda item: str(item[0])))
    return split_files


def _validate_split_file_identities(
    split_files: Mapping[str, Sequence[tuple[pathlib.Path, tuple[int, int]]]],
) -> None:
    """Reject duplicate regular-file identities within or across selected splits."""
    split_origins: dict[str, dict[tuple[int, int], pathlib.Path]] = {}
    for field, inventoried_files in split_files.items():
        origins: dict[tuple[int, int], pathlib.Path] = {}
        for split_path, identity in inventoried_files:
            previous = origins.get(identity)
            if previous is not None:
                raise ValueError(
                    f"dataset {field} paths identify the same filesystem entry and would "
                    f"select files more than once: {previous} and {split_path}"
                )
            origins[identity] = split_path
        split_origins[field] = origins

    for (left_name, left_origins), (right_name, right_origins) in combinations(
        split_origins.items(), 2
    ):
        for identity, left in left_origins.items():
            right_origin = right_origins.get(identity)
            if right_origin is not None:
                raise ValueError(
                    f"dataset {left_name} and {right_name} paths identify the same "
                    f"filesystem entry: {left} and {right_origin}"
                )


def _validate_snapshotted_split_identities(
    root: pathlib.Path,
    payload: Mapping[str, object],
    entries: Sequence[_DescriptorTreeEntry],
    *,
    require_nonempty: frozenset[str] = frozenset(),
) -> None:
    """Validate split identities on the exact source inventory being copied.

    ``ArtifactSnapshot`` invokes this callback on the immutable descriptor
    inventory that drives its copy, then proves the source inventory did not
    change before returning. This makes split disjointness a property of the
    consumed snapshot rather than an earlier observation of a mutable tree.
    Ultralytics derives label paths outside commonly selected ``images/...``
    directories, so those backend-consumed files participate in the same
    identity boundary even when the manifest does not name them directly.
    """

    by_path: dict[str, _DescriptorTreeEntry] = {}
    for entry in entries:
        if entry.relative in by_path:
            raise RuntimeError("dataset snapshot source-tree inventory contains duplicate paths")
        by_path[entry.relative] = entry

    split_files: dict[str, tuple[tuple[pathlib.Path, tuple[int, int]], ...]] = {}
    for field in ("train", "val", "test"):
        if field not in payload:
            continue
        selected: dict[str, tuple[pathlib.Path, tuple[int, int]]] = {}
        for value in _split_values(payload[field], field):
            try:
                relative_path = pathlib.Path(value).relative_to(root)
            except ValueError as exc:  # defensive: admission already normalized this
                raise RuntimeError("dataset snapshot split escaped its admitted root") from exc
            relative = relative_path.as_posix()
            if relative == ".":
                candidates = tuple(entry for entry in entries if not entry.is_directory)
            else:
                selected_root = by_path.get(relative)
                if selected_root is None:
                    raise ValueError(f"dataset {field} split changed before snapshotting: {value}")
                if not selected_root.is_directory:
                    raise ValueError(
                        f"dataset {field} split must remain a directory while snapshotting: {value}"
                    )
                prefix = f"{relative}/"
                candidates = tuple(
                    entry
                    for entry in entries
                    if not entry.is_directory and entry.relative.startswith(prefix)
                )
            if field in require_nonempty and not candidates:
                raise ValueError(
                    f"dataset {field} split must contain at least one regular file: {value}"
                )
            consumed_entries = list(candidates)
            for entry in candidates:
                if pathlib.PurePosixPath(entry.relative).suffix.lower() not in (
                    _ULTRALYTICS_IMAGE_SUFFIXES
                ):
                    continue
                image_path = root / pathlib.PurePosixPath(entry.relative)
                label_path = _ultralytics_label_path(root, image_path)
                label_relative = label_path.relative_to(root).as_posix()
                label_entry = by_path.get(label_relative)
                if label_entry is not None and not label_entry.is_directory:
                    consumed_entries.append(label_entry)
            for entry in consumed_entries:
                identity = entry.identity
                device, inode = identity[:2]
                entry_path = root / pathlib.PurePosixPath(entry.relative)
                if type(device) is not int or type(inode) is not int or device < 0 or inode <= 0:
                    raise ValueError(
                        f"{entry_path} filesystem does not expose a stable device/inode identity"
                    )
                selected.setdefault(entry.relative, (entry_path, (device, inode)))
        split_files[field] = tuple(sorted(selected.values(), key=lambda item: str(item[0])))
    _validate_split_file_identities(split_files)


def _pillow_open_function(image_module):
    """Return pristine Pillow opening, rejecting unknown process-wide hooks."""
    opener = image_module.open
    if getattr(opener, "__module__", None) == "PIL.Image":
        return opener
    if getattr(opener, "__module__", None) == "ultralytics.utils.patches":
        patches = sys.modules.get("ultralytics.utils.patches")
        original = getattr(patches, "_image_open", None)
        if callable(original) and getattr(original, "__module__", None) == "PIL.Image":
            return original
    raise RuntimeError("Pillow Image.open was replaced by an untrusted runtime hook")


def _validate_training_image_header(
    path: pathlib.Path,
    relative_path: pathlib.Path,
    expected_metadata: os.stat_result,
    *,
    expected_dimensions: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Authenticate one backend-visible still image and bound its decoded shape."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional training boundary
        raise RuntimeError("training dataset validation requires the vision runtime") from exc

    expected_formats = _CALIBRATION_IMAGE_FORMATS.get(path.suffix.lower())
    if expected_formats is None:
        raise ValueError(f"unsupported training image format: {relative_path}")
    image_open = _pillow_open_function(Image)
    try:
        with open_regular_nofollow(path.absolute(), "training image") as handle:
            before = os.fstat(handle.fileno())
            if _regular_file_identity(before) != _regular_file_identity(expected_metadata):
                raise ValueError(f"training image changed after inventory: {relative_path}")
            if before.st_size == 0 or before.st_size > _MAX_TRAINING_IMAGE_BYTES:
                raise ValueError(
                    "training image must contain 1.."
                    f"{_MAX_TRAINING_IMAGE_BYTES} bytes: {relative_path}"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with image_open(handle) as probe:
                    width, height = probe.size
                    image_format = probe.format
                    image_mode = probe.mode
                    frame_count = getattr(probe, "n_frames", 1)
                    exif_orientation = probe.getexif().get(274, 1)
                handle.seek(0)
                with image_open(handle) as verifier:
                    verifier.verify()
                if image_format == "JPEG":
                    handle.seek(-2, os.SEEK_END)
                    if handle.read(2) != b"\xff\xd9":
                        raise ValueError(
                            f"training JPEG is truncated or missing its EOI marker: {relative_path}"
                        )
            after = os.fstat(handle.fileno())
    except (OSError, RuntimeError, SyntaxError, ValueError, Image.DecompressionBombWarning) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("training image"):
            raise
        raise ValueError(
            f"training image is not a valid bounded still image: {relative_path}"
        ) from exc

    if _regular_file_identity(after) != _regular_file_identity(before):
        raise ValueError(f"training image changed while it was inspected: {relative_path}")
    if image_format not in expected_formats:
        raise ValueError(
            f"training image suffix/content mismatch for {relative_path}: {image_format!r}"
        )
    if image_mode not in _CALIBRATION_IMAGE_MODES:
        raise ValueError(f"unsupported training image mode {image_mode!r}: {relative_path}")
    if frame_count != 1:
        raise ValueError(f"training images must contain exactly one frame: {relative_path}")
    if exif_orientation != 1:
        raise ValueError(f"training images must use identity EXIF orientation: {relative_path}")
    if (
        width <= 0
        or height <= 0
        or width > _MAX_TRAINING_IMAGE_DIMENSION
        or height > _MAX_TRAINING_IMAGE_DIMENSION
        or width * height > _MAX_TRAINING_IMAGE_PIXELS
    ):
        raise ValueError(f"training image dimensions exceed the supported bounds: {relative_path}")
    if expected_dimensions is not None and (width, height) != expected_dimensions:
        raise ValueError(
            f"training image dimensions do not match annotation metadata: {relative_path}"
        )
    return width, height


def _ultralytics_label_path(root: pathlib.Path, image: pathlib.Path) -> pathlib.Path:
    """Mirror the pinned loader's last ``/images/`` to ``/labels/`` mapping."""
    image_text = os.fspath(image)
    image_marker = f"{os.sep}images{os.sep}"
    label_marker = f"{os.sep}labels{os.sep}"
    label = pathlib.Path(label_marker.join(image_text.rsplit(image_marker, 1))).with_suffix(".txt")
    try:
        label.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "training image would make the pinned backend select a label outside "
            f"the private dataset: {image.relative_to(root)}"
        ) from exc
    return label


def _validate_yolo_detection_label(
    path: pathlib.Path,
    relative_path: pathlib.Path,
    expected_metadata: os.stat_result,
    class_count: int,
) -> None:
    """Validate one stable, bounded five-column YOLO detection label."""
    try:
        with open_regular_nofollow(path.absolute(), "training label") as handle:
            before = os.fstat(handle.fileno())
            if _regular_file_identity(before) != _regular_file_identity(expected_metadata):
                raise ValueError(f"training label changed after inventory: {relative_path}")
            if before.st_size > _MAX_TRAINING_LABEL_BYTES:
                raise ValueError(
                    "training label exceeds the "
                    f"{_MAX_TRAINING_LABEL_BYTES}-byte safety limit: {relative_path}"
                )
            encoded = handle.read(_MAX_TRAINING_LABEL_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValueError(f"training label could not be read safely: {relative_path}") from exc

    if (
        _regular_file_identity(after) != _regular_file_identity(before)
        or len(encoded) != before.st_size
    ):
        raise ValueError(f"training label changed while it was inspected: {relative_path}")
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"training label must be valid UTF-8: {relative_path}") from exc
    if "\0" in text:
        raise ValueError(f"training label must not contain NUL bytes: {relative_path}")

    lines = text.splitlines()
    if len(lines) > _MAX_TRAINING_LABEL_ROWS:
        raise ValueError(
            "training label exceeds the "
            f"{_MAX_TRAINING_LABEL_ROWS}-row safety limit: {relative_path}"
        )
    rows: set[tuple[float, float, float, float, float]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(
                "training labels must use five-column YOLO detection rows "
                f"(class x_center y_center width height): {relative_path}:{line_number}"
            )
        if any(
            not token.isascii() or len(token.encode("ascii")) > _MAX_TRAINING_LABEL_TOKEN_BYTES
            for token in tokens
        ):
            raise ValueError(
                f"training label contains an invalid numeric token: {relative_path}:{line_number}"
            )
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"training label contains an invalid number: {relative_path}:{line_number}"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"training label numbers must be finite: {relative_path}:{line_number}"
            )
        class_id, x_center, y_center, width, height = values
        row = (class_id, x_center, y_center, width, height)
        if class_id != math.floor(class_id) or not 0 <= class_id < class_count:
            raise ValueError(
                f"training label class is outside [0, {class_count}): {relative_path}:{line_number}"
            )
        if (
            not 0.0 <= x_center <= 1.0
            or not 0.0 <= y_center <= 1.0
            or not 0.0 < width <= 1.0
            or not 0.0 < height <= 1.0
        ):
            raise ValueError(
                "training label boxes must use finite normalized centers and positive sizes: "
                f"{relative_path}:{line_number}"
            )
        if (
            x_center - width / 2.0 < -_TRAINING_LABEL_EDGE_TOLERANCE
            or x_center + width / 2.0 > 1.0 + _TRAINING_LABEL_EDGE_TOLERANCE
            or y_center - height / 2.0 < -_TRAINING_LABEL_EDGE_TOLERANCE
            or y_center + height / 2.0 > 1.0 + _TRAINING_LABEL_EDGE_TOLERANCE
        ):
            raise ValueError(
                "training label boxes must remain inside the normalized image bounds: "
                f"{relative_path}:{line_number}"
            )
        if row in rows:
            raise ValueError(
                f"training label contains a duplicate row: {relative_path}:{line_number}"
            )
        rows.add(row)


def _validate_ultralytics_training_snapshot(
    root: pathlib.Path,
    payload: Mapping[str, object],
) -> None:
    """Validate every image the pinned Ultralytics loader can discover."""
    from .artifacts import _tree_entries

    entries = _tree_entries(root, _MAX_DATASET_ENTRIES)
    entry_kinds = {entry: kind for entry, kind, _metadata in entries}
    entry_metadata = {entry: metadata for entry, _kind, metadata in entries}
    seen: set[pathlib.Path] = set()
    seen_labels: dict[pathlib.Path, pathlib.Path] = {}
    class_count = payload.get("nc")
    if type(class_count) is not int or not 1 <= class_count <= _MAX_CLASSES:
        raise RuntimeError("private dataset class count was corrupted")
    for field in ("train", "val", "test"):
        if field not in payload:
            continue
        candidates: list[pathlib.Path] = []
        for value in _split_values(payload[field], field):
            split_path = pathlib.Path(value)
            kind = "directory" if split_path == root else entry_kinds.get(split_path)
            if kind != "directory":
                raise RuntimeError(f"private dataset {field} split disappeared: {split_path}")
            for entry, entry_kind, _metadata in entries:
                if entry_kind != "file" or not entry.is_relative_to(split_path):
                    continue
                suffix = entry.suffix.lower()
                if suffix not in _ULTRALYTICS_IMAGE_SUFFIXES:
                    continue
                if suffix not in _CALIBRATION_IMAGE_FORMATS:
                    raise ValueError(
                        "training images use a backend-recognized format outside Manwe's "
                        f"validated still-image boundary: {entry.relative_to(root)}"
                    )
                candidates.append(entry)
        candidates.sort(key=lambda value: value.relative_to(root).as_posix())
        if field in {"train", "val"} and not candidates:
            raise ValueError(f"dataset {field} split contains no supported training images")
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if candidate in seen:
                raise ValueError(f"training split selects an image more than once: {relative}")
            seen.add(candidate)
            _validate_training_image_header(
                candidate,
                relative,
                entry_metadata[candidate],
            )
            label = _ultralytics_label_path(root, candidate)
            previous_image = seen_labels.get(label)
            if previous_image is not None:
                raise ValueError(
                    "training images resolve to the same backend label path: "
                    f"{previous_image.relative_to(root)} and {relative}"
                )
            seen_labels[label] = candidate
            label_kind = entry_kinds.get(label)
            if label_kind is None:
                continue
            if label_kind != "file":
                raise ValueError(
                    f"training label must be a regular file: {label.relative_to(root)}"
                )
            _validate_yolo_detection_label(
                label,
                label.relative_to(root),
                entry_metadata[label],
                class_count,
            )


def _validate_calibration_image(
    path: pathlib.Path,
    relative_path: pathlib.Path,
    image_size: int,
    remaining_decoded_pixels: int,
    expected_metadata: os.stat_result,
) -> _CalibrationImage:
    """Decode one stable image exactly enough to prove loader usability."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional-dependency boundary
        raise RuntimeError("INT8 calibration validation requires the export/vision extra") from exc

    suffix = path.suffix.lower()
    expected_formats = _CALIBRATION_IMAGE_FORMATS.get(suffix)
    if expected_formats is None:
        raise ValueError(f"unsupported INT8 calibration image suffix: {relative_path}")
    image_open = _pillow_open_function(Image)

    try:
        with open_regular_nofollow(path.absolute(), "INT8 calibration image") as handle:
            before = os.fstat(handle.fileno())
            expected_identity = (
                expected_metadata.st_dev,
                expected_metadata.st_ino,
                expected_metadata.st_size,
                expected_metadata.st_mtime_ns,
                expected_metadata.st_ctime_ns,
            )
            opened_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if opened_identity != expected_identity:
                raise ValueError(f"INT8 calibration image changed after inventory: {relative_path}")
            if before.st_size == 0:
                raise ValueError(f"INT8 calibration image is empty: {relative_path}")
            if before.st_size > _MAX_CALIBRATION_IMAGE_BYTES:
                raise ValueError(
                    "INT8 calibration image exceeds the "
                    f"{_MAX_CALIBRATION_IMAGE_BYTES}-byte safety limit: {relative_path}"
                )

            encoded = handle.read(_MAX_CALIBRATION_IMAGE_BYTES + 1)
            if len(encoded) != before.st_size:
                raise ValueError(f"INT8 calibration image changed while reading: {relative_path}")
            file_hasher = hashlib.sha256(encoded)

            handle.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with image_open(handle) as probe:
                    width, height = probe.size
                    image_format = probe.format
                    image_mode = probe.mode
                    frame_count = getattr(probe, "n_frames", 1)
                    exif_orientation = probe.getexif().get(274, 1)
                handle.seek(0)
                with image_open(handle) as verifier:
                    verifier.verify()

                if image_format not in expected_formats:
                    raise ValueError(
                        "INT8 calibration image suffix/content mismatch "
                        f"for {relative_path}: detected {image_format!r}"
                    )
                if image_format == "JPEG":
                    handle.seek(-2, os.SEEK_END)
                    if handle.read(2) != b"\xff\xd9":
                        raise ValueError(
                            "INT8 calibration JPEG is truncated or missing its EOI marker: "
                            f"{relative_path}"
                        )
                if image_mode not in _CALIBRATION_IMAGE_MODES:
                    raise ValueError(
                        f"INT8 calibration image mode {image_mode!r} is unsupported: {relative_path}"
                    )
                if frame_count != 1:
                    raise ValueError(
                        f"animated or multi-frame calibration images are unsupported: {relative_path}"
                    )
                if exif_orientation != 1:
                    raise ValueError(
                        "INT8 calibration images must use identity EXIF orientation: "
                        f"{relative_path}"
                    )
                if width < 10 or height < 10:
                    raise ValueError(
                        f"INT8 calibration image dimensions must both be at least 10: {relative_path}"
                    )
                if width * height > _MAX_CALIBRATION_IMAGE_PIXELS:
                    raise ValueError(
                        "INT8 calibration image exceeds the "
                        f"{_MAX_CALIBRATION_IMAGE_PIXELS}-pixel safety limit: {relative_path}"
                    )
                if width * height > remaining_decoded_pixels:
                    raise ValueError(
                        "INT8 calibration images exceed the "
                        f"{_MAX_CALIBRATION_DECODED_PIXELS}-pixel decoded-work safety limit"
                    )

                handle.seek(0)
                with image_open(handle) as decoded:
                    decoded.load()

            encoded_array = np.frombuffer(encoded, dtype=np.uint8)
            if suffix in {".tif", ".tiff"}:
                success, frames = cv2.imdecodemulti(encoded_array, cv2.IMREAD_UNCHANGED)
                backend_image = (
                    frames[0] if success and len(frames) == 1 and frames[0].ndim == 3 else None
                )
            else:
                backend_image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
                if backend_image is not None and backend_image.ndim == 2:
                    backend_image = backend_image[..., None]
            if (
                backend_image is None
                or backend_image.dtype != np.uint8
                or backend_image.ndim != 3
                or backend_image.shape[2] != 3
            ):
                shape = None if backend_image is None else backend_image.shape
                dtype = None if backend_image is None else backend_image.dtype
                raise ValueError(
                    "INT8 calibration image must decode through the pinned backend as HxWx3 "
                    f"uint8, got shape={shape}, dtype={dtype}: {relative_path}"
                )

            source_height, source_width = backend_image.shape[:2]
            if (source_width, source_height) != (width, height):
                raise ValueError(
                    "INT8 calibration image decoders disagree on dimensions "
                    f"for {relative_path}: Pillow={(width, height)}, "
                    f"OpenCV={(source_width, source_height)}"
                )
            backend_pixels = source_width * source_height
            if source_width < 10 or source_height < 10:
                raise ValueError(
                    "INT8 calibration image backend dimensions must both be at least 10: "
                    f"{relative_path}"
                )
            if backend_pixels > _MAX_CALIBRATION_IMAGE_PIXELS:
                raise ValueError(
                    "INT8 calibration image exceeds the exact backend "
                    f"{_MAX_CALIBRATION_IMAGE_PIXELS}-pixel safety limit: {relative_path}"
                )
            if backend_pixels > remaining_decoded_pixels:
                raise ValueError(
                    "INT8 calibration images exceed the exact backend "
                    f"{_MAX_CALIBRATION_DECODED_PIXELS}-pixel decoded-work safety limit"
                )
            resize_ratio = image_size / max(source_height, source_width)
            if resize_ratio != 1:
                resized_width = min(math.ceil(source_width * resize_ratio), image_size)
                resized_height = min(math.ceil(source_height * resize_ratio), image_size)
                backend_image = cv2.resize(
                    backend_image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            loaded_height, loaded_width = backend_image.shape[:2]
            letterbox_ratio = min(image_size / loaded_height, image_size / loaded_width, 1.0)
            unpadded = (
                round(loaded_width * letterbox_ratio),
                round(loaded_height * letterbox_ratio),
            )
            if (loaded_width, loaded_height) != unpadded:
                backend_image = cv2.resize(
                    backend_image,
                    unpadded,
                    interpolation=cv2.INTER_LINEAR,
                )
            horizontal_padding = image_size - unpadded[0]
            vertical_padding = image_size - unpadded[1]
            left = round(horizontal_padding / 2 - 0.1)
            right = round(horizontal_padding / 2 + 0.1)
            top = round(vertical_padding / 2 - 0.1)
            bottom = round(vertical_padding / 2 + 0.1)
            backend_image = cv2.copyMakeBorder(
                backend_image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            tensor = np.ascontiguousarray(backend_image.transpose(2, 0, 1)[::-1])
            if tensor.shape != (3, image_size, image_size) or tensor.dtype != np.uint8:
                raise RuntimeError(
                    f"pinned calibration preprocessing produced an invalid tensor for {relative_path}"
                )
            tensor_hasher = hashlib.sha256(b"manwe-ultralytics-8.4.92-calibration-tensor-v1\0")
            tensor_hasher.update(image_size.to_bytes(8, "big"))
            tensor_hasher.update(tensor.tobytes())
            after = os.fstat(handle.fileno())
    except (
        OSError,
        SyntaxError,
        cv2.error,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"invalid INT8 calibration image {relative_path}: {exc}") from exc

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"INT8 calibration image changed while decoding: {relative_path}")
    return _CalibrationImage(
        relative_path=relative_path,
        file_sha256=file_hasher.hexdigest(),
        tensor_sha256=tensor_hasher.hexdigest(),
        width=source_width,
        height=source_height,
        image_format=image_format,
    )


def _calibration_inventory(
    root: pathlib.Path,
    val: object,
    entries: Sequence[tuple[pathlib.Path, str, os.stat_result]],
    image_size: int,
) -> tuple[_CalibrationImage, ...]:
    """Return the exact validated image set represented by the source ``val`` split."""
    entry_kinds = {entry: kind for entry, kind, _metadata in entries}
    entry_metadata = {entry: metadata for entry, _kind, metadata in entries}
    candidates: list[pathlib.Path] = []
    for value in _split_values(val, "val"):
        split_path = pathlib.Path(value)
        kind = "directory" if split_path == root else entry_kinds.get(split_path)
        if kind == "file":
            if split_path.suffix.lower() not in _CALIBRATION_IMAGE_FORMATS:
                raise ValueError(
                    "INT8 calibration val files must use a supported still-image suffix: "
                    f"{split_path.relative_to(root)}"
                )
            candidates.append(split_path)
        elif kind == "directory":
            candidates.extend(
                entry
                for entry, entry_kind, _metadata in entries
                if entry_kind == "file"
                and entry.is_relative_to(split_path)
                and entry.suffix.lower() in _CALIBRATION_IMAGE_FORMATS
            )
        else:  # The validated manifest should make this unreachable.
            raise RuntimeError(f"validated calibration split disappeared: {split_path}")

    candidates.sort(key=lambda value: value.relative_to(root).as_posix())
    seen_paths: set[pathlib.Path] = set()
    encoded_bytes = 0
    for candidate in candidates:
        relative = candidate.relative_to(root)
        if relative in seen_paths:
            raise ValueError(
                f"INT8 calibration val paths select an image more than once: {relative}"
            )
        seen_paths.add(relative)
        encoded_bytes += entry_metadata[candidate].st_size
        if encoded_bytes > _MAX_CALIBRATION_ENCODED_BYTES:
            raise ValueError(
                "INT8 calibration val images exceed the "
                f"{_MAX_CALIBRATION_ENCODED_BYTES}-byte encoded-work safety limit"
            )
    if len(candidates) > _MAX_CALIBRATION_IMAGES:
        raise ValueError(
            f"INT8 calibration val split exceeds the {_MAX_CALIBRATION_IMAGES}-image safety limit"
        )
    tensor_bytes = len(candidates) * 3 * image_size * image_size
    if tensor_bytes > _MAX_CALIBRATION_TENSOR_BYTES:
        raise ValueError(
            "INT8 calibration preprocessing exceeds the "
            f"{_MAX_CALIBRATION_TENSOR_BYTES}-byte tensor-work safety limit"
        )
    images: list[_CalibrationImage] = []
    tensor_origins: dict[str, pathlib.Path] = {}
    remaining_decoded_pixels = _MAX_CALIBRATION_DECODED_PIXELS
    for candidate in candidates:
        relative = candidate.relative_to(root)
        image = _validate_calibration_image(
            candidate,
            relative,
            image_size,
            remaining_decoded_pixels,
            entry_metadata[candidate],
        )
        previous = tensor_origins.get(image.tensor_sha256)
        if previous is not None:
            raise ValueError(
                "INT8 calibration images must produce unique backend tensors; "
                f"{relative} duplicates {previous}"
            )
        tensor_origins[image.tensor_sha256] = relative
        images.append(image)
        remaining_decoded_pixels -= image.width * image.height

    if len(images) < _MIN_CALIBRATION_IMAGES:
        raise ValueError(
            "INT8 calibration val split must contain at least "
            f"{_MIN_CALIBRATION_IMAGES} unique effective backend tensors"
        )
    return tuple(images)


def _selected_calibration_images(
    images: Sequence[_CalibrationImage],
) -> tuple[_CalibrationImage, ...]:
    """Choose one backend-stable 512-image subset independently of source names."""
    ordered = sorted(images, key=lambda image: image.tensor_sha256)
    return tuple(ordered[:_BACKEND_CALIBRATION_IMAGES])


class _CalibrationLoaderSnapshot:
    """Exact label-free image view consumed by the pinned Ultralytics loader."""

    def __init__(
        self,
        source_payload: Mapping[str, object],
        artifact_root: pathlib.Path,
        images: Sequence[_CalibrationImage],
    ) -> None:
        import yaml

        temporary = tempfile.TemporaryDirectory(prefix="manwe-calibration-loader-")
        try:
            root = pathlib.Path(temporary.name).resolve(strict=True)
            image_root = root / "calibration-images"
            image_root.mkdir(mode=0o700)
            for index, image in enumerate(images):
                source = artifact_root / image.relative_path
                destination = image_root / f"{index:08d}{source.suffix.lower()}"
                os.link(source, destination, follow_symlinks=False)
                source_metadata = source.stat()
                destination_metadata = destination.stat()
                if (source_metadata.st_dev, source_metadata.st_ino) != (
                    destination_metadata.st_dev,
                    destination_metadata.st_ino,
                ):
                    raise RuntimeError("private calibration image link does not bind its source")

            payload = {
                "path": str(root),
                "train": image_root.name,
                "val": image_root.name,
                "test": image_root.name,
                "names": copy.deepcopy(source_payload["names"]),
                "nc": source_payload["nc"],
                "channels": 3,
            }
            manifest_path = root / "dataset.yaml"
            manifest_bytes = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(manifest_path, flags, 0o400)
            try:
                owned_fd = fd
                fd = -1
                with owned_binary_writer(owned_fd) as handle:
                    handle.write(manifest_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            image_root.chmod(0o500)
            root.chmod(0o500)
            root_metadata = root.stat()
            root_identity = (root_metadata.st_dev, root_metadata.st_ino)
            destinations = tuple(
                image_root / f"{index:08d}{(artifact_root / image.relative_path).suffix.lower()}"
                for index, image in enumerate(images)
            )
            self._temporary = temporary
            self.root = root
            self.image_root = image_root
            self.path = manifest_path
            self._root_identity = root_identity
            self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self._destinations = destinations
            self._closed = False
        except BaseException as primary:
            self._closed = True
            _release_resources(
                (("calibration loader staging cleanup also failed", temporary.cleanup),),
                primary=primary,
            )
            raise

    def assert_unchanged(
        self,
        artifact_root: pathlib.Path,
        images: Sequence[_CalibrationImage],
    ) -> None:
        """Require the backend-visible manifest and hard-linked inventory to remain exact."""
        if self._closed:
            raise RuntimeError("private calibration loader snapshot is closed")
        from .artifacts import _tree_entries

        metadata = self.root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._root_identity
            or metadata.st_mode & 0o222
        ):
            raise RuntimeError("private calibration loader root changed")
        entries = _tree_entries(self.root, len(images) + 2)
        expected = {
            self.path: "file",
            self.image_root: "directory",
            **{destination: "file" for destination in self._destinations},
        }
        actual = {entry: kind for entry, kind, _metadata in entries}
        if actual != expected:
            raise RuntimeError("private calibration loader inventory changed")
        if any(metadata.st_mode & 0o222 for _entry, _kind, metadata in entries):
            raise RuntimeError("private calibration loader became writable")
        try:
            manifest = read_bounded_regular_bytes(
                self.path,
                _MAX_MANIFEST_BYTES,
                "private calibration manifest",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("private calibration manifest changed") from exc
        manifest_sha256 = hashlib.sha256(manifest).hexdigest()
        if manifest_sha256 != self._manifest_sha256:
            raise RuntimeError("private calibration manifest changed")
        for image, destination in zip(images, self._destinations, strict=True):
            source_metadata = (artifact_root / image.relative_path).stat()
            destination_metadata = destination.stat()
            if (source_metadata.st_dev, source_metadata.st_ino) != (
                destination_metadata.st_dev,
                destination_metadata.st_ino,
            ):
                raise RuntimeError("private calibration image view changed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._temporary.cleanup()


class DatasetManifestSnapshot:
    """A validated source binding plus a normalized admission manifest.

    The source tree remains caller-owned and mutable. Long-running consumers
    must call :meth:`snapshot_for_training` and use that immutable private copy.
    """

    def __init__(
        self,
        payload: Mapping[str, object],
        source_manifest: _BoundSourceManifest,
        source_root: _BoundSourceDirectory,
    ) -> None:
        self._source_manifest = source_manifest
        self._source_root = source_root
        self._closed = False
        temporary = None
        try:
            import yaml

            payload_copy = copy.deepcopy(dict(payload))
            temporary = tempfile.TemporaryDirectory(prefix="manwe-dataset-")
            temporary_root = pathlib.Path(temporary.name).resolve(strict=True)
            snapshot_path = temporary_root / "dataset.yaml"
            root = pathlib.Path(str(payload_copy["path"]))
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(snapshot_path, flags, 0o400)
            try:
                owned_fd = fd
                fd = -1
                with owned_text_writer(owned_fd, encoding="utf-8", newline="\n") as handle:
                    yaml.safe_dump(payload_copy, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            if temporary is None:
                raise RuntimeError("dataset manifest staging directory was not initialized")
            self._payload = payload_copy
            self._temporary = temporary
            self.path = snapshot_path
            self.root = root
        except BaseException as primary:
            self._closed = True
            cleanups: list[_Cleanup] = []
            if temporary is not None:
                cleanups.append(("dataset manifest staging cleanup also failed", temporary.cleanup))
            cleanups.extend(
                (
                    ("source dataset root cleanup also failed", self._source_root.close),
                    ("source dataset manifest cleanup also failed", self._source_manifest.close),
                )
            )
            _release_resources(cleanups, primary=primary)
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _release_resources(
                (
                    ("dataset manifest staging cleanup failed", self._temporary.cleanup),
                    ("source dataset root cleanup also failed", self._source_root.close),
                    ("source dataset manifest cleanup also failed", self._source_manifest.close),
                )
            )

    def snapshot_for_training(self) -> TrainingDatasetSnapshot:
        """Revalidate and copy the dataset tree for one deterministic training run."""
        if self._closed:
            raise RuntimeError("dataset manifest snapshot is closed")
        self._source_root.assert_path_unchanged()
        self._source_manifest.assert_unchanged()
        # Admission and consumption are separate events. Re-run the complete
        # split/tree policy at the latter boundary so a caller cannot validate a
        # benign layout, alter it, and then ask Manwe to preserve the altered
        # layout in the backend's private copy.
        with validate_local_detection_manifest(self._source_manifest.path) as refreshed:
            if (
                refreshed._source_manifest._identity != self._source_manifest._identity
                or refreshed._source_manifest.sha256 != self._source_manifest.sha256
                or refreshed._source_root._identity != self._source_root._identity
            ):
                raise ValueError(
                    "dataset manifest or declared root identity changed after admission"
                )
            return TrainingDatasetSnapshot(refreshed)

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

    def _calibration_digests_for_root(
        self,
        root: pathlib.Path,
        content_digest: str,
        image_size: int,
        tensorrt_version: str | None = None,
        modelopt_version: str | None = None,
    ) -> tuple[str, str, tuple[_CalibrationImage, ...]]:
        """Return the policy/manifest/tree digest, tree digest, and loader inventory."""
        import yaml

        from .artifacts import _tree_entries

        image_size = _validated_calibration_image_size(image_size)
        runtime_policy = _calibration_runtime_policy(tensorrt_version, modelopt_version)
        if b"route=modelopt-max" in runtime_policy:
            # Pinned ModelOpt retains the uint8 batch list, then materializes
            # concatenated float32 input and division output: conservatively 10x.
            modelopt_work_bytes = _BACKEND_CALIBRATION_IMAGES * 3 * image_size * image_size * 10
            if modelopt_work_bytes > _MAX_MODELOPT_CALIBRATION_WORK_BYTES:
                raise ValueError(
                    "TensorRT 11 ModelOpt calibration exceeds the "
                    f"{_MAX_MODELOPT_CALIBRATION_WORK_BYTES}-byte peak-work safety limit"
                )
        if self._payload.get("channels", 3) != 3:
            raise ValueError("INT8 calibration manifests must declare or default to 3 channels")
        entries = _tree_entries(root, _MAX_DATASET_ENTRIES)
        source_bytes = sum(metadata.st_size for _entry, kind, metadata in entries if kind == "file")
        if source_bytes > _MAX_DATASET_BYTES:
            raise ValueError(
                f"calibration dataset exceeds the {_MAX_DATASET_BYTES}-byte safety limit"
            )
        images = _calibration_inventory(
            root,
            self._payload_for_root(root)["val"],
            entries,
            image_size,
        )
        canonical_payload = self._payload_for_root(pathlib.Path("."))
        manifest_bytes = yaml.safe_dump(canonical_payload, sort_keys=True).encode("utf-8")
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError(f"normalized calibration manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
        hasher = hashlib.sha256(b"manwe-calibration-dataset-v4\0")
        _update_digest_field(hasher, _CALIBRATION_POLICY)
        _update_digest_field(hasher, runtime_policy)
        _update_digest_field(hasher, image_size.to_bytes(8, "big"))
        _update_digest_field(hasher, hashlib.sha256(manifest_bytes).digest())
        _update_digest_field(hasher, bytes.fromhex(content_digest))
        for image in images:
            _update_digest_field(hasher, image.relative_path.as_posix().encode("utf-8"))
            _update_digest_field(hasher, bytes.fromhex(image.file_sha256))
            _update_digest_field(hasher, bytes.fromhex(image.tensor_sha256))
            _update_digest_field(hasher, image.width.to_bytes(8, "big"))
            _update_digest_field(hasher, image.height.to_bytes(8, "big"))
            _update_digest_field(hasher, image.image_format.encode("ascii"))
        for image in _selected_calibration_images(images):
            _update_digest_field(hasher, b"backend-image")
            _update_digest_field(hasher, bytes.fromhex(image.tensor_sha256))
        return hasher.hexdigest(), content_digest, images

    def _calibration_digests(
        self,
        image_size: int,
        tensorrt_version: str | None = None,
        modelopt_version: str | None = None,
    ) -> tuple[str, str, tuple[_CalibrationImage, ...]]:
        """Snapshot the pinned source descriptor before decoding any calibration image."""
        if self._closed:
            raise RuntimeError("dataset manifest snapshot is closed")
        from .artifacts import ArtifactSnapshot

        self._source_root.assert_path_unchanged()
        self._source_manifest.assert_unchanged()
        content_digest = self._source_root.digest()
        with ArtifactSnapshot.from_directory_fd(
            self._source_root.fd,
            content_digest,
            display=str(self.root),
            max_bytes=_MAX_DATASET_BYTES,
            max_entries=_MAX_DATASET_ENTRIES,
            source_tree_validator=partial(
                _validate_snapshotted_split_identities,
                self.root,
                self._payload,
            ),
        ) as source_snapshot:
            self._source_root.assert_path_unchanged()
            self._source_manifest.assert_unchanged()
            current_digest = self._source_root.digest()
            self._source_root.assert_path_unchanged()
            self._source_manifest.assert_unchanged()
            if current_digest != content_digest:
                raise RuntimeError("source calibration dataset changed while snapshotting")
            return self._calibration_digests_for_root(
                source_snapshot.path,
                content_digest,
                image_size,
                tensorrt_version,
                modelopt_version,
            )

    def calibration_digest(
        self,
        *,
        image_size: int = 640,
        tensorrt_version: str | None = None,
        modelopt_version: str | None = None,
    ) -> str:
        """Bind loader policy, image size, exact backend tensors, and the source tree.

        The minimum image count is a policy floor, not a claim that 1,000 images
        are statistically sufficient for every INT8 model.
        """
        digest, _content_digest, _images = self._calibration_digests(
            image_size,
            tensorrt_version,
            modelopt_version,
        )
        return digest

    def __enter__(self) -> DatasetManifestSnapshot:
        if self._closed:
            raise RuntimeError("dataset manifest snapshot is closed")
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        if exc is None:
            self.close()
            return
        _release_resources(
            (("dataset manifest snapshot cleanup also failed", self.close),),
            primary=exc,
        )


class TrainingDatasetSnapshot:
    """An immutable private dataset tree and normalized backend manifest."""

    def __init__(self, source: DatasetManifestSnapshot) -> None:
        from .artifacts import ArtifactSnapshot

        if type(source) is not DatasetManifestSnapshot or source._closed:
            raise TypeError("source must be an open DatasetManifestSnapshot")
        source_root = source._source_root.clone()
        try:
            source_manifest = source._source_manifest.clone()
        except BaseException as primary:
            _release_resources(
                (("cloned training dataset root cleanup also failed", source_root.close),),
                primary=primary,
            )
            raise

        artifact_snapshot = None
        temporary = None
        try:
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            content_digest = source_root.digest()
            artifact_snapshot = ArtifactSnapshot.from_directory_fd(
                source_root.fd,
                content_digest,
                display=str(source.root),
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
                source_tree_validator=partial(
                    _validate_snapshotted_split_identities,
                    source.root,
                    source._payload,
                    require_nonempty=frozenset({"train", "val"}),
                ),
            )
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            if source_root.digest() != content_digest:
                raise RuntimeError("source training dataset changed while snapshotting")
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()

            import yaml

            payload = source._payload_for_root(artifact_snapshot.path)
            _validate_ultralytics_training_snapshot(artifact_snapshot.path, payload)
            manifest_bytes = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
            if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
                raise ValueError(
                    f"normalized training manifest exceeds {_MAX_MANIFEST_BYTES} bytes"
                )
            temporary = tempfile.TemporaryDirectory(prefix="manwe-training-dataset-")
            loader_root = pathlib.Path(temporary.name).resolve(strict=True)
            manifest_path = loader_root / "dataset.yaml"
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(manifest_path, flags, 0o400)
            try:
                owned_fd = fd
                fd = -1
                with owned_binary_writer(owned_fd) as handle:
                    handle.write(manifest_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            loader_root.chmod(0o500)

            _release_resources(
                (
                    ("cloned training dataset root cleanup failed", source_root.close),
                    ("cloned training dataset manifest cleanup failed", source_manifest.close),
                )
            )
            self._artifact_snapshot = artifact_snapshot
            self._temporary = temporary
            self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self._closed = False
            self.root = artifact_snapshot.path
            self.path = manifest_path
            self.sha256 = content_digest
        except BaseException as primary:
            cleanups: list[_Cleanup] = []
            if temporary is not None:
                cleanups.append(
                    ("training manifest snapshot cleanup also failed", temporary.cleanup)
                )
            if artifact_snapshot is not None:
                cleanups.append(
                    ("training dataset snapshot cleanup also failed", artifact_snapshot.close)
                )
            cleanups.extend(
                (
                    ("cloned training dataset root cleanup also failed", source_root.close),
                    (
                        "cloned training dataset manifest cleanup also failed",
                        source_manifest.close,
                    ),
                )
            )
            _release_resources(cleanups, primary=primary)
            raise

    def assert_unchanged(self) -> None:
        """Re-authenticate the exact bytes exposed to the training backend."""
        if self._closed:
            raise RuntimeError("training dataset snapshot is closed")
        from .artifacts import sha256_artifact

        if (
            sha256_artifact(
                self.root,
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
            )
            != self.sha256
        ):
            raise RuntimeError("private training dataset changed")
        try:
            manifest = read_bounded_regular_bytes(
                self.path,
                _MAX_MANIFEST_BYTES,
                "private training manifest",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("private training manifest changed") from exc
        if hashlib.sha256(manifest).hexdigest() != self._manifest_sha256:
            raise RuntimeError("private training manifest changed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _release_resources(
                (
                    ("training manifest snapshot cleanup failed", self._temporary.cleanup),
                    (
                        "training dataset snapshot cleanup also failed",
                        self._artifact_snapshot.close,
                    ),
                )
            )

    def __enter__(self) -> TrainingDatasetSnapshot:
        if self._closed:
            raise RuntimeError("training dataset snapshot is closed")
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        if exc is None:
            self.close()
            return
        _release_resources(
            (("training dataset snapshot cleanup also failed", self.close),),
            primary=exc,
        )


class TrainingDirectorySnapshot:
    """A private regular-tree copy for a backend with its own dataset schema."""

    def __init__(self, artifact_snapshot) -> None:
        self._artifact_snapshot = artifact_snapshot
        self.path = artifact_snapshot.path
        self.sha256 = artifact_snapshot.sha256
        self._closed = False

    def assert_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("training directory snapshot is closed")
        from .artifacts import sha256_artifact

        if (
            sha256_artifact(
                self.path,
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
            )
            != self.sha256
        ):
            raise RuntimeError("private training directory changed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._artifact_snapshot.close()

    def __enter__(self) -> TrainingDirectorySnapshot:
        if self._closed:
            raise RuntimeError("training directory snapshot is closed")
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        if exc is None:
            self.close()
            return
        _release_resources(
            (("training directory snapshot cleanup also failed", self.close),),
            primary=exc,
        )


class CalibrationDatasetSnapshot:
    """A bounded source copy plus the exact read-only backend calibration view."""

    def __init__(
        self,
        source: DatasetManifestSnapshot,
        image_size: int,
        tensorrt_version: str | None,
        modelopt_version: str | None,
    ) -> None:
        from .artifacts import ArtifactSnapshot, sha256_artifact

        image_size = _validated_calibration_image_size(image_size)
        source_root = source._source_root.clone()
        try:
            source_manifest = source._source_manifest.clone()
        except BaseException as primary:
            _release_resources(
                (("cloned source dataset root cleanup also failed", source_root.close),),
                primary=primary,
            )
            raise
        artifact_snapshot = None
        loader_snapshot = None
        self._closed = True
        try:
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            source_content_digest = source_root.digest()
            artifact_snapshot = ArtifactSnapshot.from_directory_fd(
                source_root.fd,
                source_content_digest,
                display=str(source.root),
                max_bytes=_MAX_DATASET_BYTES,
                max_entries=_MAX_DATASET_ENTRIES,
                source_tree_validator=partial(
                    _validate_snapshotted_split_identities,
                    source.root,
                    source._payload,
                ),
            )
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            current_digest = source_root.digest()
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            if current_digest != source_content_digest:
                raise RuntimeError("source calibration dataset changed while snapshotting")
            expected_digest, _content_digest, source_images = source._calibration_digests_for_root(
                artifact_snapshot.path,
                source_content_digest,
                image_size,
                tensorrt_version,
                modelopt_version,
            )
            for image in source_images:
                private_digest = sha256_artifact(
                    artifact_snapshot.path / image.relative_path,
                    max_bytes=_MAX_CALIBRATION_IMAGE_BYTES,
                    max_entries=1,
                )
                if private_digest != image.file_sha256:
                    raise RuntimeError(
                        "private calibration image bytes differ from their validated source"
                    )
            loader_images = _selected_calibration_images(source_images)
            loader_snapshot = _CalibrationLoaderSnapshot(
                source._payload,
                artifact_snapshot.path,
                loader_images,
            )
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            current_digest = source_root.digest()
            source_root.assert_path_unchanged()
            source_manifest.assert_unchanged()
            if current_digest != source_content_digest:
                raise RuntimeError("source calibration dataset changed while snapshotting")
            self._artifact_snapshot = artifact_snapshot
            self._loader_snapshot = loader_snapshot
            self._images = loader_images
            self._source_root = source_root
            self._source_content_digest = source_content_digest
            self._source_manifest = source_manifest
            self.sha256 = expected_digest
            self.root = artifact_snapshot.path
            self.loader_root = loader_snapshot.root
            self.path = loader_snapshot.path
            self._closed = False
        except BaseException as primary:
            cleanups: list[_Cleanup] = []
            if loader_snapshot is not None:
                cleanups.append(
                    ("calibration loader snapshot cleanup also failed", loader_snapshot.close)
                )
            if artifact_snapshot is not None:
                cleanups.append(
                    ("calibration artifact snapshot cleanup also failed", artifact_snapshot.close)
                )
            cleanups.extend(
                (
                    ("cloned source dataset root cleanup also failed", source_root.close),
                    ("cloned source dataset manifest cleanup also failed", source_manifest.close),
                )
            )
            _release_resources(cleanups, primary=primary)
            raise

    def calibration_digest(self) -> str:
        """Re-hash private raw bytes so backend-time mutation fails closed."""
        if self._closed:
            raise RuntimeError("calibration snapshot is closed")
        from .artifacts import sha256_artifact

        self._loader_snapshot.assert_unchanged(self.root, self._images)
        content_digest = sha256_artifact(
            self.root,
            max_bytes=_MAX_DATASET_BYTES,
            max_entries=_MAX_DATASET_ENTRIES,
        )
        if content_digest != self._source_content_digest:
            raise RuntimeError("private calibration source copy changed")
        return self.sha256

    def assert_source_unchanged(self) -> None:
        """Require the caller-visible source tree to remain the snapshotted tree."""
        if self._closed:
            raise RuntimeError("calibration snapshot is closed")
        try:
            self._source_root.assert_path_unchanged()
        except (OSError, ValueError) as exc:
            raise RuntimeError("source calibration dataset root was replaced") from exc
        try:
            self._source_manifest.assert_unchanged()
        except (OSError, ValueError) as exc:
            raise RuntimeError("source calibration manifest changed after snapshotting") from exc
        try:
            current_digest = self._source_root.digest()
        except (OSError, ValueError) as exc:
            raise RuntimeError("source calibration dataset changed after snapshotting") from exc
        try:
            self._source_root.assert_path_unchanged()
        except (OSError, ValueError) as exc:
            raise RuntimeError("source calibration dataset root was replaced") from exc
        try:
            self._source_manifest.assert_unchanged()
        except (OSError, ValueError) as exc:
            raise RuntimeError("source calibration manifest changed after snapshotting") from exc
        if current_digest != self._source_content_digest:
            raise RuntimeError("source calibration dataset changed after snapshotting")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _release_resources(
                (
                    ("calibration loader snapshot cleanup failed", self._loader_snapshot.close),
                    (
                        "calibration artifact snapshot cleanup also failed",
                        self._artifact_snapshot.close,
                    ),
                    ("cloned source dataset root cleanup also failed", self._source_root.close),
                    (
                        "cloned source dataset manifest cleanup also failed",
                        self._source_manifest.close,
                    ),
                )
            )

    def __enter__(self) -> CalibrationDatasetSnapshot:
        if self._closed:
            raise RuntimeError("calibration snapshot is closed")
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        if exc is None:
            self.close()
            return
        _release_resources(
            (("calibration dataset snapshot cleanup also failed", self.close),),
            primary=exc,
        )


def validate_local_detection_manifest(
    path: str | pathlib.Path,
) -> DatasetManifestSnapshot:
    """Validate local splits and return a private directive-free YAML snapshot."""
    manifest = pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))
    source_manifest: _BoundSourceManifest | None = None
    source_root: _BoundSourceDirectory | None = None
    try:
        source_manifest = _BoundSourceManifest(manifest)
        try:
            payload = load_unambiguous_yaml(source_manifest.text, "dataset manifest")
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
        root_candidate = pathlib.Path(root_value).expanduser()
        if root_candidate.is_absolute():
            root = pathlib.Path(os.path.abspath(root_candidate))
            source_root = _BoundSourceDirectory.from_absolute(root)
        else:
            relative_root = pathlib.PurePath(root_candidate)
            if ".." in relative_root.parts:
                raise ValueError(
                    "relative dataset path must not contain parent-directory components"
                )
            source_root = _BoundSourceDirectory.from_manifest_relative(
                source_manifest,
                relative_root,
            )
            root = source_root.path

        sanitized: dict[str, object] = {"path": str(root)}
        split_roots: dict[str, list[tuple[pathlib.Path, pathlib.PurePath, os.stat_result]]] = {}
        for field in ("train", "val", "test"):
            if field not in payload:
                continue
            if field == "test" and payload[field] in (None, ""):
                continue
            values = _split_values(payload[field], field)
            normalized_paths: list[str] = []
            roots: list[tuple[pathlib.Path, pathlib.PurePath, os.stat_result]] = []
            for raw in values:
                _reject_remote_or_archive(raw, field)
                raw_candidate = pathlib.Path(raw).expanduser()
                if raw_candidate.is_absolute():
                    candidate = pathlib.Path(os.path.abspath(raw_candidate))
                else:
                    relative_candidate = pathlib.PurePath(raw_candidate)
                    if ".." in relative_candidate.parts:
                        raise ValueError(
                            f"dataset {field} must remain inside the declared dataset root"
                        )
                    candidate = pathlib.Path(os.path.abspath(root / relative_candidate))
                if candidate.suffix.lower() == ".txt":
                    raise ValueError(
                        f"dataset {field} path-list indirection is not accepted; use a local "
                        "directory"
                    )
                try:
                    relative = candidate.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"dataset {field} must remain inside the declared dataset root"
                    ) from exc
                metadata = _stat_relative_entry(
                    source_root.fd,
                    relative,
                    subject=f"dataset {field}",
                    require_directory=True,
                )
                normalized_paths.append(str(candidate))
                roots.append((candidate, relative, metadata))
            if len(set(normalized_paths)) != len(normalized_paths):
                raise ValueError(f"dataset {field} must not contain duplicate paths")
            paths = [pathlib.Path(value) for value in normalized_paths]
            for left, right in combinations(paths, 2):
                if left in right.parents or right in left.parents:
                    raise ValueError(
                        f"dataset {field} paths overlap and would select files more than once: "
                        f"{left} and {right}"
                    )
            split_roots[field] = roots
            sanitized[field] = (
                normalized_paths[0] if len(normalized_paths) == 1 else normalized_paths
            )

        for (left_name, left_roots), (right_name, right_roots) in combinations(
            split_roots.items(), 2
        ):
            for left, _left_relative, _left_metadata in left_roots:
                for right, _right_relative, _right_metadata in right_roots:
                    if left == right or left in right.parents or right in left.parents:
                        raise ValueError(
                            f"dataset {left_name} and {right_name} paths overlap: "
                            f"{left} and {right}"
                        )

        split_files = _inventory_selected_split_set(
            source_root.fd,
            split_roots,
            reverse=False,
        )
        try:
            confirmed_split_files = _inventory_selected_split_set(
                source_root.fd,
                split_roots,
                reverse=True,
            )
        except (OSError, ValueError) as exc:
            raise ValueError("selected dataset splits changed during aggregate inventory") from exc
        if split_files != confirmed_split_files:
            raise ValueError("selected dataset splits changed during aggregate inventory")

        _validate_split_file_identities(split_files)

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
        unique_names = (
            sanitized_names.values() if isinstance(sanitized_names, dict) else sanitized_names
        )
        if len(set(unique_names)) != class_count:
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

        # Re-observe the complete aggregate after every other manifest field has
        # been validated. The result is an admission-time observation, not an
        # immutable dataset; consumers that run later must use a private copy.
        try:
            final_split_files = _inventory_selected_split_set(
                source_root.fd,
                split_roots,
                reverse=False,
            )
            final_reverse_split_files = _inventory_selected_split_set(
                source_root.fd,
                split_roots,
                reverse=True,
            )
        except (OSError, ValueError) as exc:
            raise ValueError("selected dataset splits changed before validation completed") from exc
        if (
            final_split_files != final_reverse_split_files
            or final_split_files != confirmed_split_files
        ):
            raise ValueError("selected dataset splits changed before validation completed")
        _validate_split_file_identities(final_split_files)
        source_root.assert_path_unchanged()
        source_manifest.assert_unchanged()
        result = DatasetManifestSnapshot(sanitized, source_manifest, source_root)
        source_manifest = None
        source_root = None
        return result
    except BaseException as primary:
        cleanups: list[_Cleanup] = []
        if source_root is not None:
            cleanups.append(("source dataset root cleanup also failed", source_root.close))
        if source_manifest is not None:
            cleanups.append(("source dataset manifest cleanup also failed", source_manifest.close))
        _release_resources(cleanups, primary=primary)
        raise


def snapshot_local_calibration_dataset(
    path: str | pathlib.Path,
    *,
    image_size: int = 640,
    tensorrt_version: str | None = None,
    modelopt_version: str | None = None,
) -> CalibrationDatasetSnapshot:
    """Validate and privately snapshot one bounded local INT8 calibration dataset."""
    validated = validate_local_detection_manifest(path)
    try:
        result = CalibrationDatasetSnapshot(
            validated,
            image_size,
            tensorrt_version,
            modelopt_version,
        )
    except BaseException as primary:
        _release_resources(
            (("validated dataset manifest cleanup also failed", validated.close),),
            primary=primary,
        )
        raise
    try:
        validated.close()
    except BaseException as primary:
        _release_resources(
            (("constructed calibration snapshot cleanup also failed", result.close),),
            primary=primary,
        )
        raise
    return result


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"RF-DETR annotation JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_rfdetr_annotation(path: pathlib.Path) -> dict[str, object]:
    try:
        encoded = read_bounded_regular_bytes(
            path,
            _MAX_RFDETR_ANNOTATION_BYTES,
            "RF-DETR COCO annotation",
        )
    except OSError as exc:
        raise ValueError(f"RF-DETR annotation could not be read safely: {path}") from exc
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"RF-DETR annotation is not strict bounded JSON: {path}: {exc}") from exc
    if type(payload) is not dict:
        raise ValueError(f"RF-DETR annotation root must be an object: {path}")

    nodes = 0
    pending: list[tuple[object, int]] = [(payload, 1)]
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_RFDETR_JSON_NODES:
            raise ValueError(
                f"RF-DETR annotation exceeds the {_MAX_RFDETR_JSON_NODES}-node limit: {path}"
            )
        if depth > _MAX_RFDETR_JSON_DEPTH:
            raise ValueError(
                f"RF-DETR annotation exceeds the {_MAX_RFDETR_JSON_DEPTH}-level limit: {path}"
            )
        if type(value) is dict:
            mapping = cast(dict[object, object], value)
            if any(type(key) is not str for key in mapping):
                raise ValueError(f"RF-DETR annotation object keys must be strings: {path}")
            pending.extend((item, depth + 1) for item in mapping.values())
        elif type(value) is list:
            pending.extend((item, depth + 1) for item in cast(list[object], value))
        elif type(value) is float and not math.isfinite(value):
            raise ValueError(f"RF-DETR annotation contains a non-finite number: {path}")
        elif type(value) not in {str, int, float, bool, type(None)}:
            raise ValueError(f"RF-DETR annotation contains an unsupported JSON value: {path}")
    return cast(dict[str, object], payload)


def _rfdetr_integer(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**53:
        raise ValueError(f"{name} must be an integer in [0, 2**53]")
    return cast(int, value)


def _rfdetr_number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _rfdetr_image_relative_path(value: object, split: str) -> pathlib.PurePosixPath:
    if type(value) is not str:
        raise ValueError(f"RF-DETR {split} image file_name must be a relative string")
    name = cast(str, value)
    if (
        not name
        or not name.isprintable()
        or "\\" in name
        or "\0" in name
        or len(name.encode("utf-8")) > _MAX_RFDETR_PATH_BYTES
    ):
        raise ValueError(f"RF-DETR {split} image file_name is not a bounded portable path")
    relative = pathlib.PurePosixPath(name)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != name
    ):
        raise ValueError(f"RF-DETR {split} image file_name must stay canonically inside its split")
    if relative.suffix.lower() not in _RFDETR_IMAGE_SUFFIXES:
        raise ValueError(
            f"RF-DETR {split} image file_name has unsupported suffix {relative.suffix!r}"
        )
    return relative


def _validate_rfdetr_coco_split(
    root: pathlib.Path,
    split: str,
    *,
    require_images: bool,
) -> tuple[tuple[int, str], ...]:
    split_path = root / split
    try:
        split_metadata = split_path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"RF-DETR dataset requires a {split!r} directory") from exc
    if not stat.S_ISDIR(split_metadata.st_mode):
        raise ValueError(f"RF-DETR dataset {split!r} split must be a directory")
    annotation_path = split_path / "_annotations.coco.json"
    payload = _load_rfdetr_annotation(annotation_path)

    raw_categories = payload.get("categories")
    if type(raw_categories) is not list or not 1 <= len(raw_categories) <= _MAX_CLASSES:
        raise ValueError(f"RF-DETR {split} categories must contain 1..{_MAX_CLASSES} objects")
    categories: list[tuple[int, str]] = []
    category_ids: set[int] = set()
    category_names: set[str] = set()
    for index, raw_category in enumerate(raw_categories):
        if type(raw_category) is not dict:
            raise ValueError(f"RF-DETR {split} categories[{index}] must be an object")
        category = cast(dict[str, object], raw_category)
        category_id = _rfdetr_integer(category.get("id"), f"RF-DETR {split} categories[{index}].id")
        name_value = category.get("name")
        if type(name_value) is not str:
            raise ValueError(f"RF-DETR {split} categories[{index}].name must be a string")
        name = cast(str, name_value).strip()
        if not name or not name.isprintable() or len(name.encode("utf-8")) > _MAX_CLASS_NAME_BYTES:
            raise ValueError(
                f"RF-DETR {split} categories[{index}].name must be bounded printable text"
            )
        if category_id in category_ids or name in category_names:
            raise ValueError(f"RF-DETR {split} categories must have unique ids and names")
        category_ids.add(category_id)
        category_names.add(name)
        categories.append((category_id, name))

    raw_images = payload.get("images")
    if type(raw_images) is not list or len(raw_images) > _MAX_RFDETR_IMAGES:
        raise ValueError(
            f"RF-DETR {split} images must be a list of at most {_MAX_RFDETR_IMAGES} objects"
        )
    if require_images and not raw_images:
        raise ValueError(f"RF-DETR {split} split must contain at least one image")
    image_ids: set[int] = set()
    image_paths: set[str] = set()
    image_dimensions: dict[int, tuple[int, int]] = {}
    for index, raw_image in enumerate(raw_images):
        if type(raw_image) is not dict:
            raise ValueError(f"RF-DETR {split} images[{index}] must be an object")
        image = cast(dict[str, object], raw_image)
        image_id = _rfdetr_integer(image.get("id"), f"RF-DETR {split} images[{index}].id")
        if image_id in image_ids:
            raise ValueError(f"RF-DETR {split} image ids must be unique")
        image_ids.add(image_id)
        relative = _rfdetr_image_relative_path(image.get("file_name"), split)
        relative_text = relative.as_posix()
        if relative_text in image_paths:
            raise ValueError(f"RF-DETR {split} image file_name values must be unique")
        image_paths.add(relative_text)
        width = _rfdetr_integer(image.get("width"), f"RF-DETR {split} images[{index}].width")
        height = _rfdetr_integer(image.get("height"), f"RF-DETR {split} images[{index}].height")
        if (
            width == 0
            or height == 0
            or width > _MAX_RFDETR_IMAGE_DIMENSION
            or height > _MAX_RFDETR_IMAGE_DIMENSION
            or width * height > _MAX_RFDETR_IMAGE_PIXELS
        ):
            raise ValueError(f"RF-DETR {split} image dimensions exceed the supported bounds")
        image_path = split_path.joinpath(*relative.parts)
        try:
            metadata = image_path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"RF-DETR {split} annotation references missing image {relative_text!r}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError(
                f"RF-DETR {split} annotation image must be a nonempty regular file: "
                f"{relative_text!r}"
            )
        _validate_training_image_header(
            image_path,
            pathlib.Path(split) / relative,
            metadata,
            expected_dimensions=(width, height),
        )
        image_dimensions[image_id] = (width, height)

    raw_annotations = payload.get("annotations")
    if type(raw_annotations) is not list or len(raw_annotations) > _MAX_RFDETR_ANNOTATIONS:
        raise ValueError(
            "RF-DETR "
            f"{split} annotations must be a list of at most {_MAX_RFDETR_ANNOTATIONS} objects"
        )
    annotation_ids: set[int] = set()
    for index, raw_annotation in enumerate(raw_annotations):
        if type(raw_annotation) is not dict:
            raise ValueError(f"RF-DETR {split} annotations[{index}] must be an object")
        annotation = cast(dict[str, object], raw_annotation)
        annotation_id = _rfdetr_integer(
            annotation.get("id"), f"RF-DETR {split} annotations[{index}].id"
        )
        if annotation_id in annotation_ids:
            raise ValueError(f"RF-DETR {split} annotation ids must be unique")
        annotation_ids.add(annotation_id)
        image_id = _rfdetr_integer(
            annotation.get("image_id"),
            f"RF-DETR {split} annotations[{index}].image_id",
        )
        category_id = _rfdetr_integer(
            annotation.get("category_id"),
            f"RF-DETR {split} annotations[{index}].category_id",
        )
        if image_id not in image_ids:
            raise ValueError(f"RF-DETR {split} annotation references an unknown image id")
        if category_id not in category_ids:
            raise ValueError(f"RF-DETR {split} annotation references an unknown category id")
        bbox = annotation.get("bbox")
        if type(bbox) is not list or len(bbox) != 4:
            raise ValueError(f"RF-DETR {split} annotation bbox must be [x, y, width, height]")
        box = [
            _rfdetr_number(value, f"RF-DETR {split} annotations[{index}].bbox") for value in bbox
        ]
        image_width, image_height = image_dimensions[image_id]
        if (
            box[0] < 0.0
            or box[1] < 0.0
            or box[2] <= 0.0
            or box[3] <= 0.0
            or box[0] + box[2] > image_width
            or box[1] + box[3] > image_height
        ):
            raise ValueError(f"RF-DETR {split} annotation bbox is outside supported bounds")
        area = _rfdetr_number(annotation.get("area"), f"RF-DETR {split} annotations[{index}].area")
        if area <= 0.0 or area > image_width * image_height:
            raise ValueError(f"RF-DETR {split} annotation area is outside supported bounds")
        if "iscrowd" in annotation and (
            type(annotation["iscrowd"]) is not int or annotation["iscrowd"] not in {0, 1}
        ):
            raise ValueError(f"RF-DETR {split} annotation iscrowd must be 0 or 1")
    return tuple(sorted(categories))


def _validate_rfdetr_coco_snapshot(root: pathlib.Path) -> None:
    """Bind Manwe's RF-DETR path to the exact Roboflow COCO consumer schema."""
    train_categories = _validate_rfdetr_coco_split(root, "train", require_images=True)
    valid_categories = _validate_rfdetr_coco_split(root, "valid", require_images=True)
    if valid_categories != train_categories:
        raise ValueError("RF-DETR train and valid category tables must agree exactly")
    test_path = root / "test"
    if test_path.exists():
        test_categories = _validate_rfdetr_coco_split(root, "test", require_images=False)
        if test_categories != train_categories:
            raise ValueError("RF-DETR test category table must agree with train and valid")


def _reject_rfdetr_split_hardlinks(entries: Sequence[_DescriptorTreeEntry]) -> None:
    """Reject aliases between files that RF-DETR can treat as independent splits."""
    identities: dict[tuple[int, int], str] = {}
    for entry in entries:
        if entry.is_directory:
            continue
        relative = pathlib.PurePosixPath(entry.relative)
        if not relative.parts or relative.parts[0] not in {"train", "valid", "test"}:
            continue
        identity = (entry.identity[0], entry.identity[1])
        previous = identities.setdefault(identity, entry.relative)
        if previous != entry.relative:
            raise ValueError(
                "RF-DETR split files must not be hardlink aliases: "
                f"{previous!r} and {entry.relative!r}"
            )


def snapshot_local_training_directory(
    path: str | pathlib.Path,
) -> TrainingDirectorySnapshot:
    """Copy and validate one RF-DETR Roboflow-COCO training directory."""
    from .artifacts import ArtifactSnapshot, _descriptor_tree_entries

    root = pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))
    source = _BoundSourceDirectory.from_absolute(root)
    snapshot = None
    try:
        source.assert_path_unchanged()
        source_entries = _descriptor_tree_entries(
            source.fd,
            display=str(root),
            max_entries=_MAX_DATASET_ENTRIES,
        )
        _reject_rfdetr_split_hardlinks(source_entries)
        digest = source.digest()
        snapshot = ArtifactSnapshot.from_directory_fd(
            source.fd,
            digest,
            display=str(root),
            max_bytes=_MAX_DATASET_BYTES,
            max_entries=_MAX_DATASET_ENTRIES,
            source_tree_validator=_reject_rfdetr_split_hardlinks,
        )
        _validate_rfdetr_coco_snapshot(snapshot.path)
        source.assert_path_unchanged()
        final_entries = _descriptor_tree_entries(
            source.fd,
            display=str(root),
            max_entries=_MAX_DATASET_ENTRIES,
        )
        if final_entries != source_entries or source.digest() != digest:
            raise RuntimeError("source training dataset changed while snapshotting")
        source.assert_path_unchanged()
    except BaseException as primary:
        cleanups: list[_Cleanup] = []
        if snapshot is not None:
            cleanups.append(("private training directory cleanup also failed", snapshot.close))
        cleanups.append(("source training directory cleanup also failed", source.close))
        _release_resources(cleanups, primary=primary)
        raise
    try:
        source.close()
    except BaseException as primary:
        _release_resources(
            (("private training directory cleanup also failed", snapshot.close),),
            primary=primary,
        )
        raise
    return TrainingDirectorySnapshot(snapshot)


__all__ = [
    "CalibrationDatasetSnapshot",
    "DatasetManifestSnapshot",
    "TrainingDatasetSnapshot",
    "TrainingDirectorySnapshot",
    "snapshot_local_calibration_dataset",
    "snapshot_local_training_directory",
    "validate_local_detection_manifest",
]
