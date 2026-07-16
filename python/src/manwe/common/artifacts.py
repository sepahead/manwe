"""Fail-closed helpers for hashing and validating local model artifacts."""

from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import tempfile
import warnings
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import NamedTuple

from .config_io import open_directory_nofollow, open_regular_nofollow
from .fd_io import attach_cleanup_failure, owned_binary_reader, owned_binary_writer

TREE_HASH_DOMAIN = b"manwe-directory-tree-sha256-v1\0"
DEFAULT_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_ENTRIES = 100_000
_MAX_DESCRIPTOR_TREE_DEPTH = 256
_MAX_DESCRIPTOR_TREE_RELATIVE_PATH_BYTES = 64 * 1024 * 1024
# Authenticated reopens cost O(path depth); this caps their aggregate component
# count so a legal 100k-entry tree cannot amplify one pass into tens of millions
# of stat/open operations.
_MAX_DESCRIPTOR_TREE_COMPONENT_WORK = 2_000_000
_MAX_PRIVATE_TREE_DEPTH = _MAX_DESCRIPTOR_TREE_DEPTH + 1
# A private snapshot adds one "artifact" component to every source entry plus
# the wrapper directory itself. Since entry_count <= source component work,
# W_private <= W_source + entry_count + 1 <= 2 * W_source + 1.
_MAX_PRIVATE_TREE_COMPONENT_WORK = 2 * _MAX_DESCRIPTOR_TREE_COMPONENT_WORK + 1
_MAX_PRIVATE_TREE_RELATIVE_PATH_BYTES = 128 * 1024 * 1024

_DescriptorIdentity = tuple[int, int, int, int, int]


class _DescriptorTreeEntry(NamedTuple):
    relative: str
    is_directory: bool
    identity: _DescriptorIdentity


def _absolute_artifact_path(path: str | pathlib.Path) -> pathlib.Path:
    """Expand and anchor one caller path against exactly one cwd observation."""
    expanded = pathlib.Path(path).expanduser()
    return pathlib.Path(os.path.abspath(os.fspath(expanded)))


def _close_descriptors(
    descriptors: tuple[tuple[int, str], ...],
    *,
    primary: BaseException | None = None,
) -> None:
    cleanup_error: BaseException | None = None
    for fd, label in descriptors:
        if fd < 0:
            continue
        try:
            os.close(fd)
        except BaseException as exc:
            if primary is not None:
                attach_cleanup_failure(primary, exc, label)
            elif cleanup_error is None:
                cleanup_error = exc
            else:
                attach_cleanup_failure(cleanup_error, exc, label)
    if primary is None and cleanup_error is not None:
        raise cleanup_error.with_traceback(cleanup_error.__traceback__)


@contextmanager
def _owned_descriptor(fd: int, label: str) -> Iterator[int]:
    primary: BaseException | None = None
    try:
        yield fd
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_descriptors(((fd, label),), primary=primary)


def normalize_sha256(value: str) -> str:
    """Validate and normalize an expected SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not all(char in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _hash_field(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _update_from_open_regular_fd(
    hasher, fd: int, display: str, *, max_bytes: int, require_nonempty: bool
) -> int:
    """Hash one already anchored regular-file descriptor and close it."""
    handle = owned_binary_reader(fd)
    with handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact entry is not a regular file: {display}")
        if require_nonempty and before.st_size == 0:
            raise ValueError(f"artifact file is empty: {display}")
        if before.st_size > max_bytes:
            raise ValueError(f"artifact exceeds the {max_bytes}-byte safety limit: {display}")
        total = 0
        while True:
            chunk = handle.read(min(1 << 20, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"artifact exceeded the {max_bytes}-byte safety limit while reading: {display}"
                )
            hasher.update(chunk)
        after = os.fstat(handle.fileno())
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
    if before_identity != after_identity:
        raise ValueError(f"artifact changed while it was being hashed: {display}")
    return total


def _descriptor_tree_entries(
    directory_fd: int,
    *,
    display: str,
    max_entries: int,
    max_depth: int = _MAX_DESCRIPTOR_TREE_DEPTH,
    max_relative_path_bytes: int = _MAX_DESCRIPTOR_TREE_RELATIVE_PATH_BYTES,
    max_component_work: int = _MAX_DESCRIPTOR_TREE_COMPONENT_WORK,
    allow_nonregular_leaves: bool = False,
) -> list[_DescriptorTreeEntry]:
    """Stream a bounded inventory without retaining one descriptor per depth."""
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    if type(max_depth) is not int or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")
    if type(max_relative_path_bytes) is not int or max_relative_path_bytes <= 0:
        raise ValueError("max_relative_path_bytes must be a positive integer")
    if type(max_component_work) is not int or max_component_work <= 0:
        raise ValueError("max_component_work must be a positive integer")
    if type(allow_nonregular_leaves) is not bool:
        raise TypeError("allow_nonregular_leaves must be a boolean")
    root_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError("directory_fd must refer to a directory")
    root_identity = _descriptor_entry_identity(root_before)
    entries: list[_DescriptorTreeEntry] = []
    pending: list[tuple[str, int, _DescriptorIdentity]] = [("", 0, root_identity)]
    relative_path_bytes = 0
    component_work = 0

    while pending:
        prefix, depth, expected_directory = pending.pop()
        if prefix:
            current_fd, opened_metadata = _open_descriptor_entry(
                directory_fd,
                prefix,
                expect_directory=True,
            )
        else:
            current_fd = os.dup(directory_fd)
            opened_metadata = root_before
        with _owned_descriptor(
            current_fd,
            "artifact inventory directory descriptor cleanup failed",
        ) as bound_current_fd:
            if _descriptor_entry_identity(opened_metadata) != expected_directory:
                raise ValueError(f"artifact directory entry changed while inventorying: {prefix}")
            before_directory = os.fstat(bound_current_fd)
            if _descriptor_entry_identity(before_directory) != expected_directory:
                raise ValueError(
                    f"artifact directory changed while inventorying: {prefix or display}"
                )
            with os.scandir(bound_current_fd) as iterator:
                for child in iterator:
                    if len(entries) >= max_entries:
                        raise ValueError(
                            "artifact directory exceeds the "
                            f"{max_entries}-entry safety limit: {display}"
                        )
                    child_name = child.name
                    if not isinstance(child_name, str):
                        raise ValueError(
                            f"artifact directory contains a non-text entry name: {display}"
                        )
                    child_component_count = depth + 1
                    if child_component_count > max_component_work - component_work:
                        raise ValueError(
                            "artifact directory exceeds the "
                            f"{max_component_work}-component aggregate "
                            f"traversal-work safety limit: {display}"
                        )
                    component_work += child_component_count
                    relative = f"{prefix}/{child_name}" if prefix else child_name
                    try:
                        relative_size = len(relative.encode("utf-8"))
                    except UnicodeEncodeError as exc:
                        raise ValueError(
                            f"artifact directory path is not valid UTF-8: {relative!r}"
                        ) from exc
                    if relative_size > max_relative_path_bytes - relative_path_bytes:
                        raise ValueError(
                            "artifact directory exceeds the "
                            f"{max_relative_path_bytes}-byte aggregate "
                            f"relative-path safety limit: {display}"
                        )
                    relative_path_bytes += relative_size
                    metadata = child.stat(follow_symlinks=False)
                    identity = _descriptor_entry_identity(metadata)
                    if stat.S_ISLNK(metadata.st_mode):
                        if allow_nonregular_leaves:
                            entries.append(_DescriptorTreeEntry(relative, False, identity))
                            continue
                        raise ValueError(f"artifact directory contains a symbolic link: {relative}")
                    if stat.S_ISDIR(metadata.st_mode):
                        child_depth = depth + 1
                        entries.append(_DescriptorTreeEntry(relative, True, identity))
                        if child_depth > max_depth:
                            raise ValueError(
                                "artifact directory exceeds the "
                                f"{max_depth}-level depth safety limit: {display}"
                            )
                        pending.append((relative, child_depth, identity))
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        if allow_nonregular_leaves:
                            entries.append(_DescriptorTreeEntry(relative, False, identity))
                            continue
                        raise ValueError(
                            f"artifact directory contains an unsupported entry: {relative}"
                        )
                    entries.append(_DescriptorTreeEntry(relative, False, identity))
            after_directory = os.fstat(bound_current_fd)
            if _descriptor_entry_identity(before_directory) != _descriptor_entry_identity(
                after_directory
            ):
                raise ValueError(
                    f"artifact directory changed while inventorying: {prefix or display}"
                )
    root_after = os.fstat(directory_fd)
    if root_identity != _descriptor_entry_identity(root_after):
        raise ValueError(f"artifact directory changed while inventorying: {display}")
    entries.sort(key=lambda item: item.relative)
    return entries


def _descriptor_entry_identity(metadata: os.stat_result) -> _DescriptorIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_descriptor_entry(
    root_fd: int,
    relative: str,
    *,
    expect_directory: bool,
) -> tuple[int, os.stat_result]:
    """Open an inventoried entry without leaving the authenticated root descriptor."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | cloexec | getattr(os, "O_DIRECTORY", 0) | nofollow
    file_flags = os.O_RDONLY | cloexec | nofollow | getattr(os, "O_NONBLOCK", 0)
    components = relative.split("/")
    parent_fd = os.dup(root_fd)
    fd = -1
    operation_error: BaseException | None = None
    try:
        for component in components[:-1]:
            metadata = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"artifact directory entry changed type: {relative}")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            try:
                opened = os.fstat(next_fd)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise ValueError(
                        f"artifact directory entry was replaced while opening: {relative}"
                    )
            except BaseException as error:
                _close_descriptors(
                    ((next_fd, "artifact directory descriptor cleanup failed"),),
                    primary=error,
                )
                raise
            previous_fd = parent_fd
            parent_fd = next_fd
            try:
                os.close(previous_fd)
            except OSError as exc:
                raise RuntimeError("artifact directory descriptor could not be released") from exc
        leaf = components[-1]
        metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        expected_mode = stat.S_ISDIR if expect_directory else stat.S_ISREG
        if not expected_mode(metadata.st_mode):
            raise ValueError(f"artifact directory entry changed type: {relative}")
        fd = os.open(
            leaf,
            directory_flags if expect_directory else file_flags,
            dir_fd=parent_fd,
        )
        opened = os.fstat(fd)
        if _descriptor_entry_identity(opened) != _descriptor_entry_identity(metadata):
            raise ValueError(f"artifact directory entry was replaced while opening: {relative}")
        previous_fd = parent_fd
        parent_fd = -1
        os.close(previous_fd)
        result = fd
        fd = -1
        return result, metadata
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        _close_descriptors(
            (
                (fd, "artifact leaf descriptor cleanup failed"),
                (parent_fd, "artifact parent descriptor cleanup failed"),
            ),
            primary=operation_error,
        )


def _open_descriptor_parent(root_fd: int, relative: str) -> tuple[int, str]:
    """Open an entry's parent with a constant number of live descriptors."""
    parent_relative, separator, leaf = relative.rpartition("/")
    if not separator:
        return os.dup(root_fd), relative
    parent_fd, _ = _open_descriptor_entry(
        root_fd,
        parent_relative,
        expect_directory=True,
    )
    return parent_fd, leaf


def _tree_entries(
    path: pathlib.Path,
    max_entries: int,
) -> list[tuple[pathlib.Path, str, os.stat_result]]:
    """Expose a descriptor-bound path view for internal dataset consumers."""
    root = _absolute_artifact_path(path)
    root_fd = open_directory_nofollow(root, "artifact directory")
    with _owned_descriptor(
        root_fd,
        "artifact directory descriptor cleanup failed",
    ) as bound_root_fd:
        compact_entries = _descriptor_tree_entries(
            bound_root_fd,
            display=str(root),
            max_entries=max_entries,
        )
        result: list[tuple[pathlib.Path, str, os.stat_result]] = []
        for entry in compact_entries:
            fd, metadata = _open_descriptor_entry(
                bound_root_fd,
                entry.relative,
                expect_directory=entry.is_directory,
            )
            with _owned_descriptor(
                fd,
                "artifact entry descriptor cleanup failed",
            ):
                if _descriptor_entry_identity(metadata) != entry.identity:
                    raise ValueError(
                        f"artifact directory entry changed while inventorying: {entry.relative}"
                    )
            result.append(
                (
                    root / pathlib.PurePosixPath(entry.relative),
                    "directory" if entry.is_directory else "file",
                    metadata,
                )
            )
        after_entries = _descriptor_tree_entries(
            bound_root_fd,
            display=str(root),
            max_entries=max_entries,
        )
        if compact_entries != after_entries:
            raise ValueError(f"artifact directory changed while inventorying: {root}")
        return result


def sha256_directory_fd(
    directory_fd: int,
    *,
    display: str = "artifact directory",
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_entries: int = DEFAULT_MAX_ARTIFACT_ENTRIES,
) -> str:
    """Hash an authenticated tree with the canonical global path ordering."""
    if isinstance(directory_fd, bool) or not isinstance(directory_fd, int) or directory_fd < 0:
        raise ValueError("directory_fd must be an open directory descriptor")
    if not isinstance(display, str) or not display:
        raise ValueError("display must be a nonempty string")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")

    entries = _descriptor_tree_entries(
        directory_fd,
        display=display,
        max_entries=max_entries,
    )
    hasher = hashlib.sha256(TREE_HASH_DOMAIN)
    total_bytes = 0
    regular_files = 0
    for entry in entries:
        relative = entry.relative
        encoded = relative.encode("utf-8")
        if entry.is_directory:
            hasher.update(b"D")
            _hash_field(hasher, encoded)
            fd, metadata = _open_descriptor_entry(
                directory_fd,
                relative,
                expect_directory=True,
            )
            with _owned_descriptor(
                fd,
                "artifact hash directory descriptor cleanup failed",
            ):
                if _descriptor_entry_identity(metadata) != entry.identity:
                    raise ValueError(f"artifact directory entry changed while hashing: {relative}")
            continue
        expected_size = entry.identity[2]
        if expected_size > max_bytes - total_bytes:
            raise ValueError(
                f"artifact directory exceeds the {max_bytes}-byte safety limit: {display}"
            )
        hasher.update(b"F")
        _hash_field(hasher, encoded)
        _hash_field(hasher, str(expected_size).encode("ascii"))
        fd, metadata = _open_descriptor_entry(
            directory_fd,
            relative,
            expect_directory=False,
        )
        if _descriptor_entry_identity(metadata) != entry.identity:
            error = ValueError(f"artifact directory entry changed while hashing: {relative}")
            _close_descriptors(
                ((fd, "artifact hash file descriptor cleanup failed"),),
                primary=error,
            )
            raise error
        total_bytes += _update_from_open_regular_fd(
            hasher,
            fd,
            relative,
            max_bytes=max_bytes - total_bytes,
            require_nonempty=False,
        )
        regular_files += 1
    if regular_files == 0:
        raise ValueError(f"artifact directory contains no regular files: {display}")
    after_entries = _descriptor_tree_entries(
        directory_fd,
        display=display,
        max_entries=max_entries,
    )
    if entries != after_entries:
        raise ValueError(f"artifact directory changed while it was being hashed: {display}")
    return hasher.hexdigest()


def _copy_directory_fd(
    source_fd: int,
    destination_fd: int,
    *,
    display: str,
    max_bytes: int,
    max_entries: int,
    destination_directory_mode: int = 0o500,
    destination_file_mode: int = 0o400,
) -> None:
    """Copy an authenticated tree with live descriptors bounded independently of depth."""
    for value, name in (
        (destination_directory_mode, "destination_directory_mode"),
        (destination_file_mode, "destination_file_mode"),
    ):
        if type(value) is not int or not 0 <= value <= 0o777:
            raise ValueError(f"{name} must be an integer permission mode in [0, 0o777]")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | cloexec | getattr(os, "O_DIRECTORY", 0) | nofollow
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | cloexec | nofollow
    entries = _descriptor_tree_entries(
        source_fd,
        display=display,
        max_entries=max_entries,
    )
    total_bytes = 0
    regular_files = sum(not entry.is_directory for entry in entries)
    if regular_files == 0:
        raise ValueError(f"artifact directory contains no regular files: {display}")
    for entry in entries:
        if entry.is_directory:
            continue
        size = entry.identity[2]
        if size > max_bytes - total_bytes:
            raise ValueError(
                f"artifact directory exceeds the {max_bytes}-byte safety limit: {display}"
            )
        total_bytes += size

    for entry in entries:
        relative = entry.relative
        if entry.is_directory:
            source_directory_fd, observed = _open_descriptor_entry(
                source_fd,
                relative,
                expect_directory=True,
            )
            with _owned_descriptor(
                source_directory_fd,
                "artifact source directory descriptor cleanup failed",
            ):
                if _descriptor_entry_identity(observed) != entry.identity:
                    raise ValueError(f"artifact directory entry changed while copying: {relative}")
            destination_parent_fd, leaf = _open_descriptor_parent(destination_fd, relative)
            with _owned_descriptor(
                destination_parent_fd,
                "artifact destination parent descriptor cleanup failed",
            ) as bound_destination_parent_fd:
                os.mkdir(leaf, mode=0o700, dir_fd=destination_parent_fd)
                created = os.stat(
                    leaf,
                    dir_fd=bound_destination_parent_fd,
                    follow_symlinks=False,
                )
                created_fd = os.open(
                    leaf,
                    directory_flags,
                    dir_fd=bound_destination_parent_fd,
                )
                with _owned_descriptor(
                    created_fd,
                    "artifact created directory descriptor cleanup failed",
                ) as bound_created_fd:
                    if _descriptor_entry_identity(os.fstat(bound_created_fd)) != (
                        _descriptor_entry_identity(created)
                    ):
                        raise RuntimeError(
                            f"artifact destination was replaced while opening: {relative}"
                        )
            continue

        source_file_fd, observed = _open_descriptor_entry(
            source_fd,
            relative,
            expect_directory=False,
        )
        if _descriptor_entry_identity(observed) != entry.identity:
            error = ValueError(f"artifact directory entry changed while copying: {relative}")
            _close_descriptors(
                ((source_file_fd, "artifact source file descriptor cleanup failed"),),
                primary=error,
            )
            raise error
        owned_source_fd = source_file_fd
        source_file_fd = -1
        source_handle = owned_binary_reader(owned_source_fd)
        copied = 0
        with source_handle:
            destination_parent_fd, leaf = _open_descriptor_parent(destination_fd, relative)
            with _owned_descriptor(
                destination_parent_fd,
                "artifact destination parent descriptor cleanup failed",
            ) as bound_destination_parent_fd:
                destination_file_fd = os.open(
                    leaf,
                    destination_flags,
                    0o600,
                    dir_fd=bound_destination_parent_fd,
                )
                owned_destination_fd = destination_file_fd
                destination_file_fd = -1
                destination_handle = owned_binary_writer(owned_destination_fd)
                with destination_handle:
                    expected_size = entry.identity[2]
                    while True:
                        chunk = source_handle.read(min(1 << 20, expected_size - copied + 1))
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > expected_size:
                            raise ValueError(
                                f"artifact changed while it was being copied: {relative}"
                            )
                        destination_handle.write(chunk)
                    destination_handle.flush()
                    os.fchmod(destination_handle.fileno(), destination_file_mode)
                    os.fsync(destination_handle.fileno())
                    source_after = os.fstat(source_handle.fileno())
        if (
            _descriptor_entry_identity(source_after) != entry.identity
            or copied != entry.identity[2]
        ):
            raise ValueError(f"artifact changed while it was being copied: {relative}")

    after_entries = _descriptor_tree_entries(
        source_fd,
        display=display,
        max_entries=max_entries,
    )
    if entries != after_entries:
        raise ValueError(f"artifact directory changed while it was being copied: {display}")

    for entry in reversed(entries):
        if not entry.is_directory:
            continue
        created_fd, _ = _open_descriptor_entry(
            destination_fd,
            entry.relative,
            expect_directory=True,
        )
        with _owned_descriptor(
            created_fd,
            "artifact destination directory descriptor cleanup failed",
        ) as bound_created_fd:
            os.fchmod(bound_created_fd, destination_directory_mode)
            os.fsync(bound_created_fd)


def sha256_artifact_at(
    parent_fd: int,
    name: str,
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_entries: int = DEFAULT_MAX_ARTIFACT_ENTRIES,
) -> str:
    """Hash one entry relative to a retained parent-directory descriptor.

    Directory traversal opens every child relative to an already authenticated
    descriptor. A renamed or substituted pathname therefore cannot redirect the
    digest to a different tree.
    """
    if isinstance(parent_fd, bool) or not isinstance(parent_fd, int) or parent_fd < 0:
        raise ValueError("parent_fd must be an open directory descriptor")
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise ValueError("artifact name must be one nonempty path component")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | cloexec | getattr(os, "O_DIRECTORY", 0) | nofollow
    # O_NONBLOCK is inert for regular-file reads but prevents a TOCTOU swap-to-FIFO
    # between the stat above and os.open below from blocking this bounded hash.
    file_flags = os.O_RDONLY | cloexec | nofollow | getattr(os, "O_NONBLOCK", 0)
    root_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError(f"artifact root must not be a symbolic link: {name}")

    hasher = hashlib.sha256()
    if stat.S_ISREG(root_metadata.st_mode):
        fd = os.open(name, file_flags, dir_fd=parent_fd)
        operation_error: BaseException | None = None
        try:
            opened = os.fstat(fd)
            expected = _descriptor_entry_identity(root_metadata)
            actual = _descriptor_entry_identity(opened)
            if actual != expected:
                raise ValueError(f"artifact was replaced while it was being opened: {name}")
            owned_fd = fd
            fd = -1
            _update_from_open_regular_fd(
                hasher,
                owned_fd,
                name,
                max_bytes=max_bytes,
                require_nonempty=True,
            )
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            _close_descriptors(
                ((fd, "artifact hash file descriptor cleanup failed"),),
                primary=operation_error,
            )
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _descriptor_entry_identity(current) != expected:
            raise ValueError(f"artifact was replaced while it was being hashed: {name}")
        return hasher.hexdigest()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"artifact is neither a regular file nor a directory: {name}")

    root_fd = os.open(name, directory_flags, dir_fd=parent_fd)
    with _owned_descriptor(
        root_fd,
        "artifact hash root directory descriptor cleanup failed",
    ) as bound_root_fd:
        opened_root = os.fstat(bound_root_fd)
        expected_root = (
            root_metadata.st_dev,
            root_metadata.st_ino,
            root_metadata.st_mtime_ns,
            root_metadata.st_ctime_ns,
        )
        if (
            opened_root.st_dev,
            opened_root.st_ino,
            opened_root.st_mtime_ns,
            opened_root.st_ctime_ns,
        ) != expected_root:
            raise ValueError(f"artifact directory was replaced while it was opened: {name}")
        digest = sha256_directory_fd(
            bound_root_fd,
            display=name,
            max_bytes=max_bytes,
            max_entries=max_entries,
        )
    current_root = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        current_root.st_dev,
        current_root.st_ino,
        current_root.st_mtime_ns,
        current_root.st_ctime_ns,
    ) != expected_root:
        raise ValueError(f"artifact directory was replaced while it was being hashed: {name}")
    return digest


def sha256_artifact(
    path: str | pathlib.Path,
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_entries: int = DEFAULT_MAX_ARTIFACT_ENTRIES,
) -> str:
    """Hash one bounded regular file or symlink-free directory bundle.

    Files use the conventional SHA-256 of their bytes. Directories use a stable,
    domain-separated tree digest over entry paths, kinds, sizes, and file bytes.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    artifact = _absolute_artifact_path(path)
    try:
        if artifact.name:
            parent_fd = open_directory_nofollow(artifact.parent, "artifact parent")
            with _owned_descriptor(parent_fd, "artifact parent descriptor cleanup failed"):
                return sha256_artifact_at(
                    parent_fd,
                    artifact.name,
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
        root_fd = open_directory_nofollow(artifact, "artifact directory")
        with _owned_descriptor(root_fd, "artifact root descriptor cleanup failed"):
            return sha256_directory_fd(
                root_fd,
                display=str(artifact),
                max_bytes=max_bytes,
                max_entries=max_entries,
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"artifact does not exist: {artifact}") from exc


def verify_artifact(
    path: str | pathlib.Path,
    expected_sha256: str,
    *,
    allowed_suffixes: Collection[str] | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> pathlib.Path:
    """Validate type, suffix, bounded content, and exact digest for an artifact."""
    artifact = _absolute_artifact_path(path)
    if allowed_suffixes is not None:
        normalized = {suffix.lower() for suffix in allowed_suffixes}
        if artifact.suffix.lower() not in normalized:
            raise ValueError(
                f"artifact suffix {artifact.suffix or '<none>'!r} is not one of "
                f"{sorted(normalized)}"
            )
    expected = normalize_sha256(expected_sha256)
    actual = sha256_artifact(artifact, max_bytes=max_bytes)
    if actual != expected:
        raise ValueError(f"artifact SHA-256 does not match the expected digest: {artifact}")
    return artifact


def require_pickle_acknowledgement(path: pathlib.Path, allowed: bool) -> None:
    """Require an explicit trust decision before a pickle-backed model can load."""
    if type(allowed) is not bool:
        raise TypeError("pickle acknowledgement must be a boolean")
    if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle"} and not allowed:
        raise ValueError(
            "pickle-backed checkpoints can execute code while loading; pass "
            "allow_pickle_checkpoint=True only for an artifact whose digest and origin you trust"
        )


def _restore_mode_and_close(
    fd: int,
    *,
    restore_mode: int | None,
    primary: BaseException | None,
) -> None:
    cleanup_error: BaseException | None = None
    if restore_mode is not None:
        try:
            os.fchmod(fd, restore_mode)
        except BaseException as exc:
            cleanup_error = exc
    try:
        os.close(fd)
    except BaseException as exc:
        if cleanup_error is None:
            cleanup_error = exc
        else:
            attach_cleanup_failure(
                cleanup_error,
                exc,
                "private artifact snapshot parent descriptor cleanup failed",
            )
    if cleanup_error is None:
        return
    if primary is not None:
        attach_cleanup_failure(
            primary,
            cleanup_error,
            "private artifact snapshot parent mode restoration failed",
        )
        return
    raise cleanup_error.with_traceback(cleanup_error.__traceback__)


def _remove_private_tree_contents_fd(directory_fd: int, *, max_entries: int) -> None:
    """Remove one private tree bottom-up through a bounded, descriptor-anchored inventory."""
    entries = _descriptor_tree_entries(
        directory_fd,
        display="private artifact snapshot",
        max_entries=max_entries,
        max_depth=_MAX_PRIVATE_TREE_DEPTH,
        max_relative_path_bytes=_MAX_PRIVATE_TREE_RELATIVE_PATH_BYTES,
        max_component_work=_MAX_PRIVATE_TREE_COMPONENT_WORK,
        allow_nonregular_leaves=True,
    )
    root_metadata = os.fstat(directory_fd)
    directory_identities = {
        entry.relative: entry.identity[:2] for entry in entries if entry.is_directory
    }
    for entry in reversed(entries):
        parent_relative, separator, _ = entry.relative.rpartition("/")
        parent_fd, leaf = _open_descriptor_parent(directory_fd, entry.relative)
        operation_error: BaseException | None = None
        restore_mode: int | None = None
        try:
            parent_metadata = os.fstat(parent_fd)
            expected_parent = (
                directory_identities[parent_relative]
                if separator
                else (root_metadata.st_dev, root_metadata.st_ino)
            )
            if (
                not stat.S_ISDIR(parent_metadata.st_mode)
                or (
                    parent_metadata.st_dev,
                    parent_metadata.st_ino,
                )
                != expected_parent
            ):
                raise RuntimeError(
                    f"private artifact snapshot parent changed during cleanup: "
                    f"{parent_relative or '<root>'}"
                )
            current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            current_identity = _descriptor_entry_identity(current)
            identity_matches = (
                (current.st_dev, current.st_ino) == entry.identity[:2]
                if entry.is_directory
                else current_identity == entry.identity
            )
            kind_matches = (
                stat.S_ISDIR(current.st_mode)
                if entry.is_directory
                else not stat.S_ISDIR(current.st_mode)
            )
            if not identity_matches or not kind_matches:
                raise RuntimeError(
                    f"private artifact snapshot entry changed during cleanup: {entry.relative}"
                )
            original_mode = stat.S_IMODE(parent_metadata.st_mode)
            writable_mode = original_mode | stat.S_IWUSR | stat.S_IXUSR
            if writable_mode != original_mode:
                restore_mode = original_mode
                os.fchmod(parent_fd, writable_mode)
            if entry.is_directory:
                os.rmdir(leaf, dir_fd=parent_fd)
            else:
                os.unlink(leaf, dir_fd=parent_fd)
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            _restore_mode_and_close(
                parent_fd,
                restore_mode=restore_mode,
                primary=operation_error,
            )
    with os.scandir(directory_fd) as iterator:
        if next(iterator, None) is not None:
            raise RuntimeError("private artifact snapshot changed during cleanup")


class _PrivateSnapshotDirectory:
    """A descriptor-anchored temporary directory with bounded non-recursive cleanup."""

    def __init__(self, *, max_entries: int) -> None:
        self.name = ""
        self._max_entries = max_entries + 1
        self._root_fd = -1
        self._parent_fd = -1
        self._identity: tuple[int, int] | None = None
        self._closed = True
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        root: pathlib.Path | None = None
        root_identity: tuple[int, int] | None = None
        try:
            temporary_parent = pathlib.Path(tempfile.gettempdir()).resolve(strict=True)
            self._parent_fd = os.open(temporary_parent, directory_flags)
            root = pathlib.Path(
                tempfile.mkdtemp(
                    prefix="manwe-artifact-",
                    dir=temporary_parent,
                )
            )
            self.name = str(root)
            root_metadata = os.stat(
                root.name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
            root_identity = (root_metadata.st_dev, root_metadata.st_ino)
            self._root_fd = os.open(root.name, directory_flags, dir_fd=self._parent_fd)
            opened = os.fstat(self._root_fd)
            if (opened.st_dev, opened.st_ino) != (
                root_metadata.st_dev,
                root_metadata.st_ino,
            ):
                raise RuntimeError("private artifact snapshot root was replaced while opening")
            self._identity = (opened.st_dev, opened.st_ino)
            self._closed = False
        except BaseException as error:
            root_fd = self._root_fd
            parent_fd = self._parent_fd
            self._root_fd = -1
            self._parent_fd = -1
            if parent_fd >= 0 and root is not None and root_identity is not None:
                try:
                    visible = os.stat(
                        root.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISDIR(visible.st_mode)
                        and (
                            visible.st_dev,
                            visible.st_ino,
                        )
                        == root_identity
                    ):
                        os.rmdir(root.name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except BaseException as cleanup:
                    attach_cleanup_failure(
                        error,
                        cleanup,
                        "private artifact snapshot directory cleanup failed",
                    )
            _close_descriptors(
                (
                    (root_fd, "private artifact snapshot root descriptor cleanup failed"),
                    (parent_fd, "private artifact snapshot parent descriptor cleanup failed"),
                ),
                primary=error,
            )
            raise

    def cleanup(self) -> None:
        if self._closed:
            return
        if self._root_fd < 0 or self._parent_fd < 0 or self._identity is None:
            raise RuntimeError("private artifact snapshot cleanup state is incomplete")
        _remove_private_tree_contents_fd(
            self._root_fd,
            max_entries=self._max_entries,
        )
        self.assert_visible()
        os.rmdir(pathlib.Path(self.name).name, dir_fd=self._parent_fd)
        root_fd = self._root_fd
        parent_fd = self._parent_fd
        self._root_fd = -1
        self._parent_fd = -1
        self._closed = True
        _close_descriptors(
            (
                (root_fd, "private artifact snapshot root descriptor cleanup failed"),
                (parent_fd, "private artifact snapshot parent descriptor cleanup failed"),
            )
        )

    @property
    def fd(self) -> int:
        if self._closed or self._root_fd < 0:
            raise RuntimeError("private artifact snapshot directory is closed")
        return self._root_fd

    def assert_visible(self) -> None:
        if self._closed or self._root_fd < 0 or self._parent_fd < 0 or self._identity is None:
            raise RuntimeError("private artifact snapshot directory is closed")
        opened = os.fstat(self._root_fd)
        visible = os.stat(
            pathlib.Path(self.name).name,
            dir_fd=self._parent_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != self._identity or (
            visible.st_dev,
            visible.st_ino,
        ) != self._identity:
            raise RuntimeError("private artifact snapshot root changed")

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.cleanup()
        except BaseException as exc:
            root_fd = getattr(self, "_root_fd", -1)
            parent_fd = getattr(self, "_parent_fd", -1)
            self._root_fd = -1
            self._parent_fd = -1
            _close_descriptors(
                (
                    (root_fd, "private artifact snapshot root descriptor cleanup failed"),
                    (parent_fd, "private artifact snapshot parent descriptor cleanup failed"),
                ),
                primary=exc,
            )
            warnings.warn(
                f"implicit private artifact snapshot cleanup failed: {exc}",
                ResourceWarning,
                stacklevel=2,
            )


class ArtifactSnapshot:
    """A private verified artifact copy that remains valid until explicitly closed."""

    def __init__(
        self,
        path: str | pathlib.Path,
        expected_sha256: str,
        *,
        allowed_suffixes: Collection[str] | None = None,
        max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_entries: int = DEFAULT_MAX_ARTIFACT_ENTRIES,
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        source = _absolute_artifact_path(path)
        if allowed_suffixes is not None:
            normalized = {suffix.lower() for suffix in allowed_suffixes}
            if source.suffix.lower() not in normalized:
                raise ValueError(
                    f"artifact suffix {source.suffix or '<none>'!r} is not one of "
                    f"{sorted(normalized)}"
                )
        expected = normalize_sha256(expected_sha256)
        self._temporary = _PrivateSnapshotDirectory(max_entries=max_entries)
        root = pathlib.Path(self._temporary.name)
        destination = root / f"artifact{source.suffix.lower()}"
        try:
            try:
                destination.name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("artifact suffix must be valid UTF-8") from exc
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact root must not be a symbolic link: {source}")
            if stat.S_ISREG(metadata.st_mode):
                source_handle = open_regular_nofollow(source, "artifact file")
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                destination_fd = -1
                total = 0
                operation_error: BaseException | None = None
                try:
                    with source_handle:
                        source_before = os.fstat(source_handle.fileno())
                        if _descriptor_entry_identity(source_before) != (
                            _descriptor_entry_identity(metadata)
                        ):
                            raise ValueError(
                                f"artifact was replaced while it was being opened: {source}"
                            )
                        destination_fd = os.open(
                            destination.name,
                            flags,
                            0o400,
                            dir_fd=self._temporary.fd,
                        )
                        owned_destination_fd = destination_fd
                        destination_fd = -1
                        target = owned_binary_writer(owned_destination_fd)
                        with target:
                            while True:
                                chunk = source_handle.read(min(1 << 20, max_bytes - total + 1))
                                if not chunk:
                                    break
                                total += len(chunk)
                                if total > max_bytes:
                                    raise ValueError(
                                        f"artifact exceeds the {max_bytes}-byte safety limit: "
                                        f"{source}"
                                    )
                                target.write(chunk)
                            target.flush()
                            os.fchmod(target.fileno(), 0o400)
                            os.fsync(target.fileno())
                            source_after = os.fstat(source_handle.fileno())
                    if _descriptor_entry_identity(source_after) != (
                        _descriptor_entry_identity(source_before)
                    ):
                        raise ValueError(f"artifact changed while it was being copied: {source}")
                except BaseException as exc:
                    operation_error = exc
                    raise
                finally:
                    _close_descriptors(
                        ((destination_fd, "artifact destination descriptor cleanup failed"),),
                        primary=operation_error,
                    )
            elif stat.S_ISDIR(metadata.st_mode):
                os.mkdir(destination.name, mode=0o700, dir_fd=self._temporary.fd)
                source_fd = open_directory_nofollow(source, "artifact directory")
                with _owned_descriptor(
                    source_fd,
                    "artifact source directory descriptor cleanup failed",
                ) as bound_source_fd:
                    source_opened = os.fstat(bound_source_fd)
                    if _descriptor_entry_identity(source_opened) != (
                        _descriptor_entry_identity(metadata)
                    ):
                        raise ValueError(
                            f"artifact directory was replaced while it was opened: {source}"
                        )
                    destination_fd = os.open(
                        destination.name,
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=self._temporary.fd,
                    )
                    with _owned_descriptor(
                        destination_fd,
                        "artifact destination directory descriptor cleanup failed",
                    ) as bound_destination_fd:
                        _copy_directory_fd(
                            bound_source_fd,
                            bound_destination_fd,
                            display=str(source),
                            max_bytes=max_bytes,
                            max_entries=max_entries,
                        )
                        actual = sha256_directory_fd(
                            bound_destination_fd,
                            display=f"private snapshot of {source}",
                            max_bytes=max_bytes,
                            max_entries=max_entries,
                        )
                        os.fchmod(bound_destination_fd, 0o500)
                        os.fsync(bound_destination_fd)
            else:
                raise ValueError(f"artifact is neither a regular file nor a directory: {source}")
            if stat.S_ISREG(metadata.st_mode):
                actual = sha256_artifact_at(
                    self._temporary.fd,
                    destination.name,
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
            if actual != expected:
                raise ValueError(f"artifact SHA-256 does not match the expected digest: {source}")
            self._temporary.assert_visible()
            self.path = destination
            self.sha256 = actual
        except BaseException as error:
            try:
                self._temporary.cleanup()
            except BaseException as cleanup:
                attach_cleanup_failure(error, cleanup, "artifact snapshot cleanup failed")
            raise

    def close(self) -> None:
        self._temporary.cleanup()

    @classmethod
    def from_directory_fd(
        cls,
        directory_fd: int,
        expected_sha256: str,
        *,
        display: str = "artifact directory",
        max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_entries: int = DEFAULT_MAX_ARTIFACT_ENTRIES,
    ) -> ArtifactSnapshot:
        """Copy a verified tree from an already authenticated directory descriptor."""
        if isinstance(directory_fd, bool) or not isinstance(directory_fd, int) or directory_fd < 0:
            raise ValueError("directory_fd must be an open directory descriptor")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        expected = normalize_sha256(expected_sha256)
        instance = cls.__new__(cls)
        instance._temporary = _PrivateSnapshotDirectory(max_entries=max_entries)
        root = pathlib.Path(instance._temporary.name)
        destination = root / "artifact"
        try:
            os.mkdir(destination.name, mode=0o700, dir_fd=instance._temporary.fd)
            destination_fd = os.open(
                destination.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=instance._temporary.fd,
            )
            with _owned_descriptor(
                destination_fd,
                "artifact destination directory descriptor cleanup failed",
            ) as bound_destination_fd:
                _copy_directory_fd(
                    directory_fd,
                    bound_destination_fd,
                    display=display,
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
                actual = sha256_directory_fd(
                    bound_destination_fd,
                    display=f"private snapshot of {display}",
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
                if actual != expected:
                    raise ValueError(
                        f"artifact SHA-256 does not match the expected digest: {display}"
                    )
                os.fchmod(bound_destination_fd, 0o500)
                os.fsync(bound_destination_fd)
            instance._temporary.assert_visible()
            instance.path = destination
            instance.sha256 = actual
            return instance
        except BaseException as error:
            try:
                instance._temporary.cleanup()
            except BaseException as cleanup:
                attach_cleanup_failure(error, cleanup, "artifact snapshot cleanup failed")
            raise

    def __enter__(self) -> ArtifactSnapshot:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if _exc is None:
                raise
            attach_cleanup_failure(_exc, cleanup, "artifact snapshot cleanup failed")


__all__ = [
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_MAX_ARTIFACT_ENTRIES",
    "ArtifactSnapshot",
    "normalize_sha256",
    "sha256_artifact",
    "sha256_artifact_at",
    "sha256_directory_fd",
    "verify_artifact",
    "require_pickle_acknowledgement",
]
