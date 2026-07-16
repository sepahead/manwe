"""Fail-closed helpers for hashing and validating local model artifacts."""

from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import tempfile
from collections.abc import Collection

from .config_io import open_regular_nofollow

TREE_HASH_DOMAIN = b"manwe-directory-tree-sha256-v1\0"
DEFAULT_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_ENTRIES = 100_000
_MAX_DESCRIPTOR_TREE_DEPTH = 256


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


def _open_regular_nofollow(path: pathlib.Path):
    handle = open_regular_nofollow(path.absolute(), "artifact entry")
    return handle, os.fstat(handle.fileno())


def _update_from_file(hasher, path: pathlib.Path, *, max_bytes: int, require_nonempty: bool) -> int:
    handle, before = _open_regular_nofollow(path)
    with handle:
        if require_nonempty and before.st_size == 0:
            raise ValueError(f"artifact file is empty: {path}")
        if before.st_size > max_bytes:
            raise ValueError(f"artifact exceeds the {max_bytes}-byte safety limit: {path}")
        total_read = 0
        while True:
            chunk = handle.read(min(1 << 20, max_bytes - total_read + 1))
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_bytes:
                raise ValueError(
                    f"artifact exceeded the {max_bytes}-byte safety limit while reading: {path}"
                )
            hasher.update(chunk)
        after = os.fstat(handle.fileno())
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
        raise ValueError(f"artifact changed while it was being hashed: {path}")
    return total_read


def _copy_regular_bounded(source: pathlib.Path, destination: pathlib.Path, max_bytes: int) -> int:
    source_handle, before = _open_regular_nofollow(source)
    if before.st_size > max_bytes:
        source_handle.close()
        raise ValueError(f"artifact directory exceeds the byte safety limit: {source}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_fd = os.open(destination, flags, 0o400)
    total = 0
    try:
        with source_handle, os.fdopen(destination_fd, "wb") as destination_handle:
            destination_fd = -1
            while True:
                chunk = source_handle.read(min(1 << 20, max_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"artifact directory exceeds the byte safety limit: {source}")
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
            after = os.fstat(source_handle.fileno())
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
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
        raise ValueError(f"artifact changed while it was being copied: {source}")
    return total


def _tree_entries(
    path: pathlib.Path, max_entries: int
) -> list[tuple[pathlib.Path, str, os.stat_result]]:
    entries: list[tuple[pathlib.Path, str, os.stat_result]] = []
    for entry in path.rglob("*"):
        if len(entries) >= max_entries:
            raise ValueError(
                f"artifact directory exceeds the {max_entries}-entry safety limit: {path}"
            )
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"artifact directory contains a symbolic link: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        else:
            raise ValueError(f"artifact directory contains an unsupported entry: {entry}")
        entries.append((entry, kind, metadata))
    return sorted(entries, key=lambda item: item[0].relative_to(path).as_posix())


def _update_from_open_regular_fd(
    hasher, fd: int, display: str, *, max_bytes: int, require_nonempty: bool
) -> int:
    """Hash one already anchored regular-file descriptor and close it."""
    with os.fdopen(fd, "rb") as handle:
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
) -> list[tuple[str, str, os.stat_result]]:
    """Inventory one authenticated tree and reject mutation during traversal."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | cloexec | getattr(os, "O_DIRECTORY", 0) | nofollow
    root_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError("directory_fd must refer to a directory")
    entries: list[tuple[str, str, os.stat_result]] = []

    def walk(current_fd: int, prefix: str, depth: int) -> None:
        before_directory = os.fstat(current_fd)
        names_before = sorted(os.listdir(current_fd))
        for child_name in names_before:
            if len(entries) >= max_entries:
                raise ValueError(
                    f"artifact directory exceeds the {max_entries}-entry safety limit: {display}"
                )
            relative = f"{prefix}/{child_name}" if prefix else child_name
            metadata = os.stat(child_name, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact directory contains a symbolic link: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                entries.append((relative, "directory", metadata))
                if depth >= _MAX_DESCRIPTOR_TREE_DEPTH:
                    raise ValueError(
                        "artifact directory exceeds the "
                        f"{_MAX_DESCRIPTOR_TREE_DEPTH}-level depth safety limit: {display}"
                    )
                child_fd = os.open(child_name, directory_flags, dir_fd=current_fd)
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    os.close(child_fd)
                    raise ValueError(
                        f"artifact directory entry was replaced while opening: {relative}"
                    )
                try:
                    walk(child_fd, relative, depth + 1)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"artifact directory contains an unsupported entry: {relative}")
            entries.append((relative, "file", metadata))
        names_after = sorted(os.listdir(current_fd))
        after_directory = os.fstat(current_fd)
        before_identity = (
            before_directory.st_dev,
            before_directory.st_ino,
            before_directory.st_mtime_ns,
            before_directory.st_ctime_ns,
        )
        after_identity = (
            after_directory.st_dev,
            after_directory.st_ino,
            after_directory.st_mtime_ns,
            after_directory.st_ctime_ns,
        )
        if names_before != names_after or before_identity != after_identity:
            raise ValueError(
                f"artifact directory changed while it was being hashed: {prefix or display}"
            )

    walk(directory_fd, "", 0)
    root_after = os.fstat(directory_fd)
    if (
        root_before.st_dev,
        root_before.st_ino,
        root_before.st_mtime_ns,
        root_before.st_ctime_ns,
    ) != (
        root_after.st_dev,
        root_after.st_ino,
        root_after.st_mtime_ns,
        root_after.st_ctime_ns,
    ):
        raise ValueError(f"artifact directory changed while it was being hashed: {display}")
    return sorted(entries, key=lambda item: item[0])


def _descriptor_entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
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
    try:
        for component in components[:-1]:
            metadata = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"artifact directory entry changed type: {relative}")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened = os.fstat(next_fd)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(next_fd)
                raise ValueError(f"artifact directory entry was replaced while opening: {relative}")
            os.close(parent_fd)
            parent_fd = next_fd
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
            os.close(fd)
            raise ValueError(f"artifact directory entry was replaced while opening: {relative}")
        return fd, metadata
    finally:
        os.close(parent_fd)


def _descriptor_tree_signature(
    entries: list[tuple[str, str, os.stat_result]],
) -> list[tuple[str, str, int, int, int, int, int]]:
    return [
        (relative, kind, *_descriptor_entry_identity(metadata))
        for relative, kind, metadata in entries
    ]


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
    for relative, kind, expected_metadata in entries:
        encoded = relative.encode("utf-8")
        if kind == "directory":
            hasher.update(b"D")
            _hash_field(hasher, encoded)
            fd, metadata = _open_descriptor_entry(
                directory_fd,
                relative,
                expect_directory=True,
            )
            os.close(fd)
            if _descriptor_entry_identity(metadata) != _descriptor_entry_identity(
                expected_metadata
            ):
                raise ValueError(f"artifact directory entry changed while hashing: {relative}")
            continue
        if expected_metadata.st_size > max_bytes - total_bytes:
            raise ValueError(
                f"artifact directory exceeds the {max_bytes}-byte safety limit: {display}"
            )
        hasher.update(b"F")
        _hash_field(hasher, encoded)
        _hash_field(hasher, str(expected_metadata.st_size).encode("ascii"))
        fd, metadata = _open_descriptor_entry(
            directory_fd,
            relative,
            expect_directory=False,
        )
        if _descriptor_entry_identity(metadata) != _descriptor_entry_identity(expected_metadata):
            os.close(fd)
            raise ValueError(f"artifact directory entry changed while hashing: {relative}")
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
    if _descriptor_tree_signature(entries) != _descriptor_tree_signature(after_entries):
        raise ValueError(f"artifact directory changed while it was being hashed: {display}")
    return hasher.hexdigest()


def _copy_directory_fd(
    source_fd: int,
    destination_fd: int,
    *,
    display: str,
    max_bytes: int,
    max_entries: int,
) -> None:
    """Copy one authenticated directory tree without re-walking its source path."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | cloexec | getattr(os, "O_DIRECTORY", 0) | nofollow
    source_flags = os.O_RDONLY | cloexec | nofollow | getattr(os, "O_NONBLOCK", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | cloexec | nofollow
    entry_count = 0
    total_bytes = 0
    regular_files = 0

    def copy_tree(
        current_source_fd: int,
        current_destination_fd: int,
        prefix: str,
        depth: int,
    ) -> None:
        nonlocal entry_count, total_bytes, regular_files
        source_before = os.fstat(current_source_fd)
        names_before = sorted(os.listdir(current_source_fd))
        for child_name in names_before:
            entry_count += 1
            if entry_count > max_entries:
                raise ValueError(
                    f"artifact directory exceeds the {max_entries}-entry safety limit: {display}"
                )
            relative = f"{prefix}/{child_name}" if prefix else child_name
            metadata = os.stat(child_name, dir_fd=current_source_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact directory contains a symbolic link: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                if depth >= _MAX_DESCRIPTOR_TREE_DEPTH:
                    raise ValueError(
                        "artifact directory exceeds the "
                        f"{_MAX_DESCRIPTOR_TREE_DEPTH}-level depth safety limit: {display}"
                    )
                child_source_fd = os.open(child_name, directory_flags, dir_fd=current_source_fd)
                try:
                    opened = os.fstat(child_source_fd)
                    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                        raise ValueError(
                            f"artifact directory entry was replaced while opening: {relative}"
                        )
                    os.mkdir(child_name, mode=0o700, dir_fd=current_destination_fd)
                    child_destination_fd = os.open(
                        child_name, directory_flags, dir_fd=current_destination_fd
                    )
                    try:
                        copy_tree(child_source_fd, child_destination_fd, relative, depth + 1)
                        os.fchmod(child_destination_fd, 0o500)
                        os.fsync(child_destination_fd)
                    finally:
                        os.close(child_destination_fd)
                finally:
                    os.close(child_source_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"artifact directory contains an unsupported entry: {relative}")
            if metadata.st_size > max_bytes - total_bytes:
                raise ValueError(
                    f"artifact directory exceeds the {max_bytes}-byte safety limit: {display}"
                )
            source_file_fd = os.open(child_name, source_flags, dir_fd=current_source_fd)
            destination_file_fd = -1
            copied = 0
            try:
                opened = os.fstat(source_file_fd)
                expected = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
                actual = (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                if actual != expected:
                    raise ValueError(f"artifact entry was replaced while opening: {relative}")
                destination_file_fd = os.open(
                    child_name,
                    destination_flags,
                    0o400,
                    dir_fd=current_destination_fd,
                )
                with (
                    os.fdopen(source_file_fd, "rb") as source_handle,
                    os.fdopen(destination_file_fd, "wb") as destination_handle,
                ):
                    source_file_fd = -1
                    destination_file_fd = -1
                    while True:
                        chunk = source_handle.read(
                            min(1 << 20, max_bytes - total_bytes - copied + 1)
                        )
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > max_bytes - total_bytes:
                            raise ValueError(
                                f"artifact directory exceeds the {max_bytes}-byte safety limit: "
                                f"{display}"
                            )
                        destination_handle.write(chunk)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
                    source_after = os.fstat(source_handle.fileno())
            finally:
                if source_file_fd >= 0:
                    os.close(source_file_fd)
                if destination_file_fd >= 0:
                    os.close(destination_file_fd)
            if (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            ) != expected:
                raise ValueError(f"artifact changed while it was being copied: {relative}")
            if copied != metadata.st_size:
                raise ValueError(f"artifact changed while it was being copied: {relative}")
            total_bytes += copied
            regular_files += 1
        names_after = sorted(os.listdir(current_source_fd))
        source_after = os.fstat(current_source_fd)
        before_identity = (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        )
        after_identity = (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        )
        if names_before != names_after or before_identity != after_identity:
            raise ValueError(
                f"artifact directory changed while it was being copied: {prefix or display}"
            )

    copy_tree(source_fd, destination_fd, "", 0)
    if regular_files == 0:
        raise ValueError(f"artifact directory contains no regular files: {display}")


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
        opened = os.fstat(fd)
        expected = _descriptor_entry_identity(root_metadata)
        actual = _descriptor_entry_identity(opened)
        if actual != expected:
            os.close(fd)
            raise ValueError(f"artifact was replaced while it was being opened: {name}")
        _update_from_open_regular_fd(
            hasher,
            fd,
            name,
            max_bytes=max_bytes,
            require_nonempty=True,
        )
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _descriptor_entry_identity(current) != expected:
            raise ValueError(f"artifact was replaced while it was being hashed: {name}")
        return hasher.hexdigest()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"artifact is neither a regular file nor a directory: {name}")

    root_fd = os.open(name, directory_flags, dir_fd=parent_fd)
    opened_root = os.fstat(root_fd)
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
        os.close(root_fd)
        raise ValueError(f"artifact directory was replaced while it was opened: {name}")

    try:
        digest = sha256_directory_fd(
            root_fd,
            display=name,
            max_bytes=max_bytes,
            max_entries=max_entries,
        )
    finally:
        os.close(root_fd)
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
    artifact = pathlib.Path(path)
    try:
        root_metadata = artifact.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"artifact does not exist: {artifact}") from exc
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError(f"artifact root must not be a symbolic link: {artifact}")

    hasher = hashlib.sha256()
    if stat.S_ISREG(root_metadata.st_mode):
        _update_from_file(hasher, artifact, max_bytes=max_bytes, require_nonempty=True)
        return hasher.hexdigest()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"artifact is neither a regular file nor a directory: {artifact}")

    hasher.update(TREE_HASH_DOMAIN)
    entries = _tree_entries(artifact, max_entries)
    total_bytes = 0
    regular_files = 0
    signature_before: list[tuple[str, str, int, int]] = []
    for entry, kind, metadata in entries:
        relative_text = entry.relative_to(artifact).as_posix()
        relative = relative_text.encode("utf-8")
        signature_before.append((relative_text, kind, metadata.st_size, metadata.st_mtime_ns))
        if kind == "directory":
            hasher.update(b"D")
            _hash_field(hasher, relative)
            continue
        regular_files += 1
        if metadata.st_size > max_bytes - total_bytes:
            raise ValueError(
                f"artifact directory exceeds the {max_bytes}-byte safety limit: {artifact}"
            )
        hasher.update(b"F")
        _hash_field(hasher, relative)
        _hash_field(hasher, str(metadata.st_size).encode("ascii"))
        total_bytes += _update_from_file(
            hasher,
            entry,
            max_bytes=max_bytes - total_bytes,
            require_nonempty=False,
        )
    if regular_files == 0:
        raise ValueError(f"artifact directory contains no regular files: {artifact}")

    signature_after = [
        (
            entry.relative_to(artifact).as_posix(),
            kind,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        for entry, kind, metadata in _tree_entries(artifact, max_entries)
    ]
    if signature_before != signature_after:
        raise ValueError(f"artifact directory changed while it was being hashed: {artifact}")
    return hasher.hexdigest()


def verify_artifact(
    path: str | pathlib.Path,
    expected_sha256: str,
    *,
    allowed_suffixes: Collection[str] | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> pathlib.Path:
    """Validate type, suffix, bounded content, and exact digest for an artifact."""
    artifact = pathlib.Path(path).expanduser()
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
    return artifact.absolute()


def require_pickle_acknowledgement(path: pathlib.Path, allowed: bool) -> None:
    """Require an explicit trust decision before a pickle-backed model can load."""
    if type(allowed) is not bool:
        raise TypeError("pickle acknowledgement must be a boolean")
    if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle"} and not allowed:
        raise ValueError(
            "pickle-backed checkpoints can execute code while loading; pass "
            "allow_pickle_checkpoint=True only for an artifact whose digest and origin you trust"
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
        source = pathlib.Path(path).expanduser()
        if allowed_suffixes is not None:
            normalized = {suffix.lower() for suffix in allowed_suffixes}
            if source.suffix.lower() not in normalized:
                raise ValueError(
                    f"artifact suffix {source.suffix or '<none>'!r} is not one of "
                    f"{sorted(normalized)}"
                )
        expected = normalize_sha256(expected_sha256)
        self._temporary = tempfile.TemporaryDirectory(prefix="manwe-artifact-")
        # macOS exposes /var as a compatibility symlink to /private/var. Resolve
        # the freshly created, process-owned temporary root once so subsequent
        # descriptor walks can reject every symlink component without rejecting
        # the platform's temporary-directory alias.
        root = pathlib.Path(self._temporary.name).resolve(strict=True)
        destination = root / f"artifact{source.suffix.lower()}"
        try:
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact root must not be a symbolic link: {source}")
            if stat.S_ISREG(metadata.st_mode):
                source_handle, _ = _open_regular_nofollow(source)
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                destination_fd = os.open(destination, flags, 0o400)
                total = 0
                try:
                    with source_handle, os.fdopen(destination_fd, "wb") as target:
                        destination_fd = -1
                        while True:
                            chunk = source_handle.read(min(1 << 20, max_bytes - total + 1))
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                raise ValueError(
                                    f"artifact exceeds the {max_bytes}-byte safety limit: {source}"
                                )
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                finally:
                    if destination_fd >= 0:
                        os.close(destination_fd)
            elif stat.S_ISDIR(metadata.st_mode):
                entries = _tree_entries(source, max_entries)
                destination.mkdir(mode=0o700)
                total = 0
                for entry, kind, entry_metadata in entries:
                    relative = entry.relative_to(source)
                    destination_entry = destination / relative
                    if kind == "directory":
                        destination_entry.mkdir(mode=0o700)
                        continue
                    if entry_metadata.st_size > max_bytes - total:
                        raise ValueError(
                            f"artifact directory exceeds the {max_bytes}-byte safety limit: "
                            f"{source}"
                        )
                    total += _copy_regular_bounded(entry, destination_entry, max_bytes - total)
            else:
                raise ValueError(f"artifact is neither a regular file nor a directory: {source}")
            actual = sha256_artifact(destination, max_bytes=max_bytes, max_entries=max_entries)
            if actual != expected:
                raise ValueError(f"artifact SHA-256 does not match the expected digest: {source}")
            if destination.is_dir():
                for entry in destination.rglob("*"):
                    entry.chmod(0o500 if entry.is_dir() else 0o400)
                destination.chmod(0o500)
            else:
                destination.chmod(0o400)
            self.path = destination
            self.sha256 = actual
        except BaseException:
            self._temporary.cleanup()
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
        instance._temporary = tempfile.TemporaryDirectory(prefix="manwe-artifact-")
        root = pathlib.Path(instance._temporary.name).resolve(strict=True)
        destination = root / "artifact"
        try:
            destination.mkdir(mode=0o700)
            destination_fd = os.open(
                destination,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                _copy_directory_fd(
                    directory_fd,
                    destination_fd,
                    display=display,
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
                os.fchmod(destination_fd, 0o500)
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            actual = sha256_artifact(
                destination,
                max_bytes=max_bytes,
                max_entries=max_entries,
            )
            if actual != expected:
                raise ValueError(f"artifact SHA-256 does not match the expected digest: {display}")
            instance.path = destination
            instance.sha256 = actual
            return instance
        except BaseException:
            instance._temporary.cleanup()
            raise

    def __enter__(self) -> ArtifactSnapshot:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


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
