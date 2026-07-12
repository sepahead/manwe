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
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
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
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
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
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"artifact changed while it was being hashed: {display}")
    return total


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
        expected = (
            root_metadata.st_dev,
            root_metadata.st_ino,
            root_metadata.st_size,
            root_metadata.st_mtime_ns,
        )
        actual = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
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
        return hasher.hexdigest()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"artifact is neither a regular file nor a directory: {name}")

    root_fd = os.open(name, directory_flags, dir_fd=parent_fd)
    opened_root = os.fstat(root_fd)
    if (opened_root.st_dev, opened_root.st_ino) != (
        root_metadata.st_dev,
        root_metadata.st_ino,
    ):
        os.close(root_fd)
        raise ValueError(f"artifact directory was replaced while it was opened: {name}")

    hasher.update(TREE_HASH_DOMAIN)
    entry_count = 0
    total_bytes = 0
    regular_files = 0

    def walk(directory_fd: int, prefix: str) -> None:
        nonlocal entry_count, total_bytes, regular_files
        before_directory = os.fstat(directory_fd)
        names_before = sorted(os.listdir(directory_fd))
        for child_name in names_before:
            entry_count += 1
            if entry_count > max_entries:
                raise ValueError(
                    f"artifact directory exceeds the {max_entries}-entry safety limit: {name}"
                )
            relative = f"{prefix}/{child_name}" if prefix else child_name
            metadata = os.stat(child_name, dir_fd=directory_fd, follow_symlinks=False)
            encoded = relative.encode("utf-8")
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact directory contains a symbolic link: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                hasher.update(b"D")
                _hash_field(hasher, encoded)
                child_fd = os.open(child_name, directory_flags, dir_fd=directory_fd)
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    os.close(child_fd)
                    raise ValueError(
                        f"artifact directory entry was replaced while opening: {relative}"
                    )
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"artifact directory contains an unsupported entry: {relative}")
            if metadata.st_size > max_bytes - total_bytes:
                raise ValueError(
                    f"artifact directory exceeds the {max_bytes}-byte safety limit: {name}"
                )
            hasher.update(b"F")
            _hash_field(hasher, encoded)
            _hash_field(hasher, str(metadata.st_size).encode("ascii"))
            child_fd = os.open(child_name, file_flags, dir_fd=directory_fd)
            opened = os.fstat(child_fd)
            expected = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            actual = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            if actual != expected:
                os.close(child_fd)
                raise ValueError(f"artifact entry was replaced while opening: {relative}")
            total_bytes += _update_from_open_regular_fd(
                hasher,
                child_fd,
                relative,
                max_bytes=max_bytes - total_bytes,
                require_nonempty=False,
            )
            regular_files += 1
        names_after = sorted(os.listdir(directory_fd))
        after_directory = os.fstat(directory_fd)
        if names_before != names_after or (
            before_directory.st_dev,
            before_directory.st_ino,
            before_directory.st_mtime_ns,
        ) != (
            after_directory.st_dev,
            after_directory.st_ino,
            after_directory.st_mtime_ns,
        ):
            raise ValueError(
                f"artifact directory changed while it was being hashed: {prefix or name}"
            )

    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    if regular_files == 0:
        raise ValueError(f"artifact directory contains no regular files: {name}")
    return hasher.hexdigest()


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
    "verify_artifact",
    "require_pickle_acknowledgement",
]
