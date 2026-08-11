"""Bounded, unambiguous loading for local UTF-8 and YAML configuration files."""

from __future__ import annotations

import ctypes
import errno
import os
import pathlib
import stat
import sys
from contextlib import suppress
from typing import Any, BinaryIO

from .deps import require
from .fd_io import attach_cleanup_failure, owned_binary_reader

MAX_YAML_TOKENS = 100_000
MAX_YAML_NESTING = 64
_DARWIN_ACL_TYPE_EXTENDED = 0x00000100
_LINUX_ACCESS_CONTROL_XATTRS = frozenset(
    {
        "security.NTACL",
        "system.nfs4_acl",
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.richacl",
    }
)


def reject_unmodeled_directory_acl(directory_fd: int, display: pathlib.Path, subject: str) -> None:
    """Reject ACLs that can grant mutation rights beyond inspected mode bits."""
    if sys.platform.startswith("linux"):
        try:
            names = os.listxattr(directory_fd)
        except AttributeError as exc:  # pragma: no cover - unsupported Python build
            raise RuntimeError(f"{subject} ACL policy could not be inspected: {display}") from exc
        except OSError as exc:
            if exc.errno in {
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }:
                return
            raise RuntimeError(f"{subject} ACL policy could not be inspected: {display}") from exc
        present = _LINUX_ACCESS_CONTROL_XATTRS.intersection(names)
        if present:
            raise PermissionError(
                f"{subject} has an access-control ACL that prevents a mode-bit trust proof: "
                f"{display} ({sorted(present)})"
            )
        return
    if sys.platform != "darwin":
        raise RuntimeError(f"{subject} ACL policy is unsupported on this platform: {display}")

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        acl_get_fd_np = libc.acl_get_fd_np
        acl_free = libc.acl_free
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"{subject} ACL policy could not be inspected: {display}") from exc
    acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_free.argtypes = [ctypes.c_void_p]
    acl_free.restype = ctypes.c_int
    ctypes.set_errno(0)
    acl = acl_get_fd_np(directory_fd, _DARWIN_ACL_TYPE_EXTENDED)
    if not acl:
        error_number = ctypes.get_errno()
        if error_number in {
            errno.ENOENT,
            getattr(errno, "ENOTSUP", errno.ENOENT),
            getattr(errno, "EOPNOTSUPP", errno.ENOENT),
        }:
            return
        operation_error = OSError(
            error_number,
            os.strerror(error_number) if error_number > 0 else "unknown ACL inspection error",
            str(display),
        )
        raise RuntimeError(
            f"{subject} ACL policy could not be inspected: {display}"
        ) from operation_error

    rejection = PermissionError(
        f"{subject} has an extended ACL that prevents a mode-bit trust proof: {display}"
    )
    ctypes.set_errno(0)
    if acl_free(acl) != 0:
        error_number = ctypes.get_errno()
        cleanup = OSError(
            error_number,
            os.strerror(error_number) if error_number > 0 else "unknown ACL cleanup error",
        )
        attach_cleanup_failure(rejection, cleanup, f"{subject} ACL cleanup failed")
    raise rejection


def require_safe_publication_directory(
    directory_fd: int,
    metadata: os.stat_result,
    display: pathlib.Path,
    subject: str,
) -> None:
    """Require a directory where another ordinary identity cannot replace entries.

    A sticky shared directory is acceptable only when it is owned by this process
    or root. Processes with the same effective UID and privileged processes remain
    outside what a POSIX pathname publication protocol can exclude.
    """
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{subject} is not a directory: {display}")
    reject_unmodeled_directory_acl(directory_fd, display, subject)
    effective_uid = os.geteuid()
    if metadata.st_uid not in {0, effective_uid}:
        raise PermissionError(f"{subject} must be owned by the effective user or root: {display}")
    if not metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return
    if not metadata.st_mode & stat.S_ISVTX:
        raise PermissionError(
            f"{subject} is group/other-writable without sticky-directory protection: {display}"
        )


def validate_local_path(
    path: pathlib.Path, subject: str, *, require_directory: bool | None
) -> None:
    """Reject missing, symlinked, and special-file components in a local path."""
    if not path.is_absolute():
        raise ValueError(f"{subject} path must be absolute after normalization")
    current = pathlib.Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{subject} path does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{subject} path chain contains a symbolic link: {current}")
    metadata = path.lstat()
    if require_directory is True and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{subject} path must be a directory: {path}")
    if require_directory is False and not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{subject} path must be a regular file: {path}")
    if require_directory is None and not (
        stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
    ):
        raise ValueError(f"{subject} path must be a regular file or directory: {path}")


def read_bounded_regular_bytes(
    path: pathlib.Path, limit: int, subject: str, *, allow_empty: bool = False
) -> bytes:
    """Read one stable regular file without following any symlink in its path."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("configuration read limit must be a positive integer")
    if type(allow_empty) is not bool:
        raise TypeError("allow_empty must be a boolean")
    with open_regular_nofollow(path, subject) as handle:
        before = os.fstat(handle.fileno())
        if (before.st_size == 0 and not allow_empty) or before.st_size > limit:
            minimum = 0 if allow_empty else 1
            raise ValueError(f"{subject} must contain {minimum}..{limit} bytes: {path}")
        value = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
    if len(value) > limit:
        raise ValueError(f"{subject} exceeds {limit} bytes: {path}")
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
    if len(value) != before.st_size or identity_before != identity_after:
        raise ValueError(f"{subject} changed while it was being read: {path}")
    return value


def read_bounded_regular_bytes_at(
    directory_fd: int,
    name: str,
    limit: int,
    subject: str,
    *,
    allow_empty: bool = False,
) -> bytes:
    """Read one stable basename relative to a retained directory descriptor."""
    if type(directory_fd) is not int or directory_fd < 0:
        raise ValueError("directory_fd must be an open directory descriptor")
    if type(name) is not str or not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise ValueError(f"{subject} name must be one non-special basename")
    if type(limit) is not int or limit <= 0:
        raise ValueError("configuration read limit must be a positive integer")
    if type(allow_empty) is not bool:
        raise TypeError("allow_empty must be a boolean")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    before_named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before_named.st_mode):
        raise ValueError(f"{subject} must be a regular file: {name}")
    fd = os.open(name, flags, dir_fd=directory_fd)
    operation_error: BaseException | None = None
    try:
        opened = os.fstat(fd)
        expected_identity = (
            before_named.st_dev,
            before_named.st_ino,
            before_named.st_size,
            before_named.st_mtime_ns,
            before_named.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != expected_identity:
            raise ValueError(f"{subject} was replaced while it was being opened: {name}")
        if (opened.st_size == 0 and not allow_empty) or opened.st_size > limit:
            minimum = 0 if allow_empty else 1
            raise ValueError(f"{subject} must contain {minimum}..{limit} bytes: {name}")
        owned_fd = fd
        fd = -1
        with owned_binary_reader(owned_fd) as handle:
            value = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
        if len(value) > limit:
            raise ValueError(f"{subject} exceeds {limit} bytes: {name}")
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if (
            len(value) != opened.st_size
            or after_identity != expected_identity
            or current_identity != expected_identity
        ):
            raise ValueError(f"{subject} changed while it was being read: {name}")
        return value
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except BaseException as cleanup:
                if operation_error is None:
                    raise
                attach_cleanup_failure(
                    operation_error,
                    cleanup,
                    f"{subject} descriptor cleanup also failed",
                )


def open_regular_nofollow(path: pathlib.Path, subject: str) -> BinaryIO:
    """Open an absolute regular file through no-follow directory descriptors."""
    if not path.is_absolute():
        raise ValueError(f"{subject} path must be absolute after normalization")
    if ".." in path.parts:
        raise ValueError(f"{subject} path must not contain parent-directory components")
    directory_fd: int | None = open_directory_nofollow(path.parent, subject)
    # O_NONBLOCK is a no-op for regular-file reads but stops a FIFO at this path
    # from blocking os.open until a writer appears — the fstat below then rejects
    # it as a non-regular file instead of hanging this fail-closed read.
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    fd: int | None = None
    try:
        try:
            fd = os.open(path.name, file_flags, dir_fd=directory_fd)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{subject} path does not exist: {path}") from exc
        except OSError as exc:
            raise ValueError(f"{subject} path is a symbolic link or special file: {path}") from exc
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{subject} path must be a regular file: {path}")
        if directory_fd is None:
            raise RuntimeError(f"{subject} parent directory descriptor was lost")
        previous_directory_fd = directory_fd
        directory_fd = None
        os.close(previous_directory_fd)
        owned_fd = fd
        fd = None
        handle = owned_binary_reader(owned_fd)
        return handle
    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        if directory_fd is not None:
            with suppress(OSError):
                os.close(directory_fd)


def open_directory_nofollow(path: pathlib.Path, subject: str) -> int:
    """Open an absolute directory through a no-follow descriptor walk.

    The caller owns the returned descriptor and must close it. Retaining it lets
    a multi-step operation remain attached to the authenticated directory even
    if a pathname is concurrently renamed or replaced.
    """
    if not path.is_absolute():
        raise ValueError(f"{subject} path must be absolute after normalization")
    if ".." in path.parts:
        raise ValueError(f"{subject} path must not contain parent-directory components")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd: int | None = None
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for component in path.parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            previous_fd = directory_fd
            directory_fd = next_fd
            try:
                os.close(previous_fd)
            except OSError as exc:
                raise RuntimeError(f"{subject} directory descriptor could not be released") from exc
        result = directory_fd
        directory_fd = None
        return result
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{subject} path does not exist: {path}") from exc
    except OSError as exc:
        raise ValueError(
            f"{subject} path chain contains a symbolic link or special component: {path}"
        ) from exc
    finally:
        if directory_fd is not None:
            with suppress(OSError):
                os.close(directory_fd)


def read_bounded_utf8_regular(path: pathlib.Path, limit: int, subject: str) -> str:
    """Read one bounded stable regular file and decode it as strict UTF-8."""
    value = read_bounded_regular_bytes(path, limit, subject)
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{subject} must be valid UTF-8: {path}") from exc


def load_unambiguous_yaml(text: str, subject: str) -> object:
    """Safely load YAML while rejecting aliases, custom tags, and duplicate keys."""
    yaml: Any = require("yaml", "config")

    forbidden_tokens = (
        yaml.tokens.AliasToken,
        yaml.tokens.AnchorToken,
        yaml.tokens.DirectiveToken,
        yaml.tokens.TagToken,
    )
    nesting_start = (
        yaml.tokens.BlockMappingStartToken,
        yaml.tokens.BlockSequenceStartToken,
        yaml.tokens.FlowMappingStartToken,
        yaml.tokens.FlowSequenceStartToken,
    )
    nesting_end = (
        yaml.tokens.BlockEndToken,
        yaml.tokens.FlowMappingEndToken,
        yaml.tokens.FlowSequenceEndToken,
    )
    try:
        depth = 0
        for token_count, token in enumerate(yaml.scan(text), start=1):
            if token_count > MAX_YAML_TOKENS:
                raise ValueError(f"{subject} exceeds the {MAX_YAML_TOKENS}-token safety limit")
            if isinstance(token, forbidden_tokens):
                raise ValueError(
                    f"{subject} must not contain YAML aliases, anchors, directives, or tags"
                )
            if isinstance(token, nesting_start):
                depth += 1
                if depth > MAX_YAML_NESTING:
                    raise ValueError(
                        f"{subject} exceeds the {MAX_YAML_NESTING}-level nesting safety limit"
                    )
            elif isinstance(token, nesting_end):
                depth = max(0, depth - 1)
    except (RecursionError, yaml.YAMLError) as exc:
        raise ValueError(f"{subject} contains invalid YAML") from exc

    class _UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        if not isinstance(node, yaml.nodes.MappingNode):
            raise ValueError(f"{subject} mapping is malformed")
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ValueError(f"{subject} mapping keys must be scalar values") from exc
            if duplicate:
                raise ValueError(f"{subject} contains a duplicate key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    loader = _UniqueKeySafeLoader(text)
    try:
        return loader.get_single_data()
    except (RecursionError, yaml.YAMLError) as exc:
        raise ValueError(f"{subject} contains invalid YAML") from exc
    finally:
        loader.dispose()


def read_strict_yaml(path: pathlib.Path, limit: int, subject: str) -> object:
    """Read and parse a bounded local YAML file with one fail-closed policy."""
    return load_unambiguous_yaml(read_bounded_utf8_regular(path, limit, subject), subject)


__all__ = [
    "load_unambiguous_yaml",
    "open_directory_nofollow",
    "open_regular_nofollow",
    "read_bounded_regular_bytes",
    "read_bounded_regular_bytes_at",
    "read_bounded_utf8_regular",
    "read_strict_yaml",
    "reject_unmodeled_directory_acl",
    "require_safe_publication_directory",
    "validate_local_path",
]
