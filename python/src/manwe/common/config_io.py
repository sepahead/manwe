"""Bounded, unambiguous loading for local UTF-8 and YAML configuration files."""

from __future__ import annotations

import os
import pathlib
import stat
from typing import BinaryIO

MAX_YAML_TOKENS = 100_000
MAX_YAML_NESTING = 64


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
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"{subject} changed while it was being read: {path}")
    return value


def open_regular_nofollow(path: pathlib.Path, subject: str) -> BinaryIO:
    """Open an absolute regular file through no-follow directory descriptors."""
    if not path.is_absolute():
        raise ValueError(f"{subject} path must be absolute after normalization")
    if ".." in path.parts:
        raise ValueError(f"{subject} path must not contain parent-directory components")
    directory_fd = open_directory_nofollow(path.parent, subject)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path.name, file_flags, dir_fd=directory_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(fd)
            raise ValueError(f"{subject} path must be a regular file: {path}")
        return os.fdopen(fd, "rb")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{subject} path does not exist: {path}") from exc
    except OSError as exc:
        raise ValueError(f"{subject} path is a symbolic link or special file: {path}") from exc
    finally:
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
            os.close(directory_fd)
            directory_fd = next_fd
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
    import yaml

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
    "read_bounded_utf8_regular",
    "read_strict_yaml",
    "validate_local_path",
]
