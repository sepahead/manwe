"""Offline synthetic data so every pipeline runs with zero downloads.

Generates a tiny YOLO-format detection dataset of coloured shapes standing in for
the crebain classes, written with a dependency-free PNG encoder. Enough to smoke-
exercise dataset parsing without touching a real corpus. It does not establish
model quality or export compatibility. For
fusion/multi-modal synthetic data use :func:`manwe.fusion.make_scenario`.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import struct
import zlib
from contextlib import suppress

import numpy as np

from ..common.config_io import open_directory_nofollow
from ..common.contracts import CREBAIN_CLASSES
from ..common.fd_io import attach_cleanup_failure, owned_binary_writer

MAX_PNG_PIXELS = 32_000_000
MAX_DATASET_PIXEL_BYTES = 512 * 1024 * 1024
MAX_DATASET_IMAGES = 100_000
MAX_OBJECTS_PER_IMAGE = 1_000
MAX_DATASET_OBJECTS = 5_000_000


def _entry_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _assert_directory_path(
    path: pathlib.Path,
    expected_identity: tuple[int, int],
    subject: str,
) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{subject} was replaced while output was being written") from exc
    if not stat.S_ISDIR(metadata.st_mode) or _entry_identity(metadata) != expected_identity:
        raise RuntimeError(f"{subject} was replaced while output was being written")


def _write_exclusive_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    display: pathlib.Path,
) -> tuple[int, int]:
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise ValueError("output filename must be one nonempty path component")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = -1
    identity: tuple[int, int] | None = None
    try:
        fd = os.open(name, flags, 0o644, dir_fd=directory_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"output is not a regular file: {display}")
        identity = _entry_identity(metadata)
        owned_fd = fd
        fd = -1
        handle = owned_binary_writer(owned_fd)
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            final_metadata = os.fstat(handle.fileno())
        published = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(published.st_mode)
            or _entry_identity(published) != identity
            or final_metadata.st_size != len(payload)
            or (
                published.st_size,
                published.st_mtime_ns,
                published.st_ctime_ns,
            )
            != (
                final_metadata.st_size,
                final_metadata.st_mtime_ns,
                final_metadata.st_ctime_ns,
            )
        ):
            raise RuntimeError(f"output was replaced while it was being written: {display}")
        os.fsync(directory_fd)
        return identity
    except BaseException:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and identity is not None and _entry_identity(current) == identity:
            with suppress(OSError):
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
        raise


def _write_exclusive(path: pathlib.Path, payload: bytes) -> None:
    absolute = path.absolute()
    parent_fd = open_directory_nofollow(absolute.parent, "output parent")
    identity: tuple[int, int] | None = None
    try:
        parent_identity = _entry_identity(os.fstat(parent_fd))
        _assert_directory_path(absolute.parent, parent_identity, "output parent")
        identity = _write_exclusive_at(parent_fd, absolute.name, payload, absolute)
        _assert_directory_path(absolute.parent, parent_identity, "output parent")
    except BaseException as error:
        if identity is not None:
            try:
                current = os.stat(
                    absolute.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError:
                current = None
            if current is not None and _entry_identity(current) == identity:
                with suppress(OSError):
                    os.unlink(absolute.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
        try:
            os.close(parent_fd)
        except BaseException as cleanup:
            attach_cleanup_failure(error, cleanup, "output parent descriptor cleanup failed")
        raise
    with suppress(OSError):
        os.close(parent_fd)


def _encode_png(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as one bounded PNG payload."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 array with shape (height, width, 3)")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("rgb dimensions must be nonzero")
    if rgb.shape[0] > 0xFFFFFFFF or rgb.shape[1] > 0xFFFFFFFF:
        raise ValueError("rgb dimensions exceed the PNG format limit")
    if rgb.shape[0] * rgb.shape[1] > MAX_PNG_PIXELS:
        raise ValueError(f"rgb exceeds the {MAX_PNG_PIXELS}-pixel safety limit")
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    return png + chunk(b"IEND", b"")


def _create_directory_at(
    parent_fd: int,
    name: str,
    display: pathlib.Path,
) -> tuple[int, tuple[int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    identity: tuple[int, int] | None = None
    child_fd = -1
    try:
        created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode):
            raise RuntimeError(f"output directory is not a directory: {display}")
        identity = _entry_identity(created)
        child_fd = os.open(name, flags, dir_fd=parent_fd)
        opened = os.fstat(child_fd)
        if not stat.S_ISDIR(opened.st_mode) or _entry_identity(opened) != identity:
            raise RuntimeError(f"output directory was replaced while opening: {display}")
        os.fsync(parent_fd)
        result = child_fd
        child_fd = -1
        return result, identity
    except BaseException:
        if child_fd >= 0:
            with suppress(OSError):
                os.close(child_fd)
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and identity is not None and _entry_identity(current) == identity:
            with suppress(OSError):
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
        raise


def _open_or_create_output_root(
    root: pathlib.Path,
) -> tuple[int, int, tuple[int, int], bool]:
    """Acquire the requested empty root without a path-based creation race."""
    if root == root.parent:
        raise FileExistsError(f"output directory must be absent or empty: {root}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        initial_root = root.lstat()
    except FileNotFoundError:
        initial_root = None
    if initial_root is not None:
        if not stat.S_ISDIR(initial_root.st_mode):
            raise FileExistsError(f"output directory must be absent or empty: {root}")
        parent_fd = open_directory_nofollow(root.parent, "output root parent")
        root_fd = -1
        try:
            root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
            opened = os.fstat(root_fd)
            root_identity = _entry_identity(initial_root)
            if not stat.S_ISDIR(opened.st_mode) or _entry_identity(opened) != root_identity:
                raise RuntimeError("output directory was replaced while it was being opened")
            _assert_directory_path(root, root_identity, "output directory")
            result = root_fd
            root_fd = -1
            return parent_fd, result, root_identity, False
        except BaseException:
            if root_fd >= 0:
                with suppress(OSError):
                    os.close(root_fd)
            with suppress(OSError):
                os.close(parent_fd)
            raise

    missing: list[str] = []
    existing = root
    while True:
        try:
            metadata = existing.lstat()
        except FileNotFoundError:
            missing.append(existing.name)
            existing = existing.parent
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"output path chain contains a non-directory: {existing}")
        break
    current_fd = open_directory_nofollow(existing, "output ancestor")
    child_fd = -1
    try:
        ordered_missing = list(reversed(missing))
        for index, component in enumerate(ordered_missing):
            try:
                os.mkdir(component, mode=0o777, dir_fd=current_fd)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"output directory was occupied while it was being created: {root}"
                ) from exc
            created = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            child_fd = os.open(component, directory_flags, dir_fd=current_fd)
            opened = os.fstat(child_fd)
            if not stat.S_ISDIR(created.st_mode) or _entry_identity(created) != _entry_identity(
                opened
            ):
                raise RuntimeError("output directory was replaced while it was being created")
            if index == len(ordered_missing) - 1:
                root_identity = _entry_identity(opened)
                _assert_directory_path(root, root_identity, "output directory")
                result = child_fd
                child_fd = -1
                return current_fd, result, root_identity, True
            previous_fd = current_fd
            current_fd = child_fd
            child_fd = -1
            try:
                os.close(previous_fd)
            except OSError as exc:
                raise RuntimeError("output ancestor descriptor could not be released") from exc
    except BaseException:
        if child_fd >= 0:
            with suppress(OSError):
                os.close(child_fd)
        with suppress(OSError):
            os.close(current_fd)
        raise
    raise RuntimeError("output root creation did not produce a directory")


def write_png(path: str | pathlib.Path, rgb: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 array as a PNG using only the stdlib."""
    destination = pathlib.Path(path).expanduser().resolve(strict=False)
    try:
        _write_exclusive(destination, _encode_png(rgb))
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing PNG: {destination}") from exc


# distinct colours per class so the toy task is learnable
_CLASS_COLORS = {
    "drone": (230, 60, 60),
    "bird": (60, 200, 90),
    "aircraft": (70, 120, 240),
    "helicopter": (240, 200, 60),
    "unknown": (170, 170, 170),
}


def make_vision_smoke(
    out_dir: str | pathlib.Path,
    n_train: int = 12,
    n_val: int = 4,
    size: int = 128,
    max_objs: int = 3,
    seed: int = 0,
) -> pathlib.Path:
    """Create a tiny YOLO-format detection dataset and its ``data.yaml``.

    Returns the path to ``data.yaml`` (pass it to ``manwe vision-train``).
    """
    for value, name in (
        (n_train, "n_train"),
        (n_val, "n_val"),
        (size, "size"),
        (max_objs, "max_objs"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if size < 16:
        raise ValueError("size must be at least 16 pixels")
    image_count = n_train + n_val
    if image_count > MAX_DATASET_IMAGES:
        raise ValueError(f"dataset exceeds the {MAX_DATASET_IMAGES}-image safety limit")
    if max_objs > MAX_OBJECTS_PER_IMAGE:
        raise ValueError(f"max_objs exceeds the {MAX_OBJECTS_PER_IMAGE}-object safety limit")
    if image_count * max_objs > MAX_DATASET_OBJECTS:
        raise ValueError(
            f"dataset exceeds the {MAX_DATASET_OBJECTS}-object annotation safety limit"
        )
    if size * size > MAX_PNG_PIXELS:
        raise ValueError(f"size exceeds the {MAX_PNG_PIXELS}-pixel image safety limit")
    if image_count * size * size * 3 > MAX_DATASET_PIXEL_BYTES:
        raise ValueError(
            f"dataset exceeds the {MAX_DATASET_PIXEL_BYTES}-byte raw pixel safety limit"
        )
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    raw_output = str(out_dir)
    if any(character in raw_output for character in "\0\r\n"):
        raise ValueError("output directory path contains a control character")
    root = pathlib.Path(out_dir).expanduser().resolve(strict=False)
    root_parent_fd: int | None = None
    root_fd: int | None = None
    root_created = False
    directory_records: list[tuple[int, str, int, tuple[int, int]]] = []
    file_records: list[tuple[int, str, tuple[int, int]]] = []
    rng = np.random.default_rng(seed)
    try:
        root_parent_fd, root_fd, root_identity, root_created = _open_or_create_output_root(root)
        if os.listdir(root_fd):
            raise FileExistsError(f"output directory must be absent or empty: {root}")

        split_fds: dict[tuple[str, str], int] = {}
        for kind in ("images", "labels"):
            kind_fd, kind_identity = _create_directory_at(root_fd, kind, root / kind)
            directory_records.append((root_fd, kind, kind_fd, kind_identity))
            for split in ("train", "val"):
                split_fd, split_identity = _create_directory_at(
                    kind_fd,
                    split,
                    root / kind / split,
                )
                directory_records.append((kind_fd, split, split_fd, split_identity))
                split_fds[(kind, split)] = split_fd

        for split, n in (("train", n_train), ("val", n_val)):
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            img_fd = split_fds[("images", split)]
            lbl_fd = split_fds[("labels", split)]
            for i in range(n):
                img = rng.integers(20, 60, size=(size, size, 3), dtype=np.uint8)  # dim sky-ish bg
                lines = []
                for _ in range(int(rng.integers(1, max_objs + 1))):
                    cls_i = int(rng.integers(0, len(CREBAIN_CLASSES)))
                    cls = CREBAIN_CLASSES[cls_i]
                    bw = int(rng.integers(size // 12, size // 4))
                    bh = int(rng.integers(size // 12, size // 4))
                    cx = int(rng.integers(bw, size - bw))
                    cy = int(rng.integers(bh, size - bh))
                    x1, y1 = cx - bw // 2, cy - bh // 2
                    img[y1 : y1 + bh, x1 : x1 + bw] = _CLASS_COLORS[cls]
                    lines.append(
                        f"{cls_i} {cx / size:.6f} {cy / size:.6f} {bw / size:.6f} {bh / size:.6f}"
                    )
                image_name = f"{split}_{i:03d}.png"
                image_identity = _write_exclusive_at(
                    img_fd,
                    image_name,
                    _encode_png(img),
                    img_dir / image_name,
                )
                file_records.append((img_fd, image_name, image_identity))
                label_name = f"{split}_{i:03d}.txt"
                label_identity = _write_exclusive_at(
                    lbl_fd,
                    label_name,
                    ("\n".join(lines) + "\n").encode(),
                    lbl_dir / label_name,
                )
                file_records.append((lbl_fd, label_name, label_identity))

        data_yaml = root / "data.yaml"
        manifest_identity = _write_exclusive_at(
            root_fd,
            data_yaml.name,
            (
                json.dumps(
                    {
                        "path": str(root),
                        "train": "images/train",
                        "val": "images/val",
                        "names": list(CREBAIN_CLASSES),
                    },
                    indent=2,
                )
                + "\n"
            ).encode(),
            data_yaml,
        )
        file_records.append((root_fd, data_yaml.name, manifest_identity))
        for _parent_fd, _name, child_fd, _identity in reversed(directory_records):
            os.fchmod(child_fd, 0o755)
            os.fsync(child_fd)
        os.fsync(root_fd)
        _assert_directory_path(root, root_identity, "output directory")
        return data_yaml
    except BaseException as error:
        cleanup_failures: list[str] = []
        for parent_fd, name, identity in reversed(file_records):
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                cleanup_failures.append(f"could not inspect output file {name!r}: {exc}")
                continue
            if _entry_identity(current) != identity:
                cleanup_failures.append(f"replaced output file {name!r}")
                continue
            try:
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as exc:
                cleanup_failures.append(f"could not remove output file {name!r}: {exc}")

        for parent_fd, name, child_fd, identity in reversed(directory_records):
            try:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(child_fd)
                remaining_names = os.listdir(child_fd)
            except OSError as exc:
                cleanup_failures.append(f"could not inspect output directory {name!r}: {exc}")
                continue
            if (
                _entry_identity(named) != identity
                or _entry_identity(opened) != identity
                or remaining_names
            ):
                cleanup_failures.append(f"output directory {name!r} was replaced or is not empty")
                continue
            try:
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as exc:
                cleanup_failures.append(f"could not remove output directory {name!r}: {exc}")

        if root_created and root_fd is not None and root_parent_fd is not None:
            try:
                named_root = os.stat(
                    root.name,
                    dir_fd=root_parent_fd,
                    follow_symlinks=False,
                )
                opened_root = os.fstat(root_fd)
                if _entry_identity(named_root) == _entry_identity(opened_root) and not os.listdir(
                    root_fd
                ):
                    os.rmdir(root.name, dir_fd=root_parent_fd)
                    os.fsync(root_parent_fd)
                else:
                    cleanup_failures.append("output root was replaced or is not empty")
            except OSError as exc:
                cleanup_failures.append(f"could not remove output root: {exc}")
        if cleanup_failures and hasattr(error, "add_note"):
            error.add_note("synthetic output rollback incomplete: " + "; ".join(cleanup_failures))
        raise
    finally:
        for _parent_fd, _name, child_fd, _identity in reversed(directory_records):
            with suppress(OSError):
                os.close(child_fd)
        if root_fd is not None:
            with suppress(OSError):
                os.close(root_fd)
        if root_parent_fd is not None:
            with suppress(OSError):
                os.close(root_parent_fd)


__all__ = ["write_png", "make_vision_smoke"]
