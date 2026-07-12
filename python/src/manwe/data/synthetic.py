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
import shutil
import stat
import struct
import zlib
from contextlib import suppress

import numpy as np

from ..common.config_io import validate_local_path
from ..common.contracts import CREBAIN_CLASSES

MAX_PNG_PIXELS = 32_000_000
MAX_DATASET_PIXEL_BYTES = 512 * 1024 * 1024
MAX_DATASET_IMAGES = 100_000
MAX_OBJECTS_PER_IMAGE = 1_000
MAX_DATASET_OBJECTS = 5_000_000


def _write_exclusive(path: pathlib.Path, payload: bytes) -> None:
    validate_local_path(path.parent.absolute(), "output parent", require_directory=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, 0o644)
    metadata = os.fstat(fd)
    identity = (metadata.st_dev, metadata.st_ino)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        published = path.lstat()
        if (published.st_dev, published.st_ino) != identity:
            raise RuntimeError(f"output was replaced while it was being written: {path}")
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == identity:
            path.unlink()
        raise


def _validate_creatable_directory_chain(path: pathlib.Path) -> None:
    current = pathlib.Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"output path chain contains a symbolic link: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"output path chain contains a non-directory: {current}")


def write_png(path: str | pathlib.Path, rgb: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 array as a PNG using only the stdlib."""
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
    png += chunk(b"IEND", b"")
    destination = pathlib.Path(path).expanduser().resolve(strict=False)
    try:
        _write_exclusive(destination, png)
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
    _validate_creatable_directory_chain(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {root}")
    root_preexisted = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    validate_local_path(root, "output directory", require_directory=True)
    if any(root.iterdir()):
        raise FileExistsError(f"output directory must remain empty until generation: {root}")
    rng = np.random.default_rng(seed)
    try:
        for split, n in (("train", n_train), ("val", n_val)):
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
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
                write_png(img_dir / f"{split}_{i:03d}.png", img)
                _write_exclusive(
                    lbl_dir / f"{split}_{i:03d}.txt",
                    ("\n".join(lines) + "\n").encode(),
                )

        data_yaml = root / "data.yaml"
        _write_exclusive(
            data_yaml,
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
        )
        return data_yaml
    except BaseException:
        for child in (root / "data.yaml", root / "images", root / "labels"):
            try:
                metadata = child.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        if not root_preexisted:
            with suppress(OSError):
                root.rmdir()
        raise


__all__ = ["write_png", "make_vision_smoke"]
