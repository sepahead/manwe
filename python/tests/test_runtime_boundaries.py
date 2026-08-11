"""Adversarial boundary tests for runtime, training, and raw export preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import stat
import sys
import threading
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from manwe.cli import _cmd_vision_train
from manwe.common.artifacts import ArtifactSnapshot
from manwe.common.dataset_manifest import (
    snapshot_local_calibration_dataset,
    snapshot_local_training_directory,
    validate_local_detection_manifest,
)
from manwe.common.device import Device, resolve_device
from manwe.common.logging import configure_logging, get_logger
from manwe.eval.detection import average_precision
from manwe.export.backends import _publish_exclusive, export_model
from manwe.vision.input import prepare_single_image
from manwe.vision.postprocess import nms, scale_boxes
from manwe.vision.predict import Detector
from manwe.vision.sahi_infer import SliceConfig
from manwe.vision.train import VisionTrainConfig, train


def _assert_exception_note(error: BaseException, *fragments: str) -> None:
    """Assert PEP 678 evidence where the supported interpreter provides it."""
    if not hasattr(error, "add_note"):
        return
    assert any(
        all(fragment in note for fragment in fragments) for note in getattr(error, "__notes__", ())
    )


def _write_calibration_png(
    path,
    color: tuple[int, int, int],
    *,
    metadata: tuple[str, str] | None = None,
    size: tuple[int, int] = (10, 10),
) -> None:
    pnginfo = None
    if metadata is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(*metadata)
    Image.new("RGB", size, color).save(path, format="PNG", pnginfo=pnginfo)


def _write_rfdetr_coco_dataset(root: Path) -> None:
    categories = [{"id": 1, "name": "drone"}]
    for split, image_id in (("train", 1), ("valid", 2)):
        directory = root / split
        directory.mkdir(parents=True)
        Image.new("RGB", (16, 12), (image_id, 2, 3)).save(directory / "frame.jpg", format="JPEG")
        payload = {
            "images": [{"id": image_id, "file_name": "frame.jpg", "width": 16, "height": 12}],
            "annotations": [
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [1, 1, 4, 3],
                    "area": 12,
                    "iscrowd": 0,
                }
            ],
            "categories": categories,
        }
        (directory / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_artifact_snapshot_binds_verified_bytes_and_bounds_directory_entries(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    with ArtifactSnapshot(artifact, digest, allowed_suffixes={".pt"}) as snapshot:
        artifact.write_bytes(b"replaced")
        assert snapshot.path.read_bytes() == b"trusted"
        assert snapshot.sha256 == digest

    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactSnapshot(artifact, digest, allowed_suffixes={".pt"})

    bundle = tmp_path / "model.mlpackage"
    bundle.mkdir()
    (bundle / "a").write_bytes(b"a")
    (bundle / "b").write_bytes(b"b")
    from manwe.common.artifacts import sha256_artifact

    with pytest.raises(ValueError, match="entry safety limit"):
        sha256_artifact(bundle, max_entries=1)


def test_artifact_snapshot_enforces_declared_root_kind(tmp_path):
    file_artifact = tmp_path / "model.onnx"
    file_artifact.write_bytes(b"model")
    file_digest = hashlib.sha256(b"model").hexdigest()
    with pytest.raises(ValueError, match="directory bundle"):
        ArtifactSnapshot(file_artifact, file_digest, expect_directory=True)

    bundle = tmp_path / "model.mlpackage"
    bundle.mkdir()
    (bundle / "weights").write_bytes(b"model")
    from manwe.common.artifacts import sha256_artifact

    with pytest.raises(ValueError, match="regular file"):
        ArtifactSnapshot(bundle, sha256_artifact(bundle), expect_directory=False)


def test_closed_artifact_snapshot_cannot_be_reentered(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    snapshot = ArtifactSnapshot(artifact, hashlib.sha256(b"model").hexdigest())
    snapshot.close()
    snapshot.close()

    with pytest.raises(RuntimeError, match="snapshot is closed"):
        snapshot.__enter__()


def test_directory_snapshot_enforces_byte_limit_during_copy(tmp_path, monkeypatch):
    from manwe.common import artifacts

    bundle = tmp_path / "bundle.mlpackage"
    bundle.mkdir()
    member = bundle / "weights.bin"
    member.write_bytes(b"a")
    expected = artifacts.sha256_artifact(bundle)
    real_entries = artifacts._descriptor_tree_entries
    mutated = False

    def mutate_before_descriptor_enumeration(
        directory_fd,
        *,
        display,
        max_entries,
        **kwargs,
    ):
        nonlocal mutated
        if not mutated:
            member.write_bytes(b"x" * 100)
            mutated = True
        return real_entries(
            directory_fd,
            display=display,
            max_entries=max_entries,
            **kwargs,
        )

    monkeypatch.setattr(
        artifacts,
        "_descriptor_tree_entries",
        mutate_before_descriptor_enumeration,
    )
    with pytest.raises(ValueError, match="byte safety limit"):
        ArtifactSnapshot(bundle, expected, max_bytes=8)


def test_artifact_snapshot_closes_source_when_private_destination_open_fails(
    tmp_path,
    monkeypatch,
):
    from manwe.common import artifacts

    source = tmp_path / "model.pt"
    source.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    real_open = artifacts.os.open
    source_fds: list[int] = []

    def fail_private_destination(path, *args, **kwargs):
        if path == "artifact.pt" and kwargs.get("dir_fd") is not None:
            raise OSError("injected destination open failure")
        fd = real_open(path, *args, **kwargs)
        if path == source.name and kwargs.get("dir_fd") is not None:
            source_fds.append(fd)
        return fd

    monkeypatch.setattr(artifacts.os, "open", fail_private_destination)
    with pytest.raises(OSError, match="injected destination open failure"):
        ArtifactSnapshot(source, digest, allowed_suffixes={".pt"})
    assert len(source_fds) == 1
    with pytest.raises(OSError):
        os.fstat(source_fds[0])


def test_open_regular_nofollow_closes_file_fd_when_fdopen_fails(tmp_path, monkeypatch):
    from manwe.common import config_io, fd_io

    path = tmp_path / "config.yaml"
    path.write_bytes(b"value")
    opened_fds: list[int] = []

    def fail_fdopen(fd, _mode):
        opened_fds.append(fd)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_fdopen)
    with pytest.raises(OSError, match="injected fdopen failure"):
        config_io.open_regular_nofollow(path.absolute(), "test config")
    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])


def test_open_directory_nofollow_closes_both_fds_when_intermediate_close_fails(
    tmp_path,
    monkeypatch,
):
    from manwe.common import config_io

    directory = tmp_path / "nested"
    directory.mkdir()
    real_open = config_io.os.open
    real_close = config_io.os.close
    opened_fds: list[int] = []
    close_calls: list[int] = []
    close_failed = False

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened_fds.append(fd)
        return fd

    def fail_first_close(fd):
        nonlocal close_failed
        close_calls.append(fd)
        if not close_failed:
            close_failed = True
            # POSIX permits close(2) to have released the descriptor even when
            # reporting an error. Retrying the same integer could then close an
            # unrelated descriptor that another thread has already acquired.
            real_close(fd)
            raise OSError("injected close failure")
        return real_close(fd)

    monkeypatch.setattr(config_io.os, "open", track_open)
    monkeypatch.setattr(config_io.os, "close", fail_first_close)
    with pytest.raises(RuntimeError, match="descriptor could not be released"):
        config_io.open_directory_nofollow(directory.absolute(), "test directory")
    assert len(opened_fds) == 2
    assert close_calls.count(opened_fds[0]) == 1
    for fd in opened_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_open_directory_nofollow_never_retries_indeterminate_pre_release_close(
    tmp_path, monkeypatch
):
    from manwe.common import config_io

    directory = tmp_path / "nested"
    directory.mkdir()
    real_open = config_io.os.open
    real_close = config_io.os.close
    opened_fds: list[int] = []
    close_calls: list[int] = []
    failed = False

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened_fds.append(fd)
        return fd

    def fail_before_release(fd):
        nonlocal failed
        close_calls.append(fd)
        if not failed:
            failed = True
            raise OSError("injected pre-release close failure")
        return real_close(fd)

    monkeypatch.setattr(config_io.os, "open", track_open)
    monkeypatch.setattr(config_io.os, "close", fail_before_release)
    with pytest.raises(RuntimeError, match="descriptor could not be released"):
        config_io.open_directory_nofollow(directory.absolute(), "test directory")
    assert len(opened_fds) == 2
    assert close_calls.count(opened_fds[0]) == 1
    assert os.fstat(opened_fds[0])
    with pytest.raises(OSError):
        os.fstat(opened_fds[1])
    monkeypatch.setattr(config_io.os, "close", real_close)
    real_close(opened_fds[0])


def test_descriptor_copy_preserves_fdopen_error_and_closes_raw_fds(tmp_path, monkeypatch):
    from manwe.common import artifacts, fd_io
    from manwe.common.config_io import open_directory_nofollow

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    source_fd = open_directory_nofollow(source, "source")
    destination_fd = open_directory_nofollow(destination, "destination")
    real_fdopen = fd_io._nonowning_file
    opened_fds: list[int] = []

    def fail_second_fdopen(fd, mode):
        opened_fds.append(fd)
        if len(opened_fds) == 2:
            raise OSError("injected fdopen failure")
        return real_fdopen(fd, mode)

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_second_fdopen)
    try:
        with pytest.raises(OSError, match="injected fdopen failure"):
            artifacts._copy_directory_fd(
                source_fd,
                destination_fd,
                display="test source",
                max_bytes=1024,
                max_entries=10,
            )
    finally:
        os.close(destination_fd)
        os.close(source_fd)
    assert len(opened_fds) == 2
    for fd in opened_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_descriptor_entry_does_not_leak_leaf_when_parent_close_fails(tmp_path, monkeypatch):
    from manwe.common import artifacts
    from manwe.common.config_io import open_directory_nofollow

    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "weights.bin").write_bytes(b"trusted")
    root_fd = open_directory_nofollow(root, "root")
    real_close = artifacts.os.close
    close_calls: list[int] = []
    leaf_fds: list[int] = []
    real_open = artifacts.os.open
    failed = False
    failed_fd: int | None = None

    def track_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if path == "weights.bin":
            leaf_fds.append(fd)
        return fd

    def fail_second_close_after_release(fd):
        nonlocal failed, failed_fd
        close_calls.append(fd)
        if not failed and len(close_calls) == 2:
            failed = True
            failed_fd = fd
            real_close(fd)
            raise OSError("injected close failure")
        return real_close(fd)

    monkeypatch.setattr(artifacts.os, "open", track_open)
    monkeypatch.setattr(artifacts.os, "close", fail_second_close_after_release)
    try:
        with pytest.raises(OSError, match="injected close failure"):
            artifacts._open_descriptor_entry(root_fd, "nested/weights.bin", expect_directory=False)
    finally:
        real_close(root_fd)
    assert failed_fd is not None
    assert close_calls.count(failed_fd) == 1
    assert len(leaf_fds) == 1
    with pytest.raises(OSError):
        os.fstat(leaf_fds[0])


def test_artifact_snapshot_cleanup_never_masks_body_exception(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    snapshot = ArtifactSnapshot(source, digest, allowed_suffixes={".onnx"})
    real_close = snapshot.close

    class BodyError(Exception):
        pass

    def close_then_fail():
        real_close()
        raise OSError("injected snapshot cleanup failure")

    monkeypatch.setattr(snapshot, "close", close_then_fail)
    with pytest.raises(BodyError, match="body failure") as captured, snapshot:
        raise BodyError("body failure")
    assert isinstance(captured.value.__cause__, OSError)
    _assert_exception_note(captured.value, "artifact snapshot cleanup failed")


def test_bounded_regular_read_detects_ctime_only_mutation(tmp_path, monkeypatch):
    from manwe.common import config_io

    path = tmp_path / "config.yaml"
    path.write_bytes(b"trusted")
    original = path.stat()
    real_fstat = config_io.os.fstat
    matching_calls = 0

    def mutate_after_read(fd):
        nonlocal matching_calls
        metadata = real_fstat(fd)
        if (metadata.st_dev, metadata.st_ino) == (original.st_dev, original.st_ino):
            matching_calls += 1
            if matching_calls == 3:
                path.write_bytes(b"altered")
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
                metadata = real_fstat(fd)
        return metadata

    monkeypatch.setattr(config_io.os, "fstat", mutate_after_read)
    with pytest.raises(ValueError, match="changed while it was being read"):
        config_io.read_bounded_regular_bytes(path.absolute(), 1024, "test config")


def test_path_artifact_hash_rejects_excessive_directory_depth(tmp_path):
    from manwe.common import artifacts

    root = tmp_path / "bundle.mlpackage"
    root.mkdir()
    current = root
    for _ in range(artifacts._MAX_DESCRIPTOR_TREE_DEPTH + 1):
        current /= "d"
        current.mkdir()
    (current / "weights.bin").write_bytes(b"trusted")

    with pytest.raises(ValueError, match="depth safety limit"):
        artifacts.sha256_artifact(root)


def test_descriptor_hashes_preserve_canonical_global_path_order(tmp_path):
    from manwe.common.artifacts import (
        sha256_artifact,
        sha256_artifact_at,
        sha256_directory_fd,
    )
    from manwe.common.config_io import open_directory_nofollow

    bundle = tmp_path / "bundle.mlpackage"
    (bundle / "a").mkdir(parents=True)
    (bundle / "a" / "x").write_bytes(b"nested-a")
    (bundle / "a!").write_bytes(b"prefix-sibling-a")
    (bundle / "val").mkdir()
    (bundle / "val" / "x").write_bytes(b"nested-val")
    (bundle / "val.cache").write_bytes(b"prefix-sibling-val")
    canonical = sha256_artifact(bundle)
    root_fd = open_directory_nofollow(bundle, "test bundle")
    parent_fd = open_directory_nofollow(bundle.parent, "test bundle parent")
    try:
        assert sha256_directory_fd(root_fd) == canonical
        assert sha256_artifact_at(parent_fd, bundle.name) == canonical
        assert sha256_artifact_at(parent_fd, bundle.name, expect_directory=True) == canonical
        with pytest.raises(ValueError, match="regular file"):
            sha256_artifact_at(parent_fd, bundle.name, expect_directory=False)
        with ArtifactSnapshot.from_directory_fd(root_fd, canonical) as snapshot:
            assert snapshot.sha256 == canonical
    finally:
        os.close(parent_fd)
        os.close(root_fd)

    regular = tmp_path / "model.onnx"
    regular.write_bytes(b"model")
    parent_fd = open_directory_nofollow(tmp_path, "test artifact parent")
    try:
        assert sha256_artifact_at(
            parent_fd, regular.name, expect_directory=False
        ) == sha256_artifact(regular)
        with pytest.raises(ValueError, match="directory bundle"):
            sha256_artifact_at(parent_fd, regular.name, expect_directory=True)
        with pytest.raises(TypeError, match="boolean or None"):
            sha256_artifact_at(parent_fd, regular.name, expect_directory=1)  # type: ignore[arg-type]
    finally:
        os.close(parent_fd)


def test_dataset_manifest_is_local_directive_free_private_snapshot(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames:\n  0: drone\n  1: bird\n",
        encoding="utf-8",
    )

    snapshot = validate_local_detection_manifest(manifest)
    try:
        assert snapshot.path != manifest
        value = snapshot.path.read_text(encoding="utf-8")
        assert "download" not in value
        assert str(train_dir) in value and str(val_dir) in value
        manifest.write_text("download: __import__('os').system('false')\n", encoding="utf-8")
        assert "download" not in snapshot.path.read_text(encoding="utf-8")
    finally:
        private_path = snapshot.path
        snapshot.close()
    assert not private_path.exists()

    manifest.write_text(
        "train: https://example.invalid/train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="local filesystem"):
        validate_local_detection_manifest(manifest)
    manifest.write_text(
        "train: train\nval: val\nnames: [drone]\ndownload: __import__('os').system('false')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="download directives"):
        validate_local_detection_manifest(manifest)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            "train: train\ntrain: val\nval: val\nnames: [drone]\n",
            "duplicate key",
        ),
        (
            "train: train\nval: val\nnames: &classes [drone]\n",
            "aliases, anchors",
        ),
        (
            "train: train\nval: val\nnames: {0: '   '}\n",
            "printable strings",
        ),
        (
            "train: train\nval: val\nnames: ['drone', ' drone ']\n",
            "must be unique",
        ),
        (
            "train: train\nval: val\nnames: [drone]\nchannels: true\n",
            "channels must be 1 or 3",
        ),
    ],
)
def test_dataset_manifest_rejects_ambiguous_or_malformed_metadata(tmp_path, payload, message):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    manifest = tmp_path / "data.yaml"
    manifest.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_symlinked_path_components_and_special_splits(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "train").mkdir()
    (real_root / "val").mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: linked\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path chain contains a symbolic link"):
        validate_local_detection_manifest(manifest)

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    manifest.write_text("path: .\ntrain: fifo\nval: real/val\nnames: [drone]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_parent_directory_components(tmp_path):
    manifest_parent = tmp_path / "manifests"
    manifest_parent.mkdir()
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    manifest = manifest_parent / "data.yaml"
    manifest.write_text(
        "path: ../dataset\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent-directory components"):
        validate_local_detection_manifest(manifest)

    manifest.write_text(
        f"path: {dataset}\ntrain: ../outside\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside the declared dataset root"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_duplicate_or_nested_splits(tmp_path):
    train = tmp_path / "images"
    (train / "val").mkdir(parents=True)
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        "path: .\ntrain: images\nval: images/val\nnames: [drone]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="paths overlap"):
        validate_local_detection_manifest(manifest)

    manifest.write_text("path: .\ntrain: images\nval: images\nnames: [drone]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="paths overlap"):
        validate_local_detection_manifest(manifest)

    separate = tmp_path / "separate"
    (separate / "nested").mkdir(parents=True)
    manifest.write_text(
        "path: .\ntrain: images\nval: [separate, separate/nested]\nnames: [drone]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="would select files more than once"):
        validate_local_detection_manifest(manifest)


@pytest.mark.parametrize(
    ("train", "val", "message"),
    [
        ("null", "val", "dataset train must be a path string or nonempty path list"),
        ("train", "null", "dataset val must be a path string or nonempty path list"),
        ("''", "val", "dataset train must be a nonempty local filesystem path"),
        ("train", "[]", "dataset val must contain nonempty path strings"),
    ],
)
def test_dataset_manifest_requires_nonempty_train_and_val(tmp_path, train, val, message):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    manifest = tmp_path / "data.yaml"
    manifest.write_text(f"path: .\ntrain: {train}\nval: {val}\nnames: [drone]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_nested_cross_split_hardlink_alias(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    shared = train / "shared.jpg"
    alias = val / "alias.jpg"
    shared.write_bytes(b"shared image")
    try:
        os.link(shared, alias)
    except OSError as exc:  # pragma: no cover - filesystem capability boundary
        pytest.skip(f"hard links are unavailable: {exc}")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train and val paths identify the same filesystem entry"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_nested_within_split_hardlink_alias(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    original = train / "original.jpg"
    alias = train / "alias.jpg"
    original.write_bytes(b"shared image")
    (val / "distinct.jpg").write_bytes(b"validation image")
    try:
        os.link(original, alias)
    except OSError as exc:  # pragma: no cover - filesystem capability boundary
        pytest.skip(f"hard links are unavailable: {exc}")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train paths identify the same filesystem entry"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_accepts_distinct_same_byte_nested_files(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "image.jpg").write_bytes(b"identical image bytes")
    (val / "image.jpg").write_bytes(b"identical image bytes")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    with validate_local_detection_manifest(manifest) as snapshot:
        payload = snapshot.path.read_text(encoding="utf-8")
        assert str(train) in payload
        assert str(val) in payload


def test_training_snapshot_revalidates_split_identity_after_admission(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    train_file = train / "frame.jpg"
    val_file = val / "frame.jpg"
    train_file.write_bytes(b"training image")
    val_file.write_bytes(b"validation image")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    with validate_local_detection_manifest(manifest) as admitted:
        val_file.unlink()
        try:
            os.link(train_file, val_file)
        except OSError as exc:  # pragma: no cover - filesystem capability boundary
            pytest.skip(f"hard links are unavailable: {exc}")
        with pytest.raises(ValueError, match="train and val paths identify the same"):
            admitted.snapshot_for_training()


def test_training_snapshot_validates_the_exact_inventory_it_copies(tmp_path, monkeypatch):
    from manwe.common import artifacts

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    train_file = train / "frame.jpg"
    val_file = val / "frame.jpg"
    # The directory digest deliberately binds paths and bytes, not inode
    # topology. Equal bytes therefore isolate the hardlink-identity invariant.
    train_file.write_bytes(b"same image bytes")
    val_file.write_bytes(b"same image bytes")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    original_snapshot = artifacts.ArtifactSnapshot.from_directory_fd.__func__
    replaced = False

    def alias_immediately_before_copy(cls, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            train_file.unlink()
            try:
                os.link(val_file, train_file)
            except OSError as exc:  # pragma: no cover - filesystem capability boundary
                pytest.skip(f"hard links are unavailable: {exc}")
            replaced = True
        return original_snapshot(cls, *args, **kwargs)

    monkeypatch.setattr(
        artifacts.ArtifactSnapshot,
        "from_directory_fd",
        classmethod(alias_immediately_before_copy),
    )
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="train and val paths identify the same"),
    ):
        admitted.snapshot_for_training()
    assert replaced


def test_training_snapshot_rejects_rebound_manifest_or_root(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    def write_dataset(root, payload):
        (root / "train").mkdir(parents=True)
        (root / "val").mkdir()
        (root / "train" / "frame.jpg").write_bytes(payload)
        (root / "val" / "frame.jpg").write_bytes(payload + b" validation")
        manifest = root / "data.yaml"
        manifest.write_text(
            "path: .\ntrain: train\nval: val\nnames: [drone]\n",
            encoding="utf-8",
        )
        return manifest

    admitted_manifest = write_dataset(tmp_path / "admitted", b"admitted")
    rebound_manifest = write_dataset(tmp_path / "rebound", b"rebound")

    with validate_local_detection_manifest(admitted_manifest) as admitted:
        rebound = validate_local_detection_manifest(rebound_manifest)
        monkeypatch.setattr(
            dataset_manifest,
            "validate_local_detection_manifest",
            lambda _path: rebound,
        )
        with pytest.raises(
            ValueError,
            match="manifest or declared root identity changed after admission",
        ):
            admitted.snapshot_for_training()
        assert rebound._closed


def test_dataset_manifest_rejects_nested_symlink_escape_and_special_entry(tmp_path):
    root = tmp_path / "dataset"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    (val / "image.jpg").write_bytes(b"validation image")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    nested = train / "escape.jpg"
    nested.symlink_to(outside)
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic link"):
        validate_local_detection_manifest(manifest)

    nested.unlink()
    os.mkfifo(train / "pipe")
    with pytest.raises(ValueError, match="unsupported entry"):
        validate_local_detection_manifest(manifest)


@pytest.mark.parametrize("test_value", ["null", "''"])
def test_dataset_manifest_omits_empty_optional_test_split(tmp_path, test_value):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        f"path: .\ntrain: train\nval: val\ntest: {test_value}\nnames: [drone]\n",
        encoding="utf-8",
    )

    with validate_local_detection_manifest(manifest) as snapshot:
        assert "test:" not in snapshot.path.read_text(encoding="utf-8")


def test_dataset_manifest_bounds_selected_split_inventory(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "one.jpg").write_bytes(b"one")
    (train / "two.jpg").write_bytes(b"two")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    monkeypatch.setattr(dataset_manifest, "_MAX_DATASET_ENTRIES", 2)
    with pytest.raises(ValueError, match="entry safety limit"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_bounds_selected_split_bytes(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "image.jpg").write_bytes(b"too large")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")

    monkeypatch.setattr(dataset_manifest, "_MAX_DATASET_BYTES", 4)
    with pytest.raises(ValueError, match="byte safety limit"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_split_mutation_during_inventory(tmp_path, monkeypatch):
    from manwe.common import artifacts

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    image = train / "image.jpg"
    image.write_bytes(b"original")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    original_inventory = artifacts._descriptor_tree_entries
    replaced = False

    def replace_after_inventory(directory_fd, **kwargs):
        nonlocal replaced
        entries = original_inventory(directory_fd, **kwargs)
        if not replaced and str(kwargs["display"]).startswith("dataset train split"):
            image.unlink()
            image.write_bytes(b"attacker replacement")
            replaced = True
        return entries

    monkeypatch.setattr(artifacts, "_descriptor_tree_entries", replace_after_inventory)
    with pytest.raises(ValueError, match="changed during bounded inventory"):
        validate_local_detection_manifest(manifest)


def test_dataset_manifest_rejects_mutation_between_split_inventories(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    train_image = train / "image.jpg"
    val_image = val / "image.jpg"
    train_image.write_bytes(b"training image")
    val_image.write_bytes(b"validation image")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    original_inventory = dataset_manifest._inventory_selected_split_path
    replaced = False

    def replace_after_train(*args, **kwargs):
        nonlocal replaced
        inventory = original_inventory(*args, **kwargs)
        if not replaced and kwargs["field"] == "train":
            train_image.unlink()
            os.link(val_image, train_image)
            replaced = True
        return inventory

    monkeypatch.setattr(
        dataset_manifest,
        "_inventory_selected_split_path",
        replace_after_train,
    )
    with pytest.raises(ValueError, match="changed during aggregate inventory"):
        validate_local_detection_manifest(manifest)
    assert train_image.stat().st_ino == val_image.stat().st_ino


def test_dataset_manifest_rejects_unavailable_regular_file_identity(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "train" / "frame.jpg").write_bytes(b"train")
    (tmp_path / "val" / "frame.jpg").write_bytes(b"val")
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    checked_identity = dataset_manifest._split_entry_identity

    def identity_without_inode(metadata, *, subject):
        unavailable = SimpleNamespace(st_dev=metadata.st_dev, st_ino=0)
        return checked_identity(unavailable, subject=subject)

    monkeypatch.setattr(dataset_manifest, "_split_entry_identity", identity_without_inode)
    with pytest.raises(ValueError, match="stable device/inode identity"):
        validate_local_detection_manifest(manifest)


@pytest.mark.parametrize("suffix", [".txt", ".jpg", ".yaml"])
def test_dataset_manifest_rejects_every_path_list_file_spelling(tmp_path, suffix):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "train" / "image.jpg").write_bytes(b"train")
    (tmp_path / "val" / "image.jpg").write_bytes(b"val")
    path_list = tmp_path / f"paths{suffix}"
    path_list.write_text("/outside/private-snapshot.jpg\n", encoding="utf-8")
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        f"path: .\ntrain: {path_list.name}\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )

    message = "path-list indirection" if suffix == ".txt" else "must be a directory"
    with pytest.raises(ValueError, match=message):
        validate_local_detection_manifest(manifest)


def test_training_snapshot_rejects_an_empty_required_split(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "image.jpg").write_bytes(b"validation")
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )

    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="train split must contain at least one"),
    ):
        admitted.snapshot_for_training()


def test_calibration_digest_binds_dataset_content_and_enforces_coverage(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    _write_calibration_png(train / "not-calibration.png", (1, 2, 3))
    _write_calibration_png(val / "one.png", (4, 5, 6))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)
    snapshot = validate_local_detection_manifest(manifest)
    try:
        with pytest.raises(ValueError, match="at least 2"):
            snapshot.calibration_digest()
        second = val / "two.png"
        _write_calibration_png(second, (7, 8, 9))
        before = snapshot.calibration_digest()
        assert snapshot.calibration_digest(image_size=320) != before
        assert snapshot.calibration_digest(
            tensorrt_version="10.1.0"
        ) != snapshot.calibration_digest(
            tensorrt_version="11.0.0",
            modelopt_version="0.44.0",
        )
        assert snapshot.calibration_digest(
            tensorrt_version="11.0.0",
            modelopt_version="0.44.0",
        ) != snapshot.calibration_digest(
            tensorrt_version="11.0.0",
            modelopt_version="0.45.0",
        )
        for rejected_modelopt in ("0.44.0rc5", "0.44.dev1"):
            with pytest.raises(ValueError, match="nvidia-modelopt>=0.44"):
                snapshot.calibration_digest(
                    tensorrt_version="11.0.0",
                    modelopt_version=rejected_modelopt,
                )
        modelopt_bytes_per_pixel = dataset_manifest._BACKEND_CALIBRATION_IMAGES * 3 * 10
        max_safe_modelopt_image_size = math.isqrt(
            dataset_manifest._MAX_MODELOPT_CALIBRATION_WORK_BYTES // modelopt_bytes_per_pixel
        )
        assert max_safe_modelopt_image_size == 747
        snapshot.calibration_digest(
            image_size=max_safe_modelopt_image_size,
            tensorrt_version="11.0.0",
            modelopt_version="0.44.0",
        )
        with pytest.raises(ValueError, match="peak-work safety limit"):
            snapshot.calibration_digest(
                image_size=max_safe_modelopt_image_size + 1,
                tensorrt_version="11.0.0",
                modelopt_version="0.44.0",
            )
        with pytest.raises(ValueError, match="10.2"):
            snapshot.calibration_digest(tensorrt_version="10.2.0")
        with pytest.raises(ValueError, match="newer than the audited"):
            snapshot.calibration_digest(tensorrt_version="12.0.0")
        _write_calibration_png(second, (10, 11, 12))
        assert snapshot.calibration_digest() != before
    finally:
        snapshot.close()


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_calibration_rejects_empty_or_corrupt_image_suffix_files(tmp_path, monkeypatch, payload):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    (val / "counterfeit.png").write_bytes(payload)
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with pytest.raises(ValueError, match="empty|invalid|cannot identify"):
        snapshot_local_calibration_dataset(manifest)


def test_calibration_rejects_suffix_content_mismatch_and_duplicate_pixels(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    first = val / "first.png"
    second = val / "second.png"
    _write_calibration_png(first, (1, 2, 3), metadata=("encoding", "one"))
    _write_calibration_png(second, (1, 2, 3), metadata=("encoding", "two"))
    assert first.read_bytes() != second.read_bytes()
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)

    with pytest.raises(ValueError, match="unique backend tensors"):
        snapshot_local_calibration_dataset(manifest)

    second.unlink()
    counterfeit = val / "counterfeit.jpg"
    counterfeit.write_bytes(first.read_bytes())
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)
    with pytest.raises(ValueError, match="suffix/content mismatch"):
        snapshot_local_calibration_dataset(manifest)


def test_calibration_rejects_nonidentity_exif_orientation(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    image = Image.new("RGB", (20, 10), (1, 2, 3))
    exif = Image.Exif()
    exif[274] = 6
    image.save(val / "rotated.jpg", format="JPEG", exif=exif)
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with pytest.raises(ValueError, match="identity EXIF orientation"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_rejects_truncated_jpeg(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    image = val / "truncated.jpg"
    Image.new("RGB", (20, 10), (1, 2, 3)).save(image, format="JPEG")
    image.write_bytes(image.read_bytes()[:-2])
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with pytest.raises(ValueError, match="truncated|EOI|invalid"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


@pytest.mark.parametrize(
    ("suffix", "image_format"),
    ((".webp", "WEBP"), (".tiff", "TIFF")),
)
def test_calibration_rejects_animated_or_multiframe_images(
    tmp_path,
    monkeypatch,
    suffix,
    image_format,
):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    first = Image.new("RGB", (20, 10), (255, 0, 0))
    second = Image.new("RGB", (20, 10), (0, 0, 255))
    format_options = {"lossless": True, "loop": 0} if image_format == "WEBP" else {}
    first.save(
        val / f"multiple{suffix}",
        format=image_format,
        save_all=True,
        append_images=[second],
        duration=100,
        **format_options,
    )
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with pytest.raises(ValueError, match="animated or multi-frame"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_rejects_distinct_sources_that_preprocess_identically(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    _write_calibration_png(val / "small.png", (1, 2, 3), size=(20, 20))
    _write_calibration_png(val / "large.png", (1, 2, 3), size=(30, 30))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)

    with pytest.raises(ValueError, match="unique backend tensors"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_encoded_work_limit_fails_before_decoding(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    _write_calibration_png(val / "one.png", (1, 2, 3))
    _write_calibration_png(val / "two.png", (4, 5, 6))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)
    monkeypatch.setattr(dataset_manifest, "_MAX_CALIBRATION_ENCODED_BYTES", 1)
    monkeypatch.setattr(
        dataset_manifest,
        "_validate_calibration_image",
        lambda *_args, **_kwargs: pytest.fail("encoded-byte bound must run before decoding"),
    )

    with pytest.raises(ValueError, match="encoded-work safety limit"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_source_tree_limit_fails_before_image_decoding(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    _write_calibration_png(val / "one.png", (1, 2, 3))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)
    monkeypatch.setattr(dataset_manifest, "_MAX_DATASET_BYTES", 1)
    monkeypatch.setattr(
        dataset_manifest,
        "_calibration_inventory",
        lambda *_args, **_kwargs: pytest.fail("tree bound must run before image decoding"),
    )

    with pytest.raises(ValueError, match="(?:calibration dataset|selected dataset splits) exceed"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_rejects_unknown_process_wide_pillow_hook(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    _write_calibration_png(val / "one.png", (1, 2, 3))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    def untrusted_open(*_args, **_kwargs):
        raise AssertionError("untrusted Pillow hook must not execute")

    monkeypatch.setattr(Image, "open", untrusted_open)
    with pytest.raises(RuntimeError, match="untrusted runtime hook"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


@pytest.mark.parametrize(("mode", "color"), [("L", 1), ("RGBA", (1, 2, 3, 4))])
def test_calibration_rejects_tiff_channel_mismatch(tmp_path, monkeypatch, mode, color):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    Image.new(mode, (10, 10), color).save(val / "mismatch.tiff", format="TIFF")
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with pytest.raises(ValueError, match="HxWx3 uint8"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_rejects_manifest_grayscale_preprocessing(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "val" / "one.png", (1, 2, 3))
    manifest = root / "data.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nchannels: 1\nnames: [drone]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with pytest.raises(ValueError, match="default to 3 channels"):
        snapshot_local_calibration_dataset(manifest, image_size=32)


def test_calibration_snapshot_is_private_read_only_and_source_bound(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    first = train / "one.png"
    second = val / "two.png"
    third = val / "three.png"
    _write_calibration_png(first, (1, 2, 3))
    _write_calibration_png(second, (4, 5, 6))
    _write_calibration_png(third, (7, 8, 9))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)

    snapshot = snapshot_local_calibration_dataset(manifest)
    private_root = snapshot.root
    loader_root = snapshot.loader_root
    private_manifest = snapshot.path
    try:
        assert private_root != root
        assert loader_root != private_root
        assert str(loader_root) in private_manifest.read_text(encoding="utf-8")
        assert (private_root.stat().st_mode & 0o222) == 0
        assert ((private_root / "train" / "one.png").stat().st_mode & 0o222) == 0
        assert (private_root / "train" / "one.png").read_bytes() == first.read_bytes()
        loader_images = sorted((loader_root / "calibration-images").iterdir())
        assert len(loader_images) == 2
        assert {image.read_bytes() for image in loader_images} == {
            second.read_bytes(),
            third.read_bytes(),
        }
        assert not list(loader_root.rglob("*.txt"))
        assert not list(loader_root.rglob("*.cache"))
        bound_digest = snapshot.calibration_digest()

        _write_calibration_png(first, (10, 11, 12))
        assert (private_root / "train" / "one.png").read_bytes() != first.read_bytes()
        assert snapshot.calibration_digest() == bound_digest == snapshot.sha256
        with pytest.raises(RuntimeError, match="source calibration dataset changed"):
            snapshot.assert_source_unchanged()
    finally:
        snapshot.close()
    assert not private_root.exists()
    assert not loader_root.exists()
    assert not private_manifest.exists()


def test_calibration_close_releases_all_resources_after_cleanup_error(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    _write_calibration_png(tmp_path / "val" / "one.png", (1, 2, 3))
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)
    snapshot = snapshot_local_calibration_dataset(manifest)
    released: list[str] = []
    real_loader_close = snapshot._loader_snapshot.close
    real_artifact_close = snapshot._artifact_snapshot.close
    real_root_close = snapshot._source_root.close
    real_manifest_close = snapshot._source_manifest.close

    def fail_loader_close():
        released.append("loader")
        real_loader_close()
        raise OSError("injected loader cleanup failure")

    def close_artifact():
        released.append("artifact")
        real_artifact_close()

    def close_root():
        released.append("root")
        real_root_close()

    def close_manifest():
        released.append("manifest")
        real_manifest_close()

    monkeypatch.setattr(snapshot._loader_snapshot, "close", fail_loader_close)
    monkeypatch.setattr(snapshot._artifact_snapshot, "close", close_artifact)
    monkeypatch.setattr(snapshot._source_root, "close", close_root)
    monkeypatch.setattr(snapshot._source_manifest, "close", close_manifest)

    with pytest.raises(OSError, match="loader cleanup failure"):
        snapshot.close()
    assert released == ["loader", "artifact", "root", "manifest"]
    assert snapshot._closed


def test_calibration_loader_constructor_closes_fd_when_fdopen_fails(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest, fd_io

    opened: list[int] = []

    def fail_fdopen(fd, _mode):
        opened.append(fd)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_fdopen)
    with pytest.raises(OSError, match="fdopen failure"):
        dataset_manifest._CalibrationLoaderSnapshot(
            {"names": ["drone"], "nc": 1},
            tmp_path,
            (),
        )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_manifest_snapshot_constructor_closes_fd_when_fdopen_fails(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest, fd_io

    class SourceBinding:
        closed = False

        def close(self):
            self.closed = True

    source_manifest = SourceBinding()
    source_root = SourceBinding()
    opened: list[int] = []

    def fail_fdopen(fd, _mode):
        opened.append(fd)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_fdopen)
    with pytest.raises(OSError, match="fdopen failure"):
        dataset_manifest.DatasetManifestSnapshot(
            {"path": str(tmp_path)},
            source_manifest,
            source_root,
        )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert source_root.closed
    assert source_manifest.closed


def test_calibration_backend_view_uses_fixed_hash_ranked_subset(tmp_path, monkeypatch):
    import yaml

    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    for index, color in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9))):
        _write_calibration_png(val / f"{index}.png", color)
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 3)
    monkeypatch.setattr(dataset_manifest, "_BACKEND_CALIBRATION_IMAGES", 2)

    validated = validate_local_detection_manifest(manifest)
    try:
        _digest, _content_digest, all_images = validated._calibration_digests(640)
    finally:
        validated.close()
    expected_hashes = tuple(
        image.tensor_sha256
        for image in sorted(all_images, key=lambda image: image.tensor_sha256)[:2]
    )

    with snapshot_local_calibration_dataset(manifest) as snapshot:
        payload = yaml.safe_load(snapshot.path.read_text(encoding="utf-8"))
        assert len(list((snapshot.loader_root / payload["val"]).iterdir())) == 2
        assert tuple(image.tensor_sha256 for image in snapshot._images) == expected_hashes


def test_calibration_snapshot_rejects_source_root_replacement(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "train" / "one.png", (1, 2, 3))
    _write_calibration_png(root / "val" / "two.png", (4, 5, 6))
    _write_calibration_png(root / "val" / "three.png", (7, 8, 9))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)

    snapshot = snapshot_local_calibration_dataset(manifest)
    try:
        original = tmp_path / "original"
        root.rename(original)
        shutil.copytree(original, root)
        with pytest.raises(RuntimeError, match="root was replaced"):
            snapshot.assert_source_unchanged()
    finally:
        snapshot.close()


@pytest.mark.parametrize("source_change", ["content", "replacement"])
def test_calibration_snapshot_detects_external_manifest_change(
    tmp_path, monkeypatch, source_change
):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "val" / "one.png", (1, 2, 3))
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        "path: dataset\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    snapshot = snapshot_local_calibration_dataset(manifest)
    try:
        original = manifest.read_bytes()
        if source_change == "content":
            manifest.write_bytes(original + b"# changed after snapshot\n")
        else:
            manifest.unlink()
            manifest.write_bytes(original)
        with pytest.raises(RuntimeError, match="source calibration manifest changed"):
            snapshot.assert_source_unchanged()
    finally:
        snapshot.close()


def test_manifest_binding_rechecks_visible_leaf_after_descriptor_read(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    manifest = tmp_path / "data.yaml"
    manifest.write_text("trusted", encoding="utf-8")
    attacker = tmp_path / "attacker.yaml"
    attacker.write_text("attacker", encoding="utf-8")
    parked = tmp_path / "parked.yaml"
    binding = dataset_manifest._BoundSourceManifest(manifest)
    original_read = dataset_manifest._read_regular_utf8_at
    replaced = False

    def replace_leaf_after_read(*args, **kwargs):
        nonlocal replaced
        result = original_read(*args, **kwargs)
        if not replaced:
            manifest.rename(parked)
            attacker.rename(manifest)
            replaced = True
        return result

    monkeypatch.setattr(dataset_manifest, "_read_regular_utf8_at", replace_leaf_after_read)
    try:
        with pytest.raises(ValueError, match="manifest path was replaced"):
            binding.assert_unchanged()
    finally:
        binding.close()


def test_calibration_clone_failure_closes_previously_cloned_root(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    manifest = tmp_path / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    validated = validate_local_detection_manifest(manifest)

    class FakeRootBinding:
        closed = False

        def close(self):
            self.closed = True

    cloned_root = FakeRootBinding()
    monkeypatch.setattr(validated._source_root, "clone", lambda: cloned_root)

    def fail_manifest_clone():
        raise OSError("injected descriptor duplication failure")

    monkeypatch.setattr(validated._source_manifest, "clone", fail_manifest_clone)
    try:
        with pytest.raises(OSError, match="descriptor duplication failure"):
            dataset_manifest.CalibrationDatasetSnapshot(validated, 640, None, None)
        assert cloned_root.closed
    finally:
        validated.close()


def test_calibration_manifest_read_is_bound_to_original_parent_descriptor(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    visible = tmp_path / "visible"
    trusted = visible / "dataset"
    (trusted / "train").mkdir(parents=True)
    (trusted / "val").mkdir()
    _write_calibration_png(trusted / "val" / "trusted.png", (1, 2, 3))
    manifest = visible / "data.yaml"
    manifest.write_text(
        "path: dataset\ntrain: train\nval: val\nnames: [trusted]\n",
        encoding="utf-8",
    )

    attacker_parent = tmp_path / "attacker"
    attacker = attacker_parent / "dataset"
    (attacker / "train").mkdir(parents=True)
    (attacker / "val").mkdir()
    _write_calibration_png(attacker / "val" / "attacker.png", (9, 8, 7))
    (attacker_parent / "data.yaml").write_text(
        "path: dataset\ntrain: train\nval: val\nnames: [attacker]\n",
        encoding="utf-8",
    )
    parked = tmp_path / "parked"
    original_read = dataset_manifest._read_regular_utf8_at

    def swap_ancestor_during_read(*args, **kwargs):
        visible.rename(parked)
        attacker_parent.rename(visible)
        try:
            return original_read(*args, **kwargs)
        finally:
            visible.rename(attacker_parent)
            parked.rename(visible)

    monkeypatch.setattr(dataset_manifest, "_read_regular_utf8_at", swap_ancestor_during_read)
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with snapshot_local_calibration_dataset(manifest) as snapshot:
        assert snapshot._images[0].relative_path == (trusted / "val" / "trusted.png").relative_to(
            trusted
        )
        private_images = list((snapshot.loader_root / "calibration-images").iterdir())
        assert [image.read_bytes() for image in private_images] == [
            (trusted / "val" / "trusted.png").read_bytes()
        ]


def test_calibration_tree_copy_ignores_transient_ancestor_replacement(tmp_path, monkeypatch):
    from manwe.common import artifacts, dataset_manifest

    visible = tmp_path / "visible"
    (visible / "train").mkdir(parents=True)
    (visible / "val").mkdir()
    trusted_image = visible / "val" / "trusted.png"
    _write_calibration_png(trusted_image, (1, 2, 3))
    manifest = visible / "data.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [trusted]\n",
        encoding="utf-8",
    )

    attacker = tmp_path / "attacker"
    (attacker / "train").mkdir(parents=True)
    (attacker / "val").mkdir()
    _write_calibration_png(attacker / "val" / "attacker.png", (9, 8, 7))
    (attacker / "data.yaml").write_text(
        "path: .\ntrain: train\nval: val\nnames: [attacker]\n",
        encoding="utf-8",
    )
    parked = tmp_path / "parked"
    original_snapshot = artifacts.ArtifactSnapshot.from_directory_fd.__func__

    def swap_ancestor_during_copy(cls, *args, **kwargs):
        visible.rename(parked)
        attacker.rename(visible)
        try:
            return original_snapshot(cls, *args, **kwargs)
        finally:
            visible.rename(attacker)
            parked.rename(visible)

    monkeypatch.setattr(
        artifacts.ArtifactSnapshot,
        "from_directory_fd",
        classmethod(swap_ancestor_during_copy),
    )
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    with snapshot_local_calibration_dataset(manifest) as snapshot:
        private_images = list((snapshot.loader_root / "calibration-images").iterdir())
        assert [image.read_bytes() for image in private_images] == [trusted_image.read_bytes()]
        snapshot.assert_source_unchanged()


def test_calibration_snapshot_detects_source_parent_replacement(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    visible = tmp_path / "visible"
    (visible / "train").mkdir(parents=True)
    (visible / "val").mkdir()
    _write_calibration_png(visible / "val" / "trusted.png", (1, 2, 3))
    manifest = visible / "data.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [trusted]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    snapshot = snapshot_local_calibration_dataset(manifest)
    original = tmp_path / "original"
    visible.rename(original)
    shutil.copytree(original, visible)
    try:
        with pytest.raises(RuntimeError, match="root was replaced"):
            snapshot.assert_source_unchanged()
    finally:
        snapshot.close()


def test_calibration_source_check_brackets_retained_tree_digest(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "val" / "trusted.png", (1, 2, 3))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [trusted]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)
    snapshot = snapshot_local_calibration_dataset(manifest)

    attacker = tmp_path / "attacker"
    (attacker / "train").mkdir(parents=True)
    (attacker / "val").mkdir()
    _write_calibration_png(attacker / "val" / "attacker.png", (9, 8, 7))
    (attacker / "data.yaml").write_text(
        "path: .\ntrain: train\nval: val\nnames: [attacker]\n",
        encoding="utf-8",
    )
    parked = tmp_path / "parked"
    original_digest = snapshot._source_root.digest

    def replace_visible_root_after_digest():
        digest = original_digest()
        root.rename(parked)
        attacker.rename(root)
        return digest

    monkeypatch.setattr(snapshot._source_root, "digest", replace_visible_root_after_digest)
    try:
        with pytest.raises(RuntimeError, match="root was replaced"):
            snapshot.assert_source_unchanged()
    finally:
        snapshot.close()


def test_calibration_snapshot_detects_backend_view_mutation(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "val" / "one.png", (1, 2, 3))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    snapshot = snapshot_local_calibration_dataset(manifest)
    try:
        snapshot.loader_root.chmod(0o700)
        (snapshot.loader_root / "labels.cache").write_bytes(b"backend mutation")
        with pytest.raises(RuntimeError, match="loader root changed|loader inventory changed"):
            snapshot.calibration_digest()
    finally:
        snapshot.close()


@pytest.mark.heavy
def test_calibration_private_view_matches_actual_pinned_loader(tmp_path, monkeypatch):
    if find_spec("torch") is None or find_spec("ultralytics") is None:
        pytest.skip("requires the locked vision/export runtime")
    from manwe.common import dataset_manifest
    from manwe.common.ultralytics import (
        deterministic_ultralytics_validation_format,
        harden_ultralytics_runtime,
        verify_ultralytics_policy,
    )

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    for index, color in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9))):
        _write_calibration_png(val / f"{index}.png", color, size=(17 + index, 19 + index))
    (root / "val.cache").write_bytes(b"untrusted source cache")
    (val / "0.txt").write_text("malformed label excluded from curated view\n", encoding="utf-8")
    (val / "0.npy").write_bytes(b"untrusted adjacent array")
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 3)
    monkeypatch.setattr(dataset_manifest, "_BACKEND_CALIBRATION_IMAGES", 3)

    harden_ultralytics_runtime()
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    verify_ultralytics_policy()

    def reject_array_or_pickle_cache(*_args, **_kwargs):
        raise AssertionError("the curated loader must never call numpy.load")

    monkeypatch.setattr(np, "load", reject_array_or_pickle_cache)
    with snapshot_local_calibration_dataset(manifest, image_size=32) as snapshot:
        assert not list(snapshot.loader_root.rglob("*.cache"))
        assert not list(snapshot.loader_root.rglob("*.txt"))
        assert not list(snapshot.loader_root.rglob("*.npy"))
        with deterministic_ultralytics_validation_format():
            data = check_det_dataset(str(snapshot.path))
            cfg = get_cfg(
                DEFAULT_CFG,
                {
                    "imgsz": 32,
                    "batch": 1,
                    "task": "detect",
                    "rect": False,
                    "cache": False,
                    "fraction": 1.0,
                },
            )
            dataset = build_yolo_dataset(
                cfg,
                data["val"],
                1,
                data,
                mode="val",
                fraction=1.0,
            )
            loader = build_dataloader(dataset, batch=1, workers=0, drop_last=True)
            actual_hashes = set()
            for batch in loader:
                tensor = batch["img"].numpy()[0]
                hasher = hashlib.sha256(b"manwe-ultralytics-8.4.92-calibration-tensor-v1\0")
                hasher.update((32).to_bytes(8, "big"))
                hasher.update(tensor.tobytes())
                actual_hashes.add(hasher.hexdigest())

        assert actual_hashes == {image.tensor_sha256 for image in snapshot._images}
        assert not list(snapshot.loader_root.rglob("*.cache"))
        snapshot.calibration_digest()


@pytest.mark.parametrize(
    ("suffix", "image_format", "mode"),
    (
        (".jpg", "JPEG", "RGB"),
        (".bmp", "BMP", "RGB"),
        (".webp", "WEBP", "RGB"),
        (".tiff", "TIFF", "RGB"),
        (".png", "PNG", "P"),
        (".png", "PNG", "RGBA"),
    ),
)
@pytest.mark.heavy
def test_calibration_supported_formats_match_actual_pinned_loader(
    tmp_path,
    monkeypatch,
    suffix,
    image_format,
    mode,
):
    if find_spec("torch") is None or find_spec("ultralytics") is None:
        pytest.skip("requires the locked vision/export runtime")
    from manwe.common import dataset_manifest
    from manwe.common.ultralytics import (
        deterministic_ultralytics_validation_format,
        harden_ultralytics_runtime,
        verify_ultralytics_policy,
    )

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    if mode == "P":
        image = Image.new(mode, (19, 17), 1)
        palette = [0] * 768
        palette[3:6] = [1, 2, 3]
        image.putpalette(palette)
    elif mode == "RGBA":
        image = Image.new(mode, (19, 17), (1, 2, 3, 127))
    else:
        image = Image.new(mode, (19, 17), (1, 2, 3))
    image.save(val / f"one{suffix}", format=image_format)
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)
    monkeypatch.setattr(dataset_manifest, "_BACKEND_CALIBRATION_IMAGES", 1)

    harden_ultralytics_runtime()
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    verify_ultralytics_policy()
    with snapshot_local_calibration_dataset(manifest, image_size=32) as snapshot:
        with deterministic_ultralytics_validation_format():
            data = check_det_dataset(str(snapshot.path))
            cfg = get_cfg(
                DEFAULT_CFG,
                {
                    "imgsz": 32,
                    "batch": 1,
                    "task": "detect",
                    "rect": False,
                    "cache": False,
                    "fraction": 1.0,
                },
            )
            dataset = build_yolo_dataset(
                cfg,
                data["val"],
                1,
                data,
                mode="val",
                fraction=1.0,
            )
            loader = build_dataloader(dataset, batch=1, workers=0, drop_last=True)
            batch = next(iter(loader))
            tensor = batch["img"].numpy()[0]

        hasher = hashlib.sha256(b"manwe-ultralytics-8.4.92-calibration-tensor-v1\0")
        hasher.update((32).to_bytes(8, "big"))
        hasher.update(tensor.tobytes())
        assert hasher.hexdigest() == snapshot._images[0].tensor_sha256
        assert not list(snapshot.loader_root.rglob("*.cache"))


def test_dataset_manifest_rejects_split_outside_declared_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    manifest = tmp_path / "data.yaml"
    manifest.write_text(
        f"path: {root}\ntrain: {outside}\nval: {root}\nnames: [drone]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside the declared dataset root"):
        validate_local_detection_manifest(manifest)


def test_tensorrt_int8_preflight_rejects_pinned_silent_fp32_branch(
    monkeypatch,
):
    backends = importlib.import_module("manwe.export.backends")
    selected_devices = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(set_device=selected_devices.append)),
    )

    class FakeLogger:
        ERROR = 1

        def __init__(self, _severity):
            pass

    class NoInt8Builder:
        platform_has_fast_int8 = False

        def __init__(self, _logger):
            pass

    monkeypatch.setitem(
        sys.modules,
        "tensorrt",
        SimpleNamespace(
            __version__="8.6.1",
            Logger=FakeLogger,
            Builder=NoInt8Builder,
        ),
    )
    with pytest.raises(RuntimeError, match="silently build FP32"):
        backends._preflight_tensorrt_int8(2)
    assert selected_devices == [2]

    class TensorRT10Builder:
        def __init__(self, _logger):
            pass

    monkeypatch.setitem(
        sys.modules,
        "tensorrt",
        SimpleNamespace(
            __version__="10.1.0",
            Logger=FakeLogger,
            Builder=TensorRT10Builder,
        ),
    )
    assert backends._preflight_tensorrt_int8(1) == ("10.1.0", None)
    assert selected_devices == [2, 1]

    for version, message in (("10.2.0", "10.2"), ("12.0.0", "newer than the audited")):
        monkeypatch.setitem(
            sys.modules,
            "tensorrt",
            SimpleNamespace(
                __version__=version,
                Logger=FakeLogger,
                Builder=TensorRT10Builder,
            ),
        )
        with pytest.raises(RuntimeError, match=message):
            backends._preflight_tensorrt_int8(0)

    for name in ("modelopt", "modelopt.onnx", "modelopt.onnx.quantization"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(
        backends.importlib.metadata,
        "version",
        lambda name: (
            "0.44.2"
            if name == "nvidia-modelopt"
            else pytest.fail(f"unexpected distribution lookup: {name}")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt",
        SimpleNamespace(
            __version__="11.0.0",
            Logger=FakeLogger,
            Builder=TensorRT10Builder,
        ),
    )
    assert backends._preflight_tensorrt_int8(3) == ("11.0.0", "0.44.2")
    for invalid_version, message in (
        ("0.44.0rc5", "nvidia-modelopt>=0.44"),
        ("0.44.dev1", "nvidia-modelopt>=0.44"),
        ("not-a-version", "PEP 440"),
    ):
        monkeypatch.setattr(
            backends.importlib.metadata,
            "version",
            lambda name, value=invalid_version: (
                value
                if name == "nvidia-modelopt"
                else pytest.fail(f"unexpected distribution lookup: {name}")
            ),
        )
        with pytest.raises(RuntimeError, match=message):
            backends._preflight_tensorrt_int8(3)


@pytest.mark.parametrize("source_change", ["transient-replacement", "persistent-mutation"])
def test_int8_export_uses_digest_bound_private_snapshot_and_fails_closed(
    tmp_path, monkeypatch, source_change
):
    import yaml

    from manwe.common import dataset_manifest
    from manwe.common import device as device_module

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    train_image = root / "train" / "one.png"
    val_image_one = root / "val" / "two.png"
    val_image_two = root / "val" / "three.png"
    _write_calibration_png(train_image, (1, 2, 3))
    _write_calibration_png(val_image_one, (4, 5, 6))
    _write_calibration_png(val_image_two, (7, 8, 9))
    train_bytes = train_image.read_bytes()
    val_bytes = {val_image_one.read_bytes(), val_image_two.read_bytes()}
    (root / "val.cache").write_bytes(b"untrusted pickle-shaped cache")
    (root / "val" / "two.txt").write_text("malformed label ignored by curated view\n")
    (root / "val" / "two.npy").write_bytes(b"untrusted adjacent array cache")
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)
    with snapshot_local_calibration_dataset(
        manifest,
        image_size=320,
        tensorrt_version="10.1.0",
    ) as expected_snapshot:
        expected_digest = expected_snapshot.sha256

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    weights_digest = hashlib.sha256(b"fixture").hexdigest()
    output = tmp_path / "model.engine"
    observed = {}
    backends = importlib.import_module("manwe.export.backends")
    monkeypatch.setattr(backends, "harden_ultralytics_runtime", lambda: None)
    monkeypatch.setattr(backends, "verify_ultralytics_policy", lambda: None)
    monkeypatch.setattr(
        backends,
        "deterministic_ultralytics_validation_format",
        nullcontext,
    )
    monkeypatch.setattr(
        backends,
        "_preflight_tensorrt_int8",
        lambda _index: ("10.1.0", None),
    )
    monkeypatch.setattr(device_module, "resolve_device", lambda _device: Device("cuda", index=0))

    class FakeModel:
        task = "detect"
        model = SimpleNamespace(
            end2end=False,
            stride=[8, 16, 32],
            yaml={"channels": 3},
        )
        names = {0: "drone"}

        def __init__(self, model_path):
            self.model_path = model_path

        def export(self, **kwargs):
            private_manifest = type(manifest)(kwargs["data"])
            payload = yaml.safe_load(private_manifest.read_text(encoding="utf-8"))
            private_root = type(root)(payload["path"])
            observed["private_manifest"] = private_manifest
            observed["private_root"] = private_root
            assert private_root != root
            assert payload["train"] == payload["val"] == payload["test"] == "calibration-images"
            assert payload["channels"] == 3
            private_images = sorted((private_root / payload["val"]).iterdir())
            assert len(private_images) == 2
            assert {image.read_bytes() for image in private_images} == val_bytes
            assert train_bytes not in {image.read_bytes() for image in private_images}
            assert not list(private_root.rglob("*.txt"))
            assert not list(private_root.rglob("*.cache"))
            assert not list(private_root.rglob("*.npy"))
            assert (private_root.stat().st_mode & 0o222) == 0
            assert kwargs["batch"] == 1
            assert kwargs["fraction"] == 1.0
            assert kwargs["rect"] is False
            assert kwargs["dynamic"] is False

            if source_change == "transient-replacement":
                original = tmp_path / "source-original"
                root.rename(original)
                try:
                    (root / "train").mkdir(parents=True)
                    (root / "val").mkdir()
                    _write_calibration_png(root / "train" / "one.png", (10, 11, 12))
                    _write_calibration_png(root / "val" / "two.png", (13, 14, 15))
                    _write_calibration_png(root / "val" / "three.png", (16, 17, 18))
                    (root / "data.yaml").write_text(
                        "path: .\ntrain: train\nval: val\nnames: [attacker]\n",
                        encoding="utf-8",
                    )
                    assert {image.read_bytes() for image in private_images} == val_bytes
                finally:
                    shutil.rmtree(root)
                    original.rename(root)
            else:
                _write_calibration_png(root / "train" / "one.png", (10, 11, 12))
                assert {image.read_bytes() for image in private_images} == val_bytes

            produced = type(checkpoint)(self.model_path).with_suffix(".engine")
            produced.write_bytes(b"engine")
            return produced

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    export_kwargs = {
        "output": str(output),
        "weights_sha256": weights_digest,
        "allow_pickle_checkpoint": True,
        "int8": True,
        "data": str(manifest),
        "device": "cuda:0",
        "imgsz": 320,
    }
    if source_change == "persistent-mutation":
        with pytest.raises(RuntimeError, match="source calibration dataset changed"):
            export_model(str(checkpoint), ["tensorrt"], **export_kwargs)
        assert not output.exists()
    else:
        receipt = export_model(str(checkpoint), ["tensorrt"], **export_kwargs)
        assert receipt.calibration_manifest_sha256 == expected_digest
        assert output.read_bytes() == b"engine"
    assert not observed["private_manifest"].exists()
    assert not observed["private_root"].exists()


def test_int8_export_rejects_non_three_channel_checkpoint(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest
    from manwe.common import device as device_module

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "val" / "one.png", (1, 2, 3))
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 1)

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "model.engine"
    backends = importlib.import_module("manwe.export.backends")
    monkeypatch.setattr(
        backends,
        "_preflight_tensorrt_int8",
        lambda _index: ("10.1.0", None),
    )
    monkeypatch.setattr(backends, "harden_ultralytics_runtime", lambda: None)
    monkeypatch.setattr(backends, "verify_ultralytics_policy", lambda: None)
    monkeypatch.setattr(
        backends,
        "deterministic_ultralytics_validation_format",
        nullcontext,
    )
    monkeypatch.setattr(device_module, "resolve_device", lambda _device: Device("cuda", index=0))

    class FakeModel:
        task = "detect"
        model = SimpleNamespace(
            end2end=False,
            stride=[8, 16, 32],
            yaml={"channels": 1},
        )
        names = {0: "drone"}

        def __init__(self, _model_path):
            pass

        def export(self, **_kwargs):
            raise AssertionError("channel mismatch must fail before backend export")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    with pytest.raises(ValueError, match="3-channel detector checkpoints"):
        export_model(
            str(checkpoint),
            ["tensorrt"],
            output=str(output),
            weights_sha256=hashlib.sha256(b"fixture").hexdigest(),
            allow_pickle_checkpoint=True,
            int8=True,
            data=str(manifest),
            device="cuda:0",
        )
    assert not output.exists()


def test_export_rejects_unproducible_coreml_suffix_before_loading_model(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    with pytest.raises(ValueError, match="output suffix"):
        export_model(
            str(weights),
            ["coreml"],
            output=str(tmp_path / "model.mlmodelc"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
        )


@pytest.mark.parametrize("output", ["published\n.onnx", "published\r.onnx", "bad\0.onnx"])
def test_export_destination_rejects_control_characters_before_path_creation(tmp_path, output):
    from manwe.export.backends import _prepare_destination

    with pytest.raises(ValueError, match="bounded nonempty destination"):
        _prepare_destination(str(tmp_path / output), {".onnx"})
    assert list(tmp_path.iterdir()) == []


def test_directory_export_failure_preserves_private_partial_for_recovery(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "Manifest.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "published.mlpackage"
    digest = sha256_artifact(source)
    stage = tmp_path / f".manwe-export-stage-{'1' * 48}"

    def fail_publication(_src, destination_fd):
        fd = os.open(
            "partial",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_fd,
        )
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"partial")
        raise OSError("injected publication failure")

    monkeypatch.setattr(backends, "_copy_directory_fd_relative", fail_publication)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(OSError, match="injected publication failure") as captured:
        _publish_exclusive(source, destination, digest)
    _assert_exception_note(captured.value, "No automatic deletion was attempted")
    assert not destination.exists()
    assert (stage / "partial").read_bytes() == b"partial"
    assert stat.S_IMODE(stage.stat().st_mode) == 0o700


def test_publication_preserves_original_error_and_recovery_evidence(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "Manifest.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "published.mlpackage"
    digest = sha256_artifact(source)
    stage = tmp_path / f".manwe-export-stage-{'2' * 48}"

    def fail_publication(_source, destination_fd):
        fd = os.open(
            "partial",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_fd,
        )
        os.close(fd)
        raise ValueError("injected publication failure")

    monkeypatch.setattr(backends, "_copy_directory_fd_relative", fail_publication)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(ValueError, match="injected publication failure") as captured:
        _publish_exclusive(source, destination, digest)
    _assert_exception_note(
        captured.value,
        "No automatic deletion was attempted",
        destination.name,
        str(destination.parent),
    )
    assert not destination.exists()
    assert stage.is_dir()
    assert (stage / "partial").is_file()


def test_file_publication_rejects_private_snapshot_growth(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.onnx"
    source.write_bytes(b"trusted")
    destination = tmp_path / "published.onnx"
    digest = sha256_artifact(source)
    stage = tmp_path / f".manwe-export-stage-{'3' * 48}"
    real_copy = backends._copy_regular_snapshot

    class GrowingReader:
        def __init__(self, handle, path):
            self._handle = handle
            self._path = path
            self._grown = False

        def fileno(self):
            return self._handle.fileno()

        def read(self, size):
            if not self._grown:
                self._path.chmod(0o600)
                with self._path.open("ab") as writer:
                    writer.write(b"x")
                self._grown = True
            return self._handle.read(size)

    def grow_during_copy(source_handle, destination_handle, *, display):
        return real_copy(
            GrowingReader(source_handle, display),
            destination_handle,
            display=display,
        )

    monkeypatch.setattr(backends, "_copy_regular_snapshot", grow_during_copy)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(RuntimeError, match="grew while being published") as captured:
        _publish_exclusive(source, destination, digest)
    _assert_exception_note(captured.value, "No automatic deletion was attempted")
    assert not destination.exists()
    assert stage.read_bytes() == b""
    assert stat.S_IMODE(stage.stat().st_mode) == 0o600


def test_directory_export_replacement_cannot_redirect_descriptor_relative_writes(
    tmp_path, monkeypatch
):
    from manwe.common import artifacts
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    destination = tmp_path / "published.mlpackage"
    stage = tmp_path / f".manwe-export-stage-{'4' * 48}"
    moved = tmp_path / "moved-original.mlpackage"
    digest = sha256_artifact(source)
    real_fsync = artifacts.os.fsync
    replaced = False

    def replace_root_during_copy(fd):
        nonlocal replaced
        metadata = os.fstat(fd)
        if not replaced and stage.exists() and stat.S_ISREG(metadata.st_mode):
            stage.rename(moved)
            stage.mkdir()
            (stage / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return real_fsync(fd)

    monkeypatch.setattr(artifacts.os, "fsync", replace_root_during_copy)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(RuntimeError, match="replaced"):
        backends._publish_exclusive(source, destination, digest)
    assert not destination.exists()
    assert (stage / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not (stage / "weights.bin").exists()
    assert (moved / "weights.bin").read_bytes() == b"trusted"


def test_directory_export_binds_the_directory_created_before_copy(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    destination = tmp_path / "published.mlpackage"
    stage = tmp_path / f".manwe-export-stage-{'5' * 48}"
    moved = tmp_path / "moved-created.mlpackage"
    digest = sha256_artifact(source)
    real_open = backends.os.open
    replaced = False

    def replace_before_directory_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if (
            not replaced
            and path == stage.name
            and kwargs.get("dir_fd") is not None
            and stage.is_dir()
        ):
            stage.rename(moved)
            stage.mkdir()
            (stage / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(backends.os, "open", replace_before_directory_open)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(RuntimeError, match="replaced while it was being opened"):
        backends._publish_exclusive(source, destination, digest)
    assert not destination.exists()
    assert (stage / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert list(moved.iterdir()) == []


def test_directory_export_failure_never_deletes_foreign_replacement(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    destination = tmp_path / "published.mlpackage"
    stage = tmp_path / f".manwe-export-stage-{'6' * 48}"
    moved = tmp_path / "moved-partial.mlpackage"
    digest = sha256_artifact(source)

    def fail_copy(_source, destination_fd):
        fd = os.open("partial", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=destination_fd)
        os.close(fd)
        stage.rename(moved)
        stage.mkdir()
        (stage / "foreign.txt").write_text("foreign", encoding="utf-8")
        raise ValueError("injected publication failure")

    monkeypatch.setattr(backends, "_copy_directory_fd_relative", fail_copy)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(ValueError, match="injected publication failure") as captured:
        backends._publish_exclusive(source, destination, digest)
    _assert_exception_note(captured.value, "No automatic deletion was attempted")
    assert not destination.exists()
    assert (stage / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (moved / "partial").is_file()


def test_directory_export_copy_uses_bounded_descriptor_depth(tmp_path):
    resource = pytest.importorskip("resource")
    from manwe.common.artifacts import sha256_artifact

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    for index in range(80):
        directory = source / f"directory-{index:03d}"
        directory.mkdir()
        (directory / "weights.bin").write_bytes(str(index).encode())
    destination = tmp_path / "published.mlpackage"
    digest = sha256_artifact(source)
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    constrained_limit = min(64, soft_limit)
    if constrained_limit < 32:
        pytest.skip("open-file limit is already too small for a stable pytest process")
    resource.setrlimit(resource.RLIMIT_NOFILE, (constrained_limit, hard_limit))
    try:
        _publish_exclusive(source, destination, digest)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
    assert sha256_artifact(destination) == digest


def test_export_publication_rechecks_identity_after_digest(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.onnx"
    source.write_bytes(b"trusted")
    destination = tmp_path / "published.onnx"
    stage = tmp_path / f".manwe-export-stage-{'7' * 48}"
    moved = tmp_path / "moved-published.onnx"
    digest = sha256_artifact(source)
    original_hash = backends.sha256_artifact_at

    def replace_after_hash(parent_fd, name):
        value = original_hash(parent_fd, name)
        stage.rename(moved)
        stage.write_bytes(b"attacker")
        return value

    monkeypatch.setattr(backends, "sha256_artifact_at", replace_after_hash)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage.name)
    with pytest.raises(RuntimeError, match="digest was being verified"):
        backends._publish_exclusive(source, destination, digest)
    assert not destination.exists()
    assert stage.read_bytes() == b"attacker"
    assert moved.read_bytes() == b"trusted"


def test_export_parent_replacement_cannot_redirect_publication(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.onnx"
    source.write_bytes(b"trusted")
    parent = tmp_path / "output"
    parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    destination = parent / "published.onnx"
    stage_name = f".manwe-export-stage-{'8' * 48}"
    digest = sha256_artifact(source)
    prepared = backends._prepare_destination(str(destination), {".onnx"})
    real_copy = backends._copy_regular_snapshot
    replaced = False

    def replace_parent_during_copy(source_handle, destination_handle, *, display):
        nonlocal replaced
        if not replaced:
            parent.rename(moved_parent)
            parent.mkdir()
            (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return real_copy(source_handle, destination_handle, display=display)

    monkeypatch.setattr(backends, "_copy_regular_snapshot", replace_parent_during_copy)
    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: stage_name)

    try:
        with pytest.raises(RuntimeError, match="parent was replaced") as captured:
            backends._publish_exclusive(source, prepared, digest)
    finally:
        prepared.close()
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not (parent / destination.name).exists()
    preserved = moved_parent / stage_name
    assert preserved.read_bytes() == b"trusted"
    assert stat.S_IMODE(preserved.stat().st_mode) == 0o600
    _assert_exception_note(captured.value, "No automatic deletion was attempted")


def test_ultralytics_policy_disables_install_network_and_analytics(tmp_path, monkeypatch):
    policy = importlib.import_module("manwe.common.ultralytics")
    pristine_pillow_open = Image.open

    def unsafe_pillow_open(*_args, **_kwargs):
        raise AssertionError("Ultralytics optional-codec hook must be removed")

    monkeypatch.setattr(Image, "open", unsafe_pillow_open)
    root = ModuleType("ultralytics")
    root.SETTINGS = {"sync": True, "hub": True}  # type: ignore[attr-defined]
    utils = ModuleType("ultralytics.utils")
    utils.AUTOINSTALL = True  # type: ignore[attr-defined]
    utils.ONLINE = True  # type: ignore[attr-defined]
    checks = ModuleType("ultralytics.utils.checks")
    checks.AUTOINSTALL = True  # type: ignore[attr-defined]
    checks.ONLINE = True  # type: ignore[attr-defined]
    downloads = ModuleType("ultralytics.utils.downloads")
    downloads.safe_download = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    downloads.is_url = lambda *_args, **_kwargs: True  # type: ignore[attr-defined]
    data_utils = ModuleType("ultralytics.data.utils")
    data_utils.load_dataset_cache_file = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
    data_utils.save_dataset_cache_file = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    data_dataset = ModuleType("ultralytics.data.dataset")
    data_dataset.load_dataset_cache_file = data_utils.load_dataset_cache_file  # type: ignore[attr-defined]
    data_dataset.save_dataset_cache_file = data_utils.save_dataset_cache_file  # type: ignore[attr-defined]

    class FakeFormat:
        bgr = 0.0

        def _format_img(self, image):
            return image

    augment = ModuleType("ultralytics.data.augment")
    augment.Format = FakeFormat  # type: ignore[attr-defined]
    patches = ModuleType("ultralytics.utils.patches")
    patches._image_open = pristine_pillow_open  # type: ignore[attr-defined]
    tasks = ModuleType("ultralytics.nn.tasks")
    tasks.SAFE_LOAD = False  # type: ignore[attr-defined]
    tasks._SafeLoad = SimpleNamespace(SUPPORTED=True)  # type: ignore[attr-defined]
    events_module = ModuleType("ultralytics.utils.events")
    events_module.events = SimpleNamespace(enabled=True, events=[{"queued": True}])  # type: ignore[attr-defined]
    for name, module in {
        "ultralytics": root,
        "ultralytics.utils": utils,
        "ultralytics.utils.checks": checks,
        "ultralytics.utils.downloads": downloads,
        "ultralytics.data.utils": data_utils,
        "ultralytics.data.dataset": data_dataset,
        "ultralytics.data.augment": augment,
        "ultralytics.utils.patches": patches,
        "ultralytics.nn.tasks": tasks,
        "ultralytics.utils.events": events_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Track the policy's own constant rather than restating the version here, so a
    # vetted-version bump cannot leave a stale duplicate behind in the tests.
    vetted = policy._VETTED_ULTRALYTICS_VERSION
    monkeypatch.setattr(policy.importlib.metadata, "version", lambda _name: vetted)

    policy.verify_ultralytics_policy()
    assert root.SETTINGS == {"sync": False, "hub": False}  # type: ignore[attr-defined]
    assert checks.AUTOINSTALL is False and checks.ONLINE is False  # type: ignore[attr-defined]
    assert events_module.events.enabled is False  # type: ignore[attr-defined]
    assert Image.open is pristine_pillow_open
    assert downloads.is_url("https://example.invalid", check=False) is True  # type: ignore[attr-defined]
    assert downloads.is_url("https://example.invalid", check=True) is False  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="network downloads are disabled"):
        downloads.safe_download("https://example.invalid/model.pt")  # type: ignore[attr-defined]
    with pytest.raises(FileNotFoundError, match="cache loading is disabled"):
        data_dataset.load_dataset_cache_file(tmp_path / "labels.cache")  # type: ignore[attr-defined]
    cache_payload = {}
    cache_path = tmp_path / "labels.cache"
    data_dataset.save_dataset_cache_file("", cache_path, cache_payload, "1.0")  # type: ignore[attr-defined]
    assert cache_payload == {"version": "1.0"}
    assert not cache_path.exists()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda value: value))
    original_format = FakeFormat._format_img
    with policy.deterministic_ultralytics_validation_format():
        formatted = FakeFormat()._format_img(np.array([[[1, 2, 3]]], dtype=np.uint8))
        assert formatted[:, 0, 0].tolist() == [3, 2, 1]
        nondeterministic = FakeFormat()
        nondeterministic.bgr = 0.5
        original_image = np.array([[[1, 2, 3]]], dtype=np.uint8)
        assert nondeterministic._format_img(original_image) is original_image
    assert FakeFormat._format_img is original_format

    # The vetted-version gate is a safety control: an unvetted runtime must be
    # rejected outright, never merely hardened.
    monkeypatch.setattr(policy.importlib.metadata, "version", lambda _name: "8.4.0")
    with pytest.raises(RuntimeError, match="not the vetted"):
        policy.verify_ultralytics_policy()


def test_deterministic_ultralytics_formatter_serializes_overlapping_contexts(monkeypatch):
    policy = importlib.import_module("manwe.common.ultralytics")

    class FakeFormat:
        bgr = 1.0

        def _format_img(self, image):
            return image

    augment = ModuleType("ultralytics.data.augment")
    augment.Format = FakeFormat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics.data.augment", augment)
    original = FakeFormat._format_img
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []
    observer_results: list[object] = []
    callback_results: list[np.ndarray] = []

    def first_context() -> None:
        try:
            with policy.deterministic_ultralytics_validation_format():
                first_entered.set()
                if not release_first.wait(5):
                    raise TimeoutError("first formatter context was not released")
        except BaseException as exc:
            errors.append(exc)

    def second_context() -> None:
        try:
            if not first_entered.wait(5):
                raise TimeoutError("first formatter context never started")
            second_started.set()
            with policy.deterministic_ultralytics_validation_format():
                second_entered.set()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=first_context)
    second = threading.Thread(target=second_context)
    first.start()
    assert first_entered.wait(5)
    observer = threading.Thread(
        target=lambda: observer_results.append(FakeFormat()._format_img("original-thread-value"))
    )
    observer.start()
    observer.join(5)
    assert not observer.is_alive()
    assert observer_results == ["original-thread-value"]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda value: value))

    def callback_thread() -> None:
        formatter = FakeFormat()
        formatter.bgr = 0.0
        callback_results.append(formatter._format_img(np.array([[[1, 2, 3]]], dtype=np.uint8)))

    callback = threading.Thread(target=callback_thread)
    callback.start()
    callback.join(5)
    assert not callback.is_alive()
    assert callback_results[0][:, 0, 0].tolist() == [3, 2, 1]
    second.start()
    assert second_started.wait(5)
    assert not second_entered.wait(0.1)
    release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert second_entered.is_set()
    assert FakeFormat._format_img is original
    with (
        pytest.raises(RuntimeError, match="fixture export failure"),
        policy.deterministic_ultralytics_validation_format(),
    ):
        raise RuntimeError("fixture export failure")
    assert FakeFormat._format_img is original
    with (
        pytest.raises(RuntimeError, match="changed during calibration"),
        policy.deterministic_ultralytics_validation_format(),
    ):
        FakeFormat._format_img = original
    assert FakeFormat._format_img is original


def test_logger_cannot_mutate_the_process_root_logger():
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    with pytest.raises(ValueError, match="logger name"):
        get_logger("root", level="DEBUG")
    with pytest.raises(TypeError, match="log level"):
        configure_logging(0)  # type: ignore[arg-type]
    assert (list(root.handlers), root.level) == before


def test_device_contract_rejects_bad_types_and_prefers_available_fallback(monkeypatch):
    class StringSubclass(str):
        def strip(self, *args, **kwargs):
            raise AssertionError("string subclass callback must not run")

    with pytest.raises(TypeError, match="preference"):
        resolve_device([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="preference"):
        resolve_device(StringSubclass("auto"))
    with pytest.raises(TypeError, match="allow_fallback"):
        resolve_device("auto", allow_fallback="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device kind"):
        Device("other")  # type: ignore[arg-type]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: f"GPU {index}",
            get_device_capability=lambda index: (8, 0),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("cuda:7", allow_fallback=True).torch_device == "cuda:0"
    assert resolve_device("mps", allow_fallback=True).torch_device == "cuda:0"


def test_inference_input_is_one_owned_bounded_local_image(tmp_path):
    source = np.zeros((4, 5, 3), dtype=np.uint8)
    prepared = prepare_single_image(source)
    source[0, 0, 0] = 255
    assert prepared.shape == (4, 5, 3)
    assert prepared[0, 0, 0] == 0
    with pytest.raises(ValueError, match="uint8"):
        prepare_single_image(np.zeros((4, 5, 3), dtype=float))
    with pytest.raises(ValueError, match="URL or stream"):
        prepare_single_image("https://example.invalid/image.jpg")
    with pytest.raises(ValueError, match="videos are rejected"):
        prepare_single_image("clip.mp4")
    with pytest.raises(TypeError, match="one uint8"):
        prepare_single_image([source])

    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    prepared_rgb = prepare_single_image(rgb)
    np.testing.assert_array_equal(prepared_rgb, rgb)
    observed_images = []
    observed_kwargs = []

    class FakeTensor:
        def __init__(self, value):
            self.value = value
            self.shape = value.shape

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeModel:
        task = "detect"
        names = {0: "drone"}

        def predict(self, image, **kwargs):
            observed_images.append(image)
            observed_kwargs.append(kwargs)
            boxes = SimpleNamespace(
                xyxy=FakeTensor(np.empty((0, 4))),
                conf=FakeTensor(np.empty(0)),
                cls=FakeTensor(np.empty(0)),
            )
            return [SimpleNamespace(boxes=boxes)]

    detector = Detector.__new__(Detector)
    detector._closed = False
    detector.model = FakeModel()
    detector._model_names = ("drone",)
    detector.device = Device("cpu")
    detector.conf = 0.25
    detector.iou = 0.45
    assert detector.detect(rgb) == []
    expected_bgr = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(observed_images[0], expected_bgr)
    assert observed_kwargs[0] == {
        "conf": 0.25,
        "iou": 0.45,
        "max_det": 300,
        "device": "cpu",
        "verbose": False,
    }
    assert observed_images[0].flags.c_contiguous
    assert observed_images[0].flags.owndata
    assert not np.shares_memory(observed_images[0], rgb)
    np.testing.assert_array_equal(rgb, np.array([[[255, 0, 0], [0, 0, 255]]]))

    class MutatingModel(FakeModel):
        def predict(self, image, **kwargs):
            results = super().predict(image, **kwargs)
            self.names = {0: "airplane"}
            return results

    detector.model = MutatingModel()
    with pytest.raises(RuntimeError, match="class-name mapping changed during inference"):
        detector.detect(rgb)
    detector.model = FakeModel()

    image_module = pytest.importorskip("PIL.Image")
    color_path = tmp_path / "colors.png"
    image_module.fromarray(rgb).save(color_path)
    assert detector.detect(color_path) == []
    np.testing.assert_array_equal(observed_images[1], observed_images[0])

    first = image_module.new("RGB", (2, 2), "black")
    second = image_module.new("RGB", (2, 2), "white")
    multipage = tmp_path / "multipage.tiff"
    first.save(multipage, save_all=True, append_images=[second])
    with pytest.raises(ValueError, match="animated or multi-page"):
        prepare_single_image(multipage)

    animated = tmp_path / "animated.png"
    first.save(animated, save_all=True, append_images=[second], duration=10, loop=0)
    with pytest.raises(ValueError, match="animated or multi-page"):
        prepare_single_image(animated)

    config = SliceConfig()
    with pytest.raises(AttributeError):
        config.slice_height = 1  # type: ignore[misc]


def test_export_preflight_fails_before_loading_optional_dependencies(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    with pytest.raises(ValueError, match="exactly one"):
        export_model(
            str(checkpoint),
            ["onnx", "coreml"],
            output=str(tmp_path / "multi.onnx"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
        )
    with pytest.raises(NotImplementedError, match="not implemented"):
        export_model(
            str(checkpoint),
            ["mlx"],
            output=str(tmp_path / "model.safetensors"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
        )
    with pytest.raises(TypeError, match="formats must be a list"):
        export_model(  # type: ignore[arg-type]
            str(checkpoint),
            "onnx",
            output=str(tmp_path / "bad-list.onnx"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
        )
    with pytest.raises(TypeError, match="half must be a boolean"):
        export_model(
            str(checkpoint),
            ["onnx"],
            output=str(tmp_path / "bad-half.onnx"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
            half="false",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="verified only for TensorRT"):
        export_model(
            str(checkpoint),
            ["coreml"],
            output=str(tmp_path / "bad-int8.mlpackage"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
            int8=True,
            data="calibration.yaml",
        )
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        export_model(
            str(checkpoint),
            ["tensorrt"],
            output=str(tmp_path / "cpu.engine"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
            device="cpu",
        )
    with pytest.raises(ValueError, match="pickle-backed"):
        export_model(
            str(checkpoint),
            ["onnx"],
            output=str(tmp_path / "pickle.onnx"),
            weights_sha256=digest,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        export_model(
            str(checkpoint),
            ["onnx"],
            output=str(tmp_path / "digest.onnx"),
            weights_sha256="0" * 64,
            allow_pickle_checkpoint=True,
        )


def test_export_uses_pinned_backend_kwargs_and_validates_model_task(tmp_path, monkeypatch):
    backends = importlib.import_module("manwe.export.backends")
    monkeypatch.setattr(backends, "harden_ultralytics_runtime", lambda: None)
    monkeypatch.setattr(backends, "verify_ultralytics_policy", lambda: None)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    output = tmp_path / "published.onnx"
    observed = {}

    class FakeModel:
        task = "detect"
        model = SimpleNamespace(end2end=False, stride=[8, 16, 32])
        names = {0: "drone"}

        def __init__(self, model_path):
            self.model_path = model_path

        def export(self, **kwargs):
            observed.update(kwargs)
            produced = type(checkpoint)(self.model_path).with_suffix(".onnx")
            produced.write_bytes(b"onnx")
            return produced

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    receipt = export_model(
        str(checkpoint),
        ["onnx"],
        output=str(output),
        weights_sha256=digest,
        allow_pickle_checkpoint=True,
        device="cpu",
    )
    assert receipt.artifact_path == str(output)
    assert receipt.source_sha256 == digest
    assert output.read_bytes() == b"onnx"
    assert observed == {
        "format": "onnx",
        "imgsz": 640,
        "half": False,
        "device": "cpu",
        "nms": False,
        "opset": 17,
    }

    class WrongTask(FakeModel):
        task = "segment"

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=WrongTask))
    with pytest.raises(ValueError, match="checkpoint task"):
        export_model(
            str(checkpoint),
            ["onnx"],
            output=str(tmp_path / "wrong-task.onnx"),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("end2end", "stride", "imgsz", "message"),
    (
        (False, [8, 16, 32], 641, "divisible"),
        (False, None, 640, "strides"),
        (False, [], 640, "must not be empty"),
        (False, [8.5, 32], 640, "positive integers"),
        (False, [0, 32], 640, "positive integers"),
        (True, [8, 16, 32], 640, "end-to-end"),
    ),
)
def test_export_rejects_receipt_shape_ambiguity_before_backend_export(
    tmp_path, monkeypatch, end2end, stride, imgsz, message
):
    backends = importlib.import_module("manwe.export.backends")
    monkeypatch.setattr(backends, "harden_ultralytics_runtime", lambda: None)
    monkeypatch.setattr(backends, "verify_ultralytics_policy", lambda: None)
    checkpoint = tmp_path / "ambiguous.pt"
    checkpoint.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    output = tmp_path / "ambiguous.onnx"
    export_calls = 0

    class FakeModel:
        task = "detect"
        names = {0: "drone"}

        def __init__(self, _model_path):
            self.model = SimpleNamespace(end2end=end2end, stride=stride)

        def export(self, **_kwargs):
            nonlocal export_calls
            export_calls += 1
            raise AssertionError("backend export must not run after a failed preflight")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    with pytest.raises(ValueError, match=message):
        export_model(
            str(checkpoint),
            ["onnx"],
            output=str(output),
            weights_sha256=digest,
            allow_pickle_checkpoint=True,
            imgsz=imgsz,
            device="cpu",
        )

    assert export_calls == 0
    assert not output.exists()


def test_checkpoint_stride_metadata_has_bounded_nonrecursive_traversal():
    from manwe.export.backends import _checkpoint_max_stride

    assert _checkpoint_max_stride((8, [16, 32.0])) == 32
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="bounded structure"):
        _checkpoint_max_stride(cycle)

    class BrokenStride:
        def tolist(self):
            raise OSError("unloadable device metadata")

    with pytest.raises(ValueError, match="could not be inspected"):
        _checkpoint_max_stride(BrokenStride())


def test_rfdetr_registry_maps_every_constructor_without_pretrained_weights(monkeypatch):
    models = importlib.import_module("manwe.vision.models")
    expected = {
        "rfdetr-nano": "RFDETRNano",
        "rfdetr-small": "RFDETRSmall",
        "rfdetr-medium": "RFDETRMedium",
        "rfdetr-large": "RFDETRLarge",
    }
    registered = {name for name, spec in models.MODEL_ZOO.items() if spec.family == "rfdetr"}
    assert registered == set(expected)

    constructed = []

    def constructor(class_name):
        def build(**kwargs):
            constructed.append((class_name, kwargs))
            return SimpleNamespace(class_name=class_name)

        return build

    fake_rfdetr = SimpleNamespace(
        **{class_name: constructor(class_name) for class_name in expected.values()}
    )
    required = []

    def require(module, extra):
        required.append((module, extra))
        return fake_rfdetr

    monkeypatch.setattr(models, "require", require)
    monkeypatch.setattr(models, "_verify_rfdetr_policy", lambda: None)

    actual = {name: models.build_model(name).class_name for name in expected}

    assert actual == expected
    assert required == [("rfdetr", "rfdetr")] * len(expected)
    assert constructed == [
        (class_name, {"pretrain_weights": None}) for class_name in expected.values()
    ]


def test_rfdetr_policy_requires_pinned_runtime_and_blocks_both_download_bindings(monkeypatch):
    models = importlib.import_module("manwe.vision.models")
    assets = SimpleNamespace(download_pretrain_weights=lambda *_args: None)
    detr = SimpleNamespace(download_pretrain_weights=lambda *_args: None)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.model_weights", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.detr", detr)
    monkeypatch.setattr(models.metadata, "version", lambda name: "1.8.3")

    models._verify_rfdetr_policy()

    assert assets.download_pretrain_weights is models._blocked_rfdetr_download
    assert detr.download_pretrain_weights is models._blocked_rfdetr_download
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["WANDB_MODE"] == "disabled"


def test_rfdetr_policy_rejects_an_unvetted_runtime(monkeypatch):
    models = importlib.import_module("manwe.vision.models")
    monkeypatch.setattr(models.metadata, "version", lambda name: "9.9.9")

    with pytest.raises(RuntimeError, match="not the vetted 1.8.3"):
        models._verify_rfdetr_policy()


def test_training_config_rejects_nonfinite_and_ambiguous_values(tmp_path, monkeypatch):
    data = tmp_path / "data.yaml"
    data.write_text("names: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lr0"):
        VisionTrainConfig(data=str(data), lr0=float("nan"))
    with pytest.raises(ValueError, match="mosaic"):
        VisionTrainConfig(data=str(data), mosaic=-0.1)
    with pytest.raises(TypeError, match="pretrained"):
        VisionTrainConfig(data=str(data), pretrained="true")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="close_mosaic"):
        VisionTrainConfig(data=str(data), close_mosaic=-1)
    with pytest.raises(ValueError, match="imgsz"):
        VisionTrainConfig(data=str(data), imgsz=16)
    with pytest.raises(ValueError, match="epochs"):
        VisionTrainConfig(data=str(data), epochs=100_001)
    with pytest.raises(ValueError, match="single output-directory component"):
        VisionTrainConfig(data=str(data), name="../escape")
    with pytest.raises(ValueError, match="at most 128"):
        VisionTrainConfig(data=str(data), extra={str(index): True for index in range(129)})
    with pytest.raises(ValueError, match="weight_decay"):
        VisionTrainConfig(data=str(data), extra={"weight_decay": float("nan")})
    with pytest.raises(ValueError, match="save_period"):
        VisionTrainConfig(data=str(data), extra={"save_period": 10**20})
    with pytest.raises(ValueError, match="num_workers"):
        VisionTrainConfig(
            data=str(data), model="rfdetr-medium", extra={"num_workers": 1_000_000_000}
        )
    with pytest.raises(ValueError, match="RF-DETR does not consume"):
        VisionTrainConfig(data=str(tmp_path), model="rfdetr-medium", mosaic=0.5)

    mutable_extra = {"amp": True}
    immutable_config = VisionTrainConfig(data=str(data), extra=mutable_extra)
    mutable_extra["amp"] = False
    assert immutable_config.extra["amp"] is True
    with pytest.raises(AttributeError):
        immutable_config.epochs = 1  # type: ignore[misc]

    config = VisionTrainConfig(data=str(data), model="rfdetr-medium")
    called = {"seed": False, "model": False}
    train_module = importlib.import_module("manwe.vision.train")
    monkeypatch.setattr(
        train_module, "seed_everything", lambda _seed: called.__setitem__("seed", True)
    )
    monkeypatch.setattr(
        train_module,
        "build_model",
        lambda *_args, **_kwargs: called.__setitem__("model", True),
    )
    with pytest.raises(ValueError, match="COCO dataset directory"):
        train(config)
    assert called == {"seed": False, "model": False}


@pytest.mark.parametrize("value", [True, 0, 4097, 10**100])
def test_rfdetr_grad_accum_steps_rejects_invalid_values(tmp_path, value):
    with pytest.raises(
        ValueError,
        match=r"extra\.grad_accum_steps must be an integer in \[1, 4096\]",
    ):
        VisionTrainConfig(
            data=str(tmp_path),
            model="rfdetr-medium",
            extra={"grad_accum_steps": value},
        )


@pytest.mark.parametrize("value", [1, 4096])
def test_rfdetr_grad_accum_steps_accepts_closed_bounds(tmp_path, value):
    config = VisionTrainConfig(
        data=str(tmp_path),
        model="rfdetr-medium",
        extra={"grad_accum_steps": value},
    )
    assert config.extra["grad_accum_steps"] == value


def test_rfdetr_gradient_accumulation_misspelling_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match=(
            r"extra\.gradient_accumulation_steps is a legacy Manwe misspelling.*"
            r"use extra\.grad_accum_steps"
        ),
    ):
        VisionTrainConfig(
            data=str(tmp_path),
            model="rfdetr-medium",
            extra={"gradient_accumulation_steps": 8},
        )


def test_rfdetr_training_forwards_exact_upstream_accumulation_field(tmp_path, monkeypatch):
    dataset = tmp_path / "coco"
    _write_rfdetr_coco_dataset(dataset)
    train_module = importlib.import_module("manwe.vision.train")
    forwarded = []
    build_calls = []
    result = object()

    class FakeModel:
        def train(self, **kwargs):
            private_dataset = Path(kwargs["dataset_dir"])
            assert private_dataset != dataset
            assert (private_dataset / "train" / "_annotations.coco.json").is_file()
            assert private_dataset.stat().st_mode & 0o222 == 0
            forwarded.append(kwargs)
            return result

    def build_model(name, *, pretrained):
        build_calls.append((name, pretrained))
        return FakeModel()

    monkeypatch.setattr(train_module, "_reject_ambiguous_opencv_install", lambda: None)
    monkeypatch.setattr(train_module, "resolve_device", lambda _device: Device("cpu"))
    monkeypatch.setattr(train_module, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(train_module, "build_model", build_model)

    actual = train(
        VisionTrainConfig(
            data=str(dataset),
            model="rfdetr-medium",
            epochs=3,
            imgsz=672,
            batch=2,
            device="cpu",
            lr0=0.0002,
            patience=4,
            project="output",
            name="contract",
            extra={"grad_accum_steps": 8, "lr_drop": 2},
        )
    )

    assert actual is result
    assert build_calls == [("rfdetr-medium", False)]
    private_dataset = Path(forwarded[0]["dataset_dir"])
    assert not private_dataset.exists()
    forwarded[0].pop("dataset_dir")
    assert forwarded == [
        {
            "epochs": 3,
            "batch_size": 2,
            "device": "cpu",
            "lr": 0.0002,
            "output_dir": "output/contract",
            "resolution": 672,
            "early_stopping": True,
            "early_stopping_patience": 4,
            "grad_accum_steps": 8,
            "lr_drop": 2,
        }
    ]


def test_rfdetr_snapshot_rejects_annotation_path_escape(tmp_path):
    dataset = tmp_path / "coco"
    _write_rfdetr_coco_dataset(dataset)
    annotation = dataset / "train" / "_annotations.coco.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "../../outside.jpg"
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "outside.jpg").write_bytes(b"outside")

    with pytest.raises(ValueError, match="stay canonically inside"):
        snapshot_local_training_directory(dataset)


def test_rfdetr_snapshot_rejects_cross_split_hardlinks(tmp_path):
    dataset = tmp_path / "coco"
    _write_rfdetr_coco_dataset(dataset)
    valid_image = dataset / "valid" / "frame.jpg"
    valid_image.unlink()
    os.link(dataset / "train" / "frame.jpg", valid_image)

    with pytest.raises(ValueError, match="hardlink aliases"):
        snapshot_local_training_directory(dataset)


def test_rfdetr_snapshot_validates_hardlinks_on_the_exact_copied_inventory(tmp_path, monkeypatch):
    from manwe.common import artifacts

    dataset = tmp_path / "coco"
    _write_rfdetr_coco_dataset(dataset)
    train_image = dataset / "train" / "frame.jpg"
    valid_image = dataset / "valid" / "frame.jpg"
    # Keep the path/byte digest unchanged across the substitution so only the
    # copier's exact inode-topology validator can detect this split alias.
    valid_image.write_bytes(train_image.read_bytes())
    original_snapshot = artifacts.ArtifactSnapshot.from_directory_fd.__func__
    replaced = False

    def alias_immediately_before_copy(cls, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            valid_image.unlink()
            try:
                os.link(train_image, valid_image)
            except OSError as exc:  # pragma: no cover - filesystem capability boundary
                pytest.skip(f"hard links are unavailable: {exc}")
            replaced = True
        return original_snapshot(cls, *args, **kwargs)

    monkeypatch.setattr(
        artifacts.ArtifactSnapshot,
        "from_directory_fd",
        classmethod(alias_immediately_before_copy),
    )
    with pytest.raises(ValueError, match="RF-DETR split files must not be hardlink aliases"):
        snapshot_local_training_directory(dataset)
    assert replaced


def test_rfdetr_snapshot_binds_real_image_dimensions_and_box_extent(tmp_path):
    dataset = tmp_path / "coco"
    _write_rfdetr_coco_dataset(dataset)
    annotation = dataset / "train" / "_annotations.coco.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["images"][0]["width"] = 15
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dimensions do not match annotation"):
        snapshot_local_training_directory(dataset)

    _write_rfdetr_coco_dataset(dataset := tmp_path / "coco-box")
    annotation = dataset / "train" / "_annotations.coco.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["annotations"][0]["bbox"] = [14, 1, 4, 3]
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bbox is outside supported bounds"):
        snapshot_local_training_directory(dataset)


def test_rfdetr_snapshot_normalizes_huge_json_numbers(tmp_path):
    dataset = tmp_path / "coco-huge-number"
    _write_rfdetr_coco_dataset(dataset)
    annotation = dataset / "train" / "_annotations.coco.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["annotations"][0]["bbox"][0] = 10**400
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bbox must be a finite number"):
        snapshot_local_training_directory(dataset)


def test_ultralytics_training_snapshot_rejects_unbounded_or_unreviewed_images(
    tmp_path, monkeypatch
):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "train" / "frame.png", (1, 2, 3), size=(20, 20))
    _write_calibration_png(root / "val" / "frame.png", (4, 5, 6), size=(20, 20))
    manifest = root / "dataset.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(dataset_manifest, "_MAX_TRAINING_IMAGE_PIXELS", 399)
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="dimensions exceed"),
    ):
        admitted.snapshot_for_training()

    monkeypatch.setattr(dataset_manifest, "_MAX_TRAINING_IMAGE_PIXELS", 400)
    (root / "train" / "optional.heic").write_bytes(b"backend-visible but unreviewed")
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="backend-recognized format outside"),
    ):
        admitted.snapshot_for_training()


def test_ultralytics_training_snapshot_validates_the_backend_label_contract(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    _write_calibration_png(train / "frame.png", (1, 2, 3), size=(20, 20))
    _write_calibration_png(val / "frame.png", (4, 5, 6), size=(20, 20))
    label = train / "frame.txt"
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    manifest = root / "dataset.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )

    with (
        validate_local_detection_manifest(manifest) as admitted,
        admitted.snapshot_for_training() as snapshot,
    ):
        assert (snapshot.root / "train" / "frame.txt").read_text(encoding="utf-8") == (
            "0 0.5 0.5 0.25 0.25\n"
        )

    # Six-decimal YOLO serialization can put an exactly edge-touching box a
    # fraction of one unit in the last decimal outside normalized space.
    label.write_text("0 0.166667 0.5 0.333335 0.25\n", encoding="utf-8")
    with (
        validate_local_detection_manifest(manifest) as admitted,
        admitted.snapshot_for_training() as snapshot,
    ):
        assert (snapshot.root / "train" / "frame.txt").is_file()

    label.write_text("0 nan 0.5 0.25 0.25\n", encoding="utf-8")
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="numbers must be finite"),
    ):
        admitted.snapshot_for_training()

    label.write_text("0 0.05 0.5 0.2 0.25\n", encoding="utf-8")
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="remain inside the normalized image bounds"),
    ):
        admitted.snapshot_for_training()

    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MAX_TRAINING_LABEL_BYTES", 8)
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="label exceeds the 8-byte safety limit"),
    ):
        admitted.snapshot_for_training()


def test_ultralytics_training_snapshot_rejects_cross_split_label_hardlinks(tmp_path):
    root = tmp_path / "dataset"
    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    train_labels = root / "labels" / "train"
    val_labels = root / "labels" / "val"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    val_labels.mkdir(parents=True)
    _write_calibration_png(train_images / "train.png", (1, 2, 3), size=(20, 20))
    _write_calibration_png(val_images / "val.png", (4, 5, 6), size=(20, 20))
    train_label = train_labels / "train.txt"
    val_label = val_labels / "val.txt"
    label_bytes = b"0 0.5 0.5 0.25 0.25\n"
    train_label.write_bytes(label_bytes)
    try:
        os.link(train_label, val_label)
    except OSError as exc:  # pragma: no cover - filesystem capability boundary
        pytest.skip(f"hard links are unavailable: {exc}")
    manifest = root / "dataset.yaml"
    manifest.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [drone]\n",
        encoding="utf-8",
    )

    # Admission inventories the declared image trees; the exact consumption
    # inventory additionally binds the sibling labels derived by Ultralytics.
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="train and val paths identify the same"),
    ):
        admitted.snapshot_for_training()

    val_label.unlink()
    val_label.write_bytes(label_bytes)
    with (
        validate_local_detection_manifest(manifest) as admitted,
        admitted.snapshot_for_training() as snapshot,
    ):
        assert (snapshot.root / "labels" / "train" / "train.txt").is_file()
        assert (snapshot.root / "labels" / "val" / "val.txt").is_file()


def test_ultralytics_training_snapshot_rejects_ambiguous_labels_and_truncated_jpeg(tmp_path):
    root = tmp_path / "dataset"
    train = root / "images" / "train"
    val = root / "images" / "val"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    _write_calibration_png(train / "frame.png", (1, 2, 3), size=(20, 20))
    Image.new("RGB", (20, 20), (3, 2, 1)).save(train / "frame.jpg", format="JPEG")
    _write_calibration_png(val / "frame.png", (4, 5, 6), size=(20, 20))
    labels = root / "labels" / "train"
    labels.mkdir(parents=True)
    (labels / "frame.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    manifest = root / "dataset.yaml"
    manifest.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [drone]\n",
        encoding="utf-8",
    )

    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="same backend label path"),
    ):
        admitted.snapshot_for_training()

    (train / "frame.png").unlink()
    jpeg = train / "frame.jpg"
    jpeg.write_bytes(jpeg.read_bytes()[:-2])
    with (
        validate_local_detection_manifest(manifest) as admitted,
        pytest.raises(ValueError, match="truncated|EOI|valid bounded still image"),
    ):
        admitted.snapshot_for_training()


def test_ultralytics_training_consumes_a_private_immutable_dataset(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    root = tmp_path / "dataset"
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir()
    source_train = train_dir / "frame.png"
    _write_calibration_png(source_train, (1, 2, 3), size=(16, 12))
    source_bytes = source_train.read_bytes()
    _write_calibration_png(val_dir / "frame.png", (4, 5, 6), size=(16, 12))
    manifest = root / "dataset.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )
    train_module = importlib.import_module("manwe.vision.train")
    observed = {}
    result = object()

    class FakeModel:
        def train(self, **kwargs):
            private_manifest = Path(kwargs["data"])
            payload = yaml.safe_load(private_manifest.read_text(encoding="utf-8"))
            private_root = Path(payload["path"])
            private_train = Path(payload["train"])
            assert private_root != root
            assert private_train == private_root / "train"
            assert (private_train / "frame.png").read_bytes() == source_bytes
            assert private_root.stat().st_mode & 0o222 == 0
            source_train.write_bytes(b"mutated-after-snapshot")
            assert (private_train / "frame.png").read_bytes() == source_bytes
            observed.update(manifest=private_manifest, root=private_root)
            return result

    monkeypatch.setattr(train_module, "resolve_device", lambda _device: Device("cpu"))
    monkeypatch.setattr(train_module, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: FakeModel())

    assert train(VisionTrainConfig(data=str(manifest), epochs=1, batch=1, device="cpu")) is result
    assert not observed["manifest"].exists()
    assert not observed["root"].exists()


def test_ultralytics_training_detects_private_dataset_mutation(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    _write_calibration_png(root / "train" / "frame.png", (1, 2, 3), size=(16, 12))
    _write_calibration_png(root / "val" / "frame.png", (4, 5, 6), size=(16, 12))
    manifest = root / "dataset.yaml"
    manifest.write_text(
        "path: .\ntrain: train\nval: val\nnames: [drone]\n",
        encoding="utf-8",
    )
    train_module = importlib.import_module("manwe.vision.train")

    class MutatingModel:
        def train(self, **kwargs):
            import yaml

            payload = yaml.safe_load(Path(kwargs["data"]).read_text(encoding="utf-8"))
            target = Path(payload["train"]) / "frame.png"
            target.chmod(0o600)
            target.write_bytes(b"backend-mutated")
            return object()

    monkeypatch.setattr(train_module, "resolve_device", lambda _device: Device("cpu"))
    monkeypatch.setattr(train_module, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(train_module, "build_model", lambda *_args, **_kwargs: MutatingModel())

    with pytest.raises(RuntimeError, match="private training dataset changed"):
        train(VisionTrainConfig(data=str(manifest), epochs=1, batch=1, device="cpu"))


def test_rfdetr_training_rejects_multiple_opencv_owners_before_side_effects(tmp_path, monkeypatch):
    dataset = tmp_path / "coco"
    dataset.mkdir()
    train_module = importlib.import_module("manwe.vision.train")
    versions = {"opencv-python": "5.0.0", "opencv-contrib-python-headless": "4.10.0"}

    def installed_version(name):
        try:
            return versions[name]
        except KeyError as exc:
            raise train_module.metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(train_module.metadata, "version", installed_version)
    monkeypatch.setattr(
        train_module,
        "seed_everything",
        lambda _seed: pytest.fail("seed mutation must not happen before dependency preflight"),
    )

    with pytest.raises(RuntimeError, match="exactly one OpenCV distribution"):
        train(VisionTrainConfig(data=str(dataset), model="rfdetr-medium"))


def test_rfdetr_training_rejects_missing_opencv_owner_before_side_effects(tmp_path, monkeypatch):
    dataset = tmp_path / "coco"
    dataset.mkdir()
    train_module = importlib.import_module("manwe.vision.train")

    def missing_version(name):
        raise train_module.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(train_module.metadata, "version", missing_version)
    monkeypatch.setattr(
        train_module,
        "seed_everything",
        lambda _seed: pytest.fail("seed mutation must not happen before dependency preflight"),
    )
    with pytest.raises(RuntimeError, match="found none"):
        train(VisionTrainConfig(data=str(dataset), model="rfdetr-medium"))


def test_training_resolves_device_before_rng_mutation(tmp_path, monkeypatch):
    dataset = tmp_path / "coco"
    _write_rfdetr_coco_dataset(dataset)
    train_module = importlib.import_module("manwe.vision.train")
    monkeypatch.setattr(train_module, "_reject_ambiguous_opencv_install", lambda: None)
    monkeypatch.setattr(
        train_module,
        "resolve_device",
        lambda _device: (_ for _ in ()).throw(ValueError("invalid device")),
    )
    monkeypatch.setattr(
        train_module,
        "seed_everything",
        lambda _seed: pytest.fail("seed must not mutate before device preflight"),
    )
    with pytest.raises(ValueError, match="invalid device"):
        train(VisionTrainConfig(data=str(dataset), model="rfdetr-medium"))


def test_cli_resolves_dataset_relative_to_training_config(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    del yaml  # import availability is the contract; the CLI imports it itself.
    dataset = tmp_path / "data.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text("data: data.yaml\n", encoding="utf-8")
    observed = {}
    train_module = importlib.import_module("manwe.vision.train")
    monkeypatch.setattr(
        train_module, "train", lambda value: observed.setdefault("data", value.data)
    )
    assert _cmd_vision_train(argparse.Namespace(config=str(config))) == 0
    assert observed["data"] == str(dataset.resolve())


def test_cli_training_config_rejects_duplicate_keys_and_symlinked_components(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("data: first.yaml\ndata: second.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        _cmd_vision_train(argparse.Namespace(config=str(duplicate)))

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "train.yaml").write_text("data: data.yaml\n", encoding="utf-8")
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="path chain contains a symbolic link"):
        _cmd_vision_train(argparse.Namespace(config=str(linked_dir / "train.yaml")))

    deeply_nested = tmp_path / "deep.yaml"
    deeply_nested.write_text("value: " + "[" * 1000 + "0" + "]" * 1000, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting safety limit"):
        _cmd_vision_train(argparse.Namespace(config=str(deeply_nested)))


def test_postprocess_is_shape_safe_class_aware_and_rejects_clipped_degeneracy():
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        nms(np.zeros((1, 2, 4)), np.array([0.9]))
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    assert nms(boxes, np.array([0.9, 0.8]), labels=np.array([0, 1])) == [0, 1]
    with pytest.raises(ValueError, match="zero-area"):
        scale_boxes(np.array([[20.0, 20.0, 30.0, 30.0]]), 1.0, (0.0, 0.0), (10, 10))


def test_generic_average_precision_accepts_finite_non_probability_rankings():
    box = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert average_precision(box, np.array([12.0]), box) == 1.0
