"""Synthetic dataset generator produces a valid YOLO dataset offline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from manwe.common.config_io import read_strict_yaml
from manwe.data import make_vision_smoke, write_png


def test_make_vision_smoke_layout_and_labels():
    with tempfile.TemporaryDirectory() as d:
        yaml_path = make_vision_smoke(d, n_train=5, n_val=2, seed=1)
        root = Path(d)
        assert yaml_path.exists()
        train_imgs = list((root / "images" / "train").glob("*.png"))
        val_imgs = list((root / "images" / "val").glob("*.png"))
        assert len(train_imgs) == 5 and len(val_imgs) == 2
        # PNG magic bytes
        assert train_imgs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        # every image has a matching label with valid YOLO rows
        for img in train_imgs:
            lbl = root / "labels" / "train" / (img.stem + ".txt")
            assert lbl.exists()
            for line in lbl.read_text().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5
                cls = int(parts[0])
                assert 0 <= cls <= 4
                assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
        manifest = read_strict_yaml(yaml_path, 1 << 20, "synthetic dataset manifest")
        assert isinstance(manifest, dict)
        assert manifest["names"] == ["drone", "bird", "aircraft", "helicopter", "unknown"]


def test_synthetic_yolo_label_matches_the_exact_rasterized_box(tmp_path):
    from PIL import Image

    root = tmp_path / "aligned"
    make_vision_smoke(root, n_train=1, n_val=1, size=127, max_objs=1, seed=4)
    label = (root / "labels" / "train" / "train_000.txt").read_text().split()
    class_index = int(label[0])
    cx, cy, width, height = (float(value) for value in label[1:])
    colors = np.array(
        [
            (230, 60, 60),
            (60, 200, 90),
            (70, 120, 240),
            (240, 200, 60),
            (170, 170, 170),
        ],
        dtype=np.uint8,
    )
    with Image.open(root / "images" / "train" / "train_000.png") as image:
        pixels = np.asarray(image).copy()
    ys, xs = np.nonzero(np.all(pixels == colors[class_index], axis=2))
    assert len(xs) > 0
    actual_width = int(xs.max() - xs.min() + 1)
    actual_height = int(ys.max() - ys.min() + 1)
    assert cx == pytest.approx((xs.min() + xs.max() + 1) / (2 * pixels.shape[1]), abs=5e-7)
    assert cy == pytest.approx((ys.min() + ys.max() + 1) / (2 * pixels.shape[0]), abs=5e-7)
    assert width == pytest.approx(actual_width / pixels.shape[1], abs=5e-7)
    assert height == pytest.approx(actual_height / pixels.shape[0], abs=5e-7)


def test_synthetic_generation_rejects_shared_writable_existing_root(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    root.chmod(0o777)
    try:
        with pytest.raises(PermissionError, match="group- or other-writable"):
            make_vision_smoke(root, n_train=1, n_val=1)
        assert list(root.iterdir()) == []
    finally:
        root.chmod(0o700)


def test_synthetic_generation_rejects_invalid_or_existing_outputs(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError, match="absent or empty"):
        make_vision_smoke(occupied)
    assert (occupied / "keep.txt").read_text() == "user data"

    with pytest.raises(ValueError, match="at least 16"):
        make_vision_smoke(tmp_path / "small", size=8)
    with pytest.raises(ValueError, match="uint8"):
        write_png(tmp_path / "float.png", np.zeros((2, 2, 3), dtype=float))
    with pytest.raises(ValueError, match="image safety limit"):
        make_vision_smoke(tmp_path / "oversized", size=6000)
    with pytest.raises(ValueError, match="image safety limit"):
        make_vision_smoke(tmp_path / "too-many", n_train=100_000, n_val=1)
    with pytest.raises(ValueError, match="object safety limit"):
        make_vision_smoke(tmp_path / "too-many-objects", max_objs=1001)
    with pytest.raises(ValueError, match="annotation safety limit"):
        make_vision_smoke(tmp_path / "too-many-total-objects", n_train=5000, max_objs=1000)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    destination = tmp_path / "image.png"
    write_png(destination, image)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_png(destination, image)


def test_synthetic_generation_canonicalizes_symlinked_parents_and_rolls_back(tmp_path, monkeypatch):
    from manwe.data import synthetic

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    manifest = make_vision_smoke(linked_parent / "dataset", n_train=1, n_val=1)
    assert manifest.parent == real_parent / "dataset"

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    original_write = synthetic._write_exclusive_at
    calls = 0

    def fail_second_write(directory_fd, name, payload, display):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original_write(directory_fd, name, payload, display)

    monkeypatch.setattr(synthetic, "_write_exclusive_at", fail_second_write)
    with pytest.raises(OSError, match="injected write failure"):
        make_vision_smoke(empty_root, n_train=1, n_val=1)
    assert list(empty_root.iterdir()) == []


def test_synthetic_outputs_do_not_follow_a_final_symlink(tmp_path):
    png_target = tmp_path / "png-target"
    png_link = tmp_path / "frame.png"
    png_link.symlink_to(png_target)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_png(png_link, np.zeros((2, 2, 3), dtype=np.uint8))
    assert not png_target.exists()

    dataset_target = tmp_path / "dataset-target"
    dataset_target.mkdir()
    dataset_link = tmp_path / "dataset-link"
    dataset_link.symlink_to(dataset_target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="actual empty directory"):
        make_vision_smoke(dataset_link, n_train=1, n_val=1)
    assert list(dataset_target.iterdir()) == []


def test_synthetic_generation_does_not_follow_replaced_output_root(tmp_path, monkeypatch):
    from manwe.data import synthetic

    root = tmp_path / "dataset"
    moved = tmp_path / "moved-dataset"
    original_write = synthetic._write_exclusive_at
    replaced = False

    def replace_root_after_first_write(directory_fd, name, payload, display):
        nonlocal replaced
        identity = original_write(directory_fd, name, payload, display)
        if not replaced:
            root.rename(moved)
            root.mkdir()
            (root / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return identity

    monkeypatch.setattr(
        synthetic,
        "_write_exclusive_at",
        replace_root_after_first_write,
    )
    with pytest.raises(RuntimeError, match="output directory was replaced"):
        make_vision_smoke(root, n_train=1, n_val=1)
    assert (root / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert list(moved.iterdir()) == []


def test_synthetic_root_creation_does_not_follow_replaced_ancestor(tmp_path, monkeypatch):
    from manwe.data import synthetic

    parent = tmp_path / "output-parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-output-parent"
    root = parent / "nested" / "dataset"
    real_mkdir = synthetic.os.mkdir
    replaced = False

    def replace_ancestor_after_first_mkdir(path, *args, **kwargs):
        nonlocal replaced
        result = real_mkdir(path, *args, **kwargs)
        if not replaced and path == "nested":
            parent.rename(moved_parent)
            real_mkdir(parent, 0o755)
            (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return result

    monkeypatch.setattr(synthetic.os, "mkdir", replace_ancestor_after_first_mkdir)
    with pytest.raises(RuntimeError, match="output directory was replaced"):
        make_vision_smoke(root, n_train=1, n_val=1)
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert list((moved_parent / "nested" / "dataset").iterdir()) == []


def test_write_png_rolls_back_only_its_bound_parent_after_replacement(tmp_path, monkeypatch):
    from manwe.data import synthetic

    parent = tmp_path / "images"
    parent.mkdir()
    moved = tmp_path / "moved-images"
    destination = parent / "frame.png"
    original_write = synthetic._write_exclusive_at

    def replace_parent_after_write(directory_fd, name, payload, display):
        identity = original_write(directory_fd, name, payload, display)
        parent.rename(moved)
        parent.mkdir()
        (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
        return identity

    monkeypatch.setattr(
        synthetic,
        "_write_exclusive_at",
        replace_parent_after_write,
    )
    with pytest.raises(RuntimeError, match="output parent was replaced"):
        write_png(destination, np.zeros((2, 2, 3), dtype=np.uint8))
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not (moved / destination.name).exists()


def test_write_png_closes_and_removes_partial_file_when_fdopen_fails(tmp_path, monkeypatch):
    import os

    from manwe.common import fd_io

    destination = tmp_path / "frame.png"
    opened_fds: list[int] = []

    def fail_fdopen(fd, _mode):
        opened_fds.append(fd)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_fdopen)
    with pytest.raises(OSError, match="injected fdopen failure"):
        write_png(destination, np.zeros((2, 2, 3), dtype=np.uint8))
    assert not destination.exists()
    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])


def test_write_png_success_is_not_reclassified_by_parent_close_error(tmp_path, monkeypatch):
    from manwe.data import synthetic

    destination = tmp_path / "frame.png"
    real_write = synthetic._write_exclusive_at
    real_close = synthetic.os.close
    close_enabled = False
    close_calls: list[int] = []

    def enable_close_failure(*args, **kwargs):
        nonlocal close_enabled
        result = real_write(*args, **kwargs)
        close_enabled = True
        return result

    def close_after_release(fd):
        close_calls.append(fd)
        real_close(fd)
        if close_enabled:
            raise OSError("injected post-commit close failure")

    monkeypatch.setattr(synthetic, "_write_exclusive_at", enable_close_failure)
    monkeypatch.setattr(synthetic.os, "close", close_after_release)
    write_png(destination, np.zeros((2, 2, 3), dtype=np.uint8))
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert close_calls


def test_synthetic_manifest_quotes_unusual_paths_and_rejects_controls(tmp_path):
    from manwe.common.dataset_manifest import validate_local_detection_manifest

    unusual = tmp_path / "dataset # one: two"
    manifest = make_vision_smoke(unusual, n_train=1, n_val=1)
    with validate_local_detection_manifest(manifest) as snapshot:
        assert snapshot.path.is_file()

    with pytest.raises(ValueError, match="control character"):
        make_vision_smoke(tmp_path / "bad\npath", n_train=1, n_val=1)
