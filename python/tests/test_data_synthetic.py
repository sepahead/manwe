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
    original_write = synthetic._write_exclusive
    calls = 0

    def fail_second_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original_write(path, payload)

    monkeypatch.setattr(synthetic, "_write_exclusive", fail_second_write)
    with pytest.raises(OSError, match="injected write failure"):
        make_vision_smoke(empty_root, n_train=1, n_val=1)
    assert list(empty_root.iterdir()) == []


def test_synthetic_manifest_quotes_unusual_paths_and_rejects_controls(tmp_path):
    from manwe.common.dataset_manifest import validate_local_detection_manifest

    unusual = tmp_path / "dataset # one: two"
    manifest = make_vision_smoke(unusual, n_train=1, n_val=1)
    with validate_local_detection_manifest(manifest) as snapshot:
        assert snapshot.path.is_file()

    with pytest.raises(ValueError, match="control character"):
        make_vision_smoke(tmp_path / "bad\npath", n_train=1, n_val=1)
