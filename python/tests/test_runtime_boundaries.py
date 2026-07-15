"""Adversarial boundary tests for runtime, training, and raw export preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import os
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from manwe.cli import _cmd_vision_train
from manwe.common.artifacts import ArtifactSnapshot
from manwe.common.dataset_manifest import (
    snapshot_local_calibration_dataset,
    validate_local_detection_manifest,
)
from manwe.common.device import Device, resolve_device
from manwe.common.logging import get_logger
from manwe.eval.detection import average_precision
from manwe.export.backends import _publish_exclusive, export_model
from manwe.vision.input import prepare_single_image
from manwe.vision.postprocess import nms, scale_boxes
from manwe.vision.predict import Detector
from manwe.vision.sahi_infer import SliceConfig
from manwe.vision.train import VisionTrainConfig, train


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


def test_directory_snapshot_enforces_byte_limit_during_copy(tmp_path, monkeypatch):
    from manwe.common import artifacts

    bundle = tmp_path / "bundle.mlpackage"
    bundle.mkdir()
    member = bundle / "weights.bin"
    member.write_bytes(b"a")
    expected = artifacts.sha256_artifact(bundle)
    real_entries = artifacts._tree_entries
    mutated = False

    def mutate_after_enumeration(path, max_entries):
        nonlocal mutated
        entries = real_entries(path, max_entries)
        if path == bundle and not mutated:
            member.write_bytes(b"x" * 100)
            mutated = True
        return entries

    monkeypatch.setattr(artifacts, "_tree_entries", mutate_after_enumeration)
    with pytest.raises(ValueError, match="byte safety limit"):
        ArtifactSnapshot(bundle, expected, max_bytes=8)


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
    with pytest.raises(ValueError, match="regular file or directory"):
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


def test_calibration_digest_binds_dataset_content_and_enforces_coverage(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    (train / "one.png").write_bytes(b"one")
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)
    snapshot = validate_local_detection_manifest(manifest)
    try:
        with pytest.raises(ValueError, match="at least 2"):
            snapshot.calibration_digest()
        second = val / "two.png"
        second.write_bytes(b"two")
        before = snapshot.calibration_digest()
        second.write_bytes(b"changed")
        assert snapshot.calibration_digest() != before
    finally:
        snapshot.close()


def test_calibration_snapshot_is_private_read_only_and_source_bound(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    first = train / "one.png"
    second = val / "two.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)

    snapshot = snapshot_local_calibration_dataset(manifest)
    private_root = snapshot.root
    private_manifest = snapshot.path
    try:
        assert private_root != root
        assert str(private_root) in private_manifest.read_text(encoding="utf-8")
        assert (private_root.stat().st_mode & 0o222) == 0
        assert ((private_root / "train" / "one.png").stat().st_mode & 0o222) == 0
        assert (private_root / "train" / "one.png").read_bytes() == b"one"
        bound_digest = snapshot.calibration_digest()

        first.write_bytes(b"mutated")
        assert (private_root / "train" / "one.png").read_bytes() == b"one"
        assert snapshot.calibration_digest() == bound_digest == snapshot.sha256
        with pytest.raises(RuntimeError, match="source calibration dataset changed"):
            snapshot.assert_source_unchanged()
    finally:
        snapshot.close()
    assert not private_root.exists()
    assert not private_manifest.exists()


def test_calibration_snapshot_rejects_source_root_replacement(tmp_path, monkeypatch):
    from manwe.common import dataset_manifest

    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "train" / "one.png").write_bytes(b"one")
    (root / "val" / "two.png").write_bytes(b"two")
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
    (root / "train" / "one.png").write_bytes(b"one")
    (root / "val" / "two.png").write_bytes(b"two")
    manifest = root / "data.yaml"
    manifest.write_text("path: .\ntrain: train\nval: val\nnames: [drone]\n", encoding="utf-8")
    monkeypatch.setattr(dataset_manifest, "_MIN_CALIBRATION_IMAGES", 2)
    with snapshot_local_calibration_dataset(manifest) as expected_snapshot:
        expected_digest = expected_snapshot.sha256

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    weights_digest = hashlib.sha256(b"fixture").hexdigest()
    output = tmp_path / "model.engine"
    observed = {}
    backends = importlib.import_module("manwe.export.backends")
    monkeypatch.setattr(backends, "harden_ultralytics_runtime", lambda: None)
    monkeypatch.setattr(backends, "verify_ultralytics_policy", lambda: None)
    monkeypatch.setattr(device_module, "resolve_device", lambda _device: Device("cuda", index=0))

    class FakeModel:
        task = "detect"
        model = SimpleNamespace(end2end=False, stride=[8, 16, 32])
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
            assert (private_root / "train" / "one.png").read_bytes() == b"one"
            assert (private_root.stat().st_mode & 0o222) == 0

            if source_change == "transient-replacement":
                original = tmp_path / "source-original"
                root.rename(original)
                try:
                    (root / "train").mkdir(parents=True)
                    (root / "val").mkdir()
                    (root / "train" / "one.png").write_bytes(b"attacker")
                    (root / "val" / "two.png").write_bytes(b"attacker")
                    (root / "data.yaml").write_text(
                        "path: .\ntrain: train\nval: val\nnames: [attacker]\n",
                        encoding="utf-8",
                    )
                    assert (private_root / "train" / "one.png").read_bytes() == b"one"
                finally:
                    shutil.rmtree(root)
                    original.rename(root)
            else:
                (root / "train" / "one.png").write_bytes(b"attacker")
                assert (private_root / "train" / "one.png").read_bytes() == b"one"

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


def test_directory_export_publication_removes_partial_destination(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "Manifest.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "published.mlpackage"
    digest = sha256_artifact(source)

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
    with pytest.raises(OSError, match="injected publication failure"):
        _publish_exclusive(source, destination, digest)
    assert not destination.exists()


def test_directory_export_replacement_cannot_redirect_descriptor_relative_writes(
    tmp_path, monkeypatch
):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.mlpackage"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    destination = tmp_path / "published.mlpackage"
    moved = tmp_path / "moved-original.mlpackage"
    digest = sha256_artifact(source)
    real_copy = backends.shutil.copyfileobj
    replaced = False

    def replace_root_during_copy(src, dst, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            destination.rename(moved)
            destination.mkdir()
            (destination / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(backends.shutil, "copyfileobj", replace_root_during_copy)
    with pytest.raises(RuntimeError, match="replaced"):
        backends._publish_exclusive(source, destination, digest)
    assert (destination / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not (destination / "weights.bin").exists()
    assert (moved / "weights.bin").read_bytes() == b"trusted"


def test_export_parent_replacement_cannot_redirect_publication(tmp_path, monkeypatch):
    from manwe.common.artifacts import sha256_artifact
    from manwe.export import backends

    source = tmp_path / "source.onnx"
    source.write_bytes(b"trusted")
    parent = tmp_path / "output"
    parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)
    prepared = backends._prepare_destination(str(destination), {".onnx"})
    real_copy = backends.shutil.copyfileobj
    replaced = False

    def replace_parent_during_copy(src, dst, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            parent.rename(moved_parent)
            parent.mkdir()
            (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(backends.shutil, "copyfileobj", replace_parent_during_copy)
    try:
        with pytest.raises(RuntimeError, match="parent was replaced"):
            backends._publish_exclusive(source, prepared, digest)
    finally:
        prepared.close()
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not (parent / destination.name).exists()
    assert not (moved_parent / destination.name).exists()


def test_ultralytics_policy_disables_install_network_and_analytics(monkeypatch):
    policy = importlib.import_module("manwe.common.ultralytics")
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
    with pytest.raises(RuntimeError, match="network downloads are disabled"):
        downloads.safe_download("https://example.invalid/model.pt")  # type: ignore[attr-defined]

    # The vetted-version gate is a safety control: an unvetted runtime must be
    # rejected outright, never merely hardened.
    monkeypatch.setattr(policy.importlib.metadata, "version", lambda _name: "8.4.0")
    with pytest.raises(RuntimeError, match="not the vetted"):
        policy.verify_ultralytics_policy()


def test_logger_cannot_mutate_the_process_root_logger():
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    with pytest.raises(ValueError, match="logger name"):
        get_logger("root", level="DEBUG")
    assert (list(root.handlers), root.level) == before


def test_device_contract_rejects_bad_types_and_prefers_available_fallback(monkeypatch):
    with pytest.raises(TypeError, match="preference"):
        resolve_device([])  # type: ignore[arg-type]
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

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeModel:
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
    detector.device = Device("cpu")
    detector.conf = 0.25
    detector.iou = 0.45
    assert detector.detect(rgb) == []
    expected_bgr = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(observed_images[0], expected_bgr)
    assert observed_kwargs[0] == {
        "conf": 0.25,
        "iou": 0.45,
        "device": "cpu",
        "verbose": False,
    }
    assert observed_images[0].flags.c_contiguous
    assert observed_images[0].flags.owndata
    assert not np.shares_memory(observed_images[0], rgb)
    np.testing.assert_array_equal(rgb, np.array([[[255, 0, 0], [0, 0, 255]]]))

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
    dataset.mkdir()
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
