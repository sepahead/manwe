"""Fail-closed regressions for export contracts and fidelity metrics."""

from __future__ import annotations

import os

import numpy as np
import pytest

from manwe.common.contracts import CREBAIN_CLASSES, TensorSpec
from manwe.eval.detection import (
    Detections,
    GroundTruth,
    average_precision,
    iou_matrix,
    mean_average_precision,
)
from manwe.export import (
    ExportReceipt,
    VerifiedArtifactSignature,
    build_export_contract,
    fidelity_report,
    save_contract,
    sha256_file,
)


def _build_contract(artifact, **overrides):
    manwe_format = overrides.pop("manwe_format", "onnx")
    num_classes = overrides.pop("num_classes", len(CREBAIN_CLASSES))
    class_map = overrides.pop("class_map", None)
    digest = sha256_file(artifact)
    receipt = ExportReceipt(
        format=manwe_format,
        artifact_path=str(artifact),
        artifact_sha256=digest,
        source_sha256="1" * 64,
        source_suffix=".pt",
        image_size=640,
        precision="float32",
        embedded_nms=False,
        opset=17 if manwe_format in {"onnx", "tensorrt"} else None,
        class_count=num_classes,
        source_classes=(
            tuple(CREBAIN_CLASSES)
            if num_classes == len(CREBAIN_CLASSES)
            else tuple(f"source-{index}" for index in range(num_classes))
        ),
        end_to_end=False,
        calibration_manifest_sha256=None,
    )
    signature = VerifiedArtifactSignature(
        artifact_sha256=digest,
        precision="float32",
        embedded_nms=False,
        opset=receipt.opset,
        source_classes=receipt.source_classes,
        inputs=(TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
        outputs=(TensorSpec("output", [1, 4 + num_classes, 8400], "float32"),),
        preprocess="inspected fixture preprocessing",
        postprocess="inspected fixture postprocessing",
        failure_behavior="fixture-backed shape mismatch raises",
        evidence="tests/fixtures/signature.json sha256:fixture",
    )
    values = {
        "model_name": "manwe-aerial",
        "model_version": "0.2.0",
        "source": "test fixture",
        "rights": "test-only fixture",
        "receipt": receipt,
        "signature": signature,
        "class_map": class_map,
        "validation_data": "tests/fixtures/aerial/*.png",
        "benchmark_context": "test CPU, Linux, conf=0.25 iou=0.45",
    }
    values.update(overrides)
    return build_export_contract(**values)


def _empty_detections(image_id: str = "frame") -> Detections:
    return Detections(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), image_id=image_id)


def test_build_contract_rejects_missing_artifact_and_explicit_empty_class_map(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _build_contract(tmp_path / "missing.onnx")

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    with pytest.raises(ValueError, match="class_map must not be empty"):
        _build_contract(artifact, class_map={})


def test_identity_class_map_requires_exact_ordered_source_taxonomy(tmp_path):
    artifact = tmp_path / "reordered.onnx"
    artifact.write_bytes(b"model")
    digest = sha256_file(artifact)
    reordered = ("bird", "drone", "aircraft", "helicopter", "unknown")
    receipt = ExportReceipt(
        format="onnx",
        artifact_path=str(artifact),
        artifact_sha256=digest,
        source_sha256="1" * 64,
        source_suffix=".pt",
        image_size=640,
        precision="float32",
        embedded_nms=False,
        opset=17,
        class_count=5,
        source_classes=reordered,
        end_to_end=False,
        calibration_manifest_sha256=None,
    )
    baseline = _build_contract(artifact)
    signature = VerifiedArtifactSignature(
        artifact_sha256=digest,
        precision="float32",
        embedded_nms=False,
        opset=17,
        source_classes=reordered,
        inputs=tuple(baseline.inputs),
        outputs=tuple(baseline.outputs),
        preprocess="inspected fixture preprocessing",
        postprocess="inspected fixture postprocessing",
        failure_behavior="fixture-backed shape mismatch raises",
        evidence="tests/fixtures/reordered-signature.json",
    )
    with pytest.raises(ValueError, match="class_map is required"):
        build_export_contract(
            model_name="reordered",
            model_version="1",
            source="fixture",
            rights="fixture",
            receipt=receipt,
            signature=signature,
            validation_data="fixture",
            benchmark_context="fixture",
        )


def test_contract_rejects_bad_schema_backend_and_class_map_coverage(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)

    contract.schema_version = ""
    with pytest.raises(ValueError, match="schema_version"):
        contract.validate()

    contract.schema_version = "1.0"
    contract.backend = "other"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unsupported backend"):
        contract.validate()

    contract.backend = "onnx"
    contract.class_map = {"0": "drone"}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must be an integer"):
        contract.validate_class_map()

    contract.num_classes = 2
    contract.class_map = {0: "drone", 2: "bird"}
    with pytest.raises(ValueError, match="missing class indices.*out-of-range"):
        contract.validate_class_map()

    contract.class_map = {0: "drone", 1: "car"}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must be a crebain class"):
        contract.validate_class_map()


def test_directory_tree_digest_is_deterministic_and_content_sensitive(tmp_path):
    first = tmp_path / "first.mlpackage"
    second = tmp_path / "second.mlpackage"
    first.mkdir()
    second.mkdir()

    # Deliberately create identical trees in different orders.
    (first / "Data").mkdir()
    (first / "Data" / "weights.bin").write_bytes(b"weights")
    (first / "Manifest.json").write_text('{"version": 1}', encoding="utf-8")
    (first / "Empty").mkdir()

    (second / "Empty").mkdir()
    (second / "Manifest.json").write_text('{"version": 1}', encoding="utf-8")
    (second / "Data").mkdir()
    (second / "Data" / "weights.bin").write_bytes(b"weights")

    assert sha256_file(first) == sha256_file(second)

    (second / "Data" / "weights.bin").write_bytes(b"changed")
    assert sha256_file(first) != sha256_file(second)


def test_coreml_directory_contract_and_save_digest_gate(tmp_path):
    package = tmp_path / "detector.mlpackage"
    package.mkdir()
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    contract = _build_contract(package, manwe_format="coreml")
    assert contract.backend == "coreml"
    assert contract.is_complete()

    json_path, markdown_path = save_contract(contract, package)
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert not list(tmp_path.glob(".manwe-contract-*.in-progress"))

    (package / "Manifest.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        save_contract(contract, package)


def test_save_contract_success_is_not_reclassified_by_descriptor_close_errors(
    tmp_path, monkeypatch
):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_cleanup = contract_module._cleanup_private_stage
    real_close = contract_module.os.close
    close_enabled = False
    close_calls: list[int] = []

    def enable_close_failures(*args, **kwargs):
        nonlocal close_enabled
        result = real_cleanup(*args, **kwargs)
        close_enabled = True
        return result

    def close_after_release(fd):
        close_calls.append(fd)
        real_close(fd)
        if close_enabled:
            raise OSError("injected post-commit close failure")

    monkeypatch.setattr(contract_module, "_cleanup_private_stage", enable_close_failures)
    monkeypatch.setattr(contract_module.os, "close", close_after_release)
    json_path, markdown_path = save_contract(contract, artifact)
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert close_calls


def test_save_contract_anchors_relative_paths_to_one_cwd_snapshot(tmp_path, monkeypatch):
    from pathlib import Path

    from manwe.export import contract as contract_module

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "model.onnx").write_bytes(b"model")
    (second / "model.onnx").write_bytes(b"model")
    monkeypatch.chdir(first)
    contract = _build_contract(Path("model.onnx"))
    real_cwd = Path.cwd
    changed = False

    def snapshot_then_change(_cls):
        nonlocal changed
        captured = real_cwd()
        if not changed:
            changed = True
            os.chdir(second)
        return captured

    monkeypatch.setattr(
        contract_module.pathlib.Path,
        "cwd",
        classmethod(snapshot_then_change),
    )
    json_path, markdown_path = save_contract(contract, Path("model.onnx"))
    assert changed
    assert json_path.parent == first
    assert markdown_path.parent == first
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert not (second / json_path.name).exists()
    assert not (second / markdown_path.name).exists()


def test_save_contract_refuses_incomplete_or_wrongly_targeted_contract(tmp_path):
    artifact = tmp_path / "model.onnx"
    other = tmp_path / "other.onnx"
    artifact.write_bytes(b"model")
    other.write_bytes(b"other")
    contract = _build_contract(artifact)

    contract.benchmark_context = ""
    with pytest.raises(ValueError, match="benchmark_context"):
        save_contract(contract, artifact)

    contract.benchmark_context = "test CPU, Linux"
    with pytest.raises(ValueError, match="does not identify"):
        save_contract(contract, other)


def test_contract_rejects_root_symlinks_and_artifact_tampering(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    link = tmp_path / "link.onnx"
    link.symlink_to(artifact)

    with pytest.raises(ValueError, match="symbolic link"):
        sha256_file(link)
    with pytest.raises(ValueError, match="symbolic link"):
        _build_contract(link)

    contract = _build_contract(artifact)
    artifact.write_bytes(b"tampered")
    assert not contract.is_complete()
    with pytest.raises(ValueError, match="SHA-256"):
        contract.validate()


def test_save_contract_never_replaces_preoccupied_paths(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    json_path = artifact.with_suffix(".contract.json")
    markdown_path = artifact.with_suffix(".contract.md")

    json_path.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_contract(contract, artifact)
    assert json_path.read_text(encoding="utf-8") == "owned"
    assert not markdown_path.exists()
    assert not list(tmp_path.glob(".manwe-contract-*.in-progress"))

    json_path.unlink()
    dangling_target = tmp_path / "missing-target"
    markdown_path.symlink_to(dangling_target)
    with pytest.raises(FileExistsError):
        save_contract(contract, artifact)
    assert not json_path.exists()
    assert markdown_path.is_symlink()
    assert not dangling_target.exists()
    assert not list(tmp_path.glob(".manwe-contract-*.in-progress"))


def test_save_contract_detects_replacement_and_preserves_foreign_path(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_link = contract_module._link_staged_sidecar

    def replace_json_after_link(stage_fd, stage_name, parent_fd, final_name):
        real_link(stage_fd, stage_name, parent_fd, final_name)
        if stage_name == contract_module._STAGE_JSON:
            final_path = tmp_path / final_name
            final_path.unlink()
            final_path.write_text("foreign", encoding="utf-8")

    monkeypatch.setattr(contract_module, "_link_staged_sidecar", replace_json_after_link)
    with pytest.raises(RuntimeError, match="replaced"):
        save_contract(contract, artifact)
    assert artifact.with_suffix(".contract.json").read_text(encoding="utf-8") == "foreign"
    assert not artifact.with_suffix(".contract.md").exists()
    markers = list(tmp_path.glob(".manwe-contract-*.in-progress"))
    assert len(markers) == 1
    assert (markers[0].stat().st_mode & 0o777) == 0o700
    assert (markers[0] / "contract.json").is_file()
    assert (markers[0] / "contract.md").is_file()


def test_save_contract_preserves_same_inode_tampering_and_recovery_marker(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_link = contract_module._link_staged_sidecar

    def mutate_json_after_link(stage_fd, stage_name, parent_fd, final_name):
        real_link(stage_fd, stage_name, parent_fd, final_name)
        if stage_name == contract_module._STAGE_JSON:
            fd = os.open(final_name, os.O_WRONLY | os.O_TRUNC, dir_fd=parent_fd)
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"tampered")
                handle.flush()
                os.fsync(handle.fileno())

    monkeypatch.setattr(contract_module, "_link_staged_sidecar", mutate_json_after_link)
    with pytest.raises(RuntimeError, match="modified"):
        save_contract(contract, artifact)
    assert artifact.with_suffix(".contract.json").read_bytes() == b"tampered"
    assert not artifact.with_suffix(".contract.md").exists()
    assert len(list(tmp_path.glob(".manwe-contract-*.in-progress"))) == 1


def test_sidecar_match_rejects_path_replacement_during_descriptor_read(tmp_path, monkeypatch):
    import hashlib

    from manwe.export import contract as contract_module

    sidecar = tmp_path / "contract.json"
    payload = b"trusted!"
    sidecar.write_bytes(payload)
    metadata = sidecar.stat()
    publication = contract_module._SidecarPublication(
        identity=(metadata.st_dev, metadata.st_ino),
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    real_fstat = contract_module.os.fstat
    matching_calls = 0

    def replace_after_read(fd):
        nonlocal matching_calls
        opened = real_fstat(fd)
        if (opened.st_dev, opened.st_ino) == publication.identity:
            matching_calls += 1
            if matching_calls == 2:
                sidecar.unlink()
                sidecar.write_bytes(b"foreign!")
        return opened

    monkeypatch.setattr(contract_module.os, "fstat", replace_after_read)
    try:
        assert not contract_module._sidecar_matches(
            parent_fd,
            sidecar.name,
            publication,
        )
    finally:
        os.close(parent_fd)
    assert sidecar.read_bytes() == b"foreign!"


def test_sidecar_match_closes_raw_fd_when_fdopen_fails(tmp_path, monkeypatch):
    import hashlib

    from manwe.common import fd_io
    from manwe.export import contract as contract_module

    sidecar = tmp_path / "contract.json"
    payload = b"trusted"
    sidecar.write_bytes(payload)
    metadata = sidecar.stat()
    publication = contract_module._SidecarPublication(
        identity=(metadata.st_dev, metadata.st_ino),
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    opened_fds: list[int] = []

    def fail_fdopen(fd, _mode):
        opened_fds.append(fd)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_fdopen)
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        assert not contract_module._sidecar_matches(
            parent_fd,
            sidecar.name,
            publication,
        )
    finally:
        os.close(parent_fd)
    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])


def test_save_contract_second_link_collision_preserves_partial_publication(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_link = contract_module._link_staged_sidecar

    def occupy_markdown_before_link(stage_fd, stage_name, parent_fd, final_name):
        if stage_name == contract_module._STAGE_MARKDOWN:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(final_name, flags, 0o644, dir_fd=parent_fd)
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"foreign-markdown")
                handle.flush()
                os.fsync(handle.fileno())
        real_link(stage_fd, stage_name, parent_fd, final_name)

    monkeypatch.setattr(contract_module, "_link_staged_sidecar", occupy_markdown_before_link)
    with pytest.raises(FileExistsError):
        save_contract(contract, artifact)
    assert artifact.with_suffix(".contract.json").read_text(encoding="utf-8") == contract.to_json()
    assert artifact.with_suffix(".contract.md").read_bytes() == b"foreign-markdown"
    markers = list(tmp_path.glob(".manwe-contract-*.in-progress"))
    assert len(markers) == 1
    assert (markers[0] / "contract.json").is_file()
    assert (markers[0] / "contract.md").is_file()


def test_save_contract_cleanup_refuses_replaced_staged_entry(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_cleanup = contract_module._cleanup_private_stage

    def replace_json_before_cleanup(
        parent_fd,
        stage_fd,
        stage_name,
        stage_identity,
        publications,
        commit_boundary,
    ):
        os.unlink(contract_module._STAGE_JSON, dir_fd=stage_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(contract_module._STAGE_JSON, flags, 0o600, dir_fd=stage_fd)
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"foreign-stage-entry")
            handle.flush()
            os.fsync(handle.fileno())
        return real_cleanup(
            parent_fd,
            stage_fd,
            stage_name,
            stage_identity,
            publications,
            commit_boundary,
        )

    monkeypatch.setattr(contract_module, "_cleanup_private_stage", replace_json_before_cleanup)
    with pytest.raises(RuntimeError, match="staging cleanup failed"):
        save_contract(contract, artifact)
    assert artifact.with_suffix(".contract.json").read_text(encoding="utf-8") == contract.to_json()
    assert artifact.with_suffix(".contract.md").is_file()
    markers = list(tmp_path.glob(".manwe-contract-*.in-progress"))
    assert len(markers) == 1
    assert (markers[0] / "contract.json").read_bytes() == b"foreign-stage-entry"


def test_save_contract_revalidates_final_paths_before_marker_removal(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_cleanup = contract_module._cleanup_private_stage

    def replace_final_before_cleanup(
        parent_fd,
        stage_fd,
        stage_name,
        stage_identity,
        publications,
        commit_boundary,
    ):
        json_path = artifact.with_suffix(".contract.json")
        json_path.unlink()
        json_path.write_bytes(b"foreign-final")
        return real_cleanup(
            parent_fd,
            stage_fd,
            stage_name,
            stage_identity,
            publications,
            commit_boundary,
        )

    monkeypatch.setattr(contract_module, "_cleanup_private_stage", replace_final_before_cleanup)
    with pytest.raises(RuntimeError, match="manual recovery"):
        save_contract(contract, artifact)
    assert artifact.with_suffix(".contract.json").read_bytes() == b"foreign-final"
    assert artifact.with_suffix(".contract.md").is_file()
    markers = list(tmp_path.glob(".manwe-contract-*.in-progress"))
    assert len(markers) == 1
    assert (markers[0] / "contract.json").is_file()
    assert (markers[0] / "contract.md").is_file()


def test_save_contract_reports_indeterminate_marker_removal_sync(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    monkeypatch.setattr(contract_module, "_sync_parent_after_marker_removal", lambda _fd: False)

    with pytest.raises(RuntimeError, match="commit state is indeterminate"):
        save_contract(contract, artifact)
    assert artifact.with_suffix(".contract.json").read_text(encoding="utf-8") == contract.to_json()
    assert artifact.with_suffix(".contract.md").is_file()
    assert not list(tmp_path.glob(".manwe-contract-*.in-progress"))


def test_save_contract_revalidates_parent_path_before_marker_removal(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    parent = tmp_path / "publish"
    parent.mkdir()
    moved_parent = tmp_path / "moved-publish"
    artifact = parent / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_cleanup = contract_module._cleanup_private_stage

    def replace_parent_before_cleanup(
        parent_fd,
        stage_fd,
        stage_name,
        stage_identity,
        publications,
        commit_boundary,
    ):
        parent.rename(moved_parent)
        parent.mkdir()
        (parent / "foreign.txt").write_bytes(b"foreign")
        return real_cleanup(
            parent_fd,
            stage_fd,
            stage_name,
            stage_identity,
            publications,
            commit_boundary,
        )

    monkeypatch.setattr(contract_module, "_cleanup_private_stage", replace_parent_before_cleanup)
    with pytest.raises(RuntimeError, match="manual recovery"):
        save_contract(contract, artifact)
    assert (parent / "foreign.txt").read_bytes() == b"foreign"
    assert (moved_parent / "model.onnx").read_bytes() == b"model"
    assert (moved_parent / "model.contract.json").is_file()
    assert (moved_parent / "model.contract.md").is_file()
    markers = list(moved_parent.glob(".manwe-contract-*.in-progress"))
    assert len(markers) == 1
    assert (markers[0] / "contract.json").is_file()
    assert (markers[0] / "contract.md").is_file()


def test_save_contract_revalidates_artifact_before_marker_removal(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_cleanup = contract_module._cleanup_private_stage

    def replace_artifact_before_cleanup(
        parent_fd,
        stage_fd,
        stage_name,
        stage_identity,
        publications,
        commit_boundary,
    ):
        artifact.unlink()
        artifact.write_bytes(b"foreign-artifact")
        return real_cleanup(
            parent_fd,
            stage_fd,
            stage_name,
            stage_identity,
            publications,
            commit_boundary,
        )

    monkeypatch.setattr(
        contract_module,
        "_cleanup_private_stage",
        replace_artifact_before_cleanup,
    )
    with pytest.raises(RuntimeError, match="manual recovery"):
        save_contract(contract, artifact)
    assert artifact.read_bytes() == b"foreign-artifact"
    assert artifact.with_suffix(".contract.json").is_file()
    assert artifact.with_suffix(".contract.md").is_file()
    markers = list(tmp_path.glob(".manwe-contract-*.in-progress"))
    assert len(markers) == 1
    assert (markers[0] / "contract.json").is_file()
    assert (markers[0] / "contract.md").is_file()


def test_save_contract_parent_replacement_cannot_redirect_sidecars(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    parent = tmp_path / "publish"
    parent.mkdir()
    moved_parent = tmp_path / "moved-publish"
    artifact = parent / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    real_link = contract_module._link_staged_sidecar
    replaced = False

    def replace_parent_after_first_link(stage_fd, stage_name, parent_fd, final_name):
        nonlocal replaced
        real_link(stage_fd, stage_name, parent_fd, final_name)
        if not replaced:
            parent.rename(moved_parent)
            parent.mkdir()
            (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
            replaced = True

    monkeypatch.setattr(
        contract_module,
        "_link_staged_sidecar",
        replace_parent_after_first_link,
    )
    with pytest.raises(RuntimeError, match="parent was replaced"):
        save_contract(contract, artifact)
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (moved_parent / "model.onnx").read_bytes() == b"model"
    assert (moved_parent / "model.contract.json").is_file()
    assert not (moved_parent / "model.contract.md").exists()
    assert len(list(moved_parent.glob(".manwe-contract-*.in-progress"))) == 1


def test_contract_markdown_escapes_raw_html(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    contract = _build_contract(artifact)
    contract.model_name = '<img src=x onerror="alert(1)">'
    contract.source = "<script>alert(1)</script> & source"
    markdown = contract.to_markdown()
    assert "<script>" not in markdown
    assert "<img " not in markdown
    assert "&lt;script&gt;" in markdown
    assert "&amp; source" in markdown


def test_detection_records_reject_bad_lengths_nonfinite_values_and_label_types():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    with pytest.raises(ValueError, match="length"):
        Detections(box, np.array([]), np.array([0]))
    with pytest.raises(ValueError, match="finite"):
        Detections(np.array([[0.0, 0.0, np.nan, 10.0]]), np.array([0.9]), np.array([0]))
    with pytest.raises(ValueError, match="integer class indices"):
        GroundTruth(box, np.array([0.5]))
    with pytest.raises(ValueError, match="positive-area"):
        GroundTruth(np.array([[1.0, 1.0, 1.0, 2.0]]), np.array([0]))

    # Preserve compatible model-output forms while normalizing them safely.
    flat = Detections(np.array([0.0, 0.0, 10.0, 10.0]), np.array([0.9]), np.array([0.0]))
    assert flat.boxes.shape == (1, 4)
    assert flat.labels.dtype == np.int64
    assert Detections([], [], []).boxes.shape == (0, 4)


def test_metric_rejects_out_of_range_or_mutated_nonfinite_arrays():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = [Detections(box, np.array([0.9]), np.array([5]))]
    gts = [GroundTruth(box, np.array([0]))]
    with pytest.raises(ValueError, match="outside"):
        mean_average_precision(preds, gts, num_classes=5)

    preds[0].labels[0] = 0
    preds[0].scores[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mean_average_precision(preds, gts, num_classes=5)


def test_ap_uses_best_unmatched_ground_truth_and_exact_iou_boundary():
    gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
    pred_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    scores = np.array([0.9, 0.8])
    assert average_precision(pred_boxes, scores, gt_boxes, iou_thr=0.5) == pytest.approx(1.0)

    labels = np.array([0, 0])
    metrics = mean_average_precision(
        [Detections(pred_boxes, scores, labels)],
        [GroundTruth(gt_boxes, labels)],
        num_classes=1,
    )
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics.metric_name == "mAP50"
    assert metrics.iou_threshold == 0.5

    half_overlap = iou_matrix(np.array([[0.0, 0.0, 2.0, 1.0]]), np.array([[0.0, 0.0, 1.0, 1.0]]))
    assert half_overlap[0, 0] == 0.5
    assert (
        iou_matrix(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([[0.0, 0.0, 1.0, 1.0]]))[0, 0] == 1.0
    )
    assert average_precision(
        np.array([[0.0, 0.0, 2.0, 1.0]]),
        np.array([1.0]),
        np.array([[0.0, 0.0, 1.0, 1.0]]),
        iou_thr=0.5,
    ) == pytest.approx(1.0)

    tiny = np.array([[0.0, 0.0, 1e-200, 1e-200]])
    assert iou_matrix(tiny, tiny)[0, 0] == 1.0
    assert average_precision(tiny, np.array([1.0]), tiny) == 1.0

    elongated = np.array([[0.0, 0.0, 1e153, 1e-171]])
    assert iou_matrix(elongated, elongated)[0, 0] == 1.0
    assert average_precision(elongated, np.array([1.0]), elongated) == 1.0


def test_geometry_rejects_finite_coordinates_outside_safe_float64_envelope():
    maximum = np.finfo(np.float64).max
    extreme = np.array([[-maximum, -maximum, maximum, maximum]])
    with pytest.raises(ValueError, match="coordinate magnitude"):
        iou_matrix(extreme, extreme)


def test_equal_score_ap_is_invariant_to_frame_and_box_permutations():
    truth_box = np.array([[0.0, 0.0, 10.0, 10.0]])
    false_box = np.array([[20.0, 0.0, 30.0, 10.0]])
    first_truth = GroundTruth(truth_box, np.array([0]), image_id="first")
    second_truth = GroundTruth(truth_box, np.array([0]), image_id="second")
    true_prediction = Detections(truth_box, np.array([0.5]), np.array([0]), image_id="first")
    false_prediction = Detections(false_box, np.array([0.5]), np.array([0]), image_id="second")

    forward = mean_average_precision(
        [true_prediction, false_prediction], [first_truth, second_truth], num_classes=1
    )
    reverse = mean_average_precision(
        [false_prediction, true_prediction], [second_truth, first_truth], num_classes=1
    )
    assert forward["mAP"] == reverse["mAP"] == pytest.approx(0.25)

    # The first prediction overlaps either GT, while the second can only claim
    # the first. Canonical tie ordering must make reversal immaterial.
    ground_truth = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
    predictions = np.array([[2.0, 0.0, 12.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = np.array([0.5, 0.5])
    assert average_precision(predictions, scores, ground_truth) == pytest.approx(1.0)
    assert average_precision(predictions[::-1], scores, ground_truth[::-1]) == pytest.approx(1.0)

    # Greedy matching gets only one hit here, but the equal-score group has a
    # perfect two-edge assignment and must be evaluated as one threshold.
    greedy_trap_truth = np.array([[0.0, 0.0, 6.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    greedy_trap_predictions = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 15.0, 10.0]])
    assert average_precision(
        greedy_trap_predictions,
        np.ones(2),
        greedy_trap_truth,
        iou_thr=0.5,
    ) == pytest.approx(1.0)


def test_metrics_and_fidelity_reject_missing_partial_or_empty_frames():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    labels = np.array([0])
    frame = Detections(box, np.array([0.9]), labels)
    truth = GroundTruth(box, labels)

    with pytest.raises(ValueError, match="frame count mismatch"):
        mean_average_precision([frame], [truth, truth], num_classes=1)
    with pytest.raises(ValueError, match="frame count mismatch"):
        fidelity_report([frame, frame], [frame], [truth, truth], num_classes=1)
    with pytest.raises(ValueError, match="nonempty frame lists"):
        fidelity_report([], [], [], num_classes=1)


def test_fidelity_requires_ground_truth_and_small_object_coverage():
    empty_truth = GroundTruth(np.empty((0, 4)), np.empty(0, dtype=int), image_id="frame")
    with pytest.raises(ValueError, match="no in-range ground-truth"):
        fidelity_report([_empty_detections()], [_empty_detections()], [empty_truth], num_classes=1)

    large = np.array([[0.0, 0.0, 100.0, 100.0]])
    labels = np.array([0])
    large_frame = Detections(large, np.array([0.9]), labels, image_id="frame")
    with pytest.raises(ValueError, match="no small-object ground-truth"):
        fidelity_report(
            [large_frame],
            [large_frame],
            [GroundTruth(large, labels, image_id="frame")],
            num_classes=1,
        )


def test_fidelity_rejects_nonfinite_or_negative_tolerances():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    labels = np.array([0])
    frame = Detections(box, np.array([0.9]), labels, image_id="frame")
    truth = GroundTruth(box, labels, image_id="frame")
    with pytest.raises(ValueError, match="finite nonnegative"):
        fidelity_report([frame], [frame], [truth], num_classes=1, tolerance=np.nan)
    with pytest.raises(ValueError, match="finite nonnegative"):
        fidelity_report([frame], [frame], [truth], num_classes=1, small_tolerance=-0.1)
    for name in ("tolerance", "small_tolerance", "operating_tolerance", "max_score_delta"):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            fidelity_report(
                [frame],
                [frame],
                [truth],
                num_classes=1,
                **{name: 1.000_001},
            )

    # FPPI is a count per image, not a probability-like delta, so values above
    # one remain meaningful and accepted.
    assert fidelity_report([frame], [frame], [truth], num_classes=1, fppi_tolerance=2.0).passed


def test_fidelity_does_not_allow_one_class_gain_to_hide_another_class_loss():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]])
    truth = GroundTruth(boxes, np.array([0, 1]), image_id="frame")
    reference = Detections(boxes[:1], np.array([0.9]), np.array([0]), image_id="frame")
    exported = Detections(boxes[1:], np.array([0.9]), np.array([1]), image_id="frame")

    report = fidelity_report([reference], [exported], [truth], num_classes=2, tolerance=0.0)
    assert report.ref_map == report.exp_map == 0.5
    assert report.class_metrics[0]["drop"] == 1.0
    assert report.class_metrics[1]["drop"] == -1.0
    assert not report.passed


def test_full_class_map_is_required(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    partial = {0: CREBAIN_CLASSES[0], 1: CREBAIN_CLASSES[1]}
    with pytest.raises(ValueError, match="missing class indices"):
        _build_contract(artifact, class_map=partial, num_classes=len(CREBAIN_CLASSES))

    sparse = {index: None for index in range(80)}
    sparse[0] = "aircraft"
    sparse[14] = "bird"
    contract = _build_contract(artifact, class_map=sparse, num_classes=80)
    assert contract.is_complete()
    assert contract.class_map[1] is None
    assert "| 1 | DROP |" in contract.to_markdown()


def test_fidelity_rejects_trailing_false_positives_that_ap_does_not_penalize():
    truth_box = np.array([[0.0, 0.0, 10.0, 10.0]])
    truth = GroundTruth(truth_box, np.array([0]), image_id="frame-1")
    reference = Detections(truth_box, np.array([0.9]), np.array([0]), image_id="frame-1")
    false_boxes = np.array([[20.0 + i, 20.0, 21.0 + i, 21.0] for i in range(10)])
    exported = Detections(
        np.concatenate([truth_box, false_boxes]),
        np.array([0.9] + [0.8] * 10),
        np.zeros(11, dtype=int),
        image_id="frame-1",
    )

    # Interpolated AP remains perfect after recall reaches one, but the deployed
    # operating point and direct-agreement gates must still fail.
    assert mean_average_precision([exported], [truth], 1)["mAP"] == 1.0
    report = fidelity_report([reference], [exported], [truth], num_classes=1)
    assert not report.passed
    assert report.extra_detections == 10
    assert report.delta_fppi == 10.0


def test_image_ids_prevent_positional_frame_misalignment():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    first = Detections(box, np.array([0.9]), np.array([0]), image_id="first")
    second = Detections(box, np.array([0.9]), np.array([0]), image_id="second")
    first_truth = GroundTruth(box, np.array([0]), image_id="first")
    second_truth = GroundTruth(box, np.array([0]), image_id="second")

    with pytest.raises(ValueError, match="image_id sequences are not aligned"):
        mean_average_precision([second, first], [first_truth, second_truth], num_classes=1)
    with pytest.raises(ValueError, match="unique"):
        mean_average_precision(
            [first, Detections(box, np.array([0.8]), np.array([0]), image_id="first")],
            [first_truth, second_truth],
            num_classes=1,
        )


def test_fidelity_directly_gates_localization_and_score_drift():
    truth_box = np.array([[0.0, 0.0, 10.0, 10.0]])
    truth = GroundTruth(truth_box, np.array([0]), image_id="frame")
    reference = Detections(truth_box, np.array([0.9]), np.array([0]), image_id="frame")
    # IoU with the reference is ~0.82, still a perfect TP at AP50.
    shifted = Detections(
        np.array([[1.0, 0.0, 11.0, 10.0]]),
        np.array([0.9]),
        np.array([0]),
        image_id="frame",
    )
    localization = fidelity_report([reference], [shifted], [truth], num_classes=1)
    assert localization.ref_map == localization.exp_map == 1.0
    assert localization.missing_detections == localization.extra_detections == 1
    assert not localization.passed

    rescored = Detections(truth_box, np.array([0.7]), np.array([0]), image_id="frame")
    score = fidelity_report([reference], [rescored], [truth], num_classes=1)
    assert score.observed_max_score_delta == pytest.approx(0.2)
    assert not score.passed


def test_fidelity_required_class_coverage_fails_closed():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    truth = GroundTruth(box, np.array([0]), image_id="frame")
    prediction = Detections(box, np.array([0.9]), np.array([0]), image_id="frame")
    with pytest.raises(ValueError, match="required classes have no ground-truth coverage"):
        fidelity_report(
            [prediction],
            [prediction],
            [truth],
            num_classes=2,
            required_classes=[0, 1],
        )


def test_fidelity_requires_complete_unique_frame_identity():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    unidentified = Detections(box, np.array([0.9]), np.array([0]))
    unidentified_truth = GroundTruth(box, np.array([0]))
    with pytest.raises(ValueError, match="requires image_id"):
        fidelity_report([unidentified], [unidentified], [unidentified_truth], num_classes=1)


def test_direct_agreement_maximizes_cardinality_before_iou():
    reference_boxes = np.array([[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    exported_boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0 / 3.0, 1.0]])
    labels = np.zeros(2, dtype=int)
    reference = Detections(reference_boxes, np.array([0.9, 0.8]), labels, image_id="frame")
    exported = Detections(exported_boxes, np.array([0.9, 0.8]), labels, image_id="frame")
    truth = GroundTruth(reference_boxes, labels, image_id="frame")
    report = fidelity_report([reference], [exported], [truth], num_classes=1, agreement_iou=0.5)
    assert report.matched_detections == 2
    assert report.missing_detections == report.extra_detections == 0


def test_direct_agreement_is_order_independent_and_tolerance_is_inclusive():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    labels = np.zeros(2, dtype=int)
    truth = GroundTruth(boxes, labels, image_id="frame")
    reference = Detections(boxes, np.array([0.9, 0.7]), labels, image_id="frame")
    reordered = Detections(boxes, np.array([0.7, 0.9]), labels, image_id="frame")
    report = fidelity_report([reference], [reordered], [truth], num_classes=1)
    assert report.passed
    assert report.observed_max_score_delta == 0.0

    boundary = Detections(boxes, np.array([0.85, 0.65]), labels, image_id="frame")
    boundary_report = fidelity_report(
        [reference], [boundary], [truth], num_classes=1, max_score_delta=0.05
    )
    assert boundary_report.passed
