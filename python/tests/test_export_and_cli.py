"""Export contract + fidelity gate, and CLI smoke (numpy-only paths)."""

import hashlib
import tempfile

import numpy as np
import pytest

from manwe.cli import main
from manwe.common.contracts import TensorSpec
from manwe.eval.detection import Detections, GroundTruth
from manwe.export import (
    ExportReceipt,
    VerifiedArtifactSignature,
    build_export_contract,
    fidelity_report,
)


def test_build_export_contract_complete(tmp_path):
    artifact = tmp_path / "aerial.onnx"
    artifact.write_bytes(b"onnx fixture")
    digest = hashlib.sha256(b"onnx fixture").hexdigest()
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
        source_classes=("drone", "bird", "aircraft", "helicopter", "unknown"),
        end_to_end=False,
        calibration_manifest_sha256=None,
    )
    signature = VerifiedArtifactSignature(
        artifact_sha256=digest,
        precision="float32",
        embedded_nms=False,
        opset=17,
        source_classes=("drone", "bird", "aircraft", "helicopter", "unknown"),
        inputs=(TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
        outputs=(TensorSpec("output", [1, 9, "A"], "float32"),),
        preprocess="fixture-inspected letterbox contract",
        postprocess="fixture-inspected external NMS contract",
        failure_behavior="fixture-backed shape mismatch raises",
        evidence="tests/fixtures/export-signature.json",
    )
    c = build_export_contract(
        model_name="manwe-aerial",
        model_version="0.2.0",
        source="fine-tune of yolo11s on drone-vs-bird",
        rights="weights self-produced; MIT",
        receipt=receipt,
        signature=signature,
        validation_data="tests/fixtures/*.png",
        benchmark_context="M4 Max, CoreML EP",
    )
    assert c.backend == "onnx"
    assert c.is_complete(), c.missing_fields()
    c.validate_class_map()
    assert c.class_map[0] == "drone"


def test_receipt_and_signature_reject_forged_public_values():
    values = {
        "format": "onnx",
        "artifact_path": "model.onnx",
        "artifact_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "source_suffix": ".pt",
        "image_size": 640,
        "precision": "float32",
        "embedded_nms": False,
        "opset": 17,
        "class_count": 1,
        "source_classes": ("drone",),
        "end_to_end": False,
        "calibration_manifest_sha256": None,
    }
    for override in (
        {"image_size": True},
        {"end_to_end": "false"},
        {"precision": 123},
        {"calibration_manifest_sha256": 123},
        {"embedded_nms": True, "end_to_end": True},
        {"artifact_path": "wrong.engine"},
        {"artifact_path": "bad\nmodel.onnx"},
        {"source_suffix": ".pt/other"},
    ):
        with pytest.raises((TypeError, ValueError)):
            ExportReceipt(**(values | override))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="calibration provenance"):
        ExportReceipt(
            **(
                values
                | {
                    "format": "tensorrt",
                    "artifact_path": "model.engine",
                    "precision": "int8",
                }
            )
        )

    with pytest.raises(ValueError, match="precision"):
        VerifiedArtifactSignature(
            artifact_sha256="a" * 64,
            precision=123,  # type: ignore[arg-type]
            embedded_nms=False,
            opset=17,
            source_classes=("drone",),
            inputs=(TensorSpec("images", [1, 3, 32, 32], "float32"),),
            outputs=(TensorSpec("output", [1, 5, 1], "float32"),),
            preprocess="fixture",
            postprocess="fixture",
            failure_behavior="fixture",
            evidence="fixture",
        )


def test_fidelity_gate_pass_and_fail():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]], float)
    labels = np.array([0, 1, 2])
    gts = [GroundTruth(boxes, labels, image_id="frame-0")]
    ref = [Detections(boxes, np.array([0.9, 0.85, 0.8]), labels, image_id="frame-0")]

    identical = fidelity_report(ref, ref, gts, num_classes=5, tolerance=0.005)
    assert identical.passed and abs(identical.delta_map) < 1e-9
    assert identical.metric_name == "mAP50"
    assert identical.small_metric_name == "mAP50-small"
    assert identical.frame_count == 1
    assert identical.to_dict()["ref_map50"] == identical.ref_map

    degraded = [Detections(boxes[:2], np.array([0.9, 0.85]), labels[:2], image_id="frame-0")]
    bad = fidelity_report(ref, degraded, gts, num_classes=5, tolerance=0.005)
    assert not bad.passed and bad.delta_map > 0.005


def test_cli_numpy_only_commands(capsys):
    assert main(["doctor"]) == 0
    assert "manwe-perception[rfdetr]" in capsys.readouterr().out
    assert main(["models"]) == 0
    assert main(["models", "--track", "accuracy"]) == 0
    assert main(["data"]) == 0
    assert main(["data", "anti-uav410"]) == 0
    with tempfile.TemporaryDirectory() as d:
        assert main(["synth", d, "--n-train", "3", "--n-val", "1"]) == 0
    assert (
        main(
            [
                "fusion-sim",
                "--duration",
                "6",
                "--filters",
                "kalman",
                "ekf",
                "--modalities",
                "visual",
                "radar",
            ]
        )
        == 0
    )
