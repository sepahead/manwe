"""Export contract + fidelity gate, and CLI smoke (numpy-only paths)."""

import builtins
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

_SOURCE_CLASSES = ("drone", "bird", "aircraft", "helicopter", "unknown")


def _export_pair(
    tmp_path,
    *,
    manwe_format="onnx",
    precision="float32",
    embedded_nms=False,
    end_to_end=False,
    inputs=None,
    outputs=None,
):
    suffix = {"onnx": ".onnx", "tensorrt": ".engine"}[manwe_format]
    artifact = tmp_path / f"signature{suffix}"
    artifact.write_bytes(b"signature fixture")
    digest = hashlib.sha256(b"signature fixture").hexdigest()
    receipt = ExportReceipt(
        format=manwe_format,
        artifact_path=str(artifact),
        artifact_sha256=digest,
        source_sha256="1" * 64,
        source_suffix=".pt",
        image_size=640,
        precision=precision,
        embedded_nms=embedded_nms,
        opset=17,
        class_count=len(_SOURCE_CLASSES),
        source_classes=_SOURCE_CLASSES,
        end_to_end=end_to_end,
        calibration_manifest_sha256="2" * 64 if precision == "int8" else None,
    )
    signature = VerifiedArtifactSignature(
        artifact_sha256=digest,
        precision=precision,
        embedded_nms=embedded_nms,
        opset=17,
        source_classes=_SOURCE_CLASSES,
        inputs=inputs or (TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
        outputs=outputs or (TensorSpec("output", [1, 9, 8400], "float32"),),
        preprocess="fixture-inspected letterbox contract",
        postprocess="fixture-inspected external NMS contract",
        failure_behavior="fixture-backed shape mismatch raises",
        evidence="tests/fixtures/export-signature.json",
    )
    return receipt, signature


def _build_test_contract(receipt, signature):
    return build_export_contract(
        model_name="manwe-aerial",
        model_version="0.2.0",
        source="test fixture",
        rights="test fixture",
        receipt=receipt,
        signature=signature,
        validation_data="tests/fixtures/*.png",
        benchmark_context="test CPU",
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
        outputs=(TensorSpec("output", [1, 9, 8400], "float32"),),
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


@pytest.mark.parametrize(
    "tensor",
    (
        TensorSpec("bad name", [1, 3, 640, 640], "float32", "NCHW/RGB"),
        TensorSpec("images", [1, 3, 640, 640], "bananas", "NCHW/RGB"),
        TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGBA"),
    ),
)
def test_verified_signature_rejects_noncanonical_tensor_metadata(tensor):
    with pytest.raises(ValueError, match="invalid signature tensor metadata"):
        VerifiedArtifactSignature(
            artifact_sha256="a" * 64,
            precision="float32",
            embedded_nms=False,
            opset=17,
            source_classes=("drone",),
            inputs=(tensor,),
            outputs=(TensorSpec("output", [1, 5, "A"], "float32"),),
            preprocess="fixture",
            postprocess="fixture",
            failure_behavior="fixture",
            evidence="fixture",
        )


def test_verified_signature_rejects_unbounded_output_symbol():
    with pytest.raises(ValueError, match="canonical symbols"):
        VerifiedArtifactSignature(
            artifact_sha256="a" * 64,
            precision="float32",
            embedded_nms=False,
            opset=17,
            source_classes=("drone",),
            inputs=(TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
            outputs=(TensorSpec("output", [1, 5, "UNBOUNDED"], "float32"),),
            preprocess="fixture",
            postprocess="fixture",
            failure_behavior="fixture",
            evidence="fixture",
        )


def test_verified_signature_rejects_duplicate_interface_names():
    duplicate = TensorSpec("tensor", [1, 3, 640, 640], "float32", "NCHW/RGB")
    with pytest.raises(ValueError, match="duplicate tensor names"):
        VerifiedArtifactSignature(
            artifact_sha256="a" * 64,
            precision="float32",
            embedded_nms=False,
            opset=17,
            source_classes=("drone",),
            inputs=(duplicate,),
            outputs=(TensorSpec("tensor", [1, 5, "A"], "float32"),),
            preprocess="fixture",
            postprocess="fixture",
            failure_behavior="fixture",
            evidence="fixture",
        )


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    (
        (
            (TensorSpec("images", [99, 7, 13], "float32", "NCHW/RGB"),),
            (TensorSpec("output", [1, 9, 8400], "float32"),),
            "image input shape",
        ),
        (
            (TensorSpec("images", [1, 3, 640, 640], "float32"),),
            (TensorSpec("output", [1, 9, 8400], "float32"),),
            "image input layout",
        ),
        (
            (TensorSpec("images", [1, 640, 640, 3], "float32", "NHWC/RGB"),),
            (TensorSpec("output", [1, 9, 8400], "float32"),),
            "image input layout",
        ),
        (
            (TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
            (TensorSpec("output", [1, 8, 8400], "float32"),),
            "prediction output",
        ),
        (
            (TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
            (
                TensorSpec("output0", [1, 9, 8400], "float32"),
                TensorSpec("output1", [1, 9, 8400], "float32"),
            ),
            "exactly one",
        ),
        (
            (TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
            (TensorSpec("output", [1, 9, "A"], "float32"),),
            "concrete integer",
        ),
        (
            (TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB"),),
            (TensorSpec("output", [1, 9, 2_000_001], "float32"),),
            "anchor dimension",
        ),
    ),
)
def test_build_contract_rejects_unbound_detect_signatures(tmp_path, inputs, outputs, message):
    receipt, signature = _export_pair(tmp_path, inputs=inputs, outputs=outputs)
    with pytest.raises(ValueError, match=message):
        _build_test_contract(receipt, signature)


@pytest.mark.parametrize(
    ("embedded_nms", "end_to_end", "message"),
    ((True, False, "embedded-NMS"), (False, True, "end-to-end")),
)
def test_build_contract_rejects_unmodeled_detect_output_variants(
    tmp_path, embedded_nms, end_to_end, message
):
    receipt, signature = _export_pair(
        tmp_path,
        embedded_nms=embedded_nms,
        end_to_end=end_to_end,
    )
    with pytest.raises(ValueError, match=message):
        _build_test_contract(receipt, signature)


@pytest.mark.parametrize("anchor_count", (1, 8400, 2_000_000))
def test_build_contract_accepts_bounded_concrete_anchor_counts(tmp_path, anchor_count):
    outputs = (TensorSpec("output", [1, 9, anchor_count], "float32"),)
    receipt, signature = _export_pair(tmp_path, outputs=outputs)
    contract = _build_test_contract(receipt, signature)
    assert contract.outputs[0].shape[2] == anchor_count


def test_int8_receipt_accepts_float32_engine_io(tmp_path):
    receipt, signature = _export_pair(
        tmp_path,
        manwe_format="tensorrt",
        precision="int8",
    )
    contract = _build_test_contract(receipt, signature)
    assert contract.inputs[0].dtype == "float32"
    assert contract.outputs[0].dtype == "float32"


def test_build_contract_revalidates_mutated_signature_tensors(tmp_path):
    receipt, signature = _export_pair(tmp_path)
    signature.inputs[0].shape[1] = 99
    with pytest.raises(ValueError, match="image input shape"):
        _build_test_contract(receipt, signature)

    receipt, signature = _export_pair(tmp_path)
    signature.outputs[0].dtype = "bananas"
    with pytest.raises(ValueError, match="invalid signature tensor metadata"):
        _build_test_contract(receipt, signature)


def test_verified_signature_and_built_contract_own_tensor_snapshots(tmp_path):
    supplied_input = TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB")
    receipt, signature = _export_pair(tmp_path, inputs=(supplied_input,))
    supplied_input.shape[1] = 99
    assert signature.inputs[0].shape[1] == 3

    contract = _build_test_contract(receipt, signature)
    signature.inputs[0].shape[1] = 77
    assert contract.inputs[0].shape[1] == 3


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


def test_cli_numpy_only_commands(capsys, monkeypatch):
    real_import = builtins.__import__

    def import_without_rfdetr(name, *args, **kwargs):
        if name == "rfdetr":
            raise ImportError("simulated absent optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_rfdetr)
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
