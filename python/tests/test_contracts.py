"""Contract layer: crebain class taxonomy + model contract record."""

import hashlib
import json

from manwe.common.contracts import (
    CREBAIN_CLASSES,
    ModelContract,
    TensorSpec,
    coco_to_crebain,
    crebain_class_index,
)


def test_taxonomy_matches_crebain():
    assert CREBAIN_CLASSES == ("drone", "bird", "aircraft", "helicopter", "unknown")
    assert crebain_class_index("drone") == 0
    assert crebain_class_index("unknown") == 4


def test_coco_fallback_mapping():
    assert coco_to_crebain("airplane") == "aircraft"
    assert coco_to_crebain("bird") == "bird"
    assert coco_to_crebain("car") is None  # not an airspace object


def test_model_contract_completeness_and_serialisation(tmp_path):
    incomplete = ModelContract(
        model_name="yolo",
        model_version="v12n",
        source="",
        rights="",
        backend="onnx",
        file_path="",
    )
    assert not incomplete.is_complete()
    assert "rights" in incomplete.missing_fields()
    assert "file_sha256" in incomplete.missing_fields()
    assert "benchmark_context" in incomplete.missing_fields()

    artifact = tmp_path / "aerial.onnx"
    artifact.write_bytes(b"trusted model fixture")

    complete = ModelContract(
        model_name="manwe-aerial",
        model_version="0.2.0",
        source="manwe from-scratch yolo architecture training fixture",
        rights="MIT; weights self-produced",
        backend="onnx",
        file_path=str(artifact),
        num_classes=len(CREBAIN_CLASSES),
        source_classes=list(CREBAIN_CLASSES),
        file_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        source_sha256="1" * 64,
        export_options='{"format":"onnx","opset":17}',
        signature_evidence="tests/fixtures/export-signature.json",
        inputs=[TensorSpec("images", ["B", 3, 640, 640], "float32", "NCHW/RGB", "0..1")],
        outputs=[TensorSpec("output0", ["B", 9, 8400], "float32", "", "4 bbox + 5 cls")],
        preprocess="letterbox 640, /255, RGB",
        postprocess="NMS iou=0.45 conf=0.25 max=300",
        class_map={i: c for i, c in enumerate(CREBAIN_CLASSES)},
        validation_data="tests/fixtures/aerial/*.png",
        benchmark_context="M4 Max, macOS, ONNX Runtime, conf=0.25",
        failure_behavior="raise on missing file / wrong extension",
    )
    assert complete.is_complete()
    complete.validate_class_map()  # must not raise
    parsed = json.loads(complete.to_json())
    assert parsed["schema_version"] == "1.2"
    assert parsed["num_classes"] == 5
    assert parsed["class_map"]["0"] == "drone"
    assert "Model Contract" in complete.to_markdown()
