"""Portable Manwe taxonomy and machine-consumable model contracts."""

import hashlib
import importlib
import json

import pytest

from manwe.common.contracts import (
    AIRSPACE_CLASSES,
    COCO_KEYPOINT_NAMES,
    CREBAIN_CLASSES,
    ModelContract,
    TensorSpec,
    airspace_class_index,
    coco_to_airspace,
    coco_to_crebain,
    crebain_class_index,
    detection_runtime,
)
from manwe.export import save_contract
from manwe.schemas import candle_detect_example_bytes

_CANDLE_CONTRACT_EXAMPLE = candle_detect_example_bytes()


def _candle_contract_for_artifact(artifact) -> ModelContract:
    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["file_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return ModelContract.from_dict(payload)


def test_manwe_taxonomy_and_legacy_aliases_are_stable():
    assert AIRSPACE_CLASSES == ("drone", "bird", "aircraft", "helicopter", "unknown")
    assert CREBAIN_CLASSES == ("drone", "bird", "aircraft", "helicopter", "unknown")
    assert airspace_class_index("drone") == 0
    assert crebain_class_index("drone") == 0
    assert crebain_class_index("unknown") == 4


def test_coco_fallback_mapping():
    assert coco_to_airspace("airplane") == "aircraft"
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
        file_path=artifact.name,
        num_classes=len(CREBAIN_CLASSES),
        source_classes=list(CREBAIN_CLASSES),
        file_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        source_sha256="1" * 64,
        export_options='{"format":"onnx","opset":17}',
        signature_evidence="tests/fixtures/export-signature.json",
        inputs=[TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW/RGB", "0..1")],
        outputs=[TensorSpec("output0", [1, 9, 8400], "float32", "", "4 bbox + 5 cls")],
        preprocess="letterbox 640, /255, RGB",
        postprocess="NMS iou=0.45 conf=0.25 max=300",
        class_map={i: c for i, c in enumerate(CREBAIN_CLASSES)},
        validation_data="tests/fixtures/aerial/*.png",
        benchmark_context="M4 Max, macOS, ONNX Runtime, conf=0.25",
        failure_behavior="raise on missing file / wrong extension",
        runtime=detection_runtime(
            input_tensor="images",
            output_tensor="output0",
            image_size=640,
        ),
    )
    assert complete.is_complete()
    complete.verify_artifact(artifact)
    complete.validate(check_artifact=True, artifact_path=artifact)
    with pytest.raises(ValueError, match="artifact_path is required"):
        complete.validate(check_artifact=True)
    complete.validate_class_map()  # must not raise
    parsed = json.loads(complete.to_json())
    assert parsed["schema_version"] == "2.0"
    assert parsed["num_classes"] == 5
    assert parsed["class_map"]["0"] == "drone"
    assert "Model Contract" in complete.to_markdown()
    assert "Invalid contract" not in complete.to_markdown(
        check_artifact=True,
        artifact_path=artifact,
    )

    restored = ModelContract.from_json(complete.to_json())
    assert restored.to_dict() == complete.to_dict()

    sidecar = tmp_path / "aerial.contract.json"
    sidecar.write_text(complete.to_json(), encoding="utf-8")
    loaded = ModelContract.load(sidecar)
    assert loaded.file_sha256 == complete.file_sha256

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        ModelContract.load(sidecar)
    assert ModelContract.load(sidecar, verify_artifact=False).is_complete()


def test_cross_language_candle_contract_fixture_is_canonical():
    contract = ModelContract.from_json(_CANDLE_CONTRACT_EXAMPLE)
    assert contract.backend == "candle"
    assert contract.runtime is not None
    assert contract.runtime.adapter == "manwe-candle-yolov8-detect-v1"
    assert contract.runtime.input.width == 640
    assert contract.runtime.input.alignment == 32
    assert contract.runtime.output.max_detections == 300
    assert contract.runtime.output.keypoint_confidence_threshold is None
    assert json.loads(contract.to_json()) == json.loads(_CANDLE_CONTRACT_EXAMPLE)

    uppercase_suffix = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    uppercase_suffix["file_path"] = "model.SAFETENSORS"
    assert ModelContract.from_dict(uppercase_suffix).file_path == "model.SAFETENSORS"


def test_contract_publication_enforces_the_native_model_size_limit(tmp_path, monkeypatch):
    from manwe.export import contract as contract_module

    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"three")
    contract = _candle_contract_for_artifact(artifact)
    monkeypatch.setattr(contract_module, "MAX_NATIVE_MODEL_BYTES", 2)

    with pytest.raises(ValueError, match="2-byte safety limit"):
        save_contract(contract, artifact)
    assert not artifact.with_suffix(".contract.json").exists()
    assert not artifact.with_suffix(".contract.md").exists()


def test_contract_publication_rejects_a_nonsticky_shared_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    artifact = parent / "model.safetensors"
    artifact.write_bytes(b"trusted weights")
    contract = _candle_contract_for_artifact(artifact)

    with pytest.raises(PermissionError, match="group/other-writable.*sticky"):
        save_contract(contract, artifact)

    assert sorted(path.name for path in parent.iterdir()) == [artifact.name]


def test_contract_parent_check_binds_both_path_and_retained_descriptor(tmp_path):
    from manwe.export import contract as contract_module

    expected = tmp_path / "expected"
    other = tmp_path / "other"
    expected.mkdir()
    other.mkdir()
    expected_metadata = expected.stat()
    other_fd = contract_module.open_directory_nofollow(other.resolve(), "other parent")
    try:
        with pytest.raises(RuntimeError, match="contract parent was replaced"):
            contract_module._assert_directory_path(
                expected.resolve(),
                other_fd,
                (expected_metadata.st_dev, expected_metadata.st_ino),
            )
    finally:
        contract_module.os.close(other_fd)


def test_contract_loader_keeps_both_reads_on_one_bound_directory(tmp_path, monkeypatch):
    contract_module = importlib.import_module("manwe.common.contracts")
    visible = tmp_path / "visible"
    attacker = tmp_path / "attacker"
    parked = tmp_path / "parked"
    visible.mkdir()
    attacker.mkdir()
    trusted_artifact = b"trusted package bytes"
    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["file_sha256"] = hashlib.sha256(trusted_artifact).hexdigest()
    contract = ModelContract.from_dict(payload)
    (visible / "model.safetensors").write_bytes(trusted_artifact)
    sidecar = visible / "model.contract.json"
    sidecar.write_text(contract.to_json(), encoding="utf-8")
    (attacker / "model.safetensors").write_bytes(b"substituted bytes")

    original_verify = contract_module._verify_artifact_at

    def swap_visible_parent_during_artifact_read(*args, **kwargs):
        visible.rename(parked)
        attacker.rename(visible)
        try:
            return original_verify(*args, **kwargs)
        finally:
            visible.rename(attacker)
            parked.rename(visible)

    monkeypatch.setattr(
        contract_module, "_verify_artifact_at", swap_visible_parent_during_artifact_read
    )

    loaded = ModelContract.load(sidecar)

    assert loaded.file_sha256 == hashlib.sha256(trusted_artifact).hexdigest()
    assert (visible / "model.safetensors").read_bytes() == trusted_artifact


@pytest.mark.parametrize("verify_artifact", [0, 1, "yes", None])
def test_contract_loader_requires_an_exact_boolean_verification_policy(tmp_path, verify_artifact):
    sidecar = tmp_path / "model.contract.json"
    sidecar.write_bytes(_CANDLE_CONTRACT_EXAMPLE)
    with pytest.raises(TypeError, match="must be a boolean"):
        ModelContract.load(sidecar, verify_artifact=verify_artifact)


@pytest.mark.parametrize(
    "file_path",
    (
        "../model.onnx",
        "/model.onnx",
        "nested/model.onnx",
        "nested\\model.onnx",
        ".model.onnx",
        "model:onnx",
        "mødel.onnx",
    ),
)
def test_contract_rejects_nonportable_artifact_names(file_path):
    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["backend"] = "onnx"
    payload["file_path"] = file_path
    payload["runtime"]["adapter"] = "ultralytics-raw-detect-v1"
    payload["runtime"]["model_variant"] = None
    payload["runtime"]["input"]["interpolation"] = "bilinear"
    with pytest.raises(ValueError, match="portable ASCII artifact filename"):
        ModelContract.from_dict(payload)


def test_contract_json_parser_rejects_ambiguity_and_nonfinite_numbers():
    with pytest.raises(ValueError, match="duplicate field 'schema_version'"):
        ModelContract.from_json('{"schema_version":"2.0","schema_version":"2.0"}')
    with pytest.raises(ValueError, match="non-finite"):
        ModelContract.from_json('{"value":1e999}')
    with pytest.raises(ValueError, match="valid UTF-8 text"):
        ModelContract.from_json('"\ud800"')
    with pytest.raises(ValueError, match="depth limit|invalid JSON"):
        ModelContract.from_json("[" * 2_000 + "0" + "]" * 2_000)

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["model_name"] = chr(0xD800)
    with pytest.raises(ValueError, match="bounded printable text"):
        ModelContract.from_json(json.dumps(payload))

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        ModelContract.from_dict(payload)

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["runtime"]["output"]["confidence_threshold"] = 10**400
    with pytest.raises(ValueError, match="finite number"):
        ModelContract.from_dict(payload)


def test_runtime_contract_is_tied_to_tensor_interface():
    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["runtime"]["output"]["tensor_name"] = "wrong"
    with pytest.raises(ValueError, match="tensor_name does not match"):
        ModelContract.from_dict(payload)

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["outputs"][0]["shape"][1] = 84
    with pytest.raises(ValueError, match="concrete shape"):
        ModelContract.from_dict(payload)

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["outputs"][0]["dtype"] = "float16"
    with pytest.raises(ValueError, match="native runtime output must be float32"):
        ModelContract.from_dict(payload)

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["runtime"]["input"]["scale"] = 0.003921568627450981
    with pytest.raises(ValueError, match="preprocessing must match native prepare_image"):
        ModelContract.from_dict(payload)

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["outputs"][0]["shape"][2] = 8401
    with pytest.raises(ValueError, match=r"three stride-grid sizes \(8400\)"):
        ModelContract.from_dict(payload)

    contract = ModelContract.from_json(_CANDLE_CONTRACT_EXAMPLE)
    assert contract.runtime is not None
    native_class_errors = contract.runtime.validation_errors(
        backend="candle",
        num_classes=1001,
        inputs=contract.inputs,
        outputs=contract.outputs,
    )
    assert "native runtime supports at most 1000 classes" in native_class_errors

    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["runtime"]["adapter"] = "manwe-candle-yolov8-pose-v1"
    payload["runtime"]["keypoint_names"] = list(COCO_KEYPOINT_NAMES)
    payload["runtime"]["output"]["keypoint_confidence_threshold"] = 0.6
    payload["outputs"][0]["shape"][1] = 4 + payload["num_classes"] + 3 * len(COCO_KEYPOINT_NAMES)
    with pytest.raises(ValueError, match="pose runtime requires exactly one class"):
        ModelContract.from_dict(payload)


@pytest.mark.parametrize("threshold", [5e-324, 0.99999999])
def test_native_runtime_rejects_thresholds_that_collapse_to_float32_endpoints(threshold):
    payload = json.loads(_CANDLE_CONTRACT_EXAMPLE)
    payload["runtime"]["output"]["confidence_threshold"] = threshold

    with pytest.raises(ValueError, match="binary32 probability endpoint"):
        ModelContract.from_dict(payload)


def test_pose_runtime_makes_alignment_and_keypoint_policy_explicit():
    runtime = detection_runtime(
        input_tensor="images",
        output_tensor="predictions",
        image_size=640,
        adapter="manwe-candle-yolov8-pose-v1",
        model_variant="s",
        keypoint_names=COCO_KEYPOINT_NAMES,
    )

    assert runtime.input.alignment == 32
    assert runtime.output.keypoint_confidence_threshold == 0.6
    assert runtime.task == "pose"

    with pytest.raises(ValueError, match="detection-only"):
        detection_runtime(
            input_tensor="images",
            output_tensor="predictions",
            image_size=640,
            keypoint_confidence_threshold=0.6,
        )
