"""Adversarial admission tests for previously unreviewed numeric boundaries."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from manwe.common.contracts import MAX_CONTRACT_CLASSES, TensorSpec
from manwe.common.numeric import finite_float64_scalar
from manwe.data.synthetic import write_png
from manwe.eval.detection import Detections, GroundTruth
from manwe.export import contract as contract_module
from manwe.export import fidelity as fidelity_module
from manwe.export.contract import VerifiedArtifactSignature
from manwe.vision import sahi_infer
from manwe.vision.input import prepare_single_image
from manwe.vision.sahi_infer import SliceConfig
from manwe.vision.train import VisionTrainConfig


class _Coercive:
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("untrusted __float__ must not run")

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("untrusted __int__ must not run")


class _CoerciveFloat(float):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("numeric-subclass __float__ must not run")


class _CoerciveString(str):
    calls = 0

    def strip(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("string-subclass strip callback must not run")

    def lower(self):
        type(self).calls += 1
        raise AssertionError("string-subclass lower callback must not run")

    def encode(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("string-subclass encode callback must not run")

    def isprintable(self):
        type(self).calls += 1
        raise AssertionError("string-subclass isprintable callback must not run")


@pytest.fixture(autouse=True)
def _reset_coercion_counts():
    _Coercive.calls = 0
    _CoerciveFloat.calls = 0
    _CoerciveString.calls = 0
    yield
    assert _Coercive.calls == 0
    assert _CoerciveFloat.calls == 0
    assert _CoerciveString.calls == 0


def _signature(input_tensor: TensorSpec | None = None) -> VerifiedArtifactSignature:
    return VerifiedArtifactSignature(
        artifact_sha256="0" * 64,
        precision="float32",
        embedded_nms=False,
        opset=17,
        source_classes=("drone",),
        inputs=(input_tensor or TensorSpec("images", [1, 3, 32, 32], "float32", "NCHW/RGB"),),
        outputs=(TensorSpec("output0", [1, 5, 21], "float32"),),
        preprocess="bounded RGB",
        postprocess="raw detect",
        failure_behavior="reject malformed inputs",
        evidence="fixture inspection",
    )


def _fidelity_fixture() -> tuple[list[Detections], list[Detections], list[GroundTruth]]:
    prediction = Detections(
        np.array([[0.0, 0.0, 4.0, 4.0]]),
        np.array([0.9]),
        np.array([0]),
        image_id="frame",
    )
    truth = GroundTruth(
        np.array([[0.0, 0.0, 4.0, 4.0]]),
        np.array([0]),
        image_id="frame",
    )
    return [prediction], [prediction], [truth]


def test_shared_scalar_admission_rejects_callbacks_overflow_and_lossy_narrowing():
    for value in (_Coercive(), _CoerciveFloat(0.5), 10**10_000):
        with pytest.raises(ValueError, match="finite|representable"):
            finite_float64_scalar(value, "value")

    assert finite_float64_scalar(np.float32(0.5), "value") == 0.5
    assert finite_float64_scalar(np.array(0.5), "value") == 0.5

    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        precise = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
        with pytest.raises(ValueError, match="loses precision"):
            finite_float64_scalar(precise, "value")


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("tolerance", _Coercive()),
        ("iou_thr", _CoerciveFloat(0.5)),
        ("fppi_tolerance", 10**10_000),
    ),
    ids=("tolerance-object", "iou-subclass", "fppi-huge-int"),
)
def test_fidelity_scalar_admission_runs_before_evaluation(keyword, value):
    reference, exported, truth = _fidelity_fixture()
    with pytest.raises(ValueError, match=keyword):
        fidelity_module.fidelity_report(
            reference,
            exported,
            truth,
            num_classes=1,
            **{keyword: value},
        )


def test_fidelity_frame_cap_precedes_frame_attribute_materialization(monkeypatch):
    monkeypatch.setattr(fidelity_module, "_MAX_FIDELITY_FRAMES", 1)
    invalid_frames = [object(), object()]
    with pytest.raises(ValueError, match="frame safety limit"):
        fidelity_module.fidelity_report(
            invalid_frames,  # type: ignore[arg-type]
            invalid_frames,  # type: ignore[arg-type]
            invalid_frames,  # type: ignore[arg-type]
            num_classes=1,
        )


class _ExplosiveFrameList(list[object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("frame-list subclass length callback must not run")


class _ExplosiveCollection:
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("required-class collection callback must not run")

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("required-class collection callback must not run")


def test_fidelity_requires_exact_builtin_collections_before_len_or_iteration():
    frames = _ExplosiveFrameList()
    with pytest.raises(ValueError, match="exact built-in list"):
        fidelity_module.fidelity_report(frames, frames, frames, num_classes=1)  # type: ignore[arg-type]
    assert _ExplosiveFrameList.calls == 0

    with pytest.raises(ValueError, match="built-in"):
        fidelity_module._validate_required_classes(
            _ExplosiveCollection(),  # type: ignore[arg-type]
            1,
            "required_classes",
        )
    assert _ExplosiveCollection.calls == 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"lr0": _CoerciveFloat(0.1)},
        {"mosaic": _CoerciveFloat(0.1)},
        {"extra": {"weight_decay": _CoerciveFloat(0.1)}},
        {"lr0": 10**10_000},
        {"extra": {"weight_decay": 10**10_000}},
    ),
    ids=("lr-subclass", "mosaic-subclass", "extra-subclass", "lr-huge-int", "extra-huge-int"),
)
def test_training_config_rejects_callback_and_overflow_scalars(kwargs):
    with pytest.raises(ValueError, match="finite"):
        VisionTrainConfig(data="dataset.yaml", **kwargs)


def test_training_config_normalizes_exact_numpy_scalars():
    config = VisionTrainConfig(
        data="dataset.yaml",
        lr0=np.float32(0.5),
        mosaic=np.float32(0.25),
        extra={"weight_decay": np.float32(0.125)},
    )
    assert type(config.lr0) is float
    assert type(config.mosaic) is float
    assert type(config.extra["weight_decay"]) is float


def test_training_config_rejects_string_subclasses_without_callbacks():
    with pytest.raises(ValueError, match="model must be a built-in string"):
        VisionTrainConfig(data="dataset.yaml", model=_CoerciveString("yolo11s"))
    with pytest.raises(ValueError, match="data must be a nonempty"):
        VisionTrainConfig(data=_CoerciveString("dataset.yaml"))
    with pytest.raises(ValueError, match="supported optimizer"):
        VisionTrainConfig(
            data="dataset.yaml",
            extra={"optimizer": _CoerciveString("SGD")},
        )


class _OversizedMapping(Mapping[str, object]):
    iterated = False

    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"oversized mapping must not be read: {key}")

    def __iter__(self) -> Iterator[str]:
        type(self).iterated = True
        raise AssertionError("oversized mapping must not be iterated")

    def __len__(self) -> int:
        return 129


class _NonemptyExplosiveList(list[str]):
    iterated = False

    def __iter__(self):
        type(self).iterated = True
        raise AssertionError("rejected export formats must not be copied")


def test_training_collection_caps_precede_defensive_copies():
    _OversizedMapping.iterated = False
    with pytest.raises(ValueError, match="at most 128"):
        VisionTrainConfig(data="dataset.yaml", extra=_OversizedMapping())
    assert not _OversizedMapping.iterated

    formats = _NonemptyExplosiveList(["onnx"])
    with pytest.raises(ValueError, match="sequence"):
        VisionTrainConfig(data="dataset.yaml", export_formats=formats)
    assert not _NonemptyExplosiveList.iterated


def test_slice_config_rejects_callbacks_and_normalizes_numpy_scalars():
    with pytest.raises(ValueError, match="conf"):
        SliceConfig(conf=_CoerciveFloat(0.5))
    config = SliceConfig(
        overlap_height_ratio=np.float32(0.25),
        overlap_width_ratio=np.float32(0.25),
        conf=np.float32(0.5),
    )
    assert type(config.overlap_height_ratio) is float
    assert type(config.overlap_width_ratio) is float
    assert type(config.conf) is float


def test_sliced_ndarray_work_cap_precedes_owned_image_copy(monkeypatch):
    image = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.uint8), (1000, 1000, 3))
    config = SliceConfig(
        slice_height=32,
        slice_width=32,
        overlap_height_ratio=0.99,
        overlap_width_ratio=0.99,
    )
    monkeypatch.setattr(
        sahi_infer,
        "prepare_single_image",
        lambda _image: pytest.fail("slice work must be admitted before copying the image"),
    )
    with pytest.raises(ValueError, match="slice plan"):
        sahi_infer.sliced_predict(
            "weights.pt",
            image,
            config,
            expected_sha256="0" * 64,
        )


class _ArraySubclass(np.ndarray):
    pass


class _ArrayCoercive:
    calls = 0

    def __array__(self):
        type(self).calls += 1
        raise AssertionError("untrusted __array__ must not run")


def test_image_boundaries_reject_ndarray_subclasses_before_array_hooks():
    image = np.zeros((2, 2, 3), dtype=np.uint8).view(_ArraySubclass)
    with pytest.raises(TypeError, match="numpy array"):
        prepare_single_image(image)
    with pytest.raises(TypeError, match="numpy array"):
        sahi_infer.sliced_predict(
            "weights.pt",
            image,
            SliceConfig(),
            expected_sha256="0" * 64,
        )


def test_png_boundary_rejects_array_coercion_and_subclasses(tmp_path):
    coercive = _ArrayCoercive()
    with pytest.raises(ValueError, match="uint8 array"):
        write_png(tmp_path / "coercive.png", coercive)  # type: ignore[arg-type]
    assert _ArrayCoercive.calls == 0
    subclass = np.zeros((2, 2, 3), dtype=np.uint8).view(_ArraySubclass)
    with pytest.raises(ValueError, match="uint8 array"):
        write_png(tmp_path / "subclass.png", subclass)


class _CoerciveDimension(int):
    calls = 0

    def __le__(self, other):
        type(self).calls += 1
        raise AssertionError("numeric dimension comparison callback must not run")

    def __gt__(self, other):
        type(self).calls += 1
        raise AssertionError("numeric dimension comparison callback must not run")


class _ExplosiveShape(list[int | str]):
    iterated = False

    def __iter__(self):
        type(self).iterated = True
        raise AssertionError("invalid shape must not be copied or iterated")

    def __deepcopy__(self, _memo):
        raise AssertionError("invalid shape must not be deep-copied")


def test_tensor_dimensions_reject_numeric_subclasses_without_comparison_callbacks():
    dimension = _CoerciveDimension(1)
    errors = TensorSpec("images", [dimension], "float32").validation_errors("input")
    assert any("positive integer" in error for error in errors)
    assert _CoerciveDimension.calls == 0


def test_signature_shape_cap_precedes_defensive_copy():
    _ExplosiveShape.iterated = False
    malformed = TensorSpec("images", _ExplosiveShape([1, 3, 32, 32]), "float32", "NCHW/RGB")
    with pytest.raises(ValueError, match="invalid signature tensor metadata"):
        _signature(malformed)
    assert not _ExplosiveShape.iterated


def test_signature_and_tensor_text_reject_string_subclasses_without_callbacks():
    errors = TensorSpec(_CoerciveString("images"), [1], "float32").validation_errors("input")
    assert any("tensor identifier" in error for error in errors)
    with pytest.raises(ValueError, match="source_classes"):
        VerifiedArtifactSignature(
            artifact_sha256="0" * 64,
            precision="float32",
            embedded_nms=False,
            opset=17,
            source_classes=(_CoerciveString("drone"),),
            inputs=(TensorSpec("images", [1, 3, 32, 32], "float32", "NCHW/RGB"),),
            outputs=(TensorSpec("output0", [1, 5, 21], "float32"),),
            preprocess="bounded RGB",
            postprocess="raw detect",
            failure_behavior="reject malformed inputs",
            evidence="fixture inspection",
        )


def test_signature_tensors_are_bounded_defensive_copies():
    source = TensorSpec("images", [1, 3, 32, 32], "float32", "NCHW/RGB")
    signature = _signature(source)
    source.shape.append(64)
    assert signature.inputs[0].shape == [1, 3, 32, 32]
    signature.inputs[0].shape = _ExplosiveShape([1])
    with pytest.raises(ValueError, match="invalid signature tensor metadata"):
        contract_module._copy_validated_signature_tensors(signature.inputs, signature.outputs)


class _ExplosiveDict(dict[int, str | None]):
    copies = 0

    def copy(self):
        type(self).copies += 1
        raise AssertionError("dictionary subclass copy callback must not run")


def test_class_map_cap_and_exact_type_precede_copy():
    _ExplosiveDict.copies = 0
    with pytest.raises(TypeError, match="dictionary"):
        contract_module._copy_bounded_class_map(_ExplosiveDict({0: "drone"}))
    assert _ExplosiveDict.copies == 0
    with pytest.raises(ValueError, match=f"at most {MAX_CONTRACT_CLASSES}"):
        contract_module._copy_bounded_class_map(
            {index: None for index in range(MAX_CONTRACT_CLASSES + 1)}
        )


def test_ultralytics_formatter_bounds_shape_before_contiguous_allocation(monkeypatch):
    from manwe.common import ultralytics as policy

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda value: value))
    image = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (8000, 4001, 3))
    monkeypatch.setattr(
        np,
        "ascontiguousarray",
        lambda _value: pytest.fail("formatter must bound pixels before contiguous allocation"),
    )
    with pytest.raises(ValueError, match="pixel safety limit"):
        policy._deterministic_format_image(SimpleNamespace(bgr=0), image)


def test_ultralytics_formatter_preserves_valid_pinned_channel_order(monkeypatch):
    from manwe.common import ultralytics as policy

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda value: value))
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    formatted = policy._deterministic_format_image(SimpleNamespace(bgr=0), image)
    assert formatted[:, 0, 0].tolist() == [3, 2, 1]


def test_tiff_decoder_disagreement_rejects_before_frame_stack(tmp_path, monkeypatch):
    import cv2

    from manwe.common import dataset_manifest

    path = tmp_path / "image.tiff"
    Image.new("RGB", (19, 17), (1, 2, 3)).save(path, format="TIFF")
    frame = np.zeros((17, 19, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imdecodemulti", lambda *_args, **_kwargs: (True, [frame, frame]))
    monkeypatch.setattr(
        np,
        "stack",
        lambda *_args, **_kwargs: pytest.fail("ambiguous TIFF frames must not be stacked"),
    )
    with pytest.raises(ValueError, match="must decode.*HxWx3"):
        dataset_manifest._validate_calibration_image(
            path,
            path.relative_to(tmp_path),
            32,
            10_000,
            path.stat(),
        )


def test_single_frame_rgb_tiff_keeps_pinned_backend_semantics(tmp_path):
    from manwe.common import dataset_manifest

    path = tmp_path / "image.tiff"
    Image.new("RGB", (19, 17), (1, 2, 3)).save(path, format="TIFF")
    image = dataset_manifest._validate_calibration_image(
        path,
        path.relative_to(tmp_path),
        32,
        10_000,
        path.stat(),
    )
    assert (image.width, image.height, image.image_format) == (19, 17, "TIFF")
