"""Adversarial tests for bounded multi-camera numeric admission."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import manwe.multicam.camera as camera_module
import manwe.multicam.tracking as tracking_module
import manwe.multicam.triangulation as triangulation_module
from manwe.multicam import (
    Camera,
    Detection2D,
    Detection3D,
    reprojection_error,
    triangulate_dlt,
    triangulate_midpoint,
    triangulation_covariance,
)


class _FloatBomb:
    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("object conversion must not be invoked")


class _FloatSubclassBomb(float):
    calls: int

    def __new__(cls) -> _FloatSubclassBomb:
        instance = super().__new__(cls, 1.0)
        instance.calls = 0
        return instance

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("numeric subclass conversion must not be invoked")


class _IntSubclassBomb(int):
    def __new__(cls) -> _IntSubclassBomb:
        instance = super().__new__(cls, 2)
        instance.calls = 0
        return instance

    calls: int

    def __int__(self) -> int:
        self.calls += 1
        raise AssertionError("integer subclass conversion must not be invoked")


class _NumpyFloatSubclassBomb(np.float64):
    dtype_calls: int = 0

    @property
    def dtype(self) -> np.dtype[np.float64]:
        type(self).dtype_calls += 1
        raise AssertionError("NumPy scalar subclass attributes must not be read")


class _ContainerSubclassBomb(list[Any]):
    def __init__(self, values: list[Any]) -> None:
        super().__init__(values)
        self.calls = 0

    def __len__(self) -> int:
        self.calls += 1
        raise AssertionError("container subclass length hook must not be invoked")

    def __iter__(self) -> Any:
        self.calls += 1
        raise AssertionError("container subclass iteration hook must not be invoked")


class _NdarraySubclassBomb(np.ndarray):
    attribute_calls: int = 0

    def __getattribute__(self, name: str) -> Any:
        if name in {"dtype", "ndim", "shape", "size"}:
            type(self).attribute_calls += 1
            raise AssertionError(f"ndarray subclass {name} hook must not be invoked")
        return super().__getattribute__(name)


def _object_array(shape: tuple[int, ...]) -> tuple[np.ndarray, _FloatBomb]:
    bomb = _FloatBomb()
    value = np.empty(shape, dtype=object)
    value.fill(bomb)
    return value, bomb


def _cameras_and_pixels() -> tuple[list[Camera], list[np.ndarray]]:
    intrinsics = np.diag([100.0, 100.0, 1.0])
    cameras = [
        Camera(intrinsics, np.eye(3), np.zeros(3), name="left"),
        Camera(intrinsics, np.eye(3), np.array([-1.0, 0.0, 0.0]), name="right"),
    ]
    point = np.array([0.2, 0.1, 5.0])
    return cameras, [camera.project(point) for camera in cameras]


def _detection_2d(pixel: Any, **overrides: Any) -> Detection2D:
    arguments = {
        "camera_index": 0,
        "pixel": pixel,
        "pixels_undistorted": True,
        "pixel_std_px": 0.5,
    }
    arguments.update(overrides)
    return Detection2D(**arguments)


def _detection_3d(**overrides: Any) -> Detection3D:
    arguments: dict[str, Any] = {
        "position": np.zeros(3),
        "class_label": None,
        "confidence": 1.0,
        "camera_indices": (0, 1),
        "reprojection_error": 0.0,
        "position_covariance": np.eye(3),
    }
    arguments.update(overrides)
    return Detection3D(**arguments)


def _forbid_copy(*_args: Any, **_kwargs: Any) -> np.ndarray:
    raise AssertionError("float64 copy must not run before aggregate admission")


def _wider_finite_value() -> np.longdouble:
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    if not np.isfinite(value):
        pytest.skip("platform longdouble has no wider finite range than float64")
    return value


@pytest.mark.parametrize(
    ("field", "shape"),
    [("K", (3, 3)), ("R", (3, 3)), ("t", (3,))],
)
def test_camera_constructor_rejects_object_dtype_without_float(
    field: str,
    shape: tuple[int, ...],
) -> None:
    value, bomb = _object_array(shape)
    arguments: dict[str, Any] = {
        "K": np.eye(3),
        "R": np.eye(3),
        "t": np.zeros(3),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="dtype"):
        Camera(**arguments)

    assert bomb.calls == 0


def test_camera_constructor_admits_all_raw_arrays_before_any_float_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation, bomb = _object_array((3,))
    monkeypatch.setattr(camera_module, "_copy_finite_array", _forbid_copy)

    with pytest.raises(ValueError, match="dtype"):
        Camera(np.eye(3), np.eye(3), translation)

    assert bomb.calls == 0


def test_aggregate_preflight_precedes_sequence_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()
    invalid_vector, bomb = _object_array((3,))
    invalid_pixel, pixel_bomb = _object_array((2,))
    invalid_covariance, covariance_bomb = _object_array((3, 3))
    real_asarray = np.asarray
    sequence_materializations: list[type[Any]] = []
    intrinsics_list: Any = np.eye(3).tolist()
    rotation_list: Any = np.eye(3).tolist()
    first_vector: Any = [0.0, 0.0, 0.0]
    forward_vector: Any = [0.0, 0.0, 1.0]
    offset_vector: Any = [1.0, 0.0, 0.0]

    def checked_asarray(value: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        if type(value) in (list, tuple):
            sequence_materializations.append(type(value))
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(camera_module.np, "asarray", checked_asarray)

    with pytest.raises(ValueError, match="dtype"):
        Camera(intrinsics_list, rotation_list, invalid_vector)
    with pytest.raises(ValueError, match="dtype"):
        Camera.from_lookat(forward_vector, first_vector, up=invalid_vector)
    with pytest.raises(ValueError, match="dtype"):
        _detection_3d(
            position=[0.0, 0.0, 0.0],
            position_covariance=invalid_covariance,
        )
    with pytest.raises(ValueError, match="dtype"):
        triangulate_midpoint(
            first_vector,
            forward_vector,
            offset_vector,
            invalid_vector,
        )
    with pytest.raises(ValueError, match="dtype"):
        reprojection_error(
            cameras,
            [pixels[0].tolist(), pixels[1].tolist()],
            invalid_vector,
        )
    with pytest.raises(ValueError, match=r"pixel_stds_px\[1\]"):
        triangulation_covariance(
            cameras,
            [pixels[0].tolist(), pixels[1].tolist()],
            [0.5, bomb],  # type: ignore[list-item]
            calibration_is_exact=True,
        )
    with pytest.raises(ValueError, match="dtype"):
        triangulate_dlt(cameras, [pixels[0].tolist(), invalid_pixel])

    assert sequence_materializations == []
    assert bomb.calls == 0
    assert pixel_bomb.calls == 0
    assert covariance_bomb.calls == 0


@pytest.mark.parametrize("method_name", ["project", "in_front", "backproject_ray"])
def test_camera_methods_reject_object_dtype_without_float(method_name: str) -> None:
    camera = Camera(np.eye(3), np.eye(3), np.zeros(3))
    shape = (2,) if method_name == "backproject_ray" else (3,)
    value, bomb = _object_array(shape)

    with pytest.raises(ValueError, match="dtype"):
        getattr(camera, method_name)(value)

    assert bomb.calls == 0


@pytest.mark.parametrize("field", ["position", "target", "up"])
def test_from_lookat_rejects_object_dtype_without_float(field: str) -> None:
    value, bomb = _object_array((3,))
    arguments: dict[str, Any] = {
        "position": np.array([0.0, 0.0, 1.0]),
        "target": np.zeros(3),
        "up": np.array([0.0, 1.0, 0.0]),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="dtype"):
        Camera.from_lookat(**arguments)

    assert bomb.calls == 0


def test_from_lookat_admits_all_raw_arrays_before_any_float_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    up, bomb = _object_array((3,))
    monkeypatch.setattr(camera_module, "_copy_finite_array", _forbid_copy)

    with pytest.raises(ValueError, match="dtype"):
        Camera.from_lookat(np.array([0, 0, 1]), np.array([0, 0, 0]), up=up)

    assert bomb.calls == 0


def test_camera_boundaries_require_exact_shape_and_real_dtype() -> None:
    with pytest.raises(ValueError, match="dimensions|shape"):
        Camera(np.eye(3).reshape(9), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        Camera(np.eye(3).reshape(1, 9), np.eye(3), np.zeros(3))
    camera = Camera(np.eye(3), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="contain 3 values"):
        camera.project(np.zeros(4))
    with pytest.raises(ValueError, match="dimensions|shape"):
        camera.project(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="dimensions|shape"):
        camera.backproject_ray(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="dtype"):
        camera.in_front(np.ones(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="dtype"):
        camera.project([10**10_000, 0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype"):
        Camera.from_lookat(
            np.ones(3, dtype=np.complex128),
            np.zeros(3),
        )


@pytest.mark.parametrize(
    "value",
    [
        np.ones(3, dtype=np.bool_),
        np.array(["1", "2", "3"]),
        np.array([1, 2, 3], dtype="timedelta64[s]"),
    ],
)
def test_camera_rejects_every_non_real_raw_dtype_kind(value: np.ndarray) -> None:
    camera = Camera(np.eye(3), np.eye(3), np.zeros(3))

    with pytest.raises(ValueError, match="dtype"):
        camera.project(value)


def test_numeric_subclasses_are_rejected_without_conversion_hooks() -> None:
    bomb = _FloatSubclassBomb()
    camera = Camera(np.eye(3), np.eye(3), np.zeros(3))

    with pytest.raises(ValueError, match="primitive"):
        camera.project([bomb, bomb, bomb])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        _detection_2d([0, 0], confidence=bomb)
    with pytest.raises(ValueError, match="finite"):
        triangulate_midpoint(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.ones(3),
            np.array([1.0, 0.0, 0.0]),
            max_range_m=bomb,
        )

    assert bomb.calls == 0


def test_numpy_scalar_subclass_is_rejected_without_instance_attribute_access() -> None:
    bomb = _NumpyFloatSubclassBomb(1.0)
    _NumpyFloatSubclassBomb.dtype_calls = 0

    with pytest.raises(ValueError, match="finite"):
        _detection_2d([0, 0], confidence=bomb)

    assert _NumpyFloatSubclassBomb.dtype_calls == 0


def test_integer_subclasses_are_rejected_without_conversion_hooks() -> None:
    bomb = _IntSubclassBomb()
    cameras, pixels = _cameras_and_pixels()

    with pytest.raises(ValueError, match="width"):
        Camera(np.eye(3), np.eye(3), np.zeros(3), width=bomb, height=1)
    with pytest.raises(ValueError, match="camera_index"):
        _detection_2d([0, 0], camera_index=bomb)
    with pytest.raises(ValueError, match="camera_indices"):
        _detection_3d(camera_indices=(bomb, 1))
    with pytest.raises(ValueError, match="max_cameras"):
        triangulate_dlt(cameras, pixels, max_cameras=bomb)

    assert bomb.calls == 0


def test_container_subclasses_are_rejected_without_bound_or_iteration_hooks() -> None:
    camera = Camera(np.eye(3), np.eye(3), np.zeros(3))
    cameras, pixels = _cameras_and_pixels()
    vector = _ContainerSubclassBomb([0.0, 0.0, 1.0])
    camera_sequence = _ContainerSubclassBomb(cameras)
    index_sequence = _ContainerSubclassBomb([0, 1])
    std_sequence = _ContainerSubclassBomb([0.5, 0.5])

    with pytest.raises(ValueError, match="NumPy array, list, or tuple"):
        camera.project(vector)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequences"):
        triangulate_dlt(camera_sequence, pixels)
    with pytest.raises(ValueError, match="camera_indices"):
        _detection_3d(camera_indices=index_sequence)
    with pytest.raises(ValueError, match="align"):
        triangulation_covariance(
            cameras,
            pixels,
            std_sequence,
            calibration_is_exact=True,
        )

    assert vector.calls == 0
    assert camera_sequence.calls == 0
    assert index_sequence.calls == 0
    assert std_sequence.calls == 0


def test_ndarray_subclass_metadata_hooks_are_not_trusted() -> None:
    camera = Camera(np.eye(3), np.eye(3), np.zeros(3))
    value = np.array([0.0, 0.0, 1.0]).view(_NdarraySubclassBomb)
    _NdarraySubclassBomb.attribute_calls = 0

    assert np.array_equal(camera.project(value), np.zeros(2))
    assert _NdarraySubclassBomb.attribute_calls == 0


def test_camera_array_cast_rejects_finite_longdouble_overflow() -> None:
    value = _wider_finite_value()
    intrinsics = np.eye(3, dtype=np.longdouble)
    intrinsics[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        Camera(intrinsics, np.eye(3), np.zeros(3))


def test_float_copy_rechecks_finiteness_after_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = camera_module._admit_fixed_array(np.ones(3), "value", (3,))
    real_array = np.array

    def overflowing_copy(value: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        result = real_array(value, *args, **kwargs)
        result.flat[0] = np.inf
        return result

    monkeypatch.setattr(camera_module.np, "array", overflowing_copy)
    with pytest.raises(ValueError, match="finite"):
        camera_module._copy_finite_array(admitted, "value", maximum=None)


def test_camera_scalar_cast_rejects_finite_longdouble_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        Camera.from_lookat(
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            fov_deg=_wider_finite_value(),  # type: ignore[arg-type]
        )


def test_multicam_rejects_lossy_integer_narrowing_before_geometry_math() -> None:
    lossy = 2**53 + 1
    with pytest.raises(ValueError, match="exactly representable"):
        Camera(np.eye(3), np.eye(3), np.array([lossy, 0, 0], dtype=np.uint64))
    with pytest.raises(ValueError, match="exactly representable"):
        _detection_3d(position_covariance=np.diag(np.array([lossy, 1, 1], dtype=np.uint64)))
    with pytest.raises(ValueError, match="finite number"):
        triangulate_midpoint(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.ones(3),
            np.array([1.0, 0.0, 0.0]),
            max_range_m=lossy,
        )


def test_multicam_rejects_wider_float_precision_loss_when_available() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble has no wider precision than float64")
    lossy = np.longdouble("0.1")
    assert np.asarray(float(lossy), dtype=np.longdouble) != lossy
    with pytest.raises(ValueError, match="precision"):
        Camera(np.eye(3), np.eye(3), np.array([lossy, 0, 0], dtype=np.longdouble))
    with pytest.raises(ValueError, match="exactly representable"):
        _detection_2d([0, 0], confidence=lossy)


def test_detection_2d_rejects_object_complex_and_wrong_shape_without_float() -> None:
    pixel, bomb = _object_array((2,))
    with pytest.raises(ValueError, match="dtype"):
        _detection_2d(pixel)
    assert bomb.calls == 0

    with pytest.raises(ValueError, match="dtype"):
        _detection_2d(np.ones(2, dtype=np.complex128))
    with pytest.raises(ValueError, match="dimensions|shape"):
        _detection_2d(np.zeros((1, 2)))


@pytest.mark.parametrize(
    ("field", "shape"),
    [("position", (3,)), ("position_covariance", (3, 3))],
)
def test_detection_3d_rejects_object_dtype_without_float(
    field: str,
    shape: tuple[int, ...],
) -> None:
    value, bomb = _object_array(shape)

    with pytest.raises(ValueError, match="dtype"):
        _detection_3d(**{field: value})

    assert bomb.calls == 0


def test_detection_3d_admits_both_arrays_before_any_float_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    covariance, bomb = _object_array((3, 3))
    monkeypatch.setattr(tracking_module, "_copy_finite_array", _forbid_copy)

    with pytest.raises(ValueError, match="dtype"):
        _detection_3d(position_covariance=covariance)

    assert bomb.calls == 0


def test_detection_3d_checks_sequence_bounds_before_array_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking_module, "_copy_finite_array", _forbid_copy)

    with pytest.raises(ValueError, match="at most 64"):
        _detection_3d(camera_indices=tuple(range(65)))
    with pytest.raises(ValueError, match="camera_ids must contain at most 64"):
        _detection_3d(camera_ids=("camera",) * 65)


def test_detection_3d_requires_exact_real_array_shapes() -> None:
    with pytest.raises(ValueError, match="dimensions|shape"):
        _detection_3d(position=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="dimensions|shape"):
        _detection_3d(position_covariance=np.eye(3).reshape(9))
    with pytest.raises(ValueError, match="dtype"):
        _detection_3d(position=np.ones(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="dtype"):
        _detection_3d(position_covariance=np.eye(3, dtype=np.complex128))


def test_detection_3d_covariance_arithmetic_stays_finite_at_float64_extremes() -> None:
    diagonal = _detection_3d(position_covariance=np.eye(3) * 1e308)
    assert np.isfinite(diagonal.position_covariance).all()
    assert np.array_equal(diagonal.position_covariance, np.eye(3) * 1e308)

    with pytest.raises(ValueError, match="spectral magnitude"):
        _detection_3d(position_covariance=np.full((3, 3), np.finfo(np.float64).max))


@pytest.mark.parametrize("operation", ["detection", "triangulation"])
def test_covariance_eigendecomposition_failures_are_normalized(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()

    def fail_eigh(_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise np.linalg.LinAlgError("injected failure")

    monkeypatch.setattr(tracking_module.np.linalg, "eigh", fail_eigh)
    with pytest.raises(ValueError, match="eigendecomposition did not converge"):
        if operation == "detection":
            _detection_3d()
        else:
            triangulation_covariance(
                cameras,
                pixels,
                [0.5, 0.5],
                calibration_is_exact=True,
            )


def test_tracking_scalar_cast_rejects_finite_longdouble_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        _detection_2d([0, 0], confidence=_wider_finite_value())


def test_scalar_integer_overflow_is_normalized_to_value_error() -> None:
    huge_integer = 10**10_000

    with pytest.raises(ValueError, match="finite"):
        Camera.from_lookat(
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            fov_deg=huge_integer,
        )
    with pytest.raises(ValueError, match="finite"):
        _detection_2d([0, 0], confidence=huge_integer)
    with pytest.raises(ValueError, match="finite"):
        triangulate_midpoint(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.ones(3),
            np.array([1.0, 0.0, 0.0]),
            max_range_m=huge_integer,
        )


@pytest.mark.parametrize("operation", ["dlt", "midpoint", "reprojection", "covariance"])
def test_triangulation_arrays_reject_object_dtype_without_float(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()
    shape = (3,) if operation in {"midpoint", "reprojection"} else (2,)
    invalid, bomb = _object_array(shape)
    monkeypatch.setattr(triangulation_module, "_copy_finite_array", _forbid_copy)

    with pytest.raises(ValueError, match="dtype"):
        if operation == "dlt":
            triangulate_dlt(cameras, [pixels[0], invalid])
        elif operation == "midpoint":
            triangulate_midpoint(
                np.zeros(3),
                np.array([0.0, 0.0, 1.0]),
                np.ones(3),
                invalid,
            )
        elif operation == "reprojection":
            reprojection_error(cameras, pixels, invalid)
        else:
            triangulation_covariance(
                cameras,
                [pixels[0], invalid],
                [0.5, 0.5],
                calibration_is_exact=True,
            )

    assert bomb.calls == 0


@pytest.mark.parametrize("operation", ["dlt", "midpoint", "reprojection", "covariance"])
def test_triangulation_arrays_require_exact_shape_and_real_dtype(operation: str) -> None:
    cameras, pixels = _cameras_and_pixels()

    with pytest.raises(ValueError, match="dimensions|shape|dtype"):
        if operation == "dlt":
            triangulate_dlt(cameras, [pixels[0], np.zeros((1, 2))])
        elif operation == "midpoint":
            triangulate_midpoint(
                np.zeros(3),
                np.array([0.0, 0.0, 1.0]),
                np.ones(3),
                np.ones(3, dtype=np.complex128),
            )
        elif operation == "reprojection":
            reprojection_error(cameras, pixels, np.zeros((3, 1)))
        else:
            triangulation_covariance(
                cameras,
                [pixels[0], np.ones(2, dtype=np.complex128)],
                [0.5, 0.5],
                calibration_is_exact=True,
            )


def test_triangulation_view_bound_precedes_pixel_admission() -> None:
    cameras, _ = _cameras_and_pixels()
    invalid, bomb = _object_array((2,))

    with pytest.raises(ValueError, match="view count"):
        triangulate_dlt(
            [cameras[0]] * 65,
            [invalid] * 65,
            max_cameras=64,
        )

    assert bomb.calls == 0


def test_triangulation_work_bound_precedes_pixel_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, _ = _cameras_and_pixels()
    invalid, bomb = _object_array((2,))
    monkeypatch.setattr(triangulation_module, "_MAX_TRIANGULATION_COORDINATES", 2)

    with pytest.raises(ValueError, match="coordinate work"):
        triangulate_dlt(cameras, [invalid, invalid])

    assert bomb.calls == 0


def test_covariance_perturbation_work_bound_precedes_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()
    monkeypatch.setattr(triangulation_module, "_MAX_COVARIANCE_PIXEL_COPY_WORK", 15)
    monkeypatch.setattr(triangulation_module, "_copy_finite_array", _forbid_copy)

    with pytest.raises(ValueError, match="perturbation work"):
        triangulation_covariance(
            cameras,
            pixels,
            [0.5, 0.5],
            calibration_is_exact=True,
        )


def test_covariance_admits_all_standard_deviations_before_copy_or_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()
    bomb = _FloatBomb()
    monkeypatch.setattr(triangulation_module, "_copy_finite_array", _forbid_copy)

    def forbid_repeat(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("repeat must not run before standard-deviation admission")

    monkeypatch.setattr(triangulation_module.np, "repeat", forbid_repeat)
    with pytest.raises(ValueError, match=r"pixel_stds_px\[1\]"):
        triangulation_covariance(
            cameras,
            pixels,
            [0.5, bomb],  # type: ignore[list-item]
            calibration_is_exact=True,
        )

    assert bomb.calls == 0


def test_covariance_rejects_complex_standard_deviation_before_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()

    def forbid_repeat(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("repeat must not run for rejected standard deviations")

    monkeypatch.setattr(triangulation_module.np, "repeat", forbid_repeat)
    with pytest.raises(ValueError, match=r"pixel_stds_px\[1\]"):
        triangulation_covariance(
            cameras,
            pixels,
            [0.5, 1.0 + 0.0j],  # type: ignore[list-item]
            calibration_is_exact=True,
        )


def test_covariance_rejects_unbounded_standard_deviation_before_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()

    def forbid_repeat(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("repeat must not run for unbounded standard deviations")

    monkeypatch.setattr(triangulation_module.np, "repeat", forbid_repeat)
    with pytest.raises(ValueError, match="supported magnitude"):
        triangulation_covariance(
            cameras,
            pixels,
            [0.5, 3e154],
            calibration_is_exact=True,
        )


def test_covariance_scalar_overflow_is_rejected_before_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()

    def forbid_repeat(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("repeat must not run for non-finite float64 values")

    monkeypatch.setattr(triangulation_module.np, "repeat", forbid_repeat)
    with pytest.raises(ValueError, match="finite"):
        triangulation_covariance(
            cameras,
            pixels,
            [0.5, _wider_finite_value()],  # type: ignore[list-item]
            calibration_is_exact=True,
        )


def test_covariance_repeat_has_exact_bounded_coordinate_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras, pixels = _cameras_and_pixels()
    real_repeat: Callable[..., np.ndarray] = np.repeat
    calls: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []

    def checked_repeat(value: np.ndarray, repeats: int) -> np.ndarray:
        result = real_repeat(value, repeats)
        calls.append((value.shape, repeats, result.shape))
        return result

    monkeypatch.setattr(triangulation_module.np, "repeat", checked_repeat)
    covariance = triangulation_covariance(
        cameras,
        pixels,
        [0.5, 0.75],
        calibration_is_exact=True,
    )

    assert covariance.shape == (3, 3)
    assert calls == [((2,), 2, (4,))]


def test_triangulation_scalar_cast_rejects_finite_longdouble_overflow() -> None:
    cameras, pixels = _cameras_and_pixels()

    with pytest.raises(ValueError, match="finite"):
        triangulate_dlt(
            cameras,
            pixels,
            max_range_m=_wider_finite_value(),  # type: ignore[arg-type]
        )
