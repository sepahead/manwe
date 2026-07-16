"""Bounded, conditioned triangulation for undistorted image points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .camera import (
    Camera,
    _admit_fixed_array,
    _copy_finite_array,
    _finite_float64_scalar,
    _is_primitive_integer,
    _is_primitive_real_scalar,
    _preflight_fixed_shape,
)

_GEOMETRY_EPS = 1e-12
_MAX_GEOMETRY_MAGNITUDE = 1e12
_MAX_PIXEL_MAGNITUDE = 1e9
_MAX_TRIANGULATION_CAMERAS = 64
_MAX_TRIANGULATION_COORDINATES = 2 * _MAX_TRIANGULATION_CAMERAS
_MAX_COVARIANCE_SOLVES = 4 * _MAX_TRIANGULATION_CAMERAS
_MAX_COVARIANCE_PIXEL_COPY_WORK = _MAX_COVARIANCE_SOLVES * _MAX_TRIANGULATION_CAMERAS


def _copy_vector(
    admitted: np.ndarray,
    name: str,
    *,
    maximum: float,
) -> np.ndarray:
    return _copy_finite_array(admitted, name, maximum=maximum)


def _stable_norm(value: np.ndarray) -> float:
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0.0:
        return 0.0
    norm = scale * float(np.linalg.norm(value / scale))
    if not np.isfinite(norm):
        raise ValueError("geometry norm exceeds the finite numeric range")
    return norm


def _positive_scalar(value: Any, name: str, *, maximum: float | None = None) -> float:
    try:
        result = _finite_float64_scalar(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number > 0") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a finite number > 0")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} exceeds the supported magnitude {maximum:g}")
    return result


def _minimum_angle(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        angle = _finite_float64_scalar(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    lower_valid = angle >= 0.0 if allow_zero else angle > 0.0
    if not lower_valid or angle >= 90.0:
        interval = "[0, 90)" if allow_zero else "(0, 90)"
        raise ValueError(f"{name} must be in the interval {interval}")
    return angle


def _camera_limit(value: Any) -> int:
    if (
        not _is_primitive_integer(value)
        or int(value) < 2
        or int(value) > _MAX_TRIANGULATION_CAMERAS
    ):
        raise ValueError(f"max_cameras must be an integer in [2, {_MAX_TRIANGULATION_CAMERAS}]")
    return int(value)


def _preflight_inputs(
    cameras: Sequence[Camera], pixels: Sequence[np.ndarray], max_cameras: int
) -> list[Camera]:
    if type(cameras) not in (list, tuple) or type(pixels) not in (list, tuple):
        raise ValueError("cameras and pixels must be sequences")
    view_count = len(cameras)
    if view_count < 2:
        raise ValueError("need at least two views to triangulate")
    if view_count != len(pixels):
        raise ValueError("cameras and pixels must be the same length")
    if view_count > max_cameras:
        raise ValueError("triangulation view count exceeds max_cameras")
    if 2 * view_count > _MAX_TRIANGULATION_COORDINATES:
        raise ValueError("triangulation coordinate work exceeds the supported bound")
    if any(not isinstance(camera, Camera) for camera in cameras):
        raise TypeError("cameras must contain only Camera instances")
    for index, pixel in enumerate(pixels):
        _preflight_fixed_shape(pixel, f"pixels[{index}]", (2,))
    return list(cameras)


def _admit_pixels(pixels: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [
        _admit_fixed_array(pixel, f"pixels[{index}]", (2,)) for index, pixel in enumerate(pixels)
    ]


def _copy_pixels(admitted_pixels: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [
        _copy_vector(
            pixel,
            f"pixels[{index}]",
            maximum=_MAX_PIXEL_MAGNITUDE,
        )
        for index, pixel in enumerate(admitted_pixels)
    ]


def _acute_ray_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the acute angle between two ray lines using stable ``atan2``."""

    cross_norm = _stable_norm(np.cross(first, second))
    dot = float(np.clip(first @ second, -1.0, 1.0))
    directed = float(np.arctan2(cross_norm, dot))
    acute = min(directed, np.pi - directed)
    return float(np.degrees(max(0.0, acute)))


def _maximum_parallax_deg(
    cameras: Sequence[Camera], pixels: Sequence[np.ndarray]
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    rays = [camera.backproject_ray(pixel) for camera, pixel in zip(cameras, pixels)]
    maximum = 0.0
    for first in range(len(rays)):
        for second in range(first + 1, len(rays)):
            maximum = max(maximum, _acute_ray_angle_deg(rays[first][1], rays[second][1]))
    return maximum, rays


def triangulate_dlt(
    cameras: Sequence[Camera],
    pixels: Sequence[np.ndarray],
    *,
    require_cheirality: bool = True,
    min_ray_angle_deg: float = 1.0,
    max_range_m: float = 100_000.0,
    max_cameras: int = 16,
) -> np.ndarray:
    """Triangulate one point from ``N >= 2`` already-undistorted pixels.

    DLT is accepted only when at least one camera pair supplies the configured
    acute parallax angle. Rows are globally scaled before SVD, and the result is
    bounded by ``max_range_m`` from every contributing camera.
    """

    max_cameras = _camera_limit(max_cameras)
    camera_values = _preflight_inputs(cameras, pixels, max_cameras)
    min_ray_angle_deg = _minimum_angle(min_ray_angle_deg, "min_ray_angle_deg")
    max_range_m = _positive_scalar(max_range_m, "max_range_m")
    admitted_pixels = _admit_pixels(pixels)
    pixel_values = _copy_pixels(admitted_pixels)

    maximum_parallax, _ = _maximum_parallax_deg(camera_values, pixel_values)
    if maximum_parallax + 1e-12 < min_ray_angle_deg:
        raise ValueError(
            "degenerate triangulation geometry: maximum parallax is below min_ray_angle_deg"
        )

    rows: list[np.ndarray] = []
    for camera, pixel in zip(camera_values, pixel_values):
        projection = camera.P
        u, v = pixel
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    matrix = np.vstack(rows)
    if not np.isfinite(matrix).all():
        raise ValueError("triangulation system exceeds the finite numeric range")
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale <= _GEOMETRY_EPS:
        raise ValueError("degenerate triangulation system")
    matrix /= scale
    try:
        _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("triangulation SVD did not converge") from exc
    if not np.isfinite(singular_values).all():
        raise ValueError("triangulation SVD produced non-finite singular values")
    singular_scale = max(_GEOMETRY_EPS, float(singular_values[0]))
    if singular_values[-2] <= 1e-12 * singular_scale:
        raise ValueError("degenerate triangulation geometry")
    homogeneous = right_vectors[-1]
    if abs(float(homogeneous[3])) < _GEOMETRY_EPS:
        raise ValueError("degenerate triangulation (point at infinity)")
    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all():
        raise ValueError("triangulation produced a non-finite point")
    if np.any(np.abs(point) > _MAX_GEOMETRY_MAGNITUDE):
        raise ValueError("triangulated point exceeds the supported geometry magnitude")
    if require_cheirality and any(not camera.in_front(point) for camera in camera_values):
        raise ValueError("triangulated point is behind at least one camera")
    if any(_stable_norm(point - camera.center) > max_range_m for camera in camera_values):
        raise ValueError("triangulated point exceeds max_range_m")
    return point


def triangulate_midpoint(
    o1: np.ndarray,
    d1: np.ndarray,
    o2: np.ndarray,
    d2: np.ndarray,
    *,
    require_forward: bool = False,
    min_ray_angle_deg: float = 0.0,
    max_range_m: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return closest-ray ``(midpoint, gap)`` with optional conditioning gates.

    With no minimum angle this function retains its diagnostic parallel-line gap
    behavior. Association callers set ``min_ray_angle_deg`` and ``max_range_m``
    so an ill-conditioned or unbounded pair cannot become a candidate edge.
    """

    _preflight_fixed_shape(o1, "o1", (3,))
    _preflight_fixed_shape(d1, "d1", (3,))
    _preflight_fixed_shape(o2, "o2", (3,))
    _preflight_fixed_shape(d2, "d2", (3,))
    min_ray_angle_deg = _minimum_angle(min_ray_angle_deg, "min_ray_angle_deg", allow_zero=True)
    if max_range_m is not None:
        max_range_m = _positive_scalar(max_range_m, "max_range_m")
    admitted_first_origin = _admit_fixed_array(o1, "o1", (3,))
    admitted_first_direction = _admit_fixed_array(d1, "d1", (3,))
    admitted_second_origin = _admit_fixed_array(o2, "o2", (3,))
    admitted_second_direction = _admit_fixed_array(d2, "d2", (3,))
    first_origin = _copy_vector(
        admitted_first_origin,
        "o1",
        maximum=_MAX_GEOMETRY_MAGNITUDE,
    )
    first_direction = _copy_vector(
        admitted_first_direction,
        "d1",
        maximum=_MAX_GEOMETRY_MAGNITUDE,
    )
    second_origin = _copy_vector(
        admitted_second_origin,
        "o2",
        maximum=_MAX_GEOMETRY_MAGNITUDE,
    )
    second_direction = _copy_vector(
        admitted_second_direction,
        "d2",
        maximum=_MAX_GEOMETRY_MAGNITUDE,
    )
    norm1 = _stable_norm(first_direction)
    norm2 = _stable_norm(second_direction)
    if norm1 <= _GEOMETRY_EPS or norm2 <= _GEOMETRY_EPS:
        raise ValueError("ray directions must be non-zero")
    first_direction /= norm1
    second_direction /= norm2
    angle = _acute_ray_angle_deg(first_direction, second_direction)
    if angle + 1e-12 < min_ray_angle_deg:
        raise ValueError("ray parallax is below min_ray_angle_deg")

    offset = first_origin - second_origin
    dot = float(np.clip(first_direction @ second_direction, -1.0, 1.0))
    cross = np.cross(first_direction, second_direction)
    denominator = float(cross @ cross)
    if denominator < 1e-15:
        if require_forward:
            raise ValueError("parallel rays do not define a finite forward intersection")
        perpendicular = offset - (offset @ first_direction) * first_direction
        midpoint = 0.5 * (first_origin + second_origin)
        return midpoint, _stable_norm(perpendicular)

    first_offset = float(first_direction @ offset)
    second_offset = float(second_direction @ offset)
    first_distance = (dot * second_offset - first_offset) / denominator
    second_distance = (second_offset - dot * first_offset) / denominator
    if not np.isfinite(first_distance) or not np.isfinite(second_distance):
        raise ValueError("closest ray intersection exceeds the finite numeric range")
    if require_forward and (first_distance <= 1e-9 or second_distance <= 1e-9):
        raise ValueError("closest ray intersection lies behind a camera")
    if max_range_m is not None and (
        abs(first_distance) > max_range_m or abs(second_distance) > max_range_m
    ):
        raise ValueError("closest ray intersection exceeds max_range_m")
    first_point = first_origin + first_distance * first_direction
    second_point = second_origin + second_distance * second_direction
    if not np.isfinite(first_point).all() or not np.isfinite(second_point).all():
        raise ValueError("closest ray intersection exceeds the finite numeric range")
    return 0.5 * (first_point + second_point), _stable_norm(first_point - second_point)


def reprojection_error(
    cameras: Sequence[Camera],
    pixels: Sequence[np.ndarray],
    point: np.ndarray,
    *,
    max_cameras: int = 16,
) -> float:
    """Return mean pixel reprojection error for a bounded set of views."""

    max_cameras = _camera_limit(max_cameras)
    camera_values = _preflight_inputs(cameras, pixels, max_cameras)
    _preflight_fixed_shape(point, "point", (3,))
    admitted_pixels = _admit_pixels(pixels)
    admitted_point = _admit_fixed_array(point, "point", (3,))
    pixel_values = _copy_pixels(admitted_pixels)
    world_point = _copy_vector(
        admitted_point,
        "point",
        maximum=_MAX_GEOMETRY_MAGNITUDE,
    )
    errors: list[float] = []
    for camera, pixel in zip(camera_values, pixel_values):
        try:
            projected = camera.project(world_point)
        except ValueError:
            return float("inf")
        error = _stable_norm(projected - pixel)
        errors.append(error)
    return float(np.mean(errors))


def triangulation_covariance(
    cameras: Sequence[Camera],
    pixels: Sequence[np.ndarray],
    pixel_stds_px: Sequence[float],
    *,
    calibration_is_exact: bool = False,
    min_ray_angle_deg: float = 1.0,
    max_range_m: float = 100_000.0,
    max_cameras: int = 16,
) -> np.ndarray:
    """Propagate per-view pixel uncertainty through DLT by central differences.

    Each scalar pixel standard deviation is applied isotropically to x and y;
    coordinate and cross-view errors are assumed mutually independent. The
    returned 3x3 covariance is also conditional on every supplied ``K``, ``R``
    and ``t`` being exact. Calibration-parameter uncertainty is not represented
    by independent pixel localization noise: focal-length or pose bias can move
    the 3D solution while retaining zero reprojection residual. Callers must
    therefore set ``calibration_is_exact=True`` explicitly, and may do so only
    for exact synthetic geometry or a separately completed calibration-
    uncertainty propagation. Failure of any perturbation is treated as
    unquantifiable.
    """

    if calibration_is_exact is not True:
        raise ValueError(
            "calibration_is_exact must be explicitly True; triangulation covariance "
            "is conditional on exact K/R/t"
        )
    max_cameras = _camera_limit(max_cameras)
    camera_values = _preflight_inputs(cameras, pixels, max_cameras)
    view_count = len(pixels)
    if type(pixel_stds_px) not in (list, tuple) or len(pixel_stds_px) != view_count:
        raise ValueError("pixel_stds_px must align one-to-one with pixels")
    coordinate_count = 2 * view_count
    if coordinate_count > _MAX_TRIANGULATION_COORDINATES:
        raise ValueError("triangulation covariance work exceeds the supported bound")
    solve_count = 2 * coordinate_count
    pixel_copy_work = solve_count * view_count
    if solve_count > _MAX_COVARIANCE_SOLVES or pixel_copy_work > _MAX_COVARIANCE_PIXEL_COPY_WORK:
        raise ValueError("triangulation covariance perturbation work exceeds the supported bound")
    for index, value in enumerate(pixel_stds_px):
        if not _is_primitive_real_scalar(value):
            raise ValueError(f"pixel_stds_px[{index}] must be a finite number > 0")
    min_ray_angle_deg = _minimum_angle(min_ray_angle_deg, "min_ray_angle_deg")
    max_range_m = _positive_scalar(max_range_m, "max_range_m")
    standard_deviations = [
        _positive_scalar(
            value,
            f"pixel_stds_px[{index}]",
            maximum=_MAX_PIXEL_MAGNITUDE,
        )
        for index, value in enumerate(pixel_stds_px)
    ]
    admitted_pixels = _admit_pixels(pixels)
    pixel_values = _copy_pixels(admitted_pixels)

    jacobian = np.empty((3, coordinate_count), dtype=float)
    for view, standard_deviation in enumerate(standard_deviations):
        step = max(1e-3, min(0.25, 0.05 * standard_deviation))
        for axis in range(2):
            plus = [pixel.copy() for pixel in pixel_values]
            minus = [pixel.copy() for pixel in pixel_values]
            plus[view][axis] += step
            minus[view][axis] -= step
            point_plus = triangulate_dlt(
                camera_values,
                plus,
                min_ray_angle_deg=min_ray_angle_deg,
                max_range_m=max_range_m,
                max_cameras=max_cameras,
            )
            point_minus = triangulate_dlt(
                camera_values,
                minus,
                min_ray_angle_deg=min_ray_angle_deg,
                max_range_m=max_range_m,
                max_cameras=max_cameras,
            )
            jacobian[:, 2 * view + axis] = (point_plus - point_minus) / (2.0 * step)
    if not np.isfinite(jacobian).all():
        raise ValueError("triangulation uncertainty Jacobian is non-finite")

    standard_array = np.asarray(standard_deviations, dtype=np.float64)
    if standard_array.shape != (view_count,):
        raise ValueError("pixel_stds_px produced an invalid internal shape")
    repeated_stds = np.repeat(standard_array, 2)
    if repeated_stds.shape != (coordinate_count,):
        raise ValueError("pixel_stds_px repeat produced an invalid internal shape")
    weighted_jacobian = jacobian * repeated_stds[np.newaxis, :]
    if not np.isfinite(weighted_jacobian).all():
        raise ValueError("weighted triangulation uncertainty Jacobian is non-finite")
    covariance = weighted_jacobian @ weighted_jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("triangulation covariance is non-finite")
    covariance = 0.5 * covariance + 0.5 * covariance.T
    if not np.isfinite(covariance).all():
        raise ValueError("triangulation covariance symmetrization is non-finite")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    normalized = covariance / scale
    try:
        values, vectors = np.linalg.eigh(normalized)
    except np.linalg.LinAlgError as exc:
        raise ValueError("triangulation covariance eigendecomposition did not converge") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError("triangulation covariance eigendecomposition is non-finite")
    if float(values[0]) < -1e-10:
        raise ValueError("triangulation covariance is not positive semidefinite")
    if values[0] < 0.0:
        normalized = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
        normalized = 0.5 * normalized + 0.5 * normalized.T
        covariance = normalized * scale
        if not np.isfinite(covariance).all():
            raise ValueError("triangulation covariance repair is non-finite")
    if float(np.trace(normalized)) <= 0.0:
        raise ValueError("triangulation covariance has no measurable uncertainty")
    return covariance


__all__ = [
    "triangulate_dlt",
    "triangulate_midpoint",
    "reprojection_error",
    "triangulation_covariance",
]
