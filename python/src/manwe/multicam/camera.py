"""Validated pinhole-camera calibration and bounded rig configuration.

The camera convention is ``x_cam = R @ X_world + t``. ``R`` is therefore a
proper world-to-camera rotation and ``K`` maps positive-depth camera points to
top-left-origin image pixels. Distortion is deliberately outside this model:
pixels passed to the multi-camera boundary must already be undistorted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..common.config_io import read_strict_yaml

if TYPE_CHECKING:
    from .tracking import Detection2D, Detection3D

_ROTATION_ATOL = 1e-7
_GEOMETRY_EPS = 1e-12
_MIN_FOV_DEG = 1.0
_MAX_FOV_DEG = 170.0
_MAX_GEOMETRY_MAGNITUDE = 1e12
_MAX_INTRINSIC_MAGNITUDE = 1e9
_MAX_PIXEL_MAGNITUDE = 1e9
_MAX_IMAGE_DIMENSION = 100_000
_MAX_IMAGE_PIXELS = 1_000_000_000
_MAX_RIG_CAMERAS = 64
_MAX_RIG_DETECTIONS = 100_000
_MAX_RIG_CANDIDATE_PAIRS = 10_000_000
_MAX_RIG_HYPOTHESES = 1_000_000
_MAX_RIG_ASSOCIATION_STATES = 10_000_000
_MAX_RIG_YAML_BYTES = 1_000_000
_MAX_TIME_SKEW_S = 60.0
_MAX_SPEED_MPS = 1_000_000.0
_MAX_IDENTITY_BYTES = 256


def _finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.size != int(np.prod(shape)):
        raise ValueError(f"{name} must contain {int(np.prod(shape))} values, got {array.size}")
    array = array.reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.abs(array) > _MAX_GEOMETRY_MAGNITUDE):
        raise ValueError(f"{name} exceeds the supported geometry magnitude")
    return array


def _readonly_array(array: np.ndarray) -> np.ndarray:
    """Copy an array onto an immutable bytes backing store.

    Merely clearing NumPy's ``WRITEABLE`` flag is reversible for owning arrays.
    A bytes-backed view cannot have writeability re-enabled by a consumer.
    """

    contiguous = np.ascontiguousarray(array, dtype=float)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _stable_norm(value: np.ndarray) -> float:
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0.0:
        return 0.0
    norm = scale * float(np.linalg.norm(value / scale))
    if not np.isfinite(norm):
        raise ValueError("geometry norm exceeds the finite numeric range")
    return norm


def _positive_dimension(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    if value > _MAX_IMAGE_DIMENSION:
        raise ValueError(f"{name} exceeds the supported image dimension")
    return value


def _bounded_positive_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _finite_scalar(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0")
    if maximum is not None and abs(value) > maximum:
        raise ValueError(f"{name} exceeds the supported magnitude {maximum:g}")
    return value


def _bounded_angle(value: Any, name: str) -> float:
    angle = _finite_scalar(value, name, positive=True)
    if angle >= 90.0:
        raise ValueError(f"{name} must be in the open interval (0, 90)")
    return angle


def _validate_rotation(rotation: np.ndarray, name: str = "R") -> np.ndarray:
    if np.any(np.abs(rotation) > 1.0 + _ROTATION_ATOL):
        raise ValueError(f"{name} must be orthonormal; entries exceed rotation-matrix bounds")
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=_ROTATION_ATOL):
        raise ValueError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=0.0, atol=_ROTATION_ATOL):
        raise ValueError(f"{name} must be a proper rotation with determinant +1")
    return rotation


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        rendered = ", ".join(repr(key) for key in sorted(unknown, key=repr))
        raise ValueError(f"{name} contains unknown keys: {rendered}")


def _camera_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("name must be a string")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError("name must not contain control characters")
    name = value.strip()
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("name must be valid UTF-8 text") from exc
    if len(encoded) > _MAX_IDENTITY_BYTES:
        raise ValueError(f"name exceeds the {_MAX_IDENTITY_BYTES}-byte limit")
    return name


@dataclass(frozen=True, slots=True)
class Camera:
    """An immutable calibrated pinhole camera.

    Calibration arrays are copied onto read-only backing storage during
    validation. ``width == height == 0`` remains a compatibility representation
    for an unknown image size; otherwise both dimensions must be positive.
    """

    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 world-to-camera rotation
    t: np.ndarray  # 3, world-to-camera translation
    width: int = 0
    height: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        intrinsics = _finite_array(self.K, "K", (3, 3))
        rotation = _validate_rotation(_finite_array(self.R, "R", (3, 3)))
        translation = _finite_array(self.t, "t", (3,))
        width = _positive_dimension(self.width, "width", allow_zero=True)
        height = _positive_dimension(self.height, "height", allow_zero=True)
        if (width == 0) != (height == 0):
            raise ValueError("width and height must either both be zero or both be positive")
        if width and width * height > _MAX_IMAGE_PIXELS:
            raise ValueError("image dimensions exceed the supported pixel count")
        name = _camera_name(self.name)

        # A standard pinhole K has a nonzero homogeneous scale, positive focal
        # lengths, and no projective terms in its final row. Normalize its harmless
        # global scale so later serialization and equality are canonical.
        scale = float(intrinsics[2, 2])
        if abs(scale) <= _GEOMETRY_EPS:
            raise ValueError("K[2,2] must be non-zero")
        intrinsics /= scale
        if np.any(np.abs(intrinsics) > _MAX_INTRINSIC_MAGNITUDE):
            raise ValueError("normalized K exceeds the supported intrinsic magnitude")
        if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], rtol=0.0, atol=_GEOMETRY_EPS):
            raise ValueError("K must have final row [0, 0, 1]")
        if abs(float(intrinsics[1, 0])) > _GEOMETRY_EPS:
            raise ValueError("K must use the standard upper-triangular pinhole form")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("K focal lengths must be positive")
        determinant_sign, log_abs_determinant = np.linalg.slogdet(intrinsics)
        if determinant_sign == 0 or not np.isfinite(log_abs_determinant):
            raise ValueError("K must be invertible")

        if width:
            fov_x = float(np.degrees(2.0 * np.arctan(width / (2.0 * intrinsics[0, 0]))))
            fov_y = float(np.degrees(2.0 * np.arctan(height / (2.0 * intrinsics[1, 1]))))
            if not (_MIN_FOV_DEG <= fov_x <= _MAX_FOV_DEG):
                raise ValueError("K and width imply an unsafe horizontal field of view")
            if not (_MIN_FOV_DEG <= fov_y <= _MAX_FOV_DEG):
                raise ValueError("K and height imply an unsafe vertical field of view")
            if not -width <= intrinsics[0, 2] <= 2 * width:
                raise ValueError("K principal point lies implausibly far outside the image")
            if not -height <= intrinsics[1, 2] <= 2 * height:
                raise ValueError("K principal point lies implausibly far outside the image")
            if abs(float(intrinsics[0, 1])) > max(intrinsics[0, 0], intrinsics[1, 1]):
                raise ValueError("K skew exceeds the supported pinhole range")

        object.__setattr__(self, "K", _readonly_array(intrinsics))
        object.__setattr__(self, "R", _readonly_array(rotation))
        object.__setattr__(self, "t", _readonly_array(translation))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "name", name)

    @property
    def P(self) -> np.ndarray:
        """Return the 3x4 projection matrix ``K [R | t]``."""

        projection = self.K @ np.hstack([self.R, self.t.reshape(3, 1)])
        if not np.isfinite(projection).all():
            raise ValueError("projection matrix exceeds the finite numeric range")
        return projection

    @property
    def center(self) -> np.ndarray:
        """Return camera centre in world coordinates, ``C = -R^T t``."""

        center = -self.R.T @ self.t
        if not np.isfinite(center).all():
            raise ValueError("camera centre exceeds the finite numeric range")
        return center

    def project(self, point_world: np.ndarray) -> np.ndarray:
        """Project a finite, positive-depth world point to pixel ``(u, v)``."""

        point = _finite_array(point_world, "point_world", (3,))
        point_camera = self.R @ point + self.t
        if not np.isfinite(point_camera).all():
            raise ValueError("camera-space point exceeds the finite numeric range")
        if point_camera[2] <= 1e-9:
            raise ValueError("point must be in front of the camera (z_cam > 0)")
        homogeneous = self.K @ point_camera
        pixel = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(pixel).all():
            raise ValueError("projected pixel is non-finite")
        if np.any(np.abs(pixel) > _MAX_PIXEL_MAGNITUDE):
            raise ValueError("projected pixel exceeds the supported magnitude")
        return pixel

    def in_front(self, point_world: np.ndarray) -> bool:
        point = _finite_array(point_world, "point_world", (3,))
        point_camera = self.R @ point + self.t
        return bool(np.isfinite(point_camera).all() and point_camera[2] > 1e-9)

    def backproject_ray(self, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the world-space forward ray ``(origin, unit_direction)``.

        ``pixel`` must already be undistorted; :class:`Detection2D` makes this
        producer acknowledgement explicit at the public association boundary.
        """

        u, v = _finite_array(pixel, "pixel", (2,))
        if max(abs(float(u)), abs(float(v))) > _MAX_PIXEL_MAGNITUDE:
            raise ValueError("pixel exceeds the supported magnitude")
        try:
            direction_camera = np.linalg.solve(self.K, np.array([u, v, 1.0]))
        except np.linalg.LinAlgError as exc:
            raise ValueError("K could not be solved for the supplied pixel") from exc
        direction_world = self.R.T @ direction_camera
        norm = _stable_norm(direction_world)
        if norm <= _GEOMETRY_EPS:
            raise ValueError("pixel does not define a finite camera ray")
        direction = direction_world / norm
        if not np.isfinite(direction).all():
            raise ValueError("pixel does not define a finite camera ray")
        return self.center, direction

    @classmethod
    def from_lookat(
        cls,
        position: np.ndarray,
        target: np.ndarray,
        fov_deg: float = 60.0,
        width: int = 1920,
        height: int = 1080,
        up: np.ndarray | None = None,
        name: str = "",
    ) -> Camera:
        """Build a camera from an eye/target pose and a safe vertical FOV."""

        position = _finite_array(position, "position", (3,))
        target = _finite_array(target, "target", (3,))
        up = _finite_array([0.0, 1.0, 0.0] if up is None else up, "up", (3,))
        width = _positive_dimension(width, "width")
        height = _positive_dimension(height, "height")
        if width * height > _MAX_IMAGE_PIXELS:
            raise ValueError("image dimensions exceed the supported pixel count")
        fov_deg = _finite_scalar(fov_deg, "fov_deg", positive=True)
        if not _MIN_FOV_DEG <= fov_deg <= _MAX_FOV_DEG:
            raise ValueError(
                f"fov_deg must be in the closed interval [{_MIN_FOV_DEG:g}, {_MAX_FOV_DEG:g}]"
            )

        forward = target - position
        forward_norm = _stable_norm(forward)
        if forward_norm <= _GEOMETRY_EPS:
            raise ValueError("position and target must be distinct")
        forward /= forward_norm
        up_norm = _stable_norm(up)
        if up_norm <= _GEOMETRY_EPS:
            raise ValueError("up must be non-zero")
        up /= up_norm
        right = np.cross(forward, up)
        right_norm = _stable_norm(right)
        if right_norm <= _GEOMETRY_EPS:
            raise ValueError("up must not be parallel to the viewing direction")
        right /= right_norm
        true_up = np.cross(right, forward)

        # Camera looks down +z_cam; image v increases down, hence -true_up.
        rotation = np.vstack([right, -true_up, forward])
        translation = -rotation @ position
        focal = (height / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        if not np.isfinite(focal):
            raise ValueError("fov_deg does not produce a finite focal length")
        intrinsics = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
        )
        return cls(
            K=intrinsics,
            R=rotation,
            t=translation,
            width=width,
            height=height,
            name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "width": self.width,
            "height": self.height,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Camera:
        """Load a strict matrix calibration or typed ``position``/``look_at`` pose."""

        if not isinstance(data, Mapping):
            raise ValueError("camera record must be a mapping")
        known_keys = {
            "schema_version",
            "name",
            "K",
            "R",
            "t",
            "position",
            "look_at",
            "lookAt",
            "target",
            "fov_deg",
            "width",
            "height",
            "up",
        }
        _reject_unknown_keys(data, known_keys, "camera record")
        version = data.get("schema_version", 1)
        if (
            isinstance(version, bool)
            or not isinstance(version, (int, np.integer))
            or int(version) != 1
        ):
            raise ValueError(f"unsupported camera schema_version {version!r}")

        matrix_keys = {"K", "R", "t"}
        pose_target_keys = {"look_at", "lookAt", "target"}
        has_matrix = bool(matrix_keys & data.keys())
        has_pose = "position" in data or bool(pose_target_keys & data.keys())
        if has_matrix and has_pose:
            raise ValueError("camera record must use either K/R/t or position/look_at, not both")
        if has_matrix:
            _reject_unknown_keys(
                data,
                {"schema_version", "name", "K", "R", "t", "width", "height"},
                "matrix camera record",
            )
            missing = matrix_keys - data.keys()
            if missing:
                raise ValueError(f"matrix camera record is missing {sorted(missing)}")
            return cls(
                K=data["K"],
                R=data["R"],
                t=data["t"],
                width=data.get("width", 0),
                height=data.get("height", 0),
                name=data.get("name", ""),
            )
        if has_pose:
            _reject_unknown_keys(
                data,
                {
                    "schema_version",
                    "name",
                    "position",
                    "look_at",
                    "lookAt",
                    "target",
                    "fov_deg",
                    "width",
                    "height",
                    "up",
                },
                "pose camera record",
            )
            targets = [key for key in pose_target_keys if key in data]
            if "position" not in data or len(targets) != 1:
                raise ValueError(
                    "pose camera record requires position and exactly one look_at target"
                )
            return cls.from_lookat(
                position=data["position"],
                target=data[targets[0]],
                fov_deg=data.get("fov_deg", 60.0),
                width=data.get("width", 1920),
                height=data.get("height", 1080),
                up=data.get("up"),
                name=data.get("name", ""),
            )
        raise ValueError("camera record must contain either K/R/t or position/look_at")


@dataclass(frozen=True, slots=True)
class CameraRig:
    """A validated multi-camera rig whose gates drive :meth:`correlate`."""

    cameras: tuple[Camera, ...]
    max_speed_mps: float = field(kw_only=True)
    max_ray_gap_m: float = 8.0
    max_reprojection_px: float = 12.0
    max_time_skew_s: float = 0.05
    min_ray_angle_deg: float = 1.0
    max_range_m: float = 100_000.0
    max_cameras: int = 16
    max_detections: int = 4096
    max_candidate_pairs: int = 1_000_000
    max_hypotheses: int = 100_000
    max_association_states: int = 1_000_000

    def __post_init__(self) -> None:
        max_cameras = _bounded_positive_integer(self.max_cameras, "max_cameras", _MAX_RIG_CAMERAS)
        max_detections = _bounded_positive_integer(
            self.max_detections, "max_detections", _MAX_RIG_DETECTIONS
        )
        max_candidate_pairs = _bounded_positive_integer(
            self.max_candidate_pairs,
            "max_candidate_pairs",
            _MAX_RIG_CANDIDATE_PAIRS,
        )
        max_hypotheses = _bounded_positive_integer(
            self.max_hypotheses,
            "max_hypotheses",
            _MAX_RIG_HYPOTHESES,
        )
        max_association_states = _bounded_positive_integer(
            self.max_association_states,
            "max_association_states",
            _MAX_RIG_ASSOCIATION_STATES,
        )
        if not isinstance(self.cameras, (list, tuple)) or len(self.cameras) < 2:
            raise ValueError("rig requires at least two cameras")
        if len(self.cameras) > max_cameras:
            raise ValueError("rig camera count exceeds max_cameras")
        cameras = tuple(self.cameras)
        if any(not isinstance(camera, Camera) for camera in cameras):
            raise TypeError("cameras must contain only Camera instances")
        names = [camera.name for camera in cameras]
        if any(not name for name in names):
            raise ValueError("every rig camera requires a stable, non-empty name")
        if len(set(names)) != len(names):
            raise ValueError("rig camera names must be unique")

        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(
            self,
            "max_speed_mps",
            _finite_scalar(
                self.max_speed_mps,
                "max_speed_mps",
                nonnegative=True,
                maximum=_MAX_SPEED_MPS,
            ),
        )
        object.__setattr__(
            self,
            "max_ray_gap_m",
            _finite_scalar(self.max_ray_gap_m, "max_ray_gap_m", positive=True),
        )
        object.__setattr__(
            self,
            "max_reprojection_px",
            _finite_scalar(
                self.max_reprojection_px,
                "max_reprojection_px",
                positive=True,
                maximum=_MAX_PIXEL_MAGNITUDE,
            ),
        )
        object.__setattr__(
            self,
            "max_time_skew_s",
            _finite_scalar(
                self.max_time_skew_s,
                "max_time_skew_s",
                nonnegative=True,
                maximum=_MAX_TIME_SKEW_S,
            ),
        )
        object.__setattr__(
            self,
            "min_ray_angle_deg",
            _bounded_angle(self.min_ray_angle_deg, "min_ray_angle_deg"),
        )
        object.__setattr__(
            self,
            "max_range_m",
            _finite_scalar(
                self.max_range_m,
                "max_range_m",
                positive=True,
                maximum=_MAX_GEOMETRY_MAGNITUDE,
            ),
        )
        object.__setattr__(self, "max_cameras", max_cameras)
        object.__setattr__(self, "max_detections", max_detections)
        object.__setattr__(self, "max_candidate_pairs", max_candidate_pairs)
        object.__setattr__(self, "max_hypotheses", max_hypotheses)
        object.__setattr__(self, "max_association_states", max_association_states)

    def correlate(self, detections: Sequence[Detection2D]) -> list[Detection3D]:
        """Correlate using this rig's complete geometry, uncertainty, and work gates."""

        from .tracking import correlate_and_triangulate

        return correlate_and_triangulate(
            self.cameras,
            detections,
            max_ray_gap=self.max_ray_gap_m,
            max_reprojection=self.max_reprojection_px,
            max_time_skew=self.max_time_skew_s,
            min_ray_angle_deg=self.min_ray_angle_deg,
            max_range_m=self.max_range_m,
            max_speed_mps=self.max_speed_mps,
            max_cameras=self.max_cameras,
            max_detections=self.max_detections,
            max_candidate_pairs=self.max_candidate_pairs,
            max_hypotheses=self.max_hypotheses,
            max_association_states=self.max_association_states,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CameraRig:
        if not isinstance(data, Mapping):
            raise ValueError("rig record must be a mapping")
        allowed = {
            "schema_version",
            "cameras",
            "max_speed_mps",
            "max_ray_gap_m",
            "max_reprojection_px",
            "max_time_skew_s",
            "min_ray_angle_deg",
            "max_range_m",
            "max_cameras",
            "max_detections",
            "max_candidate_pairs",
            "max_hypotheses",
            "max_association_states",
        }
        _reject_unknown_keys(data, allowed, "rig record")
        version = data.get("schema_version", 1)
        if (
            isinstance(version, bool)
            or not isinstance(version, (int, np.integer))
            or int(version) != 1
        ):
            raise ValueError(f"unsupported rig schema_version {version!r}")
        if "max_speed_mps" not in data:
            raise ValueError("rig record requires max_speed_mps for temporal uncertainty")
        max_cameras = _bounded_positive_integer(
            data.get("max_cameras", 16), "max_cameras", _MAX_RIG_CAMERAS
        )
        records = data.get("cameras")
        if not isinstance(records, list) or len(records) < 2:
            raise ValueError("rig requires a cameras list with at least two entries")
        if len(records) > max_cameras:
            raise ValueError("rig camera count exceeds max_cameras")
        cameras = tuple(Camera.from_dict(record) for record in records)
        return cls(
            cameras=cameras,
            max_speed_mps=_finite_scalar(
                data["max_speed_mps"],
                "max_speed_mps",
                nonnegative=True,
                maximum=_MAX_SPEED_MPS,
            ),
            max_ray_gap_m=_finite_scalar(
                data.get("max_ray_gap_m", 8.0), "max_ray_gap_m", positive=True
            ),
            max_reprojection_px=_finite_scalar(
                data.get("max_reprojection_px", 12.0),
                "max_reprojection_px",
                positive=True,
                maximum=_MAX_PIXEL_MAGNITUDE,
            ),
            max_time_skew_s=_finite_scalar(
                data.get("max_time_skew_s", 0.05),
                "max_time_skew_s",
                nonnegative=True,
                maximum=_MAX_TIME_SKEW_S,
            ),
            min_ray_angle_deg=_bounded_angle(
                data.get("min_ray_angle_deg", 1.0), "min_ray_angle_deg"
            ),
            max_range_m=_finite_scalar(
                data.get("max_range_m", 100_000.0),
                "max_range_m",
                positive=True,
                maximum=_MAX_GEOMETRY_MAGNITUDE,
            ),
            max_cameras=max_cameras,
            max_detections=_bounded_positive_integer(
                data.get("max_detections", 4096),
                "max_detections",
                _MAX_RIG_DETECTIONS,
            ),
            max_candidate_pairs=_bounded_positive_integer(
                data.get("max_candidate_pairs", 1_000_000),
                "max_candidate_pairs",
                _MAX_RIG_CANDIDATE_PAIRS,
            ),
            max_hypotheses=_bounded_positive_integer(
                data.get("max_hypotheses", 100_000),
                "max_hypotheses",
                _MAX_RIG_HYPOTHESES,
            ),
            max_association_states=_bounded_positive_integer(
                data.get("max_association_states", 1_000_000),
                "max_association_states",
                _MAX_RIG_ASSOCIATION_STATES,
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraRig:
        """Load a bounded YAML rig via optional ``manwe-perception[config]`` support."""

        try:
            data = read_strict_yaml(
                Path(path).expanduser().absolute(),
                _MAX_RIG_YAML_BYTES,
                "camera rig YAML",
            )
        except ImportError as exc:  # pragma: no cover - exercised without the optional extra
            raise ImportError(
                "loading YAML rigs requires the locked config extra: "
                "`cd python && uv sync --locked --extra config`"
            ) from exc
        if not isinstance(data, Mapping):
            raise ValueError("camera rig YAML must contain a mapping")
        return cls.from_dict(data)


__all__ = ["Camera", "CameraRig"]
