from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace

import pytest

from manwe.common import dataset_manifest


def _assert_exception_note(error: BaseException, fragment: str) -> None:
    """Assert PEP 678 evidence where the supported interpreter provides it."""
    if not hasattr(error, "add_note"):
        return
    assert any(fragment in note for note in getattr(error, "__notes__", ()))


@pytest.mark.parametrize(
    "released_before_error",
    [False, True],
    ids=("close-failed-before-release", "close-failed-after-release"),
)
def test_relative_directory_walk_never_retries_ambiguous_close(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    released_before_error: bool,
) -> None:
    root = tmp_path / "root"
    (root / "child" / "leaf").mkdir(parents=True)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    real_close = os.close
    real_dup = os.dup
    real_open = os.open
    duplicated_fd = -1
    opened_fds: list[int] = []
    close_attempts: list[int] = []

    def record_dup(fd: int) -> int:
        nonlocal duplicated_fd
        duplicated_fd = real_dup(fd)
        return duplicated_fd

    def record_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_fds.append(fd)
        return fd

    injected = False

    def ambiguous_close(fd: int) -> None:
        nonlocal injected
        close_attempts.append(fd)
        if fd == duplicated_fd and not injected:
            injected = True
            if released_before_error:
                real_close(fd)
            raise OSError("injected ambiguous descriptor close")
        real_close(fd)

    monkeypatch.setattr(dataset_manifest.os, "dup", record_dup)
    monkeypatch.setattr(dataset_manifest.os, "open", record_open)
    monkeypatch.setattr(dataset_manifest.os, "close", ambiguous_close)
    try:
        with pytest.raises(OSError, match="ambiguous descriptor close"):
            dataset_manifest._open_relative_directory_nofollow(
                root_fd,
                pathlib.PurePath("child", "leaf"),
                "dataset root",
            )

        assert duplicated_fd >= 0
        assert close_attempts.count(duplicated_fd) == 1
        assert len(opened_fds) == 1
        with pytest.raises(OSError):
            os.fstat(opened_fds[0])
        if released_before_error:
            with pytest.raises(OSError):
                os.fstat(duplicated_fd)
        else:
            os.fstat(duplicated_fd)
            real_close(duplicated_fd)
    finally:
        for fd in opened_fds:
            try:
                os.fstat(fd)
            except OSError:
                continue
            real_close(fd)
        if duplicated_fd >= 0:
            try:
                os.fstat(duplicated_fd)
            except OSError:
                pass
            else:
                real_close(duplicated_fd)
        real_close(root_fd)


def test_release_resources_preserves_first_failure_and_attempts_every_cleanup() -> None:
    events: list[str] = []
    first = OSError("first cleanup failed")
    second = RuntimeError("second cleanup failed")

    def fail_first() -> None:
        events.append("first")
        raise first

    def fail_second() -> None:
        events.append("second")
        raise second

    with pytest.raises(OSError, match="first cleanup failed") as captured:
        dataset_manifest._release_resources(
            (
                ("first resource cleanup failed", fail_first),
                ("second resource cleanup also failed", fail_second),
            )
        )

    assert captured.value is first
    assert captured.value.__cause__ is second
    assert events == ["first", "second"]
    _assert_exception_note(captured.value, "second resource cleanup also failed")


def test_calibration_context_preserves_body_error_and_releases_every_resource() -> None:
    events: list[str] = []
    errors: dict[str, BaseException] = {
        "loader": OSError("loader cleanup failed"),
        "artifact": RuntimeError("artifact cleanup failed"),
        "root": ValueError("root cleanup failed"),
        "manifest": LookupError("manifest cleanup failed"),
    }

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)
            raise errors[self.name]

    snapshot = dataset_manifest.CalibrationDatasetSnapshot.__new__(
        dataset_manifest.CalibrationDatasetSnapshot
    )
    snapshot._closed = False
    snapshot._loader_snapshot = Resource("loader")
    snapshot._artifact_snapshot = Resource("artifact")
    snapshot._source_root = Resource("root")
    snapshot._source_manifest = Resource("manifest")
    body_error = RuntimeError("calibration operation failed")

    with (
        pytest.raises(RuntimeError, match="calibration operation failed") as captured,
        snapshot,
    ):
        raise body_error

    assert captured.value is body_error
    assert captured.value.__cause__ is errors["loader"]
    assert events == ["loader", "artifact", "root", "manifest"]
    assert snapshot._closed
    _assert_exception_note(captured.value, "calibration dataset snapshot cleanup also failed")
    _assert_exception_note(errors["loader"], "artifact snapshot cleanup")
    _assert_exception_note(errors["loader"], "source dataset root cleanup")
    _assert_exception_note(errors["loader"], "source dataset manifest cleanup")

    snapshot.close()
    assert events == ["loader", "artifact", "root", "manifest"]


def test_manifest_snapshot_constructor_preserves_setup_error_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Resource:
        def __init__(self, name: str, error: BaseException) -> None:
            self.name = name
            self.error = error

        def close(self) -> None:
            events.append(self.name)
            raise self.error

    source_root = Resource("root", OSError("root cleanup failed"))
    source_manifest = Resource("manifest", RuntimeError("manifest cleanup failed"))
    setup_error = MemoryError("payload copy failed")

    def fail_copy(_payload):
        raise setup_error

    monkeypatch.setattr(dataset_manifest.copy, "deepcopy", fail_copy)
    with pytest.raises(MemoryError, match="payload copy failed") as captured:
        dataset_manifest.DatasetManifestSnapshot(
            {"path": "."},
            source_manifest,
            source_root,
        )

    assert captured.value is setup_error
    assert captured.value.__cause__ is source_root.error
    assert events == ["root", "manifest"]
    _assert_exception_note(captured.value, "source dataset root cleanup")
    _assert_exception_note(captured.value, "source dataset manifest cleanup")


def test_calibration_clone_error_remains_primary_when_clone_cleanup_fails() -> None:
    clone_error = RuntimeError("manifest descriptor duplication failed")
    cleanup_error = OSError("cloned root cleanup failed")
    close_attempts = 0

    class RootClone:
        def close(self) -> None:
            nonlocal close_attempts
            close_attempts += 1
            raise cleanup_error

    class RootOwner:
        def clone(self) -> RootClone:
            return RootClone()

    class ManifestOwner:
        def clone(self):
            raise clone_error

    source = SimpleNamespace(_source_root=RootOwner(), _source_manifest=ManifestOwner())
    with pytest.raises(RuntimeError, match="descriptor duplication failed") as captured:
        dataset_manifest.CalibrationDatasetSnapshot(source, 640, None, None)

    assert captured.value is clone_error
    assert captured.value.__cause__ is cleanup_error
    assert close_attempts == 1


def test_snapshot_wrapper_destroys_unreturned_result_if_input_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_cleanup_error = OSError("validated manifest cleanup failed")

    class Validated:
        close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            raise input_cleanup_error

    class Result:
        close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1

    validated = Validated()
    result = Result()
    monkeypatch.setattr(
        dataset_manifest,
        "validate_local_detection_manifest",
        lambda _path: validated,
    )
    monkeypatch.setattr(
        dataset_manifest,
        "CalibrationDatasetSnapshot",
        lambda *_args: result,
    )

    with pytest.raises(OSError, match="validated manifest cleanup failed") as captured:
        dataset_manifest.snapshot_local_calibration_dataset("ignored.yaml")

    assert captured.value is input_cleanup_error
    assert validated.close_attempts == 1
    assert result.close_attempts == 1
