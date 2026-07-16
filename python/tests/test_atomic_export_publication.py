"""Atomic, no-replace export publication boundary tests."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from pathlib import Path

import pytest

from manwe.common.artifacts import sha256_artifact
from manwe.export import backends

_STAGE_PREFIX = ".manwe-export-stage-"


class _FakeRename:
    def __init__(self, result: int = 0, error_number: int = 0) -> None:
        self.result = result
        self.error_number = error_number
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: list[type[object]] | None = None
        self.restype: type[object] | None = None

    def __call__(self, *arguments: object) -> int:
        self.calls.append(arguments)
        ctypes.set_errno(self.error_number)
        return self.result


def _stage_entries(parent: Path) -> list[Path]:
    return sorted(
        (entry for entry in parent.iterdir() if entry.name.startswith(_STAGE_PREFIX)),
        key=lambda entry: entry.name,
    )


def _notes(error: BaseException) -> tuple[str, ...]:
    return tuple(getattr(error, "__notes__", ()))


def test_private_stage_name_uses_a_fixed_192_bit_random_token(monkeypatch):
    requested_bytes: list[int] = []

    def token_hex(byte_count):
        requested_bytes.append(byte_count)
        return "ab" * byte_count

    monkeypatch.setattr(backends.secrets, "token_hex", token_hex)
    assert backends._new_private_stage_name() == f"{_STAGE_PREFIX}{'ab' * 24}"
    assert requested_bytes == [24]


def test_occupied_destination_remains_primary_when_parent_close_fails(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "occupied.onnx"
    destination.write_bytes(b"foreign")
    opened_parent_fd = -1
    close_calls = 0
    real_open_parent = backends.open_directory_nofollow
    real_close = os.close

    def capture_parent_fd(path, name):
        nonlocal opened_parent_fd
        opened_parent_fd = real_open_parent(path, name)
        return opened_parent_fd

    def close_parent_once(fd):
        nonlocal close_calls
        if fd == opened_parent_fd:
            close_calls += 1
            real_close(fd)
            raise OSError(errno.EIO, "injected parent close failure")
        real_close(fd)

    monkeypatch.setattr(backends, "open_directory_nofollow", capture_parent_fd)
    monkeypatch.setattr(backends.os, "close", close_parent_once)

    with pytest.raises(FileExistsError, match="already occupied") as captured:
        backends._prepare_destination(str(destination), {".onnx"})

    assert close_calls == 1
    assert isinstance(captured.value.__cause__, OSError)
    assert "injected parent close failure" in str(captured.value.__cause__)
    if hasattr(captured.value, "add_note"):
        assert any("output parent cleanup failed" in note for note in _notes(captured.value))


def test_linux_libc_wrapper_uses_renameat2_noreplace_once(monkeypatch):
    rename = _FakeRename()

    class FakeLibc:
        renameat2 = rename

    loads: list[tuple[object, bool]] = []

    def load_libc(name, *, use_errno):
        loads.append((name, use_errno))
        return FakeLibc()

    monkeypatch.setattr(backends.ctypes, "CDLL", load_libc)
    backends._linux_rename_noreplace_at(37, "private-stage", "final.onnx")

    assert loads == [(None, True)]
    assert rename.calls == [(37, b"private-stage", 37, b"final.onnx", 1)]
    assert rename.argtypes == [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    assert rename.restype is ctypes.c_int


def test_linux_libc_wrapper_preserves_eexist_and_does_not_retry(monkeypatch):
    rename = _FakeRename(-1, errno.EEXIST)

    class FakeLibc:
        renameat2 = rename

    monkeypatch.setattr(backends.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    with pytest.raises(FileExistsError) as captured:
        backends._linux_rename_noreplace_at(11, "stage", "occupied")

    assert captured.value.errno == errno.EEXIST
    assert len(rename.calls) == 1


def test_linux_libc_wrapper_fails_closed_on_unsupported_filesystem(monkeypatch):
    rename = _FakeRename(-1, errno.EINVAL)

    class FakeLibc:
        renameat2 = rename

    monkeypatch.setattr(backends.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    with pytest.raises(RuntimeError, match="does not support.*no-replace") as captured:
        backends._linux_rename_noreplace_at(13, "stage", "final")

    assert isinstance(captured.value.__cause__, OSError)
    assert captured.value.__cause__.errno == errno.EINVAL
    assert len(rename.calls) == 1


def test_darwin_parent_with_extended_acl_is_rejected(monkeypatch, tmp_path):
    class FakeFunction:
        def __init__(self, result, error_number=0):
            self.result = result
            self.error_number = error_number
            self.calls = []
            self.argtypes = None
            self.restype = None

        def __call__(self, *arguments):
            self.calls.append(arguments)
            ctypes.set_errno(self.error_number)
            return self.result

    acl_get_fd_np = FakeFunction(0x1234)
    acl_free = FakeFunction(0)

    class FakeLibc:
        def __init__(self):
            self.acl_get_fd_np = acl_get_fd_np
            self.acl_free = acl_free

    fake_libc = FakeLibc()
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(backends.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc)
    parent = tmp_path.resolve()
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        with pytest.raises(PermissionError, match="extended ACL.*mode-bit trust proof"):
            backends._assert_publication_parent_trust(
                parent_fd,
                os.fstat(parent_fd),
                parent,
            )
    finally:
        os.close(parent_fd)

    assert acl_get_fd_np.calls == [(parent_fd, 0x100)]
    assert acl_free.calls == [(0x1234,)]


def test_darwin_parent_without_extended_acl_uses_mode_bit_proof(monkeypatch, tmp_path):
    class FakeFunction:
        def __init__(self, result, error_number=0):
            self.result = result
            self.error_number = error_number
            self.argtypes = None
            self.restype = None

        def __call__(self, *_arguments):
            ctypes.set_errno(self.error_number)
            return self.result

    class FakeLibc:
        def __init__(self):
            self.acl_get_fd_np = FakeFunction(None, errno.ENOENT)
            self.acl_free = FakeFunction(0)

    fake_libc = FakeLibc()
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(backends.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc)
    parent = tmp_path.resolve()
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        backends._assert_publication_parent_trust(
            parent_fd,
            os.fstat(parent_fd),
            parent,
        )
    finally:
        os.close(parent_fd)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires the Darwin libc primitive")
def test_darwin_renameatx_np_rename_excl_is_actually_exclusive(tmp_path):
    parent = tmp_path.resolve()
    stage = parent / "stage"
    final = parent / "final"
    stage.write_bytes(b"first")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        backends._darwin_rename_noreplace_at(parent_fd, stage.name, final.name)
        assert not stage.exists()
        assert final.read_bytes() == b"first"

        stage.write_bytes(b"second")
        with pytest.raises(FileExistsError) as captured:
            backends._darwin_rename_noreplace_at(parent_fd, stage.name, final.name)
        assert captured.value.errno == errno.EEXIST
        assert stage.read_bytes() == b"second"
        assert final.read_bytes() == b"first"
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("directory_artifact", [False, True])
def test_publish_populates_sibling_stage_before_atomic_rename(tmp_path, directory_artifact):
    parent = tmp_path.resolve()
    if directory_artifact:
        source = parent / "source.mlpackage"
        source.mkdir()
        (source / "Manifest.json").write_text("{}", encoding="utf-8")
        destination = parent / "published.mlpackage"
        expected_mode = 0o755
    else:
        source = parent / "source.onnx"
        source.write_bytes(b"trusted")
        destination = parent / "published.onnx"
        expected_mode = 0o644

    digest = sha256_artifact(source)
    backends._publish_exclusive(source, destination, digest)

    assert sha256_artifact(destination) == digest
    assert stat.S_IMODE(destination.stat().st_mode) == expected_mode
    assert _stage_entries(parent) == []


def test_stage_name_collision_preserves_foreign_entry_and_uses_fresh_name(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    occupied_stage = parent / f"{_STAGE_PREFIX}{'1' * 48}"
    occupied_stage.write_bytes(b"foreign")
    names = iter((occupied_stage.name, f"{_STAGE_PREFIX}{'2' * 48}"))

    monkeypatch.setattr(backends, "_new_private_stage_name", lambda: next(names))
    backends._publish_exclusive(source, destination, sha256_artifact(source))

    assert occupied_stage.read_bytes() == b"foreign"
    assert destination.read_bytes() == b"trusted"
    assert _stage_entries(parent) == [occupied_stage]


def test_stage_stays_private_until_rename_then_final_mode_is_applied(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)
    real_stage_hash = backends.sha256_artifact_at
    real_rename = backends._rename_noreplace_at
    real_finalize = backends._finalize_published_mode
    events: list[str] = []

    def inspect_then_hash(parent_fd, stage_name):
        assert events == []
        stage_metadata = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        assert stat.S_IMODE(stage_metadata.st_mode) == 0o600
        events.append("digest")
        return real_stage_hash(parent_fd, stage_name)

    def inspect_then_rename(parent_fd, stage_name, final_name):
        assert events == ["digest"]
        stage_metadata = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        assert stat.S_IMODE(stage_metadata.st_mode) == 0o600
        assert not destination.exists()
        events.append("rename")
        real_rename(parent_fd, stage_name, final_name)

    def inspect_then_finalize(parent_fd, name, identity, *, is_directory):
        assert events == ["digest", "rename"]
        assert name == destination.name
        final_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        assert (final_metadata.st_dev, final_metadata.st_ino) == identity
        assert stat.S_IMODE(final_metadata.st_mode) == 0o600
        events.append("finalize")
        real_finalize(
            parent_fd,
            name,
            identity,
            is_directory=is_directory,
        )

    monkeypatch.setattr(backends, "sha256_artifact_at", inspect_then_hash)
    monkeypatch.setattr(backends, "_rename_noreplace_at", inspect_then_rename)
    monkeypatch.setattr(backends, "_finalize_published_mode", inspect_then_finalize)
    backends._publish_exclusive(source, destination, digest)

    assert events == ["digest", "rename", "finalize"]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644


def test_nonsticky_shared_parent_is_rejected_before_staging(tmp_path):
    parent = tmp_path.resolve() / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    source = tmp_path.resolve() / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"

    with pytest.raises(PermissionError, match="group/other-writable.*sticky"):
        backends._publish_exclusive(source, destination, sha256_artifact(source))

    assert not destination.exists()
    assert _stage_entries(parent) == []


def test_sticky_shared_parent_owned_by_effective_user_is_accepted(tmp_path):
    parent = tmp_path.resolve() / "shared"
    parent.mkdir()
    parent.chmod(0o1777)
    source = tmp_path.resolve() / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)

    backends._publish_exclusive(source, destination, digest)

    assert sha256_artifact(destination) == digest
    assert _stage_entries(parent) == []


def test_parent_that_becomes_untrusted_after_preparation_is_rejected(tmp_path):
    parent = tmp_path.resolve() / "output"
    parent.mkdir(mode=0o700)
    source = tmp_path.resolve() / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    prepared = backends._prepare_destination(str(destination), {".onnx"})
    parent.chmod(0o777)
    try:
        with pytest.raises(PermissionError, match="group/other-writable.*sticky"):
            backends._publish_exclusive(source, prepared, sha256_artifact(source))
    finally:
        prepared.close()

    assert not destination.exists()
    assert _stage_entries(parent) == []


def test_racing_final_entry_is_never_overwritten(tmp_path, monkeypatch, caplog):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)
    real_rename = backends._rename_noreplace_at

    def occupy_final_then_rename(parent_fd, stage_name, final_name):
        foreign_fd = os.open(
            final_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(foreign_fd, "wb") as foreign:
            foreign.write(b"foreign")
            foreign.flush()
            os.fsync(foreign.fileno())
        real_rename(parent_fd, stage_name, final_name)

    monkeypatch.setattr(backends, "_rename_noreplace_at", occupy_final_then_rename)
    with pytest.raises(FileExistsError) as captured:
        backends._publish_exclusive(source, destination, digest)

    stages = _stage_entries(parent)
    assert destination.read_bytes() == b"foreign"
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"trusted"
    assert stat.S_IMODE(stages[0].stat().st_mode) == 0o600
    assert "No automatic deletion was attempted" in caplog.text
    if hasattr(captured.value, "add_note"):
        assert any("No automatic deletion was attempted" in note for note in _notes(captured.value))


def test_unsupported_filesystem_preserves_verified_stage(tmp_path, monkeypatch, caplog):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)

    def reject_rename(_parent_fd, _stage_name, _final_name):
        cause = OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        raise RuntimeError(
            "the operating system or destination filesystem does not support "
            "the required atomic no-replace rename operation"
        ) from cause

    monkeypatch.setattr(backends, "_rename_noreplace_at", reject_rename)
    with pytest.raises(RuntimeError, match="does not support.*no-replace") as captured:
        backends._publish_exclusive(source, destination, digest)

    stages = _stage_entries(parent)
    assert not destination.exists()
    assert len(stages) == 1
    assert sha256_artifact(stages[0]) == digest
    assert stat.S_IMODE(stages[0].stat().st_mode) == 0o600
    assert "outcome is indeterminate" in caplog.text
    if hasattr(captured.value, "add_note"):
        assert any("outcome is indeterminate" in note for note in _notes(captured.value))


def test_reported_rename_failure_can_have_committed_on_network_filesystem(
    tmp_path,
    monkeypatch,
    caplog,
):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)

    def commit_then_report_failure(parent_fd, stage_name, final_name):
        os.rename(
            stage_name,
            final_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        raise OSError(errno.EIO, "simulated lost NFS rename reply")

    monkeypatch.setattr(backends, "_rename_noreplace_at", commit_then_report_failure)
    with pytest.raises(OSError, match="lost NFS rename reply") as captured:
        backends._publish_exclusive(source, destination, digest)

    assert _stage_entries(parent) == []
    assert sha256_artifact(destination) == digest
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert "outcome is indeterminate" in caplog.text
    assert "No automatic deletion was attempted" in caplog.text
    if hasattr(captured.value, "add_note"):
        notes = _notes(captured.value)
        assert any("outcome is indeterminate" in note for note in notes)
        assert any("No automatic deletion was attempted" in note for note in notes)


def test_post_rename_permission_failure_preserves_private_final(
    tmp_path,
    monkeypatch,
    caplog,
):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)

    def fail_permission_finalization(*_args, **_kwargs):
        raise OSError(errno.EIO, "injected chmod failure")

    monkeypatch.setattr(backends, "_finalize_published_mode", fail_permission_finalization)
    with pytest.raises(OSError, match="injected chmod failure") as captured:
        backends._publish_exclusive(source, destination, digest)

    assert _stage_entries(parent) == []
    assert sha256_artifact(destination) == digest
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert "exclusive rename returned success" in caplog.text
    if hasattr(captured.value, "add_note"):
        assert any("exclusive rename returned success" in note for note in _notes(captured.value))


def test_post_publication_snapshot_cleanup_failure_reports_completed_state(
    tmp_path,
    monkeypatch,
    caplog,
):
    parent = tmp_path.resolve()
    source = parent / "source.onnx"
    source.write_bytes(b"trusted")
    destination = parent / "published.onnx"
    digest = sha256_artifact(source)
    real_snapshot = backends.ArtifactSnapshot

    class CleanupFailingSnapshot:
        def __init__(self, *args, **kwargs):
            self._snapshot = real_snapshot(*args, **kwargs)

        def __enter__(self):
            return self._snapshot.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            self._snapshot.__exit__(exc_type, exc, traceback)
            raise OSError(errno.EIO, "injected snapshot cleanup failure")

    monkeypatch.setattr(backends, "ArtifactSnapshot", CleanupFailingSnapshot)
    with pytest.raises(OSError, match="injected snapshot cleanup failure") as captured:
        backends._publish_exclusive(source, destination, digest)

    assert _stage_entries(parent) == []
    assert sha256_artifact(destination) == digest
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert "publication completed" in caplog.text
    if hasattr(captured.value, "add_note"):
        assert any("publication completed" in note for note in _notes(captured.value))


def test_copy_error_remains_primary_when_stage_descriptor_close_also_fails(
    tmp_path,
    monkeypatch,
    caplog,
):
    parent = tmp_path.resolve()
    source = parent / "source.mlpackage"
    source.mkdir()
    (source / "Manifest.json").write_text("{}", encoding="utf-8")
    destination = parent / "published.mlpackage"
    digest = sha256_artifact(source)
    stage_descriptor = -1
    injected_close_count = 0
    real_close = os.close

    def fail_copy(_source, destination_fd):
        nonlocal stage_descriptor
        stage_descriptor = destination_fd
        raise ValueError("injected copy failure")

    def fail_stage_close_once(fd):
        nonlocal injected_close_count
        if fd == stage_descriptor and injected_close_count == 0:
            injected_close_count += 1
            real_close(fd)
            raise OSError(errno.EIO, "injected stage close failure")
        if fd == stage_descriptor:
            try:
                os.fstat(fd)
            except OSError as error:
                if error.errno == errno.EBADF:
                    injected_close_count += 1
        real_close(fd)

    monkeypatch.setattr(backends, "_copy_directory_fd_relative", fail_copy)
    monkeypatch.setattr(backends.os, "close", fail_stage_close_once)
    with pytest.raises(ValueError, match="injected copy failure") as captured:
        backends._publish_exclusive(source, destination, digest)

    assert injected_close_count == 1
    assert isinstance(captured.value.__cause__, OSError)
    assert "injected stage close failure" in str(captured.value.__cause__)
    if hasattr(captured.value, "add_note"):
        assert any(
            "export stage descriptor cleanup failed" in note for note in _notes(captured.value)
        )
    assert "No automatic deletion was attempted" in caplog.text
    stages = _stage_entries(parent)
    assert not destination.exists()
    assert len(stages) == 1
    assert stages[0].is_dir()
    assert stat.S_IMODE(stages[0].stat().st_mode) == 0o700
