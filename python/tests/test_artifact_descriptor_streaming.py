"""Resource and trust-boundary tests for descriptor-anchored artifact trees."""

from __future__ import annotations

import gc
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from manwe.common import artifacts

_T = TypeVar("_T")


def _open_directory(path: pathlib.Path) -> int:
    return os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )


def _run_after_concurrent_chdir(
    monkeypatch: pytest.MonkeyPatch,
    start: pathlib.Path,
    replacement: pathlib.Path,
    operation: Callable[[], _T],
) -> _T:
    original_cwd = pathlib.Path.cwd()
    anchored = threading.Event()
    switched = threading.Event()
    worker_errors: list[BaseException] = []
    real_anchor = artifacts._absolute_artifact_path
    first_call = True

    def anchor_then_wait(path):
        nonlocal first_call
        value = real_anchor(path)
        if first_call:
            first_call = False
            anchored.set()
            if not switched.wait(timeout=5):
                raise RuntimeError("cwd replacement thread did not run")
        return value

    def replace_cwd():
        try:
            if not anchored.wait(timeout=5):
                raise RuntimeError("artifact operation did not anchor its path")
            os.chdir(replacement)
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            switched.set()

    os.chdir(start)
    worker = threading.Thread(target=replace_cwd)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(artifacts, "_absolute_artifact_path", anchor_then_wait)
            worker.start()
            result = operation()
    finally:
        switched.set()
        worker.join(timeout=5)
        os.chdir(original_cwd)
    assert not worker.is_alive()
    assert not worker_errors
    return result


def test_descriptor_inventory_streams_until_entry_limit(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    for index in range(10):
        (root / f"{index:02d}.bin").write_bytes(b"x")

    real_scandir = artifacts.os.scandir
    yielded = 0

    class CountingScandir:
        def __init__(self, path):
            self._iterator = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self._iterator.close()

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal yielded
            value = next(self._iterator)
            yielded += 1
            return value

    monkeypatch.setattr(artifacts.os, "scandir", CountingScandir)
    root_fd = _open_directory(root)
    try:
        with pytest.raises(ValueError, match="2-entry safety limit"):
            artifacts._descriptor_tree_entries(
                root_fd,
                display="test bundle",
                max_entries=2,
            )
    finally:
        os.close(root_fd)
    assert yielded == 3


def test_descriptor_inventory_bounds_aggregate_relative_path_bytes(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "abc").write_bytes(b"a")
    (root / "def").write_bytes(b"b")
    root_fd = _open_directory(root)
    try:
        with pytest.raises(ValueError, match="5-byte aggregate relative-path safety limit"):
            artifacts._descriptor_tree_entries(
                root_fd,
                display="test bundle",
                max_entries=10,
                max_relative_path_bytes=5,
            )
    finally:
        os.close(root_fd)


def test_descriptor_inventory_bounds_aggregate_component_work(tmp_path):
    root = tmp_path / "bundle"
    (root / "a" / "b" / "c").mkdir(parents=True)
    root_fd = _open_directory(root)
    try:
        with pytest.raises(
            ValueError,
            match="4-component aggregate traversal-work safety limit",
        ):
            artifacts._descriptor_tree_entries(
                root_fd,
                display="test bundle",
                max_entries=10,
                max_component_work=4,
            )
    finally:
        os.close(root_fd)


def test_private_wrapper_has_derived_depth_and_component_bounds(tmp_path):
    source = tmp_path / "source"
    (source / "a" / "b" / "c").mkdir(parents=True)
    (source / "a" / "b" / "c" / "weights.bin").write_bytes(b"x")
    private = tmp_path / "private"
    (private / "artifact" / "a" / "b" / "c").mkdir(parents=True)
    (private / "artifact" / "a" / "b" / "c" / "weights.bin").write_bytes(b"x")

    source_fd = _open_directory(source)
    private_fd = _open_directory(private)
    try:
        source_entries = artifacts._descriptor_tree_entries(
            source_fd,
            display="source",
            max_entries=4,
            max_depth=3,
            max_component_work=10,
        )
        private_entries = artifacts._descriptor_tree_entries(
            private_fd,
            display="private",
            max_entries=5,
            max_depth=4,
            max_component_work=21,
        )
    finally:
        os.close(private_fd)
        os.close(source_fd)

    assert len(source_entries) == 4
    assert len(private_entries) == 5
    assert artifacts._MAX_PRIVATE_TREE_DEPTH == artifacts._MAX_DESCRIPTOR_TREE_DEPTH + 1
    assert artifacts._MAX_PRIVATE_TREE_COMPONENT_WORK == (
        2 * artifacts._MAX_DESCRIPTOR_TREE_COMPONENT_WORK + 1
    )


def test_directory_snapshot_hashes_and_cleans_through_descriptors(tmp_path, monkeypatch):
    source = tmp_path / "bundle"
    (source / "a").mkdir(parents=True)
    (source / "a" / "nested.bin").write_bytes(b"nested")
    (source / "a!").write_bytes(b"sibling")
    expected = artifacts.sha256_artifact(source)
    source_fd = _open_directory(source)

    def reject_path_hash(*_args, **_kwargs):
        raise AssertionError("descriptor snapshot must not verify through a pathname")

    def reject_listdir(*_args, **_kwargs):
        raise AssertionError("descriptor traversal must use bounded scandir")

    monkeypatch.setattr(artifacts, "sha256_artifact", reject_path_hash)
    monkeypatch.setattr(artifacts.os, "listdir", reject_listdir)
    try:
        snapshot = artifacts.ArtifactSnapshot.from_directory_fd(source_fd, expected)
    finally:
        os.close(source_fd)
    private_root = snapshot.path.parent
    assert (snapshot.path / "a" / "nested.bin").read_bytes() == b"nested"
    assert snapshot.sha256 == expected
    snapshot.close()
    assert not private_root.exists()


def test_file_snapshot_verifies_through_the_retained_private_root(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"trusted")
    expected = artifacts.sha256_artifact(source)

    def reject_path_hash(*_args, **_kwargs):
        raise AssertionError("file snapshot must not verify through a pathname")

    monkeypatch.setattr(artifacts, "sha256_artifact", reject_path_hash)
    with artifacts.ArtifactSnapshot(source, expected) as snapshot:
        assert snapshot.path.read_bytes() == b"trusted"
        assert snapshot.sha256 == expected


def test_file_hash_uses_one_cwd_snapshot(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    replacement = tmp_path / "replacement"
    trusted.mkdir()
    replacement.mkdir()
    (trusted / "model.bin").write_bytes(b"trusted")
    (replacement / "model.bin").write_bytes(b"attacker")
    expected = artifacts.sha256_artifact(trusted / "model.bin")

    actual = _run_after_concurrent_chdir(
        monkeypatch,
        trusted,
        replacement,
        lambda: artifacts.sha256_artifact("model.bin"),
    )
    assert actual == expected


def test_directory_hash_uses_one_cwd_snapshot(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    replacement = tmp_path / "replacement"
    (trusted / "bundle").mkdir(parents=True)
    (replacement / "bundle").mkdir(parents=True)
    (trusted / "bundle" / "weights.bin").write_bytes(b"trusted")
    (replacement / "bundle" / "weights.bin").write_bytes(b"attacker")
    expected = artifacts.sha256_artifact(trusted / "bundle")

    actual = _run_after_concurrent_chdir(
        monkeypatch,
        trusted,
        replacement,
        lambda: artifacts.sha256_artifact("bundle"),
    )
    assert actual == expected


def test_verify_returns_the_path_from_its_one_cwd_snapshot(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    replacement = tmp_path / "replacement"
    trusted.mkdir()
    replacement.mkdir()
    (trusted / "model.onnx").write_bytes(b"trusted")
    (replacement / "model.onnx").write_bytes(b"attacker")
    expected = artifacts.sha256_artifact(trusted / "model.onnx")

    verified = _run_after_concurrent_chdir(
        monkeypatch,
        trusted,
        replacement,
        lambda: artifacts.verify_artifact("model.onnx", expected),
    )
    assert verified == trusted / "model.onnx"


def test_snapshot_copies_from_its_one_cwd_snapshot(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    replacement = tmp_path / "replacement"
    (trusted / "bundle").mkdir(parents=True)
    (replacement / "bundle").mkdir(parents=True)
    (trusted / "bundle" / "weights.bin").write_bytes(b"trusted")
    (replacement / "bundle" / "weights.bin").write_bytes(b"attacker")
    expected = artifacts.sha256_artifact(trusted / "bundle")

    def snapshot_bytes():
        with artifacts.ArtifactSnapshot("bundle", expected) as snapshot:
            return (snapshot.path / "weights.bin").read_bytes()

    copied = _run_after_concurrent_chdir(
        monkeypatch,
        trusted,
        replacement,
        snapshot_bytes,
    )
    assert copied == b"trusted"


def test_descriptor_hash_rejects_mutation_after_inventory(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    member = root / "weights.bin"
    member.write_bytes(b"trusted")
    root_fd = _open_directory(root)
    real_inventory = artifacts._descriptor_tree_entries
    inventory_calls = 0

    def mutate_after_inventory(*args, **kwargs):
        nonlocal inventory_calls
        entries = real_inventory(*args, **kwargs)
        inventory_calls += 1
        if inventory_calls == 1:
            member.write_bytes(b"altered")
        return entries

    monkeypatch.setattr(artifacts, "_descriptor_tree_entries", mutate_after_inventory)
    try:
        with pytest.raises(ValueError, match="changed|replaced"):
            artifacts.sha256_directory_fd(root_fd)
    finally:
        os.close(root_fd)


def test_private_cleanup_restores_read_only_parent_mode_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    expected = artifacts.sha256_artifact(source)
    source_fd = _open_directory(source)
    try:
        snapshot = artifacts.ArtifactSnapshot.from_directory_fd(source_fd, expected)
    finally:
        os.close(source_fd)

    original_unlink = artifacts.os.unlink
    injected = False

    def fail_once(path, *args, **kwargs):
        nonlocal injected
        if path == "weights.bin" and not injected:
            injected = True
            raise OSError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", fail_once)
    with pytest.raises(OSError, match="injected cleanup failure"):
        snapshot.close()
    assert (snapshot.path.stat().st_mode & 0o777) == 0o500

    monkeypatch.setattr(artifacts.os, "unlink", original_unlink)
    private_root = snapshot.path.parent
    snapshot.close()
    assert not private_root.exists()


def test_private_cleanup_restores_mode_after_indeterminate_widen_failure(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    expected = artifacts.sha256_artifact(source)
    snapshot = artifacts.ArtifactSnapshot(source, expected)
    real_fchmod = artifacts.os.fchmod
    injected = False

    def widen_then_fail(fd, mode):
        nonlocal injected
        if mode == 0o700 and not injected:
            injected = True
            real_fchmod(fd, mode)
            raise OSError("indeterminate widening failure")
        return real_fchmod(fd, mode)

    monkeypatch.setattr(artifacts.os, "fchmod", widen_then_fail)
    with pytest.raises(OSError, match="indeterminate widening failure"):
        snapshot.close()
    assert (snapshot.path.stat().st_mode & 0o777) == 0o500

    monkeypatch.setattr(artifacts.os, "fchmod", real_fchmod)
    private_root = snapshot.path.parent
    snapshot.close()
    assert not private_root.exists()


def test_private_root_open_failure_removes_only_the_created_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_mkdtemp = artifacts.tempfile.mkdtemp
    real_open = artifacts.os.open
    created: list[pathlib.Path] = []

    def track_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = pathlib.Path(real_mkdtemp(*args, **kwargs))
        created.append(path)
        return str(path)

    def fail_root_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if created and path == created[0].name and kwargs.get("dir_fd") is not None:
            raise OSError("injected private root open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifacts.tempfile, "mkdtemp", track_mkdtemp)
    monkeypatch.setattr(artifacts.os, "open", fail_root_open)
    with pytest.raises(OSError, match="injected private root open failure"):
        artifacts._PrivateSnapshotDirectory(max_entries=1)
    assert len(created) == 1
    assert not created[0].exists()


def test_implicit_snapshot_cleanup_uses_the_bounded_private_tree(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    expected = artifacts.sha256_artifact(source)
    snapshot = artifacts.ArtifactSnapshot(source, expected)
    private_root = snapshot.path.parent

    del snapshot
    gc.collect()
    assert not private_root.exists()


def test_private_cleanup_unlinks_a_foreign_symlink_without_following_it(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    expected = artifacts.sha256_artifact(source)
    snapshot = artifacts.ArtifactSnapshot(source, expected)
    private_root = snapshot.path.parent

    snapshot.path.chmod(0o700)
    (snapshot.path / "weights.bin").unlink()
    (snapshot.path / "weights.bin").symlink_to(outside)
    snapshot.close()

    assert outside.read_bytes() == b"outside"
    assert not private_root.exists()


@pytest.mark.skipif(
    not hasattr(os, "scandir") or sys.platform == "win32",
    reason="descriptor scandir and RLIMIT_NOFILE are Unix-only",
)
def test_deep_descriptor_hash_copy_and_cleanup_fit_low_fd_limit():
    source_root = pathlib.Path(__file__).parents[1] / "src"
    script = textwrap.dedent(
        """
        import os
        import pathlib
        import resource
        import tempfile

        from manwe.common.artifacts import (
            ArtifactSnapshot,
            sha256_artifact,
            sha256_directory_fd,
        )

        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary).resolve() / "source"
            source.mkdir()
            current = source
            for _ in range(96):
                current = current / "d"
                current.mkdir()
            (current / "weights.bin").write_bytes(b"trusted")
            expected = sha256_artifact(source)
            source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
            old_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            low_limit = min(24, old_limit[0])
            if low_limit < 16:
                os.close(source_fd)
                raise SystemExit(0)
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (low_limit, old_limit[1]))
                assert sha256_directory_fd(source_fd) == expected
                snapshot = ArtifactSnapshot.from_directory_fd(source_fd, expected)
                private_root = snapshot.path.parent
                assert snapshot.sha256 == expected
                snapshot.close()
                assert not private_root.exists()
            finally:
                resource.setrlimit(resource.RLIMIT_NOFILE, old_limit)
                os.close(source_fd)
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_maximum_public_directory_depth_snapshot_can_cleanup_private_wrapper(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    current = source
    for _ in range(artifacts._MAX_DESCRIPTOR_TREE_DEPTH):
        current = current / "d"
        current.mkdir()
    (current / "weights.bin").write_bytes(b"trusted")

    expected = artifacts.sha256_artifact(source)
    snapshot = artifacts.ArtifactSnapshot(source, expected)
    private_root = snapshot.path.parent
    snapshot.close()

    assert not private_root.exists()
