"""Single-owner file-descriptor adapter regressions."""

from __future__ import annotations

import os

import pytest

from manwe.common import fd_io


def test_nonowning_raw_wrapper_never_releases_descriptor(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = fd_io._nonowning_file(fd, "r")
        raw.close()
        assert os.fstat(fd)
    finally:
        os.close(fd)


def test_binary_reader_owns_and_closes_descriptor_once(tmp_path, monkeypatch):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    fd = os.open(path, os.O_RDONLY)
    real_close = os.close
    close_calls: list[int] = []

    def track_close(value: int) -> None:
        close_calls.append(value)
        real_close(value)

    monkeypatch.setattr(fd_io.os, "close", track_close)
    handle = fd_io.owned_binary_reader(fd)
    assert handle.read() == b"value"
    handle.close()
    handle.close()
    assert close_calls == [fd]
    with pytest.raises(OSError):
        os.fstat(fd)


def test_constructor_failure_closes_descriptor_and_keeps_primary(tmp_path, monkeypatch):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    fd = os.open(path, os.O_RDONLY)
    real_close = os.close
    close_calls: list[int] = []

    def fail_nonowning_file(value: int, mode: str):
        raise ValueError(f"cannot wrap {value} as {mode}")

    def track_close(value: int) -> None:
        close_calls.append(value)
        real_close(value)

    monkeypatch.setattr(fd_io, "_nonowning_file", fail_nonowning_file)
    monkeypatch.setattr(fd_io.os, "close", track_close)
    with pytest.raises(ValueError, match="cannot wrap"):
        fd_io.owned_binary_reader(fd)
    assert close_calls == [fd]
    with pytest.raises(OSError):
        os.fstat(fd)


def test_close_error_after_release_never_retries_reused_number(tmp_path, monkeypatch):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    fd = os.open(path, os.O_RDONLY)
    sentinel_source = os.open(os.devnull, os.O_RDONLY)
    real_close = os.close
    close_calls: list[int] = []

    def release_reuse_and_fail(value: int) -> None:
        close_calls.append(value)
        real_close(value)
        os.dup2(sentinel_source, value)
        raise OSError("injected post-release close failure")

    handle = fd_io.owned_binary_reader(fd)
    monkeypatch.setattr(fd_io.os, "close", release_reuse_and_fail)
    with pytest.raises(OSError, match="post-release"):
        handle.close()
    assert os.fstat(fd)
    handle.close()
    assert close_calls == [fd]
    monkeypatch.setattr(fd_io.os, "close", real_close)
    real_close(fd)
    real_close(sentinel_source)


def test_body_exception_stays_primary_when_close_fails(tmp_path, monkeypatch):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    fd = os.open(path, os.O_RDONLY)
    real_close = os.close
    close_calls: list[int] = []

    class BodyError(Exception):
        pass

    def fail_before_release(value: int) -> None:
        close_calls.append(value)
        raise OSError("injected close failure")

    handle = fd_io.owned_binary_reader(fd)
    monkeypatch.setattr(fd_io.os, "close", fail_before_release)
    with pytest.raises(BodyError, match="body") as captured, handle:
        raise BodyError("body")
    assert isinstance(captured.value.__cause__, OSError)
    assert close_calls == [fd]
    handle.close()
    assert close_calls == [fd]
    monkeypatch.setattr(fd_io.os, "close", real_close)
    real_close(fd)


def test_binary_and_text_writers_flush_before_descriptor_close(tmp_path):
    binary_path = tmp_path / "value.bin"
    binary_fd = os.open(binary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with fd_io.owned_binary_writer(binary_fd) as handle:
        handle.write(b"value")
    assert binary_path.read_bytes() == b"value"
    with pytest.raises(OSError):
        os.fstat(binary_fd)

    text_path = tmp_path / "value.txt"
    text_fd = os.open(text_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with fd_io.owned_text_writer(text_fd, newline="\n") as handle:
        handle.write("a\nb")
    assert text_path.read_text(encoding="utf-8") == "a\nb"
    with pytest.raises(OSError):
        os.fstat(text_fd)
