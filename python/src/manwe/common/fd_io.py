"""Explicit ownership adapters for already-open file descriptors."""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from types import TracebackType
from typing import BinaryIO, Literal, TextIO

_Close = Callable[[], None]


def _add_cleanup_note(primary: BaseException, label: str, cleanup: BaseException) -> None:
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(f"{label}: {cleanup}")


def _attach_cleanup_failure(
    primary: BaseException,
    cleanup: BaseException,
    label: str,
) -> None:
    """Keep ``primary`` primary while retaining one cleanup error as its cause."""
    _add_cleanup_note(primary, label, cleanup)
    if primary.__cause__ is None and not primary.__suppress_context__:
        if cleanup.__context__ is primary:
            cleanup.__context__ = None
        primary.__cause__ = cleanup
        primary.__suppress_context__ = True


def attach_cleanup_failure(
    primary: BaseException,
    cleanup: BaseException,
    label: str,
) -> None:
    """Attach a cleanup failure without replacing the operation's primary error."""
    _attach_cleanup_failure(primary, cleanup, label)


def _cleanup_failed_construction(
    primary: BaseException,
    subordinate: io.IOBase | None,
    fd: int,
) -> None:
    if subordinate is not None:
        try:
            subordinate.close()
        except BaseException as cleanup:
            _attach_cleanup_failure(
                primary,
                cleanup,
                "non-owning stream cleanup also failed",
            )
    try:
        os.close(fd)
    except BaseException as cleanup:
        _attach_cleanup_failure(
            primary,
            cleanup,
            "descriptor cleanup also failed",
        )


def _finish_close(close_stream: _Close, fd: int) -> None:
    stream_error: BaseException | None = None
    try:
        close_stream()
    except BaseException as exc:
        stream_error = exc

    descriptor_error: BaseException | None = None
    try:
        os.close(fd)
    except BaseException as exc:
        descriptor_error = exc

    if stream_error is not None:
        if descriptor_error is not None:
            _attach_cleanup_failure(
                stream_error,
                descriptor_error,
                "descriptor close also failed",
            )
        raise stream_error.with_traceback(stream_error.__traceback__)
    if descriptor_error is not None:
        raise descriptor_error.with_traceback(descriptor_error.__traceback__)


def _exit_preserving_body_exception(
    close_stream: _Close,
    body_error: BaseException | None,
) -> None:
    try:
        close_stream()
    except BaseException as cleanup:
        if body_error is None:
            raise
        _attach_cleanup_failure(
            body_error,
            cleanup,
            "stream cleanup failed",
        )


def _nonowning_file(fd: int, mode: Literal["r", "w"]) -> io.FileIO:
    """Wrap ``fd`` without ever transferring descriptor ownership to ``io``."""
    return io.FileIO(fd, mode, closefd=False)


class _OwnedBufferedReader(io.BufferedReader):
    __slots__ = ("_owned_fd",)

    def __init__(self, fd: int) -> None:
        self._owned_fd = -1
        raw: io.FileIO | None = None
        try:
            raw = _nonowning_file(fd, "r")
            super().__init__(raw)
            self._owned_fd = fd
        except BaseException as primary:
            self._owned_fd = -1
            _cleanup_failed_construction(primary, raw, fd)
            raise

    def close(self) -> None:
        fd = self._owned_fd
        if fd < 0:
            return
        self._owned_fd = -1
        _finish_close(super().close, fd)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _exit_preserving_body_exception(self.close, exc)


class _OwnedBufferedWriter(io.BufferedWriter):
    __slots__ = ("_owned_fd",)

    def __init__(self, fd: int) -> None:
        self._owned_fd = -1
        raw: io.FileIO | None = None
        try:
            raw = _nonowning_file(fd, "w")
            super().__init__(raw)
            self._owned_fd = fd
        except BaseException as primary:
            self._owned_fd = -1
            _cleanup_failed_construction(primary, raw, fd)
            raise

    def close(self) -> None:
        fd = self._owned_fd
        if fd < 0:
            return
        self._owned_fd = -1
        _finish_close(super().close, fd)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _exit_preserving_body_exception(self.close, exc)


class _OwnedTextWriter(io.TextIOWrapper):
    __slots__ = ("_owned_fd",)

    def __init__(
        self,
        fd: int,
        *,
        encoding: str,
        errors: str | None,
        newline: str | None,
    ) -> None:
        self._owned_fd = -1
        raw: io.FileIO | None = None
        buffer: io.BufferedWriter | None = None
        try:
            raw = _nonowning_file(fd, "w")
            buffer = io.BufferedWriter(raw)
            super().__init__(
                buffer,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
            self._owned_fd = fd
        except BaseException as primary:
            self._owned_fd = -1
            subordinate: io.IOBase | None = buffer if buffer is not None else raw
            _cleanup_failed_construction(primary, subordinate, fd)
            raise

    def close(self) -> None:
        fd = self._owned_fd
        if fd < 0:
            return
        self._owned_fd = -1
        _finish_close(super().close, fd)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _exit_preserving_body_exception(self.close, exc)


def owned_binary_reader(fd: int) -> BinaryIO:
    """Take ownership of ``fd`` and expose it as a buffered binary reader."""
    return _OwnedBufferedReader(fd)


def owned_binary_writer(fd: int) -> BinaryIO:
    """Take ownership of ``fd`` and expose it as a buffered binary writer."""
    return _OwnedBufferedWriter(fd)


def owned_text_writer(
    fd: int,
    *,
    encoding: str = "utf-8",
    errors: str | None = None,
    newline: str | None = None,
) -> TextIO:
    """Take ownership of ``fd`` and expose it as a buffered text writer."""
    return _OwnedTextWriter(
        fd,
        encoding=encoding,
        errors=errors,
        newline=newline,
    )


__all__ = [
    "attach_cleanup_failure",
    "owned_binary_reader",
    "owned_binary_writer",
    "owned_text_writer",
]
