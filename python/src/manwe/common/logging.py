"""Small logging helper that never configures the host application's root logger."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_PACKAGE_LOGGER = logging.getLogger("manwe")
_PACKAGE_LOGGER.addHandler(logging.NullHandler())


def _level_number(level: str) -> int:
    if not isinstance(level, str):
        raise TypeError("log level must be a string")
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level.upper()!r}")
    return numeric


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Configure Manwe's package logger for a CLI/application entry point.

    Library imports must not call :func:`logging.basicConfig` or otherwise mutate
    the process-wide root logger.  Applications that want Manwe logs opt in here.
    """
    global _CONFIGURED
    if type(force) is not bool:
        raise TypeError("force must be a boolean")
    lvl = (level or os.environ.get("MANWE_LOG_LEVEL", "INFO")).upper()
    numeric_level = _level_number(lvl)
    if _CONFIGURED and not force:
        _PACKAGE_LOGGER.setLevel(numeric_level)
        return

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
    except ImportError:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
    handler.setFormatter(formatter)
    _PACKAGE_LOGGER.handlers.clear()
    _PACKAGE_LOGGER.addHandler(handler)
    _PACKAGE_LOGGER.setLevel(numeric_level)
    _PACKAGE_LOGGER.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "manwe", level: str | None = None) -> logging.Logger:
    """Return a package logger without installing process-global handlers."""
    if not isinstance(name, str) or (name != "manwe" and not name.startswith("manwe.")):
        raise ValueError("logger name must be 'manwe' or a child in the 'manwe.' hierarchy")
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(_level_number(level))
    return logger


__all__ = ["configure_logging", "get_logger"]
