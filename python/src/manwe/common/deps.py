"""Helpers for optional dependencies — clear errors that name the extra to install."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module_name: str, extra: str) -> ModuleType:
    """Import ``module_name`` or name the locked local extra that provides it.

    >>> require("torch", "vision")  # doctest: +SKIP
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"'{module_name}' is required for this feature but is not installed.\n"
            f"From this checkout run: cd python && uv sync --locked --extra {extra}"
        ) from exc


__all__ = ["require"]
