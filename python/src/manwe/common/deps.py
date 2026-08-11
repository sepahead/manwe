"""Helpers for optional dependencies — clear errors that name the extra to install."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module_name: str, extra: str) -> ModuleType:
    """Import ``module_name`` or name the locked local extra that provides it.

    >>> require("torch", "vision")  # doctest: +SKIP
    """
    if type(module_name) is not str or not module_name:
        raise TypeError("module_name must be a nonempty string")
    if type(extra) is not str or not extra:
        raise TypeError("extra must be a nonempty string")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        target_missing = missing == module_name or (
            type(missing) is str and module_name.startswith(f"{missing}.")
        )
        problem = (
            "is not installed"
            if target_missing
            else "or one of its runtime dependencies could not be imported"
        )
        raise ImportError(
            f"'{module_name}' is required for this feature but {problem}.\n"
            f"From this checkout run: cd python && uv sync --locked --extra {extra}"
        ) from exc


__all__ = ["require"]
