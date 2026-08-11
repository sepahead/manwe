"""Versioned machine-contract examples shipped with Manwe."""

from __future__ import annotations

from importlib.resources import files


def candle_detect_example_bytes() -> bytes:
    """Return the reviewed, non-executable schema-2 detection template."""
    return (
        files("manwe.schemas").joinpath("model-contract-v2.candle-detect.example.json").read_bytes()
    )


__all__ = ["candle_detect_example_bytes"]
