"""Device resolution for training/inference across Metal (MPS), CUDA and CPU.

manwe must run the *same* pipelines on Apple Silicon (Metal/MPS) and NVIDIA
(CUDA). This module centralises that choice and the autocast dtype policy so no
pillar hard-codes ``"cuda"``. torch is imported lazily — :func:`describe_hardware`
works with numpy/stdlib only.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Literal

DeviceKind = Literal["cuda", "mps", "cpu"]


@dataclass(slots=True, frozen=True)
class Device:
    """A resolved compute device plus its recommended mixed-precision dtype."""

    kind: DeviceKind
    index: int = 0
    name: str = ""
    #: Recommended autocast dtype name: "bfloat16" | "float16" | "float32".
    autocast_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.kind not in {"cuda", "mps", "cpu"}:
            raise ValueError(f"unsupported device kind {self.kind!r}")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("device index must be a nonnegative integer")
        if self.kind != "cuda" and self.index != 0:
            raise ValueError("only CUDA devices may have a nonzero index")
        if not isinstance(self.name, str):
            raise TypeError("device name must be a string")
        if self.autocast_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported autocast dtype")

    @property
    def torch_device(self) -> str:
        if self.kind == "cuda":
            return f"cuda:{self.index}"
        return self.kind  # "mps" or "cpu"

    @property
    def supports_amp(self) -> bool:
        """Whether automatic mixed precision is worth enabling here.

        CUDA has mature AMP. MPS autocast exists but is fragile for *training*;
        we default it off and let inference opt in. CPU AMP is not useful.
        """
        return self.kind == "cuda"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        n = f" ({self.name})" if self.name else ""
        return f"{self.torch_device}{n} [autocast={self.autocast_dtype}]"


def resolve_device(prefer: str = "auto", *, allow_fallback: bool = False) -> Device:
    """Resolve a :class:`Device`, preferring accelerators when ``prefer='auto'``.

    ``prefer`` may be ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or an explicit
    ``"cuda:1"``. ``"auto"`` selects the best available device. An explicit
    accelerator request fails closed unless ``allow_fallback=True``; silently
    training on CPU can turn a bounded job into a multi-day one.
    """
    if type(allow_fallback) is not bool:
        raise TypeError("allow_fallback must be a boolean")
    kind, index = _parse_prefer(prefer)
    if kind == "cpu":
        return Device(kind="cpu", name=platform.processor() or "cpu")

    try:
        import torch
    except ImportError:
        if kind not in {"auto", "cpu"} and not allow_fallback:
            raise RuntimeError(f"requested {prefer!r}, but torch is not installed") from None
        return Device(kind="cpu", name="cpu (torch not installed)")
    except OSError as exc:
        raise RuntimeError(f"torch is installed but failed to load: {exc}") from exc

    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if kind == "auto":
        kind = "cuda" if cuda_ok else "mps" if mps_ok else "cpu"

    if kind == "cuda" and (not cuda_ok or index >= torch.cuda.device_count()):
        if not allow_fallback:
            available = torch.cuda.device_count() if cuda_ok else 0
            raise RuntimeError(
                f"requested {prefer!r}, but only {available} CUDA device(s) are available"
            )
        if cuda_ok:
            kind, index = "cuda", 0
        elif mps_ok:
            kind, index = "mps", 0
        else:
            kind, index = "cpu", 0
    if kind == "mps" and not mps_ok:
        if not allow_fallback:
            raise RuntimeError("requested 'mps', but the MPS backend is unavailable")
        kind = "cuda" if cuda_ok else "cpu"

    if kind == "cuda":
        name = torch.cuda.get_device_name(index)
        # bf16 on Ampere+ (compute capability >= 8.0); else fp16.
        try:
            major, _ = torch.cuda.get_device_capability(index)
            autocast = "bfloat16" if major >= 8 else "float16"
        except Exception:  # pragma: no cover - defensive
            autocast = "float16"
        return Device(kind="cuda", index=index, name=name, autocast_dtype=autocast)

    if kind == "mps":
        # MPS training is most stable in fp32; fp16 is fine for inference.
        return Device(kind="mps", name="Apple Metal (MPS)", autocast_dtype="float32")

    return Device(kind="cpu", name=platform.processor() or "cpu")


def _parse_prefer(prefer: str) -> tuple[str, int]:
    if not isinstance(prefer, str):
        raise TypeError("device preference must be a string")
    prefer = prefer.strip().lower()
    if not prefer:
        raise ValueError("device preference must not be empty")
    if ":" in prefer:
        kind, _, idx = prefer.partition(":")
        if kind != "cuda" or not idx.isdigit():
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<nonnegative index>")
        try:
            return kind, int(idx)
        except ValueError:
            raise ValueError(
                "device must be auto, cpu, mps, cuda, or cuda:<nonnegative index>"
            ) from None
    if prefer not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<nonnegative index>")
    return prefer, 0


def describe_hardware() -> dict[str, object]:
    """Return a JSON-serialisable summary of the available compute hardware.

    Safe to call with no ML deps installed — used by ``manwe doctor`` and by the
    benchmark harness to stamp every result with its execution context.
    """
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": None,
        "cuda": {"available": False},
        "mps": {"available": False},
        "torch_error": None,
    }
    try:
        import torch
    except ImportError:
        return info
    except OSError as exc:
        info["torch_error"] = str(exc)
        return info

    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "version": getattr(torch.version, "cuda", None),
            "bf16": torch.cuda.is_bf16_supported(),
        }
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        info["mps"] = {"available": True, "built": mps.is_built()}
    return info


__all__ = ["Device", "DeviceKind", "resolve_device", "describe_hardware"]
