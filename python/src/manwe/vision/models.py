"""Aerial-object detector zoo and lazy model factory.

Defaults follow the 2026 SOTA survey (``docs/research/SOTA-2026.md``):

* **Accuracy research track** — RF-DETR (DINOv2 backbone). NOTE: only the
  Nano→Large sizes are Apache-2.0; the RF-DETR-XL
  tier (and larger) ships under PML-1.0; dataset and weight rights still require
  separate review for every artifact.
* **Tooling / Apple-ANE track** — Ultralytics YOLO26 (NMS-free) and YOLO11/12,
  which have by far the most mature CoreML/ANE + MPS + CUDA export. NOTE:
  Ultralytics is AGPL-3.0 — a research/benchmark track unless commercially
  licensed.
* SAHI sliced inference is implemented for small aerial objects. Slicing-aided
  training and a P2 stride-4 head remain research proposals.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.deps import require
from ..common.ultralytics import harden_ultralytics_runtime, verify_ultralytics_policy

#: Default detector: mature CoreML/ANE + MPS + CUDA tooling.
#: Switch to an RF-DETR size for the accuracy ceiling (see DEFAULT_ACCURACY).
DEFAULT_DETECTOR = "yolo11s"
#: Default when maximising accuracy / domain transfer (Apache size).
DEFAULT_ACCURACY = "rfdetr-medium"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # "yolo" | "rtdetr" | "rfdetr"
    weight: str  # weight id / checkpoint name
    params_m: float
    license: str
    track: str  # "tooling" | "accuracy"
    notes: str


#: Catalogue of supported detectors with license + provenance so a shippable
#: build can review implementation licenses deliberately (see docs/research).
MODEL_ZOO: dict[str, ModelSpec] = {
    # Ultralytics YOLO — best Apple/ANE tooling, AGPL-3.0.
    "yolo11n": ModelSpec(
        "yolo11n", "yolo", "yolo11n.pt", 2.6, "AGPL-3.0", "tooling", "nano, edge/real-time"
    ),
    "yolo11s": ModelSpec(
        "yolo11s",
        "yolo",
        "yolo11s.pt",
        9.4,
        "AGPL-3.0",
        "tooling",
        "small, best speed/accuracy balance",
    ),
    "yolo11m": ModelSpec("yolo11m", "yolo", "yolo11m.pt", 20.1, "AGPL-3.0", "tooling", "medium"),
    "yolo11l": ModelSpec("yolo11l", "yolo", "yolo11l.pt", 25.3, "AGPL-3.0", "tooling", "large"),
    "yolo11x": ModelSpec("yolo11x", "yolo", "yolo11x.pt", 56.9, "AGPL-3.0", "tooling", "x-large"),
    "yolo12s": ModelSpec(
        "yolo12s",
        "yolo",
        "yolo12s.pt",
        9.3,
        "AGPL-3.0",
        "tooling",
        "attention-centric (recent ultralytics)",
    ),
    "yolo26n": ModelSpec(
        "yolo26n",
        "yolo",
        "yolo26n.pt",
        2.4,
        "AGPL-3.0",
        "tooling",
        "NMS-free end-to-end (recent ultralytics)",
    ),
    "yolo26s": ModelSpec(
        "yolo26s",
        "yolo",
        "yolo26s.pt",
        9.5,
        "AGPL-3.0",
        "tooling",
        "NMS-free, DFL-free, one .pt → 4 backends",
    ),
    # RT-DETR (via Ultralytics) — NMS-free transformer, Apache-ish upstream.
    "rtdetr-l": ModelSpec(
        "rtdetr-l",
        "rtdetr",
        "rtdetr-l.pt",
        32.0,
        "AGPL-3.0",
        "accuracy",
        "RT-DETR large (Ultralytics build)",
    ),
    "rtdetr-x": ModelSpec(
        "rtdetr-x", "rtdetr", "rtdetr-x.pt", 67.0, "AGPL-3.0", "accuracy", "RT-DETR x-large"
    ),
    # RF-DETR accuracy research track. N–L Apache-2.0; XL/2XL are PML-1.0.
    "rfdetr-nano": ModelSpec(
        "rfdetr-nano",
        "rfdetr",
        "rfdetr_nano",
        30.0,
        "Apache-2.0",
        "accuracy",
        "48.0 AP; Apache implementation tier; verify weight/data rights",
    ),
    "rfdetr-small": ModelSpec(
        "rfdetr-small",
        "rfdetr",
        "rfdetr_small",
        30.0,
        "Apache-2.0",
        "accuracy",
        "Apache implementation tier; verify weight/data rights",
    ),
    "rfdetr-medium": ModelSpec(
        "rfdetr-medium",
        "rfdetr",
        "rfdetr_medium",
        33.0,
        "Apache-2.0",
        "accuracy",
        "54.7 AP; Apache implementation tier; verify weight/data rights",
    ),
    "rfdetr-large": ModelSpec(
        "rfdetr-large",
        "rfdetr",
        "rfdetr_large",
        60.0,
        "Apache-2.0",
        "accuracy",
        "large; Apache implementation tier; verify weight/data rights",
    ),
}


def list_models(track: str | None = None) -> list[str]:
    """List zoo model names, optionally filtered by ``track`` (tooling/accuracy)."""
    return [k for k, v in MODEL_ZOO.items() if track is None or v.track == track]


def build_model(
    name: str = DEFAULT_DETECTOR,
    *,
    num_classes: int | None = None,
    pretrained: bool = False,
):
    """Build a detector by zoo name (lazy import of the right backend).

    Backend-managed pretrained downloads are deliberately disabled. Acquire and
    review bootstrap artifacts outside this API, then use a digest-bound training
    workflow once that model family has an implemented local-checkpoint adapter.
    """
    spec = MODEL_ZOO.get(name)
    if spec is None:
        raise ValueError(f"unknown model {name!r}; choose from {list_models()}")
    if type(pretrained) is not bool:
        raise TypeError("pretrained must be a boolean")
    if pretrained:
        raise ValueError("backend-managed pretrained downloads are disabled; use pretrained=False")
    if num_classes is not None and (type(num_classes) is not int or not 1 <= num_classes <= 4096):
        raise ValueError("num_classes must be an integer in [1, 4096] when provided")

    if spec.family == "rfdetr":
        rfdetr = require("rfdetr", "rfdetr")
        ctor = getattr(rfdetr, _rfdetr_class(spec.weight))
        kwargs: dict[str, object] = {}
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        if not pretrained:
            kwargs["pretrain_weights"] = None
        return ctor(**kwargs)

    if num_classes is not None:
        raise ValueError(
            "num_classes is inferred from the training dataset for Ultralytics models; "
            "passing it to build_model would otherwise be silently ignored"
        )

    harden_ultralytics_runtime()
    ultralytics = require("ultralytics", "vision")
    verify_ultralytics_policy()
    weight = spec.weight if pretrained else spec.weight.replace(".pt", ".yaml")
    if spec.family == "rtdetr":
        return ultralytics.RTDETR(weight)
    return ultralytics.YOLO(weight)


def _rfdetr_class(weight: str) -> str:
    # rfdetr exposes RFDETRNano / RFDETRSmall / RFDETRMedium / RFDETRBase / RFDETRXLarge
    mapping = {
        "rfdetr_nano": "RFDETRNano",
        "rfdetr_small": "RFDETRSmall",
        "rfdetr_medium": "RFDETRMedium",
        "rfdetr_large": "RFDETRLarge",
    }
    try:
        return mapping[weight]
    except KeyError as exc:
        raise ValueError(f"unsupported RF-DETR constructor id {weight!r}") from exc


__all__ = [
    "DEFAULT_DETECTOR",
    "DEFAULT_ACCURACY",
    "ModelSpec",
    "MODEL_ZOO",
    "list_models",
    "build_model",
]
