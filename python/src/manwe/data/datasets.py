"""Registry of the datasets manwe trains and evaluates on.

Curated from the 2026 SOTA survey (``docs/research/SOTA-2026.md``). Most
counter-UAS datasets require registration or a request — the registry records how
to obtain each one and what it is good for, so ``manwe data`` can point a user at
the right corpus instead of silently failing to download it.
"""

from __future__ import annotations

from dataclasses import dataclass

Access = str  # "open" | "registration" | "request"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str  # rgb | thermal | audio | multimodal | multicam
    task: str  # detect | track | seld | fusion | calib
    homepage: str
    access: Access
    license: str
    notes: str


DATASETS: dict[str, DatasetSpec] = {
    # --- vision: aerial / anti-UAV ---
    "anti-uav410": DatasetSpec(
        "anti-uav410",
        "thermal",
        "track",
        "https://github.com/ZhaoJ9014/Anti-UAV",
        "request",
        "research-only",
        "Thermal-IR anti-UAV tracking; the on-target thermal benchmark + export-fidelity gate.",
    ),
    "drone-vs-bird": DatasetSpec(
        "drone-vs-bird",
        "rgb",
        "detect",
        "https://wosdetc2025.wordpress.com/",
        "registration",
        "research-only",
        "WOSDETC challenge; RGB with BIRD distractors — mine birds as hard negatives.",
    ),
    "visdrone": DatasetSpec(
        "visdrone",
        "rgb",
        "detect",
        "http://aiskyeye.com/",
        "open",
        "research-only",
        "Small-object aerial detection/tracking; canonical AP-small benchmark.",
    ),
    "dota": DatasetSpec(
        "dota",
        "rgb",
        "detect",
        "https://captain-whu.github.io/DOTA/",
        "open",
        "research-only",
        "Large-scale oriented aerial objects; for oriented-box experiments.",
    ),
    "ard-mav": DatasetSpec(
        "ard-mav",
        "rgb",
        "detect",
        "https://arxiv.org/abs/2503.07115",
        "request",
        "research-only",
        "Air-to-air / moving-camera micro-air-vehicle detection.",
    ),
    # --- audio ---
    "mmaud": DatasetSpec(
        "mmaud",
        "multimodal",
        "fusion",
        "https://github.com/ntu-aris/MMAUD",
        "open",
        "research-only",
        "Mic-array + LiDAR + radar + vision anti-UAV with mm-accurate Leica GT; 3D loc + AV fusion.",
    ),
    "dregon": DatasetSpec(
        "dregon",
        "audio",
        "seld",
        "https://dregon.inria.fr/",
        "open",
        "research-only",
        "Drone ego-noise + moving-source audio; harden low-SNR / ego-noise.",
    ),
    "droneaudio-alemadi": DatasetSpec(
        "droneaudio-alemadi",
        "audio",
        "detect",
        "https://github.com/saraalemadi/DroneAudioDataset",
        "open",
        "open",
        "Binary drone presence audio; quick acoustic presence baseline.",
    ),
    "starss23": DatasetSpec(
        "starss23",
        "audio",
        "seld",
        "https://zenodo.org/record/7880637",
        "open",
        "CC-BY-4.0",
        "DCASE SELD real recordings (az/el + distance since 2024); SELD training base.",
    ),
    # --- multi-camera ---
    "wildtrack": DatasetSpec(
        "wildtrack",
        "multicam",
        "calib",
        "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
        "open",
        "research-only",
        "7-camera calibrated multi-view; validate calibration + triangulation geometry.",
    ),
    "multiviewx": DatasetSpec(
        "multiviewx",
        "multicam",
        "track",
        "https://github.com/hou-yz/MVDet",
        "open",
        "research-only",
        "Synthetic multi-view; template for a synthetic AERIAL multi-cam set.",
    ),
}


def list_datasets(modality: str | None = None) -> list[str]:
    return [k for k, v in DATASETS.items() if modality is None or v.modality == modality]


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {list_datasets()}")
    return DATASETS[name]


def access_instructions(name: str) -> str:
    d = get_dataset(name)
    how = {
        "open": "Downloadable directly from the homepage.",
        "registration": "Requires registering for the challenge/portal first.",
        "request": "Requires emailing the authors / signing a usage agreement.",
    }[d.access]
    return (
        f"{d.name}  [{d.modality} / {d.task}]  license={d.license}\n"
        f"  {d.notes}\n  Access: {how}\n  Homepage: {d.homepage}"
    )


__all__ = ["DatasetSpec", "DATASETS", "list_datasets", "get_dataset", "access_instructions"]
