"""manwe command-line interface (stdlib argparse — always importable).

manwe doctor            hardware + which optional extras are installed
manwe models            list the detector zoo (license + track)
manwe data [NAME]       list datasets / show how to obtain one
manwe synth OUT         generate an offline synthetic detection dataset
manwe fusion-sim        run a synthetic multi-sensor scenario across filters
manwe vision-train CFG  train a detector architecture from scratch (needs an extra)
manwe export W -f onnx  produce one provenance-bound raw artifact   (needs [export])
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from ._version import __version__

_MAX_TRAIN_CONFIG_BYTES = 1 << 20
_FUSION_STEP_SECONDS = 0.5
_FUSION_WARMUP_FRAMES = 3


class _CLIUsageError(ValueError):
    """An expected command-input failure that argparse should report."""


def _usage_error(error: Exception) -> NoReturn:
    message = str(error).strip() or error.__class__.__name__
    raise _CLIUsageError(message) from error


def _bounded_integer(
    name: str, *, minimum: int, maximum: int | None = None
) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if parsed < minimum or (maximum is not None and parsed > maximum):
            if maximum is None:
                raise argparse.ArgumentTypeError(f"{name} must be >= {minimum}")
            raise argparse.ArgumentTypeError(f"{name} must be in [{minimum}, {maximum}]")
        return parsed

    return parse


def _bounded_float(
    name: str, *, minimum: float, maximum: float | None = None
) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from error
        if (
            not math.isfinite(parsed)
            or parsed < minimum
            or (maximum is not None and parsed > maximum)
        ):
            if maximum is None:
                raise argparse.ArgumentTypeError(f"{name} must be finite and >= {minimum:g}")
            raise argparse.ArgumentTypeError(
                f"{name} must be finite and in [{minimum:g}, {maximum:g}]"
            )
        return parsed

    return parse


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise argparse.ArgumentTypeError("weights-sha256 must be a 64-character hexadecimal digest")
    return value.lower()


def _cmd_doctor(_args) -> int:
    from .common.device import describe_hardware, resolve_device

    hw = describe_hardware()
    print(f"manwe {__version__}")
    print(json.dumps(hw, indent=2))
    print(f"resolved device: {resolve_device('auto')}")
    print("\noptional extras:")
    for mod, extra in [
        ("torch", "vision"),
        ("ultralytics", "vision"),
        ("rfdetr", "rfdetr"),
        ("sahi", "vision"),
        ("onnxruntime", "export"),
        ("coremltools", "export"),
    ]:
        try:
            __import__(mod)
            status = "✓ installed"
        except ImportError:
            status = (
                f"— missing (local extra manwe-perception[{extra}]: "
                f"uv sync --locked --extra {extra})"
            )
        print(f"  {mod:14s} {status}")
    return 0


def _cmd_models(args) -> int:
    from .vision.models import MODEL_ZOO, list_models

    for name in list_models(args.track):
        s = MODEL_ZOO[name]
        print(f"  {name:16s} {s.params_m:6.1f}M  {s.license:10s} [{s.track}]  {s.notes}")
    return 0


def _cmd_data(args) -> int:
    from .data.datasets import DATASETS, access_instructions, list_datasets

    if args.name:
        try:
            instructions = access_instructions(args.name)
        except ValueError as error:
            _usage_error(error)
        print(instructions)
    else:
        for name in list_datasets(args.modality):
            d = DATASETS[name]
            print(f"  {name:20s} [{d.modality}/{d.task}]  access={d.access}  {d.notes}")
    return 0


def _cmd_synth(args) -> int:
    from .data.synthetic import make_vision_smoke

    try:
        yaml_path = make_vision_smoke(
            args.out,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        IsADirectoryError,
        PermissionError,
    ) as error:
        _usage_error(error)
    print(f"wrote synthetic dataset manifest: {yaml_path}")
    return 0


def _cmd_fusion_sim(args) -> int:
    from .fusion import MultiSensorTracker, TrackerConfig, make_scenario, score_tracker

    if len(set(args.modalities)) != len(args.modalities):
        raise _CLIUsageError("modalities must not contain duplicates")
    try:
        scenario = make_scenario(
            n_targets=args.targets,
            duration=args.duration,
            modalities=tuple(args.modalities),
            p_detect=args.p_detect,
            clutter_rate=args.clutter,
            seed=args.seed,
        )
    except ValueError as error:
        _usage_error(error)
    print(
        f"scenario: {args.targets} targets, {len(scenario.times)} frames, "
        f"modalities={args.modalities}\n"
    )
    print(f"{'filter':12s} {'OSPA':>8s} {'localization':>13s} {'cardinality':>12s}")
    for filt in args.filters:
        tracker = MultiSensorTracker(TrackerConfig(filter=filt))
        s = score_tracker(tracker, scenario)
        print(f"{filt:12s} {s['ospa']:8.2f} {s['localization']:13.2f} {s['cardinality']:12.2f}")
    return 0


def _cmd_vision_train(args) -> int:
    from .common.config_io import read_strict_yaml
    from .vision.train import VisionTrainConfig, train

    try:
        config_path = Path(args.config).expanduser().absolute()
        payload = read_strict_yaml(config_path, _MAX_TRAIN_CONFIG_BYTES, "vision training config")
        if not isinstance(payload, dict):
            raise ValueError("vision training config must contain a YAML mapping")
        data = payload.get("data")
        if isinstance(data, str):
            data_path = Path(data).expanduser()
            if not data_path.is_absolute():
                payload["data"] = str((config_path.parent / data_path).absolute())
        cfg = VisionTrainConfig(**payload)
    except (
        TypeError,
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        PermissionError,
    ) as error:
        _usage_error(error)
    try:
        train(cfg)
    except (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        PermissionError,
    ) as error:
        _usage_error(error)
    return 0


def _cmd_export(args) -> int:
    if not args.allow_unverified:
        raise _CLIUsageError(
            "raw export is not a trusted consumer handoff; rerun with --allow-unverified, "
            "then create a model contract and run the fidelity gate"
        )
    from .export import export_model

    try:
        receipt = export_model(
            args.weights,
            [args.formats],
            output=args.output,
            weights_sha256=args.weights_sha256,
            allow_pickle_checkpoint=args.allow_pickle_checkpoint,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            data=args.data,
            device=args.device,
            nms=args.nms,
            opset=args.opset,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        IsADirectoryError,
        PermissionError,
    ) as error:
        _usage_error(error)
    print(f"  {receipt.format:10s} → {receipt.artifact_path}")
    print(f"  SHA-256   → {receipt.artifact_sha256}")
    print("  tensor signature remains unverified; build it from backend inspection")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manwe", description="Perception research and validation workbench."
    )
    p.add_argument("--version", action="version", version=f"manwe {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report hardware + installed extras").set_defaults(
        func=_cmd_doctor
    )

    m = sub.add_parser("models", help="list the detector zoo")
    m.add_argument("--track", choices=["tooling", "accuracy"], default=None)
    m.set_defaults(func=_cmd_models)

    d = sub.add_parser("data", help="list datasets or show access instructions")
    d.add_argument("name", nargs="?", help="dataset name for detailed access instructions")
    d.add_argument(
        "--modality",
        choices=["rgb", "thermal", "audio", "multimodal", "multicam"],
        default=None,
    )
    d.set_defaults(func=_cmd_data)

    s = sub.add_parser("synth", help="generate an offline synthetic detection dataset")
    s.add_argument("out", help="output directory")
    s.add_argument("--n-train", type=_bounded_integer("n-train", minimum=1), default=12)
    s.add_argument("--n-val", type=_bounded_integer("n-val", minimum=1), default=4)
    s.add_argument("--seed", type=_bounded_integer("seed", minimum=0), default=0)
    s.set_defaults(func=_cmd_synth)

    fs = sub.add_parser("fusion-sim", help="run a synthetic multi-sensor scenario")
    fs.add_argument(
        "--filters",
        nargs="+",
        default=["kalman", "ekf", "ukf", "particle", "imm"],
        choices=["kalman", "ekf", "ukf", "particle", "imm"],
    )
    fs.add_argument("--targets", type=_bounded_integer("targets", minimum=0), default=3)
    fs.add_argument(
        "--duration",
        type=_bounded_float(
            "duration",
            minimum=_FUSION_STEP_SECONDS * _FUSION_WARMUP_FRAMES,
        ),
        default=20.0,
        help="simulation seconds (at least 1.5 so one post-warmup frame is scored)",
    )
    fs.add_argument(
        "--modalities",
        nargs="+",
        choices=["visual", "radar", "acoustic"],
        default=["visual", "radar", "acoustic"],
    )
    fs.add_argument(
        "--p-detect",
        type=_bounded_float("p-detect", minimum=0.0, maximum=1.0),
        default=0.9,
    )
    fs.add_argument("--clutter", type=_bounded_float("clutter", minimum=0.0), default=0.5)
    fs.add_argument("--seed", type=_bounded_integer("seed", minimum=0), default=0)
    fs.set_defaults(func=_cmd_fusion_sim)

    vt = sub.add_parser(
        "vision-train", help="train a detector architecture from scratch from a YAML config"
    )
    vt.add_argument("config", help="path to a vision train config YAML")
    vt.set_defaults(func=_cmd_vision_train)

    ex = sub.add_parser("export", help="produce unverified raw backend artifacts")
    ex.add_argument("weights", help="path to trained weights (.pt)")
    ex.add_argument(
        "--output",
        required=True,
        help="exclusive destination for the raw exported artifact",
    )
    ex.add_argument(
        "--weights-sha256",
        required=True,
        type=_sha256,
        help="expected SHA-256 for the exact local checkpoint",
    )
    ex.add_argument(
        "--allow-pickle-checkpoint",
        action="store_true",
        help="acknowledge that trusted .pt loading can execute serialized code",
    )
    ex.add_argument(
        "-f",
        "--format",
        dest="formats",
        required=True,
        choices=["onnx", "tensorrt", "coreml"],
    )
    ex.add_argument(
        "--imgsz",
        type=_bounded_integer("imgsz", minimum=32, maximum=4096),
        default=640,
    )
    ex.add_argument("--half", action="store_true")
    ex.add_argument("--int8", action="store_true")
    ex.add_argument("--data", default=None, help="calibration manifest (required for --int8)")
    ex.add_argument("--device", default="auto")
    ex.add_argument(
        "--opset",
        type=_bounded_integer("opset", minimum=12, maximum=20),
        default=17,
        help="pinned ONNX opset (ONNX/TensorRT)",
    )
    ex.add_argument("--nms", action="store_true", help="embed NMS when the backend supports it")
    ex.add_argument(
        "--allow-unverified",
        action="store_true",
        help="acknowledge that export alone does not satisfy a consumer contract",
    )
    ex.set_defaults(func=_cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    from .common.logging import configure_logging

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except _CLIUsageError as error:
        parser.error(str(error))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
