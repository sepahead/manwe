<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="Manwe perception proving ground — a machined sky-scope optic with a mechanical iris and a quadcopter caught in the aperture." width="200">
  </picture>
</p>

# Manwe

> A research and validation workbench for airspace perception: vision,
> audio, multi-camera geometry, and multi-target tracking on Apple Metal, NVIDIA
> CUDA, and CPU.

Manwe pairs a Python numerical/training package with Rust/Candle inference and
benchmarking tools. It is intended to produce candidate models and reference
results for downstream systems such as
[**crebain**](https://github.com/sepahead/crebain). It does **not** currently ship
a drop-in adapter for crebain, Galadriel, Engram/NCP, Prisoma, or pid-rs. Their
schemas, tensor layouts, clocks, coordinate frames, and statistical assumptions
are different; the audited gaps and required validation gates are documented in
[the integration status](docs/INTEGRATION_CREBAIN.md).

> **Alpha 1 release candidate.** The current tree targets `0.2.0-alpha.1`.
> Public Python/Rust APIs, CLI flags, manifests, and numerical defaults may change
> before a stable release. The tested host boundary is Linux and Apple Silicon
> macOS; Windows, CUDA execution, real model forwards, and downstream adapters are
> not yet validated release targets.

The design is informed by a dated research survey. That survey is context, not a
substitute for validation on the exact model, dataset, hardware, and consumer:
**[docs/research/SOTA-2026.md](docs/research/SOTA-2026.md)**.

Applications are civilian and dual-use: disaster-relief coordination, urban
delivery-drone deconfliction, infrastructure inspection, wildlife monitoring, and
airspace situational awareness.

---

## Why it's built this way

- **Pure-numpy core, lazy heavy deps.** The fusion, geometry, DOA, metrics, and
  contract layers depend only on numpy — they run without ML runtimes on the
  tested Linux/macOS hosts. Training and raw conversion dependencies are isolated
  behind the `vision` and `export` extras; platform runtimes such as TensorRT are
  installed separately.
- **Explicit device selection.** Python helpers can select CUDA, MPS, or CPU;
  actual operator coverage and numerical parity still have to be checked per path.
- **A candidate contract is code.** `manwe.common.contracts` validates a model
  manifest, artifact digest, tensor descriptions, and the five-class taxonomy. A
  valid manifest does not make an artifact compatible with a consumer by itself.
- **Promotion requires evidence.** The repo contains AP50/AP50-small,
  deployed-threshold precision/recall/FPPI and direct export-agreement gates, plus
  OSPA/GOSPA and latency tools. Consumer fixtures and operational tests remain
  mandatory before deployment.

## The four capability pillars

| Pillar | What it does | Runnable today |
|--------|--------------|----------------|
| **vision** | Detector registry, from-scratch architecture training, sliced inference, postprocess and class mapping | Ultralytics training is runnable with `[vision]`; `[rfdetr]` covers construction/inference only, and local-checkpoint fine-tuning is not implemented |
| **audio** | Microphone-array direction-of-arrival (GCC-PHAT / SRP-PHAT), log-mel/SPL features, acoustic→fusion bridge | ✅ pure numpy |
| **multicam** | Pinhole calibration, N-view DLT / midpoint triangulation, cross-camera correlation | ✅ pure numpy |
| **fusion** | KF / EKF / UKF / PF / IMM, Mahalanobis gating, M-of-N track lifecycle, OSPA/GOSPA, synthetic scenarios | ✅ pure numpy |

## Quick start

Prerequisites for the alpha development workflow:

- Python 3.10–3.14 for the core; Python 3.11–3.12 is the conservative heavy-ML
  path documented below.
- [`uv` 0.11.28](https://docs.astral.sh/uv/getting-started/installation/), matching
  CI, and Rust 1.95 or newer.
- Xcode/Metal tooling for the macOS feature path, an NVIDIA CUDA toolkit for the
  unvalidated CUDA feature, and FFmpeg for camera/video tools.

### Python training ground

```bash
cd python

# Core (numpy only) — the fusion/geometry/audio/eval core + CLI run immediately:
uv sync --locked

# Heavy pillars (use Python 3.11–3.12; torch wheels lag new releases):
uv sync --locked --extra vision --extra export  # pinned local-only training/export adapters
uv sync --locked --extra rfdetr                 # RF-DETR architecture construction
uv sync --locked --extra all                    # combined supported optional stack
```

The `manwe` CLI:

```bash
uv run --no-sync manwe doctor                    # hardware + installed extras
uv run --no-sync manwe models --track accuracy   # detector zoo and licenses
uv run --no-sync manwe data                      # dataset registry
uv run --no-sync manwe synth /tmp/smoke          # offline synthetic dataset
uv run --no-sync manwe fusion-sim                # synthetic multi-sensor comparison
# First replace the example dataset paths in configs/vision/data.example.yaml:
uv run --no-sync manwe vision-train configs/vision/aerial.yaml  # from-scratch training
uv run --no-sync manwe export /abs/best.pt -f onnx \
  --weights-sha256 <64-hex> --allow-pickle-checkpoint \
  --output /abs/candidate.onnx --allow-unverified
```

The example YAML files are repository fixtures, not wheel package data; this quick
start assumes a checkout. A wheel user must supply an equivalent local config and
dataset manifest explicitly.

Both acknowledgements are intentional. `--allow-pickle-checkpoint` confirms that
the exact digest-bound `.pt` origin is trusted even under restricted loading;
`--allow-unverified` confirms that successful conversion is not a consumer handoff
or fidelity result. The exporter works in a private snapshot and refuses to
replace an existing destination. TensorRT INT8 calibration likewise uses a bounded
read-only private dataset snapshot. Only the manifest's `val` images count: every
candidate must have matching suffix/content, identity EXIF orientation, bounded
encoded/decoded size, and decode as 3-channel `uint8`; images that collapse to the
same resized/letterboxed backend tensor are rejected. A hash-ranked 512-image
subset is then exposed in an exact label/cache/adjacent-array-free private view
with batch 1 and fraction 1.0. Ultralytics cache loading/writing and network URL
probes are disabled, and its process-global `bgr=0` validation formatter is
serialized and made deterministic only for the calibration operation, so
TensorRT 7–10 and TensorRT 11 ModelOpt consume the same bounded set. The
exporter rejects the pinned Ultralytics branch that would silently fall back to
FP32 when TensorRT reports no INT8 capability. Its receipt digest binds the
TensorRT version/route, loader policy, `imgsz`, normalized manifest, validated
inventory, and exact copied tree; both the caller-visible manifest file and its
declared source tree are pinned and rechecked through descriptor-relative POSIX
I/O before publication. Dataset roots and splits must be absolute or descendant-
relative (`..`, symlinks, and special path components are rejected).
These checks prove availability and byte/tensor uniqueness, not target-domain
representativeness or per-layer INT8 execution. `precision="int8"` records the
requested mixed-precision route; engine-inspector and fidelity evidence remain
promotion requirements. TensorRT 11 preflights locally installed
`nvidia-modelopt>=0.44`, binds its exact version into the digest, and rejects
image sizes whose conservative 10× tensor-materialization estimate exceeds
8 GiB. The ModelOpt version and TensorRT route are not yet explicit receipt
fields, so independent reconstruction still needs the build environment record.
This is not an operating-system filesystem snapshot: privileged mount changes,
SHA-256 collisions, and malicious same-UID mutation of the process-owned private
loader tree during backend reads require an isolated build worker or stronger OS
containment.

`uv run --no-sync manwe fusion-sim` on the default 3-target, 3-sensor (visual + radar + acoustic)
scenario — mean OSPA (lower is better) over 41 frames:

```
filter           OSPA  localization  cardinality
kalman           4.30          1.78         2.95
ekf              4.30          1.78         2.95
ukf              4.30          1.78         2.95
particle         9.90          2.03         9.23
imm              9.33          1.88         8.89
```

These are deterministic synthetic-regression results for the command's current
defaults and seed, not real-world accuracy or a comparison with another tracker.

### Rust/Candle reference inference CLI

```bash
cargo build --release --locked                         # CPU
cargo build --release --locked --features metal       # Apple Metal
cargo build --release --locked --features cuda        # NVIDIA CUDA (not yet CI-validated)
./target/release/manwe --model path/to/yolov8s.safetensors --model-sha256 <64-hex> --which s path/to/image.jpg
./target/release/manwe --model path/to/yolov8s-pose.safetensors --model-sha256 <64-hex> --which s --task pose path/to/image.jpg

# Experimental macOS camera viewer (adds Bevy; not a cross-platform alpha API)
cargo build --release --locked --features viewer,metal --bin camera_view
# camera_view also requires --model and --model-sha256 (or matching MANWE_* variables)
```

These commands require the repository's exact Candle YOLOv8 key/shape convention;
a generic `.safetensors` extension or an Ultralytics checkpoint is insufficient.
No checked-in converter currently produces that graph, so treat the commands as a
reference runtime until a pinned artifact manifest and golden forward fixture are
available. Neither accelerator path is a downstream deployment adapter.

Annotated JPEGs are staged, synced, and verified before no-replace publication. A
hard interruption may leave a sibling `.manwe-image-output-*.in-progress`
directory, optionally beside a complete-looking JPEG; treat the marker as an
incomplete publication and inspect it before cleanup. A failure after the final
link deliberately preserves both artifacts rather than unlinking a pathname that
may already have been replaced.

CI compiles/tests the CPU path on Linux and the Metal/viewer path on arm64 macOS.
The CUDA feature remains a target-hardware gate: it needs an NVIDIA runner, the
exact model artifact, and numerical parity evidence before it can support a
release claim.

The Rust artifact reader currently provides its no-follow and stable-identity
guarantees on Unix. It fails closed on other platforms until an equivalent
reparse-point-safe open is implemented. The pose graph is the fixed COCO
17-keypoint `(x, y, visibility)` contract; other keypoint layouts are rejected.

## Using the pillars

```python
# Fusion — an independent Python reference implementation
from manwe.fusion import MultiSensorTracker, TrackerConfig, Measurement
tr = MultiSensorTracker(TrackerConfig(filter="ekf"))
tracks = tr.step([Measurement("radar", [120.0, 0.3, 0.1], [9.0, 4e-4, 4e-4], timestamp=0.0)], 0.0)

# Multi-camera — calibrate, correlate, triangulate
import numpy as np
from manwe.multicam import Camera, Detection2D, correlate_and_triangulate
cams = [Camera.from_lookat([0,0,100],[0,0,0]), Camera.from_lookat([100,0,20],[0,0,0])]
target = np.array([0.0, 0.0, 0.0])
dets = [
    Detection2D(
        index,
        camera.project(target),
        timestamp=0.0,
        pixels_undistorted=True,
        pixel_std_px=1.0,
    )
    for index, camera in enumerate(cams)
]
dets3d = correlate_and_triangulate(
    cams,
    dets,
    max_speed_mps=50.0,
    calibration_is_exact=True,  # valid here only because geometry is synthetic
)

# Audio — one array yields DOA, not range; fuse only after independent ranging
from dataclasses import replace
from manwe.audio import detect_from_array
det = detect_from_array(signals, mic_positions, fs=16000)
det = replace(
    det,
    range_estimate=independently_measured_range_m,
    range_observed=True,
)
measurement = det.to_measurement(sensor_origin=array_xyz)

# Export — a raw receipt plus separately inspected signature are required
from manwe.export import (
    export_model, ExportReceipt, VerifiedArtifactSignature,
    build_export_contract, fidelity_report,
)
```

For a non-zero multi-camera target-speed bound, all timestamped detections in a
batch must have exactly the same capture timestamp and zero relative timestamp
uncertainty. Untimestamped batches additionally require the explicit
`simultaneous_capture=True` acknowledgement. This is a physical exposure-time
contract, not merely clock synchronization: asynchronous rays can intersect with
near-zero reprojection error at a badly biased depth, so isotropic speed/skew
covariance cannot make moving-target triangulation sound. Static-scene callers
may set `max_speed_mps=0` and opt into bounded `max_time_skew`.

The current triangulation covariance propagates pixel-localization uncertainty
only. It does not propagate uncertainty or systematic bias in camera intrinsics
or extrinsics: focal-length and baseline errors can preserve an exact
reprojection fit while biasing depth. Consequently, correlation also requires
the explicit `calibration_is_exact=True` acknowledgement, which is valid only
for analytically exact synthetic geometry. Estimated real-camera rigs remain
fail-closed until a calibration-parameter covariance model is implemented.

## Consumer integration status

No reviewed consumer is currently a zero-adaptation target. In particular,
crebain's native YOLO paths assume an 80-class COCO head while Manwe's candidate
airspace contract has five classes; artifact containers and preprocessing also
differ by backend. Galadriel, Engram/NCP, Prisoma, and pid-rs require additional
sequence, frame, identity, shape, or statistical adapters. See the
**[compatibility matrix and ten promotion gates](docs/INTEGRATION_CREBAIN.md)**.

## Benchmarks

`metal-yolo-tests/` contains bounded Candle/Metal static-image and video profiles.
Historical cross-backend numbers used incomparable timed scopes and insufficient
provenance; their runners were removed and the numbers must not rank backends. The
benchmark README defines the retained timing boundary and evidence requirements.

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — repo design & data flow
- **[docs/INTEGRATION_CREBAIN.md](docs/INTEGRATION_CREBAIN.md)** — audited consumer compatibility matrix
- **[docs/MODEL_CONTRACTS.md](docs/MODEL_CONTRACTS.md)** — candidate model manifests and fidelity limits
- **[docs/RELEASING.md](docs/RELEASING.md)** — alpha release checklist and registry policy
- **[docs/research/SOTA-2026.md](docs/research/SOTA-2026.md)** — dated, source-cited research with verification cautions
- **[CHANGELOG.md](CHANGELOG.md)** — alpha release notes and known limits
- **[SECURITY.md](SECURITY.md)** — reporting, trust boundaries, and credential response

## Technology stack

| Component | Technology |
|-----------|-----------|
| Training | PyTorch (MPS + CUDA), Ultralytics; model-family adapters are validated separately |
| Small-object | SAHI-style sliced inference; sliced training and a P2 head are planned |
| Acoustic | numpy DSP (GCC-PHAT / SRP-PHAT); deep SELD is planned |
| Fusion | numpy KF/EKF/UKF/PF/IMM, OSPA/GOSPA |
| Export | Raw ONNX, CoreML and TensorRT conversion; MLX conversion is not implemented |
| Inference (Rust) | [Candle](https://github.com/huggingface/candle) on CPU, Metal, or CUDA features |

## Use cases

Disaster-relief aerial coordination · urban delivery-drone deconfliction ·
infrastructure inspection · wildlife monitoring · traffic & airspace situational
awareness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The numpy-only core is testable without
model runtimes. Prepare the development environment before running the suite:
`make setup && make test`.

## License

The Manwe source is MIT — see [LICENSE](LICENSE). Model checkpoints, derived
weights, datasets, exporters, and runtime dependencies retain their own terms. A
locally trained derivative is not automatically MIT-redistributable. Record and
review the exact base checkpoint, training data, and toolchain rights before use;
see [docs/MODEL_CONTRACTS.md](docs/MODEL_CONTRACTS.md). The embedded annotation
font and adapted Candle example source retain their notices and license terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
