# manwe (Python training ground)

The Python package for Manwe: detector-training experiments, microphone-array
direction-of-arrival, multi-camera triangulation, evaluation, and an independent
multi-target fusion reference. Raw ONNX/CoreML/TensorRT conversion tools are
available through optional dependencies; MLX conversion is not implemented.

Crebain is an intended consumer, not a drop-in runtime. Its class heads, artifact
types, preprocessing and measurement schemas differ by backend. Galadriel,
Engram/NCP, Prisoma and pid-rs also require explicit sequence/frame/shape or
statistical adapters. Treat all generated artifacts as candidates until the
root-level compatibility gates pass.

The **core** (fusion, geometry, DOA, metrics, contracts) is pure-numpy and runs
without ML runtimes on the tested Linux/macOS hosts. Windows is not an alpha
release target because artifact/config publication currently relies on POSIX
descriptor-relative I/O. Training and raw conversion dependencies are lazy
optional extras; platform runtimes such as TensorRT are installed separately.

The distribution is named `manwe-perception`; the import package and CLI command
remain `manwe`. Alpha artifacts are built for source/GitHub release validation and
are not published under the unrelated `manwe` project on PyPI. That legacy project
also owns the `manwe` import and command, so do not co-install it; use the isolated
`uv` environment created by this checkout.

```bash
uv sync --locked --extra dev                    # core, CLI, tests, lint, and typing
uv sync --locked --extra vision --extra export # vetted local-only training/export adapters
uv sync --locked --extra rfdetr                # vetted RF-DETR 1.8 architecture construction
uv sync --locked --extra all --extra dev       # supported optional stack + tooling

uv run --no-sync manwe doctor      # hardware + installed extras
uv run --no-sync manwe fusion-sim  # synthetic KF/EKF/UKF/PF/IMM comparison
uv run --no-sync pytest tests      # core suite + config-contract dependency
```

The example YAML files under `configs/` are repository fixtures and are not wheel
package data. Source/GitHub quick starts assume a checkout; wheel users must pass
their own explicit local config and dataset manifest.

The checked-in trainers construct model architectures without pretrained weights.
Backend-managed downloads are disabled, and no digest-bound local-checkpoint
fine-tuning adapter is implemented yet. Training therefore starts from random
initialization; treat the resulting weights as experimental candidates.

Backend-managed model downloads are disabled. Runtime inference/export requires a
local artifact, exact SHA-256, and an explicit acknowledgement for pickle-backed
checkpoints. Ultralytics runtime installation, analytics, and network downloads
are disabled; dataset YAML is replaced with a private directive-free snapshot.
Raw export also requires an unoccupied caller-selected destination and returns an
`ExportReceipt`; a manifest requires separately inspected tensor-signature evidence.
TensorRT INT8 export additionally copies the bounded calibration tree into a
read-only private snapshot, binds that exact tree plus normalized manifest in the
receipt digest, and rechecks the caller-visible source before publication.

RF-DETR's upstream `train` extra currently installs both `opencv-python` and
`opencv-python-headless`, which overwrite the same `cv2` package. Manwe therefore
does not include that ambiguous extra in its lock; RF-DETR training requires a
separately curated environment with exactly one OpenCV distribution. The adapter
targets RF-DETR 1.8.3 or newer within the 1.8 release line.

Full documentation lives in the repository root: see `../README.md`,
`../docs/ARCHITECTURE.md`, `../docs/INTEGRATION_CREBAIN.md`, and
`../docs/MODEL_CONTRACTS.md`. The fidelity report emits threshold-specific
AP50/AP50-small (not COCO mAP@[.50:.95]) and also gates deployed-threshold
precision/recall/FPPI plus direct per-frame/class box and score agreement.
