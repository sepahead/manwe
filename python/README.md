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

Multi-camera position covariance currently propagates detection-pixel
localization noise conditional on analytically exact camera intrinsics and
extrinsics. It does not model calibration-parameter uncertainty or bias. The
correlation boundary therefore requires `calibration_is_exact=True`; real
estimated rigs fail closed until that covariance is supported.

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
read-only private snapshot. The ≥1,000-image floor applies to unique, fully
decodable 3-channel `uint8` `val` tensors after the pinned resize/letterbox path,
rather than filename suffixes or unrelated splits. Candidates require matching
suffix/content, identity EXIF orientation, and bounded encoded/decoded work. The
pinned backend receives a deterministic hash-ranked 512-image,
label/cache/adjacent-array-free private view with batch 1 and fraction 1.0.
Ultralytics cache I/O and network URL probes are disabled, and its process-global
`bgr=0` validation formatting is serialized and deterministic only during
calibration. This makes the consumed set stable across the legacy TensorRT
calibrator and TensorRT 11 ModelOpt. Export fails if
TensorRT reports the capability state that makes Ultralytics silently build FP32.
The receipt digest binds the TensorRT version/route, loader policy, `imgsz`,
validated inventory, exact tree, and normalized source manifest. The caller-visible
manifest file and its declared source tree are pinned and rechecked through
descriptor-relative POSIX I/O before publication. Dataset roots and splits must be
absolute or descendant-relative (`..`, symlinks, and special path components are
rejected). This
is a technical hygiene contract, not
proof that the images represent the deployment domain or that every engine layer
executes in INT8; `precision="int8"` records the requested mixed-precision route.
Retain separate dataset provenance, engine-inspector, and fidelity evidence.
TensorRT 11 preflights locally installed `nvidia-modelopt>=0.44`, binds its exact
version into the digest, and rejects image sizes whose conservative 10×
tensor-materialization estimate exceeds 8 GiB. That version and route are not
yet explicit receipt fields, so retain the build environment record.
The descriptor boundary is not an operating-system filesystem snapshot:
privileged mount changes, SHA-256 collisions, and malicious same-UID mutation of
the process-owned private loader tree during backend reads require an isolated
build worker or stronger OS containment.
Contract JSON/Markdown sidecars use an adjacent private
`.manwe-contract-*.in-progress` stage and no-replace hard links. If publication
fails after either final link appears, Manwe deliberately preserves the final paths;
failures detected before marker removal also preserve the marker for manual recovery.
A successful return requires both final links to be durable and verified and the
private marker removal to be synced. Authenticated cleanup failure preserves the
marker and reports an error; its commit check revalidates the parent path, signed
artifact, and both final sidecars immediately before removal. A parent-fsync failure
after marker removal reports an indeterminate commit instead: finals are retained,
the marker is currently absent, and a crash may restore it. Manwe never cleans final
pathnames; active same-UID races against private-stage cleanup are outside the POSIX
boundary.

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
