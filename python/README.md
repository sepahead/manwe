# Manwe Python training ground

The `manwe-perception` distribution contains Manwe’s detector experiments,
microphone-array direction-of-arrival, multi-camera geometry, independent
multi-target fusion reference, evaluation, raw conversion, and schema-2 contract
tooling. The import package and command are both `manwe`.

The numerical core uses NumPy. Heavy ML runtimes are optional and imported only
at their execution boundaries. Raw ONNX, CoreML, and TensorRT conversion is
implemented; MLX conversion is not.

## Environment

```bash
uv sync --locked --extra dev
uv sync --locked --extra vision --extra export
uv sync --locked --extra rfdetr
uv sync --locked --extra all --extra dev

uv run --locked --no-sync -- .venv/bin/manwe doctor
uv run --locked --no-sync -- .venv/bin/manwe fusion-sim
PYTHONWARNINGS=error uv run --locked --no-sync -- \
  .venv/bin/python -m pytest tests
```

Python 3.10–3.14 is supported for the core. Python 3.11–3.12 is the conservative
heavy-runtime path. Windows is not an alpha release target for workflows that
require POSIX descriptor-relative file guarantees.

The YAML files under `configs/` are source-checkout/sdist fixtures, not wheel
package data. Wheel users provide their own local config and dataset manifest.
The wheel does ship the non-executable schema example under `manwe.schemas`.

## CLI

```bash
manwe doctor
manwe models --track accuracy
manwe data
manwe synth /tmp/manwe-smoke
manwe fusion-sim
manwe vision-train configs/vision/aerial.yaml

manwe export /abs/best.pt -f onnx \
  --weights-sha256 <64-hex> --allow-pickle-checkpoint \
  --output /abs/candidate.onnx --allow-unverified

manwe contract /abs/candidate.contract.json
manwe contract /abs/candidate.contract.json --json
```

`--allow-pickle-checkpoint` acknowledges the code-execution risk of loading the
exact digest-bound checkpoint. `--allow-unverified` acknowledges that conversion
only produced raw bytes. A trusted model handoff still needs backend inspection,
`VerifiedArtifactSignature`, schema-2 contract construction, fidelity evidence,
and the target runtime/consumer fixture.

Here “signature” is tensor-interface evidence, not cryptographic authorship.

## Training data boundary

Detection manifests are local-only and directive-free. Admission rejects escape
paths, nested symlinks, special entries, overlapping roots, and duplicate/hardlink
file identities within or across train/validation/test selections under aggregate
entry and byte limits.

Admission does not freeze caller-owned storage. Immediately before training,
Manwe repeats validation, copies the bounded tree into a private read-only
snapshot while checking split identities against the copier's exact descriptor
inventory, writes a normalized private manifest, and gives only those paths to
Ultralytics. Every backend-visible still image is authenticated as single-frame,
identity-orientation, suffix/content-matched input with bounded encoded and real
decoded dimensions before model construction. Present Ultralytics labels must be
stable, bounded UTF-8 five-column detection rows with finite classes and
normalized positive boxes contained by the image, and its pinned image-to-label mapping must be
one-to-one. RF-DETR receives an equivalent private directory snapshot and must
match its COCO dimensions and boxes to the actual image headers. Both private trees are re-authenticated after the backend
returns.

TensorRT INT8 calibration additionally validates the exact decoded `val` tensor
semantics. It requires at least 1,000 unique effective images and exposes a
deterministic hash-ranked 512-image private view with cache, URL, optional-codec,
label, and adjacent-array side channels disabled. The digest binds loader policy,
image size, normalized manifest, validated inventory, exact tree, and audited
TensorRT/ModelOpt route. This establishes reproducibility and work bounds, not
domain representativeness or proof that every engine layer is INT8.

Privileged mount changes, SHA-256 collisions, or a hostile same-account process
mutating Manwe’s own private tree still require an isolated worker or stronger OS
containment.

## Model boundary

The checked-in trainers construct architectures from random initialization.
Backend-managed downloads are disabled, and no digest-bound local-checkpoint
fine-tuning adapter is implemented. Generated weights are experimental candidates.
Trainer output directories are backend-owned working state rather than an atomic
Manwe publication surface; authenticate the exact selected checkpoint by SHA-256
before passing it to inference or export.

Python checkpoint inference is a digest-bound research wrapper. It is distinct
from the native schema-2 runtime. Direct and sliced inference validate the
checkpoint's detection task and complete class table around execution, reject
out-of-image or malformed backend output, and return owned immutable Manwe
`Detection` records rather than backend-owned result objects. A schema-2 contract
binds a portable sibling artifact name and digest to exact tensors, typed
preprocessing/decoding, source classes, optional Manwe airspace mapping, and
postprocessing.

The Rust CLI/viewer execute only the reviewed Candle adapters. Python can produce
contracts for raw ONNX/CoreML/TensorRT evidence, but those records do not create an
unimplemented runtime. The packaged Candle JSON is a placeholder template; no
checked-in converter produces the required Candle weight-key layout.

Contract JSON/Markdown publication is exclusive and durable. A failure after a
final hard link appears preserves final paths and available staging evidence;
manual recovery must compare bytes and digests rather than infer state from names.
The artifact parent must pass the same mode/ACL publication policy as raw export:
shared writable parents are accepted only when sticky and owned by the effective
account or root.

## Known runtime boundary

RF-DETR’s upstream `train` extra installs competing OpenCV distributions. Manwe
therefore pins RF-DETR 1.8.3 without that extra. Training requires a separately
curated environment with exactly one OpenCV owner. CI validates the installed API
and Manwe’s argument mapping; it does not qualify a real RF-DETR training run.

Multi-camera covariance is conditional on analytically exact camera calibration.
The current implementation does not propagate intrinsic/extrinsic parameter
uncertainty, so an estimated real rig must not treat pixel-only covariance as a
complete measurement covariance.

See the root [architecture](../docs/ARCHITECTURE.md),
[model-contract guide](../docs/MODEL_CONTRACTS.md), and
[consumer compatibility audit](../docs/INTEGRATION_CREBAIN.md).
