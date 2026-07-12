# Model contracts

Manwe does not commit model weights. Raw conversion first emits an immutable
`ExportReceipt` containing source/artifact digests and exact conversion options.
A candidate `ModelContract` can then be built only with that receipt plus a
separate `VerifiedArtifactSignature` populated from backend inspection and golden
fixture evidence. `save_contract` writes the resulting sidecars exclusively.

The record is a **manifest and local validation boundary**. It does not alter a
model graph, compile an artifact, install a consumer adapter, or prove that
crebain (or another consumer) can execute it. No reviewed consumer automatically
negotiates or enforces this JSON today; see the
[compatibility matrix](INTEGRATION_CREBAIN.md).

## Required record

| Field | What it captures |
|---|---|
| Schema version | Contract parser/version boundary |
| Model name/version | Family, checkpoint and export version |
| Source and source SHA-256 | Training/checkpoint provenance and exact source identity |
| Rights | Intended use/redistribution review for code, base weights and derived weights |
| Backend and artifact path | Declared runtime plus exact file/bundle path |
| SHA-256 | Digest of a file or deterministic directory tree |
| Export options and signature evidence | Precision, opset, NMS/calibration choices plus the inspection/fixture reference |
| Number of classes and class map | Complete source-index coverage mapped to an allowed candidate label or explicit `DROP` |
| Input tensors | Names, shapes, dtypes, layouts, channel order and dynamic bounds |
| Output tensors | Names, shapes, dtypes, box/score semantics and coordinate convention |
| Preprocess/postprocess | Decode, resize, normalize, NMS, thresholds, scaling and max detections |
| Validation data | Immutable fixture manifest/digest and its rights |
| Benchmark context | Hardware, OS, runtime/provider, command, precision, thresholds and timed scope |
| Failure behavior | Missing, malformed, wrong-digest, wrong-extension and resource-limit behavior |

Schema 1.2 validation rejects missing values, incomplete class maps, invalid tensor
descriptions, wrong artifact suffix/type, empty or oversized artifacts, missing
artifacts, digest mismatches, and symlinks. CoreML bundles use a bounded,
deterministic directory-tree digest. Sidecars are first written, synced, and
verified inside an adjacent private `.manwe-contract-*.in-progress` directory,
then published with descriptor-relative no-replace hard links. An occupied path
is rejected before staging, while the link operations remain the authoritative
no-replace check. Once either final link exists, any failure preserves every final
pathname; failures detected before marker removal also preserve the staging marker
for manual recovery. Cleanup never check-then-unlinks a possibly replaced final
path. It proceeds only after revalidating the parent pathname identity,
signed-artifact digest, and original staged/final inode/content identities,
including one complete commit-boundary check immediately before marker removal.
`save_contract` returns success only after both durable final links are verified,
the marker is removed, and that removal is synced. If cleanup cannot be
authenticated, it raises and preserves the marker. If marker removal succeeds but
its parent fsync fails, it raises an **indeterminate commit** error: the current
namespace has no marker, but a crash may make it reappear.

POSIX provides no conditional unlink-by-inode operation. The absolute guarantee is
that Manwe never removes either final pathname; private-stage cleanup relies on its
high-entropy mode-0700 name plus immediate identity/content checks. A hostile
same-UID process actively racing those checks remains outside this boundary.

Manual recovery is intentionally conservative:

1. Stop concurrent writers and retain every final path and in-progress directory.
2. Compare any staged files still present, the final files, recorded digests, and
   signed artifact; a late failure may leave an empty marker, so never infer
   completeness from names alone.
3. Quarantine mismatches. Remove the marker and sync its parent only after both
   sidecars are verified, or re-sync and re-inspect the parent after an indeterminate
   marker-removal error.

The builder does not infer tensors from `family="yolo"` or a file extension. Raw
export receipts deliberately set `tensor_signature_verified=false`; a caller must
inspect the exact graph/runtime, record its real inputs/outputs and preprocessing,
and cite the fixture or inspection evidence before a candidate contract exists.

These checks establish manifest integrity, not semantic compatibility. A valid
five-class ONNX contract remains incompatible with a consumer hard-coded for an
80-class head.

## Backend status

| Backend | Manwe output/status | Consumer work still required |
|---|---|---|
| ONNX | Raw conversion is available into a caller-owned, previously absent destination | Inspect opset/providers, tensor names/layout, class count, dynamic shapes, preprocessing and NMS; run consumer fixtures. |
| CoreML | Raw conversion normally yields `.mlpackage` | Compile to `.mlmodelc` where required; pin feature names, compute units, image transform, labels and Vision output contract. |
| TensorRT | Ultralytics can produce an engine on a supported NVIDIA environment | Pin GPU/driver/TensorRT compatibility and calibration. A consumer that loads ONNX through the TensorRT execution provider does not thereby accept `.engine`. |
| MLX | The manifest type recognizes `.safetensors`, but Manwe has no implemented MLX converter | A generic safetensors file is not executable; graph architecture, key names/shapes, class head and preprocessing must match a specific loader. |

Export conversion must be treated as untrusted output until every selected backend
passes the same semantic fixture set. Never infer support from a successful file
write or extension.

## Fidelity gate: AP50 plus deployed-output agreement

`manwe.export.fidelity_report` compares exported detections with the FP32 reference
against shared ground truth. `Detections.boxes` and `GroundTruth.boxes` are
positive-area `xyxy` coordinates in **source-image pixels**, after each backend's
inverse resize/letterbox transform. Model-canvas or normalized boxes must be
converted before evaluation.

The report includes:

- macro **AP50** at one IoU threshold (`0.50`);
- a simplified **AP50-small** view based on pixel area;
- per-class AP drops, with an absolute default tolerance of `0.005` (0.5
  percentage points);
- precision, recall, and false positives per image (**FPPI**) at the detections'
  deployed confidence threshold;
- direct one-to-one, same-class reference/export agreement for every frame: by
  default paired boxes need IoU ≥ `0.95`, no reference detection may be missing,
  no exported detection may be extra, and each paired score delta must be ≤ `0.05`.

`passed` requires all of those gates. The operating-point precision/recall drop
tolerance defaults to the AP tolerance; added FPPI defaults to zero tolerance.
Callers may configure them explicitly. This closes an AP interpolation blind spot:
false positives after full recall can leave AP unchanged but will increase FPPI
and appear as extra exported detections.

The historical `ref_map`/`exp_map` field names are retained for API compatibility,
but the values are not COCO `mAP@[.50:.95]`. The small-object calculation filters by
area and does not implement all COCO ignore/crowd/max-detection rules. Use
`pycocotools` with a pinned dataset protocol for publishable or comparable COCO
metrics.

Frame order must not be an implicit assumption. The fidelity gate requires a
unique, non-empty `image_id` on every reference, exported and ground-truth frame;
it rejects duplicates and misaligned sequences. The lower-level
`mean_average_precision` helper retains positional alignment only when callers
omit every ID, so it must not be used as a promotion boundary without a separate
identity contract.

Use `required_classes` and `required_small_classes` to name the operational
coverage that must exist in ground truth. The report always rejects an evaluation
with no measured classes or no small-object classes, but it cannot infer which
classes the deployment promises; omit those arguments only for a deliberately
bounded experiment.

The fidelity report is necessary but not sufficient. It still needs representative
positive and negative frames, the exact deployed confidence threshold and consumer
pre/postprocess. Add a full COCO-style sweep, calibration, boundary cases and
consumer end-to-end fixtures before promotion.

## Minimum acceptance before trusting detections

1. Artifact path, type, digest and directory-tree rules validate.
2. Consumer load fails closed on the wrong graph, tensor names, rank, class count,
   precision and dynamic dimensions.
3. Golden preprocessing pixels and raw tensors match the consumer path; all metric
   boxes have been transformed back to source-image-pixel `xyxy`.
4. Every class and no-target/irrelevant-target fixtures produce the expected raw
   and postprocessed outputs within documented tolerance.
5. Box conversion, threshold boundaries, class-aware NMS and maximum detections
   match on extreme aspect ratios and overlapping boxes.
6. Unique frame IDs align reference/export/ground truth, required class and
   small-class coverage exists, and per-class AP50/AP50-small drops pass.
7. Deployed-threshold precision/recall/FPPI and direct same-class box/score
   agreement pass; trailing false positives cannot hide behind interpolated AP.
8. Latency is measured only after accuracy parity and records exact timing scope,
   warm-up, synchronization, hardware/runtime and repeated-run variance.
9. Missing/malformed/oversized artifacts and tensors fail without partial trust,
   path traversal, uncontrolled allocation or secret leakage.
10. The adapter/consumer versions and rollback artifact are pinned; rights review
    covers the exact base checkpoint, training data, generated artifact and
    runtime—not just this repository's source license.

## Licensing boundary

The Manwe source is MIT. That does not relicense any model checkpoint, fine-tuned
derivative, dataset or optional dependency. Licenses can vary by model family,
checkpoint tier, release date, use case and commercial agreement. Dataset terms
may also restrict redistribution of learned weights or validation fixtures.

Do not write `rights="weights self-produced; MIT"` merely because training ran
locally. Instead record verifiable facts, for example:

```text
base checkpoint: <name, version, source URL, digest, upstream license snapshot>
training datasets: <names, versions, terms, access/redistribution review>
derived artifact: <intended use and legal-review reference>
export toolchain: <versions and relevant licenses>
```

Treat the `rights` field as an auditable record, not a legal conclusion. Re-check
upstream terms before each redistribution or production use.
