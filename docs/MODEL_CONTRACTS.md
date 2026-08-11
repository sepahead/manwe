# Model contracts

Manwe schema 2.0 is an executable model-package contract, not a descriptive model
card. It binds one portable sibling artifact name and digest to a closed tensor,
preprocessing, decoding, taxonomy, and postprocessing policy.

The native Rust CLI and viewer consume schema 2 directly for the reviewed Candle
adapters. Other backends use the same record as producer evidence, but still need
an implemented runtime or downstream adapter.

## What a contract proves

A validated contract proves internal consistency. When loaded with artifact
verification, it also proves that the sibling artifact bytes match the declared
SHA-256. For a supported native adapter, validation checks the exact model variant,
input geometry, YOLOv8 output grid, class head, dtype, resource bounds, and
postprocessing policy before inference.

It does not prove:

- who authored or approved the contract;
- model accuracy, calibration, fairness, rights, or fitness for a domain;
- that an ONNX/CoreML/TensorRT artifact works in an unimplemented consumer;
- numerical parity across CPU, Metal, CUDA, or another framework; or
- compatibility with a downstream wire, clock, frame, or lifecycle contract.

The `signature_evidence` field records backend tensor-interface inspection. It is
not a digital signature. Establish contract provenance through a trusted release,
digest registry, signature/attestation system, or another operator-controlled
channel.

## Production sequence

```text
exact checkpoint
      │ raw conversion
      ▼
ExportReceipt
  source/artifact digests + options
      │ independent backend inspection and golden tensor fixture
      ▼
VerifiedArtifactSignature
  tensors + RuntimeSpec + evidence reference
      │ build_export_contract
      ▼
ModelContract 2.0
      │ save_contract (exclusive, durable sidecars)
      ▼
artifact + .contract.json + .contract.md
```

Raw conversion deliberately stops at `ExportReceipt`. The builder never guesses
an interface from “YOLO,” a suffix, or a model name. Python checkpoint inference
is a separate research wrapper with its own digest and pickle-admission boundary;
it does not pretend to execute the native schema-2 adapter.

## Schema 2.0 record

| Area | Required fields and invariant |
|---|---|
| Identity | `schema_version`, model name/version, source, rights |
| Artifact | backend, portable sibling `file_path`, lowercase SHA-256 |
| Provenance | source SHA-256, exact export options, interface-evidence reference |
| Taxonomy | class count, unique source classes, complete index-keyed `class_map` |
| Tensors | bounded unique input/output names, concrete or canonical dimensions, dtype/layout |
| Runtime | closed adapter ID plus typed image-input and detection-output policy |
| Evidence | validation data, benchmark context, failure behavior |
| Human context | matching preprocess/postprocess summaries; machine behavior comes from `runtime` |

`file_path` is one ASCII basename, resolved beside the JSON contract. Absolute
paths, traversal, nested paths, Unicode lookalikes, wrong backend suffixes, links,
empty files, and digest mismatches are rejected.

Every source class index appears exactly once in `class_map`. Its value is one of
`drone`, `bird`, `aircraft`, `helicopter`, `unknown`, or `null`. Native structured
results always retain the source index/name and add the mapped `airspace_class`;
`null` remains explicit rather than silently inventing or deleting a source
result.

### Typed image input

`runtime.input` pins tensor name, width/height, dtype, `NCHW/RGB`, RGB order,
letterbox policy, interpolation, stride alignment, pad value, scale, mean, and
standard deviation. The Candle v1 adapters require square 32-aligned FP32 input,
Catmull–Rom resize, pad 114, and exactly `1/255` scaling with zero mean/unit
standard deviation.

### Typed detection output

`runtime.output` pins tensor name, `cxcywh`, input-pixel coordinates, probability
scores, confidence threshold, class-aware NMS IoU, maximum detections, and—only
for pose—the keypoint threshold. The tensor record pins the exact raw shape and
dtype. Native adapters apply the three probability/IoU thresholds after
round-to-nearest conversion to IEEE-754 binary32, matching their FP32 model
outputs; the JSON number is not an independent higher-precision decision rule.

### Implemented adapters

| Adapter | Native status | Closed constraints |
|---|---|---|
| `manwe-candle-yolov8-detect-v1` | Implemented by shared Rust runtime | variants n/s/m/l/x; 1–1,000 classes; FP32 `[1, 4+C, N]`; `N` equals stride 8/16/32 grid sum |
| `manwe-candle-yolov8-pose-v1` | Implemented by shared Rust runtime | one source class; canonical COCO-17 `(x,y,confidence)` order; FP32 `[1, 5+51, N]` |
| `ultralytics-raw-detect-v1` | Python export evidence adapter | static raw detect tensor; no native Rust execution claim |

All adapters have bounded JSON, tensor rank/dimensions, predictions, output
elements, classes, detections, model bytes, and text. Unknown fields and duplicate
JSON keys fail closed in both Python and Rust.

## Using a contract

Validate a sidecar and its sibling artifact from Python:

```bash
manwe contract /models/model.contract.json
manwe contract /models/model.contract.json --json
```

Metadata-only inspection is explicit and weaker:

```bash
manwe contract /models/model.contract.json --no-verify-artifact
```

Execute a supported Candle package:

```bash
manwe --contract /models/model.contract.json --output-dir /outputs image.jpg
camera_view --contract /models/model.contract.json --url rtsp://camera.example/live
```

The packaged file
`manwe/schemas/model-contract-v2.candle-detect.example.json` is a schema fixture
with placeholder digests and evidence. It is intentionally non-executable. Replace
every model-specific and evidence field from actual artifact inspection; do not
treat the fixture as an attestation or a converter.

No checked-in converter currently produces the repository’s Candle YOLOv8 key
layout. A generic safetensors file—including an Ultralytics checkpoint—does not
become compatible because its extension matches.

## Native inference receipt

The Rust batch CLI writes one JSON object per input image after the annotated JPEG
is durably published. The receipt includes:

- result schema, detect/pose task, and selected device;
- model name/version;
- contract path and digest;
- artifact path and digest;
- input path and digest;
- output path and digest; and
- one `objects` payload tagged as `detections` or `poses` with its typed items.

Contract and artifact paths are the last-authenticated package paths; input paths
are canonical display labels resolved before the bounded read. A later rename can
make any of them stale. Their digests—not a subsequent lookup by pathname—identify
the bytes used by the run.

Objects are ordered deterministically by descending confidence with stable source
prediction tie-breaking. Coordinates are finite source-image-pixel `xyxy`; pose
keypoints use contract names and source pixels. Publication never replaces an
existing path. The recommended `--output-dir` boundary lets input storage remain
untrusted or read-only; it must already exist and be owner-controlled. Without the
option, output is published beside each input under the same ownership rule.
Publication is atomic per image, not across the entire argument list: a failure
on a later input does not invalidate or remove earlier outputs and receipts.

## Sidecar publication and recovery

`save_contract` snapshots mutable Python objects through canonical JSON, verifies
the exact sibling artifact again, stages JSON and Markdown in an adjacent private
directory, and publishes both with no-replace hard links. Success means both
finals are verified and durable and marker removal is synced.

The artifact parent must already exist and permit a mode/ACL trust proof. A
group/world-writable parent is accepted only when it is sticky and owned by the
effective account or root; an unmodeled access-control ACL is rejected before a
stage is created. This excludes replacement by a different ordinary account.
Same-UID and privileged mutation remain outside the POSIX pathname boundary.

Once a final link exists, an error preserves final paths and available staging
evidence. Recovery is conservative:

1. stop concurrent writers;
2. retain every final and `.manwe-contract-*.in-progress` entry;
3. compare artifact, JSON, Markdown, and recorded digests;
4. quarantine any mismatch; and
5. remove an authenticated marker only after the final state is understood and
   its parent can be synced.

Names alone never establish commit state.

## Backend status

| Backend | Manwe status | Remaining promotion work |
|---|---|---|
| Candle safetensors | Contract-bound detect/pose reference runtime | exact compatible weights, golden forward, CPU/accelerator parity, domain evaluation |
| ONNX | Raw conversion and schema-2 evidence | provider/opset fixture, exact tensors/preprocess/NMS, consumer adapter |
| CoreML | Raw `.mlpackage` conversion and evidence | compile where required; feature names, compute units, Vision output fixture |
| TensorRT | Raw engine conversion on supported NVIDIA systems | GPU/driver/runtime compatibility, calibration inspection, fidelity, consumer path |
| MLX | Contract type recognizes `.safetensors`; no converter/runtime | implement and validate a concrete graph; suffix alone is insufficient |

## Fidelity gate

`manwe.export.fidelity_report` compares exported detections with an FP32 reference
in source-image-pixel `xyxy`, aligned by unique image ID. It gates:

- macro AP50 and a documented simplified AP50-small view;
- per-class AP drops;
- deployed-threshold precision, recall, and FPPI; and
- direct one-to-one same-class box/score agreement for every frame.

By default, paired boxes require IoU at least 0.95, score delta at most 0.05, and
no missing or extra detections. `passed` requires all configured gates and required
class/small-class coverage. Historical `ref_map`/`exp_map` names are retained for
compatibility, but they are not COCO `mAP@[.50:.95]`; use a pinned pycocotools
protocol for publishable COCO metrics.

## Minimum promotion gates

1. Verify artifact type, exact digest, contract provenance, and rights.
2. Reject wrong graph, tensor names/ranks/shapes/dtypes, class count, and resource bounds.
3. Match golden preprocessing pixels and raw tensors on the actual runtime.
4. Exercise every class plus no-target and irrelevant-target fixtures.
5. Match inverse letterbox, clipping, threshold boundaries, class-aware NMS, and maximum detections.
6. Pass aligned AP50/AP50-small, operating-point, and direct agreement gates.
7. Measure latency only after fidelity, with warm-up, synchronization, timing scope, hardware, and variance recorded.
8. Prove malformed/missing/oversized inputs fail without partial trust or uncontrolled work.
9. Pin adapter and consumer versions and retain a rollback package.
10. Re-review exact checkpoint, dataset, derived-weight, and runtime rights before redistribution or production use.

The source license does not relicense weights or data. Treat `rights` as an
auditable factual record, not a legal conclusion.
