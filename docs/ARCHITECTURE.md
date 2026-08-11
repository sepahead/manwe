# Manwe architecture

Manwe is a perception research and validation workbench, not a monolithic
deployment service. Its architecture deliberately separates four boundaries that
fail for different reasons:

1. the Python research plane trains, evaluates, exports, and simulates;
2. a schema-2 model package binds one artifact to an executable interface;
3. the Rust native plane executes only the Candle adapters it implements; and
4. downstream systems require their own versioned adapters and fixtures.

Keeping those boundaries explicit prevents a successful conversion, a familiar
file suffix, or a similar class name from becoming an accidental compatibility
claim.

## Architectural invariants

### One authority per native model

The native CLI and camera viewer accept one `--contract` path. They do not accept
independent model-family, task, class-count, image-size, threshold, or digest
flags. The validated schema-2 contract is the sole authority for:

- the sibling artifact filename and SHA-256;
- Candle model variant and detect/pose adapter;
- exact input and output tensor shapes and dtypes;
- image color, letterbox, interpolation, padding, scale, and alignment;
- source classes and their optional Manwe airspace mapping;
- confidence, NMS, keypoint, and maximum-result policy.

`src/native_runtime.rs` is the only native load/inference pipeline. Both entry
points reuse it, so model construction and postprocessing cannot drift between
batch and live execution. An accepted native contract also proves the YOLOv8
three-grid prediction count implied by its input size and stays within the model,
tensor, and postprocessing work bounds.

The live viewer separates capture from inference. One worker owns the runtime and
selects fairly from one replaceable latest-frame slot per stream. Capture never
waits behind model execution and queued work cannot grow. Raw frames are displayed
during warm-up; after the first result, each view holds the newest completed exact
annotated sample and advances monotonically as later inferences finish. Superseded
queued samples are dropped, but a completed sample is not relabeled as current:
viewer latency still depends on inference time and the number of streams. This is
an annotated inference view, not an independent zero-latency raw monitor.
Frame-producing sessions reset reconnect backoff; worker panics become a failing
application exit and coordinated shutdown rather than a silent loss of inference
or capture.

The contract integrity-binds bytes; it does not authenticate who authored the
contract. Operators must establish contract provenance separately.

### Admission is not consumption

Dataset validation is an admission-time observation of caller-owned storage. It
rejects remote directives, escape paths, nested links, special entries, overlap,
and within/across-split hardlink aliases under aggregate work limits. It cannot
make a mutable filesystem immutable after return.

Training therefore revalidates at the consumption boundary and gives the backend
a private read-only copy plus a normalized manifest. Split identity is checked
against the exact descriptor inventory from which that copy is made, then the
copier proves the inventory stayed unchanged. The private copy's backend-visible
images must then pass bounded single-frame header, suffix/content, mode, and EXIF
checks before a trainer is constructed. Ultralytics label selection is mirrored
and admitted as bounded UTF-8 detection rows with finite classes and image-contained normalized
positive boxes; label aliases are rejected. RF-DETR additionally binds COCO image
dimensions and box extents to those real headers. TensorRT calibration goes
further: it validates complete decoded tensor semantics and exposes the exact
hash-ranked backend set through a private loader view. The source tree is never
treated as a stable long-running backend input merely because its YAML once
passed validation.

### Raw export is not a model package

Python conversion produces an immutable `ExportReceipt`: source digest, artifact
digest, and exact conversion options. A schema-2 `ModelContract` requires a
separate `VerifiedArtifactSignature` produced from backend inspection. Here,
“signature” means tensor-interface evidence, not a cryptographic signature.

The builder refuses to infer tensors or runtime semantics from a family name or
extension. The native Rust plane consumes only the two reviewed Candle adapters;
ONNX, CoreML, and TensorRT contracts remain evidence records until a corresponding
runtime or downstream adapter validates them.

### Structured results preserve both taxonomies

Native inference emits one JSON object per input image. Each retained object keeps
its source class index/name and, when declared, an `airspace_class` mapping. A
missing mapping is `null`, not an inferred class and not a silently dropped source
result. Detect and pose payloads are tagged, so a consumer cannot confuse their
shapes.

Python checkpoint inference applies the same ownership principle at its smaller
research boundary. Direct and sliced paths admit one bounded image, authenticate
the checkpoint digest and detection metadata around the backend call, and convert
all accepted output into immutable Manwe `Detection` records before releasing the
private artifact snapshot. Evaluation frames likewise own immutable array copies,
so later caller or backend mutation cannot rewrite measured evidence.

### Claims stop at measured boundaries

Device selection is capability routing, not numerical-equivalence evidence.
Export success is file production, not fidelity. A valid package is native-runtime
compatibility for its declared adapter, not downstream compatibility. Promotion
still requires representative fixtures, accuracy/agreement gates, operational
failure tests, and a consumer-owned adapter.

## System flow

```text
caller-owned dataset
        │ validate + revalidate
        ▼
private training snapshot ──▶ Python trainer ──▶ backend-owned candidate checkpoint
                                                     │ hash exact bytes explicitly
                                            raw backend conversion
                                                     ▼
                                              ExportReceipt
                                                     │
                                      backend tensor inspection
                                                     ▼
                              schema-2 contract + sibling artifact
                                      │                    │
                           Candle adapter only      other backends
                                      ▼                    ▼
                           shared NativeRuntime     evidence candidate
                              │            │                │
                         JSONL CLI    live viewer     downstream adapter
                              └────────────┴───────────────┘
                                             │
                                fixtures + fidelity + failure gates
                                             ▼
                                       consumer input
```

Audio, multi-camera geometry, and fusion are independent reference paths. They
produce local typed objects, not an implicit downstream wire protocol:

```text
microphones ─▶ DOA/ranging boundary ─┐
cameras ─────▶ calibrated geometry ──┼─▶ local measurements ─▶ fusion reference
radar/events ────────────────────────┘
```

An adapter must still define frames, units, clocks, identity, missingness, and
covariance semantics. Similar array shapes are not a contract.

## Repository layout

```text
manwe/
├── python/src/manwe/
│   ├── common/          # contracts, artifact/dataset boundaries, devices
│   ├── schemas/         # packaged, non-executable schema-2 examples
│   ├── vision/          # model registry, private-snapshot training, inference
│   ├── export/          # raw conversion, receipts, contract builder, fidelity
│   ├── audio/           # GCC-PHAT/SRP-PHAT and acoustic measurements
│   ├── multicam/        # camera geometry and triangulation
│   ├── fusion/          # KF/EKF/UKF/PF/IMM tracker and metrics
│   ├── data/, eval/     # datasets, synthetic fixtures, evaluation
│   └── cli.py
├── src/
│   ├── runtime_contract.rs  # bounded schema-2 native admission
│   ├── native_runtime.rs    # shared model load and inference pipeline
│   ├── model.rs             # Candle YOLOv8 detect/pose graph
│   ├── lib.rs               # preprocessing, validation, NMS, reports
│   ├── main.rs              # JSONL batch CLI + durable image publication
│   └── bin/                 # experimental viewer and credential-safe launcher
├── metal-yolo-tests/        # bounded profiling tools, not a parity benchmark
└── docs/
```

## Platform and numerical boundaries

- The NumPy core and contract layer are lightweight; heavy Python runtimes remain
  optional. POSIX descriptor-relative protections make Windows a fail-closed,
  unsupported alpha target for affected workflows.
- CUDA is the intended large-training path and Apple MPS is a development path;
  neither is promoted without exact-build operator and numerical evidence.
- Rust supports CPU, optional Metal, and optional CUDA compilation. CI exercises
  Linux CPU and macOS Metal/viewer; CUDA execution and CPU/Metal golden-forward
  parity still require a pinned real model fixture.
- Multi-camera covariance is conditional on analytically exact calibration.
  Estimated intrinsic/extrinsic uncertainty is not yet propagated, so real rigs
  must not reinterpret pixel-only covariance as complete state uncertainty.
- Fusion is an independent event-indexed reference, not a numerical twin of a
  downstream tracker.

See [MODEL_CONTRACTS.md](MODEL_CONTRACTS.md) for schema semantics and
[INTEGRATION_CREBAIN.md](INTEGRATION_CREBAIN.md) for the downstream boundary.
