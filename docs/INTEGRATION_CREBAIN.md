# Consumer integration status and required adapters

Conservative source-contract review dated 2026-07-11. The Manwe side becomes the
commit containing this document. Several local consumer checkouts had uncommitted
changes, so their observations are useful for finding gaps but are not release
evidence. Re-audit whenever any producer, adapter, or consumer contract changes.

Only committed consumer content is an anchor. Uncommitted working-tree changes
were deliberately excluded because a dirty-path count neither identifies their
bytes nor makes the review reproducible.

| Checkout | Commit anchor for clean re-audit | Boundary to re-inspect |
|---|---|---|
| crebain | `5be4f0e0415b6384306a08bf8917939f7baca219` | native/browser detector, fusion, and NCP v0.7 dependency declarations |
| Galadriel | `c4b0b3688d8fd5e3978512744525f6f2d1c5730d` | `PidObservation` and sequence/scientific gates |
| Engram | `8ca78063fb657a3d73fde7446941eee31032419b` | vendored NCP wire-0.6 manifest and gateway boundaries |
| Prisoma | `aca27af4cc97d8b2092a7ef04f74f07f40e7cd73` | VLDA/run-log boundaries and NCP v0.7 observer dependency declarations |
| pid-rs | `65139549930a0cbb7ddcad01394c897b4a3b10a4` | estimator and Python-binding array contracts |
| standalone NCP | `e3e5da4de96e8b291b3c582bd31cf41afbfad3cc` | released wire-0.7 manifest and integration rules |

This was source inspection, not a consumer build or end-to-end execution campaign.
The commit objects identify reproducible baselines; they do not encode or bless
the uncommitted files seen during review. Any adapter must start from a clean
checkout of its selected commit or immutable tag, repeat the source audit, and run
its fixtures against those exact bytes. Newer local files do not silently update
these conclusions.

## Status: no drop-in integration

No reviewed consumer imports Manwe or consumes a Manwe artifact/measurement
contract directly. `manwe.common.contracts` is a useful **producer-side manifest**,
not a negotiated wire protocol and not evidence that a consumer can execute the
artifact. All integration paths below need a versioned adapter plus consumer-owned
fixtures.

Status terms used here:

- **candidate** — Manwe can produce relevant information, but no reviewed adapter
  proves the consumer contract;
- **blocked** — a known semantic or shape mismatch prevents direct use;
- **research-only** — data may support a bounded experiment, not an operational or
  scientific conclusion without additional controls.

## Compatibility matrix

| Consumer | Consumer requires | What Manwe has now | Blocking gap | Status |
|---|---|---|---|---|
| **crebain: native ONNX / CUDA / TensorRT EP** | Ultralytics YOLOv8 tensors with exactly 84 features (`4 + 80 COCO`), known layout, native preprocessing and NMS | Rust COCO model path can produce an 80-class YOLO head; Python aerial contracts target five classes and typically produce `4 + 5` features | The five-class head is rejected by the fixed 84-feature parser. The COCO path still needs identical resize/padding, tensor names, score semantics, NMS, and fixture parity. TensorRT runtime discovers `.onnx`; a standalone `.engine` is not the same runtime input. | **blocked** for five-class models; **candidate** for a pinned COCO ONNX adapter |
| **crebain: native CoreML** | Compiled `.mlmodelc`, Vision/CoreML feature names and output type, accepted labels and coordinate conversion | Raw export normally produces `.mlpackage`; contract allows both bundle suffixes | Compilation is an extra step. The native path maps labels through its current COCO table and can drop custom labels. Manwe does not emit or test the exact Vision result contract. | **blocked** |
| **crebain: browser “CoreML” path** | An ONNX file loaded by ONNX Runtime Web, configured preprocessing, output format/tensor names, and class mapping | ONNX conversion plus a five-class manifest | The path is not a native CoreML consumer and does not ingest Manwe's manifest. A config adapter and golden browser fixtures are required. | **candidate** |
| **crebain: MLX** | A bespoke YOLOv8-style Candle graph, exact safetensor keys/shapes, and an 80-class COCO head | No implemented Manwe MLX converter; a generic `.safetensors` suffix is allowed by the manifest type | File format alone is insufficient. The consumer architecture is fixed and is not a generic Ultralytics/Manwe loader. | **blocked** |
| **crebain: RF-DETR / DETR-style models** | Consumer-specific boxes/logits tensor names, layout, preprocessing and postprocessing | Model registry/training experiments and backend conversion utilities | DETR's separate logits/boxes are not the combined YOLO tensor contract. No end-to-end exported fixture is pinned. | **blocked** |
| **crebain: sensor fusion** | `SensorMeasurement` with sensor ID, modality, epoch milliseconds, confidence, class, optional velocity and metadata; modality-dependent frames; consumer-specific tracker behavior | `Measurement` with modality, bounded float timestamp, position/covariance, optional velocity/origin/class/sensor ID; independent KF/EKF/UKF/PF/IMM tracker | Field names, clock unit/epoch, source/track identity, confidence, metadata, modality tag (`rf` vs `radiofrequency`), defaults and output schemas differ. Filter/association/process-noise/initialization behavior is not identical. | **blocked**; numerical reference only |
| **Galadriel** | Per-track, per-sequence `PidObservation`: modality, timestamp-ms, NIS/dof and optionally innovation plus covariance; scientific use additionally needs exact sequence alignment, one common residual frame and a common frozen pre-update prior | Manwe exposes measurements and final track states, not per-update innovation records | No observation serializer, stable track/sequence/session contract, explicit miss/lifecycle event, or frozen-prior/common-frame instrumented update exists. Sequentially updated and mixed polar/Cartesian residuals are not valid cross-channel PID columns. | **blocked**; scalar NIS smoke work would still need an adapter |
| **Engram / vendored NCP v0.6** | Versioned `SensorFrame`/`CommandFrame`/`ObservationFrame`, `seq >= 1` on published frames, `frame_id`, units, named channels, session/restart semantics, provenance and transport ACLs | Local Python objects with no NCP codec or publisher | No NCP v0.6 envelope, sequence join, session identity, frame/unit declaration, authorization, backpressure or restart behavior. Standalone NCP 0.7 is a breaking pre-1.0 wire and cannot be substituted silently. | **blocked** |
| **Prisoma** | Versioned run logs or offline `(V,L,D,A)` samples; exact sequence joins; finite, fixed-width vectors within a run; labels/splits/provenance; explicit missingness | Variable-size detections/tracks and synthetic tracking scenarios | A feature-definition adapter must choose stable columns, deterministic track ordering, padding/masks or selection, sequence alignment, labels and provenance. A changing target count cannot be passed as a fixed statistical vector implicitly. | **blocked**; possible offline research adapter |
| **pid-rs** | Python 3.11+ for the binding; finite, contiguous `float64` sample matrices with equal row counts and fixed dimensions; estimator-specific sample-size, dependence and geometry checks | Manwe supports Python 3.10+ and NumPy arrays can be materialized from its outputs | The shared environment must use Python 3.11+. No extraction schema defines rows/columns, missingness, track identity, standardization, temporal blocks or statistical adequacy. Tracking frames are autocorrelated and not automatically i.i.d. samples. | **research-only** after an explicit extraction/gating layer |

## Audit anchors

The compatibility conclusions above were checked against these implementation
boundaries, not project names or README aspirations:

| Repository | Reviewed boundary |
|---|---|
| crebain | `src-tauri/src/common/yolo.rs`, native ONNX/TensorRT/CoreML/MLX loaders, browser `src/detection/CoreMLDetector.ts`, `src-tauri/src/sensor_fusion.rs`, and its model-contract document |
| Galadriel | `galadriel-core`'s `PidObservation`, exact-sequence processing, and the evaluation/paper requirements for common frames and priors |
| Engram/NCP | neurocontrol protocol/transport/session types and the vendored NCP v0.6 integration rules for `seq`, `frame_id`, channels and sessions |
| Prisoma | offline VLDA sample schema, `ncp-observer` sequence joins, run-log provenance and fixed-vector statistical harnesses |
| pid-rs | PyO3 array conversion/shape checks and estimator documentation for finite samples, temporal dependence and geometry |

These are local-checkout observations, not stable claims about future upstream
versions. Pin the consumer commit in every real adapter record.

## Internal taxonomy split

Manwe itself has two distinct label regimes that must not be conflated:

| Path | Current taxonomy | Consequence |
|---|---|---|
| Root Rust/Candle CLI and benchmark checkpoint | 80 COCO classes | Shape can match an 84-feature YOLO consumer, but most COCO labels are irrelevant to the airspace contract. |
| Python candidate consumer contract | `drone`, `bird`, `aircraft`, `helicopter`, `unknown` in indices 0–4 | A raw YOLO head normally has 9 features (`4 + 5`) and cannot be parsed by a fixed 84-feature consumer. |

The fallback COCO mapping only maps `airplane → aircraft` and `bird → bird`; it
drops other COCO labels. It cannot create drone or helicopter capability. An
adapter must reject unknown head widths and carry a complete index-to-class table;
it must never guess a taxonomy from a file name.

## Artifact and tensor requirements

An artifact is compatible only if all of these match the selected consumer path:

| Dimension | Required decision |
|---|---|
| Container | `.onnx`, raw `.mlpackage`, compiled `.mlmodelc`, `.engine`, and `.safetensors` are different products. A suffix does not identify the graph or runtime. |
| Graph family | YOLO raw head, YOLO post-NMS output, and DETR boxes/logits require different parsers. |
| Tensor interface | Pin every input/output name, rank, dimension, dynamic-axis rule, dtype and layout. Reject extras or missing tensors unless explicitly allowed. |
| Image transform | Pin color order, alpha handling, orientation, letterbox versus stretch/crop, interpolation, pad value, normalization and input size. |
| Box transform | Pin `xyxy` versus `cxcywh`, normalized versus pixels, source versus model canvas, clipping and inverse-letterbox rounding. |
| Scores | Pin activation, objectness/class combination, threshold inclusivity, per-class/global NMS, IoU convention and maximum detections. |
| Precision | Record FP32/FP16/INT8 conversion and calibration data. Accuracy validation precedes latency comparison. |

Current paths are not uniform: Manwe's Rust image helper letterboxes, while
crebain has backend paths that resize/stretch and browser paths that can letterbox.
Therefore one backend passing does not validate another backend.

## Measurement, coordinate, identity and time contract

Manwe currently follows this local convention:

| Modality | Manwe `position` | Manwe covariance |
|---|---|---|
| `radar` | `[range_m, azimuth_rad, elevation_rad]` | polar `3×3` or diagonal in `[m², rad², rad²]` |
| other modalities | Cartesian `[x, y, z]` metres | Cartesian `3×3` or diagonal in `m²` |

That table is incomplete as an interchange contract. An adapter must additionally
declare:

- world/sensor frame name, handedness, axis directions, origin and datum;
- whether azimuth is clockwise/counter-clockwise, its zero axis and wrapping;
- the transform and transform timestamp from each sensor frame to the world frame;
- covariance frame and whether cross-axis terms were discarded;
- monotonic versus wall/epoch time, unit, clock domain and synchronization error;
- `session_id`, `source_id`, `sensor_id`, strictly ordered `seq`, stable track ID,
  class ID, confidence and provenance;
- explicit miss, dropped-frame, restart, out-of-order and track-delete events.

Manwe examples use a local Z-up-style Cartesian interpretation. Prisoma physics
fixtures are also Z-up, but matching the letter `z` is not proof of a shared origin,
handedness or datum. Engram/NCP carries named vectors, units and `frame_id`; examples
that use another vertical axis must be transformed explicitly. Never infer a frame
from a three-element array.

Timestamps are also not interchangeable: Manwe uses a floating-point value in its
local tracker, crebain fusion uses `u64` epoch milliseconds, and NCP requires
sequence-based joins with version/session semantics. Multiplying by 1000 is safe
only after the source clock, epoch and rounding/overflow policy are declared.

## Why Manwe fusion is not a numerical twin

Both Manwe and crebain contain 6-state tracking concepts, but shared vocabulary is
not numerical equivalence. At minimum, the reviewed implementations can differ in:

- process-noise construction and prediction-gap handling;
- birth covariance and velocity prior;
- class-aware clustering and global one-to-one association;
- measurement update order and covariance use;
- IMM motion models, transition probabilities and evidence accumulation;
- particle count/resampling and random-number streams;
- M-of-N confirmation, miss counting and deletion;
- track IDs, confidence, provenance and serialization.

A parity claim requires a common event fixture and assertions on every predicted
prior, innovation, innovation covariance, association, posterior, lifecycle change
and output—not merely similar final trajectories.

## Galadriel-specific scientific boundary

Galadriel's scalar NIS detector can parse a correctly shaped observation, but the
optional cross-channel PID analysis needs stronger semantics. For one track and
one sequence, every modality column must represent the same latent event in the
same coordinate basis and must be computed from a shared frozen prior. Radar polar
residuals beside Cartesian visual/acoustic residuals violate that condition.
Sequential updates also give later modalities a different prior. Missing/gated
measurements are informative and cannot be silently removed before exact sequence
intersection. A Manwe adapter must instrument these semantics at the tracker, not
reconstruct them from final tracks.

## Prisoma and pid-rs statistical boundary

Detections and tracks are ragged event sets; PID estimators consume rectangular
sample matrices. A defensible adapter must define, before looking at results:

1. what one row represents (frame, track-frame, window or episode);
2. how a stable target is selected or how multiple targets are ordered;
3. fixed feature columns, units and transforms;
4. padding plus masks, or an explicit exclusion rule for missing targets/modalities;
5. exact `seq` joins across V/L/D/A and labels;
6. finite/contiguous `float64` conversion and constant-column handling;
7. train/evaluation splits and leakage controls;
8. temporal dependence handling (for example block bootstrap rather than an i.i.d.
   row assumption);
9. sample-size, intrinsic-dimension and distance-concentration gates;
10. provenance for model revision, scenario, seed, preprocessing and exclusions.

Without that layer, successful function execution is not scientific validation.

## Ten promotion gates

An adapter may be called compatible only after all ten perspectives pass:

1. **Schema/version:** producer, adapter and consumer versions are pinned; unknown
   major/minor versions fail closed.
2. **Artifact/provenance:** exact bytes or directory-tree digest, source, model/data
   rights, conversion command and toolchain locks are recorded.
3. **Tensor/shape:** names, ranks, layouts, class count, dynamic bounds, dtypes and
   malformed-input behavior are checked at load time.
4. **Pre/postprocess:** golden pixels and golden raw tensors prove color, resize,
   normalization, coordinate reversal, thresholds and NMS.
5. **Taxonomy/semantics:** every index and modality maps explicitly; unsupported
   labels or fields are rejected rather than coerced to a plausible default.
6. **Geometry/uncertainty:** frames, transforms, units, covariance and angle
   conventions are documented and tested with non-axis-aligned fixtures.
7. **Time/identity:** session/source/sensor/track IDs, sequence, clock domain,
   synchronization, restart, out-of-order and missing-event behavior are tested.
8. **Numerical/scientific parity:** seeded intermediates and lifecycle decisions
   match the declared tolerance; statistical assumptions and dependence controls
   are tested separately.
9. **Quality/performance:** representative class/small-class coverage, AP,
   deployed-threshold precision/recall/FPPI and direct box/score agreement pass
   before latency is measured; timing scope, warm-up, synchronization, hardware
   and run variance are recorded.
10. **Operational/security:** malformed and oversized inputs, untrusted model files,
    resource exhaustion, transport authorization, logging/redaction, rollback and
    license obligations are exercised.

## Minimum integration fixture set

Each consumer adapter should own fixtures for:

- one target of every supported class plus no-target and irrelevant-class frames;
- extreme aspect ratios, edge/corner boxes, overlapping same/different classes and
  an intentionally wrong tensor layout/class count;
- a non-axis-aligned sensor pose with non-diagonal covariance;
- radar angle wrapping and a mixed radar/Cartesian update;
- duplicate, missing, delayed, out-of-order and restart sequences;
- simultaneous targets crossing, temporary occlusion, birth, confirmation and
  deletion;
- NaN/Inf, invalid covariance, truncated artifact, wrong digest and oversized batch;
- an accuracy regression and a latency regression that prove both gates fail.

Until those fixtures pass in the consumer repository, label the output **candidate
Manwe data**, not a production-compatible artifact.
