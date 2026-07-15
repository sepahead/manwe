# manwe architecture

Manwe is a research workbench for detector training, acoustic processing,
multi-camera geometry, fusion, evaluation, and Rust/Candle inference. Crebain is
one intended consumer, but the repositories do not currently share a complete
wire or model-runtime contract. Manwe produces candidate artifacts and reference
results; an explicit consumer adapter and validation campaign are still required.

## Design principles

1. **Pure-numpy core, lazy heavy deps.** The fusion, geometry, DOA, metrics and
   contract layers depend only on numpy, so they run without ML runtimes on the
   tested Linux and macOS hosts. Windows is not an alpha release target: several
   trust-boundary helpers intentionally rely on POSIX descriptor-relative I/O.
   Training and raw conversion dependencies are isolated behind explicit extras.
   TensorRT remains a platform installation, and MLX conversion is not implemented.
2. **One device-selection helper.** `manwe.common.resolve_device` represents CUDA,
   MPS, and CPU choices. It does not prove that a model supports the selected
   device or that outputs are numerically equivalent.
3. **Candidate contracts are evidence-bound.** Raw conversion returns an immutable
   source/artifact/options receipt. A model manifest additionally requires an
   explicit backend-inspected tensor signature and evidence reference; it is never
   inferred from a family name or extension. Consumers do not yet ingest that
   manifest automatically, so downstream drift still requires adapter fixtures.
4. **Claims are narrower than tools.** The package provides threshold-specific
   AP, operating-point precision/recall/FPPI, direct export-agreement checks,
   OSPA/GOSPA, and latency utilities. A promotion gate must add representative
   data, per-consumer fixtures, and operational failure tests.

## Repository layout

```
manwe/
├── python/                       # the training ground (primary)
│   ├── src/manwe/
│   │   ├── common/               # contracts, device (Metal/CUDA/CPU), seed, logging, deps
│   │   ├── vision/               # zoo, from-scratch training, sliced inference, postprocess
│   │   ├── audio/                # log-mel/SPL features, GCC-PHAT/SRP-PHAT DOA, acoustic→fusion
│   │   ├── multicam/             # pinhole camera, DLT/midpoint triangulation, cross-cam correlation
│   │   ├── fusion/               # KF/EKF/UKF/PF/IMM, association, M-of-N tracker, metrics, scenarios
│   │   ├── export/               # raw backend conversion, model manifest, fidelity comparison
│   │   ├── data/                 # dataset registry + offline synthetic generator
│   │   ├── eval/                 # source-pixel detection metrics + benchmark utilities
│   │   └── cli.py                # `manwe` CLI (doctor/models/data/synth/fusion-sim/vision-train/export)
│   ├── configs/                  # per-pillar YAML configs
│   ├── tests/                    # pytest suite (numpy-only core runs with zero heavy deps)
│   └── pyproject.toml
├── src/, metal-yolo-tests/       # Rust/Candle reference CLI + Candle/Metal profiling harness
└── docs/
    ├── ARCHITECTURE.md           # this file
    ├── INTEGRATION_CREBAIN.md    # audited consumer compatibility and adapter gates
    ├── MODEL_CONTRACTS.md        # candidate manifests and fidelity limits
    └── research/SOTA-2026.md     # dated, source-cited survey with explicit cautions
```

## Data flow and trust boundaries

```
 datasets ─▶ vision ─▶ checkpoint ─▶ raw backend conversion ─┐
 mic arrays ─▶ audio ─▶ local measurement ───────────────────┤
 cameras ─▶ multicam ─▶ local 3D detection ─────────────────┤
 measurements ─▶ fusion ─▶ local track state ───────────────┤
                                                            ▼
                                            untrusted Manwe output
                                                            │
                    versioned adapter + taxonomy/frame/time/shape conversion
                                                            │
                       fixtures + numerical + fidelity + failure-mode gates
                                                            ▼
                                                     consumer input
```

- **vision** → from-scratch detector experiments mapped to
  `drone/bird/aircraft/helicopter/unknown`,
  with raw ONNX/CoreML/TensorRT conversion utilities. The five-class output is not
  accepted by crebain's fixed 80-class native YOLO parser without an adapter.
- **audio** → microphone-array direction-of-arrival producing the
  local azimuth/elevation/range/SPL representation. A consumer adapter must define
  angle origin/sign, frame transform, range observability, covariance, clock, and ID.
- **multicam** → calibrated cameras + N-view triangulation producing local 3D
  detections and propagated uncertainty under declared pixel/time assumptions.
  Moving-target views must be physically simultaneous: equal zero-uncertainty
  capture timestamps, or an explicit simultaneous-capture acknowledgement for
  an untimestamped hardware-triggered batch. Bounded skew is accepted only for
  a declared static target. Camera YAML, units, distortion, handedness, world
  frame, synchronization, and ID semantics are part of the adapter contract,
  not inferred defaults.
- **fusion** → an independent Python implementation of 6-state filters,
  association, and M-of-N lifecycle. It is useful as a seeded reference, but it is
  not a numerical twin of crebain: defaults, association, process noise,
  initialization, timing, and output schemas differ.

## Hardware strategy (from the SOTA survey)

- CUDA is the intended large-training path; Apple MPS is useful for development
  and bounded experiments. Device choice does not replace a measured parity test.
- ONNX is a useful interchange candidate, not a universal executable contract.
  Tensor names/layouts, preprocessing, postprocessing, opsets, dynamic dimensions,
  precision, and provider behavior must be pinned.
- MPS promotion requires a pinned PyTorch build, explicit fallback logging, and a
  target-hardware assertion that a detector forward/backward has no NaNs or silent
  CPU offload. That heavy hardware gate is not implemented in current CI.
- The fusion stack is currently NumPy/CPU. Its latency and scaling have not been
  established for every target count or consumer workload.

See [the integration status](INTEGRATION_CREBAIN.md) for the audited boundaries,
and [the research survey](research/SOTA-2026.md) for dated background rather than
deployment guarantees.
