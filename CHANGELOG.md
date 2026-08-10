# Changelog

All notable changes are documented here. Versions follow Semantic Versioning;
the Python distribution uses the PEP 440 spelling of the same prerelease.

## 0.2.0-alpha.1 — unreleased

This is the first planned tagged alpha after the untagged Rust/Candle prototypes.

### Added

- A typed Python numerical workbench for vision, audio direction-of-arrival,
  multi-camera geometry, multi-target fusion, evaluation, synthetic data, raw
  export receipts, and candidate model contracts.
- A bounded CLI with offline dependency policy, explicit artifact digests, lazy
  heavy extras, and reproducible `uv` locking.
- Source distributions now use an explicit package/test/lock allowlist. CI
  accepts only the bounded generated metadata plus regular source members whose
  exact paths are tracked and whose bytes match the release checkout, rejecting
  ignored environments, untracked nested files, links, special entries, and
  other workstation state.
- Linux CPU and arm64 macOS Metal CI across the supported Python/Rust floors,
  package smoke tests, dependency/license audits, and current/range secret scans.
- Architecture, model-contract, integration-status, security, research, license,
  and benchmark-protocol documentation.

### Changed

- Corrected general-alpha GOSPA to follow its cut-off-metric definition and
  corrected the synthetic scenario's constant-acceleration position step; the
  deterministic `fusion-sim` reference values now reflect the exact kinematics.
- Weighted particle/IMM moment covariances now repair only roundoff-sized
  indefiniteness before exact PSD certification, while admitted Gaussian state
  remains strict. Track covariance-volume limits use exact 3×3 determinants, so
  singular PSD particle states no longer fail on a spurious floating determinant
  sign; the default five-filter `fusion-sim` command is pinned end to end.
- Particle tracks now own independent seeded random streams, so birthing an
  unrelated track cannot change an existing track's future process noise.
- New-track clustering is invariant to producer order and sensor-ID renaming;
  ambiguous distance chains remain separate instead of selecting an arbitrary
  anchor-dependent merge.
- Radar measurements and EKF updates now reject zero-range and vertical-axis
  polar singularities instead of silently consuming evidence with a zero
  Jacobian.
- Moving-target triangulation now requires physically simultaneous captures.
  Timestamped views must have exactly equal capture times with zero relative
  uncertainty; untimestamped views require an explicit simultaneous-capture
  acknowledgement. Static-scene skew remains supported. This removes an
  unsound isotropic speed/skew covariance heuristic that could understate
  geometry-amplified depth bias by orders of magnitude.
- Multi-camera covariance now fails closed unless callers explicitly acknowledge
  analytically exact calibration. Pixel localization covariance cannot represent
  focal-length or pose bias, which may preserve a zero reprojection residual
  while shifting depth; estimated real-camera rigs remain unsupported until
  calibration-parameter covariance is propagated. The rig schema is version 2
  because this acknowledgement is now mandatory.
- Tracker clock-gap budgeting no longer partitions the discrete per-cycle filter
  model: each call now applies acceleration noise and an IMM Markov transition
  exactly once, independent of the `max_dt` admission quantum. This intentionally
  changes long-gap covariance, seeded particle trajectories/RNG consumption, and
  IMM mode probabilities from the earlier numerical-substep behavior; changing
  the actual caller cycle cadence still changes this event-indexed model.
- Updated the optional viewer graph from yanked `spin` 0.10.0 to 0.10.1, which
  fixes unsound consuming operations, and made future yanked crates a
  dependency-policy error.
- Rust inference now requires local digest-bound safetensors, validates tensor and
  image bounds, preserves aspect ratio with letterboxing, and fails closed at
  unsupported filesystem boundaries.
- Rust YOLO postprocessing now uses continuous-coordinate IoU instead of
  Candle's inclusive integer-pixel convention, validates every raw output value
  before any visible result, and uses the canonical Ultralytics COCO labels.
- SPPF max pooling now pads with negative infinity like PyTorch instead of zero;
  non-zero odd kernels and a full-grid production-kernel oracle pin the shape and
  border semantics.
- Export contracts now accept only bounded canonical tensor metadata tied to the
  receipt's static image size and class count. Export preflight rejects stride-
  rounded image sizes and unsupported end-to-end heads before artifact creation.
- TensorRT INT8 export now descriptor-pins the source manifest and dataset tree,
  validates at least 1,000 unique effective `val` tensors, supplies an exact
  hash-ranked 512-image private loader view, disables
  cache/network/optional-codec side channels, binds the audited TensorRT and
  ModelOpt route, rejects silent FP32 fallback and unbounded ModelOpt work, and
  rechecks both the source manifest and declared dataset tree before publication.
- Detection manifests now require nonempty `train` and `val` splits and
  descriptor-inventory every selected tree under aggregate work limits. Nested
  links and special entries are rejected, and regular-file identities must be
  unique within and across splits, so hardlink aliases cannot bypass overlap
  validation. A reverse-order confirmation pass rejects mutations between split
  observations; mutation after a path's final check remains an OS-isolation
  boundary rather than a filesystem-snapshot claim.
- The Metal benchmark harness now reuses the root model implementation, records
  digest-verified no-replace evidence, bounds work, validates the fixed COCO
  schema and every finite output value, and removes incomparable legacy runners.
- Camera URLs and model paths are supplied at runtime; credential-bearing values
  are no longer embedded or echoed by current source.
- Annotated-image publication now requires an owner-controlled parent directory
  and creates the published inode with owner-only permissions. It removes only
  its exact known staging file and empty directory through bound descriptors.
  Recursive path-based cleanup was removed; post-link errors report the
  last-authenticated path and visibility/durability state, and incomplete
  committed cleanup is explicit.
- Updated the optional viewer graph from `event-listener` 5.4.1 to 5.4.2, which
  fixes RUSTSEC-2026-0221. Both Rust dependency gates now deny soundness
  advisories for transitive code instead of treating them as informational.
  CI now pins cargo-deny 0.20.2, whose advisory fetch remains on the database's
  main branch across repeated root/benchmark checks.
- Raised the Rust floor from 1.88 to 1.95 across both crates, CI, and the docs.
  Bevy 0.19 sets it (`rust-version = 1.95.0`, reached through the optional `viewer`
  feature); Candle 0.11 independently needs 1.94 on aarch64 (`stdarch_neon_f16`,
  stabilised in 1.94) and imageproc 0.27 needs 1.89 through its mandatory nalgebra
  0.35. Cargo declares one floor per package, so the highest of the three wins.
- The experimental camera viewer moved from Bevy 0.13 to Bevy 0.19 (required
  components, the `Message` split, `Sprite`/`Camera2d`, and the `bevy_sprite_render`
  feature split). `camera_view` now exits non-zero when Bevy reports
  `AppExit::Error` instead of discarding it.
- Upgraded Candle (core/nn/transformers) 0.9.2 to 0.11.0. Candle 0.11 rewrote its
  Metal backend onto `objc2-metal`. This repository has no digest-bound golden
  model-forward fixture, so it does not establish CPU/Metal numerical
  equivalence or unchanged inference behavior across the upgrade. Candle 0.11
  takes a non-optional `tokenizers`/Oniguruma dependency that no feature gate can
  drop, so builds now require a C toolchain and a larger cold build.
- Upgraded imageproc 0.25.1 to 0.27.0 (text drawing moved behind a `text` feature,
  now enabled explicitly) and clap 4.5.57 to 4.6.1. The exact clap pin is retained:
  clap raised its own MSRV in a minor release, so a caret range could move the
  compiler floor without a source change. CLI parsing is unchanged.
- Upgraded sha2 0.10 to 0.11 (digest 0.11 / hybrid-array). Digest output no longer
  implements `LowerHex`, so all five SHA-256 rendering sites now route through a
  single audited `secure_io::sha256_hex`, pinned by known-answer vectors; the
  benchmark crate no longer depends on `sha2` directly.
- Upgraded the vetted Ultralytics runtime from 8.4.91 to 8.4.92.
- Pinned the optional RF-DETR runtime to 1.8.3 and corrected the training adapter
  to validate and forward upstream `grad_accum_steps`. The legacy
  `gradient_accumulation_steps` misspelling is now rejected instead of being
  silently ignored upstream. CI now checks the installed `TrainConfig`,
  network-denied model construction, and exactly-one-OpenCV ownership; it does not
  execute or qualify a training run.
- Pinned CI to actions/checkout v7.0.1 and actions/setup-python v7.0.0 by commit
  SHA. checkout v7 blocks fork-PR checkout under `pull_request_target` and
  `workflow_run`; neither trigger is used here, so no opt-in is taken.
- Removed unreachable SVG definitions and styles from both logos, added native
  title/description accessibility metadata, retained reduced-motion behavior,
  and tuned genuinely distinct light/dark palettes without changing the mark's
  geometry. The model catalog now reports only the three implemented export
  targets rather than implying that MLX conversion exists.

### Known alpha limits

- No reviewed downstream repository has a drop-in adapter; consumer-owned fixtures
  and the ten documented promotion gates remain required.
- No model weights, golden model-forward fixture, CUDA hardware run, real-camera
  drill, or Windows support claim is included.
- The Bevy camera viewer is experimental and macOS-oriented.
- Python package artifacts use the distribution name `manwe-perception`; the
  import package and command remain `manwe`. Registry publication is disabled for
  this source/GitHub alpha, and the unrelated legacy `manwe` distribution must not
  be co-installed in the same environment.
- The alpha tag must not be created until historical camera credentials have been
  confirmed revoked or rotated.
