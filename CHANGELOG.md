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
- Linux CPU and arm64 macOS Metal CI across the supported Python/Rust floors,
  package smoke tests, dependency/license audits, and current/range secret scans.
- Architecture, model-contract, integration-status, security, research, license,
  and benchmark-protocol documentation.

### Changed

- Corrected general-alpha GOSPA to follow its cut-off-metric definition and
  corrected the synthetic scenario's constant-acceleration position step; the
  deterministic `fusion-sim` reference values now reflect the exact kinematics.
- Particle tracks now own independent seeded random streams, so birthing an
  unrelated track cannot change an existing track's future process noise.
- New-track clustering is invariant to producer order and sensor-ID renaming;
  ambiguous distance chains remain separate instead of selecting an arbitrary
  anchor-dependent merge.
- Rust inference now requires local digest-bound safetensors, validates tensor and
  image bounds, preserves aspect ratio with letterboxing, and fails closed at
  unsupported filesystem boundaries.
- The Metal benchmark harness now reuses the root model implementation, records
  immutable evidence, bounds work, and removes incomparable legacy runners.
- Camera URLs and model paths are supplied at runtime; credential-bearing values
  are no longer embedded or echoed by current source.
- Raised the Rust floor from 1.88 to 1.95 across both crates, CI, and the docs.
  Bevy 0.19 sets it (`rust-version = 1.95.0`, reached through the optional `viewer`
  feature); Candle 0.11 independently needs 1.94 on aarch64 (`stdarch_neon_f16`,
  stabilised in 1.94) and imageproc 0.27 needs 1.89 through its mandatory nalgebra
  0.35. Cargo declares one floor per package, so the highest of the three wins.
- The experimental camera viewer moved from Bevy 0.13 to Bevy 0.19 (required
  components, the `Message` split, `Sprite`/`Camera2d`, and the `bevy_sprite_render`
  feature split). `camera_view` now exits non-zero when Bevy reports
  `AppExit::Error` instead of discarding it.
- Upgraded Candle (core/nn/transformers) 0.9.2 to 0.11.0. No inference source
  changed. Candle 0.11 rewrote its Metal backend onto `objc2-metal`; a full YOLOv8
  forward pass was revalidated on the Metal device and is bit-identical to CPU.
  Candle 0.11 takes a non-optional `tokenizers`/Oniguruma dependency that no feature
  gate can drop, so builds now require a C toolchain and a larger cold build.
- Upgraded imageproc 0.25.1 to 0.27.0 (text drawing moved behind a `text` feature,
  now enabled explicitly) and clap 4.5.57 to 4.6.1. The exact clap pin is retained:
  clap raised its own MSRV in a minor release, so a caret range could move the
  compiler floor without a source change. CLI parsing is unchanged.
- Upgraded sha2 0.10 to 0.11 (digest 0.11 / hybrid-array). Digest output no longer
  implements `LowerHex`, so all five SHA-256 rendering sites now route through a
  single audited `secure_io::sha256_hex`, pinned by known-answer vectors; the
  benchmark crate no longer depends on `sha2` directly.
- Upgraded the vetted Ultralytics runtime from 8.4.91 to 8.4.92.
- Pinned CI to actions/checkout v7.0.0 and actions/setup-python v6.3.0 by commit
  SHA. checkout v7 blocks fork-PR checkout under `pull_request_target` and
  `workflow_run`; neither trigger is used here, so no opt-in is taken.

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
