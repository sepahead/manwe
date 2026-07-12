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

- Rust inference now requires local digest-bound safetensors, validates tensor and
  image bounds, preserves aspect ratio with letterboxing, and fails closed at
  unsupported filesystem boundaries.
- The Metal benchmark harness now reuses the root model implementation, records
  immutable evidence, bounds work, and removes incomparable legacy runners.
- Camera URLs and model paths are supplied at runtime; credential-bearing values
  are no longer embedded or echoed by current source.

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
