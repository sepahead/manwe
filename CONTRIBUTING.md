# Contributing to Manwe

Manwe is a Python research/numerical package plus Rust/Candle inference and
benchmark tooling. Crebain and the other ecosystem repositories are intended
consumers, not automatically compatible dependencies. Applications are civilian
and dual-use (disaster relief, delivery deconfliction, inspection and airspace
awareness), so failure modes and provenance matter as much as happy-path output.

## Ground rules

- **Keep the core pure-numpy.** `manwe.common`, `manwe.fusion`, `manwe.multicam`,
  `manwe.audio` (DSP), and `manwe.eval` must import and test with **numpy only**.
  Training/conversion dependencies belong behind a lazy
  `manwe.common.deps.require(module, extra)` call. TensorRT is installed as a
  platform runtime; MLX conversion is not implemented.
- **Never hard-code a device.** Use `manwe.common.resolve_device`.
- **Do not call a candidate contract “compatible.”** Class taxonomy and model
  manifests live in `manwe.common.contracts`, while the audited consumer gaps live
  in [docs/INTEGRATION_CREBAIN.md](docs/INTEGRATION_CREBAIN.md). A change touching
  tensors, preprocessing, taxonomy, coordinates, time, IDs or shapes needs a
  consumer fixture and an explicit compatibility-status update.
- **Weights and datasets are never committed** (see `.gitignore`).
- **Fail closed at trust boundaries.** Reject malformed/non-finite/oversized input,
  wrong artifact digests and unsupported schemas instead of guessing defaults.

## Python development

Use Python 3.10–3.14 for the core checks and `uv` 0.11.28, the exact resolver
version pinned in CI. The optional heavy stack is conservatively documented for
Python 3.11–3.12. Rust checks require Rust 1.88 or newer; Metal/viewer checks need
Apple tooling and CUDA checks need a target NVIDIA toolchain.

```bash
cd python
uv sync --locked --extra dev        # core + pytest/ruff/mypy
# Optional heavy stack (use a supported Python/platform and review the lock diff):
uv sync --locked --extra all --extra dev

uv run --no-sync pytest tests
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy src/manwe
```

The locked `rfdetr` extra covers construction/inference only. Do not install the
upstream training extra into the combined environment while it pulls competing
OpenCV distributions; curate and verify one package owner first.

Add a test for every new numeric routine — the filters, triangulation, DOA, and
metrics are all covered by fast, seeded tests in `python/tests/`. New pillar code
that needs torch should be split so its pure logic (postprocessing, mapping, math)
is testable without it.

## Rust development

```bash
cargo fmt --all --check
cargo test --locked --no-default-features
cargo clippy --locked --all-targets --no-default-features -- -D warnings

# Platform-specific paths (run on matching hardware/toolchain):
cargo test --locked --features metal
cargo clippy --locked --all-targets --features viewer,metal -- -D warnings
# NVIDIA host: cargo test --locked --features cuda

# Benchmark crate (Apple Silicon/Metal):
cargo test --manifest-path metal-yolo-tests/Cargo.toml --locked
cargo clippy --manifest-path metal-yolo-tests/Cargo.toml --locked --all-targets -- -D warnings
```

## Pull requests

1. Open an issue first to discuss non-trivial changes.
2. One focused change per PR.
3. Keep the applicable core, lint, package and platform-feature checks green.
4. Update docs whenever usage, a candidate contract or a default changes. Record
   empirical justification; a research citation alone does not validate a default.
5. Preserve lockfiles and explain dependency/license changes.

For model/export changes, include artifact digests, tensor/pre/postprocess fixtures,
per-class AP50/AP50-small deltas, deployed-threshold precision/recall/FPPI, direct
box/score agreement, required-class coverage and the exact consumer path. Use
unique `image_id` values and source-image-pixel `xyxy` boxes. For performance
changes, satisfy the "What a publishable comparison still requires" checklist in
`metal-yolo-tests/README.md`; numbers with different timed scopes are not comparable.

## Areas of interest

- Vision: RF-DETR / YOLO26 fine-tuning recipes, P2 head, SAHI training integration.
- Audio: a SELD ResNet-Conformer (Multi-ACCDOA) upgrade over the SRP-PHAT baseline.
- Multi-camera: RANSAC-over-subsets triangulation, LOSTU-style covariance, ReID.
- Fusion: a Coordinated-Turn IMM model, GLMB/PMBM for dense swarms, a Stone Soup
  cross-validation adapter.
- Export: an implemented and fixture-tested MLX path, TensorRT INT8 calibration,
  COCO-style evaluation, and consumer-owned integration fixtures.

## License

By contributing you agree your contributions are licensed under the MIT License.
That license covers your source contribution; it does not relicense checkpoints,
datasets, generated weights or third-party dependencies included in a test/run.

Release maintainers must also follow [docs/RELEASING.md](docs/RELEASING.md); an
alpha tag is not authorized until its external credential-response gate is closed.
