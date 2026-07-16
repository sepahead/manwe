# Candle/Metal inference benchmark harness

This directory contains two bounded Rust/Candle profiles for Apple Silicon:

- `performance_test` measures static-image raw forward execution.
- `benchmark_video` measures a decoded video pipeline with latest-frame backpressure.

These profiles are useful for one pinned configuration. They are not a
cross-backend leaderboard, an accuracy evaluation, or evidence that another runtime
will produce equivalent detections.

## Historical results are not decision evidence

An earlier version of this directory reported CoreML, Candle, and PyTorch FPS from
different implementations and timing boundaries. Those numbers did not capture
enough artifact, environment, preprocessing, or semantic provenance to support a
backend ranking. The legacy Python, Swift, plotting, export, and orchestration
scripts were removed because they auto-downloaded mutable checkpoints, trusted
pickle-backed weights, overwrote evidence, and could not enforce comparable work.

Do not use results from that legacy harness for deployment decisions.

## Enforced artifact contract

Both remaining executables:

- require a local, nonempty, regular, non-symlink `.safetensors` model;
- require the expected SHA-256 and fail before inference on a mismatch;
- use Candle 0.11.0 with the Metal backend and fail if Metal is unavailable;
- use FP32 and a 640×640 isotropic letterbox input;
- bound input dimensions, decoded allocation, requested work, FPS, and identifiers;
- hash the model and inputs used by the run;
- restrict video inputs to seekable MP4/MOV, Matroska/WebM, or AVI demuxers so
  FFmpeg cannot interpret an authenticated input as a playlist of unrecorded files;
- require owner-controlled result directories, reserve each run in a private
  staging directory, publish with no-replace hard links, and clean only the known
  staging basenames through bound directory descriptors without recursive deletion;
- publish owner-only (`0600`) evidence files and fail the command if a committed
  publication's staging removal or directory sync is incomplete;
- require native child executables whose canonical path is root/current-account
  owned, not group/world-writable, and free of mutating macOS extended ACLs;
- refuse to replace existing JSON or video evidence; and
- build from the checked-in lockfile with Rust 1.95 or newer.

The model architecture is the repository's fixed YOLOv8-small, 80-class graph.
A valid safetensors container with different keys or shapes is rejected while the
model loads. The SHA proves artifact identity, not accuracy, licensing, or semantic
compatibility.

## Static-image profile

From this directory on Apple Silicon:

```bash
cargo build --release --locked --bin performance_test

RUN_DIR=/absolute/path/to/new-results \
  ./target/release/performance_test \
  --model /absolute/path/to/yolov8s.safetensors \
  --model-sha256 <64-hex> \
  --image-dir /absolute/path/to/images \
  --num-images 500 \
  --run-id candle_metal_001
```

The harness sorts candidate paths, selects exactly the requested number of valid
images, hashes their bytes into an ordered manifest, and rechecks each digest when
the input is decoded. Images are processed one at a time rather than retained as a
large tensor collection.

The timed boundary includes CPU-to-device upload, model forward execution, a
metadata-only check for the fixed `[1, 84, 8400]` COCO output schema, and a
transfer-free device synchronization barrier. It excludes model loading, image
decode, letterboxing, full-output device compaction, CPU readback, finite-value
validation, confidence filtering, NMS, and annotation. Reported static FPS is
derived from those timed samples and therefore has the same exclusions. The JSON
records that scope explicitly. Every output is still fully validated before
evidence is published.

The owner-controlled run directory and run identifier are validated before model
work begins. Results are written, synced, and verified inside a private
`.manwe-static-benchmark-<run-id>.in-progress` directory, then atomically published
with a no-replace hard link. Pre-link failures remove only the known `result.json`
entry and the now-empty directory; unexpected content is preserved. The verified
final hard link plus successful run-directory sync commits the result. Earlier
post-link failures preserve the link and staging directory. A subsequent cleanup
failure returns nonzero without removing the committed final JSON; staging may be
present, partial, or absent if the final sync failed. The final JSON is therefore
the authoritative commit marker.

## Video profile

```bash
cargo build --release --locked --bin benchmark_video

RUN_DIR=/absolute/path/to/new-results \
  ./target/release/benchmark_video \
  --model /absolute/path/to/yolov8s.safetensors \
  --model-sha256 <64-hex> \
  --video /absolute/path/to/input.mp4 \
  --target-fps 30 \
  --max-duration-seconds 7200 \
  --run-id candle_video_001
```

Use `--target-fps 0` for an explicitly unthrottled input and `--save-video` only
when an annotated artifact is needed. The result includes the source-video digest,
model digest, FFmpeg path and digest, selected demuxer, presented/processed counts,
drop rate, selected-frame raw-pipe read wait, synchronized preprocess/upload and
model-forward stage times, raw-pipe-to-forward latency, and (when requested) the
output-video digest. The stage split uses transfer-free device synchronization
after normalized input preparation and model forward; the fixed 8,400-prediction
COCO schema check precedes the latter barrier. Latency starts when the reader
requests a selected raw frame and ends when model forward synchronizes. It
excludes source capture, the subsequent full-output device compaction, CPU
readback, and finite-value validation, plus optional NMS, rendering, and encoding.
Wall-clock processed FPS does include that full-output validation and optional
rendering/encoder-pipe writes.
FFmpeg decode and output failures are surfaced; output paths are reserved before
work begins, the encoder receives a kernel-enforced 8 GiB file-size limit, and
final names are published only after verification. Linux executes the authenticated
encoder descriptor with `fexecve`; macOS, which has no public descriptor-exec API,
rechecks the explicitly trusted path immediately before `exec`.
Failures before the first final link clean only known staging entries; unexpected
content is never traversed and keeps the staging directory in place. Once either
the optional video or result link may exist, pre-commit failures preserve all links
and the `.manwe-benchmark-*.in-progress` marker. The verified result hard link plus
successful result-directory sync is the commit point and the final JSON is the
authoritative commit marker. A later staging-cleanup failure returns nonzero, does
not remove final links, and may leave staging present, partially cleaned, or absent
when its final directory sync has uncertain durability. Inspect any staging marker
or video without its final JSON before intentionally retrying the run ID.
The optional video's staged inode and SHA-256 are authenticated before publication;
the digest is rechecked on both staged and published hard links before JSON commit.

## What a publishable comparison still requires

A cross-backend claim needs, at minimum:

1. Exact repository commit, clean/dirty state, lockfiles, OS, toolchain, framework,
   runtime/provider, hardware, power mode, thermal state, and competing workloads.
2. Checkpoint source and digest, conversion command and digest for every backend,
   precision, calibration-data digest, and artifact rights.
3. An immutable ordered input manifest and golden preprocessing fixtures covering
   color, orientation, interpolation, padding, normalization, and input layout.
4. Golden raw tensors and postprocessed detections proving box, score, class, and
   NMS parity, followed by a representative held-out accuracy/fidelity gate.
5. One identical timed boundary, appropriate synchronization, randomized run
   order, cold-start measurements, at least 100 samples, tail latency, variance,
   and confidence intervals.
6. A precise concurrency and backpressure model with sustained-load latency,
   accuracy, queueing, and dropped-frame behavior—not summed isolated FPS.

Until a shared runner enforces those rules, report these outputs only as standalone
Candle/Metal profiles with their declared scope.

## Model and software rights

The benchmark source is MIT-licensed. Checkpoints, datasets, FFmpeg builds, and
optional frameworks retain their own terms. Record exact artifacts and review their
rights before redistribution or commercial use; see
[`../docs/MODEL_CONTRACTS.md`](../docs/MODEL_CONTRACTS.md).
