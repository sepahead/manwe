# Security policy

## Reporting a vulnerability

Use the repository's private **Security → Report a vulnerability** flow. Do not
open a public issue for an unpatched vulnerability, leaked credential, unsafe
model artifact, or dataset disclosure.

Include the affected commit, entry point, prerequisites, impact, a minimal
reproducer, and any evidence that the issue crosses a trust boundary. Avoid
including live credentials, private data, or proprietary model files.

## Supported code

Security fixes target the current `main` branch. No release tag is currently
declared supported; when that changes, supported tags will be listed here
explicitly. Older snapshots may receive a fix only when a backport is announced.

## Trust boundaries

- Model checkpoints, exported graphs, CoreML bundles, datasets, camera URLs, and
  benchmark inputs are untrusted. Keep them outside the repository and verify
  provenance and digests before use.
- A schema-2 contract integrity-binds its sibling artifact; it does not
  authenticate the contract author. Select contracts through a trusted provenance
  channel. Manwe's native runtime validates the closed Candle adapter, tensor and
  allocation bounds, taxonomy, preprocessing, and executable postprocessing policy
  before loading. The required `failure_behavior` field is bounded evidence text,
  not executable policy. Other consumers must implement equivalent validation for
  their own runtime.
- Dataset-manifest admission is not a filesystem freeze. Manwe revalidates and
  privately copies training/calibration inputs before backend use, then bounds the
  images and labels selected by the pinned consumers. Image decoders remain a
  native-code attack surface; use an isolated account or worker when hostile media,
  a hostile same-UID process, or privileged mount mutation is in scope.
- Supply camera credentials at runtime. Never place them in source, examples,
  command history, logs, screenshots, benchmark results, or model metadata.
- Do not load pickle-based checkpoints from an untrusted source. Prefer bounded,
  buffered safetensors and retain the artifact digest used for a run.
- Trainer output directories are third-party backend working state, not Manwe's
  atomic publication surface. Select one exact generated checkpoint, place it in
  controlled storage, and record its SHA-256 before inference or conversion.
- Native annotated images are published owner-private without replacement. On a
  post-link error, retain the reported final path and any `.in-progress` marker;
  names alone do not establish visibility or durability state. Prefer a dedicated
  existing owner-controlled `--output-dir` when input storage is untrusted,
  shared, or read-only.
- Raw export and contract-sidecar publication require an inspectable destination
  parent. Group/world-writable parents must be sticky and owned by the effective
  account or root; unmodeled access-control ACLs fail closed. Same-UID and
  privileged mutation remain outside this pathname trust boundary.

## Credential response

If a credential is committed, revoke or rotate it first, then remove it from the
current tree. Rewriting reachable Git history is a separate coordinated operation:
it invalidates commit IDs and requires every clone and deployment to be cleaned.
Assume a credential remains compromised even after history is rewritten.
