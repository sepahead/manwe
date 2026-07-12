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
- A model-contract sidecar is evidence, not a sandbox. Consumers must validate
  tensor shapes, allocation bounds, taxonomy, preprocessing, and failure behavior.
- Supply camera credentials at runtime. Never place them in source, examples,
  command history, logs, screenshots, benchmark results, or model metadata.
- Do not load pickle-based checkpoints from an untrusted source. Prefer bounded,
  buffered safetensors and retain the artifact digest used for a run.

## Credential response

If a credential is committed, revoke or rotate it first, then remove it from the
current tree. Rewriting reachable Git history is a separate coordinated operation:
it invalidates commit IDs and requires every clone and deployment to be cleaned.
Assume a credential remains compromised even after history is rewritten.
