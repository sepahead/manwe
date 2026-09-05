# Manwe agent contract

Manwe develops perception research tools, numerical references, and checked model interfaces.
Keep each completion claim within its executed and source-bound evidence.
This file is the canonical entrypoint for coding agents.

## Read before changing

Read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) first.
Then read the document that owns the proposed change.

| Change | Owning documents |
| --- | --- |
| Architecture, ownership, datasets, native runtime | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Training, inference, export, recovery, numerical examples | [docs/WORKFLOWS.md](docs/WORKFLOWS.md) and [python/README.md](python/README.md) |
| Artifact admission, tensors, preprocessing, taxonomy, fidelity | [docs/MODEL_CONTRACTS.md](docs/MODEL_CONTRACTS.md) |
| Consumer adapters, frames, clocks, identity, PID extraction | [docs/INTEGRATION_CREBAIN.md](docs/INTEGRATION_CREBAIN.md) |
| Performance measurement | [metal-yolo-tests/README.md](metal-yolo-tests/README.md) |
| Dependencies, setup, test commands | [CONTRIBUTING.md](CONTRIBUTING.md), [Makefile](Makefile), and [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| Security, credentials, publication | [SECURITY.md](SECURITY.md) and [docs/RELEASING.md](docs/RELEASING.md) |
| Research context | [docs/research/README.md](docs/research/README.md) |

## Working method

1. Inspect the owning implementation, tests, and evidence before proposing a change.
2. Inventory staged changes, unstaged changes, branches, and worktrees before recovery work.
3. Preserve unrelated changes and their staged state.
4. Compare five to ten credible approaches before a material design decision.
5. Record assumptions, benefits, failure modes, and a decisive control for each approach.
6. Request independent review for separable, consequential decisions.
7. Implement generic behavior from explicit contracts.
8. Pair each new accept path with a negative control.
9. Pair each new rejection path with a positive control.
10. Run the applicable complete gate before publication.

Review consequential changes through these twelve lenses:
purpose, ownership, types, mathematics, time, identity, scientific validity, reproducibility, resource bounds, security, rights, and reader usability.
Resolve each failed requirement separately; a council vote cannot override a failed gate.

Freeze selected cases, seeds, exclusions, and thresholds before inspecting outcomes.
Retain failures and negative results.
Never substitute easier cases or branch on an expected fixture value.

Honor the user's delivery instructions.
When direct-main publication is authorized, commit and push only the exact gated milestone.
Preserve required checks and signing requirements.
Report a server-side rejection without bypassing repository protection.
A documentation or source milestone does not authorize an alpha tag or scientific promotion.

Do not modify another contributor's active scope.
Do not change protected PID or KG work through dependency, index, branch, or worktree operations.
Inspect published interfaces read-only when needed for an authorized adapter review.

## Implementation invariants

- Keep `manwe.common`, `manwe.fusion`, `manwe.multicam`, `manwe.audio` DSP, and `manwe.eval` usable with NumPy only.
- Load heavy dependencies lazily through the existing dependency helper.
- Use the shared device selector; do not hard-code a device.
- Preserve exact dependency pins until an installed compatibility review authorizes a change.
- Keep weights, datasets, generated models, and credentials outside Git.
- Reject malformed, non-finite, oversized, ambiguous, or unsupported input.
- Revalidate and copy mutable datasets at consumption; admission does not freeze caller storage.
- Keep the schema-2 contract as the native model authority.
- Keep batch CLI and viewer on the shared native runtime.
- Preserve source taxonomy and explicit missing mappings in output.
- Preserve bounded queues, exact sample identity, and conservative publication recovery.

Raw export stops at an `ExportReceipt`.
`VerifiedArtifactSignature` records tensor-interface evidence, not a cryptographic signature.
A model package requires separate inspection and exact artifact verification.
File suffixes, model names, and successful conversion cannot supply that evidence.

## Scientific and compatibility boundaries

Manwe remains an alpha research workbench.
Preserve the current package versions and disabled registry publication.
Follow the release guide before any release claim or tag.

Synthetic tracking results are regression evidence, not measured real-world accuracy.
Device availability and compilation do not establish a real model forward or cross-device parity.
The native Candle graph still needs a compatible real artifact and golden forward fixture.
CUDA execution and affected Windows workflows remain unqualified.

An acoustic direction estimate does not supply range.
Multi-camera covariance currently assumes analytically exact calibration.
Preserve the exact capture-time and calibration admission rules for moving targets.
Do not interpret pixel-only covariance as complete real-camera uncertainty.

Manwe has no implemented downstream adapter for CREBAIN, Galadriel, Engram/NCP, Prisoma, or pid-rs.
Historical integration observations remain dated evidence.
A new adapter needs exact consumer fixtures for schemas, shapes, units, frames, clocks, identity, lifecycle, and missingness.
Tracking frames are autocorrelated; rectangular arrays alone do not establish valid PID samples.

Accuracy and export agreement precede performance comparisons.
Record timing scope, warm-up, synchronization, hardware, variance, and model identity.
Do not rank backends using the removed, incomparable historical measurements.

## Documentation and verification

Use American English, active voice, and one term for each concept.
Follow an ASD-STE100-aligned style without claiming full standard compliance.
Keep procedural sentences within 20 words and descriptive sentences within 25 words where practical.
Put necessary conditions before commands.
Preserve exact schema, wire, model, and historical identifiers.

Keep README concise and detailed contracts in their owning guides.
Define each mathematical symbol, unit, assumption, and operating bound.
Use native SVG with accessible descriptions, readable mobile layouts, and direct zoom links.
Check rendered text, diagrams, and links after changes.

Use the commands in CONTRIBUTING, Makefile, and CI for the changed surface.
Run documented lightweight examples when their instructions change.
Check package inventories when adding files inside a distribution boundary.
Retain exact code parity when a documentation milestone relies on unchanged implementation evidence.
Report executed checks, unavailable host lanes, and remaining operational gates separately.
