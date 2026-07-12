# Research

- **[SOTA-2026.md](SOTA-2026.md)** — *State of the Art: Vision, Audio,
  Multi-Camera & Sensor Fusion for the manwe Training Ground (2026)*. A synthesis
  of multi-source web research across six topics, **adversarially fact-checked** —
  claims that came back partially-supported or refuted are flagged inline with
  ⚠️ Caution notes. It records research recommendations, not an inventory of
  implemented or consumer-compatible behavior.
- **[sota-2026-structured.json](sota-2026-structured.json)** — the machine-readable
  research record. It deliberately preserves the original `keyClaims`, models,
  recommendations, and the later `verifications` audit. Raw claims are not facts:
  join an exact claim to its verdict, prefer `correctedClaim` when present, and
  treat a missing verdict as not independently reviewed. The corrected Markdown
  survey is the safer human-facing synthesis. `recommendations`/`whyForManwe`
  preserve proposals and are not current repository status.

## TL;DR — research directions versus implemented baseline

| Pillar | Research direction and current boundary | Key caveat surfaced by verification |
|--------|-----------------|-------------------------------------|
| **Vision** | Current: from-scratch architecture training + sliced inference. Planned ablations: RF-DETR/YOLO candidates, slicing-aided training, P2 head | RF-DETR **XL/2XL are PML-1.0, not Apache**; Ultralytics is **AGPL**; "tiling beats architecture" is overreach — they stack |
| **Audio** | Current: GCC/SRP-PHAT baseline. Planned: SELD ResNet-Conformer + Multi-ACCDOA and pretrained classifier | Mamba SELD is **CUDA-only**; acoustic range is weak (~50 m/node) → distributed, good-bearing covariances |
| **Multi-cam** | Classical **N-view triangulation** (DLT + midpoint + covariance), not learned BEV | Learned BEV is ground-anchored + often **CUDA-only ops**; triangulation needs **hardware time-sync** |
| **Fusion** | Hand-rolled KF/EKF/UKF/PF/IMM + GOSPA; Stone Soup as an independent cross-check where algorithms overlap | **Stone Soup has NO IMM or GLMB** (the original claim was fabricated) — hand-roll them yourself |
| **Training/export** | Proposed checkpoint→backend conversion and fidelity workflow | Manwe currently has raw ONNX/CoreML/TensorRT conversion, no implemented MLX converter, and no drop-in consumer adapter; INT8 accuracy must be measured |
| **Hardware** | Research recommendation: evaluate CUDA for large training and MPS for bounded development; no repository-wide performance ranking | Operator coverage and numerical behavior are model/version specific; validate on target hardware |

> The report is a point-in-time snapshot (mid-2026). Treat the ⚠️ flagged claims as
> unsettled and re-verify version/benchmark specifics on your target hardware
> before making performance or licensing commitments.

For implemented status and the schema/taxonomy/frame/shape gaps, the source of
truth is [the audited compatibility matrix](../INTEGRATION_CREBAIN.md), not this
research snapshot.
