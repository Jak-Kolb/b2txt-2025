---
paths:
  - "model_training/**"
---

# Acoustic pipeline

Read root AGENTS.md and brainstorm/PLAN.md first.

- Start B2T work from the existing GRU. Model/context changes are experimental
  levers, not the presumed paper contribution. Preserve saved-argument consistency.
- ECoG requires its own appropriate features, encoder, and baseline. Establish its
  actual schema before integration; do not treat intracortical inputs as equivalent.
- Smoother lookahead changes future access and filter shape. Causal support, peak
  location, and group delay differ. The bandwidth control is not equivalence proof.
- The right-edge patcher uses past data; its cadence is not word-emission latency.
  Do not change indexing or add left padding incidentally.
- Released-feature normalization and clipping provenance remain unresolved. The C8
  check covers approximate affine behavior only on included samples. Amend the
  proposed absolute std floor before implementing R-D2/R-D3; it fails invariance.
- The dataset does not yet apply the proposed causal normalizer. Integration must
  specify state/reset behavior and match offline versus streaming execution.
- Existing anchors used former val-test for checkpoint selection. A new protocol
  does not make old checkpoints independently evaluated.
- Check CTC feasibility and finite loss. Do not change zero_infinity or suppress
  failing samples incidentally. Keep saved configs and completed runs.
- Synchronize actual CUDA timing; training launch timing is not throughput.
