---
paths:
  - "language_model/**"
---

# Language-model implementation

Read root AGENTS.md and the current PLAN.md before builds or measurements.

- Reuse the existing Python 3.9 LM environment; acoustic work uses .venv.
  Exchange array caches rather than importing the incompatible stacks together.
- The build recipe is uppercase end to end. Check corpus case and vocabulary.
- setup_lm.sh does not establish that all Kaldi command-line tools built correctly.
  Verify actual imports/binaries; use build_tlg.sh's paths if a build is authorized.
- A newly created TLG.fst may be empty while composition runs. Check completion,
  size, and errors. Never run concurrent builds into one output directory.
- LM composition has large RAM requirements. Inspect available memory before an
  authorized build and retain original models.
- Decode/LM weights depend on model and frame rate. Tune only on authorized
  development data, freeze before evaluation, and save exact settings.
- N-best extraction and neural rescoring are separate finalization stages in the
  current tools. Do not label their final WER a low-latency committed-text result.
- Prior negative sweeps describe tested configurations, not a ban on all fusion,
  adaptation, larger models, or reranking. New work still needs a specific rationale.
