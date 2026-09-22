# B2T project instructions

## Location and reading order

Develop on JakPC at /home/fishinjak/code/nejm-brain-to-text.
From the Mac use ssh jakpc with explicit commands. The interactive jakpc function
attaches a shared tmux session; do not send automation into it. The older Mac
checkout is a reference copy. This project is separate from brainBERT_test.

Inspect the current branch, git status, and relevant diffs before editing.
Start with brainstorm/PLAN.md; read RESULTS.md for existing measurements.
RESEARCH_ASSESSMENT.md records the September 18 validity review, not a blanket
pause on development. DEVELOPMENT_READY.md is a dated environment snapshot.
Read historical records only when needed to understand a specific decision.

Before touching the relevant code, explicitly read:
- model_training/**: .claude/rules/acoustic-pipeline.md.
- Benchmark code or splits.py: .claude/rules/measurement.md too.
- language_model/**: .claude/rules/language-model.md.
- Authorized training/build work: the corresponding .claude/skills/*/SKILL.md.

## Objective

Build and measure an efficient real-time neural speech-to-text pipeline on
B2T '25 intracortical data and the Makin lab's ECoG dataset. B2T is available;
lab access, schema, task, alignment, and baseline remain to be established.
Use modality-appropriate preprocessing and encoders with a shared measurement
framework where meaningful. Do not assume interchangeable inputs or pool scores.

The study is exploratory. Temporal context, neural model, LM search/rescoring,
word commitment, and runtime implementation are candidate levers within pipeline
development. Start from the existing GRU and incremental decoder. Choose small
comparisons from measured bottlenecks; do not predetermine a winning method or a
commitment-only paper. Architecture changes need a concrete question and baseline.

The slide's 200 ms, 500 ms, 1.5 s, and unconstrained comparisons are proposed
evaluation conditions. Define their clocks before adoption. They are neither
proven human-factors thresholds nor a replacement for the historical 140 ms gate.

## Scientific discipline

- Separate observations, interpretations, and hypotheses. Trace numeric claims
  to artifacts and literature claims to primary sources. Report null results.
- The former val-test influenced acoustic checkpoint selection and is exposed.
  Do not call it an independent end-to-end holdout or reuse it without an explicit
  revised evaluation scope. Do not silently invent or rename a replacement split.
- Independent evaluation is needed for confirmatory claims; its absence does not
  block instrumentation or transparently scoped exploratory development.
- Amend the proposed absolute variance floor before implementing or launching
  R-D2/R-D3: it fails affine invariance. Other work can proceed independently.
- Released-feature replay does not establish raw-signal causality. Preprocessing,
  clipping, and reset provenance remain unresolved.
- Report WER with the partition, seen/unseen definition where applicable, and
  session/participant variation. PER is diagnostic. A single replicate difference
  is not an equivalence bound or noise distribution.
- Distinguish input availability, compute, queueing, update cadence, first output,
  word commitment, endpointing, and finalization. Measure integrated percentiles
  from paired events; never sum stage p95s. RTF < 1 is not a deadline guarantee.
- Keep references, future inputs, and retrospective alignment out of online
  decisions. Full-utterance rescoring remains a finalization stage.
- Lab recordings and transcripts require authorized handling. Keep private subject
  content out of logs, commits, and external tools.

## Execution

Preserve user changes, branches, data, configurations, checkpoints, and results.
Use fresh run paths. Never overwrite completed runs.
Proceed with edits and bounded synthetic tests within the requested development
scope. Training, downloads, LM builds, full-data evaluations, and long experiments
need task authorization and a bounded smoke check. Existing authorization suffices;
do not request approval again for routine steps within it.
Do not commit, push, contact collaborators, or transfer lab data unless requested.

Use .venv/bin/python for acoustic work, benchmarks, and neural rescoring, and
/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python for the LM extension.
Keep the environments separate; exchange arrays/events across an explicit boundary.
Do not rerun setup.sh to normalize the environment. Set
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for GPU benchmark entrypoints.
Inspect live resources before jobs; the readiness snapshot is not live status.

## Context discipline

AGENTS.md holds operating rules; brainstorm/PLAN.md holds the current work sequence;
RESULTS.md holds qualified observations. CLAUDE.md points here rather than
duplicating instructions. Keep old decisions in history. Add no speculative
rankings, invented effect sizes, or long rejected-idea catalogs. Negative results
apply to tested configurations, not whole method families.
