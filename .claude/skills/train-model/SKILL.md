---
name: train-model
description: Configure or launch an authorized acoustic training run in the B2T research repository.
---

# Training

Read AGENTS.md, docs/PLAN.md, and .claude/rules/acoustic-pipeline.md.
Apply validity issues to the affected experiment: R-D2/R-D3 require an amendment;
unrelated authorized development is not blocked by that finding.

Before a costly run, establish its question, config, selection/evaluation data,
compute bounds, seed, and fresh output/checkpoint/log paths. Existing task
authorization is sufficient for routine steps within scope. Exploratory runs need
a stated rationale; confirmatory runs also need frozen criteria.

Use .venv/bin/python and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
Inspect live GPU/process/memory/disk state. Preserve existing runs and pinned
environments. Investigate failures before changing batch size or dependencies.

Smoke the actual configuration with a small explicit batch/time limit, covering
the relevant save, normalization, and evaluation paths. Choose the limit for the
change; do not inherit an expensive historical batch count. Diagnose nonfinite
loss and CTC infeasibility rather than masking them. Launch authorized long work
detached with a unique log after the smoke passes.

Record source revision and working diff, full config, data/split identities, seed,
checkpoint-selection rule, PER, runtime/resources, and authorized WER evaluation.
Keep artifacts immutable. Report uncertainty and negative results; old single-run
thresholds do not establish statistical equivalence.

Launch through `.venv/bin/python -m harness train` (harness/README.md); it smokes, trains,
registers and can benchmark the model. Multi-run queues follow results/harness/queue/.
Run one GPU process at a time, and no WFST decoding beside a GPU job (AGENTS.md).

Historical recipes in docs/history/PLAN_2026-08.md are evidence, not current launch
instructions.
