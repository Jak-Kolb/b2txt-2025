---
name: build-lm
description: Build or measure a language model for the B2T pipeline within an authorized experiment.
---

# Language-model work

Read AGENTS.md, brainstorm/PLAN.md, and .claude/rules/language-model.md.
Establish the operation, development data, compute limits, and fresh output paths.
Existing authorization covers routine steps within scope; downloads and large
builds must themselves be within the requested work.

Use /home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python for the LM extension;
use .venv for acoustic exports and neural rescoring. Verify the actual import.
Inspect build_tlg.sh and tool paths before a build. Check uppercase vocabulary,
RAM headroom, process exit status, and nonempty completed outputs. Preserve models.

Smoke a bounded decode before full work. Save exact settings and metric artifacts.
Former val-test needs an explicit revised scope for reuse; it is not independent.
Do not repeat legacy sweeps merely because historical documents show commands.

Report accuracy with the actual output stage and timing boundary. Current neural
rescoring consumes completed n-best lists; first-50 timing is not integrated
streaming latency. Search, rescoring schedules, and commitment are candidate
pipeline levers, not a predetermined paper thesis.
