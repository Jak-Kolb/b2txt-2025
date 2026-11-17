# History: superseded documents

Archive of superseded plans, reviews, logs and memos. Nothing here is a current instruction. Read a file only to
trace a specific past decision or measurement. The current plan is `docs/PLAN.md`; current results are in
`current_knowledge.md`, whose section 12 gives the development timeline these files belong to.

Every file here is tracked in git. When a document is superseded, move it here (`git mv` if it is tracked) and add
a row below.

| File | Dated | What it is | Superseded by |
|---|---|---|---|
| `audit_2026-08/AUDIT.md`, `BUDGET.md`, `LITERATURE.md`, `README.md` (was `brainstorm/audit/`) | 2026-08-06 | The August real-time latency audit: source forensics and preprocessing hypotheses, component timing budget, literature notes | Several interpretations were qualified by the September review; timing is now measured by the harness |
| `log_2026-08.md` (was `log.md`) | 2026-08-06 to 08-07 | Work log of the August block: 4-gram build (C28), neural rescoring (C29), masking (G2) | `RESULTS_2026-08.md` |
| `PLAN_2026-08.md` | 2026-08 | The complete August plan: ground state, change list, experiment groups 0-4, results ledger, excluded ideas | `PLAN_2026-09.md` |
| `RESULTS_2026-08.md` | 2026-08 | August results write-up: 42.42% to 6.89% val-dev WER, accuracy/latency frontier | `results/RESULTS.md`, then `current_knowledge.md` |
| `RESEARCH_ASSESSMENT_2026-09-18.md` | 2026-09-18 | Validity review: former val-test took part in checkpoint selection; the headline included sentence-final processing; the normalization proof did not justify the proposed experiment; the old improvement chain mixed denominators | Its constraints are carried in `AGENTS.md` and `.claude/rules/` |
| `DEVELOPMENT_READY_2026-09-18.md` | 2026-09-18 to 09-22 | Environment and git inventory snapshot taken when development moved to JakPC | `AGENTS.md`, `docs/PROTOCOL.md`, `current_knowledge.md` 3.7 |
| `PLAN_2026-09.md` (was `brainstorm/PLAN.md` as committed 2026-10-12) | 2026-09-22 to 10-12 | The September plan: milestones for output capture, paced replay, the harness, the ECoG track, the first commitment baseline and first causal Transformer | The trimmed `brainstorm/PLAN.md` |
| `model_dataset_design_2026-10-07.md` | 2026-10-07 | Long explainer of the dataset and of the reference pipeline (A2, then named `causal_la0`: 5x768 GRU, 4-gram, GPT-2 rescoring) | `current_knowledge.md` sections 2 and 3 |
| `ANALYSIS_PLAN_2026-10-11_anchors.md` (was `brainstorm/`) | 2026-10-11 | Plan for the causal / bidirectional anchor models and the look-ahead sweep | `current_knowledge.md` 4.1, 4.2 |
| `RESEARCH_AGENT_PROMPT_2026-10-11.md` (was `brainstorm/`) | 2026-10-11 | Prompt given to an outside research agent | - |
| `Intracortical_Speech_Research_Directions.md` (was `brainstorm/`) | 2026-10-11 | That agent's research-direction memo | `current_knowledge.md` sections 9 and 10 |
| `agent_research_2026-10-27.md` (was `brainstorm/agent_research.md`) | 2026-10-27 | Outside agent's assessment and experiment plan for making the Transformer competitive (shared input layer, hidden-state transfer from the GRU, controls) | `current_knowledge.md` 4.8 and 10.4 |
| `lightbeam_reproduction_gaps_2026-11-10.md` (was part of `current_knowledge.md` 4.8) | 2026-11-10 | The two tables comparing LightBeam's public code (commit 90e0ac0) with our 2026-11-03 run, and the table of fixes prepared for the corrected recipe with its arms and read-out rules | `current_knowledge.md` 4.9 ("What the corrected recipe (v2) changed") and the Y6 / Y7 / Y8 results |
| `transformer_pilots_2026-10-26.md` (was part of `current_knowledge.md` 4.8) | 2026-10-26 to 10-27 | Conv-Transformer block tests (Macaron, 800-batch), the 8k-batch pilot curves and the X3 training curve | `current_knowledge.md` 4.8 (summary) |
| `transformer_tests_explanation_2026-11-10.md` (was `transformer_tests_explanation.md` in the repository root, untracked) | 2026-11-10 | Pre-run explanation of the corrected LightBeam-recipe runs (Y6-Y8) and the other prepared runs | `current_knowledge.md` 4.9, 4.10 |

Not archived here:
- `results/RESULTS.md` stays beside the run artifacts it cites (its source paths are relative to `results/`). It
  is the record of the September measurements: paced replay, harness baselines, commitment policies, first
  rescoring and first Transformer.
- Dated analysis plans and idea files from 2026-10-13 to 10-26 were never committed. They were folded into
  `current_knowledge.md` and deleted on 2026-10-26; no copies exist.
