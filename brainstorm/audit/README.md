# brainstorm/audit — evidence archive

These are the Phase A–E audit documents (committed `1f6b52b`, 2026-08-06). They are **evidence,
not a plan**. Every decision they support has been consolidated into `/PLAN.md`, which is the
single source of truth for what to change and in what order.

Read a file here only when you need the derivation behind a number `PLAN.md` cites, or when a
claim in `PLAN.md` looks wrong and you want to check the source before contradicting it.

| File | What it holds |
|---|---|
| `AUDIT.md` | Pipeline forensics, findings F1–F16, each with `file:line` and a causality verdict |
| `BUDGET.md` | Per-stage latency ledger; emission-delay and posterior-entropy measurements; headroom |
| `LITERATURE.md` | External recon B1–B5, with retrieval dates and transfer verdicts |
| `CANDIDATES.md` | 40 scored mechanisms, master ranking, adversarial verdicts on reviewer priors |
| `candidates.json` | Machine-readable source of truth for `CANDIDATES.md` — regenerate both together |
| `RECOMMENDATIONS.md` | R1–R4 with pre-registered decision rules; the declined N=10 ensemble |
| `OPEN_QUESTIONS.md` | Q1–Q14, ordered by how much downstream work depends on the answer |
| `RUN_LEDGER.md` | The superseded 15-run / 108.8 GPU-h one-variable-per-run plan |
| `RESEARCH_PROPOSALS.md` | Earlier (2026-07-08) config-level proposals, pre-audit |

**`RUN_LEDGER.md` is superseded.** `PLAN.md` executes the same catalog in 6 training runs instead
of 15 by batching mechanistically-distinct changes. Do not work from the ledger's ordering.

**`RESEARCH_PROPOSALS.md` predates the audit** and disagrees with it in one place worth knowing:
it recommends *widening* `smooth_kernel_std` (2.5, then 3.0), while `AUDIT F2` implies *narrowing*
is what produced the L=0 result. `PLAN.md` R-D1 is the run that adjudicates; do not sweep σ before
it lands.

Cross-links between these files are relative and still resolve, since they moved together.
