# AGENTS.md — b2txt-2025

Fork of `Neuroprosthetics-Lab/nejm-brain-to-text` (Card et al. 2024, NEJM). Upstream line numbers
still match; edits are additive.

## 1. The project

Convert the Card et al. pipeline into a **fully causal, real-time streaming decoder**, then find the
best accuracy achievable *subject to* a hard real-time constraint.

The original question ("what does causality cost?") is **not** answered — the L=0 vs L=4 comparison
is confounded, and the released features turned out to be non-causally normalized. The live
questions are **"what is the causal-constrained accuracy optimum?"** and **"what does end-to-end
causality actually cost?"**

Write-up targets: ICASSP, SLT, EMBC, NeurIPS neuro tracks. Not a leaderboard entry.

**Open decision — do not assume either way.** Whether the deliverable is (a) an accuracy-vs-latency
**frontier** paper, (b) **best accuracy at fixed latency**, or (c) a **systems-and-measurement**
paper on what real-time costs once the LM is priced. It changes experiment ordering. If a task's
sequencing depends on it, ask rather than picking.

### The constraint that defines everything

| | |
|---|---|
| End-to-end latency budget | ~140 ms = `L_algo` + `L_buf` (0–60 ms, mean 30) + `L_comp` |
| Acoustic side, worst case | **60.3 ms of 140** — leaves ~79 ms per frame for the LM |
| LM stack, as shipped | **620–830 ms finalization = 4.4–5.9× the whole budget** |
| Emission | streaming; no full-utterance right context anywhere |

`L_algo = 20·L` ms where `L = smooth_lookahead` in 20 ms bins. **80 ms is the update cadence, not
the emission latency** — conflating them overstates `L_buf` and understates the LM's share.

**Ledger rule.** A latency claim is only real if it prices *every* stage, including LM decode and
rescoring. An acoustic-only ledger is incomplete — say so rather than accepting it.

---

## 2. Current state — 2026-08-07

**WER chain** (val-dev, incremental, no batch stage): 42.42 % shipped 1-gram → 35.51 % after decode
sweep → **8.49 %** with the C28 4-gram → **6.89 %** with C29 gpt2-large rescoring. Oracle over the
100-best is **2.88 %**, against a published full-stack 2.66 % **[unverified]** — but 1.51 pts of
that oracle is edit-distance partial credit no language model can claim, so the **LM-reachable
floor is ~4.39 %** (§1.13).

Per-frame LM cost p95 **0.641 ms** against 79 ms; finalization ~96 ms p95 against 140 ms.

**Evaluated operating point:** `results/lm_gen_p1e-8/data/lang_test` (3.94 GB TLG) + gpt2-large
rescoring at **γ=0.25, α=1.0, full n-best**. This is what val-test measured, and **it fails the
latency constraint there** — ~158 ms p95 against the 140 ms budget, because the harder sessions
produce longer lists (§1.14). Report that as the held-out result; it is the finding.

**top-10 is a mitigation, not a validated operating point.** It lands at ~83 ms and costs 0.15 pts,
but that number came from running a *second* config on the one-shot split, so its held-out accuracy
is **unverified** and there is no held-out set left to verify it. Do not quote 19.46 % as the
shipped system's held-out WER. Any future one-shot spend must **pre-register a latency gate** with
the same specificity as its accuracy gate — this one had none.

**Acoustic anchors:** `causal_la0` 10.04 % val PER, `causal_la4` 10.21 %, against a **0.041-pt**
replicate floor. Checkpoints live in `results/`, not `trained_models/`.

**Done:** Group 0; Group 1 (all gates, 0 training runs); Group 2 (1 run — **rejected on WER**);
C28; C29; C29 error analysis; C24 (**rejected**, 0 GPU-h); G3a (1 run — **NEUTRAL, not carried**).
Training runs spent: **2**.

**`val-test` (173 trials, 6 held-out sessions) is SPENT — 2026-08-08, once, on the frozen stack**
(§1.14). **18.84 % → 16.10 %, unseen 19.31 % [95 % CI 15.17–23.49].** Do not touch it again.

Two facts make val-test **not comparable** to val-dev, and any table showing 18.84 % beside 8.49 %
without them is misleading: the 6 held-out sessions are **2.03× harder acoustically** (per-session
PER 21.06 % vs 10.39 %), and **43.4 % of their references are verbatim-in-train vs 6.9 % of
val-dev**, so only the unseen row means anything. The number that *did* transfer is the rescorer's
headroom recovery: **26 % vs 29 %** — the LM stage is not overfit to val-dev.

**The strategic consequence.** The acoustic model is 40× under its compute budget while the LM was
4–6× over it. **The LM/decoding stage is the entire remaining budget.** Any optimization aimed at
the GRU before pricing the decoder is aimed at the wrong stage.

---

## 3. Next

**The rescorer error analysis is DONE (2026-08-07, `PLAN.md` §1.13) and it re-orders this list.**
The surviving headroom is not the LM being out-voted by the acoustic term — magnitudes are
comparable (1.1×), the (γ, α) optimum is a true interior minimum, and raising α breaks more trials
than it fixes from the first step. On the majority of the surviving mass gpt2-large **confidently
prefers the wrong candidate**. Two things follow:

- **1.51 of the 6.89 pts was never LM-reachable**: both hypotheses are wrong and the oracle pick is
  merely edit-*closer* while being worse English. The **oracle-over-n-best overstates LM-reachable
  headroom** — a manuscript-worthy caution about a metric the ASR literature quotes freely. The
  real floor on this list is **~4.39 %**, so rescoring has recovered **39 %** of what is reachable,
  not 29 %.
- **Shallow fusion (C29b) is downgraded, and must not be built as specified.** Fusion admits
  candidates *by neural score*, and high neural score is anti-correlated with correct on exactly
  this mass — so fusing gpt2-large into the beam is predicted to *lower* the oracle, the one thing
  rescoring cannot do.

**Gate N1 (`nbest` 100 → 300) also landed and is REJECTED** (§1.13). Oracle 2.88 → 2.57 % but
rescored WER only 6.89 → 6.80 %, at ~150 ms p95 finalization — **over the 140 ms budget**. The
rescorer converted 0.09 of the 0.31 new pts: **a 29 % conversion rate, the same 29 % it achieves
against any amount of material.** That makes conversion a property of the ranker, not the list.

**Every LM-side lever is now exhausted** — scale, domain adaptation, (γ, α) weighting, length
penalty (β=−1 is worth 0.025 pts), and list depth. The scorer is a linear combination of
(acoustic, 4-gram, neural, length) and that family is swept to its optimum, so **discriminative
re-ranking over these features cannot pay** — it would need a genuinely new feature, and the one
the analysis says is missing is *acoustic* discrimination. **The LM stage is done; the remaining
reachable error is acoustic.**

**G3a landed 2026-08-08 and is NEUTRAL** (§6.4): unseen WER 8.66 % vs 8.72 %, PER 10.040 → 9.922.
Per the pre-registration, **C22/C23 are not carried** — Group 3 contributes nothing to the stack and
the base model stays `causal_la0`. C20 dropped, C24 rejected, **C21 still unbuilt** and still not a
config change (`rnn_model.py:65-72` is one fused `nn.GRU(num_layers=5)`; self-conditioning needs a
rewrite that breaks every checkpoint and forces a re-run of gate G1-a).

**The finding that outlives the run: 21.4 % of reachable decisions are acoustically UNDECIDABLE.**
Where the correct candidate is in the n-best, its acoustic score is *bit-identical* to its best
competitor 21.4 % of the time — homophones (`too`/`to`, `for`/`four`, `can't`/`cant`), where CTC
emits the same phoneme sequence. No acoustic model of any size can separate them. So §1.13's "the
remaining reachable error is acoustic" needs a qualifier: **at most ~48 % of it is** (where the
acoustic score actively prefers the wrong candidate), 21 % is permanently the LM's problem, and
31 % is already acoustically correct but overridden. **Gate future acoustic work on the 48 %
subset**, not on aggregate WER, which dilutes the signal with trials the model provably cannot move.

In priority order:

1. **`val-test`** on the final stack — the acoustic and LM sides are both now measured out, so
   there is little left to change before spending it.
2. **Group 4 validity runs** (R-D1/R-D2/R-D3). C8 left the variance floor for sparse channels as an
   open design choice — decide it before the run, not after. **R-D1 (C25) is the one that cannot be
   killed** — it resolves the project's central confound, and either answer is publishable.
3. **C27 seed replication** — promoted. Every WER delta in these files rests on a single seed and
   there is still **no measured WER noise floor**; the 0.041-pt figure is PER only. G3a's −0.06 and
   G2's +0.61 are both quoted against nothing.
4. **Larger C28 corpus** — blocked on RAM, not method.

### Do not re-propose without new evidence

Each was killed by measurement in this repo, not by argument.

1. **Perplexity does not predict WER here (4 for 4).** The in-domain 4-gram had 4× worse ppl and won
   by 19 pts; fine-tuning gpt2-large improved ppl 14× and made unseen WER *worse*.
2. **In-domain adaptation buys memorization, not generalization (4 for 4).** 6.94 % of val-dev
   sentences appear verbatim in train — always split seen/unseen and report unseen.
3. **PER is the wrong gate.** G2 passed its PER gate (10.04 → 9.43) and lost on WER (8.49 → 9.10):
   masking sharpened the posterior and starved the beam.
4. **Bigger rescorers.** Plateaus at ~39 % of *reachable* headroom; Qwen2.5-1.5B is worse than
   gpt2-large at 2× the parameters and 2.5× the latency. §1.13 says why the plateau is there — the
   LM is confidently wrong, not out-voted — so more parameters at the same objective cannot move it.
5. **Re-weighting the rescorer.** The (γ, α) optimum is a verified interior minimum on a fine
   10 × 13 grid; α 1.0 → 1.5 fixes 36 trials and breaks 62. There is no weight left to find.
6. **Delay-penalized / Bayes-Risk / Peak-First CTC, TrimTail, Align-With-Purpose** — excess learned
   emission delay is **+4.64 ms**, 6 % of one frame. Nothing to remove.
7. **Reduced `patch_size` for cold start** — median time-to-first-token is 3,380 ms vs a 280 ms fill.
8. **Left-padding the patcher; deep ensembles at N=10; wider/deeper GRU.** Full reasoning and the
   rest of the excluded list: `brainstorm/PLAN.md` §9.

---

## 4. Working norms

- **No execution without approval.** No training launches, no long-running experiments without an
  explicit go.
- **Source-first verification.** Grep the source before asserting any pipeline property; cite
  `file:line`. Never assert from memory of a paper or from the shape of the code.
- **Push back.** If a request rests on a wrong premise, fix the premise first. Do not implement a
  plausible-looking version of a bad idea.
- **Traceable deltas only.** Every claimed improvement needs a mechanism and a magnitude.
- **Grouped batching, not one-variable-per-run** — but only bundle mechanistically distinct,
  **same-signed** changes, and never bundle opposed ones. Validity runs stay single-variable.
- **Pre-register the gate**: config, metric, threshold, and what each outcome implies, *before*
  launching.
- **A measurement beats any entry in these files.** When one does, correct the entry in the same
  commit rather than leaving both versions in play.

---

## 5. Where everything else lives

| Need | Where |
|---|---|
| Plan, change list, results ledger, excluded list | `brainstorm/PLAN.md` |
| Public results summary | `RESULTS.md` |
| Evidence archive (audit, literature, candidates) | `brainstorm/audit/` |
| Acoustic pipeline facts — smoothing, patching, normalization | auto-loads on `model_training/**` |
| LM traps, build limits, decode tuning | auto-loads on `language_model/**` |
| Splits, seen/unseen, PER-vs-WER, latency accounting | auto-loads on `benchmark/**` |
| Running a training job | skill `train-model` |
| Building or measuring an LM | skill `build-lm` |

**Environment:** use `.venv` (uv, py3.10) for everything training and benchmark side. conda exists
for exactly one thing — `setup_lm.sh`, which builds the py3.9 env holding `lm_decoder`.
**As of 2026-08-07 the conda install is at `~/.miniforge3` (dot-prefixed) and `conda` is NOT on
PATH**, so `$(conda info --base)` no longer resolves. Call the interpreter by absolute path:
`/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python`. Export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for benchmark entrypoints; without it the
allocator pages over PCIe and batches go 0.169 → ~1.5 s.
