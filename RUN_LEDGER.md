# RUN_LEDGER.md — ordered execution plan

Budget: **~20 runs / ~128 GPU-hours**. 1 training run = 6.4 h; 1 full-val LM evaluation = 0.3 h.

Ordered by **information gain per hour**, not accuracy gain per hour. Those differ: a run that
resolves a confound or puts an error bar on the headline claim yields no WER improvement and is
still among the most valuable things to spend an hour on. Where they conflict, information wins.

**All zero-retrain work is exhausted before any training run is spent** — Tier 0 delivers most of
the predicted accuracy gain for 9.6% of the budget.

---

## Tier 0 — no training runs (12.3 h, 9.6% of budget)

| # | Run | Hypothesis | Isolated variable | h | Information gain |
|---|---|---|---|---|---|
| 1 | **Vectorize streaming smoother + day layer** | 70% of measured streaming cost is implementation overhead, not model compute | implementation only, bit-identical output | 0.2 | **High** — every latency claim in the paper depends on the harness not overstating cost by 3× |
| 2 | **Build `lm_decoder`; drive incrementally; measure per-frame cost** | The incremental WFST path fits inside 79 ms | batch → incremental driving | 2.0 | **Critical / GATING** — establishes the constrained baseline. No other WER number means anything until this exists |
| 3 | **Per-phase greedy PER (R3 step 1)** | One model serves all 4 patch phases (`random_cut=3` already covers 3 of 4) | patch phase offset | 0.5 | **High** — decides R3 for 30 min; determines whether it costs 1.5 h or 7.9 h |
| 4 | **Blank skipping, θ ∈ {0.999, 0.99, 0.9}** | 71.8% of frames are skippable at ~zero accuracy cost | `blank_skip_thresh` | 0.3 | **High** — measured skippable fraction is already known; this confirms the accuracy is free |
| 5 | **`acoustic_scale` × `blank_penalty` × `beam` sweep** | Decode hyperparameters are untuned (repo's own two configs disagree 10× on blank penalty) | decode hyperparameters | 3.0 | **High** — largest no-retrain WER gain; must be re-run after #6 and #9 |
| 6 | **Build + evaluate pruned 4-gram ≤ 19 GB** | A 32 GB-feasible 4-gram beats the shipped 1-gram by >1 WER pt | LM order | 1.0 | **High**, but the least trustworthy estimate in the catalog (measured against a 1-gram) |
| 7 | **Endpointing policy sweep** | Finalization can shed ~1.68 s of trailing silence for <0.2 WER pts | endpoint threshold | 0.3 | **Medium** — pure latency; needed for an honest finalization number |
| 8 | **R3 merge: interleaved vs phase-averaged** | Interleaving cuts L_buf 60→15 ms; averaging buys ≥0.15 WER pts | merge mode | 1.0 | **High** — separates R3's latency mechanism from its accuracy mechanism |
| 9 | **Causal neural LM shallow fusion** | A small causal LM recovers part of the deleted OPT gain incrementally | fusion LM + weight | 1.5 | **High** — the main way back toward the unconstrained reference |
| 10 | **Adaptive-compute gate on LM confidence** | Gating on n-gram confidence recovers most rescoring gain at low *expected* latency | gate threshold | 2.0 | **Medium-High** — the frontier curve, but reports expected not worst-case latency |
| 11 | **TTA (DietCORP-style, per-trial)** | 18 ms/trial adaptation fits the budget and helps held-out sessions | test-time adaptation | 1.5 | **Medium** — no retrain; breaks streaming determinism, so report separately |

**Tier 0 subtotal: 13.3 h.** Predicted cumulative effect: constrained WER ~5.5% → **3.2–4.0% [E]**,
with a full latency and stability accounting. This is R1 in its entirety.

---

## Tier 1 — training runs, ordered (108.8 h)

| # | Run | Hypothesis | Isolated variable | h | Information gain |
|---|---|---|---|---|---|
| 12 | **Bandwidth-matched causal control** | L=0's advantage is a *bandwidth* effect, not a causality effect | smoother σ_eff at fixed lookahead=0 | 6.4 | **Critical** — the project's headline result is confounded ([AUDIT F2](AUDIT.md#f2)). Both outcomes publishable; not knowing is not |
| 13 | **Time masking, `max_frac`=0.075, seed 10** | ~53% masking cuts val PER ≥0.29 pts | time masking | 6.4 | **High** — largest single-change gain available, zero latency cost |
| 14 | **Causal normalization, seed 10** | Removing the block-z-score leak costs 0.3–1.5 PER pts | causal rolling normalization | 6.4 | **Critical** — highest-novelty item in the catalog; makes every latency claim defensible |
| 15 | **Causal normalization, seed 11** | (replication for the 2-of-3 gate) | seed | 6.4 | **Critical** — the headline claim needs an error bar |
| 16 | **Time masking, best `max_frac` from {0.05, 0.10}** | mask fraction is tunable | mask fraction | 6.4 | **Medium-High** |
| 17 | **Time masking, best config, seed 11** | (replication for the H4 gate) | seed | 6.4 | **High** — gates R4 |
| 18 | **`causal_la0` replication, seed 11** | The 0.163-pt la0-vs-la4 effect exceeds seed noise | seed | 6.4 | **Critical** — Fact A currently has n=1 and no error bar |
| 19 | **`causal_la0` replication, seed 12** | (third seed for 2-of-3) | seed | 6.4 | **High** |
| 20 | **Intermediate / self-conditioned CTC** | Auxiliary CTC heads with feedback cut PER ≥0.3 pts, causally | intermediate CTC supervision | 6.4 | **Medium-High** — strong source evidence, zero inference cost |
| 21 | **Channel / electrode masking** | Electrode masking adds on top of time masking | channel masking | 6.4 | **Medium** — may not add; targets the same error mass |
| 22 | **Label smoothing / entropy regularization** | A better-calibrated posterior gives the beam search more to work with | CTC calibration | 6.4 | **Medium-High** — the rare case where PER and WER should move in *opposite* directions |
| 23 | **Snapshot ensemble (cyclic LR)** | N members from one run capture ~⅔ of a true ensemble | LR schedule | 6.4 | **Medium** — confounds with the optimizer sweep |
| 24 | **Checkpoint averaging** (flip `save_all_val_steps: true`) | Weight averaging removes the min-over-61 selection bias | checkpoint averaging | 6.7 | **Medium** — free if the flag is flipped on any run above |
| 25 | **Intermediate CTC, seed 11** | (replication) | seed | 6.4 | **Medium** |
| 26 | **Stochastic depth / drop-path** | Drop-path adds on top of `rnn_dropout=0.4` | stochastic depth | 6.4 | **Low-Medium** |

**Tier 1 subtotal: 108.8 h across 15 runs. Running total: 122.1 h / 26 line items / 15 training runs.**

---

## ═══════════ CUT LINE — 122.1 h of ~128 h ═══════════

Everything above fits. Everything below does not, in priority order for promotion.

| Rank | Candidate | h | Why it is below the line, and what would promote it |
|---|---|---|---|
| 1 | **Causal transformer** (arXiv:2507.02800) | 25.6 | Highest ceiling of any retrain (−1.4 WER **[E]**) but 20% of budget at Medium-High risk: new architecture means re-deriving the streaming-equivalence gate, the day-layer plumbing and KV caching. **Promote immediately if run #13 (time masking) fails** — that failure would prove the source paper's gain belongs to the architecture, not the masking |
| 2 | **Deep ensemble N=3** | 12.8 | Deliberately deferred until after the regularization block so members differ by seed on a *better* base model. Promote if Tier 1 lands and ≥13 h remain. A null result here is publishable — see [RECOMMENDATIONS](RECOMMENDATIONS.md) |
| 3 | **Cross-dataset pretraining on T12** | 19.2 | Source paper measures 1.1 PER pts on T15 but explicitly notes T15 gains least, and their result confounds joint training with hierarchical CTC and a bidirectional 2048-wide GRU. Promote if run #20 (intermediate CTC) succeeds — that would de-confound the attribution |
| 4 | **`smooth_kernel_std` sweep at L=0** | 19.2 | Evidence-backed but directionally ambiguous: L=0's result argues narrower, Wairagkar's 1.5 s causal kernel argues much wider. Promote if run #12 shows the L=0 advantage *is* a bandwidth effect |
| 5 | **Larger patch (22 / 34)** | 12.8 | Cheap and evidence-backed (7th-place used both) but ΔPER is small and it grows `weight_ih_l0` 2.4× |
| 6 | **Optimizer sweep** (`eps`, weight decay, LR) | 25.6 | `eps=0.1` is 10⁷× default and untuned; the '24 retrospective credits LR tuning. Below the line only because it is 4 runs for a modest expected gain |
| 7 | **Diphone auxiliary head** | 12.8 | Credited by the '24 retrospective, but in a bidirectional setting, and the reported diphone GRU (8.39%) did not beat a plain time-masked causal transformer (8.18%) |
| 8 | **Augmentation std sweep** | 25.6 | 4 runs to tune two inherited constants that survived the original authors' own tuning on this exact dataset |
| 9 | **Deep ensemble N=10** | 57.6 | **Declined outright** — see [RECOMMENDATIONS](RECOMMENDATIONS.md). 45% of budget, sublinear return, zero novelty, and it improves the stage already 40× under budget |
| 10 | **Mamba / SSM, wider-deeper GRU, lookahead conv, mixup, day-adversarial, dynamic chunk, predictive distillation** | 96+ | Individually low or negative expected value; several are contested or pre-empted |

### Killed by measurement — not below the line, removed entirely (51.2 h freed)

| Candidate | h freed | Killed by |
|---|---|---|
| Delay-penalized CTC, Bayes-Risk CTC, Peak-First CTC, TrimTail, Align-With-Purpose | 38.4 | **Measured learned emission delay = +4.64 ms** (6% of one frame) over 13,030 paired tokens, after subtracting the smoother's predicted group delay ([BUDGET §3](BUDGET.md)). There is no delay to remove |
| Reduced patch size to cut cold start | 12.8 | **Median time-to-first-token = 3,380 ms** vs a 280 ms cold-start fill ([BUDGET §2](BUDGET.md)). The fill is entirely absorbed by the pre-speech period |
| FastEmit, delay-penalized transducer | — | **Transducer-only.** There is no transducer in this system; inapplicable before the measurement even applies |

Reviewer prior S6 ("measure before you fix") is what freed these. It was the highest-value
instruction in the brief.

---

## Dependency graph

```
#1 vectorize ──────────────────────────────────► honest latency numbers
                                                          │
#2 build lm_decoder + incremental ──► CONSTRAINED BASELINE ─┤ gates every WER claim
        │                                                   │
        ├──► #4 blank skip ──┐                              │
        ├──► #5 decode sweep ┼──► #6 4-gram ──► #5 AGAIN ────┤
        │                    │        │                      │
        │                    │        └──► #9 neural LM ──► #5 AGAIN
        │                    │                    │
        │                    └────────────────────┴──► #10 adaptive gate
        │
        └──► #7 endpointing ──► finalization latency

#3 per-phase PER ──► H3a? ──yes──► #8 merge ──► R3 complete (1.5 h)
                          └──no───► retrain random_cut=4 (+6.4 h)

#12 bandwidth control ──► resolves whether Fact A is a causality or bandwidth result
#14,#15 causal norm ────► the true price of causality (needs 2 seeds)
#18,#19 la0 seeds ──────► error bar on the headline claim
#13,#16,#17 time mask ──► better operating point for the whole frontier
```

**#5 appears three times deliberately.** The optimal `acoustic_scale` moves whenever the LM
changes, and re-sweeping is cheap (cached logits, no acoustic re-run). Failing to re-sweep after
#6 and #9 would understate both.

---

## Budget summary

| Tier | Line items | Training runs | Hours | % of 128 h |
|---|---|---|---|---|
| Tier 0 (no retrain) | 11 | 0 | 13.3 | 10.4% |
| Tier 1 (retrains) | 15 | 15 | 108.8 | 85.0% |
| **Total above cut** | **26** | **15** | **122.1** | **95.4%** |
| Freed by S6 measurement | — | (6) | (51.2) | — |
| Declined (N=10 ensemble) | — | (9) | (57.6) | — |

15 training runs, not 20 — because Tier 0 does the accuracy work that Tier 1 would otherwise have
been asked to do, and because measurement removed 6 runs before they were spent. The remaining
~6 h of slack is deliberate: reserve it for the contingent R3 retrain (#3 failing H3a) or a third
seed on whichever Tier 1 result lands closest to its gate.
