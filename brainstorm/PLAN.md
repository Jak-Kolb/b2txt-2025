# PLAN.md — grouped implementation and test plan

**Single source of truth for what to change, in what order, and how to test it.**
Everything needed to execute is in this file. The Phase A–E audit documents that produced these
decisions are archived under `audit/` as evidence only — you do not need to read them to work
a group. They are cited inline as `[AUDIT F1]`, `[BUDGET §3]` etc. when a number needs a source.

Generated 2026-08-06 from `AUDIT.md`, `LITERATURE.md`, `BUDGET.md`, `CANDIDATES.md`,
`RECOMMENDATIONS.md`, `RESEARCH_PROPOSALS.md`, `OPEN_QUESTIONS.md`, `RUN_LEDGER.md`.

Confidence tags used throughout: **[M]** measured · **[D]** derived · **[E]** estimated ·
**[U]** unverified.

---

## 0. How to use this file

Work **one group at a time**. For each group:

1. Implement every change in the group's change table.
2. Run the group's **verification block** (cheap, catches mistakes before you spend a run).
3. Run the group's **test block** (the actual measurement).
4. Fill in the group's row in the **results ledger** (§8).
5. Evaluate the **decision gate**. It tells you whether to keep the group, knock it out, or branch.

Do not start a group until the previous group's gate has been evaluated and recorded.

### What batching costs, and the deal we are making

This plan deliberately abandons the old "one variable per retrain" rule. That rule is correct for
*attribution* and wrong for *optimization throughput*: the run ledger it produced was 15 training
runs / 108.8 GPU-hours. This plan is **6 training runs / ~32 GPU-hours** (measured 5.3 h/run on
this machine, not the inherited 6.4) and expects to land at a
similar operating point.

The thing you give up is knowing *which* change inside a winning bundle did the work. That is a
real loss and you should not pretend otherwise in a manuscript. Three mitigations make it
acceptable:

- **Bundles are built from mechanistically distinct, same-signed changes.** Group 2 is all
  train-time data masking; Group 3 is all objective/optimizer. Nothing inside a bundle is expected
  to move PER in the opposite direction from its bundle-mates, so a bundle that wins, wins as a
  block, and a bundle that loses is bisectable.
- **Attribution is deferred, not deleted.** Leave-one-out ablations cost one run each and are only
  worth spending *after* you know the bundle wins and *only* for the changes a reviewer will
  challenge. See §7.6.
- **Validity runs are never bundled.** Group 4's runs each stay single-variable, because their
  entire deliverable *is* an isolated number. A bundled causal-normalization run measures nothing
  at all. This is not a style preference — it is that the run has no output if you bundle it.

**Every retrain group is preceded by a 2,000-batch smoke run (~7 min).** This is what makes
aggressive bundling safe: it catches OOM, NaN, divergence and shape errors for ~2% of a run's cost.
Skipping it is how you lose 5.3 hours.

---

## 1. Ground state

### 1.1 Measured — do not re-derive

| Quantity | Value | Source |
|---|---|---|
| val PER, `causal_la0` (L=0, causal) | **10.04 %** | training run |
| val PER, `causal_la4` (L=4, symmetric control) | **10.21 %** | training run |
| val PER, `baseline_rnn` (algorithmically identical to la4) | 10.25 % | training run |
| **Same-config run-to-run noise** | **0.041 PER pts** | [AUDIT F9] |
| Checkpoint-selection bias (min over 61 vals) | ~0.04 PER pts, consistent across runs | [AUDIT F8] |
| Streaming RTF (acoustic only) | 0.0132 | `metrics.json` |
| Per-patch compute p50 / p95 | 0.320 / 0.341 ms GPU · 1.348 / 1.438 ms CPU | [BUDGET §5] |
| **L_buf** | **0–60 ms, mean 30** (80 ms is the *cadence*, not the latency) | [AUDIT F3] |
| L_algo at L=0 | 0 ms | [AUDIT F2] |
| Acoustic total, worst case | **60.3 ms of the 140 ms budget** | [BUDGET §6] |
| **LM finalization (full stack)** | **620–830 ms — 4.4–5.9× the whole budget** | [BUDGET §1] |
| Frames with p(blank) > 0.999 | **71.83 %** | [BUDGET §5] |
| Mean posterior entropy | **0.033 nats = 0.9 % of ln(41)**; median frame numerically one-hot | [BUDGET §4] |
| Excess *learned* emission delay, la0 vs la4 | **+4.64 ms** over 13,030 paired tokens | [BUDGET §3] |
| Median time-to-first-token | **3,380 ms** into the trial | [BUDGET §2] |
| Acoustic partial-hypothesis revisions | **0 over 41,039 tokens** — flicker-free by construction | [BUDGET §4] |
| Batched ensemble cost, N=10, CPU | 7.05 ms = 8.8 % duty cycle; flat N=4→16 | [BUDGET §5] |

### 1.2 Corrections to previously-held beliefs

These were stated as settled and are not. Do not work from the old versions.

| Was believed | Actually |
|---|---|
| HDF5 features are causally normalized (0/45 sessions leak) | **Whole-block z-scored — non-causal.** The checker false-negatives because it thresholds a `max` over 512 channels. 1/√W scaling test matches prediction within 1–26 % across 4 sessions. [AUDIT F1] |
| `smooth_lookahead` is a clean causality knob | **Confounded.** L=4→L=0 removes 80 ms lookahead *and* narrows the low-pass by 37 % (σ_eff 1.852→1.161 bins) *and* raises peak weight 66 %. [AUDIT F2] |
| L=0 vs L=4 Δ is "within seed noise" | The only same-config replicate differs by **0.041 pts**; the effect is **0.163 pts** on the last-10 mean — ~4× larger. Seeds are needed to *confirm* a likely real effect, not bury a null. [AUDIT F9] |
| L_buf = 80 ms | **0–60 ms, mean 30.** [AUDIT F3] |
| Target is a pruned 3-gram at 6–8 GB | The repo's own README says the 3-gram needs **~60 GB**. Neither shipped n-gram fits 32 GB. Target is a **built, pruned 4-gram ≤ 19 GB**. [AUDIT F6] |
| The LM needs to be made streaming | **It already is.** `ctc_wfst_beam_search.cc:98` advances one frame at a time; `:113` extracts a partial best path. The *harness* is what's batch (`evaluate_model.py:237-250`). [AUDIT F4] |
| Streaming RTF measures the model | **~70 % of it is Python overhead** — per-tap FIR loop, linear ring-buffer scan, scipy kernel rebuilt per call. [AUDIT F5] |
| Test WER is the objective | **`data_test.hdf5` contains only `input_features`.** No labels of any kind; leaderboard closed. Test WER is unobtainable. [AUDIT F7] |

### 1.3 Unmeasured — these gate everything downstream

| # | Question | Blocks | Resolved by |
|---|---|---|---|
| Q1 | What does incremental WFST beam search cost per frame? | every latency claim | C13 |
| Q2 | What is the constrained-baseline WER (incremental n-gram, no rescore/OPT)? | every ΔWER in this plan | C13 |
| Q3 | Which LM produced the quoted 2.66 % WER? Only a 1-gram exists locally. | how much of 2.66 % is reachable | C13 / G0.5 |
| ~~Q11~~ | ~~Training VRAM at batch 64~~ | — | **RESOLVED: 7.35 GB of 16. Fits with ~9 GB headroom; do not reduce `batch_size`** |

### 1.4 C13 landed 2026-08-06 — what it changes

**Q1 is answered and the news is good.** The incremental WFST beam search costs **p50 0.072 ms,
p95 0.712 ms** per frame against a 79 ms budget — a **111× margin at p95** — and incremental
driving produces a byte-identical hypothesis to whole-trial driving while being *faster*
(29.9 vs 58.3 ms per trial on the probe). Gate **G1-e passes decisively**. This is the first
per-frame incremental LM cost measured for any brain-to-text system
([LITERATURE §B5](audit/LITERATURE.md) found none published).

*Carry the max, not the p95:* the worst single frame across 324,625 was **59.2 ms**, only 1.3×
under budget. Worst-case latency claims must quote that.

**Q2 is answered and the news is bad.** The constrained baseline is **42.42 % WER**, not the
estimated ~5.5 %. Two causes, pulling in different directions:

0. **Scoring note.** WER is computed with standard ASR normalization — lowercase, punctuation
   stripped, apostrophes kept. Scoring on `.lower().split()` alone reads **51.65 %**, because 98 %
   of references carry punctuation the decoder never emits. Always normalize; the 9.2-point gap is
   pure artifact.
1. **The only LM on disk is the 1-gram** (13.4 MB), which does essentially no sequential language
   modelling. Inspecting hypotheses confirms it: errors are *lexical, not acoustic* — "part ways"
   decodes as "parte weighs", "see" as "sci", "vocal" as "focal". The phonemes are right (10.04 %
   PER); with no sequential LM the decoder picks homophones arbitrarily. This confirms the fear in [Q3](#10-open-questions-that-block-claims): if the
   published 2.66 % came from 1-gram + OPT-6.7B, then **nearly all lexical work lives in the LLM**
   — inside the 620–830 ms batch wall — and R1's recoverable fraction is far smaller than assumed.
2. **`acoustic_scale` and `blank_penalty` were at untuned defaults.** C15 is expected to move this
   substantially; the repo's own two configs disagree 10× on `blank_penalty`.

**Consequence for the plan: C28 (the pruned 4-gram) is promoted from an improvement to the
load-bearing item of Group 4.** Its −2.25 WER estimate was measured 1-gram → 4-gram, and the
1-gram end is now known to be much worse than the catalog assumed. Until a real n-gram exists,
every ΔWER in this file remains **[E]**.

---

### 1.5 C15 / C14 landed 2026-08-06 — decode tuning is worth 7.4 WER points

Sweeps run on **val-dev only** (400-trial coarse pass, then confirmed on all 1,253). `val-test`
has NOT been touched — it is held until Group 1 completes.

| Config | WER (val-dev) |
|---|---|
| `acoustic_scale` 0.30 / `blank_penalty` 9 — the code defaults | **42.42 %** |
| README recipe (0.325 / 90) | 37.81 % |
| **swept optimum: 1.00 / 30** | **35.03 %** (400-trial), **35.51 %** (full dev) |

**−7.4 points for zero training runs.** Grid spread was **20.2 points**, so gate **G1-b passes**:
the decode frontier is emphatically non-degenerate.

Three things the sweep established that the catalog had wrong:

1. **Neither of the repo's own values is optimal.** `build_lm_decoder`'s default is
   `acoustic_scale=1.5`; the README recipe uses 0.3–0.325. The optimum is **1.0**, and it is a
   genuine interior minimum (0.75 → 35.15, **1.0 → 35.03**, 1.5 → 36.44, 2.0 → 37.77), confirmed
   by extending the grid after the first pass returned a boundary winner.
2. **Higher `acoustic_scale` is both more accurate and cheaper.** p95 per-frame falls 2.18 → 0.24 ms
   and wall time drops 4× across the same range that improves WER. A tighter effective search wins
   on both axes; there is no accuracy/latency trade-off to make here.
3. **`blank_penalty` barely matters at the optimum.** It is worth ~9 points at
   `acoustic_scale=0.2` but under 0.5 points at 1.0.

**C14 — blank skipping is accuracy-free but cheaper than it looks, and my first metric was wrong.**
WER is **identical (35.51 %) at every θ ∈ {1.0, 0.999, 0.99, 0.9}**, so the accuracy half of gate
G1-f passes outright. But p50/p95 per-frame showed *no* speedup, because skipping makes skipped
frames nearly free while leaving expensive frames untouched — it moves the distribution's mass, not
its median, and p95 measures exactly the frames that never skip. **Total decode time per trial is
the right instrument**: 35.7 ms (off) → 35.3 (θ=0.999) → **28.0 (θ=0.9)**, a 22 % reduction.

So θ=0.999 buys ~1 % despite 64 % of frames qualifying: the advance steps it removes are not where
the time goes. The catalog's "cuts LM work by 72 %" assumed steps ≈ cost. Moot for feasibility —
the LM is already 108× under budget — but record it as measured, not as predicted.

*Harness validity:* numpy prep is 0.0021 ms/frame against DecodeNumpy's 0.0718 ms — **2.2 %** — so
the Q1 figures are decoder cost, not measurement overhead.

---

### 1.6 C28-A landed 2026-08-06 — a trigram over the *training transcriptions* is worth 17.4 points

The C13/C15 diagnosis was that the shipped 1-gram fails **lexically**, not acoustically. The n-best
measures exactly how much that costs. At `nbest=100` on val-dev, with the C15 optimum
(`acoustic_scale` 1.0 / `blank_penalty` 30):

| | WER (val-dev) |
|---|---|
| 1-best | **36.04 %** |
| **oracle over the 100-best** | **15.17 %** |
| re-ranked by a trigram trained on the 8,072 TRAIN transcriptions | **18.33 %** |

**−17.71 points from re-ranking alone, no retrain, no FST rebuild, no corpus download.** That
recovers 85 % of the 20.87-point oracle gap. The corpus is 50,648 word tokens / 3,745 types — the
smallest sequential LM it is possible to build here — which makes this a **lower bound** on C28.

Four things this pins down:

1. **The 1-gram's 1-best is nearly arbitrary.** A sweep over the LM weight α is flat from 0.25 to
   8.0 (18.74 → 18.68, interior optimum **α=1.0** at 18.33) but collapses to 36.04 at α=0. The
   acoustic + 1-gram score barely orders the list at all; almost any sequential signal reorders it.
2. **The result is not memorization.** 6.94 % of val-dev sentences (87/1,253) appear **verbatim**
   in the training transcriptions. Scored separately: **unseen 36.24 → 18.83 % (−17.41)**, seen
   33.33 → 11.52 % (−21.82). **Report the unseen row.** Removing the overlap costs 0.5 points of
   the headline, so the finding survives.
3. **Re-ranking is a finalization-stage cost, not a per-frame one.** `nbest=100` replaces
   `GetBestPath` with `GetLattice` + `ShortestPath`, costing p50 3.46 / p95 6.42 / **max 15.33 ms**
   per trial — inside the 79 ms budget, but once per utterance, after endpointing. It does not
   change the per-frame Q1 figure.
4. **`nbest=100` costs 0.53 points on the 1-best itself** (36.04 vs 35.51 at `nbest=1`), because
   the word-level lattice n-best is not the same object as `GetBestPath`. Quote the honest delta
   as **35.51 → 18.83 %**, not 36.04 → 18.33 %.

**Consequence: C28 is confirmed as the load-bearing item of Group 4, and its estimate was far too
small.** The catalog's −2.25 pts for 1-gram → 4-gram is off by an order of magnitude at this
operating point. Shallow fusion (C29) can go *below* the 15.17 % oracle because it reorders inside
the beam rather than over a fixed list; re-ranking cannot.

Artifacts: `results/nbest_la0_1gram.json`, `results/rescore_indomain_trigram.json`,
`benchmark/stream_lm.py {nbest,rescore}`, `benchmark/indomain_lm.py`.

---

### 1.7 Group 1 Phase 1A landed 2026-08-06 — C1–C10, C12

**C1 vectorized smoother.** A preallocated circular `[K, C]` window plus a *pre-rotated* kernel
replaces the per-tap loop and the deque linear scan: the data never moves, and one matmul per bin
replaces K python-level GPU ops. Measured on this machine, smoother isolated:

| | K=5 (L=0, deployed) | K=9 (L=4) |
|---|---|---|
| CUDA | 0.0686 → **0.0364 ms/bin** (−47 %) | 0.0979 → **0.0135** (−86 %) |
| CPU | 0.0123 → **0.0054 ms/bin** (−56 %) | 0.0178 → **0.0035** (−80 %) |

Max abs difference **2.4e-07** — float reduction order only, 4,000× under the equivalence
tolerance. **Not bit-identical, contrary to what §4.3 assumed**: a matmul reduction does not sum
in kernel order. The gate that matters (tol < 1e-3, exact collapsed-sequence equality) passes.

**C3 bounded buffers.** `transformed_buffer` is now `deque(maxlen=patch_size)` driven by a counter,
with an invariant check that the due patch really is the newest window (true iff patches overlap,
P=14 > s=4). `logit_frames` retention is now a flag: the equivalence gate keeps them, the RTF path
does not, and `n_frames` comes from a counter either way.

**C4/C5/C6/C12** are one-liners and landed as specified. C5 (`zero_infinity=True`) is a **hard
prerequisite for Groups 2 and 3** — with it False, one masked trial with
`adjusted_lens < phone_seq_lens` crashes the run at the grad clip.

**C8 — G1-d PASSES, but the gate as written was not the right test.** "Reproduce the released
features to <1e-4" is not checkable: `data_val.hdf5` holds only the *val* trials of each block, so
we never observe a complete normalization unit and its sample mean is displaced by subsampling —
the same artifact that made `check_hdf5_causality.py` false-negative. What *is* checkable, and is
what the fix actually rests on, is **invariance to per-channel affine maps**. Measured residual
**3.9e-05** over 7 blocks / 4 sessions. The block constants cancel; R-D2/R-D3 are well-posed.

*One real finding fell out of this.* 0.0175 % of (t, channel) emissions hit the variance floor,
because sparse threshold-crossing channels go locally constant (channel 63 takes exactly two values
over a 20-bin window). No fixed epsilon is scale-invariant, so **the floor is a design decision
R-D2/R-D3 have to make explicitly, before the run, not a detail.**

**C9 — G1-c PASSES decisively.** Per-phase greedy PER on val-dev: **8.849 / 8.868 / 8.813 /
8.904 %**, spread **0.091 pts** against a 0.5-pt gate. Phase 3 — the one `random_cut: 3` never
trains on — is worst by 0.055 pts, i.e. barely above the 0.041-pt reproducibility floor. **C16 does
not need to wait for Group 2's `random_cut: 4`.** (These are val-dev only; 8.85 % here vs the
checkpoint's 10.04 % over all 1,426 trials is the held-out sessions being harder, not a
discrepancy.)

**C10 — temperature is worth −0.05 points. Record it as a negative result.**

| T | 1.0 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| WER (val-dev) | 35.51 | 35.47 | **35.46** | 36.00 | 36.61 |

The over-confidence is real (mean entropy 0.9 % of maximum) but **it is not what is costing WER**.
The beam does not need more acoustic alternatives — C28-A showed the 1-gram cannot *order* the
alternatives it already has. This weakens the WER half of C20's case before Group 3 spends a run
on it; C20's PER case is untouched.

---

### 1.8 C16 landed 2026-08-06 — phase merging is a frontier point, not an improvement

G1-c passed, so both modes were built and measured **separately**, as §4.2 requires. On val-dev,
against the same 1,253 references (verified identical multiset across both cache paths):

| Condition | WER | vs baseline | What it buys |
|---|---|---|---|
| baseline, phase 0 only, 12.5 Hz | **35.51 %** | — | — |
| **(a) interleaved, 50 Hz**, `acoustic_scale` 0.25 | **35.97 %** | **+0.46** | **L_buf 60 → 15 ms** |
| (b) phase-averaged, 12.5 Hz | 36.20 % | +0.69 | nothing |

**Both cost accuracy. Neither is free, and the plan's −0.2…−0.6 estimate had the sign wrong.**

1. **(a) is a real accuracy-vs-latency trade and should be reported as a frontier endpoint.** It
   cuts worst-case input buffering from 60 ms to 15 ms for +0.46 WER points, 4× the acoustic
   forward passes (4 × 0.574 ms p50 per 20 ms window = 11 % duty cycle) and 4× the LM frame rate.
   Per-frame LM cost stays trivial — p50 0.097 / p95 0.303 ms — but **worst case is 44.94 ms
   against a 20 ms cadence**, so the spike now exceeds the frame interval even though the mean is
   200× under it. Any real-time claim for (a) must quote that.
2. **The optimal `acoustic_scale` moves to exactly 1/4** (1.0 → 0.25 at 4× the frame rate), which
   is the frame-rate compensation you would predict analytically — good evidence the merge is
   wired correctly rather than accidentally reweighting the LM. Off-optimum is expensive: 0.1
   costs 836 s of wall against 46 s at 0.25, and is worse (36.72 %).
3. **(b) is declined.** Averaging log-probabilities across phases smears the four members over
   3 bins (60 ms) because frame *f* of phase *p* sits at bin `p+4f+13` — they never share a right
   edge. The ensemble gain does not survive the misalignment.

Artifacts: `benchmark/phase_ensemble.py`, `results/c16_interleaved.json`,
`results/c16_phaseavg.json`, `results/c16_interleaved_sweep.json`, `results/c9_phase_per.json`.

---

### 1.9b C28 partial 2026-08-06 — the toolchain works, and in-beam fusion is worth 19 points

**The build chain is live.** None of it was built before: SRILM (`make MAKE_PIC=yes World`), and the
six Kaldi FST tools (`arpa2fst`, `fsttablecompose`, `fstdeterminizestar`, `fstminimizeencoded`,
`fstisstochastic`, `fstaddselfloops`) via `cmake --build` in the tree `setup.py` had already
configured. `setup_lm.sh` builds only the python extension, so `path.sh` points at a `build/kaldi`
that does not exist — `language_model/build_tlg.sh` wires the real locations and is the entry point.

**First LM built end-to-end: a 4-gram over the 8,072 training transcriptions.** This was meant as a
pipeline smoke test. It is also the strongest result in the project so far.

| | WER (val-dev) | vs 1-gram baseline |
|---|---|---|
| shipped 1-gram, tuned (C15) | 35.51 % | — |
| C28-A trigram **re-ranking** the 100-best | 18.33 % | −17.2 |
| **in-domain 4-gram, in-beam** | **15.51 %** | **−20.0** |
| — unseen sentences only (1,166 trials) | **16.37 %** | **−19.1** |
| — seen sentences (87 trials, memorized) | 4.01 % | |

**In-beam fusion beats re-ranking by 2.5 points**, as predicted: re-ranking is capped by the n-best
oracle (15.17 %), fusion is not. It is also **cheaper** — p50 0.056 / p95 0.196 ms per frame against
the 1-gram's 0.072 / 0.712, because a better-informed LM keeps the beam tighter.

**Two limits on this number, both structural:**
1. **6.78 % of reference tokens are outside the training vocabulary** and this LM cannot emit them:
   `build_lm.sh` passes `-limit-vocab`, and `make_tlg.sh` greps `<unk>` out of the ARPA, so a word
   the corpus never saw has no path in G.fst. That is a hard floor. Reaching 16.37 % against a
   6.78 % floor means most of the *reachable* error is already gone.
2. It is **not deployable** — it is an in-domain LM for one participant's prompt set. C28 proper
   still needs a general corpus, which is the open decision below.

Artifacts: `language_model/build_tlg.sh`, `results/lm_indomain_4gram/` (149 MB, TLG 48 MB),
`results/c28_indomain4g.json`, `results/c28_indomain4g.hyps.tsv`.

**The general corpus, and the constraint the plan got wrong.** 2.0 GB of OpenWebText
(344.8 M tokens, 16.9 M sentences, 94.3 % of sentences kept) via `language_model/fetch_corpus.py`.
109,368 of the lexicon's 133 k words are attested, against 3,745 in-domain — that is what removes
the 6.78 % OOV floor. Counting at `-gt3min 2 -gt4min 2` gives **61.7 M n-grams** (1.8 GB ARPA) at
**12.1 GB peak / 6 min**.

**`ngram-count` was never the constraint; TLG composition is, at ~8× the artifact size.** A 10 %
pilot composed a 1.6 GB TLG at **13.0 GB peak RSS**. So a 19 GB TLG would need ~150 GB to build:
**[BUDGET]'s "pruned 4-gram ≤ 19 GB" is an *inference* budget and the plan never separated it from
the build budget.** Entropy pruning is therefore mandatory, and the pruning threshold — not RAM —
sets how large a deliverable this machine can ship.

| `ngram -prune` | n-grams | ARPA | time / peak |
|---|---|---|---|
| unpruned | 61.7 M | 1.8 GB | 6 min / 12.1 GB |
| 1e-9 | 44.3 M | 1.3 GB | 54 s / 1.6 GB |
| 3e-9 | 32.0 M | 919 MB | 52 s / 1.6 GB |
| 1e-8 | 14.9 M | 421 MB | 43 s / 1.6 GB |

Two build traps worth recording, both of which produce a *plausible-looking* wrong answer:

1. **The recipe is UPPERCASE end-to-end.** `format_lm_data.py` emits `BRING IT CLOSER` and the
   lexicon is uppercase, so lowercase text fed to `ngram-count` maps every token to `<unk>`,
   exits 0 in 1.75 s, and yields 3 bigrams. `run.sh` uppercases en route, so only direct
   `ngram-count` calls are exposed.
2. **`-gt4min 1` keeps every singleton 4-gram**, nearly doubling peak RSS for mass that does not
   survive discounting. The cutoffs are now env-overridable in `local/build_lm.sh`
   (`GT3MIN`/`GT4MIN`), defaults unchanged.

#### C28 result — the general 4-gram takes val-dev to 8.49 %

Built at `-prune 1e-8`: 14.9 M n-grams, **TLG 3.94 GB**, composed at **15.0 GB peak / 2.5 min**
(a 3.8× build-to-artifact ratio, not the 8× the pilot suggested — there is headroom for a larger
LM). Decode parameters were **re-swept**, which mattered: the 1-gram optimum of
`acoustic_scale` 1.0 is badly wrong for a 4-gram.

| Condition | all (1,253) | unseen (1,166) | seen (87) |
|---|---|---|---|
| shipped 1-gram, tuned | 35.51 % | 36.24 % | 33.33 % |
| in-domain 4-gram | 15.51 % | 16.37 % | 4.01 % |
| **general 4-gram** (`ac_scale` 0.4 / `blank_pen` 90) | **8.49 %** | **8.72 %** | 5.41 % |

**−27.0 points against the 1-gram baseline. [CANDIDATES]'s estimate for C28 was −2.25 [E]; the
measured value is 12× that.** Every ΔWER in this plan that was quoted against the 1-gram baseline
is now suspect for the same reason — the baseline was pathological, not merely untuned.

Three things this settles:

1. **It is not memorization.** The seen/unseen gap collapses to 8.72 vs 5.41, where the in-domain
   LM showed 16.37 vs 4.01. General text generalizes; the in-domain LM was partly reciting.
2. **The optimum moved to `acoustic_scale` 0.4 / `blank_penalty` 90**, from 1.0 / 30. A better LM
   deserves more weight, and the plateau is flat across 0.35–0.4 (both 6.87 % on the 400-trial
   probe), so this is robust rather than a knife-edge. Grid spread 0.84–4.22 pts.
3. **Latency is still not the problem.** p50 0.067 / p95 0.641 / max 30.6 ms per frame against a
   79 ms budget — **123× margin at p95** — with a 3.94 GB FST. Finalization p50 0.76 / p95 6.06 ms.

For context, the published full-stack figure is 2.66 % **[U]** *with* n-best rescoring and OPT-6.7B
inside a 620–830 ms batch wall. **8.49 % is incremental n-gram only: no rescore, no LLM, no
lookahead.** That is the number the latency argument has been missing since C13.

#### Interpolation is REJECTED — it buys memorization, not generalization

The mixed LM (λ=0.3, chosen by held-out perplexity) was built and decoded. It loses:

| LM | all | unseen (1,166) | seen (87) |
|---|---|---|---|
| **general 4-gram** | **8.49 %** | **8.72 %** | 5.41 % |
| mixed, λ=0.3 | 8.64 % | **9.18 %** | **1.40 %** |

Its entire advantage is on the 87 verbatim-in-train sentences (1.40 % vs 5.41 %). On the 1,166
unseen sentences it is **0.46 pts worse**. Adding a model estimated from 50 k words dilutes
well-estimated general probabilities, and the beam pays for it on any word outside the prompt set.

**Perplexity picked wrong even though it was measured on the unseen subset** (mixed 91.5 vs general
103.6). That is the third ppl/WER divergence here — the first being the in-domain LM's 4× worse
perplexity alongside a 19-point WER gain. **Do not select an LM for this system on perplexity.**
Ship the general 4-gram.

#### 1e-8 is this machine's ceiling

The mixed build peaked at **25.1 GB of 27 GiB** (16.5 min), and `fstminimizeencoded` in the LG
stage — not the final T∘LG compose — is the peak. At ~1.7 GB of peak per million n-grams, the next
threshold down (`3e-9`, 32 M n-grams) would need ~50 GB. **`-prune 1e-8` is the largest LM this box
can compose**, so C28 is at its achievable optimum here, not at a tuning optimum.

Artifacts: `results/lm_general_4gram/`, `results/lm_gen_p1e-8/data/lang_test/` (TLG 3.94 GB —
**the deliverable**), `results/c28_general4g.{json,hyps.tsv}`, `results/c28_gen_sweep{,2,3}.json`,
`results/lm_mixed_tlg/` + `results/c28_mixed4g.*` (rejected),
`language_model/{fetch_corpus.py,make_tlg_from_arpa.sh,interpolate_lm.sh}`.

---

### 1.12 C29 rescoring landed 2026-08-07 — 8.49 % → 6.89 %, and a real latency frontier

Neural rescoring of the C28 n-best (`benchmark/neural_rescore.py`). Best configuration:
**gpt2-large, γ=0.25, α=1.0 → 6.89 % val-dev** (unseen 7.09 %), recovering **29 % of the 5.62-pt
oracle headroom**.

**A scoring bug was worth more than 6× the parameters.** The first implementation *added* the
neural score on top of a full-weight 4-gram, double-counting the language model. The decoder's own
`Rescore()` subtracts the original LM first (`brain_speech_decoder.cc:65-72`). Introducing a weight
γ on the 4-gram term fixes it:

| | γ=1 (double-counted) | γ=0.25 (corrected) |
|---|---|---|
| gpt2 (124 M) | 7.83 % — 12 % of headroom | **7.38 % — 20 %** |
| gpt2-large (774 M) | 7.58 % — 16 % | **6.89 % — 29 %** |

γ=0 (full replacement) is *worse* than γ=0.25 (7.95 % vs 7.38 % on gpt2): the 4-gram is
well-estimated on this vocabulary and should be down-weighted, not discarded. **Any future fusion
or rescoring work must sweep γ** — with it fixed at 1, model scaling looks flat and the whole
approach looks dead.

**Model scaling and domain adaptation (γ swept, full list). Unseen is the reportable column:**

| rescorer | params | WER (all) | **WER (unseen)** | headroom | neural p95 |
|---|---|---|---|---|---|
| none | — | 8.49 % | 8.72 % | — | — |
| gpt2 | 124 M | 7.38 % | 7.63 % | 20 % | 15.9 ms |
| gpt2 + in-domain FT | 124 M | 7.26 % | 7.58 % | 22 % | 17.2 ms |
| Qwen2.5-0.5B | 494 M | 7.25 % | 7.54 % | 22 % | 52.3 ms |
| **gpt2-large** | **774 M** | **6.89 %** | **7.09 %** | **29 %** | 75.1 ms |
| gpt2-large + in-domain FT | 774 M | 7.12 % | 7.42 % | 25 % | 70.3 ms |
| Qwen2.5-1.5B | 1544 M | 6.95 % | 7.19 % | 28 % | 190.9 ms |

**Rescoring plateaus at ~29 % of headroom, and neither lever breaks it.**

1. **Scale stops paying past ~800 M.** Qwen2.5-1.5B is *worse* than gpt2-large at 2× the parameters
   and 2.5× the latency — and its p95 190.9 ms exceeds the whole 140 ms budget. Parameters also beat
   architecture modernity in this range (gpt2-large > Qwen2.5-0.5B on both axes).
2. **Domain adaptation is rejected — it buys memorization, and at scale it actively hurts.**
   Fine-tuning on the 8,072 TRAIN transcriptions cut held-out perplexity **395.14 → 27.44** on
   gpt2-large, a **14× improvement**, and made unseen WER **worse**: 7.09 % → 7.42 %. On gpt2-small
   it moved unseen by 0.05 pts while moving *seen* by 1.04 (4.01 → 2.97). Narrowing to the prompt
   distribution destroys the general knowledge that does the discrimination.

That perplexity result is the **fourth ppl/WER divergence** in this project, and the sharpest: a 14×
perplexity win produced a WER regression. Alongside §1.9b (interpolation) and §1.11 (in-domain
trigram, 0.00 pts), the pattern is now four-for-four — **in-domain adaptation of the LM stage does
not generalize on this task, at any model class.** The 6.94 % verbatim train/val overlap is what
makes it look otherwise on any metric that does not split seen from unseen.

**The accuracy/latency frontier, which is the deliverable this project actually wants:**

| config | WER | total finalization p95 |
|---|---|---|
| no rescoring (`nbest=1`) | 8.49 % | 6.1 ms |
| gpt2-large, top-5 | 7.45 % | ~35 ms |
| gpt2-large, top-10 | 7.35 % | ~37 ms |
| gpt2-large, all ~30 | **6.89 %** | ~96 ms |

*(total = n-best extraction p95 21.3 ms + neural forward p95)*

The oracle gain is spread **through** the list, not concentrated at the top: truncating 30 → 10
costs 9 points of headroom. Model size and list depth are substitutable — gpt2-large@top-10 ≈
gpt2-small@full on both axes. All of these sit inside the 140 ms end-to-end budget, against the
shipped stack's 620–830 ms.

**Where this leaves the project:** 8.49 % → **6.89 %** incremental (unseen 8.72 % → 7.09 %), no
batch wall, at p95 ~96 ms of a 140 ms budget. **Recommended operating point: gpt2-large, γ=0.25,
α=1.0, full n-best.** Remaining gap is 4.01 pts to our own oracle, 4.23 to the published 2.66 %
**[U]**.

**Both obvious ways to close it are now measured and refuted** — more parameters (§ above) and
in-domain adaptation (§ above). What has *not* been tried, in rough order of expected value:

- **Shallow fusion in the beam.** The only remaining mechanism that can go *below* the 2.88 %
  oracle, because it changes which candidates exist rather than reordering a fixed list. Costs
  ~+10 ms/word and decoder integration.
- **Understand the plateau before spending on it.** 71 % of the oracle headroom survives every
  rescorer tried. An error analysis of the surviving cases — are the right and wrong candidates
  genuinely equiprobable under English, or is the acoustic term overwhelming the LM? — is nearly
  free and would say whether *any* rescorer can win, or whether the acoustic scores are the
  binding constraint.
- Larger n-best (`nbest > 100`) to raise the oracle itself, at finalization cost.

Artifacts: `benchmark/neural_rescore.py`, `benchmark/finetune_rescorer.py`,
`results/c29_oracle_4gram.json`, `results/c29_rescore_*.json`, `results/c29_topk{5,10}.json`,
`results/rescorer_ft_*`.

---

### 1.11 The remaining gap is re-ranking, not search — measured 2026-08-07

With the C28 4-gram in the beam, `nbest=100` on val-dev:

| | WER |
|---|---|
| 1-best | **8.49 %** |
| **oracle over the 100-best** | **2.88 %** |
| published full-stack reference (1-gram + rescore + OPT-6.7B) | 2.66 % **[U]** |

**The correct hypothesis is already in the list, at essentially the published accuracy.** The
5.62-point gap between our 1-best and our own oracle is a *selection* problem: the search is
finding the right answer and ranking it wrong. This reframes the published system — OPT-6.7B is
largely doing n-best selection, and selection does not inherently require the 620–830 ms batch
wall.

**A cheap n-gram cannot claim any of it.** Re-ranking with the C28-A in-domain trigram recovers
**0.00 pts**; every positive weight makes it worse (α=2 → 13.48 %, +4.99). A well-estimated 4-gram
already subsumes what a 50 k-word trigram knows. This is the same lesson as the interpolation
rejection in §1.9b, and it means the 5.62 points needs a model of a *different kind*, not more
n-gram.

**So C29 is the only remaining lever of consequence**, and it is worth up to 5.62 points — more
than every acoustic-side item in this plan combined, all of which measured neutral or negative.

Two constraints on how C29 is built, both measured here:

1. **n-best rescoring pays a real finalization cost**: `nbest=100` finalization is p50 2.19 /
   **p95 21.30 / max 285.64 ms** per trial, against p50 0.76 / p95 6.06 at `nbest=1`. The max
   alone is 3.6× the whole 79 ms LM budget. **Shallow fusion inside the beam avoids this
   entirely** and is the better default; n-best rescoring must justify the tail.
2. **List sizes shrank** — mean 30.6 candidates now, against 78.5 under the 1-gram. A better LM
   keeps the beam tighter, so the oracle is computed over fewer, better candidates.

---

### 1.10 G2 landed 2026-08-07 — it passes on PER and LOSES on WER. Reject it.

340 min, 120k batches, `trained_models/g2_masking`. **Gate G2 is a strong pass on its own terms**
(≤ 9.50 %) and the improvement is real — 14× the 0.041-pt reproducibility floor:

| | val PER (best) | val PER (last-10) | val CTC loss | **val-dev WER** | WER (unseen) |
|---|---|---|---|---|---|
| `causal_la0` | 10.040 % | 10.087 % | 21.74 | **8.49 %** | 8.72 % |
| **G2** (C17+C18+C19) | **9.430 %** | **9.499 %** | 18.96 | **9.10 %** | 9.49 % |
| Δ | **−0.61** | −0.588 | −2.78 | **+0.61** | +0.77 |

**The PER gain inverts into an almost exactly equal WER loss.** This is not a tuning artifact: the
decode optimum was re-swept per model (G2's is a genuine interior minimum at `acoustic_scale` 0.5
vs the baseline's 0.4), and **G2 is worse at matched settings too** — at 0.4, 6.87 % vs 7.63 %;
at 0.5, 7.15 % vs 7.43 % on the 400-trial probe.

**Mechanism, and it is the same story as C10.** The gate asked whether masking de-sharpens the
posterior. It does the opposite: mean entropy goes **0.89 % → 0.86 % of maximum**, and frames with
p(blank) > 0.999 rise 71.83 % → 73.39 %. A sharper posterior gives the beam *fewer* acoustic
alternatives, so the LM has less to work with — exactly the mechanism C10 identified when
temperature bought only −0.05. Masking buys phoneme accuracy by making the model more decisive,
and decisiveness is what the LM stage cannot repair.

**Consequences:**

- **Do not carry G2 into the deployable stack.** Keep the run as an ablation; the checkpoint stays
  at `trained_models/g2_masking`. C19 (`random_cut: 4`) is separable and still wanted for C16.
- **PER is the wrong gate.** Groups 2–4 gate on val PER at every step except G3. Every one of those
  thresholds should be restated in WER before more GPU time is spent. §8.3's ledger already asks
  for held-out WER per run — that column is now the *only* one that decides anything.
- **Group 3's expected value drops sharply.** C20 (label smoothing) was already weakened by C10 and
  is weakened again here: its WER case rests on de-sharpening a posterior that masking just made
  sharper without helping. C21/C24 are untouched by this argument.
- The bundle is **bisectable** if you want attribution (§7.6): C17 and C18 are both sharpeners,
  C19 is not. One leave-one-out run would say whether C19 alone is neutral-to-positive on WER.

*(C6 side-note: enabling `save_all_val_steps` exposed a latent crash — `save_model_checkpoint` was
being called without its `loss` argument at both the `save_all_val_steps` and `save_final_model`
sites, dead code paths until now. Fixed; both are exercised by the smoke config.)*

---

### 1.9 Gate G1 — verdict

| Gate | Condition | Result |
|---|---|---|
| **G1-a** equivalence | tol < 1e-3, exact collapsed-sequence match after C1–C3 | **PASS** — 120 trials × 2 checkpoints (L=0 and L=4), max diff 2.4e-07 |
| **G1-b** streaming cost | per-window compute −50 % | **PARTIAL** — smoother −47 % (CUDA, K=5, deployed) / −56 % (CPU) / −86 % (K=9). The absolute 1.06 → 0.37 ms target is **not evaluable**: 1.06 ms was measured on the Mac CPU, this machine is a 5070 Ti. Report the same-machine A/B, not the cross-machine ratio |
| **G1-c** phase equivalence | per-phase PER spread ≤ 0.5 pts | **PASS** — 0.091 pts. C16 did not need `random_cut: 4` |
| **G1-d** identity check | < 1e-4 | **PASS** — 3.9e-05, on the affine-invariance form of the test (see §1.7) |
| **G1-e** incremental fits | p95 ≤ 79 ms | **PASS** — p95 0.712 ms, 111× margin; max 59.2 ms is the number worst-case claims must use |
| **G1-f** blank skipping free | ΔWER ≤ 0.10 pts | **PASS** on accuracy (0.00 pts at every θ). The "≥ 60 % of advance steps" half was the wrong instrument — see §1.5 |
| **G1-g** frontier non-degenerate | ≥ 1.0 pts of spread across 40–140 ms p95 | **N/A as written** — 20.2 pts of spread, but the whole sweep lives at **0.24–2.18 ms** p95, two orders of magnitude under the stated band. There is no accuracy/latency trade to trace on the *decode* axis; higher `acoustic_scale` is better and cheaper. The trade reappears on the **frame-rate** axis (C16a) and on the **LM-size** axis (C28) |

**Group 1 is complete. 0 training runs spent.** The two findings that reshape the rest of the plan:
the LM's *lexical* weakness is worth ~17 points (§1.6) while everything on the acoustic/decode side
is worth fractions of a point (§1.7 C10, §1.8 C16), and the accuracy/latency frontier does not live
where the plan assumed it did (§1.9 G1-g).

---

## 2. The consolidated change list

31 changes, ordered so that adjacent items share an implementation surface. Latency column is the
cost at *inference*; "0" means the change vanishes at deployment.

| ID | Change | Group | Surface | Expected Δ | Latency | Evidence |
|---|---|---|---|---|---|---|
| C1 | Vectorize streaming smoother + day layer | 1A | `benchmark/streaming_infer.py:194-208` | 0 (bit-identical) | **−0.7 ms/window** | [AUDIT F5] |
| C2 | Cache the Gaussian kernel across calls | 1A | `data_augmentations.py:15-31` | 0 (bit-identical) | −ε | [AUDIT F15] |
| C3 | Bound `transformed_buffer` / `logit_frames` | 1A | `streaming_infer.py:216,235` | 0 | 0, fixes O(T) memory | [AUDIT F14] |
| C4 | Fix unbound `frame_diff` | 1A | `benchmark/test_equivalence.py:95` | 0 | 0 | [AUDIT A7] |
| C5 | `zero_infinity=True` | 1A | `rnn_trainer.py:242` | 0 | 0 | [AUDIT F11] — **prereq for G2/G3** |
| C6 | `save_all_val_steps: true` | 1A | `rnn_args.yaml:26` | 0 | 0 | enables C25 free |
| C7 | Freeze val-dev / val-test session split | 1A | new `model_training/splits.py` | 0 | 0 | [AUDIT A8] |
| C8 | Causal-normalization identity check | 1A | new `model_training/causal_normalize.py` | 0 | 0 | de-risks R-D2/D3 |
| C9 | Per-phase greedy PER (phase-equivalence gate) | 1A | new `benchmark/phase_ensemble.py` | 0 | 0 | [AUDIT F10] |
| C10 | Temperature scaling on cached logits | 1A | `benchmark/` sweep script | **MEASURED −0.05 [M]** (est. −0.1…−0.4) | 0 | §1.7 |
| C11 | Build `lm_decoder` | 1B | `./setup_lm.sh` | — | — | **GATING** |
| C12 | Un-hardcode decoder params | 1B | `language-model-standalone.py:486-496` | 0 | 0 | [AUDIT A6] |
| C13 | Incremental LM driver → constrained baseline | 1B | new `benchmark/stream_lm.py` | **the baseline** | measures Q1 | [AUDIT F4] |
| C14 | Blank skipping θ ∈ {1.0, .999, .99, .9} | 1B | CLI flag | WER ±0.1 **[E]** | **−72 % WFST steps** | [BUDGET §5] |
| C15 | `acoustic_scale` × `blank_penalty` × beam sweep | 1B | CLI / sweep script | WER **−0.65** **[E]** | 0 | [CANDIDATES] |
| C16 | Phase merge: interleaved 20 ms / phase-averaged | 1B | `benchmark/phase_ensemble.py` | **MEASURED +0.46 / +0.69 [M]** — sign inverted | **L_buf 60→15 ms** | §1.8 |
| C17 | Time masking, n=20, `max_frac=0.075` | **2** | `data_augmentations.py` + `rnn_trainer.py:436-485` | bundle **PER −0.61 / WER +0.61 [M]** | 0 | §1.10 |
| C18 | Channel / electrode masking, rate 0.10 | **2** | same | in the same bundle — **REJECTED on WER** | 0 | §1.10 |
| C19 | `random_cut: 3 → 4` (full phase coverage) | **2** | `rnn_args.yaml:67` | PER ~0; **not separately measured** — only non-sharpener in G2 | 0, enables C16 | [AUDIT F10] |
| C20 | Label smoothing / entropy reg on CTC | **3** | `rnn_trainer.py:242,539` | **DOWNGRADED** — C10 (−0.05) and G2 (posterior got *sharper*) both undercut the mechanism | 0 | §1.7, §1.10 |
| C21 | Intermediate / self-conditioned CTC, aux weight 0.3 | **3** | `rnn_model.py`, `rnn_trainer.py` | PER **−0.60** **[E]** | 0 (heads dropped) | [LITERATURE B3.3] |
| C22 | Day-layer reg: `lr_max_day` 0.005→0.0025, `weight_decay_day` 0→0.0001 | **3** | `rnn_args.yaml:38,47` | PER −0.05…−0.40 **[E]** | 0 | [RESEARCH_PROPOSALS §2] |
| C23 | `weight_decay: 0.001 → 0.005` | **3** | `rnn_args.yaml:46` | PER ~−0.1 **[E]** | 0 | 7th-place B2T'25 |
| C24 | Checkpoint averaging over last k=10 validations | **3** | new `model_training/average_checkpoints.py` | PER **−0.12** **[E]** | 0 | [AUDIT F8] |
| C25 | Bandwidth-matched causal control (σ_eff ≈ 1.852 at L=0) | **4** | `data_augmentations.py` + config | **control — either answer publishable** | −ε | [AUDIT F2] |
| C26 | Causal rolling normalization (Welford, 10 s half-life) | **4** | new `causal_normalize.py`, `dataset.py:130` | PER **+0.80** **[E]** — a *cost* | +ε | [AUDIT F1] |
| C27 | Seed replication on the winning config | **4** | `rnn_args.yaml:48` | error bar | 0 | [AUDIT F9] |
| C28 | Build pruned 4-gram ≤ 19 GB → TLG | **4→DONE** | `language_model/build_tlg.sh` | **MEASURED −27.0 [M]** (est. −2.25) | p95 0.64 ms | §1.9b |
| C29 | Causal neural LM shallow fusion | **4** | `stream_lm.py` | WER **−1.00** **[E]** | +2–10 ms/word **[E]** | [RECOMMENDATIONS R1] |
| C30 | Wire up CTC endpointer + threshold sweep | **4** | `ctc_endpoint.{h,cc}` | WER +0.05 **[E]** | **finalization −1,680 ms** | [CANDIDATES] |
| C31 | Adaptive-compute gate on n-gram confidence | **4** | `stream_lm.py` | WER **−1.40** **[E]** | expected +5–50 ms; worst case unbounded | [CANDIDATES] |

### 2.1 Cumulative expectation

Gains inside a family do not add — several changes target the same error mass, and the source
documents are explicit about that. Discounted:

**SUPERSEDED 2026-08-07 by measurement. The arithmetic below is what the plan predicted; the
right-hand column is what happened. Keep it only as a record of how badly the priors were placed.**

```
Acoustic (val PER)                          PREDICTED        MEASURED
  causal_la0                                10.04 %  [M]     10.04 %  [M]
  + Group 2 (masking block)      −0.5…−1.0   9.0–9.5 % [E]   9.43 %   [M]  ← in band
  + Group 3 (objective block)    −0.3…−0.7   8.6–9.1 % [E]   not run
  + Group 4 causal normalization +0.3…+1.5   8.9–10.6 % [E]  not run

Deployable (val-dev WER, incremental only)  PREDICTED        MEASURED
  constrained baseline                       ~5.5 % [E]      42.42 %  [M]  ← 7.7x off
  + Group 1 decode tuning (C14/C15)          ~4.9 % [E]      35.51 %  [M]
  + C28 general 4-gram                       (−2.25 [E])     8.49 %   [M]  ← −27.0 pts
  + Group 2 on top of C28                    better [E]      9.10 %   [M]  ← WORSE
```

Three corrections the measurements force:

1. **The acoustic and LM axes are not additive, and they can have opposite signs.** Group 2 is
   −0.61 on PER and **+0.61 on WER** (§1.10). Any row above that is quoted in PER cannot be
   carried into the WER column at all.
2. **Nearly all the recoverable error was in the LM stage**, and it was recovered by one change:
   35.51 % → 8.49 %. Group 1's decode tuning (−7.4) and C28 (−27.0) together account for
   essentially the entire distance travelled. Every acoustic-side item measured so far
   (C10 −0.05, C16 +0.46/+0.69, G2 +0.61) is neutral or negative on WER.
3. **The remaining gap to the 2.66 % [U] full-stack reference is ~5.8 points**, and that reference
   buys it with n-best rescoring plus OPT-6.7B inside a 620–830 ms batch wall. The open question is
   no longer "can incremental decoding be made accurate" — it is how much of those 5.8 points is
   reachable without re-entering the batch wall. That is C29/C31 territory, not Group 2/3 territory.

Unconstrained full-stack reference is 2.66 % **[U]** and violates the budget by 4.4–5.9×.

---

## 3. Group 0 — environment. **DONE 2026-08-06. This machine runs the pipeline.**

### 3.1 Verified state

| Component | State | Note |
|---|---|---|
| GPU | **RTX 5070 Ti, 16 GB, sm_120** | `get_arch_list()` includes `sm_120`; cuDNN GRU and bf16 autocast both execute |
| torch | **2.13.0+cu130** (in `.venv`) | `setup.sh`'s `cu126` pin does **not** carry Blackwell kernels — never install from cu126 here |
| Python env | **`.venv/` (uv, py 3.10.20) — use this.** Matches all 17 `setup.sh` pins exactly; torch **2.13.0+cu130**, sm_120 verified | miniforge3 is kept **only** to run `setup_lm.sh` |
| CPU | Core Ultra 7 270K Plus, **22 cores** | plenty for the LM build and dataloading |
| System RAM | **22 GiB visible to WSL2** (host 32 GB, capped by `.wslconfig`) | see §3.3 — this is a real constraint |
| `data/hdf5_data_final/` | **45 sessions, 13 GB, extracted** | 1,426 val trials; longest train trial 2,475 bins (49.5 s), longest val 2,382 |
| Checkpoint | **`trained_models/pretrained_baseline/`** — upstream released model, 532 MB | was inside the zip; see §3.2 |
| 1-gram TLG | present, 13.4 MB + 1.8 MB `words.txt` | |
| `lm_decoder` | building. gcc 13.3.0 ✓, **cmake pinned to 3.31.10** — cmake **4.x fails** (see §3.6) | `uv tool install "cmake<4"` |
| Disk | 910 GB free | |
| Intel AI Boost NPU | **not exposed to WSL2** (`/dev/accel*` absent) and **not useful here** | see §3.4 |

### 3.2 The pretrained checkpoint changes the critical path

`brain-to-text-25.zip` contained `t15_pretrained_rnn_baseline/checkpoint/best_checkpoint`, now at
`model_training/trained_models/pretrained_baseline/`. Its `args.yaml` has **no `smooth_lookahead`
key**, so it is the symmetric, **non-causal** upstream release — not `causal_la0`.

**But it is enough to run all of Group 1.** Group 1 measures the *cost structure* of the LM stage
(Q1) and establishes the constrained baseline (Q2); neither depends on which acoustic checkpoint
supplies the logits. So:

- **Group 1 needs no training run and is unblocked right now.**
- `causal_la0` must still be retrained before Groups 2–4, because every PER delta is measured
  against its 10.04 %. That is one 5.1 h run using the config already in `rnn_args.yaml`
  (`output_dir` is already `trained_models/causal_la0`, `smooth_lookahead: 0`) — change **only**
  `gpu_number: '1' → '0'`.
- Run the `causal_la0` retrain **concurrently with Group 1's CPU/LM work.** They do not contend.

### 3.3 Measured VRAM and throughput — Q11 resolved

Measured directly at the **worst-case** sequence length in the dataset, not a typical one:

| Loop | Batch | T (bins) | Peak allocated |
|---|---|---|---|
| Train fwd+bwd+step | **64** | 2,475 (longest train trial) | **7.05 GB** |
| Train fwd+bwd+step | 32 | 2,475 | 3.74 GB |
| Val forward (`no_grad`) | **64** | 2,382 (longest val trial) | **2.72 GB** |

**The stock config fits in 16 GB with ~9 GB of headroom. Do not reduce `batch_size`** — that
would change the training dynamics for no reason.

**REQUIRED for any training run on this machine:**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Without it the run is **8.5× slower**, silently — no error, no warning. Measured on 300 real
training batches, GPU verified idle beforehand:

| | default allocator | `expandable_segments:True` |
|---|---|---|
| Peak **allocated** | 7.41 GB | 7.41 GB |
| Peak **reserved** | **22.87 GB** — exceeds the 16.3 GB card | **7.58 GB** |
| Wall s/batch | 0.40–2.82, erratic | **0.152–0.200, steady** |
| Steady state | ~1.5 s/batch | **0.169 s/batch** |
| **120k-batch run** | **~50 h** | **~5.6 h + ~0.25 h validations ≈ 5.9 h** |

**Mechanism.** Every batch pads to the longest of its 64 trials, so T varies continuously
(train trials: mean 875, p50 836, p95 1426, max 2475 bins). The default caching allocator hoards a
differently-sized freed block per shape; reservation climbs past physical VRAM; WSL2's WDDM manager
then silently pages GPU memory over PCIe rather than OOM-ing. The tell is that the trainer's logged
`time:` stays flat at ~0.13 s while wall time swings 7× — the GPU is not doing more work, it is
waiting on the driver.

**Do not trust `rnn_trainer.py`'s logged `time:` field as throughput.** `train_step_duration`
(`:562`) stops *before* `loss.item()` (`:563`), which is the first call that synchronizes with the
GPU — so it measures kernel *launch* time, not execution. Always time with `torch.cuda.synchronize()`
or from log timestamps.

Loss fell 754.6 → 642.1 → 532.6 → 170.9 and val PER 1.2295 → 0.8948, identical across torch
2.11+cu128 and 2.13+cu130 (CTC loss matched to 4 dp), so the loop learns and the envs agree.

**6 training runs ≈ 35 h.**

> **Operational warning, learned the hard way.** A training process that is orphaned by a killed
> shell **keeps running and keeps its VRAM**. Three stacked orphans took the GPU to 15.8 GB and
> produced a CUDA failure inside the cuDNN GRU that reads exactly like an OOM. Before every run:
> `pgrep -af train_model.py` and `nvidia-smi`. Launch long runs with `setsid nohup … &`, never
> from a foreground shell that a timeout can kill out from under them.

### 3.4 The NPU

`/dev/accel*` is absent and `lspci` shows no NPU — WSL2 passes through the GPU (`/dev/dxg`) but
not Intel's AI Boost. **Even with a native Linux driver it would not help this project**, and it is
not worth the OpenVINO conversion:

- The acoustic model is **kernel-launch-bound, not FLOP-bound** — the measured forward is 175× its
  arithmetic time, at 0.0024 % arithmetic duty cycle. It already costs 0.34 ms against an 80 ms
  cadence. A faster matrix engine cannot speed up a launch-latency-bound workload.
- The actual bottleneck is **WFST beam search** — irregular pointer-chasing over a multi-GB FST,
  which is the workload class NPUs are worst at.

Do not spend time on it. The 22 CPU cores are the useful accelerator here, for the LM build.

### 3.5 The one remaining hardware constraint — system RAM

`.wslconfig` at `/mnt/c/Users/fishi/.wslconfig` sets `memory=25165824000` (~23.4 GiB), so WSL2 sees
**22 GiB of the host's 32 GB**. The plan's LM target is ≤ 19 GB (C28), which does not fit
alongside Python, the cached logits and the OS.

Two honest options — **this is a user action**, since applying it needs `wsl --shutdown`, which
would kill everything currently running:

```ini
# /mnt/c/Users/fishi/.wslconfig  — raise the cap, then run `wsl --shutdown` from PowerShell
[wsl2]
memory=28GB        # was 25165824000 (~23.4 GiB); leaves ~4 GB for Windows
processors=22
swap=17179869184
```

Or **keep 22 GiB and target a ~14 GB LM for development**, treating 19 GB as the deployment figure
for a 32 GB box rather than the dev figure. Swap will not rescue you here — WFST decoding is random
access and would thrash. Record which option you took in §8.1, because it changes C28's kill
criterion.

### 3.6 Remaining setup, and the one open item

```bash
uv tool install "cmake<4"     # MUST be 3.x — see below. gcc 13.3.0 is fine.
./setup_lm.sh                 # creates conda env b2txt25_lm (py3.9) and builds lm_decoder
```

**cmake 4.x breaks this build.** CMake 4 removed compatibility with
`cmake_minimum_required(VERSION <3.5)`, and the vendored gflags declares an older minimum:

```
CMake Error at fc_base/gflags-src/CMakeLists.txt:73 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is CMake's own escape hatch, but the build chains
gflags → glog → openfst → Kaldi, all old, so pinning to **3.31.10** is the lower-risk fix and
still satisfies `setup_lm.sh`'s own `>= 3.14` check.

**`setup_lm.sh` prints "Setup complete!" even when the build fails** — there is no error check
after `python setup.py install` (`setup_lm.sh:57-63`). **Never trust that message.** Verify with:

```bash
$(conda info --base)/envs/b2txt25_lm/bin/python -c "import lm_decoder; print('OK')"
```

To retry only the C++ build after a failure (the conda env and its pip deps are already in place,
and `setup_lm.sh` aborts if `build/` or `fc_base/` exist):

```bash
cd language_model/runtime/server/x86 && rm -rf build fc_base
$(conda info --base)/envs/b2txt25_lm/bin/python setup.py install
```

**Which interpreter to use.** Everything training- and benchmark-side runs under **`.venv`**:
`../.venv/bin/python <script>` from `model_training/`, or activate with `source .venv/bin/activate`.
`.venv` was built with `uv` and reproduces `setup.sh` exactly (all 17 pins, incl. `numpy==2.1.2`),
with torch upgraded to 2.13.0+cu130 for sm_120 — `setup.sh`'s own `cu126` pin does not carry
Blackwell kernels.

**conda is used for exactly one thing:** `setup_lm.sh` hard-requires it (`conda info --base`,
`conda create -n b2txt25_lm python=3.9`) and **python3.9 is not on this system**. That script builds
a C++ extension against CMake/OpenFST/Kaldi — the fiddliest step in the project — so it runs as
written rather than being ported to uv. Do not create a conda env for training.

Note `setup_lm.sh` pins **torch 1.13.1 / Python 3.9** in a separate env. That torch has no sm_120
support — **which does not matter**, because the plan drops `--rescore` and `--do_opt`, so nothing
in the LM env needs the GPU. Install the CPU build if the CUDA wheels give trouble.

**G0.5 — resolve Q3 (5 min, high value).** Find which LM produced the 2.66 % figure: grep the eval
invocation for `--lm_path`, `--do_opt`, `--rescore`. If it was the 1-gram + OPT, essentially all
lexical work sits inside the 620–830 ms batch wall and Group 4's recoverable fraction is much
smaller than estimated.

**Gate G0: PASSED** for everything except (a) `lm_decoder` not yet built, (b) `causal_la0` not yet
retrained, (c) Q3 unresolved. None of the three blocks starting Group 1's Phase 1A.

---

## 4. Group 1 — instrument the pipeline and harvest the free frontier

**0 training runs. ~15 h wall, mostly CPU/eval. This is where most of the predicted WER gain is.**

The audit's central finding is that the acoustic model spends 60.3 ms of a 140 ms budget while the
LM stack spends 620–830 ms. Group 1 is the whole of that asymmetry, and it costs no GPU training.

Group 1 has two phases because 1B strictly depends on a compiled `lm_decoder`. Implement all of 1A
first; it is testable on its own.

### 4.1 Phase 1A — harness, prerequisites, free probes

**C1 — vectorize the streaming smoother.** `streaming_infer.py:194-199` loops over kernel taps in
Python, launching one GPU op per tap; `:201-208` linearly scans the ring buffer for every tap.
Replace both with a single stacked FIR: keep the last `past_taps+1` raw bins in a preallocated
`[K, C]` tensor and compute `(kernel[:, None] * window).sum(0)` once.
Measured target: reference GPU path 0.185 ms/bin → vectorized CPU equivalent 0.0131 ms **[M]**.

**C2 — cache the kernel.** `gauss_smooth` (`data_augmentations.py:15-31`) rebuilds the scipy
kernel and re-copies it host→device on **every call** — per batch in training, per bin in
streaming. Memoize on `(std, size, lookahead, device)`.

**C3 — bound the buffers.** `transformed_buffer` (`:216`) needs only the last `patch_size`
entries; `logit_frames` (`:235`) needs only what the collapse rule reads. Both currently grow for
the whole utterance.

**C4 — `test_equivalence.py:95`** references `frame_diff`, bound only in the `else` branch at `:91`.

**C5 — `rnn_trainer.py:242`**: `zero_infinity=False` → `True`. Combined with
`error_if_nonfinite=True` at `:554`, one trial with `adjusted_lens < phone_seq_lens` is a hard
crash. Group 2 adds masking and raises `random_cut`, both of which can trip it. **Do this before
Group 2, not with it.**

**C6 — `rnn_args.yaml:26`**: `save_all_val_steps: true`. Makes C24 free on every run from here on.

**C7 — freeze the evaluation split.** Held-out-session `val-test` = the 6 sessions from
`t15.2025.01.10` onward, all of which carry `dataset_probability_val=1`:

```
t15.2025.01.10  t15.2025.01.12  t15.2025.03.14
t15.2025.03.16  t15.2025.03.30  t15.2025.04.13
```

(Note: `AUDIT §A8` says "8 sessions"; the config has 6. Verified against `rnn_args.yaml:127-132`
and `:173-178`.) Tune on the remaining 35 val-contributing sessions (`val-dev`). **Touch `val-test`
once per group, at the gate.** Write the split to `model_training/splits.py` as a literal list so
it cannot drift. Record the trial counts in §8.1 — if `val-test` is under 8 % or over 15 % of val
trials, adjust the boundary session and record that you did.

**C8 — causal-normalization identity check.** Zero cost, and it de-risks the two most valuable runs
in Group 4. Write `model_training/causal_normalize.py::rolling_normalize(features,
half_life_bins=500, warmup_bins=50)` — EWMA mean/var, `alpha = 1 - exp(-ln2/half_life_bins)`,
strictly causal (update *after* emitting). Then verify the algebra that makes the fix exact:

> Block z-scoring gives `x_norm = (x_raw − μ_blk)/σ_blk`. Applying a causal rolling normalizer on
> top yields `(x_raw − μ_roll,raw)/σ_roll,raw` — **the block constants cancel identically.** So
> rolling-normalizing the *released* data recovers the causally-normalized raw features with no
> access to raw data.

**Test:** feed the normalizer constant (whole-block) statistics instead of rolling ones; it must
reproduce the released features to within **1e-4**. If it does not, the transform has been
misidentified and R-D2/R-D3 are void before they cost 12.8 GPU-hours.

**C9 — per-phase greedy PER.** New `benchmark/phase_ensemble.py`. For `p in {0,1,2,3}`, run the
existing `causal_la0` checkpoint on `raw_features[p:]` — frames' right edges land at bins
`4f + 13 + p` — and compute greedy PER per phase on `val-dev`.
**This is the deciding test for C16 and it takes ~30 min.**

**C10 — temperature scaling.** The posterior is extraordinarily over-confident: mean entropy
**0.9 % of maximum**, median frame numerically one-hot **[M]**. A near-one-hot posterior gives the
beam search almost no acoustic alternatives, so the LM must overcome a ~1.0 posterior to fix an
acoustic error. Temperature is a post-hoc scalar tuned on `val-dev` — free, no retrain — and it
tests the calibration hypothesis *before* Group 3 spends a run on label smoothing (C20). Sweep
`T ∈ {1.0, 1.25, 1.5, 2.0, 3.0}` on the cached logits and score WER after C13 exists.

### 4.2 Phase 1B — the LM stack (gated on C11)

**C11 — build `lm_decoder`.** `./setup_lm.sh`. Creates conda env `b2txt25_lm`; needs CMake ≥ 3.14
and gcc ≥ 10.1. This is the single hard external dependency in the whole plan.

**C12 — un-hardcode the decoder params.** `language-model-standalone.py:486-496` calls
`build_lm_decoder` with **literal** `max_active=7000, min_active=200, beam=17., lattice_beam=8.` —
the corresponding CLI flags are **dead at startup** and only take effect through the
`remote_lm_update_params` path (`:708-718`). C15 cannot sweep them until this is fixed.

**C13 — the incremental driver. This is the gating measurement of the entire project.**

New `model_training/benchmark/stream_lm.py`:
- `class IncrementalLMDecoder` wrapping `lm_decoder.BrainSpeechDecoder`, exposing
  `push_frame(logits_row) -> partial_hypothesis` and `finalize() -> final_hypothesis`.
- Feed `lm_decoder.DecodeNumpy(decoder, logits[t:t+1], zeros_like, log(blank_penalty))`
  **one frame at a time**, then read `decoder.result()[0].sentence`. This is exactly the path
  `language-model-standalone.py:769-785` already exercises, driven per-frame rather than per-trial.
- Apply `rearrange_speech_logits_pt` (`evaluate_model_helpers.py:79-83`) **once**, before the loop.
- Record per frame: wall time of `DecodeNumpy`, wall time of `result()`, the partial hypothesis,
  and the active-token count.
- Logits come from cached `val_metrics.pkl` via `benchmark/stability.py::load_val_logits` —
  **the acoustic model never re-runs.**
- Do **not** pass `--rescore`, do **not** pass `--do_opt`, set `--nbest 1` for the pure-incremental
  condition (`nbest > 1` triggers `augment_nbest`, an O(n²) batch stage).
- Pipe the per-frame partials into `stability.py::trace_partials` → `stability_metrics`. Those
  functions exist and are already exercised against the acoustic path.

**C14 — blank skipping.** `ctc_blank_skip_threshold` defaults to `1.0`, which can never fire since
`exp(log p) ≤ 1` (`ctc_wfst_beam_search.cc:79-84`, default at `:809`). At θ=0.999, **71.83 % of
frames** are skippable **[M]**. Blank Collapse reports WER 1.783 → 1.781 at θ=0.99, i.e. unchanged.

**C15 — decode sweep.** The repo's own two configurations disagree **10×** on `blank_penalty`
(code default 9.0 vs README recipe 90) and differ on `acoustic_scale` (0.3 vs 0.325) — direct
evidence neither was tuned for this checkpoint. Sweep **coordinate-wise, not full grid**:
`acoustic_scale × blank_penalty` first (they interact strongly), then `beam`/`max_active` at the
best point.

| Parameter | Values |
|---|---|
| `acoustic_scale` | 0.20, 0.25, 0.30, 0.325, 0.40, 0.50 |
| `blank_penalty` | 1, 3, 9, 30, 90 |
| `beam` | 12, 15, 17, 20 |
| `max_active` | 2000, 4000, 7000 |

**C16 — phase merge**, only if C9's gate passes. Two modes, implemented and reported **separately**
because they are different mechanisms:
- **(a) Interleaved 20 ms grid** — emit each phase's frames on a common 20 ms timeline. The LM then
  consumes 50 Hz instead of 12.5 Hz. This delivers the latency win (L_buf 60 → 15 ms worst case)
  and **quadruples the LM frame rate**, which blank skipping at θ=0.999 offsets almost exactly.
- **(b) Phase-averaged logits** — align the four streams by right-edge bin and average. Delivers
  the ensemble win, not the latency win.

Conflating (a) and (b) will make the accuracy delta look free when it is not.

### 4.3 Verification block

```bash
cd model_training
../.venv/bin/python benchmark/test_equivalence.py   # MUST PASS: tol <1e-3, exact collapsed-seq match
../.venv/bin/python benchmark/offline_benchmark.py  # → metrics.json
../.venv/bin/python benchmark/streaming_infer.py    # RTF, per-patch p50/p95
../.venv/bin/python -m benchmark.causal_norm_check  # C8 identity check, must be <1e-4
```

C1–C3 are **bit-identical by construction**. If `test_equivalence.py` fails after them, the
vectorization is wrong — do not proceed, and do not loosen the tolerance.

### 4.4 Decision gate G1

| Gate | Condition | If it fails |
|---|---|---|
| **G1-a** equivalence | `test_equivalence.py` PASSES after C1–C3, at unchanged 1e-3 tolerance | revert C1–C3, they are not free |
| **G1-b** streaming cost | measured per-window compute drops ≥ 50 % (target 1.06 → ≈0.37 ms **[D]**) | C1 did not land; profile before claiming any latency number |
| **G1-c** phase equivalence (H3a) | per-phase greedy PER spans **≤ 0.5 pts** across p ∈ {0,1,2,3} | C16 needs `random_cut: 4` first — it is already in Group 2, so **defer C16 to after Group 2** rather than spending a run |
| **G1-d** identity check | rolling normalizer reproduces released features to **< 1e-4** | R-D2/R-D3 are void. Re-derive the transform or reframe Group 4 as "causal re-normalization" rather than "leak removal" |
| **G1-e** GATING: incremental fits | p95 per-frame LM latency **≤ 79 ms** at a usable beam | reduce the search (`max_active` down, `beam` tighter, θ=0.9) and report the frontier that *does* fit. **A measured infeasibility result is still the first published latency accounting for this system** — do not treat it as failure |
| **G1-f** blank skipping is free (H1a) | θ=0.999 changes WER **≤ 0.10 pts** while cutting WFST advance steps **≥ 60 %** | fall back to θ=0.99, then 1.0 |
| **G1-g** frontier is non-degenerate (H1b) | decode sweep traces **≥ 1.0 WER pts** of spread across configs spanning 40–140 ms p95 | the frontier framing fails and the honest report is a second negative result. Say so |

**Record Q1 and Q2 in §8.2 the moment C13 produces them. Every WER estimate in this file is
provisional until then.**

---

## 5. Group 2 — the regularization block

**1 training run, ~5.9 h measured. Hypothesis: structured masking is the missing regularizer.**

`[AUDIT §A2]` finds the repo's only live augmentations are additive white noise (σ=1.0 on
unit-variance features — a 100 % noise-to-signal perturbation) and a constant offset. There is **no
time masking, no channel masking, no mixup, no warping, no label smoothing, no cutout**. Both
competition retrospectives frame this task as a regularization problem, and the closest published
causal result attributes 20–26 % relative WER improvement to masking ~53 % of every trial
`[LITERATURE B3.4]`.

All three changes are train-time only. **L_algo 0, L_buf 0, L_comp 0.** Under a hard latency cap,
mechanisms that are free at inference dominate any that are not.

### 5.1 Changes

**C17 — time masking.** Add to `data_augmentations.py`:

```python
def time_mask(features, n_masks=20, max_frac=0.075, mask_value=0.0, generator=None):
    """features [B, T, C]. Sample S ~ U[0,T), D ~ U[0, max_frac*T) per mask, per trial.
    Masks may overlap; expected coverage at (20, 0.075) is ~53%."""
```
- Sample **per trial**, not per batch — unlike `random_cut`, which draws one scalar for the whole
  batch at `rnn_trainer.py:469`.
- Apply in `transform_data` **after** `gauss_smooth` (`rnn_trainer.py:475-482`), so the mask is
  genuine information removal rather than something the smoother partially fills back in.
- Guard with `if mode == 'train'`.

**C18 — channel masking.** Zero a random 10 % of the 512 feature channels per trial. Apply
**before** the day layer, or the day-specific affine absorbs the perturbation and the change does
nothing. Same `mode == 'train'` guard.

**C19 — `random_cut: 3 → 4`** (`rnn_args.yaml:67`). `np.random.randint(0, 3)` yields `{0,1,2}` —
3 of the 4 patch phases. Raising it to 4 gives full phase coverage at zero cost and is the enabling
change for C16.

New config keys under `dataset.data_transforms:`:
```yaml
    time_mask_n: 20              # masks per trial
    time_mask_max_frac: 0.075    # max mask length as a fraction of trial length
    channel_mask_rate: 0.10      # fraction of the 512 channels zeroed per trial
    random_cut: 4                # was 3 → full patch-phase coverage
```

**Deliberately NOT in this group,** and why:

| Excluded | Reason |
|---|---|
| `smooth_kernel_std` sweep | entangled with the F2 confound. It is a **validity** question, not a regularization one → Group 4 (C25) |
| `white_noise_std` 1.0 → 0.5 | moves *opposite* to C17/C18. Bundling opposed changes makes a neutral result uninterpretable → micro-tuning (§7.6) |
| dropout changes | direction unknown, and C17/C18 already change the regularization budget → knockout knob if Group 2 over-regularizes |
| stochastic depth | requires decomposing the fused `nn.GRU(num_layers=5)` (`rnn_model.py:65-72`) into 5 modules, breaking `state_dict` compatibility with `causal_la0` and the streaming `_step_gru` path — for an expected −0.20 PER. Bad ratio |
| larger `patch_size` | a **capacity** change, and the evidence says capacity is not the lever (S2 rejected; the field moved to 83 % fewer parameters). Do not bundle a contested-direction change with expected-positive ones |

### 5.2 Config

```yaml
output_dir: trained_models/g2_masking
checkpoint_dir: trained_models/g2_masking/checkpoint
seed: 10                    # unchanged — must match causal_la0
save_all_val_steps: true    # C6
```
Everything else identical to `causal_la0`. **Never clobber `trained_models/causal_la0`.**

### 5.3 Test block

```bash
cd model_training
# 1. SMOKE (mandatory, ~7 min). Set num_training_batches: 2000 in a scratch copy of the config.
../.venv/bin/python train_model.py         # watch for: OOM, NaN loss, inf CTC, shape errors
# 2. FULL (~5.3 h at 120k batches; 0.152-0.200 s/batch measured)
#    ALWAYS detach, so a killed shell cannot orphan it holding VRAM:
setsid nohup ../.venv/bin/python -u train_model.py > run.log 2>&1 < /dev/null &
#   train_model.py now sets PYTORCH_CUDA_ALLOC_CONF itself; benchmark/eval scripts do NOT —
#   export it manually for those.
# 3. Evaluate
../.venv/bin/python evaluate_model.py               # val PER
../.venv/bin/python benchmark/test_equivalence.py   # streaming equivalence on the new checkpoint
../.venv/bin/python benchmark/stability.py          # posterior stats, blank fraction, emission delay
../.venv/bin/python benchmark/stream_lm.py          # held-out WER via the Group 1 decoder
```

The smoke run is what makes this bundle safe. C5 (`zero_infinity=True`) must already be in place —
masking plus `random_cut=4` is exactly the combination that can push `adjusted_lens` below
`phone_seq_lens` and hard-crash 5 hours in.

### 5.4 Decision gate G2

Threshold is set at **7× the measured 0.041-pt same-config reproducibility floor** `[AUDIT F9]`, so
a positive result cannot be explained by run-to-run variance.

| Outcome | Condition | Action |
|---|---|---|
| **Strong pass** | val PER ≤ **9.50 %** | keep the whole bundle; Group 3 builds on it |
| **Pass** | val PER ≤ **9.75 %** | keep the whole bundle; Group 3 builds on it |
| **Neutral** | 9.75 % < PER ≤ 10.08 % | keep — free at inference and it may still help WER. Report as neutral-on-PER. Check held-out WER before deciding |
| **Fail** | PER > 10.08 % (i.e. worse than baseline by > noise) | **bisect, do not re-run all three.** See §7.5 |

Also record, because they are the second contribution and cost nothing extra: posterior entropy
(does masking de-sharpen it?) and blank fraction (does C14's 71.83 % skippable figure survive?).

**If Group 2 fails outright at every mask fraction**, that is genuinely informative: it means the
20–26 % relative gain in `[LITERATURE B3.4]` belongs to the *transformer*, not the masking, which
materially raises the value of the causal-transformer candidate (25.6 h, currently below the cut
line). Promote it only in that case.

---

## 6. Group 3 — the objective and optimization block

**1 training run, ~5.9 h. Built on Group 2's winner** (or on `causal_la0` if G2 failed — record which).

Group 2 changed what the model sees. Group 3 changes how the loss is computed and minimized.
Mechanically disjoint from Group 2, which is why they can be sequential bundles rather than a
2×2 grid.

### 6.1 Changes

**C20 — label smoothing / entropy regularization on CTC.** This is the one change in the plan that
came out of measurement rather than the literature. Mean posterior entropy is **0.033 nats against
a maximum of ln(41) = 3.714 — 0.9 % of maximum**, with a numerically one-hot median frame **[M]**.
Add an entropy bonus or label-smoothing term at `rnn_trainer.py:242,539`, smoothing ∈ {0.05, 0.1}.

**Expect greedy PER to get slightly *worse*.** This candidate is not about PER — it trades argmax
sharpness for calibration so the beam search has acoustic alternatives to work with. It is the rare
case where PER and WER should move in opposite directions, which makes it a clean test of whether
PER is the right proxy at all. **Gate it on WER, never on PER.** C10 (temperature scaling, free, in
Group 1) is the cheap preview of this hypothesis — if C10 shows no WER movement, drop C20 from the
bundle.

**C21 — intermediate / self-conditioned CTC.** Auxiliary CTC heads after GRU layers 2 and 4,
projecting their phoneme posteriors back into the hidden state before the next block. This
conditions deeper layers on shallower layers' predictions, partially relaxing CTC's
conditional-independence assumption. Nothing about it requires future context. Aux weight 0.3.
Heads are dropped at inference — **zero latency cost**. Adds 2 × (768×41) params.

**C22 — day-layer regularization.** `lr_max_day: 0.005 → 0.0025`, `lr_min_day: 0.0001 → 0.00005`,
`weight_decay_day: 0 → 0.0001` (`rnn_args.yaml:38-39,47`). The day layers are **26.7 % of all
parameters** (45 × 262,656) trained at the same peak LR as the main model with **zero** weight
decay, while per-day training counts range 59–364 trials. That is a lot of session-specific
capacity on low-data days.

**C23 — `weight_decay: 0.001 → 0.005`** (`rnn_args.yaml:46`). Matches the 7th-place B2T'25 entry;
current value is inherited and untuned.

**C24 — checkpoint averaging.** Average the weights of the last k=10 validation checkpoints.
`best_val_PER` is a **minimum over 61 correlated evaluations** and is ~0.04 pts optimistically
biased `[AUDIT F8]`; late-training val PER has sd 0.033 pts across the last 10 evals **[M]** — real
variance to average away. **Zero inference cost** (it produces a single model). Free given C6.
Average only within the flat region of the LR schedule.

**Deliberately NOT in this group:** `epsilon: 0.1 → 1e-8`. It is 10⁷× the PyTorch default and
almost certainly a fossil, but it is **coupled to `lr_max`** — `epsilon=0.1` largely disables Adam's
adaptive term, so `lr_max=0.005` was tuned for signed-SGD-like behavior. Restoring standard Adam at
that LR is the documented instability risk in the source catalog. It is not a clean addition to a
bundle; it is a 2-variable decision. → micro-tuning (§7.6), where it gets a paired sweep.

### 6.2 Test block

Identical to §5.3, with `output_dir: trained_models/g3_objective`. **Smoke run first.** C21 changes
the model graph, so the smoke run is also the shape check.

After the full run, produce **two** checkpoints and evaluate both: the best single checkpoint and
the k=10 average (C24).

### 6.3 Decision gate G3

| Outcome | Condition | Action |
|---|---|---|
| **Pass** | held-out-session **WER** improves ≥ **0.30 pts** over the Group 2 model, at equal decode config | keep; this is the Group 4 base model |
| **Split** | WER improves but val PER regresses | **expected** if C20 is doing its job. Keep, and report the PER/WER divergence — it is a finding, not a problem |
| **Neutral** | |ΔWER| < 0.30 pts | keep C24 (free, single model) and C22/C23 (free); drop C20 and C21 from the final config to reduce the manuscript's surface |
| **Fail** | WER regresses ≥ 0.30 pts | bisect (§7.5). C20 is the first knockout candidate — over-smoothing can flatten the posterior enough that blank skipping stops firing, silently raising LM cost |

**Re-run C15 (the decode sweep) after this group.** The optimal `acoustic_scale` moves whenever the
acoustic posterior changes, and C20 changes it by design. Failing to re-sweep will understate
Group 3.

---

## 7. Group 4 — validity, seeds, and the deployable stack

**4 training runs (~21 h) + LM work that needs no GPU.**

Group 4 is grouped by *purpose*, not by bundle. **Each run inside it is single-variable, and that is
not negotiable** — the deliverable of each is an isolated number, and a bundled version of any of
them measures nothing.

**Schedule the LM work (C28–C31) to run concurrently with the training runs.** The 4-gram build is
CPU- and RAM-bound; the retrains are GPU-bound. They do not contend.

### 7.1 The runs

**R-D1 → C25 — bandwidth-matched causal control. Resolves the project's central confound.**

`smooth_lookahead` changes three things at once **[M]**:

| L | taps | lookahead | group delay | **effective σ** | peak weight |
|---|---|---|---|---|---|
| **0** | 5 | **0 ms** | +24.5 ms | **1.161 bins** | 0.339 |
| **4** | 9 | **80 ms** | 0 ms | **1.852 bins** | 0.204 |

Going L=4 → L=0 removes 80 ms of lookahead **and narrows the low-pass by 37 %**. So the observed
result — L=0 better by 0.163 pts, with a clean val-loss gap (21.74 vs 22.60) — has an obvious
alternative explanation with nothing to do with causality: **σ=2 over-smooths, and truncation
accidentally fixed it.**

Build a causal (`lookahead=0`) kernel whose σ_eff matches 1.852 bins by extending the past tail.
Run against the Group 3 winner, same seed.
- Lands near L=4's PER → the gain was **bandwidth**. `smooth_kernel_std` becomes a real tuning
  lever with a known direction.
- Lands near L=0's PER → something else is happening, and the causality claim survives.

**Both outcomes are publishable. Not knowing is not.** This run cannot be killed — that is what
makes it a control.

#### The σ disagreement, resolved 2026-08-06

`AUDIT F2` implies **narrower** smoothing is what won; `RESEARCH_PROPOSALS §1` recommends
**wider** (σ=2.5, then 3.0). These are **not two experiments — they are one experiment with two
opposite predictions**, which is the best case: the run falsifies one document either way.

Measured causal-kernel properties at `lookahead=0`, same trim rule as `data_augmentations.py:18`:

| `smooth_kernel_std` | taps | **σ_eff (bins)** | mean past lag | peak weight |
|---|---|---|---|---|
| 1.0 | 3 | 0.637 | 10.1 ms | 0.574 |
| **1.5** | 4 | **0.902** | 17.3 ms | 0.426 |
| **2.0 (current)** | 5 | **1.161** | 24.5 ms | 0.339 |
| 2.5 | 6 | 1.417 | 31.7 ms | 0.282 |
| 3.0 | 7 | 1.672 | 38.9 ms | 0.241 |
| **3.4** | 8 | **1.899** | 45.1 ms | — |
| *L=4 symmetric (the target)* | *9* | *1.852* | *0 ms* | *0.204* |

Two things fall out that neither source document had:

1. **The bandwidth-matched control is `smooth_kernel_std: 3.4` at `lookahead: 0`** (σ_eff 1.899,
   nearest reachable to L=4's 1.852). σ=3.0 reaches only 1.672 and is **not** a bandwidth match —
   a run at 3.0 would not have answered the question.
2. **Lowering the trim threshold cannot produce the control.** `RESEARCH_PROPOSALS`' proposed
   `smooth_kernel_min_weight` hook saturates at σ_eff ≈ 1.28 even at trim = 1e-4, because the
   Gaussian's own tail bounds it. Useful hook, wrong mechanism here — raise σ instead.

**Better option for the default config: narrower.** In order of weight:
- It is the only direction supported by a measurement on *this* model and dataset — σ_eff 1.161
  beat σ_eff 1.852 by 0.163 pts with a clean val-loss gap (21.74 vs 22.60).
- `patch_size=14` **already integrates 280 ms of past context**. The smoother's remaining job is
  denoising, not context aggregation; widening duplicates what the patch already does.
- `RESEARCH_PROPOSALS`' widening argument rests on Wairagkar's 1.5 s causal kernel, from a
  **voice-synthesis** system with a different signal path — a weaker transfer than an in-project
  measurement.

**So R-D1 is two runs on one axis** — the same axis the disagreement is about:

| Run | Config | AUDIT F2 predicts | RESEARCH_PROPOSALS predicts |
|---|---|---|---|
| **R-D1a** (control) | `smooth_kernel_std: 3.4`, `lookahead: 0` | PER rises toward **10.21** — the L=0 win was bandwidth | PER **falls** below 10.04 |
| **R-D1b** (lever) | `smooth_kernel_std: 1.5`, `lookahead: 0` | PER **falls** below 10.04 | PER rises |

Run **R-D1a first**. If it lands near 10.21 the bandwidth explanation holds and R-D1b is worth the
second run; if it lands near 10.04, bandwidth is not the story and R-D1b can be skipped. Either
way the streaming ring buffer must be resized for the longer past tail and the equivalence gate
re-run — 8 taps at σ=3.4 vs 5 at σ=2.

**R-D2 / R-D3 → C26 — causal rolling normalization, seeds 10 and 11. The highest-novelty item in
the project.**

Every "real-time" number this project has produced rests on inputs containing up to ~19 minutes of
future information `[AUDIT F1]`. Re-normalize causally and retrain. **Gated on C8's identity check
passing at < 1e-4.**

- Apply in `dataset.py::BrainToTextDataset.__getitem__`, immediately after `input_features` is read
  at `:130`, before `pad_sequence`.
- Per-trial application resets the estimator at each trial boundary. **Accept this for v0 and record
  it as a limitation** — trials are ~18 s so the 10 s half-life warms within each trial.
- Initialize `(mean, var) = (0, 1)`. **State this honestly as a residual assumption**: it uses the
  fact that the data is z-scored at block scale. It is far weaker than per-feature block statistics
  and matches what a deployed system does at block start using the previous block's stats.
  Document it; do not hide it.

New config keys under `dataset:`:
```yaml
  causal_normalize: true
  causal_norm_half_life_bins: 500     # 10 s at 20 ms, matching Wairagkar et al.
  causal_norm_warmup_bins: 50
```

The streaming path (`benchmark/common.py`, `streaming_infer.py`) needs the same normalizer applied
before `smooth_features` / `_append_raw`, or the equivalence gate will fail.

**Expected: PER *rises* by 0.3–1.5 pts [E].** That is the result. It is the true price of
end-to-end causality on T15, which nobody has published for any participant.

| Gate | Condition |
|---|---|
| **H2** — the leak is real and costly | PER rises ≥ **0.20 pts** on ≥ 2 of 3 seeds (~5× the 0.041 noise floor) |
| **H2-null** — real but harmless | change < 0.20 pts on ≥ 2 of 3 seeds. **A stronger and more surprising result** — it licenses every prior number in the project as real-time-valid |
| **Void** | C8's identity check failed. Report the 1/√W evidence as establishing block-*scale* normalization without claiming the exact functional form, and reframe as "causal re-normalization" rather than "leak removal." The scientific point survives |

**R-D4 → C27 — seed replication.** Re-run the winning config at seed 11 (vary **only** the
torch/numpy seed at `rnn_args.yaml:48`, never `dataset.seed` at `:78`, or the split changes). The
headline currently has n=1 and no error bar.

### 7.2 The LM/stack work (0 GPU training)

**C28 — pruned 4-gram ≤ 19 GB.** Build a KenLM 4-gram with aggressive pruning and quantization,
compiled to a TLG WFST, per the 7th-place recipe (80 % general text + 20 % conversational).
Toolchain is in `language_model/srilm-1.7.3/` and `language_model/tools/fst/`. **CPU work.**
The only LM present locally is the 1-gram (13.4 MB), so the −2.25 WER estimate is *1-gram → 4-gram*
and is **the least trustworthy number in the catalog** — if the constrained baseline was
established with a 3-gram, most of that gain vanishes.
*Fallback:* if the 4-gram cannot be pruned under 32 GB without destroying it, fall back to a 3-gram
and report the RAM/WER frontier, which is itself unpublished.

**C29 — causal neural LM shallow fusion.** 100–500 M causal LM scored incrementally during beam
search, fusion weight `lambda_nlm ∈ {0.1, 0.2, 0.3, 0.5}`. Bounded per-frame cost instead of an
end-of-utterance wall. **Do not bundle with C28** — both change the LM score scale.

**C30 — endpointing.** `ctc_endpoint.{h,cc}` ships in the decoder and is never instantiated. A
median **1,680 ms of trailing silence** follows the last emitted token **[M]**. Expect a small WER
regression (+0.05 **[E]**) bought for ~1.7 s of finalization latency. Kill if WER regresses > 0.2 pts
at any threshold saving > 1 s.

**C31 — adaptive-compute gate.** Gate on **n-gram confidence**, not acoustic entropy — only 2.9 % of
frames exceed entropy 0.5 and the model is *confidently wrong* rather than uncertain **[M]**. The
7th-place entry gates at −3.76. **Report expected latency and worst case separately, and confront
the exposure rather than burying it:** this reports an *expected* latency against a *hard*
constraint, and worst case is still 620–830 ms. A reviewer can reasonably reject that for a
real-time claim.

**Re-run C15 after C28 and again after C29.** The optimal `acoustic_scale` moves whenever the LM
changes; not re-sweeping understates both.

### 7.3 Final deliverable block

Once Group 4 lands, produce in one pass:
- End-to-end latency ledger pricing **every** stage — acoustic, WFST advance, partial extraction,
  fusion, finalization. An acoustic-only ledger is incomplete; say so if that is all you have.
- Word-level stability metrics (revisions/word, time-to-final, time-to-first-emission) against the
  LM. The acoustic decoder is provably flicker-free (0 revisions / 41,039 tokens **[M]**), so
  **100 % of user-visible flicker is LM-side** and this metric is only meaningful there. No
  brain-to-text paper reports it.
- Streaming equivalence gate PASSING on the final model.
- Held-out-session WER on `val-test`, touched once.

### 7.4 Optional, promote only if budget remains

| Item | Cost | Promote when |
|---|---|---|
| Deep ensemble N=3 (seeds only) | 12.8 h | Group 3 lands and ≥ 13 h remain. A **null result is publishable** — first evidence the field's most-cited technique stops paying at low baseline error |
| TTA, DietCORP-style, per-trial | 1.5 h | anytime. 18.21 ± 5.34 ms/trial fits the budget. **Breaks streaming determinism — report separately** |
| Causal transformer | 25.6 h | **only if Group 2 fails.** That failure would prove the source paper's gain belongs to the architecture, not the masking |

### 7.5 Knockout protocol — what to do when a bundle loses

Do **not** re-run every variable individually. Bisect:

1. Split the losing bundle into two halves, weighted by prior strength (e.g. Group 2 → {C17} vs
   {C18, C19}).
2. Run the half carrying the strongest prior. **1 run.**
3. If it passes, the other half is the culprit — drop it and continue. If it fails, the strong-prior
   change itself is the problem, which is the more interesting result.

**Two runs maximum resolves a losing bundle of three to five changes.** That is still far cheaper
than one-variable-per-run, which is the whole point of this plan.

### 7.6 Micro-tuning — deferred until Groups 1–4 land

Only after the operating point is settled. Each is single-variable, and none should be started
before Group 4's gates are recorded.

| Item | Why deferred |
|---|---|
| `epsilon` × `lr_max` paired sweep | coupled 2-variable decision with documented instability risk |
| `white_noise_std` ∈ {0.5, 1.0} | opposes C17/C18; only interpretable once masking is settled |
| `smooth_kernel_std` sweep | direction is adjudicated by R-D1, not before |
| dropout rates | regularization budget changed under it in Group 2 |
| larger `patch_size` (22/34) | capacity, and the evidence says capacity is not the lever |
| diphone auxiliary head | conflicts with C21 — both add auxiliary CTC heads |
| leave-one-out attribution ablations | 1 run each; spend only on changes a reviewer will challenge |

---

## 8. Results ledger — fill this in as you go

### 8.1 Environment (Group 0) — measured 2026-08-06

| Field | Value |
|---|---|
| GPU / VRAM | RTX 5070 Ti / 16,303 MiB, sm_120 **[M]** |
| torch / CUDA / capability | **2.13.0+cu130** / (12, 0); `sm_120` in `get_arch_list()`; matmul, cuDNN GRU, bf16 all execute **[M]** |
| CPU / system RAM | Core Ultra 7 270K Plus, 22 cores / **22 GiB visible to WSL2** (32 GB host) **[M]** |
| **Peak training VRAM @ batch 64** | allocated **7.41 GB**; reserved **7.58 GB** with expandable_segments (**22.87 GB** without) **[M]** |
| Peak validation VRAM @ batch 64 | 2.72 GB at the longest val trial (2,382 bins) **[M]** |
| **s / batch** | **0.169 steady** *with* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; ~1.5 without **[M]** |
| **Projected 120k-batch run** | **~5.9 h** (5.6 h train + ~0.25 h of 61 validations) **[D]** |
| Dataset | 45 sessions, 13 GB; 1,426 val trials **[M]** |
| Checkpoint on hand | `pretrained_baseline` (upstream, **non-causal** — no `smooth_lookahead` key) **[M]** |
| NPU | not exposed to WSL2, and not useful — workload is launch-bound **[M]** |
| WSL RAM decision (§3.5) | *TBD — raise `.wslconfig` to 28 GB, or cap the dev LM at ~14 GB* |
| `causal_la0` / `causal_la4` checkpoints | **recovered from the Mac 2026-08-06** (`fetch_checkpoints.sh`) — full checkpoint + `val_metrics.pkl` for both, `smooth_lookahead` 0 and 4 confirmed in the saved `args.yaml`. The anchors are on disk; **the retrain is unnecessary** and was abandoned at batch 24.4k/120k (it was launched at 14:33, before the pull landed at 15:15) |
| `val-dev` / `val-test` trial counts | **1,253 / 173** (12.1 %), 6 held-out sessions from `t15.2025.01.10` **[M]** |
| Which LM produced 2.66 % (Q3) | *TBD* |

### 8.2 Gating measurements (Group 1)

| Field | Value | Date |
|---|---|---|
| **Q1** — incremental WFST cost per frame | **p50 0.072 / p95 0.712 / max 59.2 ms** vs 79 ms budget — **PASS, 111x margin at p95** [M] | 2026-08-06 |
| **Q1 caveat** | max 59.2 ms is only **1.3x** under budget. Worst-case claims must quote max, not p95 | 2026-08-06 |
| **Q2** — constrained-baseline WER (1-gram, incremental, no rescore/OPT, untuned) | **42.42 %** over 1,426 trials — **not the estimated ~5.5 %** [M] | 2026-08-06 |
| Incremental vs whole-trial equivalence | byte-identical sentence; incremental also *faster* (29.9 vs 58.3 ms/trial) [M] | 2026-08-06 |
| val-dev / val-test trial counts | **1,253 / 173** (12.1 %), 6 held-out sessions from t15.2025.01.10 [M] | 2026-08-06 |
| Per-window streaming compute, before → after C1 | smoother isolated, same machine: CUDA **0.0686 → 0.0364 ms/bin** (−47 %) at K=5, **0.0979 → 0.0135** (−86 %) at K=9; CPU −56 % / −80 %. Max abs diff **2.4e-07** (float reduction order). End-to-end after: per-bin mean 0.229 / p95 0.697 ms, per-patch p50 0.574 / p95 0.649 ms, **RTF 0.0114** [M] | 2026-08-06 |
| Per-phase greedy PER, p ∈ {0,1,2,3} (max−min) | 8.849 / 8.868 / 8.813 / 8.904 % on val-dev → spread **0.091 pts**, **G1-c PASS**. Phase 3 (never trained — `random_cut` draws {0,1,2}) is worst by only +0.055 [M] | 2026-08-06 |
| Blank-skip θ=0.999: ΔWER / decode time per trial | **ΔWER 0.00 pts** (identical at θ ∈ {1.0,.999,.99,.9}); 35.7 → 35.3 ms/trial at θ=.999, **28.0 at θ=.9 (−22 %)**. p50/p95 per-frame unchanged — wrong instrument [M] | 2026-08-06 |
| Best decode config (`acoustic_scale`, `blank_penalty`, `beam`, `max_active`) | **1.0 / 30 / 17 / 7000** — interior optimum, confirmed by extending the grid [M] | 2026-08-06 |
| Decode-sweep WER spread | **20.2 pts** across the grid (35.03 → 55.2 %) [M]. **Gate G1-g does not apply as written**: the entire sweep lives at p95 **0.24–2.18 ms**, not 40–140 ms, so there is no latency axis to trade accuracy against — higher `acoustic_scale` is both more accurate *and* cheaper | 2026-08-06 |
| **n-best oracle @ nbest=100 (val-dev)** | **15.17 %** vs 36.04 % 1-best — a **20.87-pt** lexical gap the 1-gram cannot order [M] | 2026-08-06 |
| **C28-A in-domain trigram re-rank** | 36.04 → **18.33 %**; unseen-sentence subset 36.24 → **18.83 % (−17.41)**; 6.94 % verbatim train/val sentence overlap, scored separately [M] | 2026-08-06 |
| n-best finalization cost (nbest=100) | p50 3.46 / p95 6.42 / **max 15.33 ms** per trial, once per utterance [M] | 2026-08-06 |
| **C28 general 4-gram (val-dev)** | **8.49 %** all / **8.72 %** unseen / 5.41 % seen, at `acoustic_scale` 0.4 / `blank_penalty` 90. **−27.0 pts** vs the 1-gram baseline; catalog estimate was −2.25 [M] | 2026-08-07 |
| C28 build cost | 2.0 GB corpus → 61.7 M n-grams (12.1 GB peak) → `-prune 1e-8` → 14.9 M n-grams → **TLG 3.94 GB at 15.0 GB peak**. Build RSS ≈ 3.8× TLG size [M] | 2026-08-07 |
| C28 interpolation λ (unseen val-dev ppl) | **λ=0.3** on the in-domain LM: general-only 103.6, **mixed 91.5**, in-domain-only 423.5 [M] | 2026-08-07 |
| Offline acoustic stage timing (57 val trials, CUDA) | forward 13.38 ms mean / 21.14 p95; smoothing 0.21; greedy CTC 1.59; **total 15.17 ms/utterance**, RTF 0.0011 [M] | 2026-08-06 |
| C16(b) phase-averaged logits | **36.20 %** vs 35.51 % baseline — **+0.69 pts, a cost.** The accuracy half of C16 does not pay [M] | 2026-08-06 |
| C16(a) interleaved 50 Hz | **35.97 %** (+0.46 pts) at `acoustic_scale` **0.25** — buys **L_buf 60 → 15 ms**. Per-frame p50 0.097 / p95 0.303 / **max 44.94 ms against a 20 ms cadence** [M] | 2026-08-06 |
| Best temperature / ΔWER | **T=1.5 → 35.46 % vs 35.51 % at T=1.0: −0.05 pts.** Calibration is *not* the binding problem; T=2 and T=3 are worse (36.00, 36.61) [M] | 2026-08-06 |
| Identity check residual (must be < 1e-4) | **3.9e-05** affine-invariance residual over 7 blocks / 4 sessions — **G1-d PASS**. 0.0175 % of (t,channel) emissions hit the variance floor and are excluded [M] | 2026-08-06 |

### 8.3 Training runs

Record **all eight** fields per run. Missing fields are how comparisons become uninterpretable.

| Run | Config dir | Changes | val PER (best) | val PER (last-10 mean) | val CTC loss | held-out WER | p95 emission latency | equiv gate | min | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `causal_la0` | `trained_models/causal_la0` | baseline | 10.04 | 10.087 | 21.74 | — | — | PASS | 372 | anchor |
| `causal_la4` | `trained_models/causal_la4` | L=4 control | 10.21 | 10.250 | 22.60 | — | — | PASS | 396 | anchor |
| **G2** | `trained_models/g2_masking` | C17,C18,C19 | **9.430** | **9.499** | 18.96 | **9.10 % val-dev** (base 8.49) | p95 0.576 ms | — | 340 | **PER strong-pass, WER REJECT** — see §1.10 |
| **G3** | `trained_models/g3_objective` | C20–C24 | | | | | | | | |
| **R-D1** | `trained_models/d1_bandwidth` | C25 | | | | | | | | |
| **R-D2** | `trained_models/d2_causalnorm_s10` | C26 | | | | | | | | |
| **R-D3** | `trained_models/d3_causalnorm_s11` | C26, seed 11 | | | | | | | | |
| **R-D4** | `trained_models/d4_seed11` | winner, seed 11 | | | | | | | | |

### 8.4 Recording rules

- **Immutable configs.** Fresh `output_dir` / `checkpoint_dir` per run, named as in §8.3. Never
  clobber `causal_la0`, `causal_la4`, or a completed group directory.
- **Copy the resolved `args.yaml` into the run directory** — it is the only record of what actually
  ran. The trainer already does this; verify it did.
- **Record `val PER (last-10 mean)` alongside `best`.** `best` is a minimum over 61 correlated
  evaluations and carries ~0.04 pts of optimistic bias `[AUDIT F8]`. The last-10 mean is the honest
  comparator.
- **Report deltas against the 0.041-pt reproducibility floor**, not against zero. Anything under
  ~0.08 pts is not a result.
- **Tag every number** [M] / [D] / [E] / [U]. An untagged number in this ledger is a bug.
- **`val-test` is touched once per group**, at the gate. Everything else runs on `val-dev`.
- Append a terse entry to `log.md` per group: `Shipped:` / `Learned:` bullets, numbers and verdicts,
  no prose.

---

## 9. Explicitly excluded — do not re-propose

Each was proposed and refuted. Re-proposing one signals the reasoning chain drifted.

**Killed by measurement:**

| Item | Killed by |
|---|---|
| Delay-penalized CTC, Bayes-Risk CTC, Peak-First CTC, TrimTail, Align-With-Purpose (38.4 h) | **Measured learned emission delay = +4.64 ms** — 6 % of one frame — over 13,030 paired tokens, after subtracting the smoother's predicted group delay. There is nothing to remove `[BUDGET §3]` |
| Reduced `patch_size` to cut cold start (12.8 h) | **Median time-to-first-token = 3,380 ms** vs a 280 ms cold-start fill. The fill is entirely absorbed by the pre-speech period `[BUDGET §2]` |
| FastEmit, delay-penalized transducer | **Transducer-only.** There is no transducer in this system |

**Rejected on evidence:**

| Item | Reason |
|---|---|
| Deep ensemble N=10 | 45 % of budget, sublinear return (N=3 scores 3× better), zero novelty, and it improves the stage already 40× under budget. Gain was measured at a **12× higher baseline error** |
| Wider / deeper GRU | Compute premise is true but irrelevant — the binding constraint is 45 sessions from one participant. Closest published causal result won with **83 % fewer parameters**. Sign of ΔPER genuinely in doubt |
| Mamba / SSM encoder | Two independent controlled comparisons say it matches but does not beat the GRU. Its real value is ensemble diversity, not standalone accuracy |
| Dynamic chunk training | Presupposes a chunked encoder; the model is already fully causal with zero lookahead. Nothing to shrink |
| Mixup / temporal warping | Weak prior; mixup with variable-length CTC targets degenerates to input-only mixing |
| Cross-dataset T12 pretraining | 19.2 h, and the source confounds joint training with hierarchical CTC and a bidirectional 2048-wide GRU. Authors note T15 gains least |
| Predictive distillation from a bidirectional teacher | Pre-empted by Delayed-KD, and undercut by the +4.64 ms measurement — there is almost no shift to apply |
| Day-adversarial training | The day-affine layers (26.7 % of params) already provide an explicit supervised mechanism |
| Incremental LLM rescoring over a rolling context | Repeated partial rescoring thrashes — each update revises earlier words, directly worsening the stability metric this project wants to own |

**Permanently rejected (structural):**

1. **Left-padding the patcher** (`F.pad(x, (P−1, 0))`). Changes `num_patches` (22→25 at T=100),
   breaks CTC frame count and checkpoint compatibility, and adds **zero** causality. Rejected four
   times.
2. **Labeling the slid-symmetric smoother as real-time.** It carries 80 ms group delay. Sliding the
   pad is not truncating the kernel; only truncation reaches true zero delay. It is a frontier
   endpoint, not a real-time configuration.
3. **Calibration-efficiency work.** Out of scope.

---

## 10. Open questions that block claims

| # | Question | Blocks | Status |
|---|---|---|---|
| ~~Q1~~ | ~~Per-frame incremental WFST cost~~ | — | **MEASURED: p95 0.712 ms, max 59.2 ms** |
| Q2 | Constrained-baseline WER | every ΔWER in this file | C13 |
| Q3 | Which LM produced 2.66 % | how much is reachable incrementally | G0.5 |
| Q6 | Is the L=0 advantage causality or bandwidth? | the project's headline | R-D1 |
| Q7 | Does the la0-vs-la4 difference survive replication? | error bar on the headline | R-D4 |
| Q8 | Is the patch embedding phase-sensitive? | C16 | C9 |
| Q9 | Can a 4-gram be pruned under 32 GB without destroying it? | the deployable stack | C28 |
| ~~Q11~~ | ~~Training VRAM at 16 GB~~ | — | **RESOLVED 2026-08-06: 7.35 GB peak. Fits.** |

Two source-integrity items to settle before anything goes in a manuscript:
- The Wairagkar rolling-past-10 s quote is from the **voice** paper, not Card b2txt verbatim.
  Confirm the exact b2txt feature-extraction line.
- The B2T'25 7th-place figures (19 GB 4-gram, −3.76 confidence gate) come from a Medium post.
  Verify against a primary source or label them **[U]** in print.
