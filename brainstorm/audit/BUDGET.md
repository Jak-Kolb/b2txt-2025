> Historical evidence, not current instructions. Read ../../RESEARCH_ASSESSMENT.md for corrections. Links to removed proposal files are archival references recoverable from Git or .git/context-cleanup/2026-09-18/.

# BUDGET.md — Phase C latency and compute accounting

Confidence tags: **[M]** measured here, **[M-prior]** measured previously and re-verified from
the stored artifact, **[D]** derived from measured quantities, **[E]** estimated, **[U]** unverified.

Measurements below come from stored artifacts (`results/*/checkpoint/val_metrics.pkl`, full
val split n=1426, and `metrics.json`, n=1423) plus CPU micro-benchmarks of the released
`causal_la0` checkpoint on this machine (M3 Pro, 6 threads, fp32). **No training was run and no
existing pipeline file was modified.** One new additive file was created:
`model_training/benchmark/stability.py`.

---

## 0. The one-paragraph version

The acoustic model is not the problem and never was. Emission latency is **60 ms worst case,
30 ms mean**, leaving **~79 ms of the 140 ms budget unspent**, and the model occupies **1.3% of
each cadence window on GPU and 1.7% on CPU** — a 10-member ensemble fits inside the budget *on
CPU alone*. Meanwhile the reference LM stack's finalization costs **620–830 ms [D]**, which is
**4.4–5.9× the entire end-to-end budget**, and the incremental LM path — which already exists and
works — has **never been timed**. Separately, the acoustic decoder is **provably flicker-free**
(0 revisions over 41,039 emitted tokens [M]), so every user-visible instability is LM-side.
The budget question is therefore entirely a language-model question.

---

## 1. C1 — Stage table

Per emitted frame unless noted. L_algo = waiting for future data; L_buf = buffering;
L_comp = computation. "Incremental" means the stage can consume frames one at a time and emit a
partial result.

| # | Stage | file:line | L_algo | L_buf | L_comp p50 / p95 | Memory | Incremental? |
|---|---|---|---|---|---|---|---|
| 1 | Acquisition → 20 ms binning | upstream | 0 | 20 ms | — | — | Yes |
| 2 | **Feature normalization** | upstream | **non-causal, block-scale** | — | — | — | **NO** — [AUDIT F1](AUDIT.md#f1) |
| 3 | Gaussian smoother (L=0, 5 taps) | `data_augmentations.py:25-41` | **0** | 0 | 0.0131 ms/bin CPU **[M]** | 5×512 floats | Yes |
| 4 | Day affine + softsign | `rnn_model.py:95-99` | 0 | 0 | (in row 3 measurement) | 1.05 MB | Yes |
| 5 | Patching (unfold, right-edge) | `rnn_model.py:106-119` | 0 | **0–60 ms, mean 30** | ~0 | 14×512 floats | Yes |
| 6 | GRU 5×768 + head | `rnn_model.py:126-129` | 0 | 0 | **0.320 / 0.341 ms GPU [M-prior]**; **1.348 / 1.438 ms CPU [M]** | 131 MB fp32 / 65.5 MB fp16 | Yes |
| 7 | Greedy CTC collapse | `common.py:257-265` | 0 | 0 | ~0 | O(1) | Yes |
| 8 | Redis transport (per chunk) | `evaluate_model_helpers.py:206-229` | 0 | 0 | ≥1 ms round trip **[D]** | — | Yes |
| 9 | **CTC-WFST beam search** | `ctc_wfst_beam_search.cc:70-121` | 0 | 0 | **UNMEASURED [U]** | 13.4 MB (1-gram) / **19 GB (4-gram, built)** / ~60 GB (3-gram, shipped) | **Yes** |
| 10 | Partial best path | `ctc_wfst_beam_search.cc:113` | 0 | 0 | **UNMEASURED [U]** | — | Yes |
| 11 | `FinishDecoding` / n-best lattice | `brain_speech_decoder.cc:42-45` | end-of-utt | — | small **[E]** | — | End-of-utterance |
| 12 | **`Rescore()` unpruned G** | `brain_speech_decoder.cc:61-101` | **end-of-utt** | — | part of 620–830 ms | + unpruned LM | **NO — BATCH** |
| 13 | **`augment_nbest()`** O(n²) | `language-model-standalone.py:327-411` | **end-of-utt** | — | part of 620–830 ms | — | **NO — BATCH** |
| 14 | **OPT-6.7B rescoring** | `language-model-standalone.py:620-631` | **end-of-utt** | — | part of 620–830 ms | **12.4 GB VRAM** | **NO — BATCH** |
| 15 | Endpointing | `ctc_endpoint.{h,cc}` present, unused | — | — | — | — | not wired up |

### The only in-repo LM timing number, and what it implies

`evaluate_model.py:215`, from the upstream authors:

> "note: this takes ~15-20 minutes to run on the entire test split with the 5-gram LM + OPT
> rescoring (RTX 4090)"

The test split is **1,450 trials [M]**. So **[D]**:

```
900–1200 s / 1450 trials = 0.62–0.83 s per trial for the full LM stack
```

Against a 140 ms end-to-end budget that is **4.4–5.9× over**, and it is incurred entirely at
end-of-utterance. This is the wall. It is also the first time this number has been written down
in this project.

**What it does not tell us:** how that 620–830 ms splits between rows 12, 13 and 14, and — the
number that actually matters — what row 9 costs *per frame* when driven incrementally. Row 9 is
the only stage that could plausibly live inside the budget, and it has never been timed. **This is
the single highest-value missing measurement in the project** and it needs no retrain, only a
built `lm_decoder` and a wall clock.

For scale, the only external anchor found in Phase B: **17 ± 11 ms per trial** for an offline
3-gram beam search at width 18 ([LITERATURE B3.4](LITERATURE.md)). This repo runs a far wider
search (`beam=17.0`, `max_active=7000`, `nbest=100`, hard-coded at
`language-model-standalone.py:488-496`), so treat 17 ms as a floor, not an estimate.

---

## 2. C2 — Three latencies, separated

The project has been collapsing these. They differ by an order of magnitude.

### Emission latency — neural event → first appearance of corresponding text

| Term | Value | Basis |
|---|---|---|
| L_algo (smoother lookahead, L=0) | **0 ms** | `data_augmentations.py:27` keeps taps `[-4, 0]` **[M-prior]** |
| L_buf (patch geometry) | **0–60 ms, mean 30 ms** | frame f emitted at bin 4f+13; wait cycles 0,3,2,1 bins **[D]** |
| L_comp (acoustic, achievable) | **0.34 ms GPU / 1.44 ms CPU** (p95) | **[M]** |
| L_comp (acoustic, as implemented) | 1.06 ms per window | reference streaming impl **[D]** |
| **Acoustic subtotal** | **≈ 60 ms worst case, 31 ms mean** | |
| Incremental LM (row 9+10) | **UNMEASURED** | **[U]** |
| **Budget remaining for the LM** | **≈ 79 ms worst case** | 140 − 61 |

Note the smoother's *centroid* sits 1.224 bins (24.5 ms) in the past at L=0 **[M-prior]**. That
is an information-recency cost, **not** a waiting latency — no future sample is required — so
L_algo = 0 stands. It is recorded separately because it turns out to explain the emission-timing
difference between the two trained models (§3).

For `causal_la4` the same arithmetic gives **80 + 60 + 1 = 141 ms**, exactly at the ceiling — a
cleaner reason to reject it than the group-delay argument previously used.

### Finalization latency — neural event → text stops changing

- **Acoustic level: identical to emission latency.** Measured: **0 prefix violations over 41,039
  emitted tokens** across the full val split **[M]**. The greedy streaming decoder is append-only,
  so a token, once emitted, never changes. Time-to-final = 0.
- **LM level: 620–830 ms [D]** for the reference stack, because rows 12–14 are batch. The
  incremental-only path's finalization latency is **[U]**.

**This gap is the finding.** Emission is 60 ms; finalization is 620–830 ms. A user sees text
quickly and then watches it rewrite itself ~700 ms later. Reporting a single "latency" number for
this system is not meaningful, and no brain-to-text paper currently separates the two.

### Cold-start latency — one-time patch fill

**280 ms [D]** (14 bins × 20 ms) before the first frame can be emitted.

**Measured: this is a non-issue.** The first non-blank token appears at a median of **3,380 ms**
into the trial (p05 **1,860 ms**) **[M]** — the participant does not begin speaking until well
after the go cue. The 280 ms fill is entirely absorbed by the pre-speech period. Reducing
`patch_size` to cut cold start (a candidate the brief anticipated) targets a cost that does not
exist in this task. There is also a median **1,680 ms** of trailing silence after the last token
**[M]**, which is what an endpointer would have to work with.

---

## 3. Emission-delay measurement (reviewer prior S6) — resolved

Implemented in `benchmark/stability.py::emission_delay`. Paired design: for every val trial where
`causal_la0` and `causal_la4` produce the **identical** token sequence, compare the frame index at
which each token was emitted. Restricting to matching sequences makes the comparison paired and
removes any confound from the two models decoding differently.

```
512 / 1426 trials matched, 13,030 paired tokens

la0 emission frame − la4 emission frame:
  mean            +0.3640 frames  = +29.12 ms
  median          +0.0 frames        sd 0.895
  same frame       43.2%             within 1 frame  89.7%

PREDICTED from the smoother's group delay alone:  +0.306 frames = +24.5 ms
EXCESS learned emission delay:                    +0.058 frames =  +4.64 ms
```

**The causal model's emission timing is explained almost entirely by signal processing.** The
learned component is **4.6 ms — 6% of one frame.**

**Consequence: the entire emission-latency literature is inapplicable here.** Delay-penalized CTC,
Bayes-Risk CTC, Peak-First CTC, TrimTail and Align-With-Purpose all exist to remove learned
emission delay. This model has ~none to remove. Combined with the three structural arguments in
[LITERATURE §B4](LITERATURE.md), that is now a measurement, not a prediction. **S6 is confirmed and
~6 candidate runs are freed.**

---

## 4. C3 — Word-level stability metric

### Definition

For a decoder that emits a partial hypothesis `H_t` after each frame `t`, for each position `w`
in the final transcript:

| Metric | Definition |
|---|---|
| `revisions(w)` | number of times the token at position `w` changed after `w` was first occupied |
| `ttf(w)` | **time-to-final**: (frame `w` reached its final value) − (frame `w` was first occupied), in ms |
| `ttfe(w)` | **time-to-first-emission**: frame `w` was first occupied, relative to utterance start |
| `flicker_rate` | total revisions / total emitted words |
| `unstable_frac` | fraction of words revised ≥ 1 time |
| `ttf_p50`, `ttf_p95` | the honest "when can the user trust this word" latency |

`ttf_p95` is the number that belongs next to WER in any streaming brain-to-text paper. Frame
index converts to wall-clock via `frame_to_ms(f) = (4f + 13) × 20 ms`.

### Implementation

`model_training/benchmark/stability.py` (new file, nothing existing modified). It provides:

- `trace_partials()` / `stability_metrics()` — the full word-level metric, for any decoder that
  emits partial hypotheses. Ready to run against the WFST partial stream
  (`remote_lm_output_partial`, `language-model-standalone.py:785`) once `lm_decoder` is built.
- `verify_append_only()`, `posterior_stats()`, `emission_delay()`, `trial_geometry()` — the
  acoustic-level analyses, which run today from saved logits with no GPU and no LM.

```
python model_training/benchmark/stability.py \
  --val_metrics results/causal_la0/checkpoint/val_metrics.pkl \
  --compare_to  results/causal_la4/checkpoint/val_metrics.pkl \
  --expected_shift_frames 0.306
```

### First measurement, `causal_la0`, full val split

```
trials: 1426   frames: 324,625
append-only: 0 prefix violations over 41,039 tokens  ->  acoustic revisions/word = 0
blank argmax: 75.36%
entropy median 0.0000 / max possible 3.714
emission shift vs la4: +0.3640 frames (+29.12 ms); excess over predicted: +4.64 ms
```

**Result: the acoustic decoder is provably flicker-free.** `flicker_rate = 0`,
`unstable_frac = 0`, `ttf = 0` for every one of 41,039 tokens. This is not a lucky measurement —
the online collapse rule at `common.py:257-265` only ever appends, so the partial hypothesis is
monotone in the prefix order by construction. The empirical check confirms the argument holds
over the full split.

**Therefore 100% of user-visible flicker originates in the WFST beam search**, where
`GetBestPath(use_final=false)` can return a different best path as later frames arrive. The
word-level metric is meaningful only against the LM, and cannot be measured until `lm_decoder` is
built (**[U]**, and the top item in [OPEN_QUESTIONS](OPEN_QUESTIONS.md)).

**The brief's hypothesis "if the PER frontier is flat but the stability frontier is not, that is
the paper" survives, in modified form.** The stability frontier cannot be flat *for the reason the
brief imagined* (acoustic emission delay), because that delay is 4.6 ms. But it is very likely
non-flat as a function of **how much LM machinery you are willing to run incrementally**, which is
a better and more general axis.

### Posterior structure — an unexpected result

| Statistic | All frames | Non-blank frames (24.6%) |
|---|---|---|
| entropy median | **0.0000** | 0.0004 |
| entropy mean | 0.0329 | 0.0990 |
| entropy p95 | 0.2472 | 0.6466 |
| top-1 margin median | 1.0000 | 0.9999 |
| top-1 margin p05 | 0.8797 | 0.4116 |
| frames with entropy > 0.5 | 2.90% | 8.75% |
| frames with entropy > 1.0 | 0.19% | 0.69% |

Maximum possible entropy is ln(41) = 3.714. **The model is extraordinarily over-confident** —
mean entropy is **0.9% of maximum**, and the median frame posterior is numerically one-hot.

Two consequences, both actionable:

1. **Calibration is an untouched lever with a clear mechanism.** A near-one-hot acoustic posterior
   gives the beam search almost no acoustic alternatives to consider; the n-best list is then
   shaped almost entirely by the LM, and correcting an acoustic error requires the LM to overcome
   a ~1.0 posterior. Training uses no label smoothing, no entropy regularization, and no
   temperature (`rnn_trainer.py:242`). Temperature scaling is **free** (a post-hoc scalar, tuned on
   val, no retrain) and is the obvious first move. This candidate fell out of measurement rather
   than from the literature.
2. **It constrains S5 (adaptive-compute decoding).** Gating on *acoustic* entropy would fire on
   only 2.9% of frames — attractively cheap, but it also means acoustic entropy carries little
   signal, because the model is confidently wrong rather than uncertain. The 7th-place B2T'25
   solution gated on **n-gram confidence** (threshold −3.76), not acoustic entropy
   ([LITERATURE B1.2](LITERATURE.md)). That is the better signal, and S5 should adopt it.

---

## 5. C4 — Headroom

### Measured, per 80 ms cadence window

Each window must process 4 new bins and emit 1 patch.

| Device | 1 patch forward (p50 / p95) | 4 bins (FIR + day layer) | Window total | **Duty cycle** | Forwards available |
|---|---|---|---|---|---|
| GPU, as implemented | 0.320 / 0.341 ms **[M-prior]** | ~0.74 ms **[D]** | **1.06 ms** | **1.3%** | ~235 |
| GPU, vectorized | 0.320 / 0.341 ms | ~0.05 ms **[E]** | **≈0.37 ms** | **0.5%** | ~235 |
| **CPU (M3 Pro, 6 thr, fp32)** | **1.348 / 1.438 ms [M]** | **0.052 ms [M]** | **1.40 ms** | **1.7%** | **~56** |
| Laptop GPU (RTX 4060-class) | 0.35–0.45 ms **[E]** | ~0.05 ms | ≈0.5 ms **[E]** | ≈0.6% **[E]** | ~160 **[E]** |

The GPU in `metrics.json` is recorded only as `device: cuda` **[U]** — see
[AUDIT §11](AUDIT.md). The CPU column is measured here and is the safe floor.

Extrapolation across devices is by **kernel-launch and memory latency, not FLOPs**: the workload
runs at 0.0024% arithmetic duty cycle on a 3090 and the measured GPU forward is **175× its
arithmetic time** ([AUDIT §A3](AUDIT.md)). Device scaling is therefore much flatter than
FLOPs would suggest — which is why the CPU is only ~4× slower than the GPU despite ~100× less
peak throughput.

### Batched ensemble scaling — measured on CPU

The decisive number for reviewer prior S1. Members are run as a batch dimension, not sequentially:

| Batch (ensemble size N) | p50 latency | Per-member | Duty cycle of 80 ms |
|---|---|---|---|
| 1 | 1.348 ms | 1.348 ms | 1.7% |
| 4 | 7.444 ms | 1.861 ms | 9.3% |
| 8 | 7.351 ms | 0.919 ms | 9.2% |
| **10** | **7.046 ms** | **0.705 ms** | **8.8%** |
| 16 | 6.822 ms | 0.426 ms | 8.5% |

Cost is **flat from N=4 to N=16** — the batch dimension is free because the workload is
launch- and memory-bound, not arithmetic-bound. (The jump between N=1 and N=2 is a thread/kernel
selection discontinuity in PyTorch's CPU GRU path, not a scaling law.)

**A 10-member ensemble consumes 8.8% of the cadence window on a laptop CPU, with zero added
algorithmic latency.** On GPU it will be cheaper still. Memory: **655 MB fp16** for 10
inference-resident members ([AUDIT §A3](AUDIT.md)) against 24 GB. **S1's compute and memory
feasibility is settled — comfortably.** What remains open for S1 is entirely the *accuracy*
question ([LITERATURE B3.1](LITERATURE.md): the published gain was measured at a 33.7% baseline,
not 2.7%).

### The reference streaming implementation wastes ~70% of what it measures

Cross-check confirming [AUDIT F5](AUDIT.md#f5): the reference per-bin path costs **~0.185 ms on
GPU [D]**, while a vectorized equivalent costs **0.0131 ms on CPU [M]** — the GPU implementation
is **14× slower than CPU** doing identical arithmetic, because `_compute_smoothed`
(`streaming_infer.py:194-199`) launches one GPU op per kernel tap and `_raw_for_index`
(`:201-208`) linearly scans a ring buffer for each. Fixing this is free (no retrain) and should
precede any published latency claim.

### Blank skipping — measured, and it is large

`ctc_blank_skip_threshold` is implemented (`ctc_wfst_beam_search.cc:79-84`) and **disabled by
default**. Measured on `causal_la0`, full val split, 324,625 frames **[M]**:

| threshold θ | frames with p(blank) > θ | ⇒ LM frames skipped |
|---|---|---|
| 0.9 | **74.28%** | 74% |
| 0.99 | **73.08%** | 73% |
| **0.999** | **71.83%** | **72%** |

Blank argmax fraction: **75.36%**. Median blank posterior: **1.0000**.

**This vindicates the brief's "~70% of frames are skippable" claim**, which
[LITERATURE §B5](LITERATURE.md) initially doubted on the grounds that Blank Collapse measures only
43–46% removable in acoustic ASR. That doubt was wrong: acoustic ASR runs at 10–40 ms frames,
whereas this system emits 80 ms patches during a task with long silences, so blank frames
dominate far more heavily. At θ = 0.999 the skipped frames carry p(blank) ≥ 0.999, so the accuracy
risk is close to nil — Blank Collapse reports WER 1.783 → 1.781 at θ = 0.99, i.e. unchanged.

**Expected effect: ~72% fewer WFST advance steps, for a config change and no retrain.** Whether
that matters depends on row 9's unmeasured cost — but it is the cheapest available lever on the
one stage that is actually contended.

---

## 6. Where the budget actually stands

```
140 ms end-to-end budget
├─  0 ms   L_algo         smoother lookahead at L=0                      [M]
├─ 60 ms   L_buf          patch geometry, worst case (mean 30)           [D]
├─  1 ms   L_comp         acoustic, as implemented (0.34 ms achievable)  [M]
│
├─ ≈79 ms  UNALLOCATED    ← the entire remaining budget
│          must contain: incremental WFST beam search + partial extraction
│          cost: UNMEASURED                                              [U]
│
└─ EXCLUDED, currently 620–830 ms — 4.4-5.9x the whole budget:           [D]
     Rescore() unpruned G  |  augment_nbest() O(n^2)  |  OPT-6.7B rescoring
```

Three conclusions:

1. **Every accuracy mechanism that fits in the remaining 79 ms is worth more than any acoustic
   improvement**, because the acoustic side has already spent only 61 ms of 140 and is provably
   stable.
2. **The project's central empirical question is now: how much of the 620–830 ms batch stack's
   accuracy can be recovered incrementally inside 79 ms?** That is a well-posed, publishable
   question, and answering it requires no retraining — only a built `lm_decoder` and the
   measurement harness that now exists.
3. **The compute headroom is real but it is in the wrong place.** 98.5% of every cadence window is
   idle, and a 10-member ensemble fits on a laptop CPU — but spending that headroom on more
   acoustic models improves the stage that is already 40× under budget, while the stage that is
   4–6× over budget goes untouched. Any recommendation must justify itself against that asymmetry.
