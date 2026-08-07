# RECOMMENDATIONS.md — Phase E

Four recommendations, ranked. R1 and R2 are the co-headline contributions; R3 is a novel
mechanism; R4 is the largest accuracy gain per retrain and is explicitly **not** a paper on its
own. One recommendation is deliberately declined at the end — it is the brief's own highest prior.

Every number is tagged **[M]** measured, **[D]** derived, **[E]** estimated, **[U]** unverified.

---

## Framing correction that drives all four

The brief's objective is "minimize test WER subject to ~140 ms." Two facts from Phases A–C change
what that means in practice:

1. **Test WER is not measurable.** `data_test.hdf5` contains only `input_features` — no labels of
   any kind ([AUDIT F7](AUDIT.md#f7)) — and the leaderboard is closed. Every recommendation below
   is stated against **val PER** and **held-out-session WER** under the protocol in
   [AUDIT §A8](AUDIT.md), never against "test WER."
2. **The acoustic model is 40× under budget; the LM is 4–6× over it.** Emission latency is 60 ms
   worst case **[D]**; the reference LM stack's finalization is 620–830 ms **[D]**. Any
   recommendation that spends effort on the acoustic side must justify itself against that
   asymmetry, and three of the four below do so explicitly.

---

# R1 — Build the real-time-feasible frontier by relocating the contribution to the decoder

### Claim

A fully incremental decoder — n-gram WFST beam search with blank skipping, partial emission, a
32 GB-feasible 4-gram, and a causal neural LM in shallow fusion — reaches **3.2–4.0% WER
[E]** (point estimate **3.6%**) at **p95 emission latency < 140 ms**, recovering **60–75% [E]** of
the gap between the incremental baseline (~5.5% **[E]**) and the unconstrained full-stack
reference (2.66% **[U]**), **with zero training runs**.

### Why it wins under the constraint — latency arithmetic

```
140 ms budget
  L_algo   0.0 ms   smoother lookahead at L=0                            [M]
  L_buf   60.0 ms   patch geometry, worst case (mean 30)                 [D]
  L_comp   0.34 ms  acoustic forward, p95, vectorized                    [M]
  ─────────────────
  60.3 ms spent      →  79.7 ms available for the LM
```

Into that 79.7 ms the recommendation places: incremental WFST advance (**cost unmeasured [U]**,
but reduced by ~72% because **71.83% of frames carry p(blank) > 0.999 [M]**), partial best-path
extraction, and shallow-fusion scoring at an estimated +2–10 ms per emitted word **[E]**. What it
*removes* is the entire 620–830 ms batch tail: `Rescore()`, `augment_nbest()` and OPT-6.7B.

The three deleted stages are what currently buy the accuracy. R1's whole content is: **how much of
that can be bought back incrementally?** That question has never been asked for a brain-to-text
system, and answering it needs no GPU training at all.

### Implementation spec

**Prerequisite.** Build `lm_decoder`: `./setup_lm.sh` (creates conda env `b2txt25_lm`, requires
CMake ≥ 3.14, gcc ≥ 10.1). Not built on the current machine.

**New file `model_training/benchmark/stream_lm.py`:**
- `class IncrementalLMDecoder` wrapping `lm_decoder.BrainSpeechDecoder`, exposing
  `push_frame(logits_row) -> partial_hypothesis` and `finalize() -> final_hypothesis`.
- Feeds `lm_decoder.DecodeNumpy(decoder, logits[t:t+1], zeros_like, log(blank_penalty))` **one
  frame at a time**, then reads `decoder.result()[0].sentence`. This is the path
  `language-model-standalone.py:769-785` already exercises, driven per-frame instead of per-trial.
- Applies `rearrange_speech_logits_pt` (`evaluate_model_helpers.py:79-83`) **once**, before the
  loop.
- Records per-frame: wall time of `DecodeNumpy`, wall time of `result()`, the partial hypothesis,
  and the number of active tokens.
- Source of logits: `results/causal_la0/checkpoint/val_metrics.pkl` via
  `benchmark/stability.py::load_val_logits` — **the acoustic model never re-runs.**

**Wire into the existing stability harness:** pass the per-frame partial list to
`benchmark/stability.py::trace_partials` then `stability_metrics`. Those functions already exist
and are already tested against the acoustic path.

**Configuration changes, in `language-model-standalone.py`:**
- `--ctc_blank_skip_threshold 0.999` (currently `1.0`, i.e. disabled, `:803`).
- Do **not** pass `--rescore` and do **not** pass `--do_opt`.
- Set `--nbest 1` for the pure-incremental condition (nbest > 1 triggers `augment_nbest`).
- **Note a live bug**: `build_lm_decoder` is called at `:486-496` with **literal**
  `max_active=7000, min_active=200, beam=17., lattice_beam=8.` — the CLI flags for those four are
  dead at startup. To sweep them, either edit those four lines to pass `args.*`, or drive them
  through the `remote_lm_update_params` stream (`:708-718`).

**Sweep grid** (all against cached logits, no acoustic re-run):
| Parameter | Values |
|---|---|
| `acoustic_scale` | 0.20, 0.25, 0.30, 0.325, 0.40, 0.50 |
| `blank_penalty` | 1, 3, 9, 30, 90 |
| `beam` | 12, 15, 17, 20 |
| `max_active` | 2000, 4000, 7000 |
| `blank_skip_thresh` | 1.0 (off), 0.999, 0.99, 0.9 |

Sweep coordinate-wise, not full grid: `acoustic_scale` × `blank_penalty` first (they interact
strongly), then `beam`/`max_active` at the best point.

**LM artifact.** Build a pruned + quantized 4-gram KenLM → TLG, targeting ≤ 19 GB RAM, per the
7th-place B2T'25 recipe (80% Wiki/news + 20% conversational). Toolchain is in
`language_model/srilm-1.7.3/` and `language_model/tools/fst/`. **This is CPU work, not GPU.**

**Shallow fusion (second phase).** Add a causal LM (100–500 M params) scoring word hypotheses
incrementally, fused with weight `lambda_nlm` swept over {0.1, 0.2, 0.3, 0.5}.

### Pre-registered decision rule

> **H1 supported** if, on the held-out-session split, the fully incremental decoder achieves
> **WER ≤ 4.0%** with **p95 per-frame LM latency ≤ 79 ms** and **p95 word time-to-final ≤ 300 ms**,
> using no stage that requires the complete utterance.
>
> **H1a (blank skipping is free)** supported if `blank_skip_thresh = 0.999` changes WER by
> **≤ 0.10 points** while reducing measured WFST advance steps by **≥ 60%**.
>
> **H1b (the frontier is non-degenerate)** supported if sweeping the decoder configuration traces
> a curve with **≥ 1.0 WER points** of spread across configurations whose p95 latency spans
> 40–140 ms. If the curve is flat, R1's framing fails and the honest report is a second negative
> result.

### Compute plan

**Zero training runs.** ~12.3 GPU-hours total, all evaluation:

| Step | Isolates | Cost |
|---|---|---|
| E1 Vectorize streaming smoother/day layer | measurement validity | 0.2 h |
| E2 Build `lm_decoder`, drive incrementally, measure per-frame cost | **the constrained baseline** | 2.0 h |
| E3 Blank skipping at θ ∈ {0.999, 0.99, 0.9} | blank skipping alone | 0.3 h |
| E4 `acoustic_scale` × `blank_penalty` × `beam` sweep | decode hyperparameters | 3.0 h |
| E5 Build + evaluate 4-gram ≤ 19 GB | LM order | 1.0 h GPU + CPU build |
| E6 Endpointing policy | finalization latency | 0.3 h |
| E7 Causal neural LM shallow fusion | neural LM | 1.5 h |
| E8 Adaptive-compute gate on LM confidence | expected-latency curve | 2.0 h |
| E9 TTA (DietCORP-style, per-trial) | test-time adaptation | 1.5 h |

E2 gates everything. E4 must be re-run after E5 and after E7 — the optimal `acoustic_scale`
moves whenever the LM changes.

### Failure mode and fallback

**Primary failure:** the incremental WFST turns out to cost > 79 ms per frame at any useful beam
width, so nothing fits. **Fallback:** reduce the search (lower `max_active`, tighter `beam`,
θ = 0.9 blank skipping) and report the frontier that *does* fit, even if its WER is poor — a
measured infeasibility result is still the first published latency accounting for this system.

**Secondary failure:** incremental decoding proves nearly as good as the batch stack, making the
constraint uninteresting. That would be a strong positive result, reported as such.

**Tertiary:** the 4-gram cannot be pruned under 32 GB without destroying it. Fall back to a
3-gram and report the RAM/WER frontier, which is itself unpublished.

### Novelty assessment — honest

**Published already:** incremental WFST decoding (WeNet 2.0, 2022), blank skipping (Blank Collapse,
2022, ~33% WFST decode-time reduction), shallow fusion, endpointing. None of the *mechanisms* are new.

**New here:** (a) the first end-to-end latency accounting for a brain-to-text decoder that includes
the LM — [LITERATURE §B5](LITERATURE.md) found **no** published per-frame incremental LM cost for
any brain-to-text system, and the only anchor anywhere is 17 ± 11 ms/trial *offline*; (b) the
accuracy-vs-latency frontier under a hard real-time cap, which the closed competition never
measured because latency was not scored; (c) the word-level stability metric, which no
brain-to-text paper reports.

**Does the new part carry a paper? Yes** — but as a *systems and measurement* paper, not an
algorithms paper. The contribution is "here is what real-time actually costs, measured," against a
literature where every top entry is unconstrained and none report latency or memory. That is a
publishable and genuinely useful gap. It will not be accepted at a venue expecting a new
algorithm.

---

# R2 — Measure the true cost of causality by fixing the non-causal normalization

### Claim

The released features are whole-block z-scored, so every "real-time" number this project has
produced rests on inputs containing up to ~19 minutes of future information. Re-normalizing
causally (rolling 10 s, Welford) and retraining will **increase** val PER by **0.3–1.5 points
[E]** (point estimate **+0.8**), and that increase is **the true price of end-to-end causality on
T15 — a quantity nobody has published for any participant.**

### Why it wins under the constraint

It is not an accuracy gain; it is the measurement that makes every other latency claim in the
project defensible. [AUDIT F1](AUDIT.md#f1) establishes block normalization by a 1/√W scaling test
whose prediction matches observation within 1–26% across four sessions spanning two years:

```
t15.2023.08.13 blk1  T=40727 ( 815s): predicted 0.00800 | observed 0.00853 | ratio 1.07
t15.2023.11.17 blk1  T=57061 (1141s): predicted 0.00721 | observed 0.00727 | ratio 1.01
t15.2024.03.08 blk1  T=18936 ( 379s): predicted 0.01251 | observed 0.01574 | ratio 1.26
t15.2025.03.30 blk3  T=12013 ( 240s): predicted 0.01523 | observed 0.01636 | ratio 1.07
```

**The fix is exact, which is the non-obvious part.** Block z-scoring gives
`x_norm = (x_raw − μ_blk)/σ_blk` with μ_blk, σ_blk constant within a block. Applying a causal
rolling normalizer *on top* yields

```
(x_norm − μ_roll(x_norm)) / σ_roll(x_norm)
  = [ (x_raw − μ_blk)/σ_blk − (μ_roll,raw − μ_blk)/σ_blk ] / [ σ_roll,raw / σ_blk ]
  = (x_raw − μ_roll,raw) / σ_roll,raw
```

**The block constants cancel identically.** Rolling-normalizing the released data recovers exactly
the causally-normalized raw features — no access to raw data required. The only loss is at the
+10 clipping rail, which affects **0.004% of samples [M]**.

Latency cost: Welford is O(1) per feature per bin. **L_algo 0, L_buf 0, L_comp +negligible.**

### Implementation spec

**New file `model_training/causal_normalize.py`:**

```
def rolling_normalize(features: np.ndarray, half_life_bins: int = 500,
                      eps: float = 1e-5, warmup_bins: int = 50) -> np.ndarray
```
- EWMA mean and variance with `alpha = 1 - exp(-ln2 / half_life_bins)`; `half_life_bins = 500`
  = 10 s at 20 ms, matching Wairagkar et al.'s "rolling means and s.d. from the past 10 s."
- Strictly causal: output at bin *t* uses statistics from bins `< t` only (update **after**
  emitting, not before).
- Initialize `(mean, var) = (0, 1)`. **State this honestly as a residual assumption**: it uses the
  fact that the data is z-scored at block scale. It is a far weaker leak than per-feature block
  statistics, and it matches what a deployed system does at block start using the previous
  block's statistics. Document it; do not hide it.
- During the first `warmup_bins`, blend toward the prior to avoid a variance explosion.

**Application point.** In `dataset.py::BrainToTextDataset.__getitem__`, immediately after
`input_features` is read at `:130`, before `pad_sequence`. Applying it per-trial (rather than
continuously across a block) resets the estimator at each trial boundary — accept this for v0 and
record it as a limitation; trials are ~18 s so the 10 s half-life warms within each trial.

**New config keys** under `dataset:` in `rnn_args.yaml`:
```yaml
  causal_normalize: true          # default false to preserve the existing baseline
  causal_norm_half_life_bins: 500
  causal_norm_warmup_bins: 50
```

**Also change** (required, not optional): `rnn_trainer.py:242`
`zero_infinity=False` → `zero_infinity=True`. Changing the input distribution can produce
degenerate trials, and `error_if_nonfinite=True` at `:554` turns one infinite loss into a hard
crash ([AUDIT F11](AUDIT.md)).

**Streaming path.** `benchmark/common.py` and `benchmark/streaming_infer.py` need the same
normalizer applied before `smooth_features` / `_append_raw`, or the equivalence gate will fail.

### Pre-registered decision rule

> **H2 (the leak is real and costly)** supported if causal re-normalization raises val PER by
> **≥ 0.20 points** on ≥ 2 of 3 seeds. That threshold is ~5× the measured same-config
> reproducibility floor of 0.041 points ([AUDIT F9](AUDIT.md#f9)).
>
> **H2-null (the leak is real but harmless)** supported if the change is **< 0.20 points** on
> ≥ 2 of 3 seeds — which would be a *stronger* and more surprising result, licensing every prior
> number in the project as real-time-valid.
>
> **Gate on correctness before interpreting either:** the rolling normalizer must reproduce the
> released features to within 1e-4 when fed constant (block) statistics instead of rolling ones.
> If that identity check fails, the transform has been misidentified and the run is void.

### Compute plan

| Run | Isolates | Cost |
|---|---|---|
| N1 | causal normalization ON vs OFF, seed 10, everything else identical to `causal_la0` | 6.4 h |
| N2 | seed 11, same config | 6.4 h |
| N3 (contingent) | half-life 250 vs 500 vs 1000 bins, if H2 is supported | 6.4 h |

Must be run **alone**. It changes the input distribution, so every tuned hyperparameter
(`white_noise_std`, `smooth_kernel_std`, `lr_max`) is potentially invalidated — do not bundle.

### Failure mode and fallback

**Primary failure:** the identity check fails, meaning the upstream transform was not a pure block
z-score (e.g. clipping applied before normalization, or statistics computed over a superset
including unreleased inter-trial data). **Fallback:** report the 1/√W evidence as establishing
block-*scale* normalization without claiming the exact functional form, and frame the retrain as
"causal re-normalization" rather than "leak removal." The scientific point survives.

**Secondary:** PER degrades so much (> 3 points) that the causal model is no longer competitive.
That is still the answer to the question, and it is a more important result than a small one.

### Novelty assessment — honest

**Published already:** rolling causal normalization itself (Wairagkar et al., Nature 2025, on this
*same participant*); block z-scoring as the offline convention (multiple sources,
[LITERATURE B3.8](LITERATURE.md)). Neither the normalizer nor the observation that offline datasets
are block-normalized is new.

**New here:** (a) the demonstration that the *public B2T'25 benchmark data* is block-normalized and
that every result on it — including all competition entries — therefore inherits a non-causal
preprocessing step; (b) the exact-cancellation argument showing the fix is recoverable from the
released data; (c) the first measurement of what end-to-end causality actually costs on this
dataset.

**Does the new part carry a paper?** (a) and (c) together, **yes** — this is a benchmark-integrity
result with a quantified cost, of direct interest to everyone who has published on B2T'24/'25. It
is also the kind of finding that makes reviewers trust the rest of the paper. **This is the single
highest-novelty item in the entire catalog**, and it is worth more than any accuracy improvement
R4 could deliver.

---

# R3 — Phase-staggered replicas: sub-cadence emission from a single trained model

### Claim

Evaluating **one** trained model at four patch phase offsets {0, 20, 40, 60} ms and merging cuts
worst-case L_buf from **60 ms to 15 ms [D]** while acting as a test-time-augmentation ensemble
worth **−0.2 to −0.6 WER points [E]**, at **4× a compute cost that is already 1.7% of the
window** — and, per [AUDIT F10](AUDIT.md), probably with **no retraining at all**.

### Why it wins under the constraint

```
current   L_algo 0 + L_buf 0–60 (mean 30) + L_comp 0.34  =  60.3 ms worst case
staggered L_algo 0 + L_buf 0–15 (mean  7) + L_comp ~0.4  =  15.4 ms worst case   [D]
                                                     →  +45 ms returned to the LM
```

Measured compute for 4 phases as one batch: **7.44 ms at batch=4 on CPU [M]** (9.3% duty cycle),
and cost is flat from batch=4 to batch=16 because the workload is launch-bound. On GPU it is
~0.4 ms.

**Why no retrain is likely needed** — the argument that makes this cheap:
`random_cut = 3` (`rnn_args.yaml:67`) draws `cut ∈ {0,1,2}` and crops that many leading bins
before patching, so training **already** covers 3 of the 4 phases **[M]**. The smoother is an FIR
and therefore shift-equivariant; the day layer is per-bin and therefore phase-agnostic. Only the
patch embedding sees phase, and it has been trained under phase jitter.

### Implementation spec

**Step 1 — the deciding test (30 min, no training).** New file
`model_training/benchmark/phase_ensemble.py`:
- For `p in {0,1,2,3}`: run the existing `causal_la0` checkpoint on `raw_features[p:]`, producing
  frames whose right edges sit at bins `4f + 13 + p`.
- Compute greedy PER **per phase** on the val split.
- **Decision:** if max−min per-phase PER **≤ 0.5 points**, the phases are equivalent and one model
  serves all four. If not, go to Step 3.

**Step 2 — merge.** Two modes, both implemented and compared:
- **(a) Interleaved 20 ms grid** — emit each phase's frames on the common 20 ms timeline; the LM
  consumes a 50 Hz stream instead of 12.5 Hz. Delivers the latency win. Note this **4× the LM's
  frame rate**, which interacts directly with R1's budget — blank skipping at θ = 0.999 offsets it
  almost exactly (71.8% skipped **[M]**).
- **(b) Phase-averaged logits** — align the four streams by their right-edge bin and average
  overlapping frames. Delivers the ensemble win but not the latency win.
- Report both separately. They are different mechanisms and must not be conflated.

**Step 3 — contingency if phase-sensitive.** Set `random_cut: 4` in `rnn_args.yaml:67`
(`randint(0,4)` → `{0,1,2,3}`, full phase coverage) and retrain. **One run.**

**Streaming path.** `benchmark/streaming_infer.py::StreamingDecoder` needs a `phase` constructor
argument that skips the first `phase` bins before `_consume_smoothed_bin` begins buffering. The
equivalence gate must be extended to assert all four phase-decoders individually match their
offline counterparts.

### Pre-registered decision rule

> **H3a (phase equivalence)** supported if per-phase greedy val PER spans **≤ 0.5 points** across
> p ∈ {0,1,2,3} using the existing `causal_la0` checkpoint, with no retraining.
>
> **H3b (latency)** supported if the interleaved decoder's measured worst-case L_buf is
> **≤ 20 ms** and end-to-end p95 emission latency stays **< 140 ms** including the 4× LM frame rate.
>
> **H3c (accuracy)** supported if phase-averaged logits improve WER by **≥ 0.15 points** over the
> single-phase decoder on held-out sessions.
>
> H3a is the gate. If it fails, the recommendation costs one extra run before it can be retried.

### Compute plan

| Run | Isolates | Cost |
|---|---|---|
| P1 | per-phase PER on existing checkpoint — the deciding test | 0.5 h |
| P2 | interleaved vs averaged merge, latency + WER | 1.0 h |
| P3 (contingent on H3a failing) | `random_cut: 3 → 4` | 6.4 h |

Expected cost **1.5 h** if H3a passes, 7.9 h if not.

### Failure mode and fallback

**Primary failure:** H3a fails — the four phases disagree by more than 0.5 PER points, meaning the
patch embedding is phase-sensitive despite `random_cut`. **Fallback:** run P3 (`random_cut: 4`),
retest. If it still fails, the mechanism requires four separately-trained replicas, the cost rises
to 4 runs, and the score collapses — at which point drop it.

**Secondary:** the interleaved mode quadruples the LM frame rate and blows R1's budget. **Fallback:**
keep mode (b) for the ensemble benefit only, and report the latency result as achievable-but-
LM-bound, which is itself an informative finding about where the real constraint sits.

### Novelty assessment — honest

**Published already, and this is the threat:** the June 2026 deep-ensembles preprint
([LITERATURE B3.1](LITERATURE.md), Willett senior author) introduces "a computationally efficient
pseudoensembling approach based on test-time augmentation that improves decoding accuracy while
requiring only a single base decoder." That is the same family. Positioning S3 as "TTA ensembling
for BCI" would be scooped by two months.

**New here:** the *latency* half. Pseudoensembling buys accuracy; phase staggering buys accuracy
**and** converts unused compute directly into a 4× finer emission grid, cutting buffering latency
by 45 ms. Test-time augmentation over the *patch phase specifically* — a nuisance variable created
by the architecture, not by the data — is what makes the latency win possible, and no retrieved
source does this.

**Does the new part carry a paper? Not alone.** It is one mechanism with a modest accuracy gain
and a 45 ms latency win. It is a strong *section* of R1's paper — the piece that shows the
frontier can be moved, not just measured — and it should be presented that way. Claiming it as a
standalone contribution invites a direct comparison to the pseudoensembling preprint that it would
lose.

---

# R4 — Time masking, and the regularization block

### Claim

Adding time masking at ~50% of each trial reduces val PER from 10.04% to **9.2–9.5% [E]**
(point estimate **9.19%**, ΔPER **−0.85**) at **zero latency cost** — the largest single-change
accuracy gain available, and the closest published causal result attributes 20–26% relative WER
improvement to it.

### Why it wins under the constraint

Zero latency cost of any kind: **L_algo 0, L_buf 0, L_comp 0.** It is a training-time
augmentation that vanishes at inference. Under a hard latency cap, mechanisms that are free at
inference are strictly preferable to any that are not, and this is the strongest one available.

The evidence is unusually direct: arXiv:2507.02800 masks 53% of every trial and reports
15.25 → 12.17% WER (3-gram) and 11.12 → 8.18% (5-gram) against a *unidirectional GRU baseline*,
with a causal model that matched bidirectional SOTA. [AUDIT §A2](AUDIT.md) finds this repo has
**no time masking at all** — its only live augmentations are additive white noise and a constant
offset.

### Implementation spec

**`model_training/data_augmentations.py`** — add:

```
def time_mask(features: torch.Tensor, n_masks: int = 20, max_frac: float = 0.075,
              mask_value: torch.Tensor | float = 0.0,
              generator: torch.Generator | None = None) -> torch.Tensor
```
- `features` is `[B, T, C]`. For each of `n_masks` masks, sample start `S ~ U[0, T)` and duration
  `D ~ U[0, max_frac * T)`; set `features[:, S:S+D, :] = mask_value`. Masks may overlap; expected
  coverage at `n_masks=20, max_frac=0.075` is ~53%, matching the source.
- Sample masks **per trial**, not per batch (unlike `random_cut`, which uses one scalar for the
  whole batch at `rnn_trainer.py:469`).

**`model_training/rnn_trainer.py::transform_data`** — apply **after** `gauss_smooth`
(`:475-482`), so the mask is genuine information removal rather than something the smoother
partially fills back in. Guard with `if mode == 'train'`.

**New config keys** under `dataset.data_transforms:` in `rnn_args.yaml`:
```yaml
    time_mask_n: 20            # number of masks per trial
    time_mask_max_frac: 0.075  # max mask length as a fraction of trial length
    time_mask_learnable: false # v0 uses zeros; v1 uses a learned MASK vector
```

**v1 (learnable MASK token, second run only):** add
`self.mask_token = nn.Parameter(torch.zeros(1, 1, neural_dim))` to `GRUDecoder.__init__`
(`rnn_model.py:32-86`) and pass it into `transform_data`. The source paper uses a learnable token;
zeros is the cheaper first test and isolates whether the gain comes from masking at all.

**Required prerequisite:** set `zero_infinity=True` at `rnn_trainer.py:242`. Masking does not
shorten sequences, but combined with `random_cut` it can push a short trial's `adjusted_lens`
below `phone_seq_lens`, producing infinite loss → NaN gradient → hard crash via
`error_if_nonfinite=True` at `:554`.

**Sweep:** `time_mask_max_frac ∈ {0.05, 0.075, 0.10}` at fixed `n_masks=20`, i.e. ~35%/53%/65%
expected coverage.

### Pre-registered decision rule

> **H4 supported** if val PER **≤ 9.75%** on **≥ 2 of 3 seeds**, with streaming equivalence
> maintained at **< 1e-3**. The 9.75% threshold is a 0.29-point improvement — **7× the measured
> 0.041-point same-config reproducibility floor** ([AUDIT F9](AUDIT.md#f9)) — chosen so a positive
> result cannot be explained by run-to-run variance.
>
> **H4-strong** supported if val PER ≤ 9.50% on ≥ 2 of 3 seeds.
>
> **Reported alongside, not as the gate:** held-out-session WER. PER→WER transfer here is expected
> to be **sublinear**, because the LM already repairs many phoneme errors.

### Compute plan

| Run | Isolates | Cost |
|---|---|---|
| T1 | time masking at `max_frac=0.075`, zeros mask, seed 10 | 6.4 h |
| T2 | best `max_frac` from {0.05, 0.10}, seed 10 | 6.4 h |
| T3 | best config, seed 11 (for the 2-of-3 gate) | 6.4 h |
| T4 (contingent) | learnable MASK token vs zeros | 6.4 h |

**Must not be bundled** with `channel_masking`, `stochastic_depth` or the augmentation sweep —
they target the same error mass and their gains will not add.

### Failure mode and fallback

**Primary failure:** no PER improvement at any mask fraction. That would be a genuinely
informative negative — it would mean the 20–26% relative gain in arXiv:2507.02800 belongs to the
*transformer*, not the masking, which materially raises the value of the (much more expensive)
causal-transformer candidate. **Fallback:** promote `causal_transformer` above the cut line.

**Secondary:** training destabilizes (CTC loss spikes). **Fallback:** reduce `n_masks` to 10 and
confirm `zero_infinity=True` took effect.

### Novelty assessment — honest

**Published already:** essentially all of it. SpecAugment-style time masking is 2019; the specific
recipe, hyperparameters and the >50% coverage figure are arXiv:2507.02800's.

**New here:** only that it has not been demonstrated on T15 / B2T'25, or with a GRU, or in
combination with this patching scheme.

**Does the new part carry a paper? No — and I want to be unambiguous about that.** This is applying
a known technique to a new dataset. Its value is that it moves R1's frontier to a better operating
point at zero latency cost, and that it is the highest-value use of a training run in the entire
catalog. It belongs in the paper as a line in a table, not as a contribution.

---

# The recommendation I am declining to make

## Declined: deep ensembling at N=10 (reviewer prior S1)

The brief calls S1 "the highest-prior recommendation and the one most likely to survive scrutiny."
**I am not recommending it**, and the reasons are cumulative rather than any single objection.

**What survives.** The feasibility argument is correct and I verified it more strongly than the
brief did: measured **7.05 ms at batch=10 on CPU** — 8.8% of the cadence window, on a *laptop
CPU*, not a 3090 — with cost **flat from batch=4 to batch=16** because the workload is
launch-bound at 175× its arithmetic time. Memory is **655 MB fp16** against 24 GB. Logit averaging
preserves streaming equivalence provided the gate is extended to the ensemble. **Every compute
objection the brief anticipated is answered, favourably.**

**Why I still decline it, in order of weight:**

1. **The gain is measured in the wrong error regime, and the brief's framing does not acknowledge
   this.** The closed-loop result is 33.7% → 26.0% WER — a baseline **12× higher** than this
   project's. Ensemble gains compress as baseline error falls, and the source paper *explicitly
   studies that dependence* in a figure I could not retrieve ([LITERATURE B3.1](LITERATURE.md),
   bioRxiv blocks automated access). Recommending N=10 on the strength of a 23% relative gain
   measured at 33.7% baseline is an extrapolation across an order of magnitude that the source
   itself warns against.
2. **The near-one-hot posterior predicts unusually high member correlation.** Mean posterior
   entropy is **0.033 nats against a maximum of 3.714 — 0.9% of maximum [M]**, with a numerically
   one-hot median frame. Ensembles pay off where members disagree. A model this confident, trained
   on 45 sessions from one participant with identical architecture and data, will produce members
   that agree almost everywhere. The brief raises this concern ("does the gain hold when all
   members are causal and therefore more correlated") and it is the right concern — the
   measurement says it is likely to bite.
3. **The cost is 45% of the entire budget for a sublinear return.** Nine extra runs = 57.6 GPU-h
   of ~128. **N=3 scores 3× better than N=10** (0.0293 vs 0.0104) because ensemble gain is roughly
   logarithmic in N while cost is linear. The brief specified the worst point on its own curve.
4. **Zero novelty.** Ensembling is the single most-reported finding in both competition
   retrospectives — "used by all 3 top entrants" — and now has a dedicated 2026 closed-loop paper
   from the consortium that produced this dataset. A paper whose headline is "we also ensembled"
   has no contribution.
5. **It is the exact misallocation Phase C identified.** It spends 57.6 GPU-hours improving the
   stage that is **already 40× under budget**, while the stage that is **4–6× over budget** — the
   LM — goes untouched. R1 addresses that same asymmetry for **zero training runs**.

**What I recommend instead.** Keep `deep_ensemble_n3` below the cut line as a cheap diversity
probe (2 extra runs), and run it **only after** R4's regularization work, so members differ by
seed on a *better* base model. If N=3 fails to beat its best member by ≥ 0.15 WER points, that
falsifies the ensembling premise at this operating point cheaply — and *that* would be a finding
worth publishing, because it would be the first evidence that the field's most-cited technique
stops paying at low baseline error.
