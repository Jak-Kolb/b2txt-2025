# CANDIDATES.md — Phase D scored mechanism catalog

**40 candidates**, generated 2026-08-04 from `candidates.json` (same source of truth — regenerate both together, never edit one alone).

## Scoring

```
score = -(expected WER point delta) / max(GPU-hours, 0.1)
1 training run = 6.4 h    1 full-val LM evaluation = 0.3 h
```
Positive score = WER points recovered per GPU-hour. Sign convention: Δ negative = improvement.

### What everything is measured against

| Operating point | Value | Status |
|---|---|---|
| val PER, `causal_la0` | **10.04%** | measured |
| val PER, `causal_la4` | 10.21% | measured |
| **Constrained WER** (incremental n-gram only, no rescore/OPT) | **~5.5%** (range 4.0–8.0) | **UNMEASURED - established by candidate 'incremental_baseline'** |
| Unconstrained reference (full stack, violates budget) | 2.66% | UNVERIFIED - only a 1-gram LM is present locally |

The constrained baseline's basis: scaling the 12.17 -> 5.68 offline/LLM ratio in arXiv:2507.02800 onto this project's ~2.7% unconstrained full-stack figure.

> **The constrained baseline does not exist yet.** Every WER delta in this catalog is a prediction against a number nobody has measured. That is the single largest source of error in this whole catalog, and it is why `incremental_baseline` gates everything.

---

## Master ranking

| # | Candidate | Family | ΔPER | ΔWER | GPU-h | Risk | Score | Tags |
|---|---|---|---|---|---|---|---|---|
| 1 | **Build a pruned/quantized 4-gram LM that fits 32 GB** | LM & decoding | +0.00 | -2.25 | 1.0 | Medium | **2.2500** | high-value |
| 2 | **Adaptive-compute decoding gated on n-gram confidence (S5)** | LM & decoding | +0.00 | -1.40 | 2.0 | High | **0.7000** | high-value contested |
| 3 | **Small causal neural LM via shallow fusion** | LM & decoding | +0.00 | -1.00 | 1.5 | Medium | **0.6667** | high-value |
| 4 | **Incremental LLM rescoring over a rolling context** | LM & decoding | +0.00 | -1.75 | 3.0 | High | **0.5833** | contested |
| 5 | **Lightweight test-time adaptation (DietCORP-style)** | Test-time | -0.50 | -0.65 | 1.5 | Medium | **0.4333** | high-value no-retrain |
| 6 | **Online day-layer adaptation** | Test-time | -0.30 | -0.30 | 1.0 | Medium | **0.3000** | scope-check |
| 7 | **Beam / acoustic-scale / blank-penalty sweep on cached logits** | LM & decoding | +0.00 | -0.65 | 3.0 | Low | **0.2167** | free first |
| 8 | **Phase-staggered replicas for sub-cadence resolution (S3)** | Ensembling | -0.30 | -0.40 | 3.7 | Medium | **0.1081** | novel high-value latency |
| 9 | **Time masking (SpecAugment-style) with learnable MASK token** | Regularization & data | -0.85 | -1.00 | 12.8 | Low | **0.0781** | high-value best-retrain |
| 10 | **Channel / electrode masking** | Regularization & data | -0.40 | -0.40 | 6.4 | Low | **0.0625** |  |
| 11 | **Causal transformer encoder (time-masked, compact)** | Architecture | -0.90 | -1.40 | 25.6 | Medium | **0.0547** | high-ceiling |
| 12 | **Snapshot ensemble from a single cyclic-LR run** | Ensembling | -0.35 | -0.35 | 6.4 | Medium | **0.0547** | cheap |
| 13 | **Intermediate / self-conditioned CTC with feedback** | Objective | -0.60 | -0.60 | 12.8 | Low | **0.0469** | high-value |
| 14 | **Calibration: label smoothing / entropy regularization on CTC** | Objective | +0.00 | -0.50 | 12.8 | Low | **0.0391** | novel high-value |
| 15 | **Cross-dataset pretraining on Willett T12 with day+dataset affine** | Regularization & data | -0.70 | -0.65 | 19.2 | Medium | **0.0339** |  |
| 16 | **Stochastic depth / drop-path across GRU layers** | Regularization & data | -0.20 | -0.20 | 6.4 | Low | **0.0312** |  |
| 17 | **Auxiliary diphone objective** | Objective | -0.40 | -0.40 | 12.8 | Medium | **0.0312** |  |
| 18 | **Deep ensemble, N=3, batched logit averaging** | Ensembling | -0.38 | -0.38 | 12.8 | Low | **0.0293** | high-value |
| 19 | **Day-adversarial training via gradient reversal** | Regularization & data | -0.15 | -0.15 | 6.4 | Medium | **0.0234** |  |
| 20 | **Checkpoint averaging / SWA / model soup** | Ensembling | -0.12 | -0.15 | 6.7 | Low | **0.0224** | cheap |
| 21 | **Larger patch size (22 or 34) at unchanged stride** | Architecture | -0.25 | -0.25 | 12.8 | Low | **0.0195** | cheap |
| 22 | **Predictive distillation from a bidirectional teacher (S4)** | Objective | -0.20 | -0.20 | 12.8 | Medium | **0.0156** | contested |
| 23 | **Sweep smooth_kernel_std at fixed lookahead=0** | Regularization & data | -0.30 | -0.30 | 19.2 | Low | **0.0156** |  |
| 24 | **Lookahead convolution with k in {1,2}** | Architecture | -0.25 | -0.25 | 19.2 | Low | **0.0130** | frontier |
| 25 | **Optimizer sweep: Adam epsilon, weight decay, LR schedule** | Regularization & data | -0.30 | -0.30 | 25.6 | Low | **0.0117** |  |
| 26 | **Mixup and temporal warping** | Regularization & data | -0.15 | -0.15 | 12.8 | Medium | **0.0117** |  |
| 27 | **Dynamic chunk training (U2 / U2++ style)** | Architecture | -0.15 | -0.15 | 12.8 | Medium | **0.0117** |  |
| 28 | **Deep ensemble, N=10 (reviewer prior S1 as specified)** | Ensembling | -0.60 | -0.60 | 57.6 | Low | **0.0104** | contested |
| 29 | **Sweep white-noise and constant-offset augmentation strengths** | Regularization & data | -0.25 | -0.25 | 25.6 | Low | **0.0098** |  |
| 30 | **Wider / deeper causal GRU (reviewer prior S2)** | Architecture | -0.05 | -0.05 | 19.2 | Medium | **0.0026** | contested |
| 31 | **Enable CTC blank-frame skipping (ctc_blank_skip_threshold)** | LM & decoding | +0.00 | +0.00 | 0.3 | Low | **-0.0000** | free enabler first |
| 32 | **Constrained baseline: incremental n-gram only, partial emission** | LM & decoding | +0.00 | +0.00 | 2.0 | Medium | **-0.0000** | gating first |
| 33 | **Delay-penalized CTC / Bayes-Risk CTC / Peak-First / TrimTail / Align-With-Purpose** | Objective | +0.00 | +0.00 | 38.4 | Low | **-0.0000** | killed |
| 34 | **Bandwidth-matched causal smoother control (resolves AUDIT F2)** | Validity | +0.00 | +0.00 | 6.4 | Low | **-0.0000** | validity critical |
| 35 | **Vectorize the streaming smoother and day layer** | Engineering | +0.00 | +0.00 | 0.2 | Low | **-0.0000** | free engineering |
| 36 | **Seed replication on headline and close-call configs (S7)** | Validity | +0.00 | +0.00 | 25.6 | Low | **-0.0000** | validity critical |
| 37 | **Causal Mamba-2 / selective state-space encoder** | Architecture | +0.05 | +0.05 | 19.2 | Medium | **-0.0026** | contested |
| 38 | **Reduced patch size to cut cold start** | Architecture | +0.15 | +0.15 | 12.8 | Low | **-0.0117** | killed |
| 39 | **Causal rolling feature normalization (fixes AUDIT F1)** | Validity | +0.80 | +0.80 | 12.8 | Medium | **-0.0625** | validity critical |
| 40 | **Wire up the existing CTC endpointer** | LM & decoding | +0.00 | +0.05 | 0.3 | Low | **-0.1667** | free enabler latency |

---

## Detailed entries

### LM & decoding

#### `ngram_4gram` — Build a pruned/quantized 4-gram LM that fits 32 GB
*Score 2.2500 · 1.0 GPU-h (0.0 runs) · risk Medium*

**Mechanism.** Build a custom KenLM 4-gram with aggressive pruning and quantization, compiled to a TLG WFST. The 7th-place B2T'25 entry reports doing exactly this at 19 GB RAM, down from the 300 GB unpruned 5-gram, trained on 80% general text + 20% conversational.

**Source.** 7th-place B2T'25 writeup (medium.com/@jackson3b04); repo README (3-gram ~60 GB, 5-gram ~300 GB)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** -2.25 (best -3.00, worst -1.50). Measured against the 1-gram that is the only artifact present locally. n-gram order is the largest single lever in the LM stack; arXiv:2507.02800 shows 3-gram -> 5-gram worth 12.17 -> 8.18 WER (4 pts) on the sibling benchmark. Conservative because that is a different participant and LM corpus.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp beam search cost grows with graph size; UNMEASURED

**Memory.** 19 GB inference RAM (fits 32 GB with the 131 MB acoustic model)

**Risk (Medium).** LM build toolchain (SRILM/KenLM + OpenFST) is fiddly; RAM during compilation can exceed the deployment target even if the artifact fits.

**Confounds.** Changes the optimal acoustic_scale - re-run lm_param_sweep afterwards.

**Kill criterion.** Best artifact under 32 GB fails to beat the 1-gram by >1 WER pt.

#### `adaptive_compute` — Adaptive-compute decoding gated on n-gram confidence (S5)
*Score 0.7000 · 2.0 GPU-h (0.0 runs) · risk High*

**Mechanism.** Always run the cheap incremental path; invoke expensive rescoring only when the n-gram hypothesis confidence falls below a threshold. Sweeping the threshold produces a genuine expected-latency vs accuracy curve - the deliverable the project currently lacks. Gate on LM score spread, NOT acoustic entropy: only 2.9% of frames exceed entropy 0.5 and the model is confidently wrong rather than uncertain (BUDGET 4).

**Source.** Reviewer prior S5; the 7th-place B2T'25 entry already ships a version (n-gram confidence gate at -3.76)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** -1.40 (best -2.00, worst -0.80). Upper-bounded by full rescoring's gain; realized fraction depends on how well the gate identifies the utterances rescoring would have fixed. Reports EXPECTED latency, so worst-case still violates the budget - state that explicitly.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp expected +5-50 ms; worst case still 620-830 ms

**Memory.** depends on the escalation path (up to 12.4 GB VRAM if OPT is the fallback)

**Risk (High).** Reports an expected latency against a HARD constraint. A reviewer can reasonably reject expected-case latency for a real-time claim.

**Confounds.** Requires incremental_baseline and a working rescorer first.

**Kill criterion.** Gate cannot reach 80% of full-rescore gain at <25% trigger rate.

#### `neural_lm_fusion` — Small causal neural LM via shallow fusion
*Score 0.6667 · 1.5 GPU-h (0.0 runs) · risk Medium*

**Mechanism.** Replace end-of-utterance OPT-6.7B rescoring with shallow fusion of a small causal LM (100-500 M params) scored incrementally during beam search. Because it is causal and scores token-by-token, it adds bounded per-frame cost instead of an end-of-utterance wall.

**Source.** Standard streaming-ASR shallow fusion; motivated by BUDGET 1 (OPT is a 620-830 ms batch wall)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** -1.00 (best -1.50, worst -0.50). Recovers part of the gap between n-gram-only and LLM-rescored decoding. arXiv:2507.02800 shows LLM rescoring worth 12.17 -> 5.68 (53% relative); shallow fusion with a much smaller causal LM typically captures a third to a half of that.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +2-10 ms per emitted word (estimated); fits the ~79 ms slack

**Memory.** 0.2-1.0 GB fp16 inference

**Risk (Medium).** Tokenizer mismatch between the WFST word lexicon and a subword LM; fusion weight needs its own sweep.

**Confounds.** Do not bundle with ngram_4gram - both change the LM score scale.

**Kill criterion.** Cannot beat a same-RAM larger n-gram at equal latency.

#### `incremental_llm_rescore` — Incremental LLM rescoring over a rolling context
*Score 0.5833 · 3.0 GPU-h (0.0 runs) · risk High*

**Mechanism.** Instead of rescoring complete sentences at end-of-utterance, rescore a rolling window of the partial hypothesis with hypothesis pruning, updating as words stabilize. Amortizes the LLM cost across the utterance rather than paying it in one block.

**Source.** Novel for this project; standard incremental-NLU pattern

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** -1.75 (best -2.50, worst -1.00). Aims to recover most of full LLM rescoring. Discounted because rescoring a partial hypothesis is strictly weaker than rescoring a complete one - the LLM's advantage is largely right-context, which a streaming system does not have.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +10-40 ms per stabilized word (estimated)

**Memory.** 12.4 GB VRAM for OPT-6.7B; less for a smaller rescorer

**Risk (High).** Repeated partial rescoring can thrash: each update may revise earlier words, directly worsening the stability metric this project is trying to own.

**Confounds.** Interacts with endpointing and with the stability metric.

**Kill criterion.** Flicker rate exceeds 0.5 revisions/word, or latency exceeds 79 ms at p95.

#### `lm_param_sweep` — Beam / acoustic-scale / blank-penalty sweep on cached logits
*Score 0.2167 · 3.0 GPU-h (0.0 runs) · risk Low*

**Mechanism.** Sweep acoustic_scale, blank_penalty, beam, max_active and nbest against the already-saved validation logits, so the acoustic model never re-runs. The repo's own two configurations disagree by 10x on blank_penalty (code default 9.0 at language-model-standalone.py:809 vs README recipe 90) and differ on acoustic_scale (0.3 vs 0.325), which is direct evidence that neither was tuned for this checkpoint.

**Source.** Repo defaults vs README recipe; standard WFST practice

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched; PER cannot change.

**Expected ΔWER** -0.65 (best -1.00, worst -0.30). Decode hyperparameters typically move WER 0.3-1.0 pts when previously untuned. Amplified here because the posterior is near one-hot (BUDGET 4), so the acoustic_scale that balances it against the LM is far from a default.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 (offline sweep); chosen point may change beam cost

**Memory.** none (logits cached, 102 MB pkl)

**Risk (Low).** Overfitting the val split if swept too finely; hold out sessions.

**Confounds.** Must be re-run after ANY change to the acoustic model or the LM artifact.

**Kill criterion.** Best swept point is within 0.1 WER pts of current defaults.

#### `blank_skip` — Enable CTC blank-frame skipping (ctc_blank_skip_threshold)
*Score -0.0000 · 0.3 GPU-h (0.0 runs) · risk Low*

**Mechanism.** Set blank_skip_thresh below 1.0 so the WFST decoder skips frames whose blank posterior exceeds it (ctc_wfst_beam_search.cc:79-84). Currently 1.0, which can never fire because exp(log p) <= 1. Measured on causal_la0: 71.83% of frames carry p(blank) > 0.999.

**Source.** Implemented in-repo; Blank Collapse arXiv:2210.17017 (~33% WFST decode-time reduction, WER unchanged)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** +0.00 (best -0.10, worst +0.10). Blank Collapse reports WER 1.783 -> 1.781 at theta=0.99, i.e. unchanged. Skipped frames here carry p(blank) >= 0.999, so risk is near nil. Treat as accuracy-neutral.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp -72% of WFST advance steps (measured skippable fraction)

**Memory.** none

**Risk (Low).** Aggressive theta can drop a genuine low-confidence phoneme; start at 0.999.

**Confounds.** Do not bundle with the beam sweep - it changes the effective search width.

**Kill criterion.** Measured WER rises more than 0.1 pts at theta=0.999.

#### `incremental_baseline` — Constrained baseline: incremental n-gram only, partial emission
*Score -0.0000 · 2.0 GPU-h (0.0 runs) · risk Medium*

**Mechanism.** Stop sending whole-trial logits (evaluate_model.py:237-250) and instead drive lm_decoder.DecodeNumpy one emitted frame at a time, reading the partial hypothesis stream. Delete Rescore(), augment_nbest() and OPT. This measures the WER and the per-frame cost of the only LM path that can fit inside 140 ms.

**Source.** Novel for this project; the incremental path already exists (ctc_wfst_beam_search.cc:98,113)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** +0.00 (best +0.00, worst +0.00). THIS IS THE BASELINE, not a delta. Every other WER number in this catalog is measured against it. Estimated at ~5.5% WER (range 4-8) by scaling the 12.17 -> 5.68 offline/LLM-rescored ratio in arXiv:2507.02800 onto this project's ~2.7% full-stack figure. UNMEASURED.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp the number the whole project needs and does not have

**Memory.** 13.4 MB (1-gram) up to 19 GB (4-gram)

**Risk (Medium).** lm_decoder must be built (CMake/gcc, separate conda env). Not built on this machine.

**Confounds.** Must be measured before any candidate that claims a WER delta.

**Kill criterion.** Cannot be killed - it is the measurement gate for everything else.

#### `endpointing` — Wire up the existing CTC endpointer
*Score -0.1667 · 0.3 GPU-h (0.0 runs) · risk Low*

**Mechanism.** ctc_endpoint.{h,cc} ships in the decoder but is never instantiated. Wiring it lets the decoder finalize on detected end-of-speech instead of waiting for the trial buffer to end. Measured: a median 1,680 ms of trailing silence follows the last emitted token.

**Source.** In-repo (language_model/runtime/core/decoder/ctc_endpoint.cc); standard Kaldi/WeNet practice

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Acoustic model untouched.

**Expected ΔWER** +0.05 (best +0.00, worst +0.20). Premature endpointing truncates utterances and can only hurt WER. Expect a small regression bought in exchange for ~1.7 s of finalization latency.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp finalization latency -1,680 ms (measured trailing silence)

**Memory.** none

**Risk (Low).** Cutting off slow or hesitant speech; tune the silence threshold on val.

**Confounds.** Interacts with blank_skip (both key off blank posterior).

**Kill criterion.** WER regression exceeds 0.2 pts at any endpoint threshold that saves >1 s.

### Test-time

#### `tta_dietcorp` — Lightweight test-time adaptation (DietCORP-style)
*Score 0.4333 · 1.5 GPU-h (0.0 runs) · risk Medium*

**Mechanism.** At test time, build Z augmented copies of the current trial (time masking, white noise, baseline shift), take ONE gradient step updating only the patch-embedding module, then decode. The source paper measures 18.21 +/- 5.34 ms per trial at 1.33 GiB - which fits comfortably inside this project's ~79 ms of unspent budget.

**Source.** arXiv:2507.02800 (DietCORP variant); contributes to their >20% WER reduction

**Expected ΔPER** -0.50 (best -0.80, worst -0.20). Directly targets the inter-session degradation that arXiv:2605.24313 identifies as the dominant source of variability on this data. Requires NO training run - it adapts the released checkpoint.

**Expected ΔWER** -0.65 (best -1.00, worst -0.30). Their gain is entangled with time masking and the transformer; the TTA-only component is UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +18.21 +/- 5.34 ms per trial (measured in source paper)

**Memory.** 1.33 GiB peak during adaptation (source paper)

**Risk (Medium).** A gradient step at inference breaks the streaming-equivalence guarantee and makes the decoder non-deterministic across runs. Also: is it per-trial (fits) or per-frame (does not)? Per-trial only.

**Confounds.** Adapting mid-utterance would change earlier frames' interpretation; adapt only at utterance boundaries.

**Kill criterion.** No WER improvement on held-out sessions, or per-trial cost exceeds 40 ms.

#### `online_day_adapt` — Online day-layer adaptation
*Score 0.3000 · 1.0 GPU-h (0.0 runs) · risk Medium*

**Mechanism.** Update only the day-specific affine layer online during a session using an unsupervised objective (e.g. entropy minimization on the decoder's own output), leaving the GRU frozen.

**Source.** Novel for this project; related to arXiv:2507.02800's TTA

**Expected ΔPER** -0.30 (best -0.50, worst -0.10). The day layer is 262,656 parameters and is the designated locus of session variation, so it is the natural adaptation target. SCOPE NOTE: the brief excludes calibration-efficiency work; this is online adaptation during use, not calibration reduction, but the boundary is thin and should be confirmed.

**Expected ΔWER** -0.30 (best -0.50, worst -0.10). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +~1 ms per update if updates are sparse

**Memory.** +1 optimizer state for 262 K params

**Risk (Medium).** Entropy minimization on an ALREADY over-confident posterior (mean entropy 0.9% of max) has almost no gradient signal - this specific objective is likely to no-op on this model.

**Confounds.** Interacts with label_smoothing, which changes the entropy the objective keys on.

**Kill criterion.** Adaptation moves val PER by <0.05 pts, consistent with a vanishing entropy gradient.

### Ensembling

#### `phase_staggered` — Phase-staggered replicas for sub-cadence resolution (S3)
*Score 0.1081 · 3.7 GPU-h (0.5 runs) · risk Medium*

**Mechanism.** Evaluate the model at K=4 patch phase offsets {0,20,40,60} ms and merge, producing a collective emission grid at 20 ms instead of 80 ms. This simultaneously cuts L_buf and acts as a test-time-augmentation ensemble. AUDIT F10 finds random_cut=3 already exposes training to 3 of the 4 phases, and the smoother (FIR) and day layer (per-bin) are phase-agnostic, so ONE trained model probably suffices - no retrain.

**Source.** Reviewer prior S3; mechanism is TTA, which bioRxiv 2026.06.02.729705 independently validates as 'pseudoensembling' with a single base decoder

**Expected ΔPER** -0.30 (best -0.50, worst -0.10). TTA over a nuisance variable the model was already partly trained against. Modest but real, and it is measurable in ~30 min from the existing checkpoint with no training.

**Expected ΔWER** -0.40 (best -0.60, worst -0.20). Two independent effects: TTA smoothing of the posterior, plus 4x finer emission grid giving the LM more evidence per unit time. UNCERTAIN which dominates.

**Latency.** L_algo 0 ms · L_buf -45 ms (60 ms worst case -> 15 ms) · L_comp 4x forwards, batched: measured 7.4 ms at batch=4 on CPU, ~0.4 ms on GPU

**Memory.** 1x weights (same model, 4 phases) = 131 MB

**Risk (Medium).** If the patch embedding turns out to be phase-sensitive despite random_cut, the 4 phases disagree and merging degrades accuracy. This is cheap to test first and the test is the deciding experiment.

**Confounds.** Reports both a latency AND an accuracy change; must be evaluated on both axes or the accuracy delta will be misread as free.

**Kill criterion.** Per-phase greedy PER varies by >0.5 pts across the 4 phases, indicating phase sensitivity that a single model cannot serve.

#### `snapshot_ensemble` — Snapshot ensemble from a single cyclic-LR run
*Score 0.0547 · 6.4 GPU-h (1.0 runs) · risk Medium*

**Mechanism.** Replace the single cosine decay with cyclic restarts and keep the model at each cycle minimum, yielding several diverse members from ONE training run.

**Source.** Huang et al. snapshot ensembles

**Expected ΔPER** -0.35 (best -0.50, worst -0.20). Snapshot members are less diverse than independently seeded ones, so expect roughly half to two-thirds of a true N=3 ensemble's gain - at one third of the cost.

**Expected ΔWER** -0.35 (best -0.50, worst -0.20). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp same as an N-member ensemble at inference

**Memory.** N x 65.5 MB fp16

**Risk (Medium).** Cyclic restarts change the LR schedule, confounding this with the optimizer sweep. Final-cycle quality may fall below the cosine baseline.

**Confounds.** Changes the LR schedule - must not be bundled with optimizer_sweep.

**Kill criterion.** Snapshot ensemble does not beat the single cosine-trained model by >0.15 WER pts.

#### `deep_ensemble_n3` — Deep ensemble, N=3, batched logit averaging
*Score 0.0293 · 12.8 GPU-h (2.0 runs) · risk Low*

**Mechanism.** Train 3 independently seeded causal models, run them phase-aligned as a single batch dimension, and average logits before the beam search. Compute is settled: measured 7.4 ms at batch=4 on CPU - 9.3% of the 80 ms window - with zero added algorithmic latency.

**Source.** arXiv:2412.17227 (all top-3 entrants ensembled); bioRxiv 2026.06.02.729705 (33.7% -> 26.0% closed-loop)

**Expected ΔPER** -0.38 (best -0.50, worst -0.25). Ensembling reliably delivers, but the published magnitude was measured at a 33.7% baseline WER - 12x this project's operating point. Gains compress sharply as baseline error falls. Discounted hard, and N=3 captures most of the diminishing-returns curve.

**Expected ΔWER** -0.38 (best -0.50, worst -0.25). Logit averaging before the LM helps most where the acoustic model is uncertain; the near-one-hot posterior (BUDGET 4) means members may be highly correlated, further compressing the gain. UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +~0.1 ms GPU (batched); measured 7.4 ms at batch=4 on CPU

**Memory.** 3 x 65.5 MB fp16 = 197 MB inference

**Risk (Low).** Logit averaging across members breaks the bit-exact streaming-equivalence gate unless the gate is extended to the ensemble.

**Confounds.** Members must differ ONLY by seed, or this measures architectural diversity instead.

**Kill criterion.** N=3 does not beat the best single member by >0.15 WER pts.

#### `checkpoint_avg` — Checkpoint averaging / SWA / model soup
*Score 0.0224 · 6.7 GPU-h (1.0 runs) · risk Low*

**Mechanism.** Average the weights of the last k validation checkpoints into one model. Zero inference cost - it produces a single model. AUDIT F8 shows best_val_PER is a minimum over 61 correlated evaluations and is ~0.04 pts optimistically biased, which is exactly the noise that weight averaging removes.

**Source.** Izmailov et al. SWA; Wortsman et al. model soups

**Expected ΔPER** -0.12 (best -0.20, worst -0.05). Late-training val PER has sd 0.033 pts across the last 10 evaluations (measured), so there is real variance to average away. Bounded because the checkpoints are highly correlated at the cosine LR floor.

**Expected ΔWER** -0.15 (best -0.25, worst -0.05). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 - single model at inference

**Memory.** none additional

**Risk (Low).** save_all_val_steps is FALSE (rnn_args.yaml:26), so no intermediate checkpoints exist. Costs one rerun to collect them - or is FREE if the flag is flipped on the next run that happens anyway.

**Confounds.** Averaging across a LR schedule change is invalid; average only within the flat region.

**Kill criterion.** Averaged model does not beat the best single checkpoint on val PER.

#### `deep_ensemble_n10` — Deep ensemble, N=10 (reviewer prior S1 as specified)
*Score 0.0104 · 57.6 GPU-h (9.0 runs) · risk Low*

**Mechanism.** As above but with 10 members. Feasibility is fully settled: measured 7.05 ms at batch=10 on CPU (8.8% duty cycle), 655 MB fp16, cost flat from N=4 to N=16 because the workload is launch-bound rather than arithmetic-bound.

**Source.** Reviewer prior S1; same sources as N=3

**Expected ΔPER** -0.60 (best -0.80, worst -0.40). Ensemble gain is roughly logarithmic in N, so 10 members buy perhaps 1.6x what 3 do - at 4.5x the training cost. The score reflects that asymmetry.

**Expected ΔWER** -0.60 (best -0.80, worst -0.40). Same reasoning as N=3; UNCERTAIN and measured at the wrong baseline.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp measured 7.05 ms at batch=10 on CPU; ~0.4 ms GPU

**Memory.** 655 MB fp16 inference (10 x 65.5 MB); trivial against 24 GB

**Risk (Low).** Cost, not correctness. 9 extra runs is 45% of a 20-run budget for a sublinear return.

**Confounds.** Same as N=3.

**Kill criterion.** Gain from N=3 to N=10 is under 0.2 WER pts, which would make the extra 7 runs indefensible.

### Regularization & data

#### `time_masking` — Time masking (SpecAugment-style) with learnable MASK token
*Score 0.0781 · 12.8 GPU-h (2.0 runs) · risk Low*

**Mechanism.** Mask contiguous temporal spans of the input during training and replace them with a learnable MASK embedding. arXiv:2507.02800 masks ~53% of every trial (N=20 spans, max length 0.075 of the trial, start and duration uniform, overlap allowed). This repo currently has NO time masking of any kind.

**Source.** arXiv:2507.02800 (Feghhi et al., rev 2025-11-02): 20-26% relative WER improvement

**Expected ΔPER** -0.85 (best -1.20, worst -0.50). Their headline is WER on a different participant. Discounted heavily to PER because PER is closer to the acoustic model and the LM absorbs some of the gain. The prior is strong: it is the largest single regularizer in the closest published causal result, and AUDIT A2 shows this repo has only two additive-noise augmentations.

**Expected ΔWER** -1.00 (best -1.50, worst -0.50). PER->WER transfer UNCERTAIN and probably sublinear here, because the LM already repairs many phoneme errors. Range widened accordingly.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 (training-time only)

**Memory.** negligible; may reduce VRAM via shorter effective sequences

**Risk (Low).** Masking so aggressively that CTC input length falls below target length -> infinite loss, and zero_infinity=False (rnn_trainer.py:242) turns that into a hard crash. Set zero_infinity=True first.

**Confounds.** Must not be bundled with channel_masking or the augmentation sweep.

**Kill criterion.** No val PER improvement at any mask fraction in {25%, 50%, 65%} over 2 seeds.

#### `channel_masking` — Channel / electrode masking
*Score 0.0625 · 6.4 GPU-h (1.0 runs) · risk Low*

**Mechanism.** Randomly zero (or replace with a learned token) a subset of the 512 feature channels per trial during training - the frequency-axis analogue of SpecAugment, adapted to electrodes. Directly targets electrode drift and dropout, which is the dominant non-stationarity in chronic intracortical recordings.

**Source.** SpecAugment analogue; motivated by AUDIT A2 (absent) and by inter-session degradation reported in arXiv:2605.24313

**Expected ΔPER** -0.40 (best -0.60, worst -0.20). Smaller prior than time masking because the day-specific affine layers already absorb some channel-level variation, but nothing in the pipeline currently forces robustness to losing individual electrodes.

**Expected ΔWER** -0.40 (best -0.60, worst -0.20). Assumes roughly 1:1 PER->WER transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 (training-time only)

**Memory.** negligible

**Risk (Low).** Masking channels after the day layer instead of before would defeat the point.

**Confounds.** Must not be bundled with time_masking.

**Kill criterion.** No val PER improvement at mask rates {5%, 10%, 20%}.

#### `cross_dataset_t12` — Cross-dataset pretraining on Willett T12 with day+dataset affine
*Score 0.0339 · 19.2 GPU-h (3.0 runs) · risk Medium*

**Mechanism.** Train jointly on Willett-2023 (T12) and Card-2024 (T15) using per-day AND per-dataset affine input transforms to align both participants into a shared latent space, then evaluate on T15. Adds roughly an order of magnitude more neural data at zero inference cost.

**Source.** J. Neural Eng. 23(4) 2026 (10.1088/1741-2552/ae8576): T15 10.2% -> 9.1% PER, 7.34% -> 6.67% WER

**Expected ΔPER** -0.70 (best -1.10, worst -0.30). Their measured T15 gain is 1.1 PER pts, but that is confounded with their hierarchical CTC and a bidirectional 2048-wide GRU. The authors explicitly note joint training helps T12 far more than T15, so the pretraining-only component is discounted.

**Expected ΔWER** -0.65 (best -1.00, worst -0.30). They report 0.67 WER pts on T15; PER->WER transfer looks sublinear.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 (inference uses one dataset's affine only)

**Memory.** +T12 dataset on disk; training VRAM unchanged

**Risk (Medium).** T12 has a different electrode count and feature layout; the affine alignment is where this fails silently rather than loudly.

**Confounds.** Must NOT be bundled with intermediate_ctc - the source paper changes both at once, which is exactly why its T15 attribution is ambiguous.

**Kill criterion.** T15 val PER does not improve by >0.2 pts after joint training with matched schedule.

#### `stochastic_depth` — Stochastic depth / drop-path across GRU layers
*Score 0.0312 · 6.4 GPU-h (1.0 runs) · risk Low*

**Mechanism.** Randomly skip whole GRU layers during training with a per-layer survival probability, so the network learns an implicit ensemble of depths. The 7th-place B2T'25 entry used drop-path together with heavy dropout and weight decay 0.005.

**Source.** 7th-place B2T'25 writeup; Huang et al. stochastic depth

**Expected ΔPER** -0.20 (best -0.30, worst -0.10). Modest prior. rnn_dropout=0.4 is already substantial, so the marginal regularization from drop-path is likely small.

**Expected ΔWER** -0.20 (best -0.30, worst -0.10). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 (train-time only; inference uses all layers)

**Memory.** negligible

**Risk (Low).** Skipping a GRU layer means passing the hidden state through unchanged, which requires matching widths - satisfied here (all layers 768).

**Confounds.** Must not be bundled with dropout-rate changes.

**Kill criterion.** No PER improvement at survival probabilities {0.9, 0.8}.

#### `day_adversarial` — Day-adversarial training via gradient reversal
*Score 0.0234 · 6.4 GPU-h (1.0 runs) · risk Medium*

**Mechanism.** Attach a day classifier to the shared representation behind a gradient-reversal layer, forcing the encoder to produce day-invariant features while the day-specific affine layers absorb the day-specific part.

**Source.** Ganin & Lempitsky DANN; not reported in any B2T writeup retrieved

**Expected ΔPER** -0.15 (best -0.30, worst +0.00). The day-affine layers (26.7% of parameters) already provide an explicit, supervised mechanism for day adaptation, so an adversarial term is partly redundant.

**Expected ΔWER** -0.15 (best -0.30, worst +0.00). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 (discriminator dropped at inference)

**Memory.** small discriminator head

**Risk (Medium).** Adversarial training is unstable; the reversal coefficient needs a schedule.

**Confounds.** Interacts with the day-layer learning rate group.

**Kill criterion.** No PER improvement, or training instability at any reversal coefficient.

#### `smooth_std_sweep` — Sweep smooth_kernel_std at fixed lookahead=0
*Score 0.0156 · 19.2 GPU-h (3.0 runs) · risk Low*

**Mechanism.** sigma=2 is inherited from the Willett-2023 lineage and has never been re-tuned for this model or for patch_size=14, which already integrates 14 bins. AUDIT F2 shows the accidental 37% bandwidth narrowing at L=0 coincided with the PER improvement, giving an evidence-backed direction.

**Source.** AUDIT F2; Wairagkar et al. use a 1.5 s causal kernel on this participant - 15x longer than this repo's 100 ms

**Expected ΔPER** -0.30 (best -0.50, worst -0.10). Directly motivated by an effect already observed in this project's own data. The direction is ambiguous though: the L=0 result argues for NARROWER, Wairagkar's 1.5 s kernel argues for much WIDER. Sweep both directions.

**Expected ΔWER** -0.30 (best -0.50, worst -0.10). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp wider kernel = more past taps, still negligible

**Memory.** none

**Risk (Low).** A much wider causal kernel lengthens the streaming ring buffer; the equivalence gate must be re-run.

**Confounds.** Must be run at fixed lookahead=0, and separately from bandwidth_matched_control (which changes the same kernel for a different purpose).

**Kill criterion.** No sigma in {1.0, 1.5, 3.0, 5.0} beats sigma=2 by >0.1 PER pts.

#### `optimizer_sweep` — Optimizer sweep: Adam epsilon, weight decay, LR schedule
*Score 0.0117 · 25.6 GPU-h (4.0 runs) · risk Low*

**Mechanism.** epsilon=0.1 is 10^7 times the PyTorch default of 1e-8, which largely disables Adam's adaptive term and makes it behave like signed SGD with momentum. weight_decay=0.001 is 5x lower than the 0.005 the 7th-place B2T'25 entry used. Both are inherited and untuned.

**Source.** rnn_args.yaml:45-46; arXiv:2412.17227 explicitly credits LR-schedule tuning; 7th-place writeup

**Expected ΔPER** -0.30 (best -0.50, worst -0.10). The '24 retrospective names LR scheduling as one of only two training changes that helped. eps=0.1 is extreme enough that it is either load-bearing or a fossil, and either answer is worth knowing.

**Expected ΔWER** -0.30 (best -0.50, worst -0.10). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** none

**Risk (Low).** Lowering eps sharply may destabilize training given lr_max=0.005.

**Confounds.** One variable per run.

**Kill criterion.** No config beats the baseline by >0.1 PER pts over 2 seeds.

#### `mixup_warp` — Mixup and temporal warping
*Score 0.0117 · 12.8 GPU-h (2.0 runs) · risk Medium*

**Mechanism.** Interpolate pairs of trials in feature space (mixup) and/or resample the time axis by a random factor (warping) to synthesize speaking-rate variation.

**Source.** Standard; not reported in any B2T writeup retrieved

**Expected ΔPER** -0.15 (best -0.30, worst +0.00). Weak prior. Mixup with CTC targets is awkward - there is no clean way to interpolate variable-length label sequences, so it degenerates to input-only mixing.

**Expected ΔWER** -0.15 (best -0.30, worst +0.00). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** negligible

**Risk (Medium).** Time warping changes sequence length and can push adjusted_lens below target length -> infinite CTC loss -> crash (zero_infinity=False).

**Confounds.** Warping overlaps with random_cut; do not change both.

**Kill criterion.** No PER improvement from either mechanism alone.

#### `aug_std_sweep` — Sweep white-noise and constant-offset augmentation strengths
*Score 0.0098 · 25.6 GPU-h (4.0 runs) · risk Low*

**Mechanism.** white_noise_std=1.0 and constant_offset_std=0.2 are inherited unchanged. Since the features are unit-variance, sigma=1.0 is a 100% noise-to-signal perturbation - an extreme setting that has never been justified on this dataset. Sweep both.

**Source.** rnn_args.yaml:62-63; the '24 retrospective frames the task as a regularization problem

**Expected ΔPER** -0.25 (best -0.40, worst -0.10). Tuning an inherited, obviously-extreme regularization constant usually yields something, but the value may already be near-optimal since it survived the original authors' tuning on this exact dataset.

**Expected ΔWER** -0.25 (best -0.40, worst -0.10). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** negligible

**Risk (Low).** None beyond cost.

**Confounds.** One variable per run: sweep white noise and offset separately.

**Kill criterion.** Best swept value is within 0.05 PER pts of sigma=1.0 / 0.2.

### Architecture

#### `causal_transformer` — Causal transformer encoder (time-masked, compact)
*Score 0.0547 · 25.6 GPU-h (4.0 runs) · risk Medium*

**Mechanism.** Replace the 5x768 GRU with 5 transformer blocks at d=384, 6 heads, unidirectional causal attention and T5 relative positions - 9.4 M parameters against the GRU's 56.7 M in the source paper. Trained with heavy time masking, this beat the unidirectional GRU by 20-26% relative WER and matched bidirectional SOTA.

**Source.** arXiv:2507.02800 (rev 2025-11-02): 15.25 -> 12.17 (3-gram), 11.12 -> 8.18 (5-gram)

**Expected ΔPER** -0.90 (best -1.50, worst -0.40). Strongest architectural evidence available, and it is causal by construction. Discounted because it is a different participant (T12, area 6v), a different benchmark, and because their gain is entangled with time masking - which is separately available here at a fraction of the cost.

**Expected ΔWER** -1.40 (best -2.00, worst -0.80). Their 20-26% relative on a ~12% baseline; scaled to this project's lower baseline with wide uncertainty. UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 43% fewer MFLOPs than the GRU in the source paper; still launch-bound

**Memory.** training VRAM 2.66 GiB vs 5.55 GiB (source paper); inference far smaller than 131 MB

**Risk (Medium).** A new architecture in this codebase means the streaming-equivalence gate, the day-layer plumbing and the patcher all need re-derivation. Attention over a growing context needs explicit KV caching to stay O(1)/frame.

**Confounds.** Their result bundles architecture with time masking. Run time_masking on the GRU FIRST to attribute correctly - otherwise this candidate takes credit for the cheap change.

**Kill criterion.** Does not match the GRU baseline within 0.3 PER pts after equal tuning budget.

#### `larger_patch` — Larger patch size (22 or 34) at unchanged stride
*Score 0.0195 · 12.8 GPU-h (2.0 runs) · risk Low*

**Mechanism.** Increase patch_size from 14 to 22 or 34 while holding patch_stride at 4. Each frame then integrates more past context at the SAME emission cadence, and because emission is right-edge, L_buf is unchanged. Only cold start grows - which BUDGET 2 shows is irrelevant here.

**Source.** 7th-place B2T'25 used GRU variants at 34/4 and 22/4

**Expected ΔPER** -0.25 (best -0.40, worst -0.10). A top-10 competition entry chose 22 and 34 over the baseline's 14, which is weak but real evidence that 14 is not optimal. Wairagkar's causal pipeline on this same participant integrates 1.5 s of past, far more than this repo's 280 ms patch.

**Expected ΔWER** -0.25 (best -0.40, worst -0.10). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp patch embedding grows linearly: 34/14 = 2.4x on weight_ih_l0

**Memory.** weight_ih_l0 grows from 16.5 M to 40 M params at P=34

**Risk (Low).** Changes the frame count and therefore adjusted_lens; short trials can hit adjusted_lens < target_lens -> infinite CTC loss -> crash.

**Confounds.** Changes cold start (280 -> 680 ms) and parameter count simultaneously; report both.

**Kill criterion.** No PER improvement at P in {22, 34}.

#### `lookahead_conv` — Lookahead convolution with k in {1,2}
*Score 0.0130 · 19.2 GPU-h (3.0 runs) · risk Low*

**Mechanism.** Insert a convolution that reads k future frames at a chosen depth, purchasing exactly 20k ms of L_algo in exchange for future context. Unlike smooth_lookahead, this is a clean, single-purpose lookahead knob that does not also change the smoothing bandwidth.

**Source.** Deep Speech 2 lineage (no single modern citation retrieved - UNVERIFIED)

**Expected ΔPER** -0.25 (best -0.40, worst -0.10). Modest gain per k. The real value is not accuracy: it is that this is the ONLY clean instrument for building a genuine accuracy-vs-lookahead curve, which AUDIT F2 shows the project currently lacks because smooth_lookahead confounds two variables.

**Expected ΔWER** -0.25 (best -0.40, worst -0.10). UNCERTAIN.

**Latency.** L_algo +20k ms (k=1 -> 20 ms, k=2 -> 40 ms) · L_buf 0 ms · L_comp +~1 MFLOP/frame

**Memory.** negligible

**Risk (Low).** Must be implemented so the streaming decoder genuinely waits for k frames, or the equivalence gate will pass while the deployed system cheats.

**Confounds.** Changes L_algo, so it must be compared at matched total latency, not matched PER.

**Kill criterion.** PER improvement per k is under 0.1 pts, making the frontier flat in this instrument too.

#### `dynamic_chunk` — Dynamic chunk training (U2 / U2++ style)
*Score 0.0117 · 12.8 GPU-h (2.0 runs) · risk Medium*

**Mechanism.** Randomize the attention/context chunk size during training so one model serves multiple latency operating points at inference.

**Source.** arXiv:2106.05642 (U2++), arXiv:2211.00941 (Fast-U2++, latency 320 -> 80 ms)

**Expected ΔPER** -0.15 (best -0.30, worst +0.00). Mostly inapplicable: this model is ALREADY fully causal with zero lookahead, so there is no chunk to shrink. Residual value is as a regularizer only.

**Expected ΔWER** -0.15 (best -0.30, worst +0.00). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** none

**Risk (Medium).** Presupposes a chunked encoder; the GRU has no chunk parameter.

**Confounds.** Only meaningful if paired with causal_transformer.

**Kill criterion.** No PER improvement, or found to be architecturally inapplicable to a GRU.

#### `wider_deeper_gru` — Wider / deeper causal GRU (reviewer prior S2)
*Score 0.0026 · 19.2 GPU-h (3.0 runs) · risk Medium*

**Mechanism.** Scale n_units and/or n_layers by 2-5x on the argument that the model uses 0.34 ms of an 80 ms window and is therefore massively under-provisioned in compute.

**Source.** Reviewer prior S2

**Expected ΔPER** -0.05 (best -0.20, worst +0.30). ARGUED AGAINST. The compute premise is true but irrelevant: the binding constraint is data, not FLOPs - 45 sessions from one participant. The closest published causal result went the OPPOSITE way, winning with 83% FEWER parameters (arXiv:2507.02800), and both competition retrospectives frame the task as a regularization problem. Expect neutral-to-worse without regularization changes first.

**Expected ΔWER** -0.05 (best -0.20, worst +0.30). UNCERTAIN, and the sign is genuinely in doubt.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +0.3-1.5 ms/frame; still inside budget

**Memory.** 2-5x of 131 MB inference; training VRAM scales similarly

**Risk (Medium).** Overfitting. 37% of parameters are already the single patch-embedding matrix, so widening mostly grows the part least likely to be the bottleneck.

**Confounds.** Meaningless unless run AFTER the regularization candidates - otherwise it measures the capacity/regularization interaction, not capacity.

**Kill criterion.** Val PER does not improve at 1.5x width over 2 seeds.

#### `mamba_ssm` — Causal Mamba-2 / selective state-space encoder
*Score -0.0026 · 19.2 GPU-h (3.0 runs) · risk Medium*

**Mechanism.** Replace or hybridize the GRU with a selective SSM, which is causal by construction and has favorable streaming properties (constant per-step state).

**Source.** arXiv:2607.26751 (Mamba hybrid competitive, NOT better); 7th-place B2T'25 (bidirectional Mamba, valued for ensemble diversity); arXiv:2412.17227 (SSMs did not beat GRU)

**Expected ΔPER** +0.05 (best -0.20, worst +0.30). Two independent controlled comparisons say Mamba matches but does not beat the GRU. The '24 retrospective's negative verdict was in a non-causal setting and deserves retesting, but arXiv:2607.26751 already retested it in 2026 and found the same.

**Expected ΔWER** +0.05 (best -0.20, worst +0.30). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp comparable to GRU; constant state per step

**Memory.** comparable

**Risk (Medium).** mamba-ssm kernels add a build dependency and can silently fall back to a slow path.

**Confounds.** Its real value is ENSEMBLE DIVERSITY (7th place credits uncorrelated errors), not standalone accuracy. Score it as an ensemble member, not a replacement.

**Kill criterion.** Does not match the GRU within 0.2 PER pts at equal parameter count.

#### `reduced_patch` — Reduced patch size to cut cold start
*Score -0.0117 · 12.8 GPU-h (2.0 runs) · risk Low*

**Mechanism.** Shrink patch_size below 14 so the 280 ms cold-start fill shortens.

**Source.** Reviewer-anticipated candidate

**Expected ΔPER** +0.15 (best +0.00, worst +0.40). KILLED BY MEASUREMENT. BUDGET 2 measures the first non-blank token at a median of 3,380 ms into the trial (p05 1,860 ms) - the participant does not begin speaking until long after the go cue, so the 280 ms fill is entirely absorbed. This buys nothing and costs context.

**Expected ΔWER** +0.15 (best +0.00, worst +0.40). Same.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp cold start reduced; cold start is not on the critical path

**Memory.** smaller weight_ih_l0

**Risk (Low).** n/a

**Confounds.** n/a

**Kill criterion.** ALREADY KILLED: median time-to-first-token 3,380 ms >> 280 ms cold start.

### Objective

#### `intermediate_ctc` — Intermediate / self-conditioned CTC with feedback
*Score 0.0469 · 12.8 GPU-h (2.0 runs) · risk Low*

**Mechanism.** Attach auxiliary CTC heads after intermediate GRU layers, then project their phoneme posteriors back into the hidden state before the next block. This conditions deeper layers on the shallow layers' predictions, partially relaxing CTC's conditional-independence assumption. Nothing about it requires future context.

**Source.** J. Neural Eng. 23(4) 2026 (hierarchical CTC, 10.1088/1741-2552/ae8576); Nozaki & Komatsu self-conditioned CTC

**Expected ΔPER** -0.60 (best -0.90, worst -0.30). The source paper's hierarchical CTC contributes measurably on top of joint training (T12: 17.6% plain -> 16.1% hierarchical, a 1.5 pt gain). Discounted for T15, where all their gains were smaller, and for their bidirectional 2048-wide setting.

**Expected ΔWER** -0.60 (best -0.90, worst -0.30). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +~2 MFLOP/frame for the aux heads; inference can drop them entirely

**Memory.** +2 x (768x41) params, negligible

**Risk (Low).** Auxiliary loss weight needs tuning; too high and the shallow layers over-commit to phoneme decisions the deep layers cannot revise.

**Confounds.** Must NOT be bundled with cross_dataset_t12.

**Kill criterion.** No PER improvement at aux weights {0.1, 0.3, 0.5}.

#### `label_smoothing` — Calibration: label smoothing / entropy regularization on CTC
*Score 0.0391 · 12.8 GPU-h (2.0 runs) · risk Low*

**Mechanism.** Add an entropy bonus or label-smoothing term to the CTC objective so the acoustic posterior stops collapsing to one-hot. Measured on causal_la0: mean posterior entropy is 0.033 nats against a maximum of 3.714 - 0.9% of maximum, with a numerically one-hot median frame. A near-one-hot posterior gives the beam search almost no acoustic alternatives, so the LM must overcome a ~1.0 posterior to fix an acoustic error.

**Source.** Novel for this project - proposed from the BUDGET 4 measurement, not from the literature

**Expected ΔPER** +0.00 (best -0.10, worst +0.20). Greedy PER may WORSEN slightly, since smoothing trades argmax sharpness for calibration. This candidate is not about PER.

**Expected ΔWER** -0.50 (best -0.80, worst -0.20). The gain is expected in the LM stage: a wider n-best list gives the LM more to work with. PER->WER transfer here is expected to be NEGATIVE-to-positive - the rare case where PER and WER should move in opposite directions. That makes it a clean test of whether PER is the right proxy at all.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** none

**Risk (Low).** Over-smoothing flattens the posterior enough that blank skipping stops firing, silently raising LM cost.

**Confounds.** Must be evaluated on WER, not PER. Re-run lm_param_sweep afterwards - the optimal acoustic_scale WILL move.

**Kill criterion.** WER does not improve at smoothing in {0.05, 0.1} after re-sweeping acoustic_scale.

#### `diphone_aux` — Auxiliary diphone objective
*Score 0.0312 · 12.8 GPU-h (2.0 runs) · risk Medium*

**Mechanism.** Add a second CTC head predicting context-dependent diphones alongside the phoneme head, sharing the encoder. Diphones capture coarticulation that monophone targets discard; the head is dropped at inference so latency is unchanged.

**Source.** arXiv:2412.17227 names diphones as one of two training changes that helped; 1st-place B2T'24 used them; arXiv:2507.02800 reports diphone+GRU at 8.39% WER

**Expected ΔPER** -0.40 (best -0.60, worst -0.20). Credited by the organizers' retrospective, but that was in a bidirectional, unconstrained setting and the reported diphone GRU (8.39%) did not beat the plain time-masked causal transformer (8.18%).

**Expected ΔWER** -0.40 (best -0.60, worst -0.20). Assumes ~1:1 transfer; UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 at inference (aux head dropped)

**Memory.** 39^2 = 1,521 classes before pruning -> 768x1521 = 1.2 M extra training params

**Risk (Medium).** Diphone class count explodes and most classes are unobserved; needs frequency pruning or the aux CTC becomes degenerate.

**Confounds.** Must not be bundled with intermediate_ctc - both add auxiliary CTC heads.

**Kill criterion.** No PER improvement at aux weights {0.1, 0.3} with a pruned diphone inventory.

#### `predictive_distillation` — Predictive distillation from a bidirectional teacher (S4)
*Score 0.0156 · 12.8 GPU-h (2.0 runs) · risk Medium*

**Mechanism.** Train a bidirectional teacher, then distill into the causal student with a per-frame KD loss shifted by the teacher's measured emission delay, forcing the student to anticipate information the teacher only has in hindsight.

**Source.** Reviewer prior S4; largely PRE-EMPTED by Delayed-KD arXiv:2505.22069 (non-streaming teacher -> streaming student via CTC posteriors, with a Temporal Alignment Buffer)

**Expected ΔPER** -0.20 (best -0.40, worst +0.00). Doubly discounted. First, Delayed-KD already does this, and its Temporal Alignment Buffer already sweeps a delay range - S4's proposed differentiator (deriving the shift from measured peaks) is a narrow delta. Second, BUDGET 3 measures the causal/bidirectional emission offset at +4.64 ms, so there is almost no shift to apply.

**Expected ΔWER** -0.20 (best -0.40, worst +0.00). UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0 at inference (teacher discarded)

**Memory.** teacher only during training

**Risk (Medium).** Requires a bidirectional teacher run that is otherwise useless, and the KD weight/temperature interact with the CTC objective.

**Confounds.** Distillation gain must be separated from the teacher simply being better.

**Kill criterion.** Student does not beat a same-cost directly-trained causal model.

#### `delay_penalty_family` — Delay-penalized CTC / Bayes-Risk CTC / Peak-First / TrimTail / Align-With-Purpose
*Score -0.0000 · 38.4 GPU-h (6.0 runs) · risk Low*

**Mechanism.** All five shape CTC alignments to emit earlier: penalize per-frame delay via a differentiable FST, reweight paths by a Bayes risk, distill toward left-shifted posteriors, trim trailing input frames, or add a general alignment-ranking loss.

**Source.** arXiv:2305.11539, arXiv:2210.07499, arXiv:2211.03284, arXiv:2211.00522, arXiv:2307.01715 (all 3-4 yr old)

**Expected ΔPER** +0.00 (best +0.00, worst +0.10). KILLED BY MEASUREMENT. BUDGET 3 measures the learned emission delay of causal_la0 at +4.64 ms - 6% of one frame - after subtracting the smoother's predicted group delay. There is no delay to remove. These methods can only cost accuracy here.

**Expected ΔWER** +0.00 (best +0.00, worst +0.10). Same.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** none

**Risk (Low).** Not a risk - simply solving a problem this system does not have.

**Confounds.** n/a

**Kill criterion.** ALREADY KILLED: measured excess learned delay = +4.64 ms (13,030 paired tokens).

### Validity

#### `bandwidth_matched_control` — Bandwidth-matched causal smoother control (resolves AUDIT F2)
*Score -0.0000 · 6.4 GPU-h (1.0 runs) · risk Low*

**Mechanism.** Build a causal (lookahead=0) kernel whose effective sigma matches L=4's 1.852 bins rather than L=0's 1.161, by extending the past tail. AUDIT F2 shows smooth_lookahead changes lookahead AND low-pass bandwidth together (37% narrower at L=0), so the project's headline result is confounded: L=0's advantage may be a bandwidth effect, not a causality effect.

**Source.** AUDIT F2 (measured kernel properties)

**Expected ΔPER** +0.00 (best -0.20, worst +0.20). This is a CONTROL, not an improvement. If it lands near 10.21% the gain was bandwidth; if near 10.04% something else is happening. Either answer is publishable; not knowing is not.

**Expected ΔWER** +0.00 (best -0.20, worst +0.20). Not the point of this run.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +~4 taps of FIR, negligible

**Memory.** none

**Risk (Low).** Straightforward kernel change; the streaming path must be updated to match the longer past tail.

**Confounds.** Must be run alone against causal_la0 with an identical seed.

**Kill criterion.** Cannot be killed - both outcomes are informative. That is what makes it a control.

#### `seed_replication` — Seed replication on headline and close-call configs (S7)
*Score -0.0000 · 25.6 GPU-h (4.0 runs) · risk Low*

**Mechanism.** Run 2-3 seeds on the headline configurations so the central claim carries an error bar. AUDIT F9 finds the only same-config replicate available (baseline_rnn vs causal_la4, which are algorithmically identical) differs by 0.041 PER pts, against a la4->la0 effect of 0.163 pts on the last-10 mean.

**Source.** Reviewer prior S7; AUDIT F9 (measured)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Measures variance, does not reduce error.

**Expected ΔWER** +0.00 (best +0.00, worst +0.00). Same.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp 0

**Memory.** none

**Risk (Low).** Pure cost.

**Confounds.** Seeds must vary ONLY the torch/numpy seed, not the dataset seed, or the split changes.

**Kill criterion.** Cannot be killed - the variance estimate is the deliverable. NOTE: the evidence suggests this will CONFIRM a real ~0.15 pt causal advantage rather than bury it, contrary to the brief's framing of Fact A as 'inside seed noise'.

#### `causal_normalization` — Causal rolling feature normalization (fixes AUDIT F1)
*Score -0.0625 · 12.8 GPU-h (2.0 runs) · risk Medium*

**Mechanism.** Re-normalize the released features with a causal rolling estimator (Welford, ~10 s half-life) instead of the whole-block z-score the data actually carries, then retrain. AUDIT F1 establishes block normalization by a 1/sqrt(W) scaling test matching prediction within 1-26% across four sessions. Wairagkar et al. use exactly this rolling-past-10 s scheme in their real-time system on the SAME participant.

**Source.** AUDIT F1 (measured); Wairagkar et al., Nature 644:145-152 (2025)

**Expected ΔPER** +0.80 (best +0.30, worst +1.50). This is a COST, not a gain - the sign is deliberately positive. Removing a leak makes the task harder. The magnitude is the number the project needs and nobody has published: the true price of end-to-end causality on T15.

**Expected ΔWER** +0.80 (best +0.30, worst +1.50). Same. UNCERTAIN.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp +negligible (Welford is O(1) per feature per bin)

**Memory.** +2 x 512 floats of running state

**Risk (Medium).** The exact upstream transform (clip before or after z-score) cannot be recovered from the released files, so the re-normalization is an approximation of an unknown original.

**Confounds.** Must be run alone. Changes the input distribution, so every other tuned hyperparameter is potentially invalidated.

**Kill criterion.** Cannot be killed on accuracy grounds - a regression IS the result. Killed only if the rolling estimator cannot be made to reproduce the released features when given block statistics, indicating the transform was misidentified.

### Engineering

#### `vectorized_streaming` — Vectorize the streaming smoother and day layer
*Score -0.0000 · 0.2 GPU-h (0.0 runs) · risk Low*

**Mechanism.** Replace the per-tap Python loop in _compute_smoothed (streaming_infer.py:194-199) and the linear ring-buffer scan in _raw_for_index (:201-208) with a single vectorized FIR, and hoist the scipy kernel rebuild out of gauss_smooth. Measured: the reference GPU path costs ~0.185 ms/bin while a vectorized CPU equivalent costs 0.0131 ms - the GPU implementation is 14x SLOWER than CPU doing identical arithmetic.

**Source.** AUDIT F5, BUDGET 5 (measured)

**Expected ΔPER** +0.00 (best +0.00, worst +0.00). Bit-identical arithmetic; PER cannot change.

**Expected ΔWER** +0.00 (best +0.00, worst +0.00). Bit-identical.

**Latency.** L_algo 0 ms · L_buf 0 ms · L_comp -0.7 ms per cadence window (70% of measured streaming cost)

**Memory.** also fixes unbounded transformed_buffer / logit_frames growth (AUDIT F14)

**Risk (Low).** Must keep the streaming-equivalence gate passing at <1e-3.

**Confounds.** none

**Kill criterion.** Equivalence gate fails, or measured RTF does not improve.


---

## What this ranking gets wrong

The score is **marginal efficiency**, not a plan. Read it as "where is the next hour best spent,"
never as "do these in order." Five specific distortions:

**1. It systematically over-rewards LM work and under-rewards retrains.** The top seven entries
require zero training runs. That is genuinely where the efficiency is — but the cheap LM
candidates are *not independent*. Every one of them changes the LM score scale, so
`lm_param_sweep` has to be re-run after each, and all of them depend on `incremental_baseline`
existing first. Their scores are individually correct and jointly misleading.

**2. `ngram_4gram`'s 2.25/h is inflated by the baseline it is measured against.** The only LM
artifact present locally is the 1-gram (13.4 MB), so the −2.25 WER point estimate is
1-gram → 4-gram. If the constrained baseline is established with a 3-gram instead, the marginal
gain shrinks by most of that. This is the single least trustworthy number in the catalog.

**3. Validity candidates score ≤ 0 by construction and are still load-bearing.**
`causal_normalization` scores −0.0625 because removing a leak makes the task *harder*.
`bandwidth_matched_control` scores 0.0000 because it is a control. `seed_replication` scores
0.0000 because variance estimation is not error reduction. None of that makes them optional —
without them the project's central claim is confounded ([AUDIT F2](AUDIT.md#f2)), unsound
([AUDIT F1](AUDIT.md#f1)), and un-error-barred ([AUDIT F9](AUDIT.md#f9)) respectively.

**4. Two entries were killed by Phase C measurement, freeing 51.2 GPU-hours.**
`delay_penalty_family` (5 methods, 38.4 h) and `reduced_patch` (12.8 h). Both were solving
problems this system does not have — a learned emission delay of **+4.64 ms** and a cold start
that is absorbed by a **3,380 ms** pre-speech period. They are retained in the catalog with
score 0 precisely so the kill is on the record rather than silently omitted.

**5. Score ignores dependency order and diminishing returns across candidates.** Several
candidates target the same error mass — `time_masking`, `channel_masking`, `stochastic_depth`
and `aug_std_sweep` are all regularization, and their gains will not add. The run ledger, not
this table, is where that gets resolved.

---

## Reviewer-seeded priors: adversarial verdicts

| Prior | Verdict | Evidence |
|---|---|---|
| **S1** — causal deep ensembling | **PARTIALLY SUPPORTED. Feasibility settled, magnitude not.** | Compute and memory are decisively fine: measured **7.05 ms at batch=10 on CPU** (8.8% duty cycle), 655 MB fp16, cost flat N=4→16 because the workload is launch-bound ([BUDGET §5](BUDGET.md)). But the published gain (33.7% → 26.0%) was measured at a baseline **12× higher** than this project's. Gains compress as baseline error falls. **N=3 scores 3× better than N=10** (0.0293 vs 0.0104) — the brief's N=10 specification is the worst point on its own curve. |
| **S2** — capacity is under-spent | **REJECTED.** | Scores 0.0026, third from bottom, and the sign of ΔPER is genuinely in doubt. The compute premise is true but irrelevant: the binding constraint is 45 sessions from one participant. The closest published causal result won with **83% fewer parameters** ([arXiv:2507.02800](LITERATURE.md)), and both competition retrospectives frame this as a regularization problem. 37% of parameters are already the single patch-embedding matrix. |
| **S3** — phase-staggered replicas | **SUPPORTED, and cheaper than the brief assumed.** | [AUDIT F10](AUDIT.md) finds `random_cut=3` already exposes training to **3 of 4** patch phases, and the smoother (FIR, shift-equivariant) and day layer (per-bin) are phase-agnostic — so one trained model probably serves all four phases with **no retrain**. Testable in ~30 min from the existing checkpoint. Ranks 8th at 0.1081, the highest-scoring candidate that touches latency. **Novelty is threatened**: the June 2026 ensembles preprint independently proposes "pseudoensembling" via test-time augmentation with a single base decoder. S3 must be positioned as TTA *over patch phase*, whose distinctive payoff is sub-cadence temporal resolution — not as TTA for BCI ensembling. |
| **S4** — predictive distillation | **LARGELY PRE-EMPTED, and undercut by measurement.** | Delayed-KD ([arXiv:2505.22069](LITERATURE.md)) already does non-streaming-teacher → streaming-student KD on CTC posteriors, with a Temporal Alignment Buffer that already sweeps a delay range. S4's differentiator — deriving the shift from measured peaks — is narrow. Worse, [BUDGET §3](BUDGET.md) measures the causal/symmetric emission offset at **+4.64 ms**, so there is almost no shift to apply. Score 0.0156. **Demote or re-scope.** |
| **S5** — adaptive-compute decoding | **SUPPORTED as high-value; partially pre-empted; one real reviewer risk.** | Ranks 2nd at 0.70. But the 7th-place B2T'25 entry already ships a version (gate on n-gram confidence at −3.76), so the novelty is in the *latency framing*, not the mechanism. The real exposure: it reports **expected** latency against a **hard** constraint — worst case is still 620–830 ms. A reviewer can reasonably reject expected-case latency for a real-time claim, and the recommendation must confront that rather than bury it. [BUDGET §4](BUDGET.md) also corrects the gating signal: acoustic entropy fires on only 2.9% of frames and carries little information because the model is confidently wrong rather than uncertain. Gate on LM score spread. |
| **S6** — measure before you fix | **CONFIRMED. This was the highest-value instruction in the brief.** | [BUDGET §3](BUDGET.md), 13,030 paired tokens across 512 trials: la0 emits **+0.364 frames** later than la4, of which **+0.306** is predicted by the smoother's group delay alone. **Excess learned delay: +4.64 ms — 6% of one frame.** The entire emission-latency literature is inapplicable. **Frees 6 runs / 38.4 GPU-h.** |
| **S7** — the n=1 problem | **CONFIRMED, but the brief has the direction backwards.** | [AUDIT F9](AUDIT.md#f9): `baseline_rnn` and `causal_la4` are *algorithmically identical* (for an odd 9-tap kernel `padding='same'` pads exactly (4,4)) yet differ by **0.041 PER pts**. The la4→la0 effect is **0.163 pts** on the last-10 mean — about 4× the only same-config replicate available. Seeds are needed to **confirm a likely real ~0.15-point causal advantage**, not to bury a null. Fact A's "the difference is inside seed noise" is not what the evidence says. |

---

## Coverage check against the required families

| Required family | Candidates |
|---|---|
| Objective / alignment shaping | `intermediate_ctc`, `diphone_aux`, `label_smoothing`, `predictive_distillation`, `delay_penalty_family` (delay-CTC, BRCTC, Peak-First, TrimTail, AWP — all killed) |
| Causal architecture | `causal_transformer`, `wider_deeper_gru`, `mamba_ssm`, `lookahead_conv`, `larger_patch`, `reduced_patch` (killed), `dynamic_chunk` |
| Regularization and data | `time_masking`, `channel_masking`, `aug_std_sweep`, `optimizer_sweep`, `stochastic_depth`, `mixup_warp`, `day_adversarial`, `cross_dataset_t12`, `smooth_std_sweep` |
| Ensembling and averaging | `phase_staggered`, `deep_ensemble_n3`, `deep_ensemble_n10`, `checkpoint_avg`, `snapshot_ensemble` |
| LM and decoding | `lm_param_sweep`, `blank_skip`, `endpointing`, `incremental_baseline`, `ngram_4gram`, `neural_lm_fusion`, `adaptive_compute`, `incremental_llm_rescore` |
| Test-time | `tta_dietcorp`, `online_day_adapt` |
| *(added by this audit)* Validity & engineering | `causal_normalization`, `bandwidth_matched_control`, `seed_replication`, `vectorized_streaming` |

FastEmit and the delay-penalized transducer are folded into `delay_penalty_family` with an explicit
note: both are **transducer-only** and there is no transducer in this system, so they do not apply
even before the emission-delay measurement kills the family.

`dilated causal TCN front end` was considered and **not carried forward as a separate candidate**:
it occupies the same design slot as `causal_transformer` and `larger_patch` (buying more past
context at fixed cadence) with weaker published support on this benchmark family, so it would have
added a row without adding information.
