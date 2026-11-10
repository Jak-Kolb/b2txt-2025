> Historical evidence, not current instructions. Read ../../RESEARCH_ASSESSMENT.md for corrections. Links to removed proposal files are archival references recoverable from Git or .git/context-cleanup/2026-09-18/.

# AUDIT.md — Phase A pipeline forensics

Scope: `Jak-Kolb/b2txt-2025` @ `main` (4617886), which is **ahead of** `origin/causal-preprocessing`
(the benchmark harness exists only on `main`; `causal-preprocessing` is the older branch).
Everything below is read from source or measured on this machine. Confidence is tagged
**[measured]**, **[derived]**, **[estimated]**, or **[UNVERIFIED]**.

No pipeline code was modified and no training was launched.

---

## 0. Headline: four claims in the ground state are wrong

| # | Ground-state claim | Verdict | Where |
|---|---|---|---|
| **A** | "HDF5 features are causally normalized (0/45 sessions show a block z-score signature)" | **FALSE.** Features are whole-block z-scored. The repo's own checker returns a false negative because it thresholds a `max` over 512 channels. | [F1](#f1) |
| **B** | `smooth_lookahead` is a clean lookahead knob; L=0 vs L=4 measures the cost of causality | **FALSE.** It confounds lookahead with low-pass bandwidth. L=0 also narrows effective smoothing by 37%. | [F2](#f2) |
| **C** | "L_buf = 80 ms from the patch stride" | **WRONG QUANTITY.** 80 ms is the *update cadence*. Emission buffering is 0–60 ms, mean 30 ms. | [F3](#f3) |
| **D** | "A pruned 3-gram (~6–8 GB) is the current target" | **UNSUPPORTED.** The repo's own README says the 3-gram needs **~60 GB**. Neither shipped n-gram fits 32 GB. | [F6](#f6) |

Two further corrections, less severe: the LM beam search **is already incremental** ([F4](#f4)),
and the streaming RTF is ~70% Python bookkeeping rather than model compute ([F5](#f5)).

---

## 1. Finding table

Causality verdict: **CAUSAL** / **NON-CAUSAL** / **N/A**. Latency verdict: contribution to
L_algo (waiting for future data), L_buf (buffering), L_comp (compute), or none.

| ID | Component | file:line | What the code actually does | Causal? | Latency | Lever? |
|---|---|---|---|---|---|---|
| F1 | Feature normalization (upstream) | `analyses/check_hdf5_causality.py:76`; data | Whole-block z-score, ~240–1140 s blocks | **NON-CAUSAL** | 0 ms (offline), but invalidates real-time claim | Yes — high |
| F2 | Gaussian smoother | `data_augmentations.py:25-41` | Truncate+renormalize 9-tap σ=2 kernel; changes bandwidth *and* lookahead together | CAUSAL at L=0 | L_algo = 20·L ms | Yes — high |
| F3 | Patcher | `rnn_model.py:106-119`; `benchmark/streaming_infer.py:218-221` | `unfold(3,14,4)`, right-edge emission | **CAUSAL** | L_buf 0–60 ms; cadence 80 ms; cold start 280 ms | Yes |
| F4 | WFST beam search | `.../ctc_wfst_beam_search.cc:70-121` | Frame-by-frame `AdvanceDecoding(...,1)`; partial best path each call | **CAUSAL, incremental** | L_comp — unmeasured | Yes — highest |
| F5 | Streaming harness cost | `benchmark/streaming_infer.py:194-208` | Python per-tap FIR + linear ring-buffer scan | N/A | 70% of measured L_comp is overhead | Yes (free) |
| F6 | n-gram artifact | `README.md:101`; `language_model/README.md:1` | 1-gram shipped (13.4 MB); 3-gram ~60 GB, 5-gram ~300 GB | N/A | RAM wall | Yes |
| F7 | Test labels | `data/hdf5_data_final/*/data_test.hdf5` | **Contains only `input_features`.** No labels of any kind | N/A | — | Blocker |
| F8 | Checkpoint selection | `rnn_trainer.py:599-614` | `best_val_PER` = min over 61 validations | N/A | — | Yes |
| F9 | Reproducibility | `results/*/training_log` | Two *algorithmically identical* configs differ by 0.041 PER pts | N/A | — | Informs S7 |
| F10 | `random_cut` | `rnn_trainer.py:468-471` | Crops 0–2 bins from sequence start → 3 of 4 patch phases | CAUSAL | — | Yes — enables S3 |
| F11 | CTC loss | `rnn_trainer.py:242` | `zero_infinity=False` + `error_if_nonfinite=True` at `:554` | N/A | — | Robustness risk |
| F12 | Blank skipping | `.../ctc_wfst_beam_search.cc:79-84` | Implemented; `blank_skip_thresh` default 1.0 = **disabled** | CAUSAL | Cuts LM work | Yes — free |
| F13 | Rescore / OPT | `language-model-standalone.py:591-631` | Only inside the finalize branch | **BATCH — latency wall** | Unbounded | Must replace |
| F14 | Streaming buffers | `benchmark/streaming_infer.py:216,235` | `transformed_buffer` / `logit_frames` never trimmed | N/A | Memory grows with utterance | Fix before deploy |
| F15 | Kernel rebuild | `data_augmentations.py:15-31` | scipy kernel rebuilt + H2D copy on **every** call | N/A | Part of the 70% | Yes (free) |
| F16 | Duplicate loss append | `rnn_trainer.py:720` and `:760` | `metrics['losses']` appended twice per batch | N/A | — | Cosmetic |

---

## 2. A1 — Feature path

**Shape and content [measured].** 512 features/bin, 20 ms bins (`benchmark/common.py:25`),
`float32` on disk. Two 256-channel blocks over 256 electrodes:

| Block | mean | std | min | max | unique values (32,946 samples) | Identity |
|---|---|---|---|---|---|---|
| `[0:256]` | −0.006 | 0.995 | −1.264 | **+10.000** | 170 | threshold crossings (discrete) |
| `[256:512]` | −0.013 | 0.997 | −2.769 | **+10.000** | 32,935 | spike-band power (continuous) |

Both are **clipped at exactly +10.0** upstream (`0.004%` of samples at the rail) and z-scored.
No normalization, clipping, or log transform happens anywhere in this repo — `dataset.py:130`
reads `input_features` verbatim. The only in-pipeline transform is the Gaussian smoother.

<a name="f1"></a>
### F1 — The features are whole-block z-scored. This is non-causal, and the repo concludes the opposite.

`log.md` and `analyses/check_hdf5_causality.py` report *"0/45 sessions show a whole-block
z-score (future-leak) signature"*, and the ground state elevates this to "HDF5 features are
causally normalized." **That conclusion does not survive.**

The checker's test (`check_hdf5_causality.py:76`) is
`max_over_512_channels(|mean|) < 0.01 AND max(|std−1|) < 0.01`. Using `max` over 512 channels
means one badly-behaved channel fails the whole session. The **median** tells a different story
[measured, all 45 sessions]:

```
block-level  max |mean|   median across sessions = 0.1308     <- the statistic the checker uses
trial-level  max |mean|   median across sessions = 0.4366     <- 3.3x larger, so NOT per-trial
block-level  median|mean| (session t15.2023.08.13, block 1)   = 0.0085
```

Decisive test: **if a single global normalization was applied at block scale, sub-window means
are pure subsampling noise and must shrink as 1/√W.** If instead a causal rolling window (e.g.
Wairagkar's past-10 s) were used, the 10-s windowed means would be *pinned near zero* and would
not shrink further with W. Measured on session `t15.2023.08.13`, block 1 (815 s):

| window W | 5.0 s | 10.0 s | 20.0 s | full block (815 s) |
|---|---|---|---|---|
| median per-feature abs mean | 0.1002 | 0.0736 | 0.0511 | **0.0085** |
| ratio to previous | — | 1.36 (√2 = 1.414) | 1.44 | — |

Extrapolating the 1/√W law from W=1000 bins to the full block predicts **0.0080**; observed
**0.0085**. Across four sessions spanning two years and block lengths 240 s–1141 s, the
prediction matches within 1–26%:

```
t15.2023.08.13 blk1  T= 40727 ( 815s): predicted 0.00800 | observed 0.00853 | ratio 1.07
t15.2023.11.17 blk1  T= 57061 (1141s): predicted 0.00721 | observed 0.00727 | ratio 1.01
t15.2024.03.08 blk1  T= 18936 ( 379s): predicted 0.01251 | observed 0.01574 | ratio 1.26
t15.2025.03.30 blk3  T= 12013 ( 240s): predicted 0.01523 | observed 0.01636 | ratio 1.07
```

The residual 0.0085 is exactly the sampling noise expected because the *released* trials are a
subset of the block (inter-trial periods are not distributed) — not evidence against block
normalization.

**Verdict: NON-CAUSAL.** Every feature at time *t* was divided by a standard deviation and
offset by a mean computed over up to ~19 minutes of *future* recording.

**Why this is the single most important finding.** The project's contribution is a real-time
latency claim. That claim is currently made about a model whose *inputs* contain block-scale
future information. No amount of causal smoothing, causal patching, or streaming-equivalence
testing repairs a leak that happens before the data reaches this repository. Three consequences:

1. The 10.04% / 10.21% numbers are **not** achievable in a genuine real-time system. They are
   upper bounds obtained with leaked normalization statistics.
2. The streaming-equivalence gate is sound but proves less than advertised: it certifies that
   streaming reproduces offline, on inputs that are themselves non-causal.
3. This is *fixable and cheap to quantify*: re-normalize the released features with a causal
   rolling estimator (Welford, ~10 s half-life) and retrain. That single run measures the true
   cost of end-to-end causality — which is the number the paper actually needs and does not have.

Caveat [UNVERIFIED]: the exact upstream transform (clip before or after z-score, whether the
+10 rail is pre- or post-normalization) cannot be recovered from the released files. The
block-scale conclusion does not depend on resolving it.

---

## 3. A2 — Augmentation

All augmentation lives in `rnn_trainer.transform_data` (`rnn_trainer.py:436-485`), applied on
GPU, **train mode only** (`:447`). `data_augmentations.py` contains only `gauss_smooth`.

| Augmentation | Config value | Active? | Code | Notes |
|---|---|---|---|---|
| static gain | `static_gain_std: 0.0` | **off** | `:449-453` | full 512×512 warp matrix per example |
| white noise | `white_noise_std: 1.0` | on | `:456-457` | σ=1.0 on unit-variance features — a **100% noise-to-signal** perturbation |
| constant offset | `constant_offset_std: 0.2` | on | `:460-461` | per-trial, per-channel DC shift |
| random walk | `random_walk_std: 0.0` | **off** | `:464-465` | |
| random cut | `random_cut: 3` | on | `:468-471` | `randint(0,3)` → crop 0/1/2 bins |
| Gaussian smooth | `smooth_kernel_std: 2` | on (train+val) | `:475-482` | not an augmentation |

**Absent entirely** — and this is the gap the '24 retrospective's "it's a regularization problem"
framing points straight at: no channel/electrode masking, no time masking (SpecAugment), no
mixup, no temporal warping or resampling, no stochastic depth, no label smoothing, no
cutout. The only two live augmentations are additive noise. There is also **no dropout between
the last GRU layer and the output head** (`rnn_model.py:126-129`); `rnn_dropout=0.4` applies only
*between* GRU layers, per `nn.GRU` semantics.

<a name="f10"></a>
### F10 — `random_cut` already covers 3 of the 4 patch phases

`cut = np.random.randint(0, 3)` yields `cut ∈ {0,1,2}`, shifting the whole batch before patching.
With `patch_stride=4` this exposes the model to **3 of 4** patch phase alignments during training
(phase 3 is never sampled). One scalar `cut` is drawn per batch, not per trial.

This materially de-risks **S3 (phase-staggered replicas)**: the reviewer's spec worried that
replicas might need separate training. They very likely do not — the smoother is an FIR (shift-
equivariant) and the day layer is per-bin (phase-agnostic), so only the patch embedding sees
phase, and it has already been trained under phase jitter. Raising `random_cut` to `4` gives full
phase coverage at **zero** additional cost and is the natural enabling change.

---

## 4. A3 — Architecture

`GRUDecoder`, `rnn_model.py:4-134`. Unidirectional (`bidirectional=False`, `:71`).

**Exact parameter count [measured]:**

```
total                       44,315,177
  day layers (45 × 262,656) 11,819,520   26.7%     45 × (512×512 + 512), init to identity
  GRU                       32,463,360   73.3%
    gru.weight_ih_l0        16,515,072   37.3%  <- 2304 × 7168, the patch embedding alone
    gru.weight_hh_l0         1,769,472
    gru.{weight_ih,hh}_l1-4 14,155,776
  out (768→41)                  31,529
  h0 (learned init)                 768
INFERENCE-RESIDENT (1 day)  32,758,313
```

Two structural facts worth carrying forward: **26.7% of parameters are day layers, of which
1/45 is used at inference**; and **37% of all parameters are the single patch-embedding matrix**
`weight_ih_l0`, a direct consequence of `patch_size=14 × 512 = 7168` input width.

**FLOPs per emitted frame, batch=1 [derived]:**

| Stage | MFLOP | share |
|---|---|---|
| smoother (5 taps × 4 bins) | 0.020 | 0.0% |
| day layer (512×512 × 4 bins) | 2.097 | 3.1% |
| GRU L0 (7168→2304, + recurrent) | 36.569 | 54.5% |
| GRU L1–4 | 28.312 | 42.2% |
| output head | 0.063 | 0.1% |
| **total** | **67.06** | |

At 12.5 frames/s that is **0.838 GFLOP/s sustained**.

| Device | peak | arithmetic duty cycle |
|---|---|---|
| RTX 3090 fp32 | 35.6 TFLOP/s | **0.0024%** |
| RTX 3090 TF32 | 71.0 TFLOP/s | 0.0012% |
| laptop RTX 4060 | 15.0 TFLOP/s | 0.0056% |
| CPU (M3 Pro, ~0.3 TFLOP/s) | 0.3 TFLOP/s | 0.28% |

**The measured 0.3196 ms patch forward is 175× the arithmetic time (0.0018 ms).** The model is
**kernel-launch-bound, not FLOP-bound** — five sequential `cuDNN` GRU steps plus a GEMM, each a
separate launch, on tensors far too small to fill the device. This single fact reframes the
capacity and ensembling questions: widening the model or batching *N* replicas into one batch
dimension costs almost nothing in wall time until the arithmetic term becomes comparable, which
is ~2–3 orders of magnitude away.

**Inference weight memory [derived]:** fp32 full checkpoint 177.3 MB; inference-resident (one
day layer) 131.0 MB; fp16 65.5 MB. A 10-member fp16 ensemble is **655 MB** — trivial against
24 GB. VRAM at *training* time is **[UNVERIFIED]** (no GPU on this machine); it must be measured
before any capacity recommendation is finalized.

---

## 5. A4 — Training loop

`rnn_trainer.py`. Optimizer `AdamW(fused=True)` (`:283-290`) with three param groups (`:267-276`):
biases (`weight_decay=0`), day layers (own LR + `weight_decay_day=0`), everything else.

| Setting | Value | Line |
|---|---|---|
| LR schedule | cosine, warmup 1000 → decay to 120k | `:294-363`, cfg `:33-41` |
| lr_max / lr_min | 0.005 / 0.0001 (same for day layers) | cfg `:34-41` |
| betas | (0.9, 0.999) | cfg `:43-44` |
| **eps** | **0.1** | cfg `:45` |
| weight decay | 0.001 (main), 0 (day, bias) | cfg `:46-47` |
| grad clip | 10.0, `error_if_nonfinite=True` | `:551-556` |
| batch | 64 trials, 4 days/batch, sampled **with replacement** | cfg `:74-77`, `dataset.py:197` |
| total batches | 120,000 | cfg `:32` |
| validation every | 2,000 batches → **61 validations** | cfg `:52` |
| early stopping | `false` | cfg `:29` |
| AMP | bfloat16 | cfg `:19`, `:528` |
| compile | `torch.compile(model)` | `:134` |
| seed | 10 (torch/np/random), dataset seed 1 | cfg `:48`, `:78` |

`epsilon: 0.1` is **10⁷× the PyTorch default** (1e-8). Inherited from the Stanford NPTL lineage.
It heavily damps the adaptive term for all but the largest gradients, making AdamW behave closer
to signed-SGD-with-momentum. It has, as far as this repo shows, never been re-tuned. Given that
the '24 retrospective credits LR-schedule tuning with real gains, this is a genuinely untouched
and cheap knob.

<a name="f8"></a>
### F8 — `best_val_PER` is a minimum over 61 correlated evaluations

`rnn_trainer.py:599-614` checkpoints whenever `avg_PER < best_val_PER`. The reported headline is
therefore an **order statistic**, not an estimate of converged performance, and is optimistically
biased. Measured on the three completed runs:

| run | best (reported) | at batch | mean of last 10 vals | sd of last 10 | selection bias |
|---|---|---|---|---|---|
| `baseline_rnn` | 0.10250 | 116,000 | 0.10286 | 0.00018 | −0.00036 |
| `causal_la4` | 0.10210 | 102,000 | 0.10250 | 0.00023 | −0.00040 |
| `causal_la0` | 0.10040 | 106,000 | 0.10087 | 0.00033 | −0.00047 |

Bias is a consistent **~0.04 PER points**. It is roughly equal across runs, so *differences*
between runs are less affected than absolute numbers — but any comparison against an external
figure (e.g. arXiv:2607.26751's 12.62% PER) must account for it. Validation cadence (every 2,000
batches) is ample for checkpoint averaging over the last *k* evaluations, which is a
**zero-inference-cost** lever that the pipeline already has the data for.

<a name="f9"></a>
### F9 — An in-hand estimate of run-to-run noise: 0.041 PER points

`baseline_rnn` and `causal_la4` are **algorithmically identical**. Diffing their saved configs
shows the only substantive difference is that `baseline_rnn` has no `smooth_lookahead` key
(→ `lookahead=None` → `F.conv1d(padding='same')` with the full 9-tap kernel) while `causal_la4`
sets `lookahead=4` (→ keep taps `[:4+4+1]` = all 9, renormalize by a sum that is already 1.0,
then `F.pad(inputs,(4,4))`). For an odd 9-tap kernel `padding='same'` pads exactly (4,4).
**These are the same computation.**

They differ only in `gpu_number` ('0' vs '1') and output directory. Yet:

```
baseline_rnn  0.10251   (last-10 mean 0.10286)
causal_la4    0.10210   (last-10 mean 0.10250)
                Δ = 0.041 pts (best)  /  0.036 pts (last-10 mean)
```

This is a single draw from the same-config, different-environment distribution — weak, but it is
the *only* replicate the project has, and it is real evidence. Set against it, the la4 → la0
difference is **0.163 points on the last-10 mean, ~4× the one observed same-config difference**.

**This argues against Fact A as stated.** "The difference is inside seed noise" is not what the
available evidence says; the evidence is more consistent with a real effect that n=1 cannot
establish. S7's seed plan is therefore not hygiene — it may confirm a genuine and counterintuitive
positive result. The val CTC loss gap points the same way and is much cleaner than PER:
la0 **21.74** vs la4 **22.60** vs baseline **22.38** (final validation).

---

## 6. A5 — Loss

```python
rnn_trainer.py:242   self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)
rnn_trainer.py:533   adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
rnn_trainer.py:539   loss = ctc_loss(logits.log_softmax(2).permute(1,0,2), labels, adjusted_lens, phone_seq_lens)
rnn_trainer.py:546   loss = torch.mean(loss)
```

`blank=0`; 41 classes = blank + 39 ARPAbet phonemes + `' | '` (silence/word boundary) at index 40
(`evaluate_model_helpers.py:9-20`). `adjusted_lens` uses float division then `int32` truncation,
which correctly reproduces `unfold`'s `floor((T−P)/s)+1`. **No auxiliary losses exist** — no
diphone head, no intermediate CTC, no self-conditioning. Reduction is `'none'` then an unweighted
`mean`, so long and short utterances contribute equally per-trial (not per-frame).

<a name="f11"></a>
**F11 — robustness risk.** `zero_infinity=False` combined with `error_if_nonfinite=True` in the
gradient clip (`:554`) means a single trial with `adjusted_lens < phone_seq_lens` produces
`inf` loss → `nan` grad → **hard crash**. It has not fired in three runs, but any change that
shortens sequences (larger `patch_size`, larger `random_cut`, aggressive time masking) can trip
it. Set `zero_infinity=True` before any such experiment.

---

## 7. A6 — Decoding stack (priority)

Full path from logits to text:

```
logits [T_frames, 41]                       rnn_model.py:129
  └─ rearrange [BLANK, phonemes…, SIL] → [BLANK, SIL, phonemes…]
                                            evaluate_model_helpers.py:79-83, called at evaluate_model.py:220
  └─ redis xadd, float32 bytes              evaluate_model_helpers.py:215
  └─ lm_decoder.DecodeNumpy(...,log(blank_penalty))
                                            language-model-standalone.py:769-772
       └─ CtcWfstBeamSearch::Search         ctc_wfst_beam_search.cc:70-121
            ├─ optional blank skip          ctc_wfst_beam_search.cc:79-84
            ├─ AdvanceDecoding(&decodable,1) PER FRAME   ctc_wfst_beam_search.cc:98
            └─ GetBestPath(&lat,false)  → PARTIAL hypothesis   :113
  └─ partial published                      language-model-standalone.py:785
  ─────────────── end of utterance ───────────────
  └─ FinishDecoding() → FinalizeSearch()    brain_speech_decoder.cc:42-45
  └─ [optional] Rescore()  unpruned G.fst lattice compose   brain_speech_decoder.cc:61-101
  └─ [optional] augment_nbest()  O(n²) word-swap over top-20  language-model-standalone.py:327-411
  └─ [optional] OPT-6.7B rescore of all candidates  language-model-standalone.py:620-631
  └─ argmax of  acoustic_scale·AC + (1−α)·LM + α·LLM        :233
```

**Parameters [measured from source defaults, `language-model-standalone.py:799-814`]:**

| Parameter | Default | Used in README recipe |
|---|---|---|
| `max_active` / `min_active` | 7000 / 200 | hard-coded at `:488-489`, **ignores CLI** |
| `beam` / `lattice_beam` | 17.0 / 8.0 | hard-coded at `:490-491`, **ignores CLI** |
| `acoustic_scale` | 0.3 | 0.325 |
| `blank_penalty` | 9.0 | 90 |
| `nbest` | 100 | 100 |
| `alpha` (LLM weight) | 0.5 | 0.55 |
| `ctc_blank_skip_threshold` | **1.0 (disabled)** | 1.0 |
| `top_candidates_to_augment` | 20 | 20 |

Note `build_lm_decoder` is called at `:486-496` with **literal** `max_active=7000, min_active=200,
beam=17., lattice_beam=8.` — the corresponding CLI flags are dead at startup and only take effect
through the `remote_lm_update_params` path (`:708-718`). Worth knowing before sweeping them.

### Per-stage incremental vs. batch

| Stage | Incremental? | Verdict |
|---|---|---|
| Smoothing, day layer, patching, GRU, head | **Yes** | proven by the equivalence gate |
| CTC-WFST beam search | **Yes** — `AdvanceDecoding(…,1)` per frame, partial best path after every call | **not a latency wall** |
| Blank skipping | Yes, per frame | disabled by default |
| `FinishDecoding` / `FinalizeSearch` | End of utterance | small; needed for n-best lattice |
| **`Rescore()` (unpruned G)** | **No** — composes the whole lattice | **WALL** |
| **`augment_nbest()`** | **No** — O(n²) over full n-best | **WALL** |
| **OPT-6.7B rescoring** | **No** — scores complete sentences | **WALL** |

<a name="f4"></a>
### F4 — The incremental LM already exists; the harness just doesn't use it

This is the most consequential correction to the project's mental model. The brief lists
"streaming/incremental KenLM and WFST beam search with partial-hypothesis emission" as the
"least-explored" area to investigate. **It is already implemented and working in this
repository.** `ctc_wfst_beam_search.cc:98` advances Kaldi's lattice-faster decoder one frame at a
time, and `:113` extracts a partial best path (`use_final=false`) after every `Search()` call,
which `language-model-standalone.py:785` publishes to `remote_lm_output_partial`.

What makes the *reference stack* batch is `evaluate_model.py:237-250`: it sends the entire
trial's logits in one `xadd` and then immediately finalizes. That is an evaluation-harness
choice, not an architectural limit.

So the real question is not "can the LM be made incremental" but **"what does the incremental LM
cost per 80 ms frame, and how much accuracy is lost by deleting the three batch stages?"**
Neither number is measured anywhere in this repo. Both are Phase C work.

<a name="f6"></a>
### F6 — The RAM plan does not close

`README.md:101` and `language_model/README.md:1`, from the upstream authors:

> "Note that the 3gram model requires ~60GB of RAM, and the 5gram model requires ~300GB of RAM.
> Furthermore, OPT 6.7b requires a GPU with at least ~12.4 GB of VRAM."

Against a 32 GB deployment budget: the 5-gram is out (already known), **and so is the 3-gram at
~60 GB**. The brief's "pruned 3-gram (~6–8 GB) is the current target" is not sourced in this
repo and does not match the only figure the repo states. Getting to 6–8 GB means *building and
pruning a new LM*, which is real work with its own accuracy cost — not a download. The only
artifact actually present locally is the 1-gram: `openwebtext_1gram_lm_sil/TLG.fst` (13.4 MB) +
`words.txt` (1.8 MB) [measured].

[UNVERIFIED] Which LM produced the quoted 2.66% local WER. Only the 1-gram exists on this
machine, and the reference recipe pairs it with OPT-6.7B — which would place essentially all of
the lexical work in the LLM rescoring stage, i.e. squarely inside the batch wall. This needs to
be pinned down before any WER claim is made, because it determines how much of the 2.66% is
reachable incrementally.

---

## 8. A7 — Benchmark harness

`model_training/benchmark/{common,offline_benchmark,streaming_infer,test_equivalence}.py`.

### Does the equivalence test test what it claims?

**Mostly yes.** `test_equivalence.py:55-108` recomputes logits two ways — offline
(`common.offline_logits`, the production path) and streaming (`streaming_infer.run_streaming_trial`,
an independent reimplementation of smoother + day layer + patcher + GRU) — then asserts max
absolute logit difference `< 1e-3` **and** exact equality of the collapsed greedy sequences.
That is a real, non-trivial acceptance gate and the streaming reimplementation is genuinely
independent, so it catches real bugs.

Three caveats:

1. **It is an implementation-equivalence test, not a causality test.** It passes for
   `causal_la4` too (`test_equivalence.py:155-159` defaults to checking both), where the streaming
   decoder simply waits for 4 future bins. Passing this gate does not certify L_algo = 0.
2. The online collapse rule (`common.py:257-265`, append when `argmax != previous_argmax` and
   `!= blank`) differs syntactically from the offline rule (`common.py:245-250`,
   `unique_consecutive` then strip blanks) but is provably equivalent — verified on
   `[A,A,blank,A]` and `[A,blank,blank,A]`, both yield `[A,A]`. No issue.
3. `test_equivalence.py:95` references `frame_diff` which is only bound in the `else` branch at
   `:91`. Unreachable today (empty logits give `max_diff = 0.0 < tolerance`), but fragile.

### Does the RTF include the LM? **No — and it is named honestly.**

`streaming_infer.py:294-295`:
```python
"total_acoustic_sec": total_sec,
"rtf": float(total_sec / (n_bins * BIN_SECONDS)) ...
```
`total_compute_sec()` (`:169-170`) sums only per-bin and finish time. **Ground-state Fact C is
confirmed from source: every published RTF in this project is acoustic-model-only.** The n-gram
beam search, WFST, and rescoring contribute exactly zero to 0.00072 and 0.0132.

<a name="f5"></a>
### F5 — 70% of the measured streaming cost is the reference implementation, not the model

Measured, `results/trained_models/causal_la0/checkpoint/metrics.json`, n=1423 val trials
(= 1426 val trials − 3 warmup, i.e. the full val split), device recorded only as `cuda`:

| | mean | p50 | p95 | max |
|---|---|---|---|---|
| streaming RTF | 0.013183 | 0.013074 | 0.013803 | 0.015131 |
| per-bin (ms) | 0.2638 | 0.1826 | 0.5127 | |
| per-emitted-patch (ms) | 0.3196 | 0.3165 | 0.3409 | |
| total per trial (ms) | 243.36 | 234.59 | 397.17 | |
| offline RTF | 0.000722 | 0.000711 | 0.000762 | 0.009766 |
| offline forward (ms/trial) | 12.62 | 12.18 | 20.29 | |

Decomposition [derived]. Mean trial ≈ 923 bins (from RTF and total). Patch timing is **nested
inside** bin timing (`streaming_infer.py:224,228` sits within the `process_bin` timed region at
`:132-136`), so:

```
total per trial                 243.5 ms
  patch forwards  228 × 0.3196 =  72.9 ms   (30%)   <- the actual network
  everything else              = 170.6 ms   (70%)   <- Python FIR + ring-buffer scan + day layer
per 80 ms cadence window: 4 × 0.185 + 0.320 = 1.06 ms busy  ->  98.7% idle
```

This is fully consistent with the reported percentiles: a non-emitting bin costs ≈0.185 ms
(≈ p50 0.1826) and an emitting bin ≈0.185+0.320 = 0.505 ms (≈ p95 0.5127).

The 70% is avoidable overhead, not physics. `_compute_smoothed` (`:194-199`) runs a **Python loop
over kernel taps**, each launching a separate GPU multiply-add; `_raw_for_index` (`:201-208`)
does a **linear scan** of the ring buffer for every tap; and `gauss_smooth`
(`data_augmentations.py:15-31`) rebuilds the scipy kernel and re-copies it host→device on every
call. Offline processes the same 923 bins in 12.6 ms — **19× faster** — doing identical math.

Two consequences: (a) the true achievable acoustic L_comp is well under 1 ms and the "76× real
time" figure understates headroom by roughly 3×; (b) fixing this is free (no retrain) and should
precede any latency claim in a paper.

<a name="f14"></a>
**F14 — not yet a bounded-memory streaming decoder.** `transformed_buffer` (`:216`) and
`logit_frames` (`:235`) grow without eviction for the whole utterance. Fine for a benchmark,
wrong for deployment; `transformed_buffer` needs only the last `patch_size` entries.

---

## 9. A8 — Evaluation protocol

**Split [measured].** Not the `test_percentage: 0.1` in the config — that key is **dead**. The
trainer builds the split from *separate files*: `data_train.hdf5` with `test_percentage=0`
(all → train) and `data_val.hdf5` with `test_percentage=1` (all → val), `rnn_trainer.py:153-174`.
The split is the competition's own, fixed on disk.

```
train: 45 sessions,  8,072 trials
val:   41 sessions,  1,426 trials      (4 sessions have dataset_probability_val=0)
test:  41 sessions,  1,450 trials
total              10,948 trials
```

**PER is micro-averaged, not per-session.** `rnn_trainer.py:765`:
`avg_PER = total_edit_distance / total_seq_length`, summed over all trials. Per-day PERs are
logged separately (`:588-590`) but are not what the headline reports. Long sentences therefore
dominate the metric.

<a name="f7"></a>
### F7 — Test labels do not exist. At all.

Measured directly on `data_test.hdf5` for session `t15.2023.08.13`:

| split | datasets present | attrs present |
|---|---|---|
| train | `input_features`, `seq_class_ids`, `transcription` | `block_num`, `n_time_steps`, `sentence_label`, `seq_len`, `session`, `trial_num` |
| val | same as train | same as train |
| **test** | **`input_features` only** | **`block_num`, `n_time_steps`, `session`, `trial_num`** |

No `seq_class_ids`, no `transcription`, no `sentence_label`, no `seq_len`. With the leaderboard
closed, **test WER is unobtainable by any means available to this project.**

The brief's objective — "minimize test WER" — is therefore not directly measurable, and every
comparison to the 1.5% / 1.78% / 2.79% leaderboard figures is a comparison against a metric this
project cannot compute. This needs to be stated plainly in any writeup.

**Proposed held-out protocol.** Split the 1,426-trial val set into `val-dev` and `val-test` by
**session**, not by trial: hold out a contiguous block of the latest sessions (e.g. the 8 sessions
from `t15.2025.01.10` onward, ~10% of trials) as `val-test`, tune everything on the remaining 33,
and touch `val-test` once per recommendation. Splitting by session rather than trial preserves
the day-layer generalization structure and avoids leaking block-level normalization statistics
between dev and test — which, given [F1](#f1), would otherwise be a genuine leak.

How it differs from the leaderboard metric, explicitly: (i) different and smaller trial set
(~140 vs 1,450); (ii) drawn from val blocks, which the organizers may have balanced differently
from test; (iii) the day layers for those sessions **were trained on**, since every session
contributes training trials — so this measures within-session-day generalization, whereas the
leaderboard also does. Both are copy-task sentences from the same participant, so corpus
composition is comparable. The honest framing is *"held-out sessions from the public validation
split"*, never *"test WER."*

---

## 10. A trap in the central experiment

<a name="f2"></a>
### F2 — `smooth_lookahead` is not a clean causality knob

`data_augmentations.py:25-28` truncates the symmetric kernel to offsets `[−p, +L]` and
renormalizes. With `smooth_kernel_std=2, smooth_kernel_size=100`, the surviving kernel
(`gaussKernel > 0.01`, `:18`) has **K=9 taps, p=4** [measured], so `L=4` reproduces the symmetric
baseline exactly and `L=4` is the maximum. Full kernel:

```
offset:  -4      -3      -2      -1       0      +1      +2      +3      +4
weight:  0.0276  0.0663  0.1238  0.1802  0.2042  0.1802  0.1238  0.0663  0.0276
```

Truncating changes **three** things at once, not one [measured]:

| L | taps | span | lookahead | group delay | **effective σ** | peak weight |
|---|---|---|---|---|---|---|
| **0** | 5 | −4…0 | **0 ms** | +24.5 ms | **1.161 bins (23.2 ms)** | 0.339 |
| 1 | 6 | −4…+1 | 20 ms | +14.2 ms | 1.384 (27.7 ms) | 0.261 |
| 2 | 7 | −4…+2 | 40 ms | +6.8 ms | 1.588 (31.8 ms) | 0.225 |
| 3 | 8 | −4…+3 | 60 ms | +2.3 ms | 1.749 (35.0 ms) | 0.210 |
| **4** | 9 | −4…+4 | **80 ms** | 0 ms | **1.852 bins (37.0 ms)** | 0.204 |

Going L=4 → L=0 removes 80 ms of lookahead **and simultaneously narrows the low-pass by 37%**
(σ_eff 1.852 → 1.161 bins) and raises the peak weight by 66%. The L=0 model sees measurably
**sharper, less temporally blurred features**.

So the observed result — L=0 *better* by 0.163 points on the last-10 mean, with a large and clean
val-loss gap (21.74 vs 22.60) — has an obvious alternative explanation that has nothing to do with
causality: **σ=2 over-smooths, and truncation accidentally fixed it.** σ=2 is inherited from the
Willett-2023 lineage and, on the evidence in this repo, has never been re-tuned for this model or
for `patch_size=14` (which already aggregates 14 bins and provides its own temporal integration).

Consequences:

- **Fact A's negative result is confounded.** "Causality is free" is not established. What is
  established is "this particular truncation is free-or-better," which is a weaker and different
  claim. A reviewer will find this.
- **The true cost of causality is still unmeasured.** The clean control is a **bandwidth-matched
  causal kernel**: keep `lookahead=0` but extend the past tail / raise σ until σ_eff ≈ 1.85 bins,
  matching L=4. If that matches L=4's 10.21%, the gain was bandwidth. If it matches L=0's 10.04%,
  something else is going on. **One run, single variable, resolves the project's central claim.**
- **It also exposes a free lever.** If narrower smoothing helps, `smooth_kernel_std` is an
  untuned hyperparameter with an evidence-backed direction. Sweeping σ ∈ {1.0, 1.5, 2.0, 3.0} at
  fixed L=0 is single-variable and directly motivated by an effect already observed.

Note also: at L=0 the kernel centroid sits **1.224 bins (24.5 ms) in the past**. This is not a
waiting latency — no future sample is required, so L_algo = 0 stands — but it does mean the
feature at frame *t* predominantly reflects activity centred 24.5 ms earlier. That distinction
belongs in BUDGET.md.

<a name="f3"></a>
### F3 — L_buf is 0–60 ms, not 80 ms

The brief decomposes the budget as "L_buf (buffering, currently 80 ms from the patch stride)".
80 ms is the **update cadence** — the interval between successive emissions — which is a
different quantity from emission latency.

From `streaming_infer.py:218-221`, frame *f* covers smoothed bins `[4f, 4f+13]` and is emitted as
soon as bin `4f+13` arrives. The newest bin in each patch is emitted **immediately**; the patch
looks backwards, not forwards. A neural event in bin *b* is first covered by frame
`f = ceil((b−13)/4)`, emitted at bin `4f+13`, so the wait is `(4f+13) − b ∈ {0,1,2,3}` bins:

```
b = 13 → emit 13, delay 0 bins
b = 14 → emit 17, delay 3 bins = 60 ms   (worst case)
b = 15 → emit 17, delay 2 bins = 40 ms
b = 16 → emit 17, delay 1 bin  = 20 ms
b = 17 → emit 17, delay 0
```

**L_buf: max 60 ms, mean 30 ms.** Cold start (first patch requires 14 bins) is 280 ms, one-time,
already correctly identified as separate.

Corrected acoustic budget for `causal_la0` [derived]:

| term | brief's figure | audited figure |
|---|---|---|
| L_algo | 0 ms | 0 ms ✓ |
| L_buf | 80 ms | **0–60 ms (mean 30)** |
| L_comp | <1 ms | ~1.06 ms per window as implemented; **<0.35 ms** achievable ([F5](#f5)) |
| **acoustic total** | ~81 ms | **~61 ms worst case, ~31 ms mean** |
| remaining for the LM within 140 ms | ~59 ms | **~79 ms worst case, ~109 ms mean** |

For `causal_la4` the same arithmetic gives 80 (L_algo) + 60 (L_buf) + ~1 = **141 ms — exactly at
the budget ceiling**, which is a tidier justification for rejecting it than the group-delay
argument currently in use.

---

## 11. Smaller findings

- **F12 — blank skipping is implemented and disabled.** `ctc_wfst_beam_search.cc:79-84` skips a
  frame when `exp(logp[0]) > blank_skip_thresh`. Default 1.0 (`params.h:42`,
  `language-model-standalone.py:809`) can never trigger, since `exp(log p) ≤ 1`. Note
  `ctc_wfst_beam_search.h:51` carries a different default (0.98) that the Python path overrides.
  Enabling it is a config change with zero retrain cost.
- **F15 — `gauss_smooth` rebuilds its kernel on every call** (`data_augmentations.py:15-31`):
  numpy allocation, `scipy.ndimage.gaussian_filter1d`, threshold, normalize, then a fresh
  host→device tensor — per batch in training, per bin in streaming. Hoisting it is free.
- **F16 — `metrics['losses']` is appended twice per validation batch** (`rnn_trainer.py:720` and
  `:760`), once as a numpy array and once as a float. `avg_loss` is numerically unaffected (both
  entries are equal) but the list is 2× the expected length.
- **Day-layer gather is a training-time cost**: `rnn_model.py:95-96` builds a `[64, 512, 512]`
  stacked weight tensor per forward (67 MB fp32) via a Python list comprehension over the batch.
  Irrelevant at inference (batch=1, fixed day) but it is real training overhead.
- **`create_attention_mask`** (`rnn_trainer.py:408-434`) is dead code — never called, and its
  `torch.where(mask, True, False)` is a no-op identity.
- **Hardware inconsistency [UNVERIFIED]**: `model_training/CLAUDE.md` says training runs on a
  RunPod A100; the brief says a single RTX 3090; `metrics.json` records only `device: cuda`.
  Compute budgeting in units of "6.4-hour runs" needs this pinned down — the three completed runs
  took 372–396 min, consistent with the 6.4 h figure, but on an unidentified device.

---

## 12. What Phase A changes about the plan

1. **The causal-normalization finding ([F1](#f1)) outranks everything else on the list.** The
   project's real-time claim is currently built on non-causally normalized inputs. One retrain
   with a causal rolling normalizer converts the paper's weakest point into its most defensible
   measurement — the true, end-to-end cost of causality, which nobody has published for this
   participant.
2. **The central experiment is confounded ([F2](#f2)).** A bandwidth-matched causal control is one
   run and it decides whether Fact A's negative result survives review.
3. **Fact A is probably not "inside seed noise" ([F9](#f9)).** The only same-config replicate in
   hand differs by 0.041 points against a 0.163-point effect. Seeds are needed to confirm a likely
   *positive* result, not to bury a null one.
4. **The LM is the entire remaining latency budget, and it is already incremental ([F4](#f4)).**
   The work is not "make it streaming" — it is "measure the incremental path, then buy back the
   accuracy the three batch stages currently provide."
5. **Headroom is ~3× larger than reported ([F5](#f5))**, and the model is launch-bound rather than
   FLOP-bound ([A3](#4-a3--architecture)) — which makes batched *N*-way ensembling and phase-staggered
   replicas far cheaper than the brief's arithmetic assumes.
6. **"Test WER" is not measurable ([F7](#f7)).** A session-held-out protocol on the val split must
   be defined and pre-registered before any number is quoted.
