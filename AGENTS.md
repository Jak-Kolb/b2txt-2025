# AGENTS.md — b2txt-2025

Repo: `Jak-Kolb/b2txt-2025`, forked from `Neuroprosthetics-Lab/nejm-brain-to-text`.
Working branch: `audit/realtime-latency-frontier`. Upstream line numbers still match; edits are
additive.

## 0. `PLAN.md` is the single source of truth

**Read `brainstorm/PLAN.md` first.** It holds the consolidated change list, the four execution groups, the
test protocol, and the results ledger. Work one group at a time; do not plan from anything else.

The Phase A–E audit documents (`AUDIT`, `BUDGET`, `LITERATURE`, `CANDIDATES`, `RECOMMENDATIONS`,
`OPEN_QUESTIONS`, `RUN_LEDGER`, `RESEARCH_PROPOSALS`) are archived under `brainstorm/audit/` as
**evidence, not plans**. Open one only to check a derivation behind a number `PLAN.md` cites.
`brainstorm/audit/RUN_LEDGER.md` is superseded — do not work from its ordering.

`brainstorm/causal_audit.md` and `brainstorm/optimization_plan.md` predate the audit and are stale.
Ignore them; they have not been deleted only because nobody has confirmed they are safe to drop.

---

## 1. What this project is

Converting the Card et al. 2024 NEJM intracortical speech neuroprosthesis pipeline into a
**fully causal, real-time streaming decoder**, and then finding the best accuracy achievable
*subject to* a hard real-time constraint.

The original question ("what does causality cost?") is **not** answered — the L=0 vs L=4
comparison that appeared to answer it is confounded, and the released features turned out to be
non-causally normalized. See §3 and §4. The live questions are **"what is the causal-constrained
accuracy optimum?"** and **"what does end-to-end causality actually cost?"**

Write-up targets: ICASSP, SLT, EMBC, NeurIPS neuro tracks. Not a leaderboard entry; the
Brain-to-Text '25 competition is closed and there is no '26.

### Open decision — do not assume either way

Whether the deliverable is (a) an accuracy-vs-latency **frontier paper** or (b) a
**"best accuracy at fixed latency"** paper is **unresolved**. It changes experiment ordering
materially. The audit added a third framing — (c) a **systems-and-measurement paper** on what
real-time actually costs once the LM is priced — which is what `PLAN.md`'s ordering serves best,
and which is compatible with either (a) or (b) as the accuracy story. Still the user's call.
If a task's correct sequencing depends on it, ask rather than picking.

---

## 2. Hard constraints — every proposal must respect these

| Constraint | Value |
|---|---|
| End-to-end latency budget | ~140 ms = `L_algo` + `L_buf` (**0–60 ms, mean 30**) + `L_comp` |
| Acoustic side, worst case | **60.3 ms of 140** — leaves ~79 ms for the entire LM stage |
| LM stack, as shipped | **620–830 ms finalization = 4.4–5.9× the whole budget** |
| Real-time factor | RTF < 1 |
| Training hardware | **RTX 5070 Ti, 16 GB — verified. 7.41 GB peak, ~5.9 h per 120k run** (needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, §8) |
| Inference RAM | 32 GB → **a built, pruned 4-gram ≤ 19 GB**. The shipped 3-gram needs ~60 GB, the 5-gram ~300 GB |
| Emission | streaming; no full-utterance right context anywhere |

`L_algo = 20·L` ms, where `L = smooth_lookahead` in 20 ms bins. At `L=0`, `L_algo = 0`.
**80 ms is the update cadence, not the emission latency** — those are different quantities and
conflating them overstates `L_buf` by 20 ms and understates the LM's share of the budget.

**Ledger rule.** A latency claim is only real if it prices *every* stage, including LM decode
and rescoring. Acoustic-model compute is not the budget. Any proposal presented with an
acoustic-only ledger is incomplete — say so rather than accepting it.

---

## 3. Current measured state (all numbers real, do not re-derive)

**Accuracy**
- `causal_la4` (L=4, non-causal control): **10.21 %** val PER, 396 min. Anchor baseline.
- `causal_la0` (L=0, `L_algo=0`): **10.04 %** val PER, 372 min.
- Δ = **−0.163 points** on the last-10 mean. This is **~4× the only same-config replicate in
  hand (0.041 pts)** — so "within seed noise" is *not* what the evidence says. Seeds are needed to
  confirm a likely real effect, not to bury a null.
- **But the comparison is confounded**: truncating the kernel removes lookahead *and* narrows the
  low-pass by 37 %. See §4.
- Released-checkpoint reference: 2.66 % avg WER full-stack **[unverified — only a 1-gram exists
  locally]**; 10.20 % greedy-CTC PER (acoustic-only).
- **Run-to-run noise floor: 0.041 PER pts.** Report deltas against this, not against zero.
- **`best_val_PER` is a min over 61 correlated evaluations**, ~0.04 pts optimistically biased.
  Record the last-10 mean alongside it.

**Streaming** (`model_training/benchmark/`, harness built *and* run)
- Equivalence gate **PASSED** on `causal_la0` (logit tol < 1e-3, exact collapsed-sequence match).
- Streaming RTF ≈ 0.0132 — but **~70 % of that is Python overhead**, not the model. True
  per-window compute is ~0.37 ms achievable against 1.06 ms measured.
  *(Those two figures are Mac-CPU. On the 5070 Ti after C1: RTF **0.0114**, per-bin mean 0.229 /
  p95 0.697 ms, per-patch p50 0.574 / p95 0.649 ms. Do not compare the two machines' absolutes.)*
- Per-patch compute p95 ≈ **0.34 ms** GPU / **1.44 ms** CPU against an **80 ms** cadence.
- **C1 landed 2026-08-06**: the smoother is now a circular window + pre-rotated kernel, one matmul
  per bin. Same-machine A/B: **−47 %** CUDA at K=5 (the deployed L=0), −86 % at K=9, −56 %/−80 % on
  CPU. It is **not** bit-identical — a matmul reduction does not sum in kernel order — but the
  difference is 2.4e-07 and the equivalence gate passes at unchanged tolerance.
- Batched N=10 ensemble: **7.05 ms on a laptop CPU**, flat from N=4 to N=16 — the model is
  kernel-launch-bound, not FLOP-bound (measured forward is 175× its arithmetic time).

**Measured by the audit — new, and each one killed or created work**
- **Excess learned emission delay: +4.64 ms** over 13,030 paired tokens, after subtracting the
  smoother's predicted group delay. Kills the entire delay-penalized-CTC family (38.4 GPU-h).
- **Median time-to-first-token: 3,380 ms.** The 280 ms cold start is fully absorbed by the
  pre-speech period. Kills reduced-patch work (12.8 GPU-h).
- **71.83 % of frames carry p(blank) > 0.999.** Blank skipping is implemented and disabled.
- **Mean posterior entropy is 0.9 % of maximum**, median frame numerically one-hot. The model is
  confidently wrong rather than uncertain — so gate adaptive compute on *n-gram* confidence, never
  on acoustic entropy. Also makes calibration a free untouched lever.
- **The acoustic decoder is provably flicker-free**: 0 revisions over 41,039 tokens, because the
  online collapse rule only ever appends. **100 % of user-visible flicker is LM-side.**

**The strategic consequence.** Compute is not the binding resource — the acoustic model is 40×
under budget while the LM is 4–6× over it. **The LM/decoding stage is the entire remaining
budget.** Any optimization that targets the GRU before pricing the decoder is aimed at the wrong
stage. Spending the idle headroom on more acoustic models improves the stage that is already fine.

---

## 4. Verified ground truth — established from source

Re-deriving these has already cost several cycles. If you believe one is wrong, say so
explicitly and show the `file:line` — do not silently work around it.

**But this section is not immune.** The normalization entry below was stated here as settled for
months and was wrong; measurement overturned it. Treat these as the current best reading of the
source, not as unfalsifiable. A measurement beats an entry in this file — when one does, correct
the entry in the same commit rather than leaving both versions in play.

**Smoothing**
- Upstream kernel: symmetric 9-tap Gaussian (±4 bins / ±80 ms). Train used `padding='same'`;
  eval used `padding='valid'`. `'valid'` only trims boundary frames — it does **not** one-side
  the kernel and is **not** causal. The original ~10.25 % PER is a non-causal number.
- Current implementation: parameterized **truncated half-Gaussian**,
  `gauss_smooth(..., lookahead=L)`. Keeps offsets `−p..+L`, peak always on the current bin,
  renormalized, asymmetric pad `(p, L)`, valid conv, length-preserving `T→T` (so `adjusted_lens`
  never changes). `L=4` is bit-identical to symmetric `'same'` (max diff 0.0); `L=0` is fully
  causal with zero group delay — verified weights `[0.339, 0.299, 0.206, 0.110, 0.046]`, zero
  response before `t0`.
- **The subtlety that keeps getting missed:** left-padding `K−1` while keeping the *symmetric*
  kernel is causal but **not latency-free** — it converts 80 ms of future leak into 80 ms of
  group delay (peak lands at `t−4`). Sliding the pad is not truncating the kernel. Only
  truncation reaches true zero delay.
- Wiring: `smooth_lookahead` in `rnn_args.yaml` → `rnn_trainer.py:476` →
  `evaluate_model_helpers.py:92`, which reads it from the saved `model_args` so train/eval
  consistency is structural rather than conventional.
- **`smooth_lookahead` is not a clean causality knob.** Truncating changes three things at once:
  L=4→L=0 removes 80 ms of lookahead **and narrows effective σ by 37 %** (1.852 → 1.161 bins)
  **and raises peak weight 66 %** (0.204 → 0.339). So "causality is free" is not established —
  what is established is "this particular truncation is free-or-better." The clean control is a
  **bandwidth-matched causal kernel** (σ_eff ≈ 1.852 at `lookahead=0`); one run resolves it.
  This is `PLAN.md` R-D1.

**Patching** (`rnn_model.py`, `unfold`, P=14, s=4)
- **Causal** under right-edge / last-bin emission. CTC is alignment-free; nothing pins frame `i`
  to time `i·s`. The 280 ms cold start is real but irrelevant — it is absorbed by a 3,380 ms
  median pre-speech period.
- `random_cut: 3` draws `{0,1,2}`, so training already covers **3 of the 4 patch phases**.

**Normalization — CORRECTED 2026-08-06. The previous entry here was wrong.**
- The released HDF5 features are **whole-block z-scored, i.e. NON-CAUSAL**. Every feature at time
  *t* was offset and scaled by statistics computed over up to ~19 minutes of *future* recording.
- The earlier "0/45 sessions leak" conclusion came from `analyses/check_hdf5_causality.py:76`,
  which thresholds a **`max` over 512 channels** — one badly-behaved channel fails a whole session,
  so the test false-negatives. The decisive evidence is a 1/√W scaling test: sub-window means shrink
  as 1/√W exactly as pure subsampling noise would, matching prediction within 1–26 % across four
  sessions spanning two years.
- **Consequence:** the 10.04 % / 10.21 % figures are upper bounds obtained with leaked
  normalization statistics, and the streaming-equivalence gate certifies streaming reproduces
  offline *on inputs that are themselves non-causal*.
- **The fix is exact and needs no raw data**: applying a causal rolling normalizer on top of
  block-z-scored data cancels the block constants identically, recovering
  `(x_raw − μ_roll,raw)/σ_roll,raw`. Wairagkar et al. use exactly this rolling-past-10 s scheme on
  this same participant. `PLAN.md` C8 verifies the algebra at zero cost; R-D2/R-D3 measure the price.
- Manuscript caveat: the Wairagkar quote is from the **voice** paper, not Card b2txt verbatim.
  Confirm the exact b2txt feature-extraction line before it goes in a paper.

**Evaluation**
- **`data_test.hdf5` contains only `input_features`** — no `seq_class_ids`, no `transcription`, no
  labels of any kind. With the leaderboard closed, **test WER is unobtainable**. Never quote a
  "test WER"; the honest framing is *"held-out sessions from the public validation split."*
- Split is fixed on disk from separate files; `test_percentage: 0.1` in the config is **dead**.
  train 45 sessions / 8,072 trials; val 41 sessions / 1,426 trials.
- PER is **micro-averaged** (`total_edit_distance / total_seq_length`), so long sentences dominate.

**Other**
- `adjusted_lens` is computed at `rnn_trainer.py:532` and `:707`.
- **`zero_infinity=False` (`rnn_trainer.py:242`) + `error_if_nonfinite=True` (`:554`) is a hard
  crash** the moment any trial has `adjusted_lens < phone_seq_lens`. Set `zero_infinity=True`
  before masking, larger `patch_size`, or larger `random_cut`.
- The incremental WFST beam search **already exists and works** (`ctc_wfst_beam_search.cc:98`
  advances one frame at a time; `:113` extracts a partial best path). What makes the reference
  stack batch is `evaluate_model.py:237-250`, a harness choice. The question is not "can it be made
  incremental" but "what does it cost per frame."
- `language-model-standalone.py:486-496` passes **literal** `max_active`/`beam`/`lattice_beam`
  values; those CLI flags are dead at startup. Fix before sweeping them.
- Raising `patch_size` or `smooth_kernel_std` to recover accuracy is **not free** — both spend
  latency budget. `smooth_kernel_std` widens the *past* tail and is near-free; future taps and
  `patch_size` are not.

---

## 5. Permanently rejected — do not re-propose

Each of these has been proposed and refuted. Re-proposing one is a signal that the reasoning
chain drifted; stop and re-read §4.

1. **Left-padding the patcher** (`F.pad(x, (P−1, 0))`). Changes `num_patches` (22→25 at T=100),
   breaks CTC frame count and checkpoint compatibility, and adds **zero** causality. Rejected
   four times.
2. **Labeling the slid-symmetric smoother as real-time.** It carries 80 ms group delay. Sliding
   the pad is not truncating the kernel. It is a frontier endpoint, not a real-time configuration.
3. **Calibration-efficiency work.** Out of scope for this project.
4. **Delay-penalized CTC, Bayes-Risk CTC, Peak-First CTC, TrimTail, Align-With-Purpose.** Killed by
   measurement: excess learned emission delay is **+4.64 ms**, 6 % of one frame. There is nothing
   to remove. FastEmit and delay-penalized transducers are additionally transducer-only, and there
   is no transducer here.
5. **Reduced `patch_size` to cut cold start.** Killed by measurement: median time-to-first-token is
   3,380 ms against a 280 ms fill.
6. **Deep ensemble at N=10.** 45 % of budget for a sublinear return, zero novelty, and it improves
   the stage already 40× under budget. The published gain was measured at a **12× higher baseline
   error**. N=3 scores 3× better on the same curve; keep it as a cheap diversity probe only.
7. **Wider / deeper GRU on the "we have idle compute" argument.** The compute premise is true and
   irrelevant — the binding constraint is 45 sessions from one participant, and the closest
   published causal result won with 83 % fewer parameters.

Full excluded list with reasons: `PLAN.md` §9.

---

## 6. Working norms

- **Grouped batching, not one-variable-per-run.** *(Changed 2026-08-06 — the previous rule was
  "one variable per retrain, non-negotiable.")* Changes are bundled into the four groups in
  `PLAN.md` and tested one group at a time: 6 training runs instead of 15. Bundles are built only
  from **mechanistically distinct, same-signed** changes, so a bundle that wins wins as a block and
  a bundle that loses is bisectable. Never bundle changes that push in opposite directions — a
  neutral result from opposed changes is uninterpretable.
- **Three carve-outs where single-variable still holds, and they are not negotiable:**
  1. **Validity runs** (bandwidth-matched control, causal normalization, seed replication). Their
     entire deliverable *is* an isolated number; a bundled version measures nothing.
  2. **Changes with a documented instability risk** (e.g. Adam `epsilon` × `lr_max`, which are
     coupled). These get their own paired sweep, not a slot in a bundle.
  3. **Contested-direction changes.** If the sign of ΔPER is genuinely in doubt, it does not go in
     a bundle of expected-positives.
- **Smoke run before every retrain.** 2,000 batches, ~7 min. Catches OOM, NaN, divergence and shape
  errors for ~2 % of a run's cost. This is what makes aggressive bundling safe; skipping it is how
  6.4 hours get lost.
- **Bisect, don't re-run.** When a bundle loses, split it in two weighted by prior strength and run
  the strong half. Two runs maximum resolves a losing bundle of three to five changes.
- **Attribution is deferred, not deleted.** Leave-one-out ablations cost one run each; spend them
  only after a bundle wins, and only on changes a reviewer will challenge. Do not claim
  per-change attribution for a bundled result in a manuscript.
- **Pre-registered decision gates.** Before a run: state the config, the metric, the threshold,
  and what each outcome implies. Write it down *before* launching, not after seeing the number.
  `PLAN.md` carries a gate for every group; if a task needs a new one, write it first.
- **Immutable configs.** Fresh `output_dir` / `checkpoint_dir` per run, named as in `PLAN.md` §8.3.
  Never clobber `trained_models/baseline_rnn`, an existing `causal_la*`, or a completed group.
- **Record the last-10 mean, not just `best`.** `best_val_PER` is an order statistic over 61
  correlated evaluations.
- **Compare against the 0.041-pt noise floor**, never against zero. Under ~0.08 pts is not a result.
- **Source-first verification.** Grep the source before asserting any pipeline property. Cite
  `file:line`. Never assert a property from memory of a paper or from the shape of the code.
- **Push back.** If a request rests on a wrong premise, fix the premise first. Do not implement a
  plausible-looking version of a bad idea, and do not validate an incorrect assumption to be
  agreeable.
- **Traceable deltas only.** Every claimed PER improvement needs a mechanism and a magnitude.
  "Should help" is not a proposal; cut it before sequencing.
- **No execution without approval.** No training launches, no pipeline edits, no long-running
  experiments without an explicit go.
- **Flag stale or unverified numbers.** Figures pulled from research reports (LightBeam, B2T '25
  winner details, exact ms claims) need primary-source verification before manuscript use. Label
  anything estimated as `estimated`.

---

## 7. Repo map

```
AGENTS.md                      this file. CLAUDE.md is a one-line `@AGENTS.md` import so Claude
                               Code picks it up — it reads CLAUDE.md, not AGENTS.md
README.md                      upstream paper README, plus a fork notice at the top
RESULTS.md                     PUBLIC-FACING results summary: the 42.42 -> 6.89 % chain, the
                               accuracy/latency frontier, and what the measurements overturned
brainstorm/                    planning + exploratory material (tracked, ships publicly)
  PLAN.md                      ← START HERE. Change list, 4 groups, test protocol, results ledger
  audit/                       Phase A–E evidence archive (see its README); RUN_LEDGER is superseded
  optimization_plan.md         predates the audit — STALE, kept for provenance only
  causal_audit.md              predates the audit — STALE, kept for provenance only
model_training/
  rnn_model.py                 GRU, day-affine layer, patching (unfold, P=14 s=4)
                               :65-72 fused nn.GRU(num_layers=5)  :95-99 day layer  :106-119 unfold
  rnn_trainer.py               training loop; :242 CTCLoss; :436-485 transform_data;
                               :476 smoothing call; :532/:707 adjusted_lens; :554 grad clip
  rnn_args.yaml                config; :26 save_all_val_steps; :46-48 wd/eps/seed; :67 random_cut;
                               :70-71 smooth_kernel_std / smooth_lookahead
  dataset.py                   HDF5 loading, batching; :130 input_features read
  data_augmentations.py        gauss_smooth (truncated half-Gaussian, lookahead kwarg)
  evaluate_model_helpers.py    :79-83 logit rearrange; :92 smoothing at eval (reads saved args)
  evaluate_model.py            eval entrypoint; :237-250 the batch xadd that makes the ref harness batch
  trained_models/              GITIGNORED — runs THIS machine is producing; may be mid-training
    pretrained_baseline/       upstream release, non-causal (no smooth_lookahead key)
  causal_normalize.py          C8 rolling normalizer + the affine-invariance identity check
  splits.py                    C7 frozen val-dev / val-test session split
  benchmark/
    common.py                  checkpoint/args loading, HDF5 discovery, CTC collapse, RTF stats
    offline_benchmark.py       offline acoustic timing → metrics.json
    streaming_infer.py         one-bin streaming decoder (C1 circular window, C3 bounded buffers)
    test_equivalence.py        acceptance gate: logit tol <1e-3, exact collapsed-sequence equality
    stability.py               word-stability, posterior stats, emission delay, append-only check
    stream_lm.py               C13 incremental WFST driver; subcommands export/decode/sweep/temp/
                               nbest/rescore. THE measurement surface for everything LM-side
    indomain_lm.py             C28-A trigram over the TRAIN transcriptions (the +17-pt result)
    phase_ensemble.py          C9 per-phase PER; C16 interleaved / phase-averaged merges
    neural_rescore.py          C29 n-best rescoring with any HF causal LM. Sweep GAMMA (the 4-gram
                               weight) — pinned at 1 it double-counts the LM and hides the gains
    finetune_rescorer.py       in-domain adaptation of a rescorer (measured, REJECTED — §1.12)
results/                       GITIGNORED — authenticated reference checkpoints + all measurements
    causal_la0/ causal_la4/    the real checkpoints (10.04 % / 10.21 %), recovered from the Mac.
                               Benchmark defaults point HERE, not at trained_models/
language_model/                n-gram decode / LM fusion — the actual bottleneck
  build_tlg.sh                 C28 entry point: wires SRILM + the six Kaldi FST binaries that
                               setup_lm.sh does NOT build, then counts/prunes/composes a TLG
  make_tlg_from_arpa.sh        compose TLG from an existing ARPA, reusing L.fst/T.fst
  fetch_corpus.py              stream + normalize an LM corpus (numbers spelled out, URLs dropped)
  interpolate_lm.sh            LM interpolation w/ held-out lambda selection (measured, REJECTED)
  examples/speech/s0/          the LM recipe. run.sh -> local/build_lm.sh (GT*MIN cutoffs are now
                               env-overridable) -> tools/fst/make_tlg.sh. UPPERCASE end to end
  language-model-standalone.py :486-496 decoder params (C12: now honour the CLI); :769-785 incremental
  runtime/.../ctc_wfst_beam_search.cc  :79-84 blank skip (disabled); :98 per-frame advance; :113 partial
  pretrained_language_models/openwebtext_1gram_lm_sil/   the ONLY LM artifact present (13.4 MB)
data/hdf5_data_final/          Kaggle brain-to-text-25, 45 sessions, ~13 GB — GITIGNORED
log.md                         running project log — GITIGNORED
```

---

## 8. Environment and commands

### Local machine — set up and verified working 2026-08-06

This is the primary training machine now. **It runs the pipeline at the stock config.**

| | |
|---|---|
| GPU | **RTX 5070 Ti, 16,303 MiB, sm_120** (Blackwell) — not the 3090 assumed earlier |
| torch | **2.13.0+cu130** in `.venv`, `sm_120` in `get_arch_list()`. `setup.sh`'s `cu126` pin does **not** carry Blackwell kernels — never install from cu126 here |
| **Python env** | **`.venv/` (uv, py 3.10.20) — use this for everything training/benchmark side.** Reproduces `setup.sh` exactly (all 17 pins, incl. `numpy==2.1.2`) |
| CPU / RAM | Core Ultra 7 270K Plus, 22 cores / **22 GiB visible to WSL2** (32 GB host, capped in `.wslconfig`) |
| Data | `data/hdf5_data_final/`, 45 sessions, 13 GB, extracted |
| Checkpoint | `model_training/trained_models/pretrained_baseline/` — upstream release, **non-causal** (no `smooth_lookahead` key). Enough to run all of Group 1; `causal_la0` still needs one retrain |
| `lm_decoder` | needs **cmake 3.x** (`uv tool install "cmake<4"`); cmake 4 fails on vendored gflags. `setup_lm.sh` prints "Setup complete" even when the build fails — verify with `import lm_decoder` |
| NPU | **not exposed to WSL2, and not useful.** The model is launch-bound (forward is 175× its arithmetic time); the real bottleneck is WFST pointer-chasing, the workload NPUs are worst at. Do not spend time on it |

**Measured, replacing the inherited estimates:**

| | |
|---|---|
| Peak training VRAM, batch 64 | allocated **7.41 GB of 16** — fits with ~9 GB spare. **Do not reduce `batch_size`** |
| Peak validation VRAM, batch 64 | 2.72 GB |
| **s/batch** | **0.169 steady** — *only* with the env var below; **~1.5 without** |
| **120k-batch run** | **~5.9 h** (5.6 h train + 0.25 h validations) |

```bash
# train_model.py sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True itself (see below).
# The benchmark/eval entrypoints do NOT — export it manually for those:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# training / benchmarks — use .venv, NOT conda
cd model_training && ../.venv/bin/python train_model.py

# long runs: always detach, so a killed shell cannot orphan them holding VRAM
setsid nohup ../.venv/bin/python -u train_model.py > run.log 2>&1 < /dev/null &
```

**Why the env var is mandatory.** Every batch pads to the longest of its 64 trials, so T varies
continuously (mean 875, max 2475 bins). The default caching allocator hoards a differently-sized
freed block per shape and its **reservation reached 22.87 GB on a 16.3 GB card** — at which point
WSL2's WDDM manager silently pages GPU memory over PCIe instead of OOM-ing. Measured on 300 real
batches: **0.169 s/batch with the var, ~1.5 s/batch without** (reserved 7.58 GB vs 22.87 GB).
Peak *allocated* is 7.41 GB either way, so VRAM was never the constraint — reservation was.

**Do not trust the trainer's logged `time:` as throughput.** `train_step_duration`
(`rnn_trainer.py:562`) stops *before* `loss.item()` (`:563`), the first call that synchronizes with
the GPU, so it measures kernel *launch* time. It reads a flat ~0.13 s even while wall time swings
7×. Time with `torch.cuda.synchronize()` or from log timestamps.

**conda exists for exactly one purpose: `setup_lm.sh`.** That script hard-requires it
(`conda info --base`; `conda create -n b2txt25_lm python=3.9`) and **python3.9 is not on this
system**. It builds a C++ extension against CMake/OpenFST/Kaldi — the fiddliest step in the
project — so it runs as written rather than being ported to uv. **Do not create a conda env for
training.** One was created and deleted on 2026-08-06 as redundant: `.venv` already existed, with a
newer working torch and correct pins.

**Orphaned-process trap — this cost a debugging cycle.** A training process orphaned by a killed
shell **keeps running and keeps its VRAM**. Three stacked orphans took the GPU to 15.8 GB and
produced a CUDA failure inside the cuDNN GRU that reads exactly like an OOM. Before any run:
`pgrep -af train_model.py` and `nvidia-smi`. Launch long runs with `setsid nohup … &`. Note that
`pkill -f <pattern>` will kill its own shell if the pattern appears in that shell's command line —
use `'train_mode[l]\.py'`.

**RunPod (training) — the pod disk is ephemeral local NVMe, not a network volume.**
It has been lost twice to GPU migration. Before rebuilding, **verify on GitHub that
`causal-preprocessing` actually contains the lookahead changes** — otherwise the rebuild
retrains the non-causal baseline by mistake. `scp` checkpoints off the pod as they are written.

Rebuild: miniconda into `/workspace` → clone `causal-preprocessing` → `./setup.sh` → re-pull the
Kaggle `brain-to-text-25` data into `data/hdf5_data_final`.

*(RunPod only — on the local machine use `.venv`, see above. Do not copy these lines locally.)*

```bash
# train (from model_training/)
conda activate b2txt25          # post-rebuild pods have run from base/py3.13 — check which exists
python train_model.py           # 120k batches, ~5.9 h locally / ~6.4 h on the old device, seed 10
                                # PER needs no redis/LM; the LM stack is WER-only

# benchmark (from model_training/)
python benchmark/offline_benchmark.py   # → metrics.json
python benchmark/test_equivalence.py    # acceptance gate; must PASS before any streaming claim
python benchmark/streaming_infer.py     # RTF, per-patch p50/p95
```

**Local eval (M3 Pro Mac, CPU path):** requires `map_location='cpu'` and bf16 neural input.
This is where the Phase A–E audit measurements were taken. The full LM stack does not fit in RAM
here — part of why the deployable target is a pruned 4-gram (≤ 19 GB), not the shipped 3-gram
(~60 GB).

---

## 9. `log.md` style

Reverse-chronological. Terse first person. Brief `Shipped:` and `Learned:` bullets. Keep numbers
and verdicts; cut explanatory reasoning and prose elaboration. Do not narrate.

---

## 10. Execution groups — details in `PLAN.md`

| Group | What | Runs | Gate |
|---|---|---|---|
| **0** | Prerequisites: extract data, build env, restore `causal_la0`, pin VRAM | 0–1 | **DONE** — original `causal_la0`/`causal_la4` checkpoints recovered from the Mac into `results/`; no retrain needed |
| **1** | Instrument the pipeline; build the incremental LM decoder; harvest the free decode gains | **0** | **DONE 2026-08-06** — p95 0.712 ms vs 79 ms. See `PLAN.md` §1.4–1.9 |
| **2** | Regularization block: time masking, channel masking, `random_cut=4` | **1** | val PER ≤ 9.75 % |
| **3** | Objective block: label smoothing, intermediate CTC, day-layer reg, checkpoint averaging | **1** | held-out WER −0.30 pts |
| **4** | Validity (bandwidth control, causal normalization ×2, seed) + deployable stack (4-gram, fusion, endpointing, adaptive gate) | **4** | H2 / R-D1 both publishable either way |

**6 training runs, ~38 GPU-hours**, against the 15-run / 108.8-hour one-variable ledger it replaces.
Group 1 costs zero training runs and is where most of the predicted WER gain lives.

**Group 1 is COMPLETE as of 2026-08-06, at 0 training runs.** All gates evaluated in `PLAN.md`
§1.9. What it established, in order of how much it changes the plan:

1. **The LM's lexical weakness is worth ~17 WER points and everything else is worth fractions of
   one.** A trigram over just the 8,072 training transcriptions re-ranks the 100-best from
   36.04 % → 18.33 % (unseen-sentence subset 36.24 → 18.83 %). **C28 is the whole game.**
2. **Temperature is worth −0.05 pts** and **phase merging costs 0.46–0.69 pts.** Both were expected
   to help. The acoustic and decode sides are done being optimized cheaply.
3. **There is no accuracy/latency frontier on the decode axis** — the entire sweep lives at
   0.24–2.18 ms p95 against a 79 ms budget. The trade only exists on frame rate (C16a) and LM size.

The old "highest-prior lever: causal deep ensembling with phase-staggered replicas" has been
**split**. Phase staggering is C9/C16 and is **measured**: it is the only mechanism that converts
idle compute into *latency* (L_buf 60 → 15 ms), and it **costs +0.46 WER pts** to do so — a
frontier endpoint, not an improvement. Its accuracy half (phase-averaging) is declined outright:
+0.69 pts for nothing. Deep ensembling at N=10 is declined (§5.6). The word-level output-stability
metric survives as the second contribution, but it is only measurable against the LM — the acoustic
decoder is flicker-free by construction.

---

## 11. Response contract

Lead with the answer; justify only where it changes the conclusion. No preamble, no restatement
of the question, no closing summary. Short prose by default; bullets for 3+ discrete comparisons
or ordered steps. Assume fluency in code, math, and papers — skip the ELI5. Web-search or read
the docs for any API, library, paper, or current spec rather than reciting from training data,
and flag uncertainty and stale results explicitly.
