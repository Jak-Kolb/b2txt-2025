# CLAUDE.md — b2txt-2025 (`causal-preprocessing`)

Repo: `Jak-Kolb/b2txt-2025`, forked from `Neuroprosthetics-Lab/nejm-brain-to-text`.
Working branch: `causal-preprocessing`. Upstream line numbers still match; edits are additive.

---

## 1. What this project is

Converting the Card et al. 2024 NEJM intracortical speech neuroprosthesis pipeline into a
**fully causal, real-time streaming decoder**, and then finding the best accuracy achievable
*subject to* a hard real-time constraint.

The original question ("what does causality cost?") is answered and the answer is ~nothing —
see §3. The live question is **"what is the causal-constrained accuracy optimum?"**

Write-up targets: ICASSP, SLT, EMBC, NeurIPS neuro tracks. Not a leaderboard entry; the
Brain-to-Text '25 competition is closed and there is no '26.

### Open decision — do not assume either way

Whether the deliverable is (a) an accuracy-vs-latency **frontier paper** or (b) a
**"best accuracy at fixed latency"** paper is **unresolved**. It changes experiment ordering
materially. The flat L=0/L=4 delta argues for (b). If a task's correct sequencing depends on
this, ask rather than picking.

---

## 2. Hard constraints — every proposal must respect these

| Constraint | Value |
|---|---|
| End-to-end latency budget | ~140 ms = `L_algo` + `L_buf` (80 ms) + `L_comp` (<60 ms) |
| Real-time factor | RTF < 1 |
| Training hardware | single RTX 3090, ~6.4 h per 120k-batch run |
| Inference RAM | 32 GB → **pruned 3-gram LM (~6–8 GB)**, not a 300 GB LM |
| Emission | streaming; no full-utterance right context anywhere |

`L_algo = 20·L` ms, where `L = smooth_lookahead` in 20 ms bins. At `L=0`, `L_algo = 0`.

**Ledger rule.** A latency claim is only real if it prices *every* stage, including LM decode
and rescoring. Acoustic-model compute is not the budget. Any proposal presented with an
acoustic-only ledger is incomplete — say so rather than accepting it.

---

## 3. Current measured state (all numbers real, do not re-derive)

**Accuracy**
- `causal_la4` (L=4, non-causal control): **10.21 %** val PER, 396 min. This is the anchor
  baseline for all frontier deltas — not the older 10.25 % figure, which was a different env.
- `causal_la0` (L=0, fully causal, `L_algo=0`): **10.04 %** val PER, 372 min.
- Δ = **−0.17 points**, i.e. within seed noise. **Headline: zero-lookahead causal smoothing
  costs no measurable accuracy.**
- Released-checkpoint reference: 2.66 % avg WER full-stack; 10.20 % greedy-CTC PER
  (acoustic-only).

**Streaming** (`model_training/benchmark/`, harness built *and* run)
- Equivalence gate **PASSED** on `causal_la0` (logit tol < 1e-3, exact collapsed-sequence match).
- Streaming RTF ≈ **0.0132** (~76× real time).
- Per-patch compute p95 ≈ **0.34 ms** against an **80 ms** cadence.
- The system is idle **~99.6 %** of every cadence window.

**The strategic consequence.** Compute is not the binding resource; the frontier is degenerate.
The unspent headroom is the lever — causal ensembling, phase-staggered replicas, larger causal
models. And **the LM/decoding stage is the actual streaming bottleneck**, not the acoustic model.
Any optimization work that targets the GRU before pricing the decoder is aimed at the wrong stage.

---

## 4. Verified ground truth — established from source, treat as settled

Re-deriving these has already cost several cycles. If you believe one is wrong, say so
explicitly and show the `file:line` — do not silently work around it.

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

**Patching** (`rnn_model.py`, `unfold`, P=14, s=4)
- **Causal** under right-edge / last-bin emission. CTC is alignment-free; nothing pins frame `i`
  to time `i·s`. The 280 ms cold-start and 80 ms cadence are real latency, but they are a
  frontier target, not a causality bug.

**Normalization**
- The released HDF5 features are **causally safe**: 0/45 sessions show a per-trial or whole-block
  z-score signature. Corroborated by Wairagkar et al. (PMC11370360, same participant T15), which
  documents a rolling past-10 s mean/std.
- Caveat for manuscript: that quote is from the voice paper, not Card b2txt verbatim. Confirm the
  exact b2txt feature-extraction line before it goes in a paper.

**Other**
- `adjusted_lens` is computed at `rnn_trainer.py:532` and `:707`.
- Raising `patch_size` or `smooth_kernel_std` to recover accuracy is **not free** — both spend
  latency budget. `smooth_kernel_std` widens the *past* tail and is near-free; future taps and
  `patch_size` are not.

---

## 5. Permanently rejected — do not re-propose

Each of these has been proposed and refuted. Re-proposing one is a signal that the reasoning
chain drifted; stop and re-read §4.

1. **Left-padding the patcher** (`F.pad(x, (P−1, 0))`). Changes `num_patches` (22→25 at T=100),
   breaks CTC frame count and checkpoint compatibility, and adds **zero** causality. Rejected
   three times.
2. **Labeling the slid-symmetric smoother as real-time.** It carries 80 ms group delay. It is a
   frontier endpoint, not a real-time configuration.
3. **Bundling independent changes into a single retrain.** Confounds the PER delta and destroys
   the run's value.
4. **Calibration-efficiency work.** Out of scope for this project.

---

## 6. Working norms

- **One variable per retrain.** Non-negotiable. A run that changes two things measures neither.
- **Pre-registered decision gates.** Before a run: state the config, the metric, the threshold,
  and what each outcome implies. Write it down *before* launching, not after seeing the number.
- **Immutable configs.** Fresh `output_dir` / `checkpoint_dir` per run. Never clobber
  `trained_models/baseline_rnn` or an existing `causal_la*`.
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
model_training/
  rnn_model.py                 GRU, day-affine layer, patching (unfold, P=14 s=4)
  rnn_trainer.py               training loop; :476 smoothing call; :532/:707 adjusted_lens
  rnn_args.yaml                config; dataset.data_transforms.smooth_lookahead
  dataset.py                   HDF5 loading, batching
  data_augmentations.py        gauss_smooth (truncated half-Gaussian, lookahead kwarg)
  evaluate_model_helpers.py    :92 smoothing at eval, reads lookahead from saved model_args
  evaluate_model.py            eval entrypoint
  trained_models/
    causal_la4/                L=4 control, 10.21 % val PER
    causal_la0/                L=0 causal, 10.04 % val PER
  benchmark/
    common.py                  checkpoint/args loading, HDF5 trial discovery, CTC collapse, RTF stats
    offline_benchmark.py       offline acoustic timing → metrics.json
    streaming_infer.py         one-bin streaming decoder, per-bin/per-patch timing
    test_equivalence.py        acceptance gate: logit tol <1e-3, exact collapsed-sequence equality
language_model/                n-gram decode / LM fusion — the real bottleneck; profile before touching
data/hdf5_data_final/          Kaggle brain-to-text-25, 45 sessions, ~13 GB
log.md                         running project log
```

---

## 8. Environment and commands

**RunPod (training) — the pod disk is ephemeral local NVMe, not a network volume.**
It has been lost twice to GPU migration. Before rebuilding, **verify on GitHub that
`causal-preprocessing` actually contains the lookahead changes** — otherwise the rebuild
retrains the non-causal baseline by mistake. `scp` checkpoints off the pod as they are written.

Rebuild: miniconda into `/workspace` → clone `causal-preprocessing` → `./setup.sh` → re-pull the
Kaggle `brain-to-text-25` data into `data/hdf5_data_final`.

```bash
# train (from model_training/)
conda activate b2txt25          # post-rebuild pods have run from base/py3.13 — check which exists
python train_model.py           # 120k batches, ~6.4 h on a 3090, seed 10
                                # PER needs no redis/LM; the LM stack is WER-only

# benchmark (from model_training/)
python benchmark/offline_benchmark.py   # → metrics.json
python benchmark/test_equivalence.py    # acceptance gate; must PASS before any streaming claim
python benchmark/streaming_infer.py     # RTF, per-patch p50/p95
```

**Local eval (M3 Pro Mac, CPU path):** requires `map_location='cpu'` and bf16 neural input.
The full LM stack does not fit in RAM here — this is part of why the deployable target is a
pruned 3-gram.

---

## 9. `log.md` style

Reverse-chronological. Terse first person. Brief `Shipped:` and `Learned:` bullets. Keep numbers
and verdicts; cut explanatory reasoning and prose elaboration. Do not narrate.

---

## 10. Remaining phases

2. Honest comparator (offline minus LLM/ensemble) + frontier scaffold.
3. Recovery arms: FastEmit / delay-penalized CTC, distillation from a bidirectional teacher,
   lookahead conv k ∈ {0,2,4}.
4. Deployable WER via pruned 3-gram.
5. Real-time replay harness.
6. Write-up.

Highest-prior open lever given the 99.6 % idle window: **causal deep ensembling with
phase-staggered replicas**, plus a word-level output-stability metric as a second contribution.

---

## 11. Response contract

Lead with the answer; justify only where it changes the conclusion. No preamble, no restatement
of the question, no closing summary. Short prose by default; bullets for 3+ discrete comparisons
or ordered steps. Assume fluency in code, math, and papers — skip the ELI5. Web-search or read
the docs for any API, library, paper, or current spec rather than reciting from training data,
and flag uncertainty and stale results explicitly.
