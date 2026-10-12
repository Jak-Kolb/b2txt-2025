# Analysis plan: anchor runs (written 2026-10-11, before any anchor result exists)

## Purpose

Re-establish the causal vs bidirectional comparison under a protocol suited to screening, and
measure the two quantities needed to read every later screening result: seed-only WER variation
and whether the bidirectional gain is a capacity effect.

## Protocol (identical for every anchor)

- la0 recipe, with a complete 20k-batch schedule: `num_training_batches = lr_decay_steps =
  lr_decay_steps_day = 20000` (cosine to the minimum learning rate, 1,000 warmup batches).
- Evaluation: the fixed final snapshot (batch 19999), chosen before training. No checkpoint
  selection on validation data.
- Decoding:
  - val-dev full (1,253 trials, 35 sessions, 8,277 words), gen4g, blank penalty 90, dataset trial
    ends (oracle endpoints).
  - Acoustic scales 0.25, 0.3, 0.4, 0.5 and 0.6 for every model: an equal tuning budget.
  - Report both the fixed common setting (0.4) and each model's best grid point.
- Uncertainty: paired session-cluster bootstrap (10,000 resamples). This is session-sampling
  uncertainty, not seed variability.

## Runs

| Name | Model | Seeds (model / batch order) |
|---|---|---|
| anc_causal_s10 | causal 5x768 (32.5M shared params) | 10 / 1 (la0's) |
| anc_bi768_s10 | bidirectional 5x768 (79.1M) | 10 / 1 |
| anc_bi448_s10 | bidirectional 5x448 (35.0M, parameter-matched to la0) | 10 / 1 |
| anc_causal_s11 | causal 5x768 | 11 / 2 |

## How results will be read

1. **Seed yardstick:** |WER(anc_causal_s11) − WER(anc_causal_s10)| at 0.4 and at the best grid
   point. This is one replicate, a rough scale for seed variation, not a noise distribution.
2. **Future context at matched size:** anc_bi448_s10 vs anc_causal_s10.
   - A gain larger than about twice the seed yardstick, with a session interval excluding zero, is
     evidence that whole-sentence context helps beyond capacity.
3. **Capacity:** anc_bi768_s10 vs anc_bi448_s10.
   - A difference within the seed yardstick means capacity is not the driver, and either
     bidirectional model can serve as the ceiling in the look-ahead study.
4. **Protocol fidelity:** anc_causal_s10 at 0.4 vs fully trained la0 (8.51%, same decode settings).
   - Reported as context for how far the short protocol sits from full training.
   - No threshold: it informs, but does not decide, whether later screening uses this protocol.
5. **Agreement with the earlier 20k-prefix result:** the earlier stopped-at-20k comparison (7.90%
   vs 9.76%, −1.86 points) is a separate data point. If the complete-schedule gap differs, report
   both and treat the complete schedule as primary.
6. **Secondary:** seen/unseen WER, greedy PER, and the per-position and per-source breakdown
   (learning/bidir_gain.py) for the matched-size pair.

## What these runs cannot establish

- **Not a live result:** the bidirectional models are offline-only.
- **Not the gain at bounded latency:** that is the job of the look-ahead study.
- **Seed variation of the bidirectional model itself:** only one seed is planned.
- **Generalization beyond the exposed val-dev split or this participant:** the decode settings
  are tuned on val-dev itself.

## Addendum (2026-10-11, 23:40 UTC): overnight runs, written after the anchor results and before these runs

Anchor results that motivated these runs (results/harness/queue/20261011T170834Z_anchors/summary.md):
- Seed yardstick: 0.07–0.13 WER points.
- bi448 vs causal: −0.59 to −0.65, with a session interval touching zero.
- bi768 vs bi448: −0.48 [−0.90, −0.05].
- Causal at 20k (8.51–8.64%) matches fully trained la0 (8.51%).

Overnight runs, same protocol and decode grid as the anchors:

| Name | Model | Seeds |
|---|---|---|
| anc_causal1344_s10 | causal 5x1344 (77.8M) | 10/1 |
| anc_bi448_s11 | bidirectional 5x448 | 11/2 |
| anc_causal_s12 | causal 5x768 | 12/3 |
| anc_causal_tmask_s10 | causal 5x768 + published per-trial time masking (20 spans, ≤7.5% each, zeroed bins) | 10/1 |
| anc_causal1344_s11 | causal 5x1344 | 11/2 |
| anc_causal1024_s10 | causal 5x1024 (about 50M) | 10/1 |

How they will be read:

- **Capacity without latency:**
  - causal1344 vs causal at both seed pairs.
  - A gain beyond about twice the (now three-seed) yardstick, consistent across both pairs, with
    session intervals excluding zero, means extra capacity helps the live model.
  - causal1344 vs bi768 then compares future context at matched (large) size.
  - causal1024 places a middle point on the curve.
- **Future context at matched size:** the bi448 vs causal contrast at seeds 11/2 (anc_bi448_s11 vs
  anc_causal_s11), alongside the seed-10 contrast. Both pairs showing the same sign and roughly the
  same size, beyond the yardstick, strengthens the suggestive result. A shrinking or flipping
  contrast weakens it.
- **Seed yardstick:** pairwise differences among causal seeds 10, 11 and 12.
- **Masking:** tmask vs causal_s10, judged against the same yardstick. One seed only, so read it as
  screening.

## Addendum (2026-10-12): bounded-future runs, written after the overnight results and before these runs

Overnight results that motivate them (results/harness/queue/20261011T234231Z_overnight/summary.md),
WER at acoustic scale 0.4:

- **Future context helps by about 0.7 points:**
  - bi448 vs causal: −0.65 and −0.72 over two seed pairs.
  - bi768 vs causal1344, at matched large size: −0.71 [−1.21, −0.22].
- **Size helps the causal model by about 0.4 points:** causal1344 vs causal, −0.41 and −0.42.
- **Seed yardstick:** at most 0.13 among three causal seeds.
- **Published per-trial masking:** +5.17, far worse.

**Look-ahead family** (code: model.lookahead_frames / lookahead_units in model_training/rnn_model.py):
- la0's causal 5x768 GRU, plus a 256-unit GRU run backward over the causal outputs of frames
  t..t+R, restarted at every frame.
- 33.3M shared parameters.
- R = 0, 2, 6 and 18 frames (0, 160, 480 and 1,440 ms). R = 0 is the capacity control.
- Same protocol, seeds (10/1) and decode grid as the anchors.

**Sliding-window family**, to follow once implemented and smoke-tested:
- A bidirectional 5x448 GRU rerun over frames t−P..t+R at every step, outputting at t.
- (P, R) = (0, 2), (6, 2) and (6, 6).

**How they will be read** (WER at 0.4, plus each model's best grid point):

1. **Future, at identical parameters:** la_rR vs la_r0. The seed yardstick is about 0.13, so a
   one-seed difference of about 0.3 points or more is readable; smaller ones are not.
2. **Share of the ceiling recovered:**
   (WER(la_r0) − WER(la_rR)) / (WER(la_r0) − WER(bi448)), using bi448_s10 (same seeds) as the
   matched-size whole-sentence reference. Reported with the caveat that the ceiling itself is
   only about 0.7 points.
3. **Module effect without future:** la_r0 vs anc_causal_s10.
4. **Long past memory:**
   - win_p0_r2 vs la_r2 (same 160 ms future; 280 ms vs whole-trial past).
   - win_p6_r2 vs both.
5. **Best design within a 500 ms budget:** win_p6_r6 vs la_r6.
6. **Second seeds** only for points whose one-seed result is interesting (within about 0.3 points of
   a decision-relevant comparison).

**Limits:**
- These give final-text WER only. Live-revision and commitment timing for these models need the
  harness timestamps shifted by R frames, which is not yet implemented.
