# Transformer tests: what we are changing and why

Written 2026-11-10. **Nothing in this document has been run on the GPU yet.** Unless stated otherwise, every number is
sentence-end word error rate (WER) with the 4-gram decoder on val-dev (1,253 validation sentences, 35 sessions), from
one training seed per model. Val-dev is exploratory: there is no independent test set. Launch commands are in
`brainstorm/PLAN.md`; full result tables are in `current_knowledge.md` section 4.8.

## Short version

- **The problem.** Our best Transformer scores 9.15% WER; our best comparable GRU scores 7.77%. The published
  LightBeam causal Transformer, trained on this same dataset, beats its GRU baseline by 19%.
- **The finding.** Our 2026-11-03 "faithful" LightBeam run was not faithful. Reading their public code found about
  ten differences from what we trained. Most of them leave our model less regularized than theirs, which matches
  what we measured: it memorizes the training set and is overconfident.
- **The test.** Three trainings of about 100 minutes each: an exact-recipe control, a version that fits our live
  pipeline, and that version plus a consistency loss aimed at the overconfidence.
- **What I expect.** An improvement on 9.15% is likely. Reaching the 7.0% that their numbers imply is uncertain: the
  fixes are untested here, bundled, and judged on one seed. A Transformer that still trails the GRU is useful anyway
  as an ensemble member.

## 1. The problem

| Model | Phoneme error (PER) | Best WER (acoustic scale) | Calibrated CTC loss |
|---|---|---|---|
| GRU, 5x1344, 160 ms look-ahead (`G15`) | 9.85 | **7.77** (0.4) | 0.367 |
| Our own Transformer design, shared input layer (`X6`, averaged) | 9.40 | 8.67 (0.4) | 0.353 |
| LightBeam-recipe Transformer, 2026-11-03 (`Y4`, averaged) | **8.82** | 9.15 (0.2) | 0.442 |

"Acoustic scale" is how much the decoder trusts the acoustic model against the language model. "Calibrated CTC loss"
is the model's loss on val-dev after the best single temperature is applied, in nats per phoneme; lower means its
probabilities are more useful to a decoder.

The 2026-11-03 run has three symptoms:

1. **Best phoneme error, worst word error.** Its single best guess per frame is right more often than any other
   model's, but the decoder cannot make use of its probabilities.
2. **Overconfidence.** On frames where it emits a phoneme, it puts an average of 0.974 probability on its top
   choice. Its loss is lowest at temperature 3, and its best acoustic scale is 0.2, half the GRU's.
3. **Memorization.** Its validation loss is lowest at batch 5,000 (21) and rises to 34 by the end of training, while
   its phoneme error keeps improving.

Overconfidence costs words because the decoder weighs acoustic evidence against the language model. When the acoustic
model gives nearly all its probability to one phoneme per frame, a wrong phoneme cannot be overruled unless the
acoustic scale is turned down, and turning it down discards the correct evidence too. On our earlier Transformer the
language model fixed 65% of the acoustic word errors against 71% for the GRU, and the correct sentence was missing
from the 100-best list more often.

### Why keep working on the Transformer

- **The published result is large and on our data.** Their repository's own validation logs (10 seeds each, same
  decoder for both models) give 5.09% for the causal Transformer and 6.23-6.36% for the baseline GRU, a ratio of 0.81.
  On the private test set with a WFST decoder it is 4.81% against 6.72%. Our best GRU is already at 0.91 of our
  baseline GRU (7.77 against 8.51), so a faithful run should land near 7.0%, about 10% below our best GRU.
- **A different architecture helps the ensemble even when it is weaker.** Two Transformers added as scoring members
  took live WER at 500 ms down by 0.27, at a point where adding more GRUs had stopped helping. In sentence-end word
  voting, three Transformers that were each worse than every GRU took the vote from 7.00% to 6.69%.
- **Timing.** Our earlier Transformer design emitted word ends 80-160 ms earlier than the GRU, which matters for
  the 200 ms budget. This has not been measured on the LightBeam recipe.

## 2. Why I think the training recipe is the cause

1. **Their model was not overconfident in the way ours is.** Their decoder ran the Transformer at the acoustic scale
   tuned for the GRU (0.325) and it still won. Ours needs 0.2. So the overconfidence comes from our run, not from the
   architecture.
2. **The differences point the same way as the symptoms.** We used about half their time masking and a fixed context
   length, and our batch sampler repeated some trials three times as often as others. Each of these makes memorizing
   easier.
3. **Their own ablation gives masking a large effect.** On the 2024 benchmark, removing time masking cost them 18%
   WER.
4. **The first diagnosis was built on a misreading.** I had read `max_mask_pct: 0.05` as "masks of up to 5 frames".
   It means up to 5% of the trial.

## 3. The changes

All three runs carry every change in sections 3.1 and 3.2. Each is an option that is off by default, so existing
models and recipes are untouched.

### 3.1 Changes I expect to matter most

**Heavier time masking** (`time_mask_max_frac=0.05`, no `time_mask_max_len`)
- Was: 20 masked spans of up to 5 frames each, about 22% of a trial hidden.
- Now: 20 spans of up to 5% of the trial's own length, about 40% of a median trial hidden.
- Problem: memorization and overconfidence. With most of the input visible, the model can fit each training sentence
  closely.
- Why it should help: the model has to infer hidden stretches from context, so it cannot be certain about every
  frame. This is the regularizer their ablation found most important.

**Batches drawn from shuffled passes over all trials** (`dataset.sampling=trials`)
- Was: four random days per batch, every day equally likely. A trial from the smallest day (59 trials) was drawn 3.0
  times as often as it would be if all trials were equally likely, and one from the largest day (364 trials) 0.49
  times as often. The ten smallest days hold 10.5% of the trials and received 22% of the draws.
- Now: every trial once per pass, with each batch mixing many days.
- Problem: repeated trials are what a memorizing model exploits, and the days with the most data were under-used.
- Why it should help: the day-balanced sampler exists so that each day's own input layer gets updated. This model
  has no per-day layer, so the balancing has no benefit here and only skews the data.
- Open question: days with few trials now get fewer updates. Our earlier Transformer was weakest on exactly those
  days, so per-session results need checking.

**Left context sampled per batch from 5 to 20 s** (`model.transformer.window_min=63`)
- Was: every frame could always see 20 s back, which for most trials is the whole sentence so far.
- Now: in training, the visible history is drawn between 5 and 20 s for each batch; evaluation still uses 20 s.
- Problem: memorization through long-range cues, such as position within the sentence.
- Why it should help: the model cannot rely on a fixed amount of history. This is my reasoning; I found no ablation
  of it.

**Dropout moved from the input into the layers** (`input_layer_dropout=0`, `embed_dropout=0.2`, `ffn_dropout=true`)
- Was: 0.2 on the raw input and 0.35 on the embedded frames, and none after the feed-forward activation.
- Now: 0.2 on the embedded frames only, and 0.35 inside the feed-forward layers.
- Problem: we were adding input noise their recipe does not have, while the layers that do the memorizing had less
  regularization. With masking doubled, the extra input dropout would hide too much of the signal.
- Why it should help: it puts the regularization where theirs is. I am less sure of this one on its own.

### 3.2 Smaller changes, included so the run matches theirs

I do not expect any of these to move the result much alone. They are in because the earlier run failed with
"small" differences left in, and each costs nothing.

| Change | Was | Now | Reason it could matter |
|---|---|---|---|
| Learned position bias (`pos_bias=relative`) | Fixed penalty that grows with distance (ALiBi) | One learned value per layer and distance | Each layer learns how far back to look. Their ablation: replacing it with sinusoidal positions cost 10% (not a direct comparison with ours) |
| CTC loss divided by sentence length (`ctc_length_norm=true`) | Per-trial sum | Per-trial sum / number of phonemes | Long sentences no longer dominate each update, and gradient clipping engages as it does in their code |
| Initialization (`init=torch`) | Xavier-uniform, about 3x the default variance for square layers | PyTorch defaults | Larger starting weights are a possible, unconfirmed contributor to overconfidence |
| Random cut | Up to 3 bins removed from the start | None | Fidelity only |
| Warmup | 1,000 batches | 1,260 batches (their 10 epochs) | Fidelity only |

### 3.3 Already in place, and kept different on purpose

- **No per-session input layer** (since the 2026-11-03 run). Their logs show how much this matters: adding a
  per-session layer takes their Transformer from 5.09% to 8.11%, and to 9.61% with the softsign we used. All our
  earlier Transformers had that layer except the one that shared it, which was our best.
- **Checkpoint choice.** They keep the checkpoint with the best validation phoneme error. We score a fixed final
  snapshot and an average of the last six, both set before training, because val-dev is also what we evaluate on.
- **Masked frames** get a learned vector where theirs get zeros. After their embedding layers the two are equivalent.

## 4. The three runs

About 100 minutes of training each, one at a time, then 1.5-3 hours of decoding for all three (my estimate).

| Run | What differs | Question it answers |
|---|---|---|
| `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s10` (control) | Their smoothing, which is centred and reads 80 ms of future input. No attention look-ahead | Does their recipe reproduce in our trainer at all? |
| `tfm384x5_lb_p20000_f160_noday_v2_63k_s10` | Causal smoothing; the last layer looks 160 ms ahead, as our GRU does | What does the recipe give in the form our live pipeline uses? |
| `tfm384x5_lb_p20000_f160_noday_v2_crctc_32k_s10` | The second run plus a consistency loss (CR-CTC, weight 0.2), on half the batches at twice the cost each | Does consistency training fix the overconfidence? |

**Why a separate control.** Their smoothing sees 80 ms into the future, which is the one part of their recipe our
live models do not copy. Without the control, a miss by the second run could be either our code or the smoothing.

**The consistency loss.** Each batch goes through the model twice with different random masking and noise, and the
loss penalizes disagreement between the two outputs.
- Problem: overconfidence, the main measured symptom.
- Why it should help: where the two views disagree, the model is pushed to spread its probability, which is what a
  decoder needs. In its paper the probability on emitted labels fell from 98.5% to 89.4% and error rates fell
  16-23% on speech benchmarks.
- Why it is a risk: both confidence regularizers tried here so far failed. A confidence penalty took a Transformer
  from about 9.2% to 12.1%, and uniform label smoothing cost 2.4 points. Those pushed toward a uniform distribution;
  this one pushes toward the model's own second view. Its paper reports nothing with an external language model.

## 5. How the results will be read

Set before training. Phoneme error is not the criterion, because the 2026-11-03 run already had the best phoneme
error and the worst word error.

For each run, on both the final snapshot and the six-snapshot average:

| Measure | 2026-11-03 run | GRU | Wanted |
|---|---|---|---|
| Calibrated CTC loss | 0.442 | 0.35-0.37 | lower |
| Best acoustic scale | 0.2 | 0.4 | 0.3 or above |
| Best WER over scales 0.2-0.5 | 9.15 | 7.77 | see below |

| Outcome | Meaning | Next step |
|---|---|---|
| Control at 7.0% or below, best scale 0.3 or above | Reproduced | Put the live-usable run through the deadline ranker as decider and member |
| Control reproduces, live-usable run clearly worse | The gain depends on the 80 ms of future smoothing | Count smoothing look-ahead in the ranker (it currently refuses such models) and use the control itself; 80 ms is less than the 160 ms our 500 ms decider already uses |
| Control between 7.0% and 8.5% | Inconclusive on one seed; their seeds spread 4.5-6.0% | Second seed before a verdict |
| Control at 8.5% or above | Not reproduced; our trainer still differs somewhere | Run their code unmodified before any more Transformer work |
| Either live-usable run at 8.85% or below | Kept as an ensemble member | It counts as beating the GRU only below 7.77% |

## 6. What could go wrong

- **The 7.0% target is my conversion**, not their number: their 0.81 ratio applied to our baseline GRU. Their figures
  come from a different decoder, with a large language model in the search.
- **The fixes are bundled.** If the result improves, we will not know which change did it.
- **One seed per run.**
- **Heavy masking hurt our GRU** (+5.17 WER), though that was combined with the GRU's much stronger input noise.
  Their recipe pairs heavy masking with light noise, and that pairing is what these runs test.
- **The gain may belong to the recipe, not to attention.** On the 2024 benchmark their GRU trained with the same
  masking was only 4% behind their Transformer. If the Transformer reproduces, the same recipe on our GRU is an
  obvious follow-up; it is not in the queue.
- **Sentence-end WER is not the goal.** Lower phoneme error and lower sentence-end WER have both failed to move
  live WER before. A reproduced model still has to prove itself at the word deadline.

## 7. The other runs and what each is for

Live WER at 500 ms is 8.26%. If the system always picked the right word when it was among its top three candidates it
would be 5.5%, and 3.8% if it picked the right word whenever it was offered at all. The ranker that makes the choice
has stopped improving with more language models, more data or more capacity. So each run below either improves the
acoustic model that produces the candidates, or gives the ranker a kind of evidence it does not have yet.

| Run | Time | What it is for | How it is read |
|---|---|---|---|
| **Candidate pooling across models** | under 1 h, CPU | Today the ranker chooses only among one model's 50 candidates, and the right word is missing at about 7% of deadlines. This adds the other models' candidates and their votes, using outputs already saved. No training | Live WER with own candidates, with votes, and with added candidates; how often the right word is offered. Under 0.3 gain means candidates are not the limit |
| **GPT-2 Large rescoring of the existing Transformer** | minutes | How much of the Transformer's deficit is the 4-gram decoder? Their result used a much stronger decoder; on our earlier Transformers rescoring narrowed the gap from about 0.9 to 0.5 | Sentence-end WER after rescoring against the GRU's 6.10%. Not a live number |
| **Word cut points** | minutes | Preparation only: marks where each word ends in every training trial. Needed by the splicing runs and the second scorer | - |
| **Phrase splicing** (GRU and Transformer) | 2-4 h each | More training sentences from the same recordings: with probability 0.3 a trial becomes the first words of one sentence joined to the last words of another from the same day. Published gain 7.0% to 5.5% (bidirectional model) | Kept if WER is at least 0.3 below the matched model and word ends are no more than 40 ms later |
| **Diphone targets** (GRU and Transformer) | 2-4 h each | An extra training target that makes the model predict each phoneme together with the one before it, so its features carry how sounds run into each other. Dropped at inference, so decoding is unchanged. Published gain 9.46% to 8.06% (bidirectional GRU) | Same keep rule |
| **Live WER at 2, 3 and 4 s** | about 12 h | A measurement, not an improvement. Going from 500 ms to 1.5 s bought only 0.25, yet rescoring the finished sentence is about 2 points better than live. This shows how quickly the gain arrives with waiting, and whether under 6% is reachable at a longer budget | Live WER per budget, with the share of deadlines that run past the end of the trial |
| **Non-CTC second scorer** | 1-2 h training, plus untimed scoring | Every acoustic score the ranker sees comes from CTC, which scores each frame independently. A small attention decoder on the same frozen model predicts phonemes one at a time, each conditioned on the ones before, so its mistakes differ | Live WER with and without the new score. The ranking step has never been run end to end |

Two notes on this list:

- **Hold the two Transformer arms of splicing and diphones** until the control and the live-usable run are scored.
  They are built on our older Transformer design (8.67%), which becomes obsolete if the corrected recipe reproduces.
  The two GRU arms can run as planned.
- **Not ready to run: a fine-tuned selector that reads several models' live guesses.** This is the largest published
  lever: 29-57% at sentence end when a fine-tuned language model merges 10-29 models' outputs, against 6-10% without
  fine-tuning. Nobody has reported it for words committed at a deadline. It needs one to one and a half weeks of
  building, and it is the reason architecture diversity from the Transformer matters.

## 8. Sources

- LightBeam code and validation logs: github.com/ebrahimfeghhi/brainaudio at commit 90e0ac0
  (`neurips_b2t_25_chunked_transformer.yaml`, `models/transformer_chunking.py`, `training/trainer.py`,
  `auxiliary_folders/day_parameters_experiment/output_logging/day_specific_log_b2t_25.md`).
- LightBeam paper: arXiv 2603.14002. Time-masking and position ablations: Feghhi et al. 2025, arXiv 2507.02800v2
  (2024 benchmark, 3-gram, validation).
- Consistency loss (CR-CTC): arXiv 2410.05101.
- Diphone targets (DCoND): arXiv 2411.10657. Phrase splicing: "Cut, Stitch, Adapt", MICCAI 2026.
- Ensemble merging: arXiv 2412.17227 (2024 benchmark lessons), arXiv 2511.21740 (BIT), bioRxiv
  10.64898/2026.06.02.729705 (deep neural ensembles).
- Our measurements: `current_knowledge.md` sections 0, 4.5, 4.8 and 6.4.
- Queue scripts: `results/harness/queue/tfm_lbfix_20261111.py` (Transformer runs),
  `results/harness/queue/acoustic_aug_20261110.py` (splicing and diphones),
  `results/harness/queue/late_deadlines_20261109.py` (2-4 s budgets).
