# Streaming neural decoding plan — updated 2026-11-10

**Where things stand:** live-word WER is 8.81 / 8.26 / 8.01% at 200 ms / 500 ms / 1.5 s on val-dev against a target
of below 6% (`current_knowledge.md` section 0, which also holds every result and the ledger of what was tried). The
experiments below are prepared and not yet run. The research direction remains exploratory.

Models are named as in `harness/registry/MODELS.md`, which also lists the letter IDs used here (A2, G2, X6).

## Objective and scope

Build an efficient real-time speech-to-text pipeline on B2T '25 and the Makin
lab's ECoG dataset, and use its development to investigate what accuracy,
stability, and delay cost. The class presentation remains the engineering frame.
A paper should explain a reproducible finding from this work; its central result
and novelty are not decided in advance.

Use modality-appropriate features and encoders. Share timing, output-event, and
evaluation conventions where meaningful; report each dataset separately.
B2T implementation can proceed now. ECoG is an intended second development track,
with access, task, data format, alignment, and baseline still unknown.

The September 2026 validity review limits particular claims and experiments; it does not
freeze unrelated development. The August plan, the September milestones of this file
and the review itself are in `docs/history/` (index: `docs/history/README.md`).
`results/RESULTS.md` (local) is the source-linked record of the September measurements.

## Next experiments, prepared 2026-11-10 (none run yet)

Target: live-word WER below 6% (`current_knowledge.md` section 0; now 8.81 / 8.26 / 8.01 at 200 ms / 500 ms / 1.5 s).
The ranker is saturated on its current inputs (more LLMs, data and capacity add nothing), so each step adds new
evidence. Code is in place and smoke-checked on CPU; GPU smokes happen at launch (`harness train` smokes 50 batches
before the real run). Rules as always: one GPU process at a time, no 8.5 GB decoder alongside GPU jobs, nice 19,
fresh run paths. All results are val-dev and exploratory; one seed per arm.

1. **Pool candidate lists across deciders** (CPU, under an hour). Other deciders' saved lists and partials vote on
   each option and add options the deciding model lacks.
   `nice -n 19 .venv/bin/python learning/live_eval/llm_live/ranker_pool.py learning/live_eval/out/llm_live/ranker_pool.json > learning/live_eval/out/llm_live/ranker_pool.log 2>&1`
   Read: WER per decider for "own lists" / "+ member votes" / "+ member-only options", and "reference offered:
   own / pooled". The members' decoders are assumed to run concurrently; their compute is not in the commit time.
   If the gain is under 0.3, candidates are not the limit: go to step 3.
2. **Budget curve at 2, 3 and 4 s** (CPU decoding then GPU scoring, about 12 h; not alongside training).
   `nohup .venv/bin/python results/harness/queue/late_deadlines_20261109.py > results/harness/queue/late_deadlines_20261109_run.log 2>&1 &`
   Result in `learning/live_eval/out/llm_live/ranker_lists_late.{json,log}`. Late deadlines often clip to the
   trial end (oracle endpointing): report that share with the numbers.
3. **Phrase splicing and diphone targets** (4 trainings of 2-4 h, then decoding).
   First the cut points for every training trial (a few minutes of GPU):
   `.venv/bin/python data_prep/word_boundaries.py`
   then `nohup .venv/bin/python results/harness/queue/acoustic_aug_20261110.py > results/harness/queue/acoustic_aug_20261110_run.log 2>&1 &`
   (arm names as arguments to run a subset). Config keys: `dataset.splice.prob`, `dataset.splice.boundaries`,
   `diphone_weight`. Keep rule (in the script): best WER at least 0.3 below the matched 20k reference and word ends
   no more than 40 ms later; kept models go to the ranker as decider / ensemble member.
4. **Non-CTC second scorer** (one training of roughly 1-2 h per decider, untimed; needs step 3's cut points).
   `.venv/bin/python model_training/aed_scorer.py --encoder gru1344_head256_pall_f160_20k_s10 --out model_training/trained_models/aed_gru1344_head256_pall_f160_20k_s10_$(date -u +%Y%m%dT%H%M%SZ)`
   then, per deadline, `learning/live_eval/llm_live/aed_features.py features HEAD_DIR la1344_r2_s10_final 1` (G2
   under its old name, which the cached files in `learning/` are keyed by; and
   `13`), then `aed_features.py rank OUT_JSON 1`. The `rank` path has not been exercised end to end. The encoder
   saw the training trials, so the head may over-trust it; step 5's fold models remove that.
5. **Candidate-aware fine-tuned LLM** (about 1-1.5 weeks; only the first piece exists). Give it the live partial
   hypotheses of several deciders as well as the candidates (see "What the 2026-11-10 literature check changes").
   Ready: `dataset.train_fold.n_folds=5` / `dataset.train_fold.holdout=K` trains without every fifth training trial.
   Still to build: a harness scope that decodes held-out training trials; deadline lists on them
   (`llm_live/run.py` pattern); the selector fine-tune (prefix + candidates + phonemes -> next word). `peft` is not
   installed and `bitsandbytes` cannot find its CUDA library, so 4-bit loading does not work today; OPT-2.7B in fp16
   fits 16 GB without it. Environment changes need approval.

### Transformer fixes, prepared 2026-11-10 (none run yet)

The 2026-11-03 "faithful" LightBeam run was not faithful (`current_knowledge.md` 4.8, "Reproduction gaps"). Their own
repository reports, on full validation with their decoder, causal Transformer 5.09% vs baseline GRU 6.2-6.4% (10
seeds each), so a faithful run should land near 0.81 of our baseline GRU. Fixes are coded, unit-tested and checked on
CPU under `torch.compile`; no GPU run yet.

| Fix | Config key | Was |
|---|---|---|
| Masks up to 5% of the trial's frames (about 40% of a median trial masked) | `time_mask_max_frac=0.05`, no `time_mask_max_len` | up to 5 frames (22%) |
| Left context drawn per batch from 5-20 s in training | `model.transformer.window_min=63` | fixed 20 s |
| Learned relative position bias | `model.transformer.pos_bias=relative` | ALiBi |
| CTC loss divided by target length | `ctc_length_norm=true` | per-trial sum |
| No random cut | `dataset.data_transforms.random_cut=0` | 3 |
| Batches are shuffled passes over all trials | `dataset.sampling=trials` | 4 random days per batch (small days drawn up to 3x as often) |
| Dropout 0.2 on frame embeddings only; dropout inside the feed-forward | `model.input_network.input_layer_dropout=0`, `model.transformer.embed_dropout=0.2`, `model.transformer.ffn_dropout=true` | 0.2 on raw bins + 0.35 on embeddings; none in the feed-forward |
| PyTorch default initialization | `model.transformer.init=torch` | Xavier-uniform, zero biases |
| Centred smoothing, 80 ms of future input (arm A only) | `dataset.data_transforms.smooth_lookahead=4` | causal |
| Consistency-regularized CTC (arm C only) | `cr_ctc_weight=0.2` | - |

- **T0. Rescore the existing model** (minutes of GPU, no training): how much of the gap is the decoder?
  `.venv/bin/python -m harness rescore 20261104T002238Z_tfm_lb_r2_63k_s10_avg+gen4g-acoustic_scale-0.2_sweep`
- **T1-T3. Three trainings of about 100 min each, then decoding:**
  `nohup .venv/bin/python results/harness/queue/tfm_lbfix_20261111.py > results/harness/queue/tfm_lbfix_20261111_run.log 2>&1 &`
  (arm names as arguments to run a subset).
  - A `tfm384x5_lb_p20000_f80_sm80_noday_v2_63k_s10`: their recipe as closely as this trainer allows (centred smoothing, no attention
    look-ahead). The reproduction control. Reproduced if its best 4-gram WER is 7.0 or below (0.81 x A2's
    8.51) with the best acoustic scale at 0.3 or above; if it is 8.5 or above, run their code unmodified next.
  - B `tfm384x5_lb_p20000_f160_noday_v2_63k_s10`: the same with causal smoothing and R = 2; plugs into the live pipeline as is.
  - C `tfm384x5_lb_p20000_f160_noday_v2_crctc_32k_s10`: B + CR-CTC on half the batches.
  Read-out and keep rule are in the script (best-temperature CTC loss, best acoustic scale, best WER; reference 0.442 /
  0.2 / 9.15; GRU 7.77). PER is not the criterion. The fixes are bundled, so a gain is not attributed to any one. One
  seed per arm; their seeds spread 4.5-6.0%.
- Arm A cannot join the deadline ranker until `ctc_features.lookahead` counts smoothing look-ahead (it asserts none).
- Suggested order for the first day: step 1 (CPU, under an hour), T0, arms A and B, then C or step 3's queue overnight.

### What the 2026-11-10 literature check changes

- **Step 5 is the largest published lever, and it should read several models.** Merging the decoded sentences of
  10-29 models with a fine-tuned LLM cuts sentence-end WER by 29-57% (DCoND-LIFT 8.09 -> 5.77; BIT 4.06 -> 1.76; RNN
  6.67 -> 3.09; Stanford deep ensembles, T15: -35%); the same merge without fine-tuning gives 6-10%. Every result is
  sentence-end; none is for words committed at a deadline. Here, plain word voting over the saved sentence-end
  outputs of 14 GRUs gives 7.67 -> 7.00, and 6.69 with three Transformers added
  (`learning/per_eval/sentence_vote.py`), in line with the untrained numbers. So build the step-5 selector to read
  the live partial hypotheses of several deciders, not one decider's 50-best.
- **Different architectures are worth more as members than more seeds**: three Transformers, each worse than every
  GRU, added more to the vote than going from 5 to 14 GRUs. Transformer arms that miss the GRU's WER are still
  candidates for members.
- **Untested and cheap:** members made from one model with test-time input noise (Stanford: -17% on T15 at sentence
  end, against -35% for ten separately trained models); a less heavily pruned first-pass n-gram (ours is a 4-gram
  pruned at 1e-8; the literature's 5-gram needs ~300 GB).
- **Not available to us:** multi-user training (Fogg et al.: -30% on this dataset) needs five other participants'
  unreleased data; no pretrained weights are published.

## Open tracks and standing constraints

- **Lab ECoG track (not started).** With the user and lab, establish authorized storage and access, participants
  and sessions, task, sampling and features, labels and alignment, split constraints, and the lab's existing
  decoder. Implement the smallest adapter the real schema supports; do not build speculative loaders or assume the
  B2T phoneme vocabulary or checkpoints transfer. This blocks ECoG integration only, not B2T work. No collaborator
  messages, data transfers or downloads follow automatically.
- **Paper question (open).** Choose it with the lab from the measured tradeoffs, then freeze the pipeline and
  protocol before any independent evaluation. Without independent data, present a disclosed retrospective
  exploratory study and limit generalization claims.
- **Evaluation.** Val-dev has been used for selection and tuning; former val-test took part in checkpoint selection.
  Neither is an independent holdout, and no replacement split is created silently. Define selection data and
  independent evaluation before any confirmatory comparison.
- **Uncertainty.** Match conditions and use paired seeds when measuring training effects; report session
  uncertainty separately from seed variation. A single-replicate difference is not an equivalence margin.
- **Timing.** Trial ends are the dataset's (oracle endpoints). Word-end times are alignment proxies, not ground
  truth. Raw acquisition and feature extraction are outside the replay boundary. Report deadline misses and timing
  coverage; observed maxima are not hard bounds.
- **Normalization experiment (dormant).** Resolve preprocessing, clipping and reset provenance and amend the
  proposed variance floor (R-D2 / R-D3) before that treatment; see `.claude/rules/acoustic-pipeline.md`.

The September milestones (output capture, paced replay, the harness, the first commitment baseline and the first
causal Transformer) are complete; their text is in `docs/history/PLAN_2026-09.md` and their results in
`results/RESULTS.md` sections 1 and 4-6.
