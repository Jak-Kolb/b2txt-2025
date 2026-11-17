# Streaming neural decoding plan — updated 2026-11-17

**Where things stand:** live-word WER is 8.81 / 7.61 / 7.42% at 200 ms / 500 ms / 1.5 s on val-dev against a target
of below 6% (`current_knowledge.md` section 0, which also holds every result and the ledger of what was tried). The
500 ms and 1.5 s results come from the LightBeam-recipe Transformer Y7 deciding the words (at 500 ms with more late
words than the GRU decider's 8.18); the best sentence-end result is Y6 with GPT-2 Large rescoring, 5.47. No Transformer
reliably meets 200 ms yet. No queue is running. The research direction remains exploratory.

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

## Next experiments, prepared 2026-11-10 (step 3 done with no arm kept, step 4 running, the rest not run)

Target: live-word WER below 6% (`current_knowledge.md` section 0; now 8.81 / 7.61 / 7.42 at 200 ms / 500 ms / 1.5 s).
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
3. **Phrase splicing and diphone targets.** Done 2026-11-16: no arm passed its keep rule on the GRU or the conv
   Transformer (results in `current_knowledge.md` 4.10). The cut points from `data_prep/word_boundaries.py` remain
   for step 4.
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

### Language-model steps, planned 2026-11-11 (nothing built or run)

Where the language side loses words at the 500 ms deadline (`current_knowledge.md` 6.4): the ranker picks a wrong
candidate for 5.88% of reference words, and 1.87% are absent from the candidates and from the final sentence. More
off-the-shelf LLMs add nothing, so each step changes what the LLM knows or what the first pass offers. Both steps
need approval before they start.

- **L1. Fine-tune OPT-2.7B on the 8,072 training sentences and swap it into the deadline ranker** (estimated one
  day: fine-tune, re-score the saved deadline lists, re-fit the ranker). Fine-tuning was the largest single-LLM gain
  for GPT-2 Large here, OPT is still used off the shelf, and LightBeam's fine-tuning was worth 7.26 -> 6.36 at
  sentence end. Scoring cost at the deadline does not change (about 35 ms per 50 candidates). Needs `peft` for
  LoRA (an environment change). Pattern: `model_training/benchmark/finetune_rescorer.py`, which made the fine-tuned
  GPT-2 Large. Reference: the current ranker (fine-tuned GPT-2 Large + OPT-2.7B + CTC ensemble) on the same
  cross-fitted lists. Keep rule: live WER at least 0.2 lower at two of the three budgets, and deadline-respecting
  WER no worse.
- **L2. A less heavily pruned first-pass n-gram** (estimated one to two days, mostly a CPU build; `build-lm`
  skill). Ours is a 4-gram pruned at 1e-8 (3.9 GB graph, about 8.5 GB of RAM); published systems use an unpruned
  5-gram (~300 GB), and BEST reports 2.57 vs 2.92 for 5-gram vs 4-gram in its pipeline. First a size table: graph
  size and decoder RAM at two or three lighter pruning thresholds, to find what fits in 27 GB. Then, for the
  largest that fits: sentence-end WER, the 100-best oracle, the share of deadlines where the reference word is
  offered (93% now), decoder memory and step time, and the ranker on the new lists. Reference: the same decider
  with `gen4g`. Risk: a larger graph decodes more slowly, and late words already cost WER at 200 ms.
- **Step 5 above is the third language-model step.** L1's fine-tuning path (LoRA on OPT-2.7B) is its first piece.

### Improving on the reproduced LightBeam model, planned 2026-11-16

Y6 is LightBeam's recipe reproduced (6.97; `current_knowledge.md` 4.9), and nothing of ours has beaten it. Ours so
far: averaging the last six snapshots (-0.30 on Y6, -0.01 on Y7) and the causal-smoothing variant Y7 (7.43, worse
than Y6 with twice the future input). One seed each, and their own seeds spread 4.5-6.0, so a difference under about
0.3 is not a result. Only the items marked "queued" are scheduled; the others need a go-ahead.

**A. Measure what the two models already give (no training).**
- A1. **Y6 through the deadline ranker** as decider and as ensemble member. Y7 was done 2026-11-17
  (`current_knowledge.md` 6.6): at commit rules matched to its later word ends it gives 7.61 at 500 ms (more late
  words than G2's 8.18) and 7.42 at 1.5 s (keep rule met). Without the LLM step Y6 is 0.15-0.25 better than Y7 (final
  weights) and its average about 0.6 better (6.2).
  Keep rule: live WER at least 0.2 below the G2 decider at the same budget, deadline-respecting WER no worse. Needs A4.
- A2. **Live replay of Y6 / Y7 at their best acoustic scale (0.3)**, CPU: the recorded replays used 0.4, where both
  are 0.5 worse at sentence end.
- A3. **Word voting and the CTC ensemble with Y6 / Y7 added**, from saved outputs (`learning/per_eval/sentence_vote.py`;
  reference 7.00 with 14 GRUs, 6.69 with three older Transformers), and their word-end timing against G15 from the
  saved replays (late words cost 2.1 points at 200 ms).
- A4. **Count smoothing look-ahead in the ranker** (`ctc_features.lookahead` asserts none; `ranker.budget()`), so Y6
  can be a member and a decider. Code only.

**B. Trainings on the corrected recipe (about 2 h each at 63k; one GPU process at a time).**
- B1. **Seeds.** Y7 seed 2 done (Y9: 7.66 / 7.58 vs Y7 7.44 / 7.43; live without the LLM unchanged). Y6 seed 2
  is not run: it is the model every arm here is compared with.
- B2. **CR-CTC (Y8: 8.71 / 8.57 at scale 0.5, the grid edge; higher scales queued).** As an improvement on Y7 it has
  to reach 7.13 or below (0.3 under
  Y7's 7.43, average model) at the same compute. If it does: the same on Y6's setting.
- B3. **Where the future input is spent.** Y6 (80 ms, centred smoothing) beat Y7 (160 ms, last-layer attention), with
  two things changed at once. Two arms on the corrected recipe:
  - `tfm384x5_lb_p20000_f160_sm80_noday_v2_63k_s10`: centred smoothing plus one frame of attention look-ahead, 160 ms
    in total. Does attention look-ahead add to Y6? Keep rule: at least 0.3 below Y6's average; then it is the 500 ms
    and 1.5 s decider candidate.
  - `tfm384x5_lb_p20000_f0_noday_v2_63k_s10`: no future input at all, the 200 ms candidate. Reference D1 (8.22 at
    sentence end, 11.6 live without the LLM step). Keep rule: at least 0.3 below D1.
- B4. **Phrase splicing and diphone targets on this recipe:** not run, since neither passed its keep rule on the GRU
  or the conv Transformer (2026-11-16).
- B5. **More seeds of the best variant as ensemble members**, only after A1 / A3 show that a Y member helps.

**C. Make it real time (needed before any latency claim).** A streaming forward pass for the relative-bias
Transformer (per-layer caches over the 20 s window; the output for frame t leaves when frame t + R arrives), checked
against the offline outputs as the GRU head was (2e-5), then the harness timing check. Today the Y models are
replayed from outputs computed offline under the causal mask.

**D. Outside ideas.** `docs/research/research_prompt.md` (not tracked) is a brief for an external deep-research run
on new architectures and training ideas; the report it produced is `docs/research/research_report_2026-11-17.md`.
Its proposals are triaged into this plan; none is adopted unread.

- **Follow-up queue, started 2026-11-16 12:24 EST, reordered 13:50** (`results/harness/queue/followup_20261116.py`,
  log `results/harness/queue/followup_20261116_run.log`). Done: X9 (conv Transformer + diphone) trained; decoding of
  G16, G17, Y8, X8 and X9 (`results/harness/queue/20261116T185056Z_decode`, recorded in `current_knowledge.md`).
  Running since 2026-11-17 00:01 UTC: the step-4 second scorer on G2, then on Y7, then Y7's second seed (`..._63k_s11`; its first attempt was stopped at batch
  8000 on the user's request) and its decoding. Then `results/harness/queue/y8_scale_up_20261116.py` (waits for the
  queue's pid) decodes Y8 at scales 0.6, 0.7, ... until WER rises (log `y8_scale_up_20261116.log`).
  Restart the queue from a stage with `--from aed_g2` etc. Afterwards: Y7 (and any
  winners) in the deadline ranker.

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
