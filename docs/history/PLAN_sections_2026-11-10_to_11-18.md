# Plan sections finished or superseded 2026-11-10 to 11-18

Moved out of `docs/PLAN.md` on 2026-11-18, unchanged. Results are in `current_knowledge.md` (4.9-4.11, 6.10,
6.11 and the ledger in section 7). The current plan is `docs/PLAN.md`.

## Next experiments, prepared 2026-11-10 (steps 1, 3 and 4 done; 2 and 5 not run)

Target: live-word WER below 6% (`current_knowledge.md` section 0; now 8.81 / 7.27 / 7.10 at 200 ms / 500 ms / 1.5 s).
The ranker is saturated on its current inputs (more LLMs, data and capacity add nothing), so each step adds new
evidence. Code is in place and smoke-checked on CPU; GPU smokes happen at launch (`harness train` smokes 50 batches
before the real run). Rules as always: one GPU process at a time, no 8.5 GB decoder alongside GPU jobs, nice 19,
fresh run paths. All results are val-dev and exploratory; one seed per arm.

1. **Pool candidate lists across deciders.** Done 2026-11-17 in chain 1 (`current_knowledge.md` 6.10): member votes
   took the Y6 decider from 7.27 to 7.06 (500 ms) and 7.10 to 6.58 (1.5 s), as an upper bound (the members' decoders
   and compute are not charged); options only members offer did not help. Next: see "Chain 1" below.
2. **Budget curve at 2, 3 and 4 s** (CPU decoding then GPU scoring, about 12 h; not alongside training).
   `nohup .venv/bin/python results/harness/queue/late_deadlines_20261109.py > results/harness/queue/late_deadlines_20261109_run.log 2>&1 &`
   Result in `learning/live_eval/out/llm_live/ranker_lists_late.{json,log}`. Late deadlines often clip to the
   trial end (oracle endpointing): report that share with the numbers.
3. **Phrase splicing and diphone targets.** Done 2026-11-16: no arm passed its keep rule on the GRU or the conv
   Transformer (results in `current_knowledge.md` 4.10). The cut points from `data_prep/word_boundaries.py` remain
   for step 4.
4. **Non-CTC second scorer.** Done 2026-11-17 (`current_knowledge.md` 6.10): the head trained on Y7 average's
   training-set states, scoring Y7's lists, did not help (+0.13 at 500 ms, -0.03 at 1.5 s). Its encoder saw the
   training trials; a head on out-of-fold encoders (step 5's fold models) is the untested variant.
5. **Candidate-aware fine-tuned LLM** (about 1-1.5 weeks; only the first piece exists). Give it the live partial
   hypotheses of several deciders as well as the candidates (see "What the 2026-11-10 literature check changes").
   Ready: `dataset.train_fold.n_folds=5` / `dataset.train_fold.holdout=K` trains without every fifth training trial.
   Still to build: a harness scope that decodes held-out training trials; deadline lists on them
   (`llm_live/run.py` pattern); the selector fine-tune (prefix + candidates + phonemes -> next word). `peft` is not
   installed and `bitsandbytes` cannot find its CUDA library, so 4-bit loading does not work today; OPT-2.7B in fp16
   fits 16 GB without it. Environment changes need approval.

### Language-model steps, planned 2026-11-11 (nothing built or run)

Superseded 2026-11-18 by "Next steps" below: L2 is step A2, a larger general LLM is step B, L1 is not now.

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
- A1, A2, A4. Done 2026-11-17 in chain 1 (`current_knowledge.md` 6.10): Y6 average as the decider at scale 0.3,
  7.27 / 7.10 at 500 ms / 1.5 s and on time (keep rule met); Y7 at scale 0.3 -0.13 / -0.28; smoothing look-ahead is
  counted in the ranker.
- A3. **Word voting and the CTC ensemble with Y6 / Y7 added**, from saved outputs (`learning/per_eval/sentence_vote.py`;
  reference 7.00 with 14 GRUs, 6.69 with three older Transformers), and their word-end timing against G15 from the
  saved replays (late words cost 2.1 points at 200 ms).

**B. Trainings on the corrected recipe (about 2 h each at 63k; one GPU process at a time).**
- B1. **Seeds.** Y7 seed 2 done (Y9: 7.66 / 7.58 vs Y7 7.44 / 7.43; live without the LLM unchanged). Y6 seed 2,
  the model every arm here is compared with: done 2026-11-18 (Y10: 6.95 / 6.95 vs Y6 7.27 / 6.97).
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
Its proposals were audited against this project on 2026-11-17; the ones that survived were run in chain 1 or are
listed below.

### Chain 1 (2026-11-17, done)

Chain 1 (`results/harness/queue/chain1_20261117.py`; outputs `results/harness/queue/20261117T180730Z_chain1/`;
results in `current_knowledge.md` 6.10): the Y6 average as the decider at scale 0.3 met its keep rule (7.27 / 7.10 at
500 ms / 1.5 s, on time); Y7 at scale 0.3 gained 0.13 / 0.28; the 11-17 Y7 runs' fp32 OPT timing was corrected;
candidate pooling gave 7.06 / 6.58 as an upper bound; the second scorer, previous-sentence context and two-thread LLM
scoring did not help.

Code changed 2026-11-17 (in `learning/live_eval/llm_live/`; checked: member shifts unchanged for every existing pair,
the 11-17 Y7 ranker reproduced at 7.61 / 7.42, Y6's commit delays match `clock_a`): smoothing look-ahead counted
(`ctc_features.lookahead`, `end_bins`, `ranker.load`); list variants read their own decode run (`run.json`);
`ranker_pool.py` aligns members by input bin (before, 14-bin members of a 4-bin decider read ~200 ms past its
deadline); `aed_features.py` lets the head's encoder differ from the decider and charges the head's time. New:
`ranker_chain1.py`, `score_llm_ctx.py`, `llm_concurrency.py`.

### Done 2026-11-17 night to 11-18 (`current_knowledge.md` 4.9, 4.11, 6.11)

- Word familiarity: words seen fewer than five times in the 8,072 training sentences are 15% of val-dev words and two
  thirds of the errors, live and at sentence end; about 60% of those errors are misrankings, 40% words never offered.
- Ranker: familiarity features kept (8.64 / 7.06 / 6.69); the 1d inputs kept at 1.5 s (6.51); a second decoder's
  votes give 6.77 / 6.48 at 500 ms / 1.5 s; the 1d inputs and the votes do not add up.
- Overnight models (Y10-Y13): Y6's recipe holds over two seeds (best WER 6.95 / 6.97); none of the four helps as a
  CTC member (the ensemble is saturated); Y6's heavy time masks delay its word ends (Y13: p95 120 vs 360 ms, +1 WER).
- Conclusion: acoustic recipe variants and more CTC members no longer move live WER. The gaps to 6% (2.6 / 1.1 / 0.5
  points with one decoder) must come from the language side and from words rare in training. No more GPU nights on
  acoustic variants of the Y6 recipe until one of the steps below needs them.

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
