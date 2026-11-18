# Streaming neural decoding plan — updated 2026-11-18

**Where things stand:** live-word WER is 8.64 / 7.06 / 6.51% at 200 ms / 500 ms / 1.5 s on val-dev with one decoder,
against a target of below 6% (`current_knowledge.md` section 0, which also holds every result and the ledger of what
was tried); with a second decoder's votes 6.77 / 6.48 at 500 ms / 1.5 s. The 500 ms and 1.5 s results come from the Y6
average deciding the words, with word-familiarity features in the ranker. Words seen fewer than five times in the
training sentences are 15% of val-dev words and two thirds of the errors, and no acoustic model has improved on
never-seen words; the overnight models of 2026-11-17 (Y10-Y13) did not move live WER. The work now is on the language
side, led by a phoneme-aware word selector (step 1, started 2026-11-18; its fold models are training). The research
direction remains exploratory.

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

## Next steps (2026-11-18)

Earlier plan sections (the 2026-11-10 list, the language-model steps, the LightBeam follow-ups, chain 1, the night of
11-17) are finished or superseded: `docs/history/PLAN_sections_2026-11-10_to_11-18.md`. Their results are in
`current_knowledge.md`. Acoustic recipe variants and more CTC members no longer move live WER (4.9, 6.11); the gaps
to 6% (2.6 / 1.1 / 0.5 points at 200 ms / 500 ms / 1.5 s) must come from the language side and from words rare in
training (two thirds of the errors; about 60% misranked, 40% never offered). All numbers val-dev (1,253 trials, 8,277
words), exploratory, reported by familiarity bin and with deadline-respecting WER. Machine: one GPU job at a time,
never beside a WFST decoder; CPU-only untimed decodes can run two at a time (`harness bench/sweep --no-job-lock
--device cpu`; RAM fits two 8.5 GB decoders).

### 1. Phoneme-aware word selector (started 2026-11-18; 1-1.5 weeks)

**Idea.** At each deadline an LLM reads the words already shown, the decision's candidate words with their
pronunciations, and the decider's phoneme string for the word being decided, and scores each candidate (later also
proposes a word outside the list). Never-seen words keep about 65% of their phonemes right, which can be enough to
recognise the word the way spelling correction does. The ranker's CTC features already measure each candidate's
acoustic match, so the selector has to add what they cannot: context and sound judged together, the decider's typical
confusions (learned from held-out errors), and words the list does not contain. Precedent (sentence end only):
DCoND-LIFT's fine-tuned LLM reading candidates and phonemes, 8.09 -> 5.77 with ten models, 7.29 in-context only.

**1a. Out-of-fold models (running since 15:56 UTC, done about 01:20 UTC 11-19).** On their own training trials models
have near-perfect phonemes and have seen every word, so the selector's training decisions come from models that never
saw them: five Y6-recipe models, each without one fifth of the training sentences
(`results/harness/queue/folds_y6_20261118.py`; `dataset.train_fold` with `by: sentence`, new 2026-11-18, because 17.6%
of training trials repeat a sentence). Per fold about 1,600 trials and 10,000 words, 4.5-5.3% never seen by its
model (val-dev 6.8%). Check: each fold model's val-dev PER near Y6's (it sees 80% of the data).

**1b. Held-out decisions (CPU, after training; then GPU for scoring).** Per fold: the fold model's log-probs on its
held-out trials, a WFST decode with partials, and prefix 50-best lists at the deadlines (D = 0 and D = 10, as the Y6
decider), as `llm_live/prefix_nbest.py` does for val-dev. The harness scopes cover val only, so this needs a small
driver over the training trials. About six times val-dev's size; two decoders in parallel. Leakage: GPT-2 L ft and
every CTC member were trained on all training sentences, so none of their scores enters the selector's inputs; the
candidate order comes from the decoder score (acoustic + 4-gram) and OPT-2.7B. Familiarity, where used, counts the
fold model's own training sentences.

**1c. Selector inputs.** Context: the live partial's words before the position (what a live system has), not the
reference. Phonemes: the decider's greedy CTC phonemes between the previous word boundary and this one, at the
deadline frame. Candidates: the top K (about 10-20) by decoder score, each with its lexicon pronunciation. Target: the
reference word. About 50k decisions per deadline from the folds; one fold's sessions held out for the selector's own
validation.

**Queued (2026-11-18 16:10 UTC):** `results/harness/queue/selector_probe_20261118.py` waits for the fold models,
then runs 1b (held-out forward passes on the GPU, lists with two CPU decoders), 1c (records) and 1d (the probe on
the GPU, then its ranker runs on four splits per deadline). Code: `learning/selector/` (`inputs.py`, `heldout.py`,
`score.py`, `ranker_selector.py`); the val-dev records are in `learning/selector/val/`. The reference word is in
the top 20 by decoder score for 78% of never-seen words but first for only 42% (`current_knowledge.md` 6.11).

**1d. Zero-shot probe (GPU, about 1 h, when the GPU is free).** Qwen2.5-3B (downloaded 2026-11-18) prompted with a
few examples in the same format scores the val-dev candidates; its score enters the cross-fitted ranker as a feature.
DCoND's in-context result says this should help a little; if it adds nothing beyond the CTC features, the fine-tune
has to make up the whole case.

**1e. Fine-tune.** LoRA on Qwen2.5-1.5B (then 3B) with the 1c decisions (`peft` needed, an environment change),
about 1-2 h of GPU per run. At the deadline one prompt of about 100-150 tokens plus the K options: roughly 20-40 ms,
inside the 500 ms budget (may replace OPT).

**1f. Evaluation.** The selector's log-probability of each candidate, and its interaction with the rare flag, as
ranker features for the Y6 decider at 500 ms and 1.5 s and for D1 at 200 ms (cross-fitted by session halves, four
splits), its time charged. Keep rule (set now): live WER at least 0.2 lower at 500 ms or 1.5 s, deadline-respecting
no worse, on the standard split and two of three random splits. Then 1g: words the selector generates outside the
list as extra options (the only route besides step 2 to the never-offered 40%).

**Stop rule.** If neither the probe nor the first fine-tune lowers live WER by 0.1 at any budget, stop and write it
up as a negative result.

### 2. Coverage of rare words (CPU, can run beside 1)

- 2a. Diagnostic (minutes, saved outputs): for each error whose word was absent at its deadline, is the word in the
  decoder lexicon, in the sentence-end 100-best (5.1), and how probable is it under the 4-gram?
- 2b. 100-best deadline lists for the Y6 decider at D = 0 and D = 10 (CPU decode; GPT-2 L ft and OPT scoring, about
  twice the LLM time per decision). Keep rule: live WER at least 0.1 lower at 1.5 s, deadline-respecting no worse.
- 2c. A less pruned first-pass 4-gram is not buildable here: composing the decoding graph past pruning 1e-8 needs
  about 50 GB of RAM (August build log, `docs/history/log_2026-08.md`; 1e-8 peaked at 25 GB of 27). The unpruned
  4-gram already exists (`results/lm_general_4gram/lm_orig.arpa`, 61 M n-grams, 2 GB OpenWebText; pruned versions at
  3e-9 and 1e-9 beside it), so instead: rescore every deadline candidate with it as a ranker feature (CPU). That
  helps ranking of rare sequences, not coverage.

### 3. A stronger general LLM in the ranker (GPU, about a day)

Qwen2.5-3B (downloaded) scores the saved deadline lists (Y6 decider D = 0 and D = 10, D1's lists) in fp16; it enters
the ranker with its rare-word interaction, as a third LLM at 1.5 s and in place of OPT at 500 ms (a third LLM made
words late there before). The October Qwen 0.5B / 1.5B test used weights tuned for GPT-2; the ranker now learns
them. Keep rule: live WER at least 0.15 lower at one budget, deadline-respecting no worse. Shares the GPU with step 1.

### 4. Y13 as the 200 ms decider (about half a day)

Y13 (Y6 with shorter masks) commits at its own word end with p95 120 ms and is 11.21 live before any LLM step (D1:
11.57). Deadline lists at D = 0 and D = 1 (CPU decode), GPT-2 L ft and OPT scoring, CTC features (members with at most
one frame of future input: the causal GRUs and the Y6-recipe models), the familiarity ranker on its own lists and
pooled with D1's. Keep rule: live WER at least 0.15 below 8.64, deadline-respecting no worse than 10.77. Then the
deadline compute (fp16 GPT-2, fewer scored candidates; about half the 200 ms lateness).

### Not now

More acoustic recipe variants or CTC members (including the 11-16 attention look-ahead and causal arms); the 2-4 s
budget curve; the corpus oracle; marginalized diphones; phrase splicing; fine-tuning OPT on the training sentences
(favours the vocabulary that is already right); Deadline-MWER and the neural-token adapter (they can reuse 1a's fold
models later).

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
- **Real-time Transformer forward pass (needed before any latency claim with a Y model).** Today the Y models are
  replayed from outputs computed offline under the causal mask. A streaming forward pass for the relative-bias
  Transformer (per-layer caches over the 20 s window; frame t's output leaves when frame t + R arrives), checked
  against the offline outputs, then the harness timing check.
- **Normalization experiment (dormant).** Resolve preprocessing, clipping and reset provenance and amend the
  proposed variance floor (R-D2 / R-D3) before that treatment; see `.claude/rules/acoustic-pipeline.md`.

The September milestones (output capture, paced replay, the harness, the first commitment baseline and the first
causal Transformer) are complete; their text is in `docs/history/PLAN_2026-09.md` and their results in
`results/RESULTS.md` sections 1 and 4-6.
