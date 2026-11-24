# Streaming neural decoding plan — updated 2026-11-23

**Where things stand (2026-11-23):** live-word WER is 8.06 / **5.80** / **5.42%** at 200 ms / 500 ms / 1.5 s on val-dev
(mean delay 78 / 229 / 1,246 ms; mean of four session splits 8.01 / 5.79 / 5.43): under the 6% target at 500 ms and 1.5 s,
not yet at 200 ms (`current_knowledge.md` section 0, which also holds every result and the ledger of what was tried).
The 500 ms and 1.5 s decider is T1_1 avg, Y6's recipe at width 512 and 8 layers (26.3M parameters; one run), which
replaced V1_1 avg (512 x 6: 5.97 / 5.80; its second run V1_2 avg 5.81 / 5.61) by the rule set before its live run.
The full system adds the CTC ensemble, GPT-2
Large ft + OPT-2.7B, word familiarity and the fine-tuned Qwen2.5-3B phoneme-aware selector (one per limit, its time
charged). At 200 ms the causal GRU D1_1 decides; V2_1 (V1 + Y13's short masks) reaches 6.66 there, but only 91.0% of its
words are final within 200 ms, because list extraction and the two LLMs take too long. After the sentence, the learned
rescorer over T1_1 avg's 100-best reaches 3.71% (one model 4.13; GPT-2 Large alone 4.76; V1_1 avg 3.88), and over
D1_1's 5.46% (one model 5.76).

**Reporting (2026-11-19, `current_knowledge.md` 1.4):** a budget is met when 95% of words are final within it; correct
but late words then count as correct; every live number carries its mean delay; per budget, one model and the full
system; per model at sentence end, one model and the full system. Charts and slides name recipes (Y6), not runs.

**Order of work after the 2026-11-19 presentation:** fold model 4 reran (done 02:22 UTC, Y6sf4_1); the zero-shot
selector probe ran and added nothing (1d); the fold models' lists are being rebuilt after a class-order bug (1b), then
a quick fine-tune on a small subset decides whether the full one is worth running (1f). Words seen fewer than five times in the
training sentences are 15% of val-dev words and two thirds of the errors, and no acoustic model has improved on
never-seen words; the overnight models of 2026-11-17 (Y6_2, Y11-Y13) did not move live WER. The work now is on the language
side, led by a phoneme-aware word selector (step 1, started 2026-11-18; its five fold models are trained). The research
direction remains exploratory.

Models are named as in `harness/registry/MODELS.md`: recipe IDs (`Y6`) for configurations, run IDs (`Y6_1`, `Y6_2`)
for single training runs, "avg" for a run's averaged weights.

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
words), exploratory, reported by familiarity bin with mean delay and the share of words within the budget (a budget is
met when 95% of words are final within it; late-but-correct words then count as correct). Machine: one GPU job at a time,
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

**1a. Out-of-fold models (done 2026-11-19 02:22 UTC: Y6sf0_1 to Y6sf4_1, full-val PER 10.05-10.17).** On their own training trials models
have near-perfect phonemes and have seen every word, so the selector's training decisions come from models that never
saw them: five Y6-recipe models, each without one fifth of the training sentences
(`results/harness/queue/folds_y6_20261118.py`; `dataset.train_fold` with `by: sentence`, new 2026-11-18, because 17.6%
of training trials repeat a sentence). Per fold about 1,600 trials and 10,000 words, 4.5-5.3% never seen by its
model (val-dev 6.8%). Check: each fold model's val-dev PER near Y6_1's (it sees 80% of the data).

**1b. Held-out decisions (CPU, after training; then GPU for scoring).** Per fold: the fold model's log-probs on its
held-out trials, a WFST decode with partials, and prefix 50-best lists at the deadlines (D = 0 and D = 10, as the Y6_1
decider), as `llm_live/prefix_nbest.py` does for val-dev. The harness scopes cover val only, so this needs a small
driver over the training trials. About six times val-dev's size; two decoders in parallel. Leakage: GPT-2 L ft and
every CTC member were trained on all training sentences, so none of their scores enters the selector's inputs; the
candidate order comes from the decoder score (acoustic + 4-gram) and OPT-2.7B. Familiarity, where used, counts the
fold model's own training sentences.

**Fix (2026-11-19):** the first lists were wrong: `heldout.py` saved the logits in the acoustic class order, which the
WFST decoder misread (every phoneme as its neighbour), so the reference was in the top 20 for under 1% of decisions.
Fixed and rebuilt from the saved logits (folds 0 and 4 first, 1-3 next); nothing had been trained on them, and the
val-dev records and the synthetic set were not affected (`current_knowledge.md` 6.11).

**1c. Selector inputs.** Context: the words before the position in the decision's best candidate by decoder score
(what a live system displays), not the reference. Phonemes: the decider's greedy CTC phonemes between the previous
word boundary and this one. Candidates: the top 20 by decoder score (they hold nearly everything the 50 hold), each
with its lexicon pronunciation. Target: the reference word, by minimum-edit alignment. About 50k decisions per
deadline from the folds; fold 4 held out for the selector's own validation.

**Queued (2026-11-18 16:10 UTC):** `results/harness/queue/selector_probe_20261118.py` waits for the fold models,
then runs 1b (held-out forward passes on the GPU, lists with two CPU decoders), 1c (records) and 1d (the probe on
the GPU, then its ranker runs on four splits per deadline). Code: `learning/selector/` (`inputs.py`, `heldout.py`,
`score.py`, `ranker_selector.py`); the val-dev records are in `learning/selector/val/`. The reference word is in
the top 20 by decoder score for 78% of never-seen words but first for only 42% (`current_knowledge.md` 6.11).

**1d. Zero-shot probe (GPU, about 1 h, when the GPU is free).** Qwen2.5-3B (downloaded 2026-11-18) prompted with a
few examples in the same format scores the val-dev candidates; its score enters the cross-fitted ranker as a feature.
DCoND's in-context result says this should help a little; if it adds nothing beyond the CTC features, the fine-tune
has to make up the whole case.
**Result (2026-11-19, `results/harness/queue/20261119T022309Z_selector_probe/`):** nothing added. With the probe's score
as a ranker input, live WER is 7.06 -> 7.03 at 500 ms and 6.51 -> 6.50 at 1.5 s on the standard split, -0.02 to +0.05
on three random session splits, and never-seen words do not move (500 ms 43.0 -> 42.8). The fine-tune (step 2) has to
make the whole case.

**1e. Synthetic training data (CPU, about a day of code; added 2026-11-18).** The folds give about 50k real
decisions, of which only about 11% are words rare in training. A text corpus can supply millions: take sentences
from the local OpenWebText sample (`results/corpus/openwebtext_2gb.fmt.txt`, 2 GB), pick a word, look up its
pronunciation (125k-word lexicon), pass it through a phoneme noise channel estimated from the fold models' held-out
decodes (substitution, deletion and insertion rates per phoneme, by word familiarity: never-seen words about a third
of phonemes wrong, familiar ones about 5%), and build the candidate list the way the decoder does (the lexicon words
nearest to the noisy phonemes, weighted by word frequency, the true word present about as often as in the real lists:
79% for never-seen words). The selector then sees the whole 125k vocabulary, most of it never in the B2T training
sentences. The noise channel comes from the folds only, never from val-dev. Public sets checked against this:
HyPoradise (LLM correction of ASR n-best lists; text only, no phonemes) and phoneme recognizers run on LibriSpeech
(audio error patterns, not this decoder's); the synthetic set matches our decoder's errors and vocabulary better, so
it comes first, with HyPoradise-style pretraining as an option if the synthetic set alone falls short.

**Progress 2026-11-18 evening:** fold 0's held-out phoneme error by familiarity is 38.2 / 26.5 / 7.8 / 2.4
(val-dev, Y6_1 avg: 35.3 / 24.1 / 7.4 / 2.2), so the held-out trials do look like val-dev. The noise channel
from fold 0 (`learning/selector/noise.py`) and the generator (`learning/selector/synth.py`) give lists with the
right word first / in the top 20 for 44 / 74% of never-seen words (real lists 42 / 78%); a 300k-decision set is
being generated. Re-estimate the channel from all five folds once they finish.

**1f. Fine-tune.** First (2026-11-19) a quick run: LoRA on Qwen2.5-1.5B with about 4k contested decisions of fold 0
(`finetune.py --contested --folds 0`), then pick accuracy on fold 4 and on val-dev against the decoder's first candidate
(`eval_choice.py`; val-dev: decoder 85.4%, zero-shot Qwen2.5-3B 76.3%). Only if it beats the decoder there: the full
run on folds 0-3, then 1g.
**Result of the quick run:** pick accuracy 86.2% on val-dev against the decoder's 85.4% (fold 4: 87.8 vs 85.7). In the
ranker, 500 ms 7.06 -> 6.75 (-0.23 to -0.31 on all four splits); with its 56 ms charged the 95% rule holds (p95 432 ms,
98.0% of words within 500 ms). 1.5 s: 6.51 -> 6.31 on the standard split only. Next: rebuild folds 1-3
and train on folds 0-3 (about 23k contested decisions), and cut the selector's time (merged adapter, fewer options,
scoring only close decisions, or in place of OPT at 500 ms).
**Result of the full run (2026-11-19, `results/harness/queue/20261119T174419Z_selector_full/`):** folds 0-3, 23k
contested decisions, merged adapter (28 ms). Pick accuracy 89.3% on val-dev (decoder 85.4%). Ranker, time charged:
500 ms 7.06 -> 6.58 (all four splits -0.34 to -0.48), 1.5 s 6.51 -> 6.29 (standard and two of three random splits);
the 95% rule holds at both (98.7% / 99.4% of words within the budget). Kept. Gaps to 6% now 0.58 at 500 ms and 0.29 at 1.5 s. Next for the selector:
the synthetic set and a 3B model (more rare words; it already cuts never-seen-word WER 43.0 -> 39.4), D1's decisions
for 200 ms, and words outside the list (1h).
**200 ms (2026-11-19):** the selector on close decisions only (`SEL_GATE=3`, 29% of contested decisions) keeps 96.0% of
D1's words within 200 ms: full system 8.64 -> 8.36, one model 9.27 -> 8.84, on all four splits.
**Queued 2026-11-19 afternoon, to get 500 ms and 1.5 s under 6%** (gaps 0.58 and 0.29):
`results/harness/queue/selector_3b_20261119.py` (Qwen2.5-3B selector, D 0 and 10), then
`selector_ideas_20261119.py`: one 1.5B model on both deadlines' decisions; the words already decoded after the decision
word at 1.5 s; a 0.5B selector for 200 ms (faster, so more decisions can use it); 60k synthetic decisions mixed in
(noise channel re-estimated from all five folds). Dropped: Y6_2's lists voting. With Y6_1's own look-ahead it is not
admitted at D = 0 (a voting model needs R_M <= R_S + D - 1), and at 1.5 s a second decoder gave only 6.51 -> 6.48.
**Results and next (2026-11-19 evening):** the 3B selector is the new best (6.44 / 6.22). On the 1.5B, one model for
both deadlines and the following words at 1.5 s each helped a little; the 0.5B lost at 200 ms (8.58 vs 8.36). Queued:
`selector_combined_20261119.py`, a 3B on both deadlines with the following words, then the synthetic-mix test again at
batch 4 x 4 (at batch 8 its long prompts filled GPU memory and the run slowed 15x at step 3,000).
**Done (21:05 EST):** the combined 3B (both deadlines + following words) is the new best: 1.5 s 6.22 -> 6.07 (four-split
mean 6.29 -> 6.11, all four lower), 500 ms 6.44 -> 6.40 (mean 6.49 -> 6.45). The synthetic mix helped the 1.5B by about
as much as the other two ideas (means 6.65 -> 6.60 at 500 ms, 6.38 -> 6.30 at 1.5 s). Gaps to 6%: 0.07 at 1.5 s (about
6 words of 8,277), 0.40 at 500 ms. Next candidates: the combined 3B with the synthetic mix (batch 4 x 4, about 3.5 h);
the combined 3B on the GRU deciders' decisions at 200 ms (needs their following-word records; about 30 min).
**Overnight 2026-11-19/20** (`results/harness/queue/selector_night_20261119.py`): averaging selectors did not help
(1.5 s mean 6.11 -> 6.19 with the separate 3B), so the night goes to a stronger single selector: (1) the combined 3B at
200 ms; (2) a 3B on everything that helped (both deadlines, following words, 60k synthetic decisions, all five folds),
batch 2 x 8 with a slowdown watchdog; (3) the combined recipe on five folds, no synthetic; (4) the same with the
decoder's confidence per candidate in the prompt (`--conf`, new).
**Overnight results:** (1) a tie at 200 ms. (2) "Everything" 3B: pick accuracy 90.8 / 90.9% (combined 89.9 / 90.1%);
new best at 200 ms, 8.12 -> 8.06 (mean 8.16 -> 8.01, all four splits lower; one model 8.64 -> 8.40) and 500 ms 6.40 ->
6.39 (mean 6.45 -> 6.41; one model 7.43 -> 7.29); worse at 1.5 s (6.10, mean 6.16 vs 6.11), so 1.5 s keeps the combined
3B. Possibly because half of its training decisions (the synthetic ones) have no following words. (3) and (4) did not
finish: WSL shut down cleanly at 04:29 EST (Windows side) with (3) at step 3,100 of 3,644, which had also slowed to
5 s per step from step 1,550 (GPU memory spill at batch 8 with five folds; the watchdog's rising median missed it).
To rerun: (3) and (4) at batch 4 x 4, the watchdog against the first blocks' median.
**Reruns (2026-11-20 morning):** (3) five folds, no synthetic: pick accuracy 90.3 / 90.5%, but the full system is not
better (500 ms 6.46, mean 6.41, same as the everything 3B; 1.5 s 6.17, mean 6.21, higher than the combined 3B on all four
splits). It is the best selector for one model alone (Y6: 500 ms 7.42 -> 7.24, 1.5 s 7.15 -> 7.03, every split); to use it
in the one-model rows, score the GRU decisions with it too, so chart 03 keeps one selector. (4) `--conf` spilled GPU
memory at batch 4 x 4 (2.65 s per step from the start); restarted at batch 2 x 8. Result: a tie at 1.5 s (6.08 vs
6.07, mean 6.08 vs 6.11; split values 6.08 / 6.05 / 6.02 / 6.17), worse at 500 ms (mean 6.46). Against (3), its control,
the confidence lowers 1.5 s on all four splits (mean 6.21 -> 6.08) with the same pick accuracy. Shown numbers unchanged:
8.06 / 6.39 / 6.07. Next candidate: the combined recipe on folds 0-3 with `--conf` (about 1.5 h at batch 2 x 8). If the
five-fold cost (+0.10) and the confidence gain (-0.13) add up, 1.5 s would be about 5.98. Result: a tie (6.08, mean
6.12 vs 6.07 / 6.11). The four 1.5 s selector variants sit at means 6.08-6.21, a plateau about 0.1 above 6%, so the
next gains have to come from elsewhere (below).
**Toward 6% at 500 ms (queued 2026-11-20 afternoon, no training; `results/harness/queue/waits_avg_20261120.py`).** At
500 ms the 529 errors are 304 misranked words and 190 never offered. (1) **Use the waits the budget allows:** 500 ms
commits at D 0 (98.3% of words in time) and 1.5 s at D 10 (Y7_1's old timing). Without LLMs, D 1 and D 12 stay inside
the 95% rule and were worth -0.10 / -0.18. New list set at D 1 / 12, scored through the whole system with the current
selectors. (2) **Y6_1 avg and Y6_2 avg averaged frame by frame into one decider** (same recipe and timing; better lists
would act on the never-offered words). Sentence-end gate first: it goes to the live pipeline if it is at least 0.3
below the better single run (<= 6.65). If it passes, two more Y6 runs (about 110 min each) for a four-way average.
Result: not passed. Best 6.84 (mean of the logits, scale 0.35; controls 6.97 / 6.95). The runs share most errors, so
more seeds of the same recipe will not buy much.
Training a new acoustic recipe is last: recipe changes have not moved live WER since October.
**Result (17:13 EST):** (1) 1.5 s at D 12 with the combined 3B: **5.96** (5.96 / 6.09 / 6.07 / 6.11, mean 6.06; D 10 6.07 /
6.10 / 6.02 / 6.26, mean 6.11), 98.9% of words within 1.5 s, mean delay 1,148 ms; without the selector 6.46 (D 10
6.51). Adopted. With conf4 at D 12: 5.99 (mean 6.10). 500 ms at D 1: 6.37 (mean 6.41, D 0's 6.39 / 6.41) with 95.0%
of words in time (p95 499 ms): a tie at the rule's edge, so 500 ms stays at D 0. (2) not passed (above).
Next at 1.5 s: D 13 / 14 may also keep 95% in time with the full system (98.9% at D 12). Next at 500 ms: the Transformer
architecture changes (shorter attention window, causal conv module, size), proposed, waiting for the user.
**Queued 2026-11-20 17:44 EST (user: "go on both"):** (a) D 13 / 14 at 1.5 s (`waits_avg_20261120.py --waits 13,14`;
the longest wait inside the 95% rule becomes the 1.5 s rule, fixed before the D 12 run). (b) Then
`results/harness/queue/tfm_arch_20261120.py`: Y15 (2 s attention window per layer), W1 (causal conv module in every
block), V1 (512 wide, 6 layers), two runs each (seeds as Y6_1 / Y6_2), about 19 h with decoding, done around midday
2026-11-21. Rule set before training (recipe means, average weights): into the live pipeline if best 4-gram WER
<= 6.66 (Y6 6.96 - 0.3), or never-seen-word phoneme error <= 32.75 (Y6 35.75 - 3) with WER <= 7.06.
**(a) result (18:10 EST):** D 13 keeps 97.5% of words within 1.5 s and gives 5.99 (5.99 / 6.10 / 6.05 / 6.15, mean
6.07; without the selector 6.49); D 14 keeps only 92.5% (fails; 6.05). By the rule fixed beforehand the 1.5 s rule is
D 13: 5.99, mean delay 1,227 ms (D 12 gave 5.96 / mean 6.06, not used: choosing it for its WER would be selection on
val-dev). README, charts 06 and 12 updated.
**(b) first runs (2026-11-21 09:15 EST; val-dev, 1,253 sentences; one run each, final / avg weights; best 4-gram WER
over scales 0.2-0.5, greedy PER, never-seen-word phoneme error):** V1_1 (512 x 6) **6.77 / 6.54**, PER 6.98 / 6.85,
never-seen 33.2 / 33.1 (Y6_1 7.27 / 6.97, 8.07 / 7.95, 35.3; Y6_2 avg 6.95, 7.65, 36.2); better in every familiarity
bin (seen 1-4 times 19.7 vs 24.1 / 22.2). Y15_1 (2 s window) 7.39 / 7.09, PER 8.17 / 8.13, never-seen 34.4 / 34.9:
the short window did not help unseen words. W1_1 (conv) diverged after warmup (best in-training val PER at batch
1,000, then blank-only output from about batch 16k; WER 96); W1_2 diverged the same way and was stopped at batch 26.4k.
Y6's lr 0.002 and gradient clip 100 are twice / ten times the X conv runs' settings, the likely cause. Relaunched
09:04 EST as `tfm_arch_20261120.py --seeds s11 --trained win2s win2s d512`: V1_2 training (about 2.5 h), then scoring
of Y15_2 and V1_2, done around 12:30 PM. V1 passes the rule if V1_2 avg is at most 6.78 (recipe mean <= 6.66). Not
re-run: conv at a lower learning rate (a recipe change; for the user).
**Queued 2026-11-21 09:35 EST (user: "okay try V2. also score V1 with the full pipeline. and yes"; the user is away all
weekend, so runs may go at any hour), one after the other:** (1) `results/harness/queue/v1_live_20261121.py`: V1_1 avg
as the decider in Y6_1 avg's place, nothing else changed (scale 0.3, same members, LLMs and selectors), D 0 at 500 ms
and D 12 / 13 / 14 at 1.5 s, four session splits; about 40 min. Rule: V1_1 avg replaces Y6_1 avg at a limit if its
full-system WER, mean of four splits, is lower at that limit's rule wait (1.5 s: the longest inside the 95% rule) with
at least 95% of words in time. (2) V2 (`tfm_arch_20261120.py d512mask5f`): V1 + Y13's short time masks, for a 200 ms
Transformer decider; through the 200 ms full pipeline if its best WER (avg) is below Y13's 8.01, replacing D1_1 if
below 8.06 there with 95% of words within 200 ms. (3) U1 (640 x 6, 30.9M) and T1 (512 x 8, 26.3M), one run each,
scored as each finishes; a second run if run 1 is at least 0.15 below V1's recipe mean. Training about 2.5 / 3.4 / 3.2 h;
all done around 1 AM Sunday.
**(b) second runs (12:25 EST):** V1_2 **6.05 / 6.15** (final / avg), PER 6.79 / 6.67, never-seen 32.7 / 32.8. V1's
recipe mean (avg) **6.35** (6.54 / 6.15) vs Y6 6.96: **passes the rule** (<= 6.66). Y15_2 7.16 / 7.14, never-seen 34.0 /
33.9; Y15's mean 7.12, never-seen 34.4: not passed. U1 / T1 get a second run if run 1 (avg) is at most 6.20 (V1's mean
- 0.15). Added before any V1 live result: V1_2 avg through the same full pipeline after V1_1 avg (`v1_live_20261121.py
--decider s11`), so the recipe's live result is the mean of its two runs; the replace rule stays on V1_1 avg. V2 starts
after it (about 2:00 PM), then U1 and T1.
**V1_1 avg live result (13:16 EST; `results/harness/queue/20261121T143527Z_v1_live/`; full system, standard / random
splits 1-3):** 500 ms D 0 **5.97** (5.97 / 5.79 / 5.82 / 6.05, mean 5.91; Y6 6.39, mean 6.41), 96.8% of words within
500 ms, mean delay 276 ms (Y6 203). 1.5 s: D 13 keeps only 94.0% (fails), so by the rule the wait is D 12: **5.80** (5.80
/ 5.67 / 5.80 / 5.87, mean 5.78; Y6 at D 13 5.99, mean 6.07), 97.7% within, mean delay 1,226 ms. D 14 83.4%. One model
(the per-limit selector): 6.62 / 6.55 (Y6 7.29 / 6.96). Without the selector: 6.63 / 6.26. **Rule met at both limits:
V1_1 avg is the 500 ms and 1.5 s decider; both are under 6% on the standard split and on the mean of four splits.** V1
marks each word about one 80 ms frame later than Y6 (delays about 78 ms longer at every wait), so its D 12 has Y6's
D 13 timing. The selectors were trained on Y6 decisions and used unchanged. Queued before V2
(`v1_extras_20261121.py`): V1's one-model cells with the separate 3B (chart 03, README) and GPT-2 rescoring of V1_1 /
V1_2 avg; charts and README regenerated after it.
**Overnight results (read 2026-11-22 14:42 EST):** V1_2 avg live 5.81 at 500 ms (mean 5.81, 98.2% within, mean delay 208
ms) and 5.61 at 1.5 s D 13 (mean 5.56; D 13 is V1_2's longest wait inside the 95% rule). V1_1 avg one model with the
separate 3B: 6.93 / 6.51 (Y6 7.42 / 7.15). GPT-2 rescoring: V1_1 avg 4.78, V1_2 avg 4.85. V2 6.86 (avg; passes its
gate), U1 (640 x 6) 6.74, T1 (512 x 8) 6.46: neither size step earns a second run (rule <= 6.20). **V2 at 200 ms (15:15
EST, `v2_200_20261122.py`):** full system 6.66 (mean 6.66) vs D1_1 8.06, but 91.0% of words within 200 ms (p95 233 ms;
92.6% without the selector): the rule (95%) is not met; D1_1 stays. Charts, README, figures README and the 595 deck
updated to V1 (2026-11-22).
**Four reporting cells (18:31 EST, `four_cells_20261122.py`):** every headline decider now has one model and full system,
live and at sentence end. Learned sentence-end rescorer, one model / full system: D1_1 5.76 / 5.46, V1_1 avg 4.55 /
**3.88**, V1_2 avg 4.20 / 3.73; V1_2 avg live one model 6.68 / 6.27 (`current_knowledge.md` 6.11).
**T1_1 avg live (2026-11-23, 11:54 PM EST Sunday; `t1_live_20261123.py`, folder `20261123T032447Z_t1_live`):** through
V1's full live pipeline at its best decode scale, 0.25. Expected beforehand: about V1_1 avg's result (acoustic WER 6.46
vs 6.54). Result: full system 500 ms D 0 **5.80** (mean of four splits 5.79; V1_1 avg 5.97 / 5.91), 97.8% within, mean
delay 229 ms; 1.5 s D 13 (the longest wait inside the 95% rule; D 14 90.3%) **5.42** (mean 5.43; V1_1 avg 5.80 / 5.78),
96.7% within, 1,246 ms. Rule met at both limits: **T1_1 avg is the 500 ms and 1.5 s decider.** One model (separate 3B)
6.57 / 6.39; sentence end: GPT-2 L 4.76, learned 4.13 / **3.71**. One run of T1; V1's two runs differ by 0.16 / 0.19 in
the full system, so part of the gain may be seed variation.
**Next (proposed, not queued):** a second run of T1 (T1_2, seed 11; about 3.2 h of training, then this live pipeline,
about 1.5 h) so the decider is a recipe result, as V1's was. Expected: between V1's recipe (5.89 / 5.70) and T1_1.
**Next at 200 ms (proposed, not queued):** V2's lists miss the rule on compute, not accuracy. (a) GPT-2 L ft in fp16 on
V2's lists (fp32 now; OPT in fp16 left WER unchanged) and re-time; (b) a tighter selector gate or none; (c) a second run
of V2 (V1's two runs differed by 0.39 in WER and V1_2 marked words earlier, 208 vs 276 ms mean delay at 500 ms). (a)
and (b) need no training (about 15 min of GPU).
 LoRA on Qwen2.5-1.5B (then 3B): first on the synthetic decisions, then on the fold decisions (the
real decoder's lists); `learning/selector/finetune.py`; `peft` needed (an environment change). About 1-2 h of GPU per
run. At the deadline one prompt of about 100-150 tokens plus the K options: roughly 20-40 ms, inside the 500 ms
budget (may replace OPT).

**1g. Evaluation.** The selector's log-probability of each candidate, and its interaction with the rare flag, as
ranker features for the Y6_1 decider at 500 ms and 1.5 s and for D1_1 at 200 ms (cross-fitted by session halves, four
splits; `learning/selector/ranker_selector.py`), its time charged. Keep rule (set now): live WER at least 0.2 lower
at 500 ms or 1.5 s, with 95% of words within the budget (compute charged), on the standard split and two of three
random splits. Then 1h:
words the selector generates outside the list as extra options (the only route besides step 2 to the never-offered
40%).

**Stop rule.** If neither the probe nor the first fine-tune lowers live WER by 0.1 at any budget, stop and write it
up as a negative result.

### 2. Coverage of rare words (CPU, can run beside 1)

- 2a. Diagnostic (minutes, saved outputs): for each error whose word was absent at its deadline, is the word in the
  decoder lexicon, in the sentence-end 100-best (5.1), and how probable is it under the 4-gram?
- 2b. 100-best deadline lists for the Y6_1 decider at D = 0 and D = 10 (CPU decode; GPT-2 L ft and OPT scoring, about
  twice the LLM time per decision). Keep rule: live WER at least 0.1 lower at 1.5 s, 95% of words within the budget.
- 2c. A less pruned first-pass 4-gram is not buildable here: composing the decoding graph past pruning 1e-8 needs
  about 50 GB of RAM (August build log, `docs/history/log_2026-08.md`; 1e-8 peaked at 25 GB of 27). The unpruned
  4-gram already exists (`results/lm_general_4gram/lm_orig.arpa`, 61 M n-grams, 2 GB OpenWebText; pruned versions at
  3e-9 and 1e-9 beside it), so instead: rescore every deadline candidate with it as a ranker feature (CPU). That
  helps ranking of rare sequences, not coverage.

### 3. A stronger general LLM in the ranker (GPU, about a day)

Qwen2.5-3B (downloaded) scores the saved deadline lists (Y6_1 decider D = 0 and D = 10, D1_1's lists) in fp16; it enters
the ranker with its rare-word interaction, as a third LLM at 1.5 s and in place of OPT at 500 ms (a third LLM made
words late there before). The October Qwen 0.5B / 1.5B test used weights tuned for GPT-2; the ranker now learns
them. Keep rule: live WER at least 0.15 lower at one budget, 95% of words within the budget. Shares the GPU with step 1.

### 4. Y13 as the 200 ms decider (about half a day)

Y13 (Y6_1 with shorter masks) commits at its own word end with p95 120 ms and is 11.21 live before any LLM step (D1_1:
11.57). Deadline lists at D = 0 and D = 1 (CPU decode), GPT-2 L ft and OPT scoring, CTC features (members with at most
one frame of future input: the causal GRUs and the Y6-recipe models), the familiarity ranker on its own lists and
pooled with D1_1's. Keep rule: live WER at least 0.15 below 8.64, with 95% of words within 200 ms (compute charged). Then the
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
- **Real-time Transformer forward pass (needed before any latency claim with a Y model).** Done for the LightBeam
  recipe without attention look-ahead (Y6) on 2026-11-19: the streaming path caches keys and values over the 20 s
  window, streamed logits match offline within 2.5e-5, and a paced CPU replay of `Y6_gen4g` on 2 trials passed the
  harness equivalence check (lag p95 94 ms). Still to do before a latency claim: the harness timing check on the GPU
  over `val-dev-sample`; and a streaming path for attention look-ahead (Y7).
- **Normalization experiment (dormant).** Resolve preprocessing, clipping and reset provenance and amend the
  proposed variance floor (R-D1_2 / R-D3) before that treatment; see `.claude/rules/acoustic-pipeline.md`.

The September milestones (output capture, paced replay, the harness, the first commitment baseline and the first
causal Transformer) are complete; their text is in `docs/history/PLAN_2026-09.md` and their results in
`results/RESULTS.md` sections 1 and 4-6.
