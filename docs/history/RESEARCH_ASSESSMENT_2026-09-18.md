# Publication assessment — 2026-09-18

**Judgment:** the project has useful engineering and preliminary results, but the
current evidence does not yet support a publishable claim of end-to-end causal
decoding at a specified user-visible latency. Keep the systems question and narrow
the work. Repair measurement and evaluation before more optimization.

This review inspected the PC source, saved configuration YAMLs, aggregate JSON
results, historical planning, and primary literature. It ran only a small synthetic
normalization counterexample, not training or a scientific dataset evaluation.
Source line references below describe the current code at main 05500b8; no runtime
source was changed.

**Current scope:** the user has retained the original pipeline objective on B2T '25
and the Makin lab's ECoG dataset. Temporal context, neural models, LM search,
commitment, and runtime choices are exploratory levers within that work.
[The current plan](brainstorm/PLAN.md) governs development order. This review
constrains affected experiments and claims; it does not fix a paper thesis or
block instrumentation while independent evaluation or lab access is unresolved.

## 1. What the class pitch gets right

Accuracy at explicit delay budgets is a useful research question. A working replay
pipeline, defensible measurement, and a compact set of controlled comparisons can
support a serious systems/methods submission. Makin lab involvement could strengthen
the evaluation design and supply a distinct replication setting if access is approved.

The labels on the slide need qualification. 200 ms is a proposed design target, not
a universal threshold for natural conversation. 1.5 s is not one comparable latency
of all current streaming decoders. The unconstrained run is a reference configuration,
not a guaranteed accuracy ceiling.

Littlejohn et al. report 80 ms updates and, separately, a 1.56 s median GO-cue-to-text
onset for one task. They also measure per-step deadline success. Those clocks should
not be placed on a single axis without explanation.
[Primary paper](https://www.nature.com/articles/s41593-025-01905-6).

Wairagkar et al. report neural acquisition-to-synthesized-samples processing within
10 ms, with audio playback latency treated separately. Their voice system is not
the same task as stable word transcription. Their feature processing uses a
log transform and past-10-second normalization; the repository's EWMA half-life
implementation is not an exact reproduction of that procedure.
[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12369848/).

LightBeam is directly relevant prior art: it studies memory-efficient CTC decoding,
delayed neural-LM fusion, and an accuracy/RTF tradeoff on B2T '24/'25. It discusses
output jitter and uses a 1.2-second fusion interval for B2T '25. A new paper here
must distinguish its stable-output/deadline evaluation from existing efficiency
work rather than claiming streaming decoding is new. Its numerical WER cannot be
compared directly with this project's custom subsets.
[Primary preprint, March 2026](https://arxiv.org/html/2603.14002v1).

This is a targeted literature check, not a systematic novelty review or a promise
of acceptance. Before committing substantial compute, agree the contribution and
evaluation protocol with the lab.

## 2. Problems that currently block the strong claims

### A. The former holdout participated in acoustic checkpoint selection

rnn_trainer.py:153–174 constructs train/validation input lists for all configured
sessions; :206–218 builds the validation loader. Lines 597–633 evaluate that loader
and select best_checkpoint from aggregate validation PER. Lines 711–726 include
days whose dataset_probability_val flag is one.

The saved args.yaml for causal_la0, causal_la4, and rd1a_bwmatched all contain
[1, 1, 1, 1, 1, 1] at day indices 39–44, precisely the six sessions in splits.py.
Training also uses the train partition from these sessions. Thus these are neither
unseen recording sessions for the neural model nor untouched examples for the
whole model-selection pipeline. The decoder-level split does not fix this.

The old numbers remain observations. Describe the first frozen-decoder application
accurately, disclose upstream selection and later reuse, and stop calling it an
independent end-to-end test. Do not rerun it during cleanup. A future confirmatory
claim needs a frozen evaluation not used in any model/preprocessing/decoder selection.
If that is unavailable, explicitly frame a retrospective exploratory study.
Newly naming a subset of already-examined data does not make it pristine.

### B. The headline result includes utterance-final processing

stream_lm.py:403–481 reads cached logits, consumes every frame, then calls
FinishDecoding and extracts a whole-trial n-best list. neural_rescore.py:115–156
reads those completed lists and scores their sentences. A causal language model
does not turn an utterance-final scoring procedure into incremental stable text.

The system has an incremental 4-gram stage. The 6.8866% WER adds full-utterance
rescoring. Describe both outputs separately. Final WER is not the accuracy a user
would see 200 ms after a word, and known file/trial endings are not a deployed
endpoint detector.

The latency loop times only the first min(50, number_of_trials) lists. Separate
p95 extraction and forward times were added to report approximately 96/158 ms.
Quantiles of sums are not sums of quantiles; there are no paired integrated
measurements in those summaries. The stored development extraction maximum is
285.6446 ms, already beyond the old 140 ms budget.
Sources: results/c29_oracle_4gram.json, c29_rescore_large_gamma.json,
valtest_nbest.json, valtest_rescore.json.

### C. The normalization proof does not justify the proposed experiment

The historical audit inferred block normalization from aggregate statistics,
including 1/sqrt(W) scaling. Such scaling is not unique to block z-scoring; stationary
processes can show it too. Obtain source-level preprocessing provenance rather than
claiming that this diagnostic proves the transform.

The same audit reports an upper clipping rail and says the clipping order is
unverified (brainstorm/audit/AUDIT.md, feature path/F1). Clipping is not invertible.
Cancellation of an affine transform does not recover samples lost to a nonlinear
transform, and per-trial resets/gaps must match the deployment claim.

The August 18 proposed floor is mathematically wrong as an invariance claim:
std(a*x+b) = |a|*std(x), so a fixed 0.001 threshold can select different branches.
Using the existing rolling_normalize implementation on 80 alternating samples
[-0.0005, +0.0005], then applying the proposed zero-below-threshold rule:
at t=50 the original std is 0.0005 and output is zero; scaling by three gives
std 0.0015 and output approximately -0.997785. This synthetic counterexample is
reproducible without research data. The prior check excluded low-variance samples,
so its reported pass does not validate the new threshold branch.

Hold R-D2/R-D3 until the algorithm, causality scope, and gate are amended. A causal
transform of already-released features is a useful experiment even if exact raw
equivalence cannot be established; just do not call it recovered raw causal data.
A small performance change after re-normalization would not make the old noncausal
preprocessing causal or retroactively validate all earlier latency claims.

### D. Statistical and interpretive claims exceed the evidence

- One 0.041-point same-configuration difference estimates neither a noise floor nor
  an equivalence bound. The R-D1a control is useful but a small single-seed delta
  is not proof that causality is free.
- The three-number bandwidth/causality decomposition closes by subtraction.
  Algebraic closure alone supplies no evidence of causal identification.
- Two normalization runs at different seeds compared to one anchor confound
  treatment and seed variability. Use paired seeds and pre-specified uncertainty.
- Trial bootstrap intervals ignore session dependence. Report paired per-session
  effects and use session-aware uncertainty; with few sessions, show the individual
  effects and limited precision instead of implying participant-level replication.
- Similar headroom-recovery percentages do not establish absence of overfitting.
  Score ties are properties of this representation/decoder, not a proof that no
  neural system could distinguish the intended words.
- A failed sweep bounds the tested family. It does not prove that larger models,
  fusion, or all reranking are impossible. Remove those absolute prohibitions.

### E. The old improvement chain mixed denominators

results/stream_lm_la0_1gram.json reports 42.4246% on 1,426 trials; the 8.4934% and
6.8866% files use 1,253 val-dev trials. The complete 42.42-to-6.89 chain was labeled
val-dev in several documents. Keep populations explicit and use the same subset
for comparisons. The existing 8.4934-to-6.8866 comparison does share a subset.
RESULTS.md now lists source fields and limitations instead of a misleading headline.

## 3. Minimal defensible paper

Provisional contribution: a reproducible study of how preprocessing, output
commitment, and finalization affect accuracy, text revision, and deadline success
for intracortical speech decoding on commodity hardware.

The essential artifact is one integrated, timestamped replay harness. Measure
from feature availability to provisional and committed output, include queues,
device synchronization, endpoint waiting and finalization, and retain per-event
records. Raw acquisition/extraction remain explicitly outside the boundary when
using precomputed B2T features.

For word delay, define a reference event. B2T provides sequences without
ground-truth word alignment; a CTC alignment or retrospective stable-prefix time
is a proxy, not the participant's intended word time. An offline alignment may
help score a metric, but cannot supply information to the online decoder.
If true alignment is unavailable, report processing lag, first-output time,
revision/stabilization metrics, and finalization with their actual definitions;
do not relabel them intent-to-word latency.

Use the current incremental decoder, a small number of bounded settings, and a
full-utterance reference; include a relevant modern decoder if feasible.
Report WER at the actual committed output, provisional revisions, resource use,
deadline-miss fraction, p50/p95/p99 and observed max. RTF is supplementary.
No empirical max proves a hard real-time bound.

For the class, the public-data implementation and transparent limitations are a
substantial deliverable. For publication, independent evaluation and a clear
increment over the prior art matter more than adding models. Lab ECoG replication
could strengthen generality, but requires a suitable encoder/preprocessing and
its own protocol; intracortical spike features do not transfer directly.
Do not make the semester dependent on unconfirmed access.

## 4. Cleanup and preservation

The active plan and instructions now contain current work and safeguards only.
The complete previous PLAN.md and RESULTS.md were preserved as historical records.
The old candidate ranking, recommendations, open-question list, execution ledger,
early proposals, and candidate JSON were removed from the working tree; they
conflicted with later decisions or supplied unsupported predicted effect sizes.
The run-specific destructive R-D1a watchdog was also removed after backup.

Pre-cleanup files, hashes, Git status, refs and working diff are under
.git/context-cleanup/2026-09-18/. This is a same-disk recovery copy, not an off-machine
backup. No data, checkpoints, scientific result files, branches, or runtime
configuration were deleted. No source algorithm was patched, experiment launched,
commit made, or push performed.
