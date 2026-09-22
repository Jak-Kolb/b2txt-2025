# Recorded B2T results

Historical artifact fields were checked on 2026-09-18. A separately labeled
engineering smoke below was run on September 19 and verified September 22. See [the assessment](RESEARCH_ASSESSMENT.md) for validity limitations,
[the current plan](brainstorm/PLAN.md) for next work, and
[the original research record](docs/history/PLAN_2026-08.md) for historical gates.

## Accuracy observations

| Configuration | Evaluation subset | WER (%) | Source under results/ |
| --- | --- | ---: | --- |
| Default 1-gram | All public validation, 1,426 trials | 42.4246 | stream_lm_la0_1gram.json: wer_percent |
| Tuned incremental 4-gram | val-dev, 1,253 trials | 8.4934 | c28_general4g.json: wer_percent |
| 4-gram + gpt2-large, full-utterance rescoring | Same val-dev | 6.8866 | c29_rescore_large_gamma.json: best.wer_percent |
| Same rescored configuration, unseen sentences | val-dev unseen subset | 7.0872 | same: wer_unseen_rescored |
| Incremental 4-gram | Former val-test, 173 trials | 18.8356 | valtest_decode.json: wer_percent |
| + full-utterance rescoring | Same former val-test | 16.0959 | valtest_rescore.json: best.wer_percent |
| Same rescored configuration, unseen sentences | Former val-test unseen subset | 19.3062 | same: wer_unseen_rescored |
| R-D1a bandwidth control, re-tuned 4-gram | val-dev, 1,253 trials | 9.0733 | rd1a_decode.json: wer_percent |

The 42.42-to-6.89 narrative mixed evaluation populations; it is not a paired
val-dev improvement estimate. The 8.4934-to-6.8866 comparison uses the same subset
but includes an utterance-final rescoring stage.

The former val-test is not an independent end-to-end holdout: its sessions'
validation flags were enabled in saved acoustic configurations and their trials
contributed to checkpoint selection. They also have training trials from the same
sessions. Later decoder settings were frozen for the first reported application,
but that does not undo upstream selection. The subsequent top-10 experiment is
exploratory. No further evaluation of that partition is authorized.

## Timing observations

- 4-gram frame compute p95: 0.641249 ms, from
  c28_general4g.json: per_frame_ms.p95. This is cached-logit decoder timing.
- Development n-best extraction p95/max: 21.3004 / 285.6446 ms, from
  c29_oracle_4gram.json: finalize_ms.
- Development neural rescoring p95: 75.0503 ms, from
  c29_rescore_large_gamma.json: latency_ms_per_utterance.p95.
- Former val-test extraction p95: 62.3929 ms; neural rescoring p95: 95.4695 ms,
  from valtest_nbest.json and valtest_rescore.json respectively.

The rescoring timing loop measures at most the first 50 utterances. The old
approximately 96 ms / 158 ms figures add separate component percentiles; they are
not measured percentiles of an integrated pipeline. They exclude endpoint waiting
and cannot establish stable-word delay. The observed development extraction maximum
alone exceeded the historical 140 ms threshold; an empirical maximum also does
not constitute a hard real-time guarantee.

## Findings that need qualification

The archived R-D1a record reports 10.240% best PER versus 10.210% for the symmetric
anchor and 10.040% for the narrower causal anchor. Training summaries were not
recomputed during this review. Treat this as preliminary evidence that bandwidth
matters, not proof that causality is free. One 0.041-point replicate difference
does not define statistical equivalence.

Whole-block normalization is an inference in the historical audit, with unresolved
clipping order. Exact recovery of raw causal features is unproven. The proposed
absolute variance cutoff demonstrably breaks affine invariance.

Claims that all LM approaches are exhausted, that acoustic ties prove biological
undecidability, or that a hand-classified oracle correction defines an achievable
WER floor are withdrawn from the active summary. They exceed the tested evidence.

## Engineering smoke — 2026-09-19, verified 2026-09-22

Source: results/paced_replay_smoke_20260919_02/{manifest.json,summary.json,
output_trace.jsonl}. The manifest marks completion; trace and summary hashes
match, and trace replay reproduces all summary fields derived from the trace.

The existing causal_la0 GRU and tuned 4-gram ran with fixed 20 ms feature arrivals
in separate acoustic/LM processes. Selection was the first two enabled development
validation trials, excluding former val-test day indices 39–44: one session
(t15.2023.08.13), one participant, 33.18 seconds, 409 partial frame updates.
Both streamed logits matched offline execution within 1e-3 (maximum observed error
3.814697265625e-05); final native text matched exactly.

Paired frame-window-availability-to-output lag p95 was 10.3931 and 10.3005 ms
by trial, with maximum 11.4161 ms. These are summary.json:
trials[*].paced.frame_window_to_output_ms, not sums of stage percentiles.
There were zero observed 200 ms processing-budget exceedances out of 409 updates.
The pooled raw-event p95 is 10.3510864 ms (linear percentile of
(output_ready_ns - frame_window_available_ns)/1e6 for partial outputs).
This boundary includes acoustic computation, synchronization, IPC, and native
decoding; excludes raw acquisition/extraction, display, and word alignment.
Loading, synthetic warmup, and trial reset are outside timed replay.

The tiny smoke had zero word edits over 17 reference words and no output revisions.
Seen/unseen text was not stratified. This is a plumbing check, not a general WER
estimate or evidence of improved accuracy. First nonempty output occurred 3.0084
and 3.8848 seconds after trial start; these include recorded lead-in and are not
aligned word delays. Endpoints were dataset-provided and no commitment policy ran.

Native worker peak RSS was 8,403,160 KiB; peak allocated CUDA memory was
405,696,000 bytes (summary.json: resources). The first attempt, retained as
paced_replay_smoke_20260919_01, completed decoding but failed its 5-second teardown
limit. The successful run used a separately bounded 60-second shutdown grace
period. No training, full-data evaluation, or historical-result replacement occurred.
