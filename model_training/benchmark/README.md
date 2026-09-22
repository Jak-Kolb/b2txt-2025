# Decoder output tracing

Run development commands on JakPC from /home/fishinjak/code/nejm-brain-to-text.
The new trace path uses synthetic tests and does not require training.

## Offline regression checks

    .venv/bin/python -m unittest discover -s tests -v
    /home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python -m unittest discover -s tests -v

Verified 2026-09-22: 52 tests passed in .venv; 49 passed and three were skipped
in the LM environment. Compile, paced-replay/worker CLI, and diff checks passed.

Offline tests use generated arrays, a fake LM worker, and a small random GRU.
The legacy acoustic metric, actual GRU adapter, and HDF5 selector checks require
the Python 3.10 acoustic stack and are skipped in the Python 3.9 LM environment.
They cover scheduling/backlog, frame order, equivalence, protocol failures, timeout
cleanup, and separate shutdown grace. Native validation is recorded below.

## Capturing and replaying output

For an authorized development-data smoke test, supply the actual cache and LM
directories and fresh output paths:

    /home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python model_training/benchmark/stream_lm.py decode \
      --cache /path/to/logits.npz --lm /path/to/lm \
      --split dev --n_trials 2 --capture_partials \
      --trace_out results/new_trace.jsonl --out results/new_decode.json

Both capture flags are required. Decode outputs refuse to overwrite existing
files. The historical test partition remains exposed; its CLI name does not
make it an independent holdout.

Recompute a saved trace summary without loading an LM or dataset:

    .venv/bin/python model_training/benchmark/stream_lm.py trace-summary \
      --trace results/new_trace.jsonl --out results/new_summary.json

A JSONL file has a versioned run header, trial headers, one partial output per
logit frame, one final output per trial, and a completion record. Interrupted
traces remain inspectable but the summary command rejects them.

## Cached-logit trace clocks (schema 1)

- Timestamps are measured monotonic nanoseconds relative to the start of each
  decoder trial, before reset. Each output has processing-start and output-ready
  times. Output-ready follows result retrieval/tokenization and precedes trace
  serialization; bookkeeping contributes to later elapsed times.
- Input is an unpaced logit cache. Source-feature availability is null, and trial
  ends are provided by the dataset. No acoustic, queueing, speech-to-word, or
  end-to-end deadline claim follows from these timestamps.
- Raw output tokens are retained, including empty outputs and finalization
  changes. No commitment policy runs; finalization and online commitment differ.
- Token edit counts distinguish tail appends from internal insertions, deletions,
  and substitutions, using deterministic minimum-edit alignment. Appending new
  text is growth; revision_edits counts the other three operations.
- Display-position histories also count disappearance and reappearance. Positions
  are not persistent word identities. Their retrospective stabilization intervals
  describe the observed output sequence, not a guarantee available online.
- The legacy stability_metrics API now reports metric_version 2 and retains
  retracted positions. Its flicker_rate denominator is all ever-visible positions;
  final-word timing excludes positions absent at the end. Old saved metrics are
  unchanged and should not be silently mixed with version 2.

Trace headers include decoding settings, cache/reference-sidecar hashes, source
hashes, and original cache/session indices. The LM is identified by its resolved
path; preserve its immutable graph/config provenance for future scientific
comparisons. Keep traces under ignored results paths: hypotheses may contain
subject content.

## Paced integrated replay (schema 2)

For an authorized bounded development run, use a fresh output directory:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u \
      model_training/benchmark/paced_replay.py \
      --out_dir results/new_paced_smoke --n_trials 2 --max_trial_seconds 60 \
      --device cuda --shutdown_timeout 60

Defaults use the existing causal_la0 checkpoint, tuned 4-gram graph, and separate
Python 3.9 native LM environment. This is a synchronous two-process baseline.
The first N enabled validation trials exclude former val-test day indices 39–44.
Overlong trials fail the bound rather than being silently truncated.

Features become available on a fixed 20 ms bin-end schedule. Delayed computation
does not reset that schedule: queueing/backlog remains visible. Acoustic CUDA
completion is synchronized, and IPC/native decoder/coordinator events share the
Linux monotonic clock. Trial reset, model/graph loading, and synthetic warmup are
outside the timed boundary. Reference text is read only after timed decoding.
Native output receives one ordered logit frame per request, without references.

Schema 2 adds input/endpoint records and source-availability timestamps to partial
and final events. Frame-window-to-output lag includes any acoustic lookahead;
latest-required-input-to-output lag isolates processing after all needed input.
The summary reports both per trial, alongside queueing, stage times, first output,
revisions, provided-endpoint-to-final time, budget exceedances, and paced RTF.
Paced RTF includes deliberate waiting and is not a maximum-throughputput estimate.
The 200/500/1500 ms thresholds apply to frame processing lag, not word latency.
No online commitment policy or online endpoint detector is implemented.

Each run preserves manifest.json, output_trace.jsonl, summary.json, and the worker
stderr log. Manifest hashes identify checkpoint, arguments, LM graph/vocabulary,
input files, relevant source files, and final trace/summary. A complete manifest
and valid trace completion marker distinguish success from interrupted runs.
The trace header is the initial metadata snapshot and can still say running.
The existing trace-summary command also reads schema 2 without native/data access;
it reproduces event-derived metrics, not post-run accuracy/resource additions.

The September 19 smoke in results/paced_replay_smoke_20260919_02 completed two
trials and 409 updates. Streamed/offline logits and native final text agreed;
per-trial processing-lag p95 was 10.39/10.30 ms. See ../../RESULTS.md for sources
and limits. This verifies the released-feature integration boundary only.
Raw-feature causality, speech-to-word delay, larger-sample accuracy, and lab ECoG
integration remain unresolved. Keep traces private under ignored results paths.
