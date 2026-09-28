# Pipeline harness

Benchmark, compare, and watch B2T decoding pipelines. Run from the repository root on JakPC
with the acoustic environment; outputs go to ignored `results/harness/`.

    .venv/bin/python -m harness bench la0_gen4g            # standard benchmark (default)
    .venv/bin/python -m harness bench la0_gen4g --limit 5  # smoke test (flagged, hidden in lists)
    .venv/bin/python -m harness runs                        # list runs
    .venv/bin/python -m harness compare RUN_A RUN_B         # paired comparison, same scope only
    .venv/bin/python -m harness bench la0_gen4g --tier paced # 1x paced timing on the 70-trial sample (~25 min)
    .venv/bin/python -m harness serve                       # viewer + live player on 127.0.0.1:8765
    .venv/bin/python -m harness train --from causal_la0 --name wide1024 \
        --set model.n_units=1024 --question "Does width help?" --then-bench la0_gen4g
    .venv/bin/python -m harness train-status

From the Mac: `ssh -L 8765:127.0.0.1:8765 jakpc`, then open http://127.0.0.1:8765.

## Pipelines

A pipeline is an acoustic model + preprocessing + LM graph + eight native decode settings.
Presets live in `registry/pipelines/`; components in `registry/acoustic/` and `registry/lm/`.
Ad-hoc compositions: `bench --acoustic causal_la4 --lm gen4g --decode acoustic_scale=0.5`.
Preprocessing is the model's training smoothing; `--preprocess smooth_lookahead=2` is allowed
but flagged `preprocess_mismatch`. Decode settings depend on the acoustic model; a preset
records where its settings were tuned (`decode_tuned_on`).

## Standard benchmark (about 5-9 minutes)

1. **Accuracy** on val-dev (1,253 trials, 35 sessions): per-trial fp32 logits (cached under
   `results/harness/logits/`, reused while model, preprocessing, scope, and code are unchanged)
   decoded by `model_training/benchmark/stream_lm.py decode` in the LM environment, one frame at
   a time with every partial output recorded. Reports pooled WER with a session-bootstrap 95%
   interval, per-session WER, seen/unseen WER, revisions, and diagnostic greedy PER.
2. **Timing check** on `val-dev-sample` (2 trials per session): the real streaming path
   (`paced_replay.replay_trial`, the same code as paced and live runs) with arrivals compressed
   to 1 ns. Reports structural buffering delay from the config, start-to-start service times
   (p50/p95/p99/max, overruns of the 20 ms bin and 80 ms frame cycle), a single-server queue
   simulation of lag against the 20 ms schedule, and streamed-vs-batched logit equivalence.

The simulated lag is optimistic (warm hardware, no sleep/wake jitter). Paced 1x runs measure
it (`--tier paced`) and report the calibration gap. Endpoints are the dataset's trial ends
(oracle); no word-commitment policy runs.

## Paced runs (occasional)

`--tier paced` replays `val-dev-sample` at the real 20 ms cadence (`paced_replay.replay_trial`
with its hard streamed/offline logit and native-text checks) and reports measured
frame-window-to-output lag, busy time per bin, input queueing, and endpoint-to-final time.
`sim_calibration` compares it with the simulated lag of the latest standard run of the same
pipeline on the same trials. Use it after changes that affect runtime; paced runs refuse to
start while a harness-launched training run is active. Full val-dev (`--scope val-dev-full`)
is about 6.3 hours; reserve it for pipelines frozen for a claim, on an otherwise idle machine.

## Live player (`serve`, Live tab)

Pick a preset (or compose acoustic model + LM + decode settings, optionally overriding the
smoothing lookahead, flagged as a train/inference mismatch), load it, then play a trial, a
session from a trial onward, or a random trial. The engine (`live_engine.py`, a separate
process) streams bins at 20 ms through the same replay code as paced runs and records each
played trial under `results/harness/live/<start>/` (stopped trials stay incomplete). The page
shows threshold-crossing counts per 20 ms bin (integer counts reconstructed from the released
block-z-scored features; not spike times), spike-band power, per-array activity, pinned
electrode traces, the greedy phoneme stream, partial and final text beside the reference
(hidden in blind mode; the engine never sees it), and frame processing lag. Speeds other than
1x, or playback during a benchmark, are labeled display only.

## Retraining (`train`)

`train` forks a registered model's saved config with OmegaConf dotlist overrides (unknown keys
are errors), writes it with fresh absolute output paths under `model_training/trained_models/`,
runs a smoke (default 50 batches with two validations: exit status, finite loss and PER,
checkpoint written, streamed-vs-offline logits), then launches a detached supervisor. On
success the supervisor registers `registry/acoustic/<name>.yaml` (provenance, question, git
state, and the trainer's selection data) and optionally runs a standard benchmark with the
given preset's LM and decode settings (flagged `decode_tuned_for_other_acoustic`). Records live
in `results/harness/train/<id>/`. `--smoke-only` stops after the smoke.

## Labels that travel with every number

- `acoustic_selected_on_scope`: the trainer picked the checkpoint by val PER over all val
  sessions, including val-dev. `decode_tuned_on_scope`: decode settings were swept on val-dev.
  All val-dev numbers are exploratory development results, not independent evaluation.
- `former-val-test-full` is refused unless `--allow-exposed "<reason>"` is given.
- `smoke_limited`, `preprocess_mismatch`, `offline_noncausal` (no streaming/timing),
  `decode_tuned_for_other_acoustic`, `decode_tuning_unrecorded`.
- Intervals resample sessions: session-sampling uncertainty within the scope, not seed
  variability. Compare pipelines only on identical scope hashes (`compare` enforces this).

## Run directory

`results/harness/runs/<UTC>_<pipeline>_<tier>/`: `manifest.json` (resolved spec, file hashes,
scope with ordered trials, flags, git state and `git_diff.patch` if dirty, status),
`output_trace.jsonl` (schema 1 accuracy trace), `timing_trace.jsonl` (schema 2, compressed
schedule: its lag fields are not lag), `lm_decode.json`, `trials.jsonl`, `summary.json`, logs.
Runs never overwrite; a failed run keeps its files and a `traceback.txt`.

## Resources

One benchmark at a time (`locks/job.lock`). The timing check, paced runs, and live trials
never overlap (`locks/paced.lock`). An LM process starts only if MemAvailable exceeds its
expected RSS + 2 GB (the general 4-gram needs about 8.5 GB of the machine's 27 GB).

## Tests

    .venv/bin/python -m unittest discover -s tests -v

Harness tests use synthetic HDF5, a tiny CPU GRU, and `tests/fake_native/lm_decoder.py` in
place of the native decoder; they skip themselves in the Python 3.9 LM environment.
