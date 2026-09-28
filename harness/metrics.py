"""Scoring, uncertainty, paired comparison, structural delay, and queue simulation (numpy only).

Conventions (see .claude/rules/measurement.md):
- WER is pooled edits / reference words after stream_lm.norm_words normalization.
- Intervals resample sessions (the unit that varies); they describe session-sampling uncertainty
  within this scope, not training-seed variability or generalization to new participants.
- Percentiles come from paired per-event values; stage percentiles are never summed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_training.benchmark.stream_lm import levenshtein, norm_words

BIN_NS = 20_000_000
BUDGETS_MS = (200.0, 500.0, 1500.0)
BOOTSTRAP_METHOD = ("session-cluster percentile bootstrap of pooled WER; session-sampling "
                    "uncertainty within this scope, not seed variability")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return dict(n=0, mean=None, p50=None, p95=None, p99=None, max=None)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return dict(n=int(values.size), mean=float(values.mean()), p50=float(p50), p95=float(p95),
                p99=float(p99), max=float(values.max()))


def word_errors(reference, hypothesis):
    ref, hyp = norm_words(reference), norm_words(hypothesis)
    return levenshtein(ref, hyp), len(ref)


def pooled(rows, edits="edits", words="ref_words"):
    e = sum(r[edits] for r in rows)
    w = sum(r[words] for r in rows)
    return dict(percent=100.0 * e / w if w else None, edits=e, ref_words=w, n_trials=len(rows))


def _session_totals(rows, value_key, words_key="ref_words"):
    sessions = sorted({r["session"] for r in rows})
    index = {s: i for i, s in enumerate(sessions)}
    values = np.zeros(len(sessions))
    words = np.zeros(len(sessions))
    for row in rows:
        values[index[row["session"]]] += row[value_key]
        words[index[row["session"]]] += row[words_key]
    return sessions, values, words


def session_bootstrap(rows, value_key="edits", n_boot=10_000, seed=0):
    """95% interval for 100 * sum(value) / sum(words), resampling whole sessions."""
    sessions, values, words = _session_totals(rows, value_key)
    record = dict(method=BOOTSTRAP_METHOD, n_boot=n_boot, seed=seed, n_sessions=len(sessions))
    if len(sessions) < 2:
        return dict(record, ci95=None, note="fewer than two sessions; no interval")
    idx = np.random.default_rng(seed).integers(0, len(sessions), size=(n_boot, len(sessions)))
    stats = 100.0 * values[idx].sum(axis=1) / words[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(record, ci95=[float(lo), float(hi)])


def per_session(rows):
    out = []
    for session in sorted({r["session"] for r in rows}):
        subset = [r for r in rows if r["session"] == session]
        out.append(dict(session=session, **pooled(subset)))
    return out


def seen_unseen(rows):
    return dict(definition="seen = normalized reference text appears in the TRAIN split; "
                           "unseen text is not an unseen participant or session",
                seen=pooled([r for r in rows if r["seen"]]),
                unseen=pooled([r for r in rows if not r["seen"]]))


def accuracy_summary(rows, n_boot=10_000, seed=0):
    finals = sum(r["n_final_words"] for r in rows)
    revisions = sum(r["revision_edits"] for r in rows)
    finalization = {}
    for row in rows:
        for key, value in row["finalization_edit_counts"].items():
            finalization[key] = finalization.get(key, 0) + value
    return dict(
        wer=dict(pooled(rows), **session_bootstrap(rows, n_boot=n_boot, seed=seed)),
        per_session=per_session(rows),
        seen_unseen=seen_unseen(rows),
        per_diagnostic=dict(pooled(rows, "phone_edits", "phone_len"),
                            label="greedy-CTC phoneme error rate on acoustic logits; diagnostic only"),
        revisions=dict(revision_edits=revisions, n_final_words=finals,
                       revision_edits_per_100_final_words=100.0 * revisions / finals if finals else None,
                       finalization_edit_counts=finalization,
                       definition="insertions+deletions+substitutions between consecutive partial "
                                  "outputs (appends excluded); no commitment policy"))


def paired_compare(rows_a, rows_b, n_boot=10_000, seed=0):
    """Paired effect of B relative to A on identical trials and references."""
    keys_a = [(r["session"], r["split"], r["trial_key"], r["ref"]) for r in rows_a]
    keys_b = [(r["session"], r["split"], r["trial_key"], r["ref"]) for r in rows_b]
    if keys_a != keys_b:
        raise ValueError("Paired comparison needs identical ordered trials and references")
    diff_rows = [dict(session=a["session"], delta=b["edits"] - a["edits"], ref_words=a["ref_words"])
                 for a, b in zip(rows_a, rows_b)]
    words = sum(r["ref_words"] for r in rows_a)
    delta = sum(r["delta"] for r in diff_rows)
    sessions, deltas, session_words = _session_totals(diff_rows, "delta")
    interval = session_bootstrap(diff_rows, "delta", n_boot, seed)
    differing = [dict(i=i, session=a["session"], trial_key=a["trial_key"], ref=a["ref"],
                      hyp_a=a["hyp"], hyp_b=b["hyp"], edits_a=a["edits"], edits_b=b["edits"])
                 for i, (a, b) in enumerate(zip(rows_a, rows_b)) if a["hyp"] != b["hyp"]]
    return dict(
        delta_wer_points=100.0 * delta / words if words else None, delta_edits=delta, ref_words=words,
        ci95=interval["ci95"], method=interval["method"], n_boot=n_boot, seed=seed,
        per_session=[dict(session=s, delta_edits=int(d), ref_words=int(w),
                          delta_wer_points=100.0 * d / w if w else None)
                     for s, d, w in zip(sessions, deltas, session_words)],
        trials=dict(improved=sum(r["delta"] < 0 for r in diff_rows),
                    worsened=sum(r["delta"] > 0 for r in diff_rows),
                    unchanged_edits=sum(r["delta"] == 0 for r in diff_rows),
                    different_text=len(differing)),
        differing=differing,
        note="negative delta means B has fewer errors; a single paired run is not a noise floor")


def smoothing_kernel(std, size, lookahead):
    """Numpy mirror of common.smoothing_kernel_from_args / gauss_smooth weights."""
    from scipy.ndimage import gaussian_filter1d
    impulse = np.zeros(size, dtype=np.float32)
    impulse[size // 2] = 1
    kernel = gaussian_filter1d(impulse, std)
    kernel = kernel[kernel > 0.01]
    kernel = kernel / kernel.sum()
    past = kernel.shape[0] // 2
    future = past if lookahead is None else int(lookahead)
    kernel = kernel[: past + future + 1]
    return kernel / kernel.sum(), past, future


def structural_delay(preprocess, patch_size, patch_stride, bin_ms=20.0):
    """Buffering delays implied by the configuration alone (exact for buffering only)."""
    window, stride = (patch_size, patch_stride) if patch_size > 0 else (1, 1)
    if preprocess["smooth_data"]:
        kernel, past, future = smoothing_kernel(preprocess["smooth_kernel_std"],
                                                preprocess["smooth_kernel_size"],
                                                preprocess["smooth_lookahead"])
        offsets = np.arange(-past, future + 1)
        centroid = float(np.dot(kernel, offsets))
    else:
        past = future = 0
        centroid = 0.0
    noncausal = bool(preprocess["smooth_data"] and preprocess["smooth_lookahead"] is None)
    return dict(
        label="structural buffering delay from configuration; excludes the model's learned "
              "emission delay, LM word finalization, commitment, and endpointing",
        bin_ms=bin_ms, noncausal_offline_only=noncausal,
        lookahead_ms=future * bin_ms,
        smoothing=dict(past_bins=past, future_bins=future, support_bins=past + future + 1,
                       centroid_delay_ms=-centroid * bin_ms),
        patch=dict(window_bins=window, stride_bins=stride, window_ms=window * bin_ms),
        first_output_ms=(window + future) * bin_ms,
        update_cadence_ms=stride * bin_ms,
        frame_entry_wait_ms=dict(min=0.0, max=(stride - 1) * bin_ms, mean=(stride - 1) * bin_ms / 2))


def trace_trials(path):
    """Group a schema-2 trace into per-trial step lists (input/endpoint + following outputs)."""
    trials, current, header = [], None, None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = record["record"]
            if kind == "run_start":
                header = record
            elif kind == "trial_start":
                current = dict(start=record, steps=[])
                trials.append(current)
            elif kind in ("input", "endpoint"):
                current["steps"].append(dict(record=record, outputs=[]))
            elif kind == "output":
                current["steps"][-1]["outputs"].append(record)
    if header is None or header.get("schema_version") != 2:
        raise ValueError("Expected a schema-2 trace")
    return trials


def service_times(trial):
    """Start-to-start service per input bin (compressed schedule); includes all bookkeeping."""
    starts = [s["record"]["acoustic_start_ns"] for s in trial["steps"]]
    return np.diff(np.asarray(starts, dtype=np.int64))


def simulate_trial(trial, bin_ns=BIN_NS):
    """Replay measured service times against a paced arrival schedule (single-server FIFO).

    The coordinator is synchronous (acoustic, then one LM round trip per emitted frame), so
    each step starts at max(its arrival, previous start + previous service) and each output
    keeps its measured offset from its step's start.
    """
    start = trial["start"]
    n_bins, window, stride = start["n_bins"], start["window_bins"], start["stride_bins"]
    services = service_times(trial)
    frame_lags, latest_lags, endpoint_to_final = [], [], None
    sim_start = None
    for index, step in enumerate(trial["steps"]):
        arrival = (index + 1) * bin_ns if step["record"]["record"] == "input" else n_bins * bin_ns
        sim_start = arrival if sim_start is None else max(arrival, sim_start + int(services[index - 1]))
        origin = step["record"]["acoustic_start_ns"]
        for output in step["outputs"]:
            ready = sim_start + output["output_ready_ns"] - origin
            if output["kind"] == "partial":
                frame_lags.append((ready - (window + output["frame_index"] * stride) * bin_ns) / 1e6)
                latest_lags.append((ready - arrival) / 1e6)
            else:
                endpoint_to_final = (ready - n_bins * bin_ns) / 1e6
    return frame_lags, latest_lags, endpoint_to_final


def exceedances(values, budgets=BUDGETS_MS):
    return {str(b): dict(n=int(sum(v > b for v in values)), denominator=len(values)) for b in budgets}


def timing_check_summary(trace_path, bin_ns=BIN_NS):
    trials = trace_trials(trace_path)
    bins, cycles, frame_lags, latest_lags, finals = [], [], [], [], []
    for trial in trials:
        stride = trial["start"]["stride_bins"]
        services = service_times(trial) / 1e6
        bins.extend(services)
        cycles.extend(services[i:i + stride].sum() for i in range(0, len(services) - stride + 1, stride))
        frame, latest, final = simulate_trial(trial, bin_ns)
        frame_lags.extend(frame)
        latest_lags.extend(latest)
        if final is not None:
            finals.append(final)
    bin_ms = bin_ns / 1e6
    cycle_ms = bin_ms * (trials[0]["start"]["stride_bins"] if trials else 1)
    return dict(
        service_ms=dict(
            definition="start-to-start time of each synchronous coordinator step, measured "
                       "unpaced (acoustic + CUDA sync + LM IPC/decode + trace bookkeeping)",
            per_bin=distribution(bins), per_frame_cycle=distribution(cycles),
            frac_bins_over_bin=float(np.mean(np.asarray(bins) > bin_ms)) if bins else None,
            frac_cycles_over_cycle=float(np.mean(np.asarray(cycles) > cycle_ms)) if cycles else None,
            bin_budget_ms=bin_ms, cycle_budget_ms=cycle_ms),
        simulated_lag=dict(
            label="simulated from unpaced service times against a 20 ms arrival schedule; "
                  "optimistic (warm hardware, no sleep/wake jitter); not a measured latency",
            frame_window_to_output_ms=distribution(frame_lags),
            latest_input_to_output_ms=distribution(latest_lags),
            exceedances=exceedances(frame_lags),
            endpoint_to_final_ms=distribution(finals)),
        n_trials=len(trials), n_bins=len(bins))


def paced_lags(trace_path, budgets=BUDGETS_MS):
    """Measured pooled lag distributions over all partial outputs of a paced trace."""
    frame, latest, finals = [], [], []
    for trial in trace_trials(trace_path):
        n_bins, bin_ns = trial["start"]["n_bins"], trial["start"]["bin_ns"]
        for step in trial["steps"]:
            for output in step["outputs"]:
                if output["kind"] == "partial":
                    frame.append((output["output_ready_ns"] - output["frame_window_available_ns"]) / 1e6)
                    latest.append((output["output_ready_ns"] - output["input_available_ns"]) / 1e6)
                else:
                    finals.append((output["output_ready_ns"] - n_bins * bin_ns) / 1e6)
    return dict(frame_window_to_output_ms=distribution(frame),
                latest_input_to_output_ms=distribution(latest),
                exceedances=exceedances(frame, budgets), endpoint_to_final_ms=distribution(finals))


def quantiles(values, n=101):
    values = np.asarray(values, dtype=np.float64)
    return [] if values.size == 0 else [float(v) for v in np.quantile(values, np.linspace(0, 1, n))]


def timing_quantiles(trace_path, simulated, bin_ns=BIN_NS, n=101):
    """Quantile curves for CDF charts: simulated (compressed trace) or measured (paced trace)."""
    frame, services = [], []
    for trial in trace_trials(trace_path):
        services.extend(service_times(trial) / 1e6 if simulated else [])
        if simulated:
            frame.extend(simulate_trial(trial, bin_ns)[0])
        else:
            frame.extend((o["output_ready_ns"] - o["frame_window_available_ns"]) / 1e6
                         for s in trial["steps"] for o in s["outputs"] if o["kind"] == "partial")
    return dict(frame_window_to_output_ms=quantiles(frame, n), service_per_bin_ms=quantiles(services, n),
                simulated=simulated)


def busy_times(trial):
    """Per input bin in a paced trace: step start to its last output (or acoustic ready)."""
    out = []
    for step in trial["steps"]:
        record = step["record"]
        if record["record"] != "input":
            continue
        end = max([o["output_ready_ns"] for o in step["outputs"]] or [record["acoustic_ready_ns"]])
        out.append((end - record["acoustic_start_ns"]) / 1e6)
    return out


def paced_summary(trace_path, budgets=BUDGETS_MS):
    """Measured (1x paced) pooled lag distributions plus busy-time and queueing context."""
    busy, queue = [], []
    for trial in trace_trials(trace_path):
        busy.extend(busy_times(trial))
        queue.extend((s["record"]["acoustic_start_ns"] - s["record"]["input_available_ns"]) / 1e6
                     for s in trial["steps"] if s["record"]["record"] == "input")
    out = paced_lags(trace_path, budgets)
    out.update(label="measured on the paced 20 ms schedule: queueing, CUDA sync, IPC, and native decode "
                     "included; raw acquisition, feature extraction, and display excluded",
               busy_per_bin_ms=distribution(busy), input_queue_ms=distribution(queue))
    return out


def sim_calibration(paced, standard_summary, standard_run_id):
    """Measured paced lag vs the queue simulation of the matching standard run (same trials)."""
    measured = paced["frame_window_to_output_ms"]
    timing = standard_summary["timing_check"]
    simulated = timing["simulated_lag"]["frame_window_to_output_ms"]
    keys = ("p50", "p95", "p99", "max")
    return dict(
        standard_run=standard_run_id,
        measured=dict((k, measured[k]) for k in keys), simulated=dict((k, simulated[k]) for k in keys),
        measured_minus_simulated_ms=dict((k, measured[k] - simulated[k]) for k in keys),
        paced_busy_per_bin_ms=dict((k, paced["busy_per_bin_ms"][k]) for k in keys),
        unpaced_service_per_bin_ms=dict((k, timing["service_ms"]["per_bin"][k]) for k in keys),
        note="positive differences mean the simulation was optimistic; busy vs unpaced service shows "
             "how much of the gap is slower steps when the hardware idles between bins")
