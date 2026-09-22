"""Versioned output traces and retrospective revision accounting.

Version 1 measures cached decoder output. Version 2 uses a paced, scheduled
feature-availability clock across acoustic and LM processes. Neither supplies
speech-word alignment or an online commitment policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class WordTrace:
    """History of a display position, including disappearance (None).

    Positions are not semantic word identities: an insertion can shift several
    positions. Use word_edits for minimum token-edit counts between outputs.
    """

    position: int
    first_frame: int
    final_frame: int
    revisions: int
    final_word: Optional[str]
    values: List[Optional[str]] = field(default_factory=list)


def trace_partials(partials: Sequence[Sequence[str]]) -> List[WordTrace]:
    """Retain every occupied position, even if absent from the final output."""
    traces: Dict[int, WordTrace] = {}
    for frame, hypothesis in enumerate(partials):
        for position in range(max(len(hypothesis), len(traces))):
            word = hypothesis[position] if position < len(hypothesis) else None
            trace = traces.get(position)
            if trace is None:
                traces[position] = WordTrace(position, frame, frame, 0, word, [word])
            elif word != trace.final_word:
                trace.revisions += 1
                trace.final_word = word
                trace.final_frame = frame
                trace.values.append(word)
    return [traces[p] for p in sorted(traces)]


def word_edits(previous: Sequence[str], current: Sequence[str]) -> Dict[str, int]:
    """Minimum edits, separating tail growth from edits to an existing output.

    Ties prefer match, substitution, deletion, then insertion. An insertion after
    consuming the entire previous output is an append; other insertions revise it.
    Repeated words can have ambiguous alignments; this is an operational convention.
    """
    n, m = len(previous), len(current)
    cost = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i]
        for j in range(1, m + 1):
            row.append(min(cost[i - 1][j] + 1, row[j - 1] + 1,
                           cost[i - 1][j - 1] + (previous[i - 1] != current[j - 1])))
        cost.append(row)
    counts = dict(appends=0, insertions=0, deletions=0, substitutions=0)
    i, j = n, m
    while i or j:
        if i and j and previous[i - 1] == current[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i and j and cost[i][j] == cost[i - 1][j - 1] + 1:
            counts["substitutions"] += 1
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["appends" if i == n else "insertions"] += 1
            j -= 1
    return counts


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class _TraceState:
    """Validate cached (v1) and paced feature replay (v2), including completion."""

    def __init__(self) -> None:
        self.started = self.ended = False
        self.active = None
        self.ids = set()
        self.frames = self.last_ns = self.trials = self.outputs = 0
        self.version = 1
        self.index_key = "cache_index"
        self.inputs = self.pending = 0
        self.stage = None
        self.endpoint_seen = False

    def accept(self, record: Dict[str, Any]) -> None:
        if not isinstance(record, dict) or self.ended:
            raise ValueError("Invalid record or content after run_end")
        kind = record.get("record")
        if not self.started:
            version = record.get("schema_version")
            if kind != "run_start" or type(version) is not int or version not in (1, 2):
                raise ValueError("Trace must start with schema_version 1 or 2")
            self.version = version
            self.index_key = "cache_index" if version == 1 else "trial_index"
            expected = ("perf_counter_ns_relative_to_trial_start" if version == 1
                        else "monotonic_ns_relative_to_trial_start")
            if record.get("clock") != expected:
                raise ValueError("Unknown trace clock")
            if not isinstance(record.get("metadata"), dict):
                raise ValueError("Missing run metadata")
            self.started = True
            return
        if kind == "trial_start":
            if self.active is not None:
                raise ValueError("Previous trial has no final output")
            for key in (self.index_key, "n_frames"):
                _integer(record.get(key), key)
            _integer(record.get("day_index"), "day_index", -1)
            if record[self.index_key] in self.ids:
                raise ValueError("Duplicate trial index")
            self.ids.add(record[self.index_key])
            if self.version == 2:
                _integer(record.get("n_bins"), "n_bins")
                for key in ("bin_ns", "window_bins", "stride_bins"):
                    _integer(record.get(key), key, 1)
                expected = max(0, 1 + (record["n_bins"] - record["window_bins"]) // record["stride_bins"])
                if record["n_frames"] != expected:
                    raise ValueError("Trial frame geometry does not match")
            self.active = record
            self.frames = self.last_ns = self.inputs = self.pending = 0
            self.stage = None
            self.endpoint_seen = False
        elif kind in ("input", "endpoint"):
            self._accept_input(record)
        elif kind == "output":
            if self.active is None or record.get(self.index_key) != self.active[self.index_key]:
                raise ValueError("Output does not belong to an active trial")
            for key in ("processing_start_ns", "output_ready_ns"):
                _integer(record.get(key), key)
            if not self.last_ns <= record["processing_start_ns"] <= record["output_ready_ns"]:
                raise ValueError("Output clock must be monotone and nonnegative")
            if self.version == 1 and record.get("input_available_ns") is not None:
                raise ValueError("Cached replay has no measured source-feature availability")
            words = record.get("words")
            if not isinstance(words, list) or any(
                not isinstance(w, str) or not w or len(w.split()) != 1 or w.strip() != w
                for w in words
            ):
                raise ValueError("words must be a list of nonempty tokens")
            if record.get("frame_index") is not None:
                _integer(record["frame_index"], "frame_index")
            if self.version == 2:
                self._accept_pipeline(record)
            if record.get("kind") == "partial":
                if record.get("frame_index") != self.frames or self.frames >= self.active["n_frames"]:
                    raise ValueError("Partial outputs must cover each logit frame in order")
                self.frames += 1
            elif record.get("kind") == "final":
                expected = self.frames - 1 if self.frames else None
                if self.frames != self.active["n_frames"] or record.get("frame_index") != expected:
                    raise ValueError("Final output must follow all trial frames")
                self.active = None
                self.trials += 1
            else:
                raise ValueError("Output kind must be partial or final; no commitment policy is active")
            self.last_ns = record["output_ready_ns"]
            self.outputs += 1
        elif kind == "run_end":
            if self.active is not None:
                raise ValueError("Run ended before the final trial output")
            if record.get("n_trials") != self.trials or record.get("n_outputs") != self.outputs:
                raise ValueError("Run completion counts do not match")
            self.ended = True
        else:
            raise ValueError(f"Unknown trace record: {kind}")

    def _accept_input(self, record):
        if self.version != 2 or self.active is None or self.endpoint_seen or self.pending:
            raise ValueError("Input/endpoint is out of order")
        if record.get("trial_index") != self.active["trial_index"]:
            raise ValueError("Input belongs to a different trial")
        for key in ("input_available_ns", "acoustic_start_ns", "acoustic_ready_ns", "n_emitted"):
            _integer(record.get(key), key)
        if not max(self.last_ns, record["input_available_ns"]) <= record["acoustic_start_ns"] <= record["acoustic_ready_ns"]:
            raise ValueError("Input clock is early or nonmonotone")
        if record["record"] == "input":
            if record.get("bin_index") != self.inputs or self.inputs >= self.active["n_bins"]:
                raise ValueError("Input bins must arrive in order")
            self.inputs += 1
        else:
            if self.inputs != self.active["n_bins"]:
                raise ValueError("Endpoint arrived before all input bins")
            self.endpoint_seen = True
        if record["input_available_ns"] != self.inputs * self.active["bin_ns"]:
            raise ValueError("Input availability must follow the fixed replay schedule")
        if self.frames + record["n_emitted"] > self.active["n_frames"]:
            raise ValueError("Too many emitted frames")
        self.pending = record["n_emitted"]
        self.stage = record
        self.last_ns = record["acoustic_ready_ns"]

    def _accept_pipeline(self, record):
        if self.stage is None:
            raise ValueError("Output has no input/endpoint event")
        for key in ("input_available_ns", "acoustic_start_ns", "acoustic_ready_ns"):
            if record.get(key) != self.stage[key]:
                raise ValueError("Output does not match its acoustic event")
        for key in ("worker_received_ns", "worker_output_ready_ns"):
            _integer(record.get(key), key)
        if not (record["acoustic_ready_ns"] <= record["worker_received_ns"]
                <= record["processing_start_ns"] <= record["worker_output_ready_ns"]
                <= record["output_ready_ns"]):
            raise ValueError("Pipeline timestamps are not ordered on one clock")
        if record.get("kind") == "partial":
            if not self.pending:
                raise ValueError("No emitted frame is pending")
            frame = record.get("frame_index")
            _integer(frame, "frame_index")
            window = (self.active["window_bins"] + frame * self.active["stride_bins"]) * self.active["bin_ns"]
            if record.get("frame_window_available_ns") != window or window > record["input_available_ns"]:
                raise ValueError("Invalid frame-window timestamp")
            expected_phase = "endpoint_flush" if self.endpoint_seen else "stream"
            if record.get("phase") != expected_phase:
                raise ValueError("Incorrect output phase")
            self.pending -= 1
        elif record.get("kind") == "final":
            if not self.endpoint_seen or self.pending or record.get("phase") != "finalization":
                raise ValueError("Final output must follow endpoint and flushed frames")
            if record.get("frame_window_available_ns") is not None:
                raise ValueError("Finalization has an endpoint clock, not a frame-window clock")


class OutputTraceWriter:
    """Write buffered JSONL to a fresh path; failed runs have no run_end marker."""

    def __init__(self, path: Path, metadata: Dict[str, Any], schema_version: int = 1) -> None:
        self.path = Path(path)
        self.schema_version = schema_version
        self.metadata = metadata
        self.state = _TraceState()
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        try:
            self._write(dict(record="run_start", schema_version=self.schema_version,
                             clock=("perf_counter_ns_relative_to_trial_start" if self.schema_version == 1
                                    else "monotonic_ns_relative_to_trial_start"),
                             metadata=self.metadata))
        except Exception:
            self.handle.close()
            raise
        return self

    def _write(self, record: Dict[str, Any]) -> None:
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.state.accept(record)
        self.handle.write(serialized + "\n")

    def start_trial(self, cache_index: int, day_index: int, n_frames: int, **geometry) -> None:
        """The first argument is the cache index in v1, replay trial index in v2."""
        self._write(dict(record="trial_start", **{self.state.index_key: cache_index},
                         day_index=day_index, n_frames=n_frames, **geometry))

    def input(self, *, endpoint=False, **fields) -> None:
        if self.state.active is None:
            raise ValueError("start_trial must precede input")
        self._write(dict(record="endpoint" if endpoint else "input",
                         trial_index=self.state.active[self.state.index_key], **fields))

    def output(self, *, kind: str, frame_index: Optional[int],
               processing_start_ns: int, output_ready_ns: int,
               words: Sequence[str], pipeline=None) -> None:
        if self.state.active is None:
            raise ValueError("start_trial must precede output")
        record = dict(record="output", **{self.state.index_key: self.state.active[self.state.index_key]},
                      kind=kind, frame_index=frame_index, words=list(words),
                      processing_start_ns=processing_start_ns, output_ready_ns=output_ready_ns,
                      input_available_ns=None)
        if pipeline is not None:
            allowed = {"input_available_ns", "acoustic_start_ns", "acoustic_ready_ns",
                       "worker_received_ns", "worker_output_ready_ns",
                       "frame_window_available_ns", "phase"}
            if self.schema_version != 2 or set(pipeline) != allowed:
                raise ValueError("Pipeline fields require the complete version 2 timing boundary")
            record.update(pipeline)
        self._write(record)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self._write(dict(record="run_end", n_trials=self.state.trials,
                                 n_outputs=self.state.outputs))
        finally:
            self.handle.close()
        return False


def _distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return dict(mean=None, p50=None, p95=None, max=None)
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), max=float(max(values)))


def summarize_trial(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe observed display changes; this never drives online decisions."""
    previous: List[str] = []
    totals = dict(appends=0, insertions=0, deletions=0, substitutions=0)
    final_edits = dict(totals)
    for event in events:
        changes = word_edits(previous, event["words"])
        for key in totals:
            totals[key] += changes[key]
        if event["kind"] == "final":
            final_edits = changes
        previous = event["words"]
    traces = trace_partials([event["words"] for event in events])
    stable_ms = [
        (events[t.final_frame]["output_ready_ns"] - events[t.first_frame]["output_ready_ns"]) / 1e6
        for t in traces if t.final_word is not None
    ]
    first = next((e["output_ready_ns"] / 1e6 for e in events if e["words"]), None)
    return dict(
        n_final_words=len(previous), n_ever_visible_positions=len(traces),
        position_revisions=sum(t.revisions for t in traces),
        edit_counts=totals,
        revision_edits=sum(totals[k] for k in ("insertions", "deletions", "substitutions")),
        finalization_edit_counts=final_edits,
        first_output_elapsed_ms=first,
        final_output_elapsed_ms=events[-1]["output_ready_ns"] / 1e6,
        final_position_stabilization_ms=_distribution(stable_ms),
    )


def paced_metrics(events, inputs, bin_ns, budgets_ms):
    """Compute paired-clock lag distributions; never add component percentiles."""
    partials = [e for e in events if e["kind"] == "partial"]
    bins = [e for e in inputs if e["record"] == "input"]
    lags = [(e["output_ready_ns"] - e["frame_window_available_ns"]) / 1e6 for e in partials]
    stream_lags = [(e["output_ready_ns"] - e["frame_window_available_ns"]) / 1e6
                   for e in partials if e["phase"] == "stream"]
    latest_lags = [(e["output_ready_ns"] - e["input_available_ns"]) / 1e6 for e in partials]
    def exceedances(values):
        return {str(b): dict(n=sum(v > b for v in values), denominator=len(values))
                for b in budgets_ms}
    return dict(
        n_partial_outputs=len(partials), n_endpoint_flush_outputs=sum(
            e["phase"] == "endpoint_flush" for e in partials),
        frame_window_to_output_ms=_distribution(lags),
        latest_input_to_output_ms=_distribution(latest_lags),
        input_queue_lag_ms=_distribution(
            [(e["acoustic_start_ns"] - e["input_available_ns"]) / 1e6 for e in bins]),
        acoustic_ms=_distribution(
            [(e["acoustic_ready_ns"] - e["acoustic_start_ns"]) / 1e6 for e in bins]),
        worker_compute_ms=_distribution(
            [(e["worker_output_ready_ns"] - e["processing_start_ns"]) / 1e6 for e in partials]),
        worker_to_coordinator_ms=_distribution(
            [(e["output_ready_ns"] - e["worker_output_ready_ns"]) / 1e6 for e in partials]),
        budget_exceedances=exceedances(lags), stream_budget_exceedances=exceedances(stream_lags),
        endpoint_to_final_ms=(events[-1]["output_ready_ns"] - inputs[-1]["input_available_ns"]) / 1e6,
        elapsed_rtf=(events[-1]["output_ready_ns"] / (len(bins) * bin_ns)) if bins else None,
    )


def summarize_trace(path: Path) -> Dict[str, Any]:
    """Replay one completed trace, keeping at most one trial's events in memory."""
    state = _TraceState()
    trials = []
    events = []
    current = {}
    inputs = []
    metadata = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                state.accept(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid trace at line {line_number}: {exc}") from exc
            kind = record["record"]
            if kind == "run_start":
                metadata = record["metadata"]
            elif kind == "trial_start":
                current = record
                events = []
                inputs = []
            elif kind in ("input", "endpoint"):
                inputs.append(record)
            elif kind == "output":
                events.append(record)
                if record["kind"] == "final":
                    summary = dict(**{state.index_key: current[state.index_key]},
                                   day_index=current["day_index"], **summarize_trial(events))
                    if state.version == 2:
                        summary.update(source=current.get("source"), n_bins=current["n_bins"],
                                       paced=paced_metrics(events, inputs, current["bin_ns"],
                                                           metadata.get("budgets_ms", [])))
                    trials.append(summary)
    if not state.ended:
        raise ValueError("Incomplete trace: missing run_end; do not use it as a completed result")
    return dict(schema_version=state.version, metadata=metadata, n_trials=state.trials,
                n_outputs=state.outputs, revision_edits=sum(t["revision_edits"] for t in trials),
                trials=trials)
