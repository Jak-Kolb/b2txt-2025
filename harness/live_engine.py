"""Live replay engine: a .venv process holding one loaded pipeline and a warm LM worker.

JSON lines on stdin (commands) and on a dedicated protocol stdout (events). Every played
trial runs through paced_replay.replay_trial, the same code as paced benchmarks, and is
recorded as a schema-2 trace under results/harness/live/<engine start>/. The engine never
reads reference text; the server adds it for display.

Commands: {"op": "load", "pipeline": {...}} | {"op": "unload"} |
          {"op": "play", "items": [{split, session, trial_key}], "speed": 1|2|4|"max", "gap_s": s} |
          {"op": "stop"}
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os
from pathlib import Path
import queue
import sys
import threading
import traceback

import numpy as np

from . import registry
from .scopes import DATA_DIR, Trial, partition_of
from .store import (DEFAULT_ROOT, HashCache, env_info, flocked, fresh_dir, lock_held, lock_path, ram_guard,
                    utc_stamp)

ARRAYS = [dict(name="ventral 6v", first=0, n=64), dict(name="area 4", first=64, n=64),
          dict(name="55b", first=128, n=64), dict(name="dorsal 6v", first=192, n=64)]  # repo README.md
N_ELECTRODES = 256
CLIP_RAIL = 10.0
SATURATED = 255
SBP_SCALE = 127 / 4.0  # int8 carries z in [-4, 4]
SPEEDS = {1: 20_000_000, 2: 10_000_000, 4: 5_000_000, "max": 1}


class Stopped(Exception):
    """Raised inside the replay loop when the user stops playback."""


def tx_counts(tx, rail=CLIP_RAIL, tol=1e-3):
    """Integer threshold-crossing counts from whole-block z-scored TX features (display only).

    Within a trial each electrode's values lie on a lattice min + k*step (k = counts relative
    to the trial minimum). Values on the clip rail are marked SATURATED; electrodes that are
    not lattice-consistent are reported for fallback display.
    """
    counts = np.zeros(tx.shape, dtype=np.uint8)
    fallback = []
    for e in range(tx.shape[1]):
        column = tx[:, e]
        saturated = column >= rail - 1e-6
        values = np.unique(column[~saturated])
        counts[saturated, e] = SATURATED
        if values.size < 2:
            continue
        k = (column[~saturated] - values[0]) / np.diff(values).min()
        if not np.allclose(k, np.round(k), atol=tol) or k.max() >= SATURATED:
            fallback.append(e)
            counts[~saturated, e] = 0
            continue
        counts[~saturated, e] = np.round(k).astype(np.uint8)
    return counts, fallback


def neural_display(features):
    """Per-bin display payloads computed before the timed loop; never fed to the model."""
    counts, fallback = tx_counts(features[:, :N_ELECTRODES])
    sbp = np.clip(np.round(features[:, N_ELECTRODES:2 * N_ELECTRODES] * SBP_SCALE), -127, 127).astype(np.int8)
    return dict(tx=counts, sbp=sbp, fallback=fallback)


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def top_phonemes(logits, names, k=3):
    z = np.asarray(logits, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(p)[::-1][:k]
    return [[names[i].strip() or "|", round(float(p[i]), 3)] for i in order]


class Sink:
    """Bounded event queue drained by a writer thread; only bin events may be dropped."""

    def __init__(self, stream, maxsize=4096):
        self.stream = stream
        self.queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def emit(self, event, droppable=False):
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            if droppable:
                self.dropped += 1
            else:
                self.queue.put(event, timeout=5)

    def _drain(self):
        while True:
            event = self.queue.get()
            if event is None:
                return
            self.stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.stream.flush()

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=5)


class TrackedLM:
    """LMClient proxy that knows whether a trial is open (a stop must send finish, not reset)."""

    def __init__(self, client):
        self.client = client
        self.active = False

    def request(self, op, **payload):
        response = self.client.request(op, **payload)
        if op == "reset":
            self.active = True
        elif op == "finish":
            self.active = False
        return response


class ObservingAcoustic:
    """Wraps AcousticAdapter: honors stop between bins and keeps emitted rows for display."""

    def __init__(self, inner, stop):
        self.inner, self.stop = inner, stop
        self.pending = collections.deque()

    def process_bin(self, row):
        if self.stop.is_set():
            raise Stopped()
        rows = self.inner.process_bin(row)
        self.pending.extend(rows)
        return rows

    def finish(self):
        if self.stop.is_set():
            raise Stopped()
        rows = self.inner.finish()
        self.pending.extend(rows)
        return rows


class TeeWriter:
    """Writes each trace record first, then derives a display event from it."""

    def __init__(self, writer, acoustic, display, sink, play_id, phonemes):
        self.writer, self.acoustic, self.display = writer, acoustic, display
        self.sink, self.play_id, self.phonemes = sink, play_id, phonemes

    def start_trial(self, *args, **kwargs):
        self.writer.start_trial(*args, **kwargs)

    def input(self, *, endpoint=False, **fields):
        self.writer.input(endpoint=endpoint, **fields)
        if not endpoint:
            i = fields["bin_index"]
            self.sink.emit(dict(type="bin", play_id=self.play_id, bin=i, tx=b64(self.display["tx"][i]),
                                sbp=b64(self.display["sbp"][i]), avail_ms=fields["input_available_ns"] / 1e6,
                                start_ms=fields["acoustic_start_ns"] / 1e6, ready_ms=fields["acoustic_ready_ns"] / 1e6),
                           droppable=True)

    def output(self, *, kind, frame_index, processing_start_ns, output_ready_ns, words, pipeline=None):
        self.writer.output(kind=kind, frame_index=frame_index, processing_start_ns=processing_start_ns,
                           output_ready_ns=output_ready_ns, words=words, pipeline=pipeline)
        if kind == "partial":
            logits = self.acoustic.pending.popleft()
            top = top_phonemes(logits, self.phonemes)
            self.sink.emit(dict(
                type="frame", play_id=self.play_id, frame=frame_index, phase=pipeline["phase"], words=list(words),
                lag_ms=(output_ready_ns - pipeline["frame_window_available_ns"]) / 1e6,
                latest_lag_ms=(output_ready_ns - pipeline["input_available_ns"]) / 1e6,
                worker_ms=(pipeline["worker_output_ready_ns"] - processing_start_ns) / 1e6,
                phone=top[0][0] if top[0][0] != "BLANK" else None, top=top))
        else:
            self.sink.emit(dict(type="final", play_id=self.play_id, words=list(words),
                                endpoint_to_final_ms=(output_ready_ns - pipeline["input_available_ns"]) / 1e6))


class Engine:
    def __init__(self, root, data_dir, lm_python, device, sink, registry_root=registry.REGISTRY_DIR):
        self.root, self.data_dir, self.lm_python, self.device = Path(root), Path(data_dir), lm_python, device
        self.registry_root = registry_root
        self.sink = sink
        self.stop = threading.Event()
        self.worker = None
        self.live_dir = fresh_dir(self.root / "live" / utc_stamp())
        (self.live_dir / "engine.json").write_text(json.dumps(dict(
            pid=os.getpid(), env=env_info(), lm_python=lm_python, device=device,
            note="one schema-2 trace per played trial; stopped trials stay incomplete"), indent=2))
        self.spec = self.identity = self.runtime = self.lm = None
        self.acoustic_key = self.lm_key = None
        self.plays = self.loads = 0
        self.state = "idle"

    # -- events -----------------------------------------------------------------------------
    def emit_state(self, state, message=None):
        self.state = state
        self.sink.emit(dict(type="state", state=state, message=message,
                            pipeline=None if self.spec is None else dict(
                                name=self.spec["name"], acoustic=self.spec["acoustic"]["name"],
                                lm=self.spec["lm"]["name"], decode=self.spec["decode"],
                                preprocess=self.spec["preprocess"], pipeline_hash=self.identity["pipeline_hash"]),
                            lm_expected_rss_gb=None if self.lm is None else self.spec["lm"]["expected_rss_gb"],
                            live_dir=str(self.live_dir)))

    # -- commands ---------------------------------------------------------------------------
    def handle(self, command):
        op = command.get("op")
        if op == "stop":
            self.stop.set()
            return
        if self.worker is not None and self.worker.is_alive():
            self.sink.emit(dict(type="error", message=f"Busy ({self.state}); stop playback first"))
            return
        target = {"load": self.load, "unload": self.unload, "play": self.play}.get(op)
        if target is None:
            self.sink.emit(dict(type="error", message=f"Unknown op {op!r}"))
            return
        self.worker = threading.Thread(target=self._run, args=(target, command), daemon=True)
        self.worker.start()

    def _run(self, target, command):
        try:
            target(command)
        except Exception as exc:
            (self.live_dir / f"error_{utc_stamp()}.txt").write_text(traceback.format_exc())
            self.recover_lm()
            self.sink.emit(dict(type="error", message=f"{type(exc).__name__}: {exc}"))
            self.emit_state("ready" if self.lm is not None and self.runtime is not None else "idle")

    def recover_lm(self):
        """After a failure, keep the LM worker only if it is alive and back between trials."""
        if self.lm is None:
            return
        try:
            if self.lm.client.process.poll() is None and self.lm.active:
                self.lm.request("finish")
            if self.lm.client.process.poll() is None:
                return
        except Exception:
            pass
        self.lm.client._stop()
        self.lm = self.lm_key = None

    def load(self, command):
        pipeline = command.get("pipeline") or {}
        spec = registry.resolve(pipeline.get("preset"), acoustic=pipeline.get("acoustic"), lm=pipeline.get("lm"),
                                decode=pipeline.get("decode"), preprocess=pipeline.get("preprocess"),
                                root=self.registry_root)
        if not spec["streaming_capable"]:
            raise ValueError("offline_noncausal pipeline (symmetric smoothing) cannot stream")
        self.emit_state("loading", f"Loading {spec['name']}")
        identity = registry.identity(spec, HashCache(self.root))
        if identity["acoustic_key"] != self.acoustic_key:
            from .acoustic import Runtime
            self.runtime = None
            self.runtime = Runtime(spec, self.device)
            self.acoustic_key = identity["acoustic_key"]
        lm_key = (spec["lm"]["graph_dir"], json.dumps(spec["decode"], sort_keys=True))
        if lm_key != self.lm_key:
            self.close_lm()
            from model_training.benchmark.lm_worker import LMClient
            from .bench import lm_worker_command
            ram_guard(spec["lm"]["expected_rss_gb"], f"LM worker {spec['lm']['name']}")
            self.loads += 1
            client = LMClient(lm_worker_command(spec, self.lm_python),
                              self.live_dir / f"lm_worker_{self.loads}.log", 30.0, 300.0, shutdown_seconds=60.0)
            client.__enter__()
            if client.ready["n_classes"] != spec["model"]["n_classes"]:
                client.__exit__(None, None, None)
                raise ValueError("Acoustic and LM class counts differ")
            self.lm, self.lm_key = TrackedLM(client), lm_key
        self.spec, self.identity = spec, identity
        with (self.live_dir / f"load_{self.loads}_{utc_stamp()}.json").open("x") as handle:
            json.dump(dict(pipeline=spec, identity=identity), handle, indent=2)
        self.emit_state("ready", f"Loaded {spec['name']}")

    def close_lm(self):
        if self.lm is not None:
            client, self.lm, self.lm_key = self.lm.client, None, None
            client.__exit__(None, None, None)

    def unload(self, command=None):
        self.close_lm()
        self.runtime = self.spec = self.identity = self.acoustic_key = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.emit_state("idle", "Unloaded")

    def play(self, command):
        if self.lm is None or self.runtime is None:
            raise RuntimeError("Load a pipeline first")
        speed = command.get("speed", 1)
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {list(SPEEDS)}")
        gap_s = float(command.get("gap_s", 1.5))
        self.stop.clear()
        items = command.get("items") or []
        for n, item in enumerate(items):
            if self.stop.is_set():
                break
            try:
                with flocked(lock_path(self.root, "paced"), blocking=False):
                    self.emit_state("playing", f"{item['session']} {item['trial_key']} ({n + 1}/{len(items)})")
                    self.play_one(item, speed)
            except BlockingIOError:
                raise RuntimeError("A paced benchmark or timing check is running; live play would perturb it")
            if gap_s and n + 1 < len(items):
                self.stop.wait(gap_s)
        self.emit_state("ready", "Stopped" if self.stop.is_set() else "Playback finished")

    def play_one(self, item, speed):
        from model_training.benchmark.output_trace import OutputTraceWriter, summarize_trace
        from model_training.benchmark.paced_replay import (AcousticAdapter, frame_geometry, replay_trial,
                                                           synthetic_warmup)
        from model_training.evaluate_model_helpers import LOGIT_TO_PHONEME
        split, session, trial_key = item["split"], item["session"], item["trial_key"]
        trial = Trial(session, split, trial_key, str(self.data_dir / session / f"data_{split}.hdf5"))
        runtime, spec = self.runtime, self.spec
        features, _ = runtime.features(trial)
        display = neural_display(features)
        day = runtime.day_index(session)
        warmup = synthetic_warmup(runtime.model, runtime.args, runtime.device, day, self.lm)
        from .train import active_training
        contention = dict(job_running=lock_held(self.root, "job"), training_running=bool(active_training(self.root)))
        bin_ns = SPEEDS[speed]
        timing_valid = speed == 1 and not any(contention.values())
        self.plays += 1
        play_id = self.plays
        path = self.live_dir / f"{play_id:04d}_{session}_{trial_key}.jsonl"
        n_bins = int(features.shape[0])
        n_frames, window, stride = frame_geometry(n_bins, runtime.patch_size, runtime.patch_stride)
        partition = partition_of(session, split)
        self.sink.emit(dict(type="trial_start", play_id=play_id,
                            item=dict(split=split, session=session, trial_key=trial_key), partition=partition,
                            day_index=day, n_bins=n_bins, n_frames=n_frames, window_bins=window, stride_bins=stride,
                            bin_ns=bin_ns, speed=speed, timing_valid=timing_valid, contention=contention,
                            flags=registry.flags(spec, partition), arrays=ARRAYS, tx_fallback=display["fallback"],
                            trace=path.name))
        metadata = dict(
            replay_mode="live_paced_synchronous_two_process", speed=speed, timing_valid=timing_valid,
            input_clock=f"bin_end_schedule_{bin_ns}_ns", endpoint="provided_trial_end (oracle endpointing)",
            commitment_policy="none", budgets_ms=[200.0, 500.0, 1500.0], contention=contention,
            pipeline=spec["name"], pipeline_hash=self.identity["pipeline_hash"], config=spec["decode"],
            preprocess=spec["preprocess"], warmup=warmup,
            display_extras="display payloads are derived after each trace record inside the timed loop")
        observing = ObservingAcoustic(AcousticAdapter(runtime.model, runtime.args, day, runtime.device), self.stop)
        status, summary = "complete", None
        dropped_before = self.sink.dropped
        try:
            with OutputTraceWriter(path, metadata, schema_version=2) as writer:
                tee = TeeWriter(writer, observing, display, self.sink, play_id, LOGIT_TO_PHONEME)
                runtime.sync()
                replay_trial(features, observing, self.lm, tee, trial_index=0, day_index=day,
                             patch_size=runtime.patch_size, patch_stride=runtime.patch_stride, bin_ns=bin_ns,
                             source=dict(session=session, trial_key=trial_key, split=split))
            summary = summarize_trace(path)["trials"][0]
        except Stopped:
            status = "stopped"
            if self.lm.active:
                self.lm.request("finish")
        self.sink.emit(dict(type="trial_end", play_id=play_id, status=status, summary=summary,
                            dropped_bins=self.sink.dropped - dropped_before))

    def shutdown(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=30)
        self.close_lm()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--lm-python", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--registry", default=str(registry.REGISTRY_DIR))
    options = parser.parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Keep library prints (CUDA, native) off the protocol stream.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sink = Sink(protocol)
    engine = Engine(options.root, options.data_dir, options.lm_python, options.device, sink, options.registry)
    engine.emit_state("idle", "Engine started")
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                sink.emit(dict(type="error", message="Malformed command"))
                continue
            engine.handle(command)
    finally:
        engine.shutdown()
        sink.close()
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
