"""Offline paced-replay tests; real subprocesses, synthetic data/decoders only."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from model_training.benchmark.lm_worker import LMClient, serve
from model_training.benchmark.output_trace import OutputTraceWriter, summarize_trace
from model_training.benchmark.paced_replay import frame_geometry, replay_trial, reorder_logits, wait_until


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)

    def advance(self, ns):
        self.now += ns
        return self.now


class Acoustic:
    def __init__(self, clock, cost_ns=0, window=1, stride=1, lookahead=0):
        self.clock, self.cost_ns = clock, cost_ns
        self.window, self.stride, self.lookahead = window, stride, lookahead
        self.rows, self.frames = [], 0

    def _ready(self, end):
        result = []
        while self.window + self.frames * self.stride <= end:
            result.append(np.array([0, self.frames + 2, 1], dtype=np.float32))
            self.frames += 1
        return result

    def process_bin(self, row):
        self.rows.append(row.copy())
        self.clock.advance(self.cost_ns)
        return self._ready(len(self.rows) - self.lookahead)

    def finish(self):
        self.clock.advance(self.cost_ns)
        return self._ready(len(self.rows))


class FakeClient:
    def __init__(self, clock, cost_ns=1_000_000):
        self.clock, self.cost_ns = clock, cost_ns
        self.frames = []

    def request(self, op, **payload):
        received = self.clock.advance(100_000)
        start = self.clock.advance(100_000)
        if op == "reset":
            self.words = []
        if op == "frame":
            self.frames.append(payload)
            self.words.append(str(max(range(len(payload["logits"])), key=payload["logits"].__getitem__)))
        ready = self.clock.advance(self.cost_ns)
        output = self.clock.advance(100_000)
        return dict(worker_received_ns=received, processing_start_ns=start,
                    worker_output_ready_ns=ready, output_ready_ns=output, words=list(self.words))


class PacedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.jsonl"

    def replay(self, n_bins=5, window=1, stride=1, lookahead=0, cost_ns=0):
        clock = Clock()
        acoustic = Acoustic(clock, cost_ns, window, stride, lookahead)
        lm = FakeClient(clock)
        features = np.arange(n_bins * 2, dtype=np.float32).reshape(n_bins, 2)
        with OutputTraceWriter(self.path, {"budgets_ms": [20, 200]}, schema_version=2) as writer:
            result = replay_trial(features, acoustic, lm, writer, trial_index=2, day_index=1,
                                  patch_size=window, patch_stride=stride,
                                  clock=clock, sleep=clock.sleep, keep_logits=True)
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        return result, records, acoustic, lm

    def test_fixed_schedule_exposes_backlog(self):
        result, records, _, _ = self.replay(cost_ns=35_000_000)
        inputs = [r for r in records if r["record"] == "input"]
        self.assertEqual([r["input_available_ns"] for r in inputs],
                         [20_000_000 * (i + 1) for i in range(5)])
        lag = [r["acoustic_start_ns"] - r["input_available_ns"] for r in inputs]
        self.assertEqual(lag[0], 0)
        self.assertGreater(lag[-1], lag[1])
        report = summarize_trace(self.path)
        metrics = report["trials"][0]["paced"]
        self.assertEqual(metrics["budget_exceedances"]["20"]["n"], 5)
        self.assertGreater(metrics["elapsed_rtf"], 1)
        self.assertEqual(result["n_frames"], 5)

    def test_lookahead_and_endpoint_flush_have_separate_clocks(self):
        _, records, _, _ = self.replay(n_bins=7, window=3, stride=2, lookahead=2)
        outputs = [r for r in records if r["record"] == "output" and r["kind"] == "partial"]
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0]["frame_window_available_ns"], 60_000_000)
        self.assertEqual(outputs[0]["input_available_ns"], 100_000_000)
        self.assertEqual([r["phase"] for r in outputs], ["stream", "stream", "endpoint_flush"])
        report = summarize_trace(self.path)["trials"][0]
        self.assertEqual(report["paced"]["n_endpoint_flush_outputs"], 1)
        self.assertEqual(report["paced"]["stream_budget_exceedances"]["20"]["denominator"], 2)

    def test_ordered_transfer_reorders_classes_once_and_sends_no_reference(self):
        _, _, acoustic, lm = self.replay(n_bins=2)
        self.assertEqual(lm.frames[0]["logits"], [0.0, 1.0, 2.0])
        self.assertEqual(set(lm.frames[0]), {"frame_index", "logits"})
        np.testing.assert_array_equal(acoustic.rows[0], [0, 1])
        np.testing.assert_array_equal(acoustic.rows[1], [2, 3])

    def test_empty_trial_and_short_trial_finish(self):
        result, _, _, _ = self.replay(n_bins=0, window=3, stride=2)
        self.assertEqual(result["n_frames"], 0)
        report = summarize_trace(self.path)
        self.assertIsNone(report["trials"][0]["paced"]["elapsed_rtf"])

    def test_wait_until_handles_early_wakeup_and_does_not_sleep_when_late(self):
        clock = Clock()
        target = clock() + 10
        waits = []
        def early(seconds):
            waits.append(seconds)
            clock.advance(5)
        wait_until(target, clock, early)
        self.assertEqual(len(waits), 2)
        wait_until(target - 1, clock, early)
        self.assertEqual(len(waits), 2)

    def test_geometry_without_patching(self):
        self.assertEqual(frame_geometry(3, 0, 0), (3, 1, 1))
        self.assertEqual(frame_geometry(2, 3, 1), (0, 3, 1))

    def test_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            reorder_logits(np.array([0, np.nan, 1]))

    def test_reader_rejects_fabricated_availability(self):
        _, records, _, _ = self.replay()
        next(r for r in records if r["record"] == "input")["input_available_ns"] = 0
        self.path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        with self.assertRaisesRegex(ValueError, "fixed replay schedule"):
            summarize_trace(self.path)

    def test_reader_rejects_cross_process_time_reversal(self):
        _, records, _, _ = self.replay()
        next(r for r in records if r["record"] == "output")["worker_received_ns"] = 0
        self.path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        with self.assertRaisesRegex(ValueError, "Pipeline timestamps"):
            summarize_trace(self.path)

    @unittest.skipIf(sys.version_info < (3, 10), "actual acoustic adapter requires Python 3.10")
    def test_actual_small_gru_matches_offline_with_lookahead(self):
        import torch
        from model_training.benchmark.common import build_model, offline_logits
        from model_training.benchmark.paced_replay import AcousticAdapter
        torch.manual_seed(4)
        args = {"model": {"n_input_features": 2, "n_units": 4, "n_layers": 1, "patch_size": 3,
                         "patch_stride": 2, "rnn_dropout": 0, "input_network": {"input_layer_dropout": 0}},
                "dataset": {"sessions": ["synthetic"], "n_classes": 3,
                            "data_transforms": {"smooth_data": True, "smooth_kernel_size": 8,
                                                "smooth_kernel_std": 1, "smooth_lookahead": 2}}}
        model = build_model(args).eval()
        features = np.random.default_rng(3).normal(size=(11, 2)).astype(np.float32)
        clock = Clock()
        with OutputTraceWriter(self.path, {"budgets_ms": []}, schema_version=2) as writer:
            result = replay_trial(features, AcousticAdapter(model, args, 0, torch.device("cpu")),
                                  FakeClient(clock), writer, trial_index=0, day_index=0,
                                  patch_size=3, patch_stride=2, clock=clock, sleep=clock.sleep,
                                  keep_logits=True)
        with torch.inference_mode():
            expected = offline_logits(model, torch.from_numpy(features).unsqueeze(0), 0,
                                      args, torch.device("cpu"))[0].numpy()
        np.testing.assert_allclose(np.stack(result["logits"]), expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(summarize_trace(self.path)["schema_version"], 2)

    @unittest.skipIf(sys.version_info < (3, 10), "HDF5 selector uses acoustic stack")
    def test_selection_excludes_former_val_test(self):
        import h5py
        from model_training.benchmark.paced_replay import select_development_trials
        path = self.path.with_suffix(".h5")
        with h5py.File(path, "w") as h:
            h.create_dataset("trial_0001/input_features", data=np.zeros((5, 2), np.float32))
        trials = [SimpleNamespace(day_idx=day, hdf5_path=path, trial_key="trial_0001")
                  for day in [39, 1, 40, 2]]
        with patch("model_training.benchmark.common.discover_trials", return_value=trials):
            selected = select_development_trials({}, "unused", 2, 1)
        self.assertEqual([t.day_idx for t in selected], [1, 2])


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "worker.log"
        self.fixture = Path(__file__).with_name("fake_lm_worker.py")
        remote = Path("/home/fishinjak/.miniforge3/envs/b2txt25_lm/bin/python")
        self.python = str(remote) if remote.exists() else sys.executable

    def test_real_process_protocol_roundtrip_and_shutdown(self):
        with LMClient([self.python, "-u", str(self.fixture)], self.log) as client:
            pid = client.process.pid
            self.assertEqual(client.ready["n_classes"], 3)
            client.request("reset")
            reply = client.request("frame", frame_index=0, logits=[0, 5, 1])
            self.assertEqual(reply["words"], ["1"])
            self.assertLessEqual(reply["worker_output_ready_ns"], reply["output_ready_ns"])
            self.assertEqual(client.request("finish")["words"], ["1"])
        self.assertEqual(client.process.returncode, 0)
        self.assertGreater(pid, 0)

    def test_native_style_failure_is_reported_and_child_reaped(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic decoder failure"):
            with LMClient([self.python, "-u", str(self.fixture), "--fail"], self.log) as client:
                client.request("reset")
                client.request("frame", frame_index=0, logits=[0, 5, 1])
        self.assertIsNotNone(client.process.poll())

    def test_timeout_reaps_child(self):
        with self.assertRaises(TimeoutError):
            with LMClient([self.python, "-u", str(self.fixture), "--delay", "1"],
                          self.log, timeout_seconds=0.02) as client:
                client.request("reset")
                client.request("frame", frame_index=0, logits=[0, 5, 1])
        self.assertIsNotNone(client.process.poll())

    def test_unexpected_frame_cannot_advance_decoder(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected frame"):
            with LMClient([self.python, "-u", str(self.fixture)], self.log) as client:
                client.request("reset")
                client.request("frame", frame_index=2, logits=[0, 5, 1])
        self.assertIsNotNone(client.process.poll())


    def test_teardown_has_a_separate_grace_period(self):
        with LMClient([self.python, "-u", str(self.fixture), "--shutdown_delay", "0.1"],
                      self.log, timeout_seconds=0.02, shutdown_seconds=1) as client:
            client.request("reset")
            client.request("finish")
        self.assertEqual(client.process.returncode, 0)

    def test_stalled_teardown_is_bounded_and_reaped(self):
        import subprocess
        with self.assertRaises(subprocess.TimeoutExpired):
            with LMClient([self.python, "-u", str(self.fixture), "--shutdown_delay", "1"],
                          self.log, shutdown_seconds=0.02) as client:
                client.request("reset")
                client.request("finish")
        self.assertIsNotNone(client.process.poll())

    def test_protocol_rejects_duplicate_request_ids(self):
        backend = SimpleNamespace(n_classes=3, reset=lambda: None)
        source = io.StringIO('{"id":0,"op":"reset"}\n{"id":0,"op":"reset"}\n')
        output = io.StringIO()
        self.assertEqual(serve(backend, source, output), 1)
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["type"], "error")


if __name__ == "__main__":
    unittest.main()
