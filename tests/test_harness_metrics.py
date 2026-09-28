"""Harness scoring, uncertainty, structural-delay, and queue-simulation tests (synthetic only)."""
import sys
import tempfile
import unittest
from pathlib import Path

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

import numpy as np

from harness import metrics
from model_training.benchmark.output_trace import OutputTraceWriter
from model_training.benchmark.paced_replay import replay_trial
from test_paced_replay import Acoustic, Clock, FakeClient


def row(session, edits, words, seen=False, key="t", ref="r", hyp="h"):
    return dict(session=session, split="val", trial_key=key, ref=ref, hyp=hyp, edits=edits,
                ref_words=words, seen=seen, phone_edits=1, phone_len=4, n_final_words=words,
                revision_edits=0, finalization_edit_counts=dict(appends=0))


class ScoringTests(unittest.TestCase):
    def test_pooled_and_seen_unseen(self):
        rows = [row("a", 1, 10, True), row("a", 0, 5), row("b", 3, 5)]
        self.assertAlmostEqual(metrics.pooled(rows)["percent"], 20.0)
        split = metrics.seen_unseen(rows)
        self.assertEqual(split["seen"]["edits"] + split["unseen"]["edits"], 4)
        self.assertEqual(metrics.word_errors("Hello, world.", "hello word"), (1, 2))

    def test_bootstrap_is_deterministic_and_needs_two_sessions(self):
        rows = [row(s, e, 10) for s, e in zip("abcde", [0, 1, 2, 3, 4])]
        first = metrics.session_bootstrap(rows, n_boot=500)
        self.assertEqual(first, metrics.session_bootstrap(rows, n_boot=500))
        lo, hi = first["ci95"]
        self.assertLessEqual(lo, 20.0)
        self.assertGreaterEqual(hi, 20.0)
        self.assertIsNone(metrics.session_bootstrap([row("a", 1, 3)])["ci95"])

    def test_paired_compare(self):
        a = [row("a", 2, 10, key="1", hyp="x"), row("b", 1, 10, key="2", hyp="y")]
        b = [row("a", 1, 10, key="1", hyp="z"), row("b", 1, 10, key="2", hyp="y")]
        result = metrics.paired_compare(a, b, n_boot=200)
        self.assertAlmostEqual(result["delta_wer_points"], -5.0)
        self.assertEqual(result["trials"]["improved"], 1)
        self.assertEqual(len(result["differing"]), 1)
        with self.assertRaises(ValueError):
            metrics.paired_compare(a, b[::-1])


class StructuralDelayTests(unittest.TestCase):
    def preprocess(self, lookahead):
        return dict(smooth_data=True, smooth_kernel_std=2, smooth_kernel_size=100, smooth_lookahead=lookahead)

    def test_la0_and_la4(self):
        la0 = metrics.structural_delay(self.preprocess(0), 14, 4)
        self.assertEqual(la0["first_output_ms"], 280.0)
        self.assertEqual(la0["update_cadence_ms"], 80.0)
        self.assertEqual(la0["frame_entry_wait_ms"], dict(min=0.0, max=60.0, mean=30.0))
        self.assertEqual(la0["lookahead_ms"], 0.0)
        self.assertGreater(la0["smoothing"]["centroid_delay_ms"], 0)
        la4 = metrics.structural_delay(self.preprocess(4), 14, 4)
        self.assertEqual(la4["first_output_ms"], 360.0)
        self.assertEqual(la4["lookahead_ms"], 80.0)
        self.assertTrue(metrics.structural_delay(self.preprocess(None), 14, 4)["noncausal_offline_only"])

    def test_kernel_matches_streaming_kernel(self):
        import torch
        from model_training.benchmark.common import smoothing_kernel_from_args
        for lookahead in (0, 2, 4):
            args = {"dataset": {"data_transforms": self.preprocess(lookahead)}}
            expected, past, future = smoothing_kernel_from_args(args, torch.device("cpu"))
            kernel, p, f = metrics.smoothing_kernel(2, 100, lookahead)
            np.testing.assert_allclose(kernel, expected.numpy(), rtol=1e-6)
            self.assertEqual((p, f), (past, future))


class SimulationTests(unittest.TestCase):
    """Simulating a compressed (bin_ns=1) run must reproduce a paced run's lags exactly."""

    def run_trace(self, path, bin_ns, cost_ns, n_bins=12, window=3, stride=2, lookahead=1):
        clock = Clock()
        acoustic = Acoustic(clock, cost_ns, window, stride, lookahead)
        features = np.arange(n_bins * 2, dtype=np.float32).reshape(n_bins, 2)
        with OutputTraceWriter(path, {"budgets_ms": []}, schema_version=2) as writer:
            for trial in range(2):
                replay_trial(features, acoustic if trial == 0 else Acoustic(clock, cost_ns, window, stride, lookahead),
                             FakeClient(clock, cost_ns=3_000_000), writer, trial_index=trial, day_index=0,
                             patch_size=window, patch_stride=stride, bin_ns=bin_ns,
                             clock=clock, sleep=clock.sleep)
        return path

    def check(self, cost_ns):
        with tempfile.TemporaryDirectory() as temp:
            paced = self.run_trace(Path(temp) / "paced.jsonl", 20_000_000, cost_ns)
            compressed = self.run_trace(Path(temp) / "fast.jsonl", 1, cost_ns)
            measured = metrics.paced_lags(paced)
            for paced_trial, fast_trial in zip(metrics.trace_trials(paced), metrics.trace_trials(compressed)):
                frame, latest, final = metrics.simulate_trial(fast_trial)
                expected_frame = [(o["output_ready_ns"] - o["frame_window_available_ns"]) / 1e6
                                  for s in paced_trial["steps"] for o in s["outputs"] if o["kind"] == "partial"]
                expected_latest = [(o["output_ready_ns"] - o["input_available_ns"]) / 1e6
                                   for s in paced_trial["steps"] for o in s["outputs"] if o["kind"] == "partial"]
                self.assertEqual(frame, expected_frame)
                self.assertEqual(latest, expected_latest)
                final_output = paced_trial["steps"][-1]["outputs"][-1]
                self.assertEqual(final, (final_output["output_ready_ns"] - 12 * 20_000_000) / 1e6)
            summary = metrics.timing_check_summary(compressed)
            self.assertEqual(summary["simulated_lag"]["frame_window_to_output_ms"],
                             measured["frame_window_to_output_ms"])
            self.assertEqual(summary["n_bins"], 24)
            return summary

    def test_no_backlog(self):
        summary = self.check(cost_ns=2_000_000)
        self.assertEqual(summary["service_ms"]["frac_bins_over_bin"], 0.0)

    def test_backlog(self):
        summary = self.check(cost_ns=25_000_000)
        self.assertGreater(summary["service_ms"]["frac_bins_over_bin"], 0.5)
        lags = summary["simulated_lag"]["frame_window_to_output_ms"]
        self.assertGreater(lags["max"], lags["p50"])


if __name__ == "__main__":
    unittest.main()
