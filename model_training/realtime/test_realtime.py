"""Unit + smoke tests for realtime package (no GPU required).

Run from repo root:
  python -m model_training.realtime.test_realtime
  # or
  python model_training/realtime/test_realtime.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MODEL_TRAINING = HERE.parent
BENCHMARK = MODEL_TRAINING / "benchmark"
for p in (str(HERE), str(MODEL_TRAINING), str(BENCHMARK)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ensemble import average_logits  # noqa: E402
from streaming_decode import (  # noqa: E402
    AdaptiveRescoreConfig,
    blank_skip_mask,
    filter_frames_by_blank_skip,
    frame_entropy,
    greedy_ctc_with_blank_skip,
    run_adaptive_policy_over_logits,
    should_trigger_rescore,
    stream_greedy_partials,
    StreamingBeamState,
)
from stability import compute_stability_metrics, StabilityTracker  # noqa: E402
from peak_delay import measure_peak_delays, summarize_peak_delays  # noqa: E402
from checkpoint_avg import average_state_dicts  # noqa: E402


class TestAverageLogits(unittest.TestCase):
    def test_prob_mean_shape(self):
        a = torch.randn(10, 41)
        b = torch.randn(10, 41)
        out = average_logits([a, b], mode="prob_mean")
        self.assertEqual(tuple(out.shape), (10, 41))

    def test_identical_members(self):
        a = torch.randn(5, 8)
        out = average_logits([a, a, a], mode="logit_mean")
        self.assertTrue(torch.allclose(out, a.float(), atol=1e-5))

    def test_logprob_mean_normalized(self):
        a = torch.randn(4, 6)
        b = torch.randn(4, 6)
        out = average_logits([a, b], mode="logprob_mean")
        # rows should be valid log-probs
        self.assertTrue(torch.allclose(out.exp().sum(-1), torch.ones(4), atol=1e-4))


class TestBlankSkip(unittest.TestCase):
    def test_high_blank_skipped(self):
        # Craft logits: frame 0 high blank, frame 1 high class 3
        logits = torch.zeros(3, 5)
        logits[0, 0] = 10.0  # blank
        logits[1, 3] = 10.0
        logits[2, 0] = 10.0
        mask = blank_skip_mask(logits, blank_threshold=0.7)
        self.assertTrue(bool(mask[0].item()))
        self.assertFalse(bool(mask[1].item()))
        self.assertTrue(bool(mask[2].item()))

    def test_filter_keeps_min(self):
        logits = torch.zeros(2, 4)
        logits[:, 0] = 10.0  # all blank
        kept, idx = filter_frames_by_blank_skip(logits, blank_threshold=0.5, min_keep=1)
        self.assertGreaterEqual(kept.shape[0], 1)

    def test_greedy_blank_skip_runs(self):
        logits = torch.randn(20, 41)
        seq = greedy_ctc_with_blank_skip(logits, blank_threshold=0.7)
        self.assertIsInstance(seq, np.ndarray)


class TestAdaptive(unittest.TestCase):
    def test_trigger_gate(self):
        cfg = AdaptiveRescoreConfig(entropy_threshold=1.0, min_partial_frames=2, cooldown_frames=3)
        st = StreamingBeamState()
        self.assertFalse(should_trigger_rescore(st, 0, 5.0, cfg))
        self.assertTrue(should_trigger_rescore(st, 5, 5.0, cfg))
        st.last_rescore_frame = 5
        self.assertFalse(should_trigger_rescore(st, 6, 5.0, cfg))  # cooldown

    def test_policy_dry_run(self):
        logits = torch.randn(15, 41)
        out = run_adaptive_policy_over_logits(logits, rescorer=None)
        self.assertIn("trigger_rate", out)
        self.assertGreaterEqual(out["trigger_rate"], 0.0)


class TestStability(unittest.TestCase):
    def test_no_revisions(self):
        partials = ["a", "a b", "a b c"]
        times = [80.0, 160.0, 240.0]
        m = compute_stability_metrics(partials, times, "a b c")
        self.assertEqual(m["n_words"], 3)
        self.assertEqual(m["mean_rpv"], 0.0)

    def test_with_revisions(self):
        partials = ["the", "a", "a cat", "the cat"]
        times = [80.0, 160.0, 240.0, 320.0]
        m = compute_stability_metrics(partials, times, "the cat")
        self.assertGreaterEqual(m["mean_rpv"], 0.0)
        self.assertEqual(m["n_words"], 2)

    def test_tracker(self):
        tr = StabilityTracker()
        tr.update("hi", 80)
        tr.update("hi there", 160)
        m = tr.finalize("hi there")
        self.assertEqual(m["n_words"], 2)


class TestPeakDelay(unittest.TestCase):
    def test_measure_and_summarize(self):
        # Synthetic: peaks late vs linear midpoints
        T, C = 40, 10
        logits = torch.full((T, C), -5.0)
        # Emit classes 1,2,3 near the end
        for i, c in enumerate([1, 2, 3]):
            t = 20 + i * 5
            logits[t, c] = 10.0
        phones = [1, 2, 3]
        trial = measure_peak_delays(logits, phones, patch_stride=4, bin_ms=20.0)
        self.assertGreater(trial["n_matched"], 0)
        summary = summarize_peak_delays([trial])
        self.assertIn(summary["decision"], {
            "KILL_EMISSION_REGULARIZERS",
            "AUTHORIZE_DELAY_CTC",
            "BORDERLINE",
            "INCONCLUSIVE",
        })


class TestCheckpointAvg(unittest.TestCase):
    def test_average(self):
        a = {"w": torch.ones(3), "i": torch.tensor(1)}
        b = {"w": torch.ones(3) * 3, "i": torch.tensor(1)}
        out = average_state_dicts([a, b])
        self.assertTrue(torch.allclose(out["w"], torch.ones(3) * 2))
        self.assertEqual(int(out["i"].item()), 1)


class TestEntropy(unittest.TestCase):
    def test_peaked_low_entropy(self):
        logits = torch.zeros(1, 5)
        logits[0, 2] = 20.0
        ent = frame_entropy(logits)
        self.assertLess(float(ent[0]), 0.1)

    def test_uniform_high_entropy(self):
        logits = torch.zeros(1, 8)
        ent = frame_entropy(logits)
        # ln(8) ≈ 2.07
        self.assertGreater(float(ent[0]), 1.5)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
