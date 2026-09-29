"""Per-trial time masking (the published treatment) on synthetic tensors."""
import sys
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("requires torch (acoustic environment)")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from data_augmentations import time_mask_per_trial


def masked_bins(out):
    return (out == 0).all(dim=-1)                                   # [B, T]


class TimeMaskPerTrialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_padding_is_never_masked_and_coverage_is_exact(self):
        features = torch.ones(3, 400, 4)
        lengths = torch.tensor([400, 120, 37])
        out, coverage = time_mask_per_trial(features, lengths, n_masks=20, max_frac=0.075)
        masked = masked_bins(out)
        for b, n in enumerate(lengths.tolist()):
            self.assertFalse(masked[b, n:].any())                   # nothing beyond the valid length
            self.assertAlmostEqual(coverage[b].item(), masked[b, :n].float().mean().item(), places=6)

    def test_each_span_is_at_most_max_frac_of_the_trial(self):
        lengths = torch.full((500,), 200)
        out, _ = time_mask_per_trial(torch.ones(500, 200, 1), lengths, n_masks=1, max_frac=0.075)
        widths = masked_bins(out).sum(dim=1)
        self.assertLessEqual(widths.max().item(), 15)                # floor(0.075 * 200)
        self.assertEqual(widths.max().item(), 15)                    # the maximum width is reachable

    def test_short_trials_get_no_masks(self):
        out, coverage = time_mask_per_trial(torch.ones(2, 13, 2), torch.tensor([13, 13]), n_masks=20, max_frac=0.075)
        self.assertTrue(torch.equal(out, torch.ones(2, 13, 2)))     # floor(0.075 * 13) = 0
        self.assertEqual(coverage.tolist(), [0.0, 0.0])

    def test_published_regime_covers_about_half_of_a_trial(self):
        # 20 spans of U{0..75} on 1000 bins: 1 - (1 - 37.5/1000)^20 = 0.53 before edge effects
        _, coverage = time_mask_per_trial(torch.ones(2000, 1000, 1), torch.full((2000,), 1000),
                                          n_masks=20, max_frac=0.075)
        self.assertGreater(coverage.mean().item(), 0.49)
        self.assertLess(coverage.mean().item(), 0.56)


if __name__ == "__main__":
    unittest.main()
