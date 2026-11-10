"""Phrase splicing (dataset.splice_trials) and the auxiliary diphone CTC targets (rnn_trainer.diphone_targets)."""
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
    import torch
    import torchaudio  # noqa: F401  (rnn_trainer dependency)
    import omegaconf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("requires torch/torchaudio/omegaconf (acoustic environment)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from dataset import splice_trials
from rnn_model import GRUDecoder
from rnn_trainer import diphone_targets


class SpliceTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = torch.arange(40.).view(20, 2), -torch.arange(60.).view(30, 2)
        self.a_ids, self.b_ids = torch.tensor([3, 40, 5, 40, 0, 0, 0, 0]), torch.tensor([7, 40, 8, 9, 40, 0, 0, 0])

    def test_first_words_of_a_then_last_words_of_b(self):
        features, ids, n_steps, seq_len = splice_trials(self.a, self.a_ids, [[10, 2]], self.b, self.b_ids, 5,
                                                        [[12, 2]], np.random.default_rng(0))
        self.assertEqual((n_steps, seq_len), (28, 5))
        self.assertEqual(ids.tolist(), [3, 40, 8, 9, 40, 0, 0, 0])
        self.assertTrue(torch.equal(features[:10], self.a[:10]) and torch.equal(features[10:], self.b[12:]))

    def test_labels_that_do_not_fit_are_refused(self):
        self.assertIsNone(splice_trials(self.a, self.a_ids[:4], [[10, 4]], self.b, self.b_ids, 5, [[12, 2]],
                                        np.random.default_rng(0)))


class DiphoneTests(unittest.TestCase):
    def test_classes_are_unique_per_context_and_padding_stays_blank(self):
        d = diphone_targets(torch.tensor([[3, 40, 5, 0], [5, 3, 5, 3]]), 41)
        self.assertEqual(d[0].tolist(), [3, 1 + 3 * 40 + 39, 1 + 40 * 40 + 4, 0])
        self.assertNotEqual(d[1, 0].item(), d[1, 2].item())            # same phoneme, different left context
        self.assertLess(d.max().item(), 1 + 41 * 40)

    def test_hidden_features_feed_the_output_layer(self):
        torch.manual_seed(0)
        model = GRUDecoder(neural_dim=8, n_units=16, n_days=2, n_classes=41, rnn_dropout=0.0, input_dropout=0.0,
                           n_layers=1, patch_size=4, patch_stride=2).eval()
        x, day = torch.randn(2, 20, 8), torch.tensor([0, 1])
        hidden = model(x, day, return_hidden=True)
        self.assertEqual(hidden.shape[-1], model.out.in_features)
        self.assertTrue(torch.allclose(model.out(hidden), model(x, day), atol=1e-6))


class TrialSamplingTests(unittest.TestCase):
    def test_each_pass_uses_every_trial_once(self):
        from dataset import BrainToTextDataset
        trials = {0: {"trials": list(range(5)), "session_path": "a"}, 1: {"trials": list(range(30)), "session_path": "b"}}
        ds = BrainToTextDataset(trials, n_batches=7, batch_size=10, days_per_batch=4, random_seed=3, sampling="trials")
        drawn = [(d, t) for i in range(7) for d, ts in ds.batch_index[i].items() for t in ts]
        self.assertTrue(all(sum(map(len, ds.batch_index[i].values())) == 10 for i in range(7)))
        from collections import Counter
        self.assertEqual(len(set(drawn[:30])), 30)               # inside the first pass: no trial twice
        counts = Counter(drawn)                                  # 70 draws = two full passes over the 35 trials
        self.assertEqual(len(counts), 35)
        self.assertEqual(set(counts.values()), {2})
        with self.assertRaises(ValueError):
            BrainToTextDataset(trials, n_batches=1, batch_size=10, days_per_batch=1, sampling="sessions")
        with self.assertRaises(ValueError):                      # the default sampler still checks days_per_batch
            BrainToTextDataset(trials, n_batches=1, batch_size=10, days_per_batch=4)


if __name__ == "__main__":
    unittest.main()
