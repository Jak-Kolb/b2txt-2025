"""Confidence penalty: per-frame entropy summed over each trial's valid frames, averaged over the batch."""
import math
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("requires the acoustic environment")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model_training"))
from rnn_trainer import frame_entropy, frame_uniform_kl


class FrameEntropyTests(unittest.TestCase):
    def test_uniform_frames_give_log_classes_per_valid_frame(self):
        logits = torch.zeros(2, 7, 5)
        lens = torch.tensor([7, 3])
        self.assertAlmostEqual(float(frame_entropy(logits, lens)), (7 + 3) / 2 * math.log(5), places=5)

    def test_frames_beyond_length_are_ignored(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 9, 5)
        lens = torch.tensor([9, 6])
        changed = logits.clone()
        changed[1, 6:] = torch.randn(3, 5) * 10
        torch.testing.assert_close(frame_entropy(logits, lens), frame_entropy(changed, lens))

    def test_penalty_gradient_flattens_a_peaked_frame(self):
        logits = torch.tensor([[[4.0, 0.0, 0.0]]], requires_grad=True)
        (-frame_entropy(logits, torch.tensor([1]))).backward()     # the trainer subtracts weight * entropy
        self.assertGreater(float(logits.grad[0, 0, 0]), 0.0)          # descent lowers the peak logit



class UniformKLTests(unittest.TestCase):
    def test_zero_for_uniform_frames_and_lengths_respected(self):
        lens = torch.tensor([7, 3])
        self.assertAlmostEqual(float(frame_uniform_kl(torch.zeros(2, 7, 5), lens)), 0.0, places=5)
        torch.manual_seed(0)
        logits = torch.randn(2, 9, 5)
        changed = logits.clone()
        changed[1, 6:] = torch.randn(3, 5) * 10
        lens = torch.tensor([9, 6])
        torch.testing.assert_close(frame_uniform_kl(logits, lens), frame_uniform_kl(changed, lens))

    def test_matches_kl_of_uniform_from_p(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 4, 6)
        logp = logits.log_softmax(-1)
        u = torch.full_like(logp, 1 / 6)
        expected = (u * (u.log() - logp)).sum(-1).sum()
        torch.testing.assert_close(frame_uniform_kl(logits, torch.tensor([4])), expected)

    def test_optimum_with_a_certain_label(self):
        # -log p_y + w * KL(u || p) is minimised at p_c = (w / C) / (1 + w) for every c != y
        w, C = 0.05, 41
        z = torch.zeros(1, 1, C, requires_grad=True)
        opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = -z.log_softmax(-1)[0, 0, 0] + w * frame_uniform_kl(z, torch.tensor([1]))
            loss.backward()
            return loss
        opt.step(closure)
        p = z.detach().softmax(-1)[0, 0]
        self.assertAlmostEqual(float(p[1]), (w / C) / (1 + w), places=5)
        self.assertAlmostEqual(float(p[0]), 0.954, places=3)


if __name__ == "__main__":
    unittest.main()
