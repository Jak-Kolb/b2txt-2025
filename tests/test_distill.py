"""Frame-level distillation loss: zero for identical outputs, masked beyond each trial's frames, gradient to the
student only."""
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("requires the acoustic environment")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model_training"))
from rnn_trainer import distill_loss


class DistillTests(unittest.TestCase):
    def test_zero_for_identical_and_positive_otherwise(self):
        torch.manual_seed(0)
        teacher = torch.randn(2, 9, 5)
        lens = torch.tensor([9, 6])
        self.assertAlmostEqual(float(distill_loss(teacher.clone(), teacher, lens, 2.0)), 0.0, places=5)
        self.assertGreater(float(distill_loss(torch.randn(2, 9, 5), teacher, lens, 2.0)), 0.0)

    def test_frames_beyond_length_are_ignored(self):
        torch.manual_seed(1)
        student, teacher = torch.randn(2, 9, 5), torch.randn(2, 9, 5)
        lens = torch.tensor([9, 6])
        changed = student.clone(); changed[1, 6:] += 5.0
        torch.testing.assert_close(distill_loss(student, teacher, lens, 2.0), distill_loss(changed, teacher, lens, 2.0))

    def test_gradient_reaches_student_only(self):
        torch.manual_seed(2)
        student = torch.randn(1, 4, 5, requires_grad=True)
        teacher = torch.randn(1, 4, 5)
        distill_loss(student, teacher, torch.tensor([4]), 1.0).backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
