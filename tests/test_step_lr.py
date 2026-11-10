"""Step learning-rate schedule: linear warmup, then x lr_gamma at each milestone, for every param group."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("requires the acoustic environment")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model_training"))
from rnn_trainer import BrainToTextDecoder_Trainer


class StepScheduleTests(unittest.TestCase):
    def test_warmup_then_steps(self):
        a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
        optim = torch.optim.AdamW([{"params": [a]}, {"params": [b], "lr": 0.5}], lr=2.0)
        fake = SimpleNamespace(args=dict(lr_warmup_steps=4, lr_gamma=0.1, lr_milestones=[10, 20]))
        sched = BrainToTextDecoder_Trainer.create_step_lr_scheduler(fake, optim)
        seen = []
        for _ in range(25):
            seen.append([g["lr"] for g in optim.param_groups])
            optim.step()
            sched.step()
        self.assertEqual(seen[0], [0.0, 0.0])
        self.assertAlmostEqual(seen[2][0], 1.0)
        for step, factor in ((4, 1.0), (9, 1.0), (10, 0.1), (19, 0.1), (20, 0.01), (24, 0.01)):
            self.assertAlmostEqual(seen[step][0], 2.0 * factor, msg=step)
            self.assertAlmostEqual(seen[step][1], 0.5 * factor, msg=step)


if __name__ == "__main__":
    unittest.main()
