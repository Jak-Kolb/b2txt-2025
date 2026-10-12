"""Bidirectional GRU option: default GRU unchanged, non-causal outputs, and offline-only handling."""
import sys
import tempfile
import unittest
from pathlib import Path

if sys.version_info < (3, 10):
    raise unittest.SkipTest("acoustic model tests require the Python 3.10 acoustic stack")

import numpy as np
import torch

import harness_fixture as fx
from harness import registry
from harness.train import streaming_check
from model_training.benchmark.common import build_model, offline_logits
from model_training.benchmark.streaming_infer import run_streaming_trial


def args(bidirectional):
    cfg = fx.model_args(0)
    cfg["model"]["bidirectional"] = bidirectional
    return cfg


class BidirectionalGRUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = np.random.default_rng(1).normal(size=(41, fx.N_FEATURES)).astype(np.float32)

    def logits(self, model, cfg, features):
        with torch.inference_mode():
            return offline_logits(model, torch.as_tensor(features).unsqueeze(0), 1, cfg, torch.device("cpu"))[0]

    def test_default_gru_state_dict_is_unchanged(self):
        default = build_model(fx.model_args(0)).state_dict()
        explicit = build_model(args(False)).state_dict()
        self.assertEqual({k: v.shape for k, v in default.items()}, {k: v.shape for k, v in explicit.items()})
        self.assertFalse(any("reverse" in key for key in default))

    def test_bidirectional_outputs_depend_on_future_frames(self):
        cfg = args(True)
        model = build_model(cfg).eval()
        n_units = cfg["model"]["n_units"]
        self.assertEqual(model.out.in_features, 2 * n_units)
        base = self.logits(model, cfg, self.features)
        changed = self.features.copy()
        changed[-8:] += 5.0   # perturb only the end of the trial
        moved = self.logits(model, cfg, changed)
        self.assertEqual(base.shape, moved.shape)
        self.assertGreater(float((base[0] - moved[0]).abs().max()), 1e-6)   # the first frame sees the future

    def test_bidirectional_cannot_stream(self):
        cfg = args(True)
        model = build_model(cfg).eval()
        with self.assertRaises(ValueError):
            run_streaming_trial(model, cfg, torch.device("cpu"), self.features, 1, enable_timing=False)

    def test_registry_and_smoke_treat_it_as_offline_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg = fx.make_registry(root)
            checkpoint = fx.make_checkpoint(root / "models" / "tiny_bi" / "checkpoint", bidirectional=True)
            (reg / "acoustic" / "tiny_bi.yaml").write_text(
                f"name: tiny_bi\ncheckpoint_dir: {checkpoint}\nselection: {{rule: test, data: [val-dev]}}\n")
            spec = registry.resolve(acoustic="tiny_bi", lm="fake", root=reg)
            self.assertFalse(spec["streaming_capable"])
            self.assertIn("offline_noncausal", registry.flags(spec, "val-dev"))
            self.assertEqual(streaming_check(checkpoint, root, device="cpu"), dict(skipped="bidirectional model"))


if __name__ == "__main__":
    unittest.main()
