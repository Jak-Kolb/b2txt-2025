"""Sliding-window bidirectional GRU: matches the full GRU on an exact window, bounded support, padded batches, flags."""
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


def args(past, future, layers=2):
    cfg = fx.model_args(0)   # patch 3 bins, stride 2 bins, causal smoothing (kernel 8): frame k = bins 2k..2k+2
    cfg["model"].update(bidirectional=True, n_layers=layers, window_past_frames=past, window_future_frames=future)
    return cfg


class WindowModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = np.random.default_rng(1).normal(size=(41, fx.N_FEATURES)).astype(np.float32)

    def logits(self, model, cfg, features):
        with torch.inference_mode():
            return offline_logits(model, torch.as_tensor(features).unsqueeze(0), 1, cfg, torch.device("cpu"))[0]

    def test_exact_window_matches_full_bidirectional_gru(self):
        model = build_model(args(0, 0)).eval()
        frames = torch.randn(2, 7, model.input_size)
        states = model.h0.expand(model.n_layers * 2, 2, model.n_units).contiguous()
        with torch.inference_mode():
            full, _ = model.gru(frames, states)
            for t in range(7):   # a window covering exactly frames 0..6, seen from frame t
                model.window_past_frames, model.window_future_frames = t, 6 - t
                torch.testing.assert_close(model._window_gru(frames)[:, t], full[:, t], rtol=1e-5, atol=1e-6)

    def test_support_is_the_window(self):
        for past, future in ((0, 2), (3, 1)):
            cfg = args(past, future)
            model = build_model(cfg).eval()
            base = self.logits(model, cfg, self.features)
            k = 10
            first_bin, last_bin = 2 * (k - past), 2 * (k + future) + 2
            after = self.features.copy()
            after[last_bin + 1:] += 5.0
            torch.testing.assert_close(self.logits(model, cfg, after)[k], base[k], rtol=0, atol=1e-6)
            before = self.features.copy()
            before[:max(first_bin - 8, 0)] += 5.0      # beyond the causal smoothing support (8 taps)
            torch.testing.assert_close(self.logits(model, cfg, before)[k], base[k], rtol=0, atol=1e-6)
            for inside in (first_bin, last_bin):
                changed = self.features.copy()
                changed[inside] += 5.0
                self.assertGreater(float((self.logits(model, cfg, changed)[k] - base[k]).abs().max()), 1e-6)

    def test_padded_batch_matches_single_trial(self):
        cfg = args(2, 2)
        model = build_model(cfg).eval()
        long = torch.as_tensor(self.features).unsqueeze(0)
        short = long[:, :25]                                  # 12 output frames
        batch = torch.cat([long, torch.nn.functional.pad(short, (0, 0, 0, 16))], dim=0)
        with torch.inference_mode():
            single = model(short, torch.tensor([1]))[0]
            masked = model(batch, torch.tensor([1, 1]), frame_lens=torch.tensor([20, 12]))[1, :12]
        torch.testing.assert_close(masked, single, rtol=0, atol=1e-5)

    def test_parameters_match_the_plain_bidirectional_model_and_training_runs(self):
        plain = fx.model_args(0)
        plain["model"].update(bidirectional=True, n_layers=2)
        window = build_model(args(1, 2))
        self.assertEqual({k: v.shape for k, v in build_model(plain).state_dict().items()},
                         {k: v.shape for k, v in window.state_dict().items()})
        window.train()
        x = torch.randn(2, 21, fx.N_FEATURES)
        window(x, torch.tensor([0, 1]), frame_lens=torch.tensor([10, 7])).sum().backward()   # checkpointed path
        self.assertTrue(all(p.grad is not None for n, p in window.named_parameters() if n.startswith("gru.")))

    def test_invalid_and_offline_handling(self):
        cfg = args(1, 2)
        cfg["model"]["bidirectional"] = False
        with self.assertRaises(ValueError):
            build_model(cfg)
        cfg = args(1, 2)
        model = build_model(cfg).eval()
        with self.assertRaises(ValueError):
            run_streaming_trial(model, cfg, torch.device("cpu"), self.features, 1, enable_timing=False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg = fx.make_registry(root)
            checkpoint = root / "models" / "tiny_win" / "checkpoint"
            checkpoint.mkdir(parents=True)
            torch.save({"model_state_dict": model.state_dict()}, checkpoint / "best_checkpoint")
            from omegaconf import OmegaConf
            OmegaConf.save(OmegaConf.create(cfg), checkpoint / "args.yaml")
            (reg / "acoustic" / "tiny_win.yaml").write_text(
                f"name: tiny_win\ncheckpoint_dir: {checkpoint}\nselection: {{rule: test, data: [val-dev]}}\n")
            spec = registry.resolve(acoustic="tiny_win", lm="fake", root=reg)
            self.assertFalse(spec["streaming_capable"])
            self.assertIn("offline_lookahead_80ms", registry.flags(spec, "val-dev"))   # 2 frames x 2 bins x 20 ms
            self.assertEqual(streaming_check(checkpoint, root, device="cpu"), dict(skipped="bidirectional model"))


if __name__ == "__main__":
    unittest.main()
