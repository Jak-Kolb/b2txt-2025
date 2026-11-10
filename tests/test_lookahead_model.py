"""Bounded look-ahead GRU: exact right-context support, padded-batch consistency, capacity control, streaming."""
import sys
import tempfile
import unittest
from pathlib import Path

if sys.version_info < (3, 10):
    raise unittest.SkipTest("acoustic model tests require the Python 3.10 acoustic stack")

import numpy as np
import torch

import harness_fixture as fx
from harness import metrics, registry
from harness.train import streaming_check
from model_training.benchmark.common import build_model, offline_logits
from model_training.benchmark.streaming_infer import StreamingDecoder, run_streaming_trial


def args(frames, units=6):
    cfg = fx.model_args(0)   # patch 3 bins, stride 2 bins, causal smoothing: frame k ends at bin 2k+2
    cfg["model"].update(lookahead_frames=frames, lookahead_units=units)
    return cfg


class LookaheadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = np.random.default_rng(1).normal(size=(41, fx.N_FEATURES)).astype(np.float32)

    def logits(self, model, cfg, features):
        with torch.inference_mode():
            return offline_logits(model, torch.as_tensor(features).unsqueeze(0), 1, cfg, torch.device("cpu"))[0]

    def test_support_is_exactly_r_frames(self):
        for frames in (0, 1, 3):
            cfg = args(frames)
            model = build_model(cfg).eval()
            base = self.logits(model, cfg, self.features)
            for k in (2, 7):
                last_bin = 2 * (k + frames) + 2      # last raw bin inside frame k + R
                beyond = self.features.copy()
                beyond[last_bin + 1:] += 5.0
                moved = self.logits(model, cfg, beyond)
                torch.testing.assert_close(moved[:k + 1], base[:k + 1], rtol=0, atol=1e-6)
                inside = self.features.copy()
                inside[last_bin] += 5.0
                self.assertGreater(float((self.logits(model, cfg, inside)[k] - base[k]).abs().max()), 1e-6,
                                   f"R={frames}: frame {k} should see bin {last_bin}")

    def test_padded_batch_matches_single_trial(self):
        cfg = args(3)
        model = build_model(cfg).eval()
        long = torch.as_tensor(self.features).unsqueeze(0)
        short = long[:, :25]                                  # 12 output frames
        batch = torch.cat([long, torch.nn.functional.pad(short, (0, 0, 0, 16))], dim=0)
        days = torch.tensor([1, 1])
        with torch.inference_mode():
            single = model(short, torch.tensor([1]))[0]
            masked = model(batch, days, frame_lens=torch.tensor([20, 12]))[1, :12]
            unmasked = model(batch, days)[1, :12]
        torch.testing.assert_close(masked, single, rtol=0, atol=1e-5)
        self.assertGreater(float((unmasked - single).abs().max()), 1e-5)   # without lengths, padding would leak in

    def test_parallel_recurrence_matches_windowed_gru(self):
        for frames in (0, 2, 5):
            model = build_model(args(frames)).eval()
            h = torch.randn(2, 9, fx.model_args(0)["model"]["n_units"])
            lens = torch.tensor([9, 6])
            with torch.inference_mode():
                fast = model._lookahead(h, lens)
                masked = h * (torch.arange(9).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1)
                padded = torch.nn.functional.pad(masked, (0, 0, 0, frames))
                for b in range(2):
                    for t in range(9):
                        window = padded[b, t:t + frames + 1].flip(0).unsqueeze(0)   # t+R first
                        out, _ = model.lookahead_gru(window)
                        torch.testing.assert_close(fast[b, t], out[0, -1], rtol=1e-5, atol=1e-6)

    def test_r0_is_a_capacity_control(self):
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(build_model(args(0))), count(build_model(args(5))))
        default = build_model(fx.model_args(0))
        self.assertFalse(any("lookahead" in key for key in default.state_dict()))
        self.assertEqual(default.out.in_features, fx.model_args(0)["model"]["n_units"])

    def test_invalid_combinations_raise(self):
        cfg = args(2)
        cfg["model"]["bidirectional"] = True
        with self.assertRaises(ValueError):
            build_model(cfg)
        with self.assertRaises(ValueError):
            build_model(args(2, units=0))

    def test_streaming_matches_offline(self):
        for frames in (0, 1, 3):
            cfg = args(frames)
            model = build_model(cfg).eval()
            for n_bins in (41, 9):                           # 9 bins: 4 frames, fewer than R + 1 for R = 3 at the end
                features = self.features[:n_bins]
                result = run_streaming_trial(model, cfg, torch.device("cpu"), features, 1, enable_timing=False)
                offline = self.logits(model, cfg, features)
                self.assertEqual(result["n_frames"], offline.shape[0])
                torch.testing.assert_close(result["logits"].float(), offline.float(), rtol=1e-5, atol=1e-5)

    def test_streaming_emits_frame_t_when_frame_t_plus_r_arrives(self):
        cfg = args(3)
        model = build_model(cfg).eval()
        decoder = StreamingDecoder(model, cfg, 1, torch.device("cpu"))
        lookahead = int(cfg["dataset"]["data_transforms"]["smooth_lookahead"])
        emitted = []
        for raw in self.features:
            emitted.append(len(decoder.process_bin(raw)))
        tail = len(decoder.finish())
        causal_frames = decoder.n_emitted_frames              # all outputs, streamed plus flushed
        first = next(i for i, n in enumerate(emitted) if n)
        # Causal frame k is ready at bin 2k + 2 (+ smoothing look-ahead); output 0 waits for causal frame 3
        self.assertEqual(first, 2 * 3 + 2 + lookahead)
        self.assertEqual(sum(emitted) + tail, causal_frames)
        self.assertEqual(tail, 3)

    def test_registry_and_streaming_check(self):
        cfg = args(2)
        model = build_model(cfg).eval()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg = fx.make_registry(root)
            checkpoint = root / "models" / "tiny_la" / "checkpoint"
            checkpoint.mkdir(parents=True)
            torch.save({"model_state_dict": model.state_dict()}, checkpoint / "best_checkpoint")
            from omegaconf import OmegaConf
            OmegaConf.save(OmegaConf.create(cfg), checkpoint / "args.yaml")
            (reg / "acoustic" / "tiny_la.yaml").write_text(
                f"name: tiny_la\ncheckpoint_dir: {checkpoint}\nselection: {{rule: test, data: [val-dev]}}\n")
            spec = registry.resolve(acoustic="tiny_la", lm="fake", root=reg)
            self.assertTrue(spec["streaming_capable"])
            flags = registry.flags(spec, "val-dev")
            self.assertIn("lookahead_80ms", flags)             # 2 frames x stride 2 bins x 20 ms
            self.assertNotIn("offline_noncausal", flags)
            self.assertFalse(any(f.startswith("offline_lookahead") for f in flags))
            check = streaming_check(checkpoint, fx.make_data(root), device="cpu")
            self.assertTrue(check["passed"], check)
            delay = metrics.structural_delay(spec["preprocess"]["effective"], 3, 2, lookahead_frames=2)
            self.assertEqual(delay["model_lookahead_ms"], 80.0)
            self.assertEqual(delay["first_output_ms"],
                             metrics.structural_delay(spec["preprocess"]["effective"], 3, 2)["first_output_ms"] + 80.0)

if __name__ == "__main__":
    unittest.main()
