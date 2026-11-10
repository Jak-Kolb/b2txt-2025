"""Look-ahead variants (raw / both inputs, palindrome mode) and input compression: support, equivalence, streaming."""
import sys
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("acoustic model tests require the Python 3.10 acoustic stack")

import numpy as np
import torch

import harness_fixture as fx
from model_training.benchmark.common import build_model, offline_logits
from model_training.benchmark.streaming_infer import run_streaming_trial


def args(frames=None, source="hidden", mode="backward", bottleneck=0):
    cfg = fx.model_args(0)   # patch 3 bins, stride 2 bins, causal smoothing: frame k ends at bin 2k+2
    if frames is not None:
        cfg["model"].update(lookahead_frames=frames, lookahead_units=6, lookahead_input=source,
                            lookahead_raw_dim=5, lookahead_mode=mode)
    if bottleneck:
        cfg["model"]["input_bottleneck"] = bottleneck
    return cfg


VARIANTS = [dict(source="raw"), dict(source="both"), dict(source="raw", mode="palindrome"),
            dict(source="hidden", mode="palindrome"), dict(source="hidden", bottleneck=3),
            dict(source="raw", mode="palindrome", bottleneck=3)]


class LookaheadVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = np.random.default_rng(1).normal(size=(41, fx.N_FEATURES)).astype(np.float32)

    def logits(self, model, cfg, features):
        with torch.inference_mode():
            return offline_logits(model, torch.as_tensor(features).unsqueeze(0), 1, cfg, torch.device("cpu"))[0]

    def test_support_is_exactly_r_frames(self):
        for variant in VARIANTS:
            for frames in (0, 2):
                cfg = args(frames, **variant)
                model = build_model(cfg).eval()
                base = self.logits(model, cfg, self.features)
                k = 6
                last_bin = 2 * (k + frames) + 2
                beyond = self.features.copy()
                beyond[last_bin + 1:] += 5.0
                torch.testing.assert_close(self.logits(model, cfg, beyond)[:k + 1], base[:k + 1], rtol=0, atol=1e-6,
                                           msg=f"{variant} R={frames}")
                inside = self.features.copy()
                inside[last_bin] += 5.0
                self.assertGreater(float((self.logits(model, cfg, inside)[k] - base[k]).abs().max()), 1e-6,
                                   f"{variant} R={frames}")

    def test_padded_batch_matches_single_trial(self):
        for variant in VARIANTS:
            cfg = args(3, **variant)
            model = build_model(cfg).eval()
            long = torch.as_tensor(self.features).unsqueeze(0)
            short = long[:, :25]                              # 12 output frames
            batch = torch.cat([long, torch.nn.functional.pad(short, (0, 0, 0, 16))], dim=0)
            with torch.inference_mode():
                single = model(short, torch.tensor([1]))[0]
                masked = model(batch, torch.tensor([1, 1]), frame_lens=torch.tensor([20, 12]))[1, :12]
            torch.testing.assert_close(masked, single, rtol=0, atol=1e-5, msg=str(variant))

    def test_palindrome_matches_explicit_loop(self):
        model = build_model(args(2, source="raw", mode="palindrome")).eval()
        u, h = torch.randn(2, 8, 5), torch.randn(2, 8, fx.model_args(0)["model"]["n_units"])
        lens = torch.tensor([8, 5])
        with torch.inference_mode():
            fast = model._palindrome(u, h, lens)
            masked = u * (torch.arange(8).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1)
            padded = torch.nn.functional.pad(masked, (0, 0, 0, 2))
            for b in range(2):
                for t in range(8):
                    window = padded[b, t:t + 3]
                    sequence = torch.cat([window, window.flip(0)]).unsqueeze(0)     # t..t+R, then t+R..t
                    start = torch.tanh(model.lookahead_init(h[b, t])).view(1, 1, -1)
                    out, _ = model.lookahead_gru(sequence, start)
                    torch.testing.assert_close(fast[b, t], out[0, -1], rtol=1e-5, atol=1e-6)

    def test_compression_streams_and_defaults_are_unchanged(self):
        default = build_model(fx.model_args(0))
        self.assertFalse(any(k.startswith(("input_compress", "lookahead")) for k in default.state_dict()))
        cfg = args(bottleneck=3)
        model = build_model(cfg).eval()
        self.assertEqual(model.gru.input_size, 3 * cfg["model"]["patch_size"])
        streamed = run_streaming_trial(model, cfg, torch.device("cpu"), self.features, 1, enable_timing=False)["logits"]
        torch.testing.assert_close(streamed.float(), self.logits(model, cfg, self.features).float(), rtol=1e-4, atol=1e-4)

    def test_streaming_matches_offline(self):
        for variant in VARIANTS:
            cfg = args(2, **variant)
            model = build_model(cfg).eval()
            for n_bins in (41, 9):
                features = self.features[:n_bins]
                streamed = run_streaming_trial(model, cfg, torch.device("cpu"), features, 1, enable_timing=False)["logits"]
                torch.testing.assert_close(streamed.float(), self.logits(model, cfg, features).float(),
                                           rtol=1e-5, atol=1e-5, msg=str(variant))

    def test_invalid_settings_raise(self):
        bad = fx.model_args(0)
        bad["model"]["lookahead_input"] = "raw"           # no look-ahead module
        with self.assertRaises(ValueError):
            build_model(bad)
        with self.assertRaises(ValueError):
            build_model(args(2, mode="sideways"))


if __name__ == "__main__":
    unittest.main()
