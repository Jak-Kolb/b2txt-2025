"""Causal Transformer acoustic model: streaming equals offline, causality, and builder dispatch."""
import sys
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("acoustic model tests require the Python 3.10 acoustic stack")

import numpy as np
import torch

from model_training.benchmark.common import build_model, offline_logits
from model_training.benchmark.streaming_infer import run_streaming_trial


def args(model_type="transformer", lookahead=0, window=3):
    model = {"n_input_features": 4, "n_units": 8, "n_layers": 2, "patch_size": 3, "patch_stride": 2,
             "rnn_dropout": 0.0, "input_network": {"input_layer_dropout": 0.0}, "type": model_type,
             "transformer": {"d_model": 16, "n_layers": 2, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                             "window": window}}
    return {"model": model,
            "dataset": {"sessions": ["a", "b"], "n_classes": 5,
                        "data_transforms": {"smooth_data": True, "smooth_kernel_size": 8,
                                            "smooth_kernel_std": 1.0, "smooth_lookahead": lookahead}}}


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = np.random.default_rng(1).normal(size=(41, 4)).astype(np.float32)

    def model(self, **kwargs):
        cfg = args(**kwargs)
        return build_model(cfg).eval(), cfg

    def test_streaming_matches_offline(self):
        # window 3 < 19 patch frames exercises cache truncation; lookahead exercises the flush path
        for window, lookahead in ((3, 0), (3, 2), (64, 0)):
            model, cfg = self.model(window=window, lookahead=lookahead)
            streamed = run_streaming_trial(model, cfg, torch.device("cpu"), self.features, 1,
                                           enable_timing=False)["logits"]
            with torch.inference_mode():
                offline = offline_logits(model, torch.from_numpy(self.features).unsqueeze(0), 1, cfg,
                                         torch.device("cpu"))[0]
            self.assertEqual(streamed.shape, offline.shape)
            torch.testing.assert_close(streamed, offline, atol=1e-5, rtol=1e-4)

    def test_causal_and_padding_safe(self):
        model, _ = self.model(window=64)
        x = torch.from_numpy(self.features).unsqueeze(0)
        day = torch.tensor([0])
        with torch.inference_mode():
            full = model(x, day)
            padded = model(torch.cat((x, torch.zeros(1, 20, 4)), dim=1), day)
            changed = model(torch.cat((x[:, :25], x[:, 25:] + 5.0), dim=1), day)
        torch.testing.assert_close(padded[:, :full.shape[1]], full)
        # frames whose patch window ends before bin 25 cannot see the change
        unaffected = (25 - 3) // 2 + 1
        torch.testing.assert_close(changed[:, :unaffected], full[:, :unaffected])
        self.assertFalse(torch.allclose(changed[:, unaffected:], full[:, unaffected:]))

    def test_front_end_matches_gru(self):
        transformer, _ = self.model()
        gru = build_model(args("gru"))
        for i in range(2):
            torch.nn.init.normal_(transformer.day_weights[i])
            torch.nn.init.normal_(transformer.day_biases[i])
            gru.day_weights[i].data.copy_(transformer.day_weights[i].data)
            gru.day_biases[i].data.copy_(transformer.day_biases[i].data)
        x = torch.from_numpy(self.features).unsqueeze(0)
        captured = {}

        def capture(module, inputs, output):
            captured["x"] = inputs[0]
        gru.gru.register_forward_hook(capture)
        with torch.inference_mode():
            gru.eval()(x, torch.tensor([1]))
            torch.testing.assert_close(transformer.front_end(x, torch.tensor([1])), captured["x"])

    def test_training_step_and_dispatch(self):
        model, _ = self.model()
        model.train()
        logits = model(torch.randn(2, 30, 4), torch.tensor([0, 1]))
        logits.log_softmax(-1).mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        gru = build_model(args("gru"))
        self.assertEqual(type(gru).__name__, "GRUDecoder")
        legacy = args("gru")
        del legacy["model"]["type"]
        self.assertEqual(type(build_model(legacy)).__name__, "GRUDecoder")
        with self.assertRaises(ValueError):
            build_model(args("mamba"))
        with self.assertRaises(NotImplementedError):
            model(torch.randn(1, 10, 4), torch.tensor([0]), return_state=True)


if __name__ == "__main__":
    unittest.main()
