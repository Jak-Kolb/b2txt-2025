"""Causal Transformer acoustic model: streaming equals offline, causality, and builder dispatch."""
import sys
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("acoustic model tests require the Python 3.10 acoustic stack")

import numpy as np
import torch

from model_training.benchmark.common import build_model, offline_logits
from model_training.benchmark.streaming_infer import run_streaming_trial


def args(model_type="transformer", lookahead=0, window=3, future=0, **extra):
    model = {"n_input_features": 4, "n_units": 8, "n_layers": 2, "patch_size": 3, "patch_stride": 2,
             "rnn_dropout": 0.0, "input_network": {"input_layer_dropout": 0.0}, "type": model_type,
             "transformer": {"d_model": 16, "n_layers": 2, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                             "window": window, "lookahead_frames": future, **extra}}
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

    def test_mask_token_replaces_masked_frames(self):
        cfg = args(window=64)
        cfg["model"]["transformer"]["mask_token"] = True
        model = build_model(cfg).eval()
        x, day = torch.randn(1, 30, 4), torch.tensor([0])
        full = torch.ones(1, 14, dtype=torch.bool)                 # (30 - 3) // 2 + 1 = 14 frames
        a = model(x, day, frame_mask=full)
        b = model(torch.randn(1, 30, 4), day, frame_mask=full)
        self.assertTrue(torch.allclose(a, b))                       # fully masked: the input no longer matters
        none = torch.zeros(1, 14, dtype=torch.bool)
        self.assertTrue(torch.allclose(model(x, day, frame_mask=none), model(x, day)))
        with self.assertRaises(ValueError):
            self.model()[0](x, day, frame_mask=full)                # model without a token
        self.assertNotIn("mask_token", self.model()[0].state_dict())  # existing checkpoints still load

    def test_masked_leading_frames_keep_gradients_bounded(self):
        # Zero-filled input bins at a trial start leave exactly constant rows through the whole stack at
        # init; LayerNorm's backward then scales gradients by 1/sqrt(eps) per norm. The token avoids that.
        cfg = args(window=64)
        cfg["model"]["transformer"].update(n_layers=4, mask_token=True)
        model = build_model(cfg).train()
        x = torch.randn(2, 60, 4)
        mask = torch.zeros(2, 29, dtype=torch.bool)
        mask[:, :10] = True
        model(x, torch.tensor([0, 1]), frame_mask=mask).log_softmax(-1).mean().backward()
        norm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters() if p.grad is not None))
        self.assertLess(norm.item(), 1e3)
        zeroed = build_model(dict(cfg, model=dict(cfg["model"], transformer=dict(cfg["model"]["transformer"],
                                                                                  mask_token=False)))).train()
        x0 = x.clone()
        x0[:, :25] = 0.0                                            # frames 0-9 see only zeroed bins
        zeroed(x0, torch.tensor([0, 1])).log_softmax(-1).mean().backward()
        norm0 = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in zeroed.parameters() if p.grad is not None))
        self.assertGreater(norm0.item(), 1e6)                       # documents the failure the token avoids


    def test_lookahead_support_is_exactly_r_frames_in_total(self):
        # frame k of patch 3 / stride 2 ends at bin 2k + 2; causal smoothing adds no future bins
        for future in (0, 1, 3):
            model, cfg = self.model(window=64, future=future)
            with torch.inference_mode():
                base = offline_logits(model, torch.as_tensor(self.features)[None], 1, cfg, torch.device("cpu"))[0]
                k = 6
                last_bin = 2 * (k + future) + 2
                beyond = self.features.copy()
                beyond[last_bin + 1:] += 5.0
                moved = offline_logits(model, torch.as_tensor(beyond)[None], 1, cfg, torch.device("cpu"))[0]
                torch.testing.assert_close(moved[:k + 1], base[:k + 1], rtol=0, atol=1e-6)
                inside = self.features.copy()
                inside[last_bin] += 5.0
                changed = offline_logits(model, torch.as_tensor(inside)[None], 1, cfg, torch.device("cpu"))[0]
                self.assertGreater(float((changed[k] - base[k]).abs().max()), 1e-6)

    def test_lookahead_padded_batch_matches_single_trial_and_r0_is_causal_model(self):
        model, _ = self.model(window=64, future=2)
        long = torch.as_tensor(self.features)[None]
        short = long[:, :25]                                          # 12 output frames
        batch = torch.cat([long, torch.nn.functional.pad(short, (0, 0, 0, 16))], dim=0)
        with torch.inference_mode():
            single = model(short, torch.tensor([1]))[0]
            masked = model(batch, torch.tensor([1, 1]), frame_lens=torch.tensor([20, 12]))[1, :12]
        torch.testing.assert_close(masked, single, rtol=0, atol=1e-5)
        causal, _ = self.model(window=64, future=0)
        causal.load_state_dict(model.state_dict())
        with torch.inference_mode():
            a = causal(long, torch.tensor([1]))
            model.lookahead_frames = 0
            b = model(long, torch.tensor([1]))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_conformer_support_padding_and_drop_path(self):
        # causal convolution adds past context only: total future support stays exactly R frames
        conformer = dict(block_type="conformer", conv_kernel=3, drop_path=0.3)
        for block, future in (("conformer", 0), ("conformer", 2), ("conv", 0), ("conv", 2)):
            model, cfg = self.model(window=64, future=future, **dict(conformer, block_type=block))
            with torch.inference_mode():
                base = offline_logits(model, torch.as_tensor(self.features)[None], 1, cfg, torch.device("cpu"))[0]
                k = 6
                last_bin = 2 * (k + future) + 2
                beyond = self.features.copy()
                beyond[last_bin + 1:] += 5.0
                moved = offline_logits(model, torch.as_tensor(beyond)[None], 1, cfg, torch.device("cpu"))[0]
                torch.testing.assert_close(moved[:k + 1], base[:k + 1], rtol=0, atol=1e-6)
                inside = self.features.copy()
                inside[last_bin] += 5.0
                changed = offline_logits(model, torch.as_tensor(inside)[None], 1, cfg, torch.device("cpu"))[0]
                self.assertGreater(float((changed[k] - base[k]).abs().max()), 1e-6)
        model, _ = self.model(window=64, future=2, **conformer)
        long = torch.as_tensor(self.features)[None]
        short = long[:, :25]
        batch = torch.cat([long, torch.nn.functional.pad(short, (0, 0, 0, 16))], dim=0)
        with torch.inference_mode():
            single = model(short, torch.tensor([1]))[0]
            masked = model(batch, torch.tensor([1, 1]), frame_lens=torch.tensor([20, 12]))[1, :12]
        torch.testing.assert_close(masked, single, rtol=0, atol=1e-5)
        # stochastic depth is active only in training, and the last layer gets the full rate
        self.assertEqual([layer.drop_path for layer in model.layers], [0.0, 0.3])
        model.train()
        torch.manual_seed(0)
        a = model(long, torch.tensor([1]), frame_lens=torch.tensor([20]))
        torch.manual_seed(1)
        b = model(long, torch.tensor([1]), frame_lens=torch.tensor([20]))
        self.assertFalse(torch.allclose(a, b))
        with self.assertRaises(NotImplementedError):
            self.model(window=64, **conformer)[0].stream_step(torch.zeros(1, 1, 12), [None, None])

    def test_lookahead_model_refuses_to_stream(self):
        model, cfg = self.model(window=64, future=2)
        with self.assertRaises(ValueError):
            run_streaming_trial(model, cfg, torch.device("cpu"), self.features, 1, enable_timing=False)

    def test_compiled_gradients_match_eager(self):
        # inductor (torch 2.13) got the day-layer gradient ~150% wrong for matmul + overlapping-unfold patching
        import copy
        from model_training.rnn_model import GRUDecoder
        from model_training.transformer_model import TransformerDecoder
        x = torch.randn(2, 41, 4, dtype=torch.float64)
        day = torch.tensor([0, 1])
        for base in (GRUDecoder(4, 8, 2, 5, n_layers=1, patch_size=3, patch_stride=2),
                     TransformerDecoder(4, 2, 5, d_model=16, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0,
                                        patch_size=3, patch_stride=2, window=8, block_type="conv", conv_kernel=3)):
            base = base.double()
            for w in base.day_weights:
                torch.nn.init.normal_(w, std=0.5)
            grads = []
            for model in (copy.deepcopy(base), torch.compile(copy.deepcopy(base))):
                model(x, day).log_softmax(-1)[:, :, 0].sum().backward()
                grads.append({n: q.grad for n, q in getattr(model, "_orig_mod", model).named_parameters()})
            for name, g in grads[0].items():
                torch.testing.assert_close(grads[1][name], g, rtol=1e-9, atol=1e-12, msg=name)
            torch._dynamo.reset()

    def test_shared_input_layer(self):
        from model_training.transformer_model import TransformerDecoder
        torch.manual_seed(0)
        model = TransformerDecoder(4, 3, 5, d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0, patch_size=3,
                                   patch_stride=2, window=8, block_type="conv", conv_kernel=3, lookahead_frames=1,
                                   shared_input=True).eval()
        self.assertFalse(hasattr(model, "day_weights"))
        # the trainer puts every parameter named day_* in the day-layer group (its own lr, no weight decay)
        self.assertEqual(sorted(n for n, _ in model.named_parameters() if "day_" in n),
                         ["day_biases.0", "day_biases.1", "day_biases.2", "day_weight_shared"])
        torch.testing.assert_close(model.day_weight_shared.detach(), torch.eye(4))
        x = torch.randn(1, 41, 4)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x, torch.tensor([0])), model(x, torch.tensor([2]))))
            model.day_biases[2] += 0.5                              # sessions now differ only through their bias
            self.assertFalse(torch.equal(model(x, torch.tensor([0])), model(x, torch.tensor([2]))))

    def test_no_day_input_layer(self):
        from model_training.transformer_model import TransformerDecoder
        torch.manual_seed(0)
        model = TransformerDecoder(4, 3, 5, d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0, patch_size=3,
                                   patch_stride=2, window=8, no_day_input=True).eval()
        self.assertEqual([n for n, _ in model.named_parameters() if "day_" in n], [])
        x = torch.randn(1, 41, 4)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x, torch.tensor([0])), model(x, torch.tensor([2]))))
            # the front end is plain patching of the input
            torch.testing.assert_close(model.front_end(x, torch.tensor([0])),
                                       x.unfold(1, 3, 2).permute(0, 1, 3, 2).flatten(2))
            with self.assertRaises(NotImplementedError):
                model.stream_step(torch.zeros(1, 1, 12), model.initial_state())
        with self.assertRaises(ValueError):
            TransformerDecoder(4, 3, 5, d_model=16, n_layers=2, n_heads=2, no_day_input=True, shared_input=True)

    def test_intermediate_ctc_output(self):
        import copy
        from model_training.transformer_model import TransformerDecoder
        kw = dict(d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0, patch_size=3, patch_stride=2, window=8,
                  block_type="conv", conv_kernel=3, lookahead_frames=1)
        with self.assertRaises(ValueError):
            TransformerDecoder(4, 3, 5, inter_ctc_layer=2, **kw)
        torch.manual_seed(0)
        model = TransformerDecoder(4, 3, 5, inter_ctc_layer=1, **kw).double()
        x = torch.randn(2, 41, 4, dtype=torch.float64)
        day, lens = torch.tensor([0, 2]), torch.tensor([20, 14])
        model.eval()
        self.assertIsInstance(model(x, day, frame_lens=lens), torch.Tensor)
        model.train()
        logits, mid = model(x, day, frame_lens=lens)
        self.assertEqual(logits.shape, mid.shape)
        # the intermediate output is taken after block 1 of 2, so it cannot depend on the second block
        mid.sum().backward()
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.layers[1].parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.layers[0].parameters() if p.grad is not None), 0)
        # compiled training forward (tuple output) matches eager gradients
        model.zero_grad()
        compiled_base = copy.deepcopy(model)
        grads = []
        for m in (model, torch.compile(compiled_base)):
            out, inter = m(x, day, frame_lens=lens)
            (out.log_softmax(-1)[:, :, 0].sum() + inter.log_softmax(-1)[:, :, 1].sum()).backward()
            grads.append({n: q.grad for n, q in getattr(m, "_orig_mod", m).named_parameters()})
        for name, g in grads[0].items():
            torch.testing.assert_close(grads[1][name], g, rtol=1e-9, atol=1e-12, msg=name)
        torch._dynamo.reset()

    def test_time_span_length_cap(self):
        from model_training.data_augmentations import per_trial_time_spans
        torch.manual_seed(0)
        masked, _ = per_trial_time_spans(torch.full((200,), 100), 100, n_masks=1, max_frac=1.0, max_len=5)
        self.assertLessEqual(int(masked.sum(dim=1).max()), 5)
        self.assertGreater(int(masked.sum(dim=1).max()), 0)

    def test_relative_bias_is_causal_and_bounded_by_lookahead(self):
        from model_training.transformer_model import TransformerDecoder
        torch.manual_seed(0)
        kw = dict(d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0, patch_size=3, patch_stride=2, window=8,
                  no_day_input=True, pos_bias="relative", max_rel_dist=4)
        day = torch.tensor([0])
        for future in (0, 2):
            model = TransformerDecoder(4, 3, 5, lookahead_frames=future, **kw).double().eval()
            self.assertEqual(tuple(model.rel_bias.shape), (2, 7))
            x = torch.randn(1, 41, 4, dtype=torch.float64, requires_grad=True)
            model(x, day)[0, 6].sum().backward()
            # frame 6 covers bins 12..14; with R frames of look-ahead it may read up to frame 6 + R (bins .. 14 + 2R)
            last = 14 + 2 * future
            self.assertTrue(bool((x.grad[0, last + 1:] == 0).all()))
            self.assertTrue(bool((x.grad[0, last] != 0).any()))
            self.assertTrue(bool((model.rel_bias.grad != 0).any()))
        with self.assertRaises(NotImplementedError):
            TransformerDecoder(4, 3, 5, **kw).eval().stream_step(torch.zeros(1, 1, 12), [None, None])
        with self.assertRaises(ValueError):
            TransformerDecoder(4, 3, 5, d_model=16, n_layers=2, n_heads=2, pos_bias="sinusoidal")

    def test_sampled_window_only_in_training(self):
        from model_training.transformer_model import TransformerDecoder
        torch.manual_seed(0)
        kw = dict(d_model=16, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0, patch_size=3, patch_stride=2,
                  no_day_input=True)
        sampled = TransformerDecoder(4, 3, 5, window=8, window_min=2, **kw)
        fixed = TransformerDecoder(4, 3, 5, window=8, **kw)
        fixed.load_state_dict(sampled.state_dict())
        x, day = torch.randn(1, 41, 4), torch.tensor([0])
        with torch.no_grad():
            torch.testing.assert_close(sampled.eval()(x, day), fixed.eval()(x, day))
            sampled.train()
            outs = {round(float(sampled(x, day)[0, -1].sum()), 5) for _ in range(40)}
            self.assertGreater(len(outs), 2)                    # the window changes between forwards
            self.assertLessEqual(len(outs), 7)                  # windows 2..8
            # never more than `window` frames of context: frame 19 (bins 38..40) reads frames 12..19 at most
            changed = x.clone()
            changed[0, :24] += 1.0                              # bins 0..23 belong to frames <= 11 only
            for _ in range(20):
                state = torch.get_rng_state()
                a = sampled(x, day)[0, -1]
                torch.set_rng_state(state)
                torch.testing.assert_close(a, sampled(changed, day)[0, -1])
        with self.assertRaises(ValueError):
            TransformerDecoder(4, 3, 5, window=8, window_min=9, **kw)

    def test_consistency_loss(self):
        from model_training.rnn_trainer import consistency_loss
        torch.manual_seed(0)
        a = torch.randn(2, 6, 5, dtype=torch.float64)
        lens = torch.tensor([6, 4, 6, 4])
        torch.testing.assert_close(consistency_loss(torch.cat([a, a]), lens), torch.zeros(2), atol=1e-6, rtol=0)
        b = a.clone()
        b[1, 4:] += 3.0                                         # only padding frames of trial 1 differ
        self.assertEqual(float(consistency_loss(torch.cat([a, b]), lens)[1]), 0.0)
        b = (a + torch.randn_like(a)).requires_grad_()
        loss = consistency_loss(torch.cat([a, b]), lens)
        self.assertTrue(bool((loss > 0).all()))
        # one frame, by hand: 0.5 * [KL(b || a) + KL(a || b)]
        pa, pb = a[0, 0].log_softmax(-1), b[0, 0].detach().log_softmax(-1)
        one = consistency_loss(torch.stack([a[0, :1], b[0, :1].detach()]), torch.tensor([1, 1]))
        torch.testing.assert_close(one[0], 0.5 * ((pb.exp() * (pb - pa)).sum() + (pa.exp() * (pa - pb)).sum()), check_dtype=False, rtol=1e-5, atol=1e-7)
        loss.sum().backward()
        self.assertTrue(bool((b.grad[1, 4:] == 0).all()) and bool((b.grad[0] != 0).any()))

    def test_lightbeam_dropout_and_init_options(self):
        from model_training.transformer_model import TransformerDecoder
        kw = dict(d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.35, patch_size=4, patch_stride=4, window=8,
                  no_day_input=True)
        torch.manual_seed(0)
        old = TransformerDecoder(4, 3, 5, **kw)
        torch.manual_seed(0)
        new = TransformerDecoder(4, 3, 5, embed_dropout=0.2, ffn_dropout=True, init="torch", **kw)
        self.assertEqual(old.input_drop.p, 0.35)
        self.assertEqual(new.input_drop.p, 0.2)
        self.assertEqual([type(m).__name__ for m in new.layers[0].ffn], ["Linear", "GELU", "Dropout", "Linear"])
        self.assertEqual(new.layers[0].ffn[2].p, 0.35)
        self.assertIn("layers.0.ffn.2.weight", old.state_dict())       # existing checkpoints keep their names
        self.assertEqual(float(old.out.bias.abs().sum()), 0.0)         # Xavier path zeroes biases
        self.assertGreater(float(new.out.bias.abs().sum()), 0.0)       # PyTorch default does not
        bound = 1 / 16 ** 0.5                                          # default Linear init: U(-1/sqrt(fan_in), ...)
        self.assertLessEqual(float(new.out.weight.abs().max()), bound + 1e-6)
        x = torch.randn(2, 40, 4)
        self.assertEqual(tuple(new.eval()(x, torch.tensor([0, 1])).shape), (2, 10, 5))
        with self.assertRaises(ValueError):
            TransformerDecoder(4, 3, 5, init="zeros", **kw)

if __name__ == "__main__":
    unittest.main()
