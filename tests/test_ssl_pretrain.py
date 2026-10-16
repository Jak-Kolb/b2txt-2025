"""Masked pretraining: hidden bins stay hidden, mask ratio, loss trains, day-layer transfer, label-free pool."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import h5py
    import numpy as np
    import torch
    from omegaconf import OmegaConf
except ImportError:
    raise unittest.SkipTest("requires the acoustic environment")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model_training"))
from model_factory import build_model
import ssl_pretrain as sp
from rnn_trainer import BrainToTextDecoder_Trainer
from data_prep import build_ssl_pool as pool


def cfg(kind="gru", lookahead=0, sessions=("t15.2023.08.11", "t12.2022.04.28")):
    model = dict(n_input_features=6, n_units=8, n_layers=2, patch_size=3, patch_stride=2, rnn_dropout=0.0,
                 input_network=dict(input_layer_dropout=0.0))
    if kind == "gru" and lookahead:
        model.update(lookahead_frames=lookahead, lookahead_units=4)
    if kind == "transformer":
        model.update(type="transformer", transformer=dict(d_model=8, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0,
                                                          window=16, lookahead_frames=lookahead))
    return OmegaConf.create(dict(model=model, dataset=dict(sessions=list(sessions), n_classes=5)))


class SSLTests(unittest.TestCase):
    def test_masked_bins_do_not_influence_any_output(self):
        torch.manual_seed(0)
        for kind, lookahead in (("gru", 0), ("gru", 2), ("transformer", 2)):
            model = build_model(cfg(kind, lookahead)).eval()
            x = torch.randn(1, 31, 6)
            mask = torch.zeros(1, 31, dtype=torch.bool); mask[0, 8:20] = True
            fill = torch.randn(6)
            with torch.no_grad():
                a = model(x, torch.tensor([0]), bin_mask=mask, mask_fill=fill, return_hidden=True)
                y = x.clone(); y[0, 8:20] += 7.0
                b = model(y, torch.tensor([0]), bin_mask=mask, mask_fill=fill, return_hidden=True)
                y2 = x.clone(); y2[0, 25] += 7.0                       # an unmasked bin does matter
                c = model(y2, torch.tensor([0]), bin_mask=mask, mask_fill=fill, return_hidden=True)
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f"{kind} R={lookahead}")
            self.assertGreater(float((a - c).abs().max()), 1e-6)

    def test_mask_ratio_and_padding(self):
        g = torch.Generator().manual_seed(0)
        mask = sp.draw_bin_mask(torch.tensor([100, 40]), 120, 0.3, 5, 10, g)
        self.assertGreaterEqual(int(mask[0, :100].sum()), 30)
        self.assertLessEqual(int(mask[0, :100].sum()), 30 + 10)
        self.assertFalse(mask[1, 40:].any())

    def test_loss_decreases_and_participant_heads_differ(self):
        torch.manual_seed(0)
        c = cfg("gru", 2)
        model = build_model(c)
        heads = sp.SSLHeads(6, 8 + 4, 2, 6)
        opt = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=1e-2)
        t = torch.linspace(0, 6.28, 41)
        x = torch.stack([torch.sin(t * (k + 1)) for k in range(6)], dim=1).repeat(4, 1, 1)
        n_bins = torch.tensor([41, 41, 35, 30])
        g = torch.Generator().manual_seed(1)
        losses = []
        for _ in range(250):
            mask = sp.draw_bin_mask(n_bins, 41, 0.3, 3, 6, g)
            loss, n, _ = sp.ssl_loss(model, heads, x, n_bins, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]), mask, c)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.assertGreater(n, 0)
        zero_predictor = float((x ** 2).mean())                     # MSE of predicting 0 for every hidden bin
        self.assertLess(np.mean(losses[-20:]), 0.5 * zero_predictor)

    def test_gauss_nll_and_contrastive_objectives_train(self):
        torch.manual_seed(0)
        t = torch.linspace(0, 12.56, 81)
        x = torch.stack([torch.sin(t * (k + 1) * 0.5 + k) for k in range(6)], dim=1).repeat(4, 1, 1)
        n_bins = torch.tensor([81, 81, 70, 60])
        for objective in ("gauss_nll", "contrastive"):
            c = cfg("gru", 2)
            c.ssl = dict(temperature=0.1)
            model = build_model(c)
            heads = sp.SSLHeads(6, 8 + 4, 2, 6, objective=objective, proj_dim=16)
            opt = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=1e-2)
            g = torch.Generator().manual_seed(1)
            first = last = None
            for step in range(200):
                mask = sp.draw_bin_mask(n_bins, 81, 0.4, 4, 8, g)
                loss, n, extra = sp.ssl_loss(model, heads, x, n_bins, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]), mask, c)
                opt.zero_grad(); loss.backward(); opt.step()
                first = float(loss.detach()) if first is None else first
                last = float(loss.detach())
            self.assertGreater(n, 0, objective)
            self.assertLess(last, first, objective)
            if objective == "contrastive":
                self.assertGreater(extra["acc"], 0.3)          # far above chance for ~10-frame candidate sets

    def test_day_layers_transfer_by_session_name(self):
        source = build_model(cfg(sessions=("t15.a", "t15.b", "t12.c")))
        for i, w in enumerate(source.day_weights):
            torch.nn.init.constant_(w, float(i + 1))
        target_cfg = cfg(sessions=("t15.b", "t15.new"))
        target = build_model(target_cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt"
            torch.save(dict(model_state_dict=source.state_dict(), sessions=["t15.a", "t15.b", "t12.c"]), path)
            fake = SimpleNamespace(model=target, logger=SimpleNamespace(info=lambda m: None),
                                   args=dict(init_shared_day_layers=True, dataset=dict(sessions=["t15.b", "t15.new"])))
            BrainToTextDecoder_Trainer.load_shared_weights(fake, str(path))
        self.assertTrue(torch.all(target.day_weights[0] == 2.0))                  # t15.b <- source index 1
        torch.testing.assert_close(target.day_weights[1], torch.eye(6))           # new session keeps fresh identity
        torch.testing.assert_close(target.gru.weight_hh_l0, source.gru.weight_hh_l0)

    def test_pool_blanks_labels_and_skips_val(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sess = root / "b2t" / "t15.2023.08.11"; sess.mkdir(parents=True)
            for split, n in (("train", 2), ("test", 1), ("val", 3)):
                with h5py.File(sess / f"data_{split}.hdf5", "w") as f:
                    for i in range(n):
                        g = f.create_group(f"trial_{i:04d}")
                        g.create_dataset("input_features", data=np.ones((5 + i, 512), np.float32))
                        g.create_dataset("seq_class_ids", data=np.ones(500, np.int32))
                        g.attrs.update(n_time_steps=5 + i, seq_len=3, block_num=1, trial_num=i, sentence_label="secret")
            summary = pool.main(["--b2t-dir", str(root / "b2t"), "--out-dir", str(root / "pool")])
            self.assertEqual(summary["n_trials"], 3)
            with h5py.File(root / "pool" / "t15.2023.08.11" / "data_train.hdf5", "r") as f:
                self.assertEqual(len(f), 3)
                self.assertTrue(all(f[k].attrs["sentence_label"] == "" and int(f[k].attrs["seq_len"]) == 0 for k in f))
                self.assertEqual(sorted(f[k].attrs["source_split"] for k in f), ["test", "train", "train"])


if __name__ == "__main__":
    unittest.main()
