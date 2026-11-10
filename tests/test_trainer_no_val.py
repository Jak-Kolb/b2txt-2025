"""Trainer validation when no session has enabled validation data (e.g. T12-only pretraining)."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
    import torchaudio  # noqa: F401  (rnn_trainer dependency)
    import omegaconf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("requires torch/torchaudio/omegaconf (acoustic environment)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from rnn_trainer import BrainToTextDecoder_Trainer


class NoValidationDataTests(unittest.TestCase):
    def fake_trainer(self, probability_val):
        args = dict(dataset=dict(sessions=[f"t12.2022.05.0{i}" for i in range(len(probability_val))],
                                 dataset_probability_val=probability_val),
                    log_val_skip_logs=False)
        return SimpleNamespace(args=args, model=torch.nn.Linear(1, 1), device=torch.device("cpu"))

    def test_empty_validation_set_reports_nan_instead_of_dividing_by_zero(self):
        metrics = BrainToTextDecoder_Trainer.validation(self.fake_trainer([0, 0]), loader=[])
        self.assertTrue(math.isnan(metrics["avg_PER"]))
        self.assertTrue(math.isnan(metrics["avg_loss"]))
        self.assertEqual(metrics["day_PERs"], {})
        self.assertFalse(metrics["avg_PER"] < math.inf)          # never selected as a new best checkpoint

    def test_batches_from_disabled_days_are_skipped(self):
        batch = dict(input_features=torch.zeros(2, 20, 512), seq_class_ids=torch.ones(2, 3, dtype=torch.int32),
                     n_time_steps=torch.tensor([20, 20]), phone_seq_lens=torch.tensor([3, 3]),
                     day_indicies=torch.tensor([1, 1]))
        metrics = BrainToTextDecoder_Trainer.validation(self.fake_trainer([0, 0]), loader=[batch])
        self.assertTrue(math.isnan(metrics["avg_PER"]))


class SharedWeightTransferTests(unittest.TestCase):
    def model(self, n_days, seed):
        from model_factory import build_model
        torch.manual_seed(seed)
        args = dict(model=dict(n_input_features=6, n_units=8, n_layers=2, patch_size=3, patch_stride=2, rnn_dropout=0.0,
                               lookahead_frames=2, lookahead_units=4, input_network=dict(input_layer_dropout=0.0)),
                    dataset=dict(sessions=[f"s{i}" for i in range(n_days)], n_classes=5))
        return build_model(args)

    def test_loads_everything_but_day_layers_from_a_compiled_checkpoint(self):
        import tempfile
        source, target = self.model(2, seed=1), self.model(3, seed=2)
        fresh_days = {k: v.clone() for k, v in target.state_dict().items() if k.startswith("day_")}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt"
            torch.save({"model_state_dict": {f"_orig_mod.{k}": v for k, v in source.state_dict().items()}}, path)
            fake = SimpleNamespace(model=target, logger=SimpleNamespace(info=lambda msg: None), args={})
            BrainToTextDecoder_Trainer.load_shared_weights(fake, str(path))
        for key, value in target.state_dict().items():
            expected = fresh_days[key] if key.startswith("day_") else source.state_dict()[key]
            torch.testing.assert_close(value, expected, rtol=0, atol=0, msg=key)

    def test_shrink_and_perturb_mixes_pretrained_and_fresh_weights(self):
        import tempfile
        source, target = self.model(2, seed=1), self.model(3, seed=2)
        fresh = {k: v.clone() for k, v in target.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt"
            torch.save({"model_state_dict": source.state_dict()}, path)
            fake = SimpleNamespace(model=target, logger=SimpleNamespace(info=lambda msg: None),
                                   args=dict(init_shared_shrink=0.4, init_shared_perturb=0.5))
            BrainToTextDecoder_Trainer.load_shared_weights(fake, str(path))
        for key, value in target.state_dict().items():
            expected = fresh[key] if key.startswith("day_") else 0.4 * source.state_dict()[key] + 0.5 * fresh[key]
            torch.testing.assert_close(value, expected, msg=key)

    def test_architecture_mismatch_raises(self):
        import tempfile
        source = self.model(2, seed=1)
        target_args_units = self.model(2, seed=2)
        target_args_units.lookahead_gru = torch.nn.GRU(8, 6, batch_first=True)   # different look-ahead size
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt"
            torch.save({"model_state_dict": source.state_dict()}, path)
            fake = SimpleNamespace(model=target_args_units, logger=SimpleNamespace(info=lambda msg: None), args={})
            with self.assertRaises((ValueError, RuntimeError)):
                BrainToTextDecoder_Trainer.load_shared_weights(fake, str(path))


if __name__ == "__main__":
    unittest.main()
