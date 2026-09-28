"""Retraining workflow tests with a fake trainer (no GPU training)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

import harness_fixture as fx
from harness import registry, train


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.data = fx.make_data(root)
        self.registry = fx.make_registry(root)
        self.results = root / "results"
        self.models = root / "trained_models"
        self.env = patch.dict(os.environ, fx.fake_native_env())
        self.env.start()
        self.addCleanup(self.env.stop)
        source = self.registry.parent / "models" / "tiny" / "checkpoint"
        self.fake = lambda loss="2.50": (lambda config: [sys.executable, str(fx.TESTS / "fake_trainer.py"),
                                                          str(config), str(source), loss])

    def fork(self, name="wide", overrides=("model.n_units=16",)):
        return train.fork("tiny", name, list(overrides), "Does width help?", root=self.results,
                          registry_root=self.registry, models_dir=self.models, data_dir=self.data,
                          smoke_batches=20)

    def test_fork_writes_fresh_absolute_paths_and_rejects_unknown_keys(self):
        train_dir = self.fork()
        config = OmegaConf.load(train_dir / "config.yaml")
        smoke = OmegaConf.load(train_dir / "smoke_config.yaml")
        self.assertEqual(config.model.n_units, 16)
        self.assertTrue(Path(config.output_dir).is_absolute())
        self.assertFalse(Path(config.output_dir).exists())
        self.assertEqual(Path(config.checkpoint_dir), Path(config.output_dir) / "checkpoint")
        self.assertEqual(Path(config.dataset.dataset_dir), self.data.resolve())
        self.assertEqual(smoke.num_training_batches, 20)
        self.assertEqual(smoke.batches_per_val_step, 20)
        self.assertTrue(str(smoke.output_dir).startswith(str(train_dir.resolve())))
        request = json.loads((train_dir / "request.json").read_text())
        self.assertEqual(request["selection"]["data"], ["val-dev", "former-val-test"])
        self.assertIn("head", request["code"])
        with self.assertRaises(ConfigKeyError):
            self.fork(name="typo", overrides=["model.n_unitz=16"])
        with self.assertRaises(FileExistsError):
            self.fork(name="tiny")
        with self.assertRaises(ValueError):
            train.fork("tiny", "x", [], " ", root=self.results, registry_root=self.registry)

    def test_additions_introduce_new_keys_explicitly(self):
        train_dir = train.fork("tiny", "tfm", [], "Transformer?", root=self.results,
                               registry_root=self.registry, models_dir=self.models, data_dir=self.data,
                               additions=["model.type=transformer", "model.transformer.d_model=16"])
        config = OmegaConf.load(train_dir / "config.yaml")
        self.assertEqual(config.model.type, "transformer")
        self.assertEqual(config.model.transformer.d_model, 16)
        self.assertEqual(json.loads((train_dir / "request.json").read_text())["additions"][0], "model.type=transformer")
        with self.assertRaisesRegex(ValueError, "already exist"):
            train.fork("tiny", "tfm2", [], "q", root=self.results, registry_root=self.registry,
                       models_dir=self.models, data_dir=self.data, additions=["model.n_units=4"])
        with self.assertRaises(ConfigKeyError):
            train.fork("tiny", "tfm3", ["model.type=transformer"], "q", root=self.results,
                       registry_root=self.registry, models_dir=self.models, data_dir=self.data)

    def test_log_parser_flags_nonfinite_values(self):
        path = Path(self.temp.name) / "training_log"
        path.write_text("Train batch 0: loss: nan grad norm: 1\nVal batch 0: PER (avg): 0.9 CTC\n")
        self.assertFalse(train.parse_log(path)["finite"])
        path.write_text("Train batch 0: loss: 3.1 grad\nVal batch 0: PER (avg): 0.9 CTC\n")
        self.assertTrue(train.parse_log(path)["finite"])
        self.assertFalse(train.parse_log(path.with_name("missing"))["finite"])

    def test_smoke_checks(self):
        train_dir = self.fork()
        passed, result = train.run_smoke(train_dir, data_dir=self.data, trainer=self.fake(), device="cpu")
        self.assertTrue(passed, result)
        self.assertTrue(result["streaming"]["passed"])
        bad_dir = self.fork(name="bad")
        passed, result = train.run_smoke(bad_dir, data_dir=self.data, trainer=self.fake("nan"), device="cpu")
        self.assertFalse(passed)
        self.assertEqual(json.loads((bad_dir / "state.json").read_text())["state"], "smoke_failed")

    def test_supervise_registers_then_benchmarks(self):
        train_dir = self.fork()
        train.set_state(train_dir, then_bench="tiny_fake")
        code = train.supervise(train_dir, root=self.results, trainer=self.fake(), registry_root=self.registry,
                               bench_kwargs=dict(data_dir=self.data, device="cpu", lm_python=sys.executable,
                                                 log=lambda *_: None, timing_check=False))
        self.assertEqual(code, 0)
        state = json.loads((train_dir / "state.json").read_text())
        self.assertEqual(state["state"], "benchmarked")
        entry = registry.load_kind("acoustic", self.registry)["wide"]
        self.assertEqual(entry["training"]["forked_from"], "tiny")
        self.assertEqual(entry["selection"]["data"], ["val-dev", "former-val-test"])
        manifest = json.loads((self.results / "runs" / state["benchmark_run"] / "manifest.json").read_text())
        self.assertEqual(manifest["pipeline"]["acoustic"]["name"], "wide")
        self.assertIn("decode_tuned_for_other_acoustic", manifest["flags"])
        with self.assertRaises(FileExistsError):
            train.register(train_dir, self.registry)
        self.assertEqual(train.status(self.results)[0]["state"], "benchmarked")


if __name__ == "__main__":
    unittest.main()
