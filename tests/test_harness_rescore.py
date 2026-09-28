"""Finalization rescoring as a derived record: fake native n-best, tiny local GPT-2 (no downloads)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

import harness_fixture as fx
from harness import bench, registry, rescore


class RescoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.data = fx.make_data(root)
        cls.registry = fx.make_registry(root)
        cls.results = root / "results"
        cls.env = patch.dict(os.environ, dict(fx.fake_native_env(), HF_HUB_OFFLINE="1"))
        cls.env.start()
        try:
            from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except OSError as exc:  # no cached GPT-2 tokenizer on this machine
            raise unittest.SkipTest(f"gpt2 tokenizer not cached: {exc}")
        cls.model_dir = root / "tiny_gpt2"
        GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(tokenizer))).save_pretrained(cls.model_dir)
        tokenizer.save_pretrained(cls.model_dir)
        (cls.registry / "rescorer").mkdir()
        (cls.registry / "rescorer" / "tiny.yaml").write_text(
            f"name: tiny\nmodel: {cls.model_dir}\nnbest: 5\nweights: {{gamma: 0.25, alpha: 1.0, beta: 0.0}}\n"
            "tuned_on: {partition: val-dev, acoustic: someone_else, lm: fake, artifact: none}\n")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.temp.cleanup()

    def test_rescoring_record(self):
        spec = registry.resolve("tiny_fake", root=self.registry)
        run = bench.run_standard(spec, root=self.results, data_dir=self.data, device="cpu", lm_python=sys.executable,
                                 log=lambda *_: None, timing_check=False)
        out = rescore.rescore_run(run, "tiny", root=self.results, device="cpu", lm_python=sys.executable,
                                  log=lambda *_: None, registry_root=self.registry)
        record = json.loads((out / "record.json").read_text())
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(record["status"], "complete")
        self.assertIn("rescorer_tuned_for_other_acoustic", record["flags"])
        self.assertEqual(summary["self_check"]["identity_weights_reproduce_run_1best"], summary["self_check"]["n_trials"])
        self.assertLessEqual(summary["oracle_wer"], summary["wer"]["percent"] + 1e-9)
        self.assertEqual(len((out / "trials.jsonl").read_text().splitlines()), 8)
        self.assertEqual(rescore.latest(run.name, self.results)[0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
