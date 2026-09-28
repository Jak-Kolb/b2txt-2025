"""End-to-end standard benchmark on a tiny CPU GRU, synthetic HDF5, and the fake native LM.

The LM stage runs the real stream_lm.py CLI and the timing check the real lm_worker.py, both
as subprocesses whose PYTHONPATH puts tests/fake_native/lm_decoder.py first.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

import numpy as np

import harness_fixture as fx
from harness import bench, registry
from harness.store import flocked, list_runs, lock_path


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.data = fx.make_data(root)
        cls.registry = fx.make_registry(root)
        cls.results = root / "results"
        cls.env = patch.dict(os.environ, fx.fake_native_env())
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.temp.cleanup()

    def run_bench(self, pipeline="tiny_fake", results=None, **kwargs):
        spec = registry.resolve(pipeline, root=self.registry, **kwargs.pop("resolve", {}))
        return bench.run_standard(spec, root=results or self.results, data_dir=self.data, device="cpu",
                                  lm_python=sys.executable, log=lambda *_: None, **kwargs)

    def test_standard_run_end_to_end_and_cache_reuse(self):
        results = self.results.parent / "results_e2e"
        first = self.run_bench(results=results)
        manifest, summary, rows = bench.load_run(first)
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["logits_cache"]["hit"])
        self.assertEqual(len(rows), 8)
        accuracy = summary["accuracy"]
        self.assertEqual(accuracy["cross_checks"]["lm_stage_edits"], accuracy["wer"]["edits"])
        self.assertIsNotNone(accuracy["wer"]["ci95"])
        self.assertEqual(len(accuracy["per_session"]), 2)
        self.assertIn("counts", accuracy["error_attribution"])
        self.assertIn("seen", accuracy["seen_unseen"])
        self.assertTrue(any(r["seen"] for r in rows))
        timing = summary["timing_check"]
        self.assertEqual(timing["equivalence"]["n_trials"], 4)
        self.assertLess(timing["equivalence"]["max_logit_error"], 1e-3)
        self.assertEqual(timing["equivalence"]["final_text_compared"], 4)
        self.assertEqual(timing["structural_delay"]["first_output_ms"], 60.0)
        self.assertGreater(timing["service_ms"]["per_bin"]["n"], 0)
        for name in ("output_trace.jsonl", "timing_trace.jsonl", "lm_decode.json", "trials.jsonl"):
            self.assertIn(name, manifest["files_sha256"])

        second = self.run_bench(results=results)
        manifest2, _, rows2 = bench.load_run(second)
        self.assertTrue(manifest2["logits_cache"]["hit"])
        self.assertEqual([r["hyp"] for r in rows], [r["hyp"] for r in rows2])
        compared = bench.compare_runs(first, second)
        self.assertEqual(compared["delta_edits"], 0)
        self.assertEqual(compared["trials"]["different_text"], 0)

    def test_limit_scope_differs_and_compare_refuses(self):
        limited = self.run_bench(limit=2, timing_check=False)
        manifest, summary, _ = bench.load_run(limited)
        self.assertIn("smoke_limited", manifest["flags"])
        self.assertEqual(summary["timing_check"], {"skipped": "disabled (--no-timing-check)"})
        full = self.run_bench(timing_check=False)
        with self.assertRaisesRegex(ValueError, "different scopes"):
            bench.compare_runs(limited, full)

    def test_transformer_streams_through_the_same_benchmark(self):
        run = self.run_bench("tiny_fake", resolve=dict(acoustic="tiny_tfm"))
        manifest, summary, rows = bench.load_run(run)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(rows), 8)
        equivalence = summary["timing_check"]["equivalence"]
        self.assertLess(equivalence["max_logit_error"], 1e-3)
        self.assertEqual(equivalence["final_text_mismatches"], [])

    def test_sweep_runs_one_record_per_grid_point(self):
        spec = registry.resolve("tiny_fake", root=self.registry)
        runs = bench.run_sweep(spec, {"acoustic_scale": [0.4, 0.5], "blank_penalty": [2]}, root=self.results,
                               data_dir=self.data, device="cpu", lm_python=sys.executable, log=lambda *_: None)
        self.assertEqual(len(runs), 2)
        manifests = [bench.load_run(r)[0] for r in runs]
        self.assertEqual({m["tier"] for m in manifests}, {"sweep"})
        self.assertEqual([m["pipeline"]["decode"]["acoustic_scale"] for m in manifests], [0.4, 0.5])
        self.assertTrue(all(m["pipeline"]["decode"]["blank_penalty"] == 2.0 for m in manifests))
        self.assertTrue(all("decode_tuning_unrecorded" in m["flags"] for m in manifests))
        self.assertTrue(manifests[0]["pipeline"]["name"].startswith("tiny_fake@acoustic_scale=0.4"))
        self.assertIn("error_attribution", bench.load_run(runs[0])[1]["accuracy"])

    def test_noncausal_skips_timing_check(self):
        run = self.run_bench("sym_fake")
        manifest, summary, _ = bench.load_run(run)
        self.assertIn("offline_noncausal", manifest["flags"])
        self.assertEqual(summary["timing_check"]["skipped"], "offline_noncausal pipeline cannot stream")

    def test_lm_failure_marks_run_failed(self):
        with patch.dict(os.environ, {"PYTHONPATH": str(fx.REPO)}):
            with self.assertRaisesRegex(RuntimeError, "LM stage failed"):
                self.run_bench(limit=1, timing_check=False)
        failed = [r for r in list_runs(self.results) if r["status"] == "failed"]
        self.assertTrue(failed)
        self.assertTrue((Path(failed[0]["run_dir"]) / "traceback.txt").exists())

    def test_job_lock_prevents_concurrent_benchmarks(self):
        with flocked(lock_path(self.results, "job")):
            with self.assertRaises(BlockingIOError):
                self.run_bench(limit=1, timing_check=False)


class PacedTests(unittest.TestCase):
    """1x paced tier on the synthetic sample (4 short trials, about 3 s of real time)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.data = fx.make_data(root)
        cls.registry = fx.make_registry(root)
        cls.results = root / "results"
        cls.env = patch.dict(os.environ, fx.fake_native_env())
        cls.env.start()
        cls.spec = registry.resolve("tiny_fake", root=cls.registry)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.temp.cleanup()

    def common(self):
        return dict(root=self.results, data_dir=self.data, device="cpu", lm_python=sys.executable,
                    log=lambda *_: None)

    def test_paced_run_measures_lag_and_calibrates_against_standard(self):
        standard = bench.run_standard(self.spec, **self.common())
        paced = bench.run_paced(self.spec, **self.common())
        manifest, summary, rows = bench.load_run(paced)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["tier"], "paced")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["offline_native_text_match"] for r in rows))
        measured = summary["paced"]["frame_window_to_output_ms"]
        self.assertGreater(measured["n"], 0)
        self.assertGreater(measured["p50"], 0)
        self.assertEqual(summary["paced"]["exceedances"]["200.0"]["n"], 0)
        calibration = summary["sim_calibration"]
        self.assertEqual(calibration["standard_run"], standard.name)
        self.assertIn("p95", calibration["measured_minus_simulated_ms"])
        # Same trials and pipeline: paced and standard final text agree trial by trial.
        _, _, standard_rows = bench.load_run(standard)
        by_key = {(r["session"], r["trial_key"]): r["hyp"] for r in standard_rows}
        for row in rows:
            if (row["session"], row["trial_key"]) in by_key:
                self.assertEqual(row["hyp"], by_key[(row["session"], row["trial_key"])])

    def test_paced_refuses_noncausal_and_concurrent_jobs(self):
        with self.assertRaisesRegex(ValueError, "offline_noncausal"):
            bench.run_paced(registry.resolve("sym_fake", root=self.registry), **self.common())
        with flocked(lock_path(self.results, "job")):
            with self.assertRaises(BlockingIOError):
                bench.run_paced(self.spec, limit=1, **self.common())


class DecoderPathConsistencyTests(unittest.TestCase):
    """stream_lm.run_decode (accuracy stage) and lm_worker.NativeBackend (streaming) agree."""

    def test_identical_partials(self):
        sys.path.insert(0, str(fx.FAKE_NATIVE))
        try:
            import lm_decoder
        finally:
            sys.path.remove(str(fx.FAKE_NATIVE))
        from model_training.benchmark.lm_worker import NativeBackend
        from model_training.benchmark.output_trace import OutputTraceWriter
        from model_training.benchmark.stream_lm import run_decode
        logits = [np.random.default_rng(i).normal(size=(9, 5)).astype(np.float32) for i in range(3)]
        config = dict(acoustic_scale=0.4, blank_penalty=2.0, beam=17, lattice_beam=8, max_active=7000,
                      min_active=200, length_penalty=0, blank_skip_thresh=1.0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            with OutputTraceWriter(path, {}) as writer:
                run_decode(lm_decoder, temp, logits, ["x"] * 3, capture_partials=True,
                           trace_writer=writer, **config)
            partials = {}
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record["record"] == "output":
                    partials.setdefault(record["cache_index"], []).append(record["words"])
        with patch.dict(sys.modules, {"lm_decoder": lm_decoder}):
            backend = NativeBackend(temp, config, n_classes=5)
            for index, trial in enumerate(logits):
                backend.reset()
                # JSON transport round trip, as in the worker protocol.
                words = [backend.step(json.loads(json.dumps(row.tolist()))) for row in trial]
                words.append(backend.finish())
                self.assertEqual(words, partials[index])


if __name__ == "__main__":
    unittest.main()
