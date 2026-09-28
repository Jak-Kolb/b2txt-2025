"""Viewer API tests on synthetic runs (aiohttp test client; no torch in the server process)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

from aiohttp.test_utils import AioHTTPTestCase

import harness_fixture as fx
from harness import bench, registry
from harness.server import create_app


class ServerTests(AioHTTPTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        data = cls.data = fx.make_data(root)
        cls.registry = fx.make_registry(root)
        cls.results = root / "results"
        with patch.dict(os.environ, fx.fake_native_env()):
            spec = registry.resolve("tiny_fake", root=cls.registry)
            run = lambda **kw: bench.run_standard(spec, root=cls.results, data_dir=data, device="cpu",
                                                  lm_python=sys.executable, log=lambda *_: None, **kw)
            cls.full_a = run().name
            cls.full_b = run().name
            cls.limited = run(limit=2, timing_check=False).name

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    async def get_application(self):
        return create_app(self.results, registry_root=self.registry, data_dir=self.data)

    async def test_runs_and_detail(self):
        response = await self.client.get("/api/runs")
        rows = await response.json()
        self.assertEqual(len(rows), 3)
        row = next(r for r in rows if r["run_id"] == self.full_a)
        self.assertIsNotNone(row["wer"])
        self.assertIsNotNone(row["lag_p95_ms"])
        self.assertEqual(row["lag_kind"], "simulated")
        detail = await (await self.client.get(f"/api/runs/{self.full_a}")).json()
        self.assertEqual(detail["manifest"]["status"], "complete")
        trials = await (await self.client.get(f"/api/runs/{self.full_a}/trials")).json()
        self.assertEqual(len(trials), 8)
        self.assertEqual({t["corpus"] for t in trials}, {"Switchboard", "Random"})

    async def test_trial_events_and_timing(self):
        events = await (await self.client.get(f"/api/runs/{self.full_a}/trials/2/events")).json()
        self.assertEqual(events["schema_version"], 1)
        self.assertEqual(events["records"][0]["cache_index"], 2)
        self.assertEqual(events["records"][-1]["kind"], "final")
        timing = await (await self.client.get(f"/api/runs/{self.full_a}/trials/1/events?trace=timing")).json()
        self.assertEqual(timing["schema_version"], 2)
        quantiles = await (await self.client.get(f"/api/runs/{self.full_a}/timing")).json()
        self.assertEqual(len(quantiles["frame_window_to_output_ms"]), 101)
        self.assertTrue(quantiles["simulated"])
        missing = await self.client.get(f"/api/runs/{self.full_a}/trials/99/events")
        self.assertEqual(missing.status, 404)

    async def test_error_attribution_endpoints(self):
        errors = await self.client.get(f"/api/runs/{self.full_a}/errors")
        self.assertEqual(errors.status, 200)
        body = await errors.json()
        self.assertEqual(len(body["trials"]), 8)
        self.assertIn("pct_acoustic_word_errors_fixed_by_lm", body["summary"])
        ok = [t["i"] for t in body["trials"] if t["status"] == "ok"]
        self.assertTrue(ok)
        detail = await (await self.client.get(f"/api/runs/{self.full_a}/trials/{ok[0]}/phonemes")).json()
        self.assertEqual(detail["status"], "ok")
        self.assertIn("greedy", detail["words"][0])
        rows = await (await self.client.get("/api/runs")).json()
        self.assertIn("lm_fixed_pct", next(r for r in rows if r["run_id"] == self.full_a))

    async def test_commitment_endpoint(self):
        missing = await self.client.get(f"/api/runs/{self.full_b}/commitment")
        self.assertEqual(missing.status, 404)
        computed = await self.client.get(f"/api/runs/{self.full_b}/commitment?compute=1")
        self.assertEqual(computed.status, 200)
        policies = {p["policy"]: p for p in (await computed.json())["policies"]}
        self.assertIn("lag-2", policies)
        self.assertEqual(policies["none"]["commit_errors"]["early_commits"], 0)
        again = await self.client.get(f"/api/runs/{self.full_b}/commitment")
        self.assertEqual(again.status, 200)

    async def test_compare_same_scope_and_refusal(self):
        ok = await self.client.get(f"/api/compare?a={self.full_a}&b={self.full_b}")
        self.assertEqual(ok.status, 200)
        self.assertEqual((await ok.json())["delta_edits"], 0)
        refused = await self.client.get(f"/api/compare?a={self.full_a}&b={self.limited}")
        self.assertEqual(refused.status, 409)

    async def test_run_ids_are_not_paths(self):
        for bad in ("..%2F..%2Fetc", "nonexistent", "..", "%2Fetc%2Fpasswd"):
            response = await self.client.get(f"/api/runs/{bad}/trials")
            self.assertEqual(response.status, 404, bad)

    async def test_index_and_static(self):
        self.assertEqual((await self.client.get("/")).status, 200)
        self.assertEqual((await self.client.get("/static/js/app.js")).status, 200)
        registry_view = await (await self.client.get("/api/registry")).json()
        self.assertIn("tiny_fake", registry_view["pipelines"])


class ImportTests(unittest.TestCase):
    def test_server_does_not_import_torch(self):
        code = "import sys, harness.server; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=fx.REPO, check=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
