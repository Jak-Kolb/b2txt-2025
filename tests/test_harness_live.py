"""Live engine and relay tests: tiny CPU GRU, real lm_worker with the fake native decoder."""
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

import numpy as np
from aiohttp.test_utils import AioHTTPTestCase

import harness_fixture as fx
from harness import live_engine
from harness.live import LiveRelay
from harness.server import create_app
from model_training.benchmark.output_trace import summarize_trace


class Collect:
    """Stream stand-in for the engine's protocol output."""

    def __init__(self):
        self.events = []
        self.cond = threading.Condition()

    def write(self, text):
        with self.cond:
            self.events.extend(json.loads(line) for line in text.splitlines() if line)
            self.cond.notify_all()

    def flush(self):
        pass

    def wait_for(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        with self.cond:
            while not any(predicate(e) for e in self.events):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No matching event; last: {self.events[-3:]}")
                self.cond.wait(remaining)
        return next(e for e in self.events if predicate(e))


class CountTests(unittest.TestCase):
    def test_counts_recovered_from_block_zscore(self):
        rng = np.random.default_rng(0)
        counts = rng.poisson(1.2, size=(300, 5)).astype(np.float32)
        counts[:, 3] = 0                      # silent electrode
        z = (counts - counts.mean(0)) / np.where(counts.std(0) > 0, counts.std(0), 1)
        z[10, 1] = 10.0                       # clip rail
        z[:, 4] = rng.normal(size=300)        # not a lattice
        recovered, fallback = live_engine.tx_counts(z)
        base = counts - counts.min(0)
        np.testing.assert_array_equal(recovered[:, 0], base[:, 0])
        self.assertEqual(recovered[10, 1], live_engine.SATURATED)
        mask = np.arange(300) != 10
        np.testing.assert_array_equal(recovered[mask, 1], (counts[mask, 1] - counts[mask, 1].min()))
        self.assertTrue((recovered[:, 3] == 0).all())
        self.assertEqual(fallback, [4])

    def test_sink_drops_only_bins_when_full(self):
        gate = threading.Event()

        class Blocked(Collect):
            def write(self, text):
                gate.wait()
                super().write(text)
        stream = Blocked()
        sink = live_engine.Sink(stream, maxsize=2)
        for i in range(6):
            sink.emit(dict(type="bin", bin=i), droppable=True)
        self.assertGreater(sink.dropped, 0)
        gate.set()
        sink.emit(dict(type="final"))
        sink.close()
        self.assertEqual(stream.events[-1]["type"], "final")

    def test_engine_never_touches_reference_text(self):
        source = Path(live_engine.__file__).read_text()
        for name in ("transcription", "sentence_label", "read_labels", "seq_class_ids"):
            self.assertNotIn(name, source)


class EngineTests(unittest.TestCase):
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

    def engine(self):
        stream = Collect()
        sink = live_engine.Sink(stream)
        engine = live_engine.Engine(self.results, self.data, sys.executable, "cpu", sink, self.registry)
        self.addCleanup(lambda: (engine.shutdown(), sink.close()))
        return engine, stream

    def test_load_play_stop_and_replay(self):
        engine, stream = self.engine()
        engine.handle(dict(op="load", pipeline=dict(preset="tiny_fake")))
        stream.wait_for(lambda e: e["type"] == "state" and e["state"] == "ready")
        item = dict(split="val", session="t15.2099.01.01", trial_key="trial_0002")
        engine.handle(dict(op="play", items=[item], speed=1))
        end = stream.wait_for(lambda e: e["type"] == "trial_end")
        start = next(e for e in stream.events if e["type"] == "trial_start")
        self.assertEqual(end["status"], "complete")
        self.assertTrue(start["timing_valid"])
        self.assertNotIn("reference", start)
        bins = [e for e in stream.events if e["type"] == "bin" and e["play_id"] == start["play_id"]]
        frames = [e for e in stream.events if e["type"] == "frame" and e["play_id"] == start["play_id"]]
        self.assertEqual(len(bins) + end["dropped_bins"], start["n_bins"])
        self.assertEqual(len(frames), start["n_frames"])
        # Synthetic features are narrower than the 512-feature release: TX takes the first columns.
        self.assertEqual(len(base64.b64decode(bins[0]["tx"])), fx.N_FEATURES)
        trace = engine.live_dir / start["trace"]
        self.assertEqual(summarize_trace(trace)["trials"][0], end["summary"])
        final = next(e for e in stream.events if e["type"] == "final")
        self.assertEqual(end["summary"]["n_final_words"], len(final["words"]))

        # Stop during a multi-trial playlist, then play again on the same LM worker.
        stream.wait_for(lambda e: e["type"] == "state" and e["state"] == "ready" and e["message"] == "Playback finished")
        long_play = [dict(item, trial_key=f"trial_000{i}") for i in range(4)]
        engine.handle(dict(op="play", items=long_play, speed=1, gap_s=0))
        stream.wait_for(lambda e: e["type"] == "bin" and e["play_id"] == start["play_id"] + 1 and e["bin"] >= 3)
        engine.handle(dict(op="stop"))
        stopped = stream.wait_for(lambda e: e["type"] == "trial_end" and e["status"] == "stopped")
        stream.wait_for(lambda e: e["type"] == "state" and e.get("message") == "Stopped")
        stopped_trace = next(engine.live_dir.glob(f"{stopped['play_id']:04d}_*.jsonl"))
        with self.assertRaisesRegex(ValueError, "Incomplete trace"):
            summarize_trace(stopped_trace)
        engine.handle(dict(op="play", items=[item], speed="max"))
        again = stream.wait_for(lambda e: e["type"] == "trial_end" and e["play_id"] > stopped["play_id"])
        self.assertEqual(again["status"], "complete")
        replay_start = next(e for e in stream.events if e["type"] == "trial_start" and e["play_id"] == again["play_id"])
        self.assertFalse(replay_start["timing_valid"])

    def test_noncausal_pipeline_is_refused(self):
        engine, stream = self.engine()
        engine.handle(dict(op="load", pipeline=dict(preset="sym_fake")))
        error = stream.wait_for(lambda e: e["type"] == "error")
        self.assertIn("offline_noncausal", error["message"])


class RelayTests(AioHTTPTestCase):
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

    async def get_application(self):
        command = [sys.executable, "-u", "-m", "harness.live_engine", "--root", str(self.results),
                   "--data-dir", str(self.data), "--lm-python", sys.executable, "--device", "cpu",
                   "--registry", str(self.registry)]
        self.relay = LiveRelay(self.results, self.data, sys.executable, engine_command=command)
        return create_app(self.results, registry_root=self.registry, live=self.relay)

    async def receive_until(self, ws, predicate, timeout=60):
        events = []
        async def loop():
            while True:
                event = json.loads((await ws.receive()).data)
                events.append(event)
                if predicate(event):
                    return event
        return await asyncio.wait_for(loop(), timeout), events

    async def test_relay_adds_reference_and_wer_only_on_the_server_side(self):
        sessions = await (await self.client.get("/api/data/sessions?split=val")).json()
        self.assertEqual([s["partition"] for s in sessions], ["former-val-test", "val-dev", "val-dev"])
        trials = await (await self.client.get("/api/data/trials?split=val&session=t15.2099.01.02")).json()
        self.assertEqual(trials[0]["trial_key"], "trial_0000")
        self.assertEqual(trials[0]["corpus"], "Random")
        self.assertNotIn("reference", json.dumps(trials))
        ws = await self.client.ws_connect("/api/live")
        await ws.send_str(json.dumps(dict(op="play", items=[dict(split="val", session="../etc", trial_key="trial_0000")])))
        error, _ = await self.receive_until(ws, lambda e: e["type"] == "error")
        self.assertIn("Invalid play item", error["message"])
        await ws.send_str(json.dumps(dict(op="load", pipeline=dict(preset="tiny_fake"))))
        await self.receive_until(ws, lambda e: e["type"] == "state" and e["state"] == "ready")
        await ws.send_str(json.dumps(dict(op="play", items=[dict(split="val", session="t15.2099.01.02",
                                                                 trial_key="trial_0001")], speed="max")))
        end, events = await self.receive_until(ws, lambda e: e["type"] == "trial_end")
        start = next(e for e in events if e["type"] == "trial_start")
        self.assertEqual(start["reference"], "w4")
        self.assertEqual(start["corpus"], "Random")
        self.assertIn("wer", end)
        await ws.send_str(json.dumps(dict(op="play", items=[dict(split="test", session="t15.2099.01.02",
                                                                 trial_key="trial_0000")], speed="max")))
        end, events = await self.receive_until(ws, lambda e: e["type"] == "trial_end")
        start = next(e for e in events if e["type"] == "trial_start")
        self.assertNotIn("reference", start)
        self.assertFalse(start["labeled"])
        self.assertNotIn("wer", end)
        await ws.close()
        await self.relay.shutdown()


if __name__ == "__main__":
    unittest.main()
