"""Jobs page data: queue-script status reporting, older queue logs, train jobs, log tails, GPU lock (no torch)."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

from aiohttp.test_utils import AioHTTPTestCase

import harness_fixture as fx
from harness import jobs
from harness.server import create_app

SCRIPT = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    from harness import jobs
    folder = Path({folder!r}); folder.mkdir(parents=True)
    log = jobs.queue_logger(folder)
    log("waiting for PID 999999 to exit")
    log("PID 999999 exited")
    log("stage lists: start")
    log("built 20990101T000000Z_tiny_standard")
    log("stage lists: done")
    log("stage llm: start")
    log("stage llm: FAILED (['x'] failed (logs in /tmp/q)); going on")
    log("stage rank: start")
    if {crash}:
        raise RuntimeError("boom")
    log("stage rank: done")
    log("done")
""")


def stamp(offset=0):
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() + offset))


class JobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results" / "harness"
        self.queue = self.root / "queue"
        self.queue.mkdir(parents=True)
        (self.root / "runs" / "20990101T000000Z_tiny_standard").mkdir(parents=True)

    def run_script(self, name, crash=False):
        folder = self.queue / f"{stamp()}_{name}"
        script = self.queue / f"{name}_20990101.py"
        script.write_text(SCRIPT.format(repo=str(fx.REPO), folder=str(folder), crash=crash))
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        return folder, out

    def test_reporting_writes_status_and_events(self):
        folder, out = self.run_script("good")
        self.assertEqual(out.returncode, 0, out.stderr)
        status = json.loads((folder / "job.json").read_text())
        self.assertEqual(status["state"], "done")
        self.assertEqual([(s["name"], s["state"]) for s in status["stages"]],
                         [("lists", "done"), ("llm", "failed"), ("rank", "done")])
        self.assertEqual(status["runs"], ["20990101T000000Z_tiny_standard"])
        self.assertEqual(status["after_pid"], 999999)
        self.assertTrue(status["ended"])
        events = [json.loads(l) for l in (self.queue / "jobs.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[-1]["event"], "end")
        self.assertIn("stage_failed", [e["event"] for e in events])
        job = next(j for j in jobs.list_jobs(self.root)["jobs"] if j["id"] == folder.name)
        self.assertEqual(job["state"], "done")
        self.assertEqual(job["failed_stages"], 1)
        self.assertEqual(job["links"]["runs"], ["20990101T000000Z_tiny_standard"])
        self.assertNotIn("_logs", job)

    def test_crash_marks_failed(self):
        folder, out = self.run_script("bad", crash=True)
        self.assertNotEqual(out.returncode, 0)
        status = json.loads((folder / "job.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["exit"]["error"], "RuntimeError")
        self.assertEqual(status["stages"][-1]["state"], "failed")

    def test_dead_running_status_is_interrupted(self):
        folder = self.queue / f"{stamp()}_gone"
        folder.mkdir()
        (folder / "job.json").write_text(json.dumps(dict(
            id=folder.name, name="gone", script="results/harness/queue/gone_20990101.py", pid=999999,
            state="running", stage="x", stages=[dict(name="x", state="running", started=None, ended=None)],
            started="2099-01-01T00:00:00+00:00", runs=[])))
        job = jobs.get_job(folder.name, self.root)
        self.assertEqual(job["state"], "interrupted")
        self.assertIsNone(job["stage"])

    def test_old_queue_logs_are_parsed(self):
        done = self.queue / "20260101T000000Z_old"
        done.mkdir()
        (done / "queue.log").write_text("2026-01-01T00:00:00Z stage a: start\n2026-01-01T00:10:00Z stage a: done\n"
                                        "2026-01-01T00:10:01Z stage b: start\n2026-01-01T00:20:00Z stage b: FAILED (x)\n"
                                        "2026-01-01T00:20:00Z done\n")
        (self.queue / "old_20260101.log").write_text("stdout line\n")
        rerun = self.queue / "20260103T000000Z_rerun"
        rerun.mkdir()
        (rerun / "queue.log").write_text("2026-01-03T00:00:00Z stage a: start\n2026-01-03T00:01:00Z stage a: start\n"
                                         "2026-01-03T00:02:00Z stage a: done\n")
        cut = self.queue / "20990102T000000Z_cut"
        cut.mkdir()
        (cut / "queue.log").write_text("2099-01-02T00:00:00Z stage a: start\n")
        found = {j["id"]: j for j in jobs.list_jobs(self.root)["jobs"]}
        self.assertEqual(found[done.name]["state"], "done")
        self.assertEqual(found[done.name]["failed_stages"], 1)
        self.assertEqual(found[done.name]["elapsed_s"], 1200)
        self.assertEqual(found[done.name]["logs"], ["queue", "stdout"])
        self.assertEqual(found[cut.name]["state"], "interrupted")
        self.assertEqual(found[rerun.name]["state"], "done")
        self.assertEqual([s["state"] for s in found[rerun.name]["stages"]], ["unfinished", "done"])
        tail = jobs.job_log_tail(done.name, 2, root=self.root)
        self.assertEqual(len(tail["lines"]), 2)
        self.assertEqual(tail["file"], f"queue/{done.name}/queue.log")
        self.assertEqual(jobs.job_log_tail(done.name, 5, "stdout", self.root)["lines"], ["stdout line"])

    def test_live_old_script_is_running_and_waiting_detected(self):
        script = self.queue / "live_20990101.py"
        script.write_text("import time\ntime.sleep(30)\n")
        proc = subprocess.Popen([sys.executable, str(script)])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        folder = self.queue / f"{stamp()}_live"
        folder.mkdir()
        (folder / "queue.log").write_text(f"2099-01-01T00:00:00Z waiting for PID {proc.pid} to exit\n")
        other = self.queue / "orphan_20990101.py"
        other.write_text("import time\ntime.sleep(30)\n")
        orphan = subprocess.Popen([sys.executable, str(other), "--after", str(proc.pid)])
        self.addCleanup(orphan.wait)
        self.addCleanup(orphan.kill)
        time.sleep(0.3)
        found = {j["id"]: j for j in jobs.list_jobs(self.root)["jobs"]}
        self.assertEqual(found[folder.name]["state"], "waiting")
        self.assertEqual(found[folder.name]["pid"], proc.pid)
        self.assertEqual(found[f"pid-{orphan.pid}"]["state"], "waiting")
        self.assertEqual(found[f"pid-{orphan.pid}"]["waiting_on"]["job"], folder.name)
        (folder / "queue.log").write_text(f"2099-01-01T00:00:00Z PID 1 exited\n2099-01-01T00:00:01Z stage s1: start\n")
        job = jobs.get_job(folder.name, self.root)
        self.assertEqual((job["state"], job["stage"]), ("running", "s1"))

    def test_training_queue_status_files(self):
        (self.root / "train" / "20990101T000000Z_m1").mkdir(parents=True)
        folder = self.queue / "20260101T000000Z_tq"
        folder.mkdir()
        (folder / "status.json").write_text(json.dumps(dict(started="2026-01-01T00:00:00Z", protocol="p", jobs=dict(
            m1=dict(state="done", updated="2026-01-01T03:00:00Z", train_dir=str(self.root / "train" / "20990101T000000Z_m1"),
                    runs={"0.4": "20990101T000000Z_tiny_standard"}),
            m2=dict(state="gave up after retries", updated="2026-01-01T04:00:00Z"),
            m3=dict(state="training (attempt 1)", updated="2026-01-01T05:00:00Z")))))
        (folder / "commands.log").write_text("$ train m1\n")
        job = jobs.get_job(folder.name, self.root)
        self.assertEqual(job["state"], "interrupted")      # no "finished" and no live process
        self.assertEqual([(s["name"], s["state"]) for s in job["stages"]],
                         [("m1", "done"), ("m2", "failed"), ("m3", "unfinished")])
        self.assertEqual(job["links"]["runs"], ["20990101T000000Z_tiny_standard"])
        self.assertEqual(job["links"]["train"], ["20990101T000000Z_m1"])
        self.assertEqual(job["failed_stages"], 1)
        self.assertEqual(jobs.job_log_tail(folder.name, root=self.root)["log"], "commands")
        status = json.loads((folder / "status.json").read_text())
        status["finished"] = "2026-01-01T06:00:00Z"
        (folder / "status.json").write_text(json.dumps(status))
        job = jobs.get_job(folder.name, self.root)
        self.assertEqual((job["state"], job["ended"], job["elapsed_s"]), ("done", "2026-01-01T06:00:00+00:00", 6 * 3600))

    def test_train_jobs_are_merged(self):
        train = self.root / "train" / "20990101T000000Z_wide"
        train.mkdir(parents=True)
        (train / "request.json").write_text(json.dumps(dict(name="wide", forked_from="tiny", overrides=["a=1"],
                                                            question="Q?", model_dir=str(train / "m"),
                                                            created="2099-01-01T00:00:00+00:00")))
        (train / "state.json").write_text(json.dumps(dict(state="benchmarked", benchmark_run="20990101T000000Z_tiny_standard",
                                                          updated="2099-01-01T05:00:00+00:00")))
        (train / "train.log").write_text("t: Train batch 0: loss: 2.5\n")
        job = jobs.get_job("train-20990101T000000Z_wide", self.root)
        self.assertEqual((job["kind"], job["state"], job["stage"]), ("train", "done", "benchmarked"))
        self.assertEqual(job["links"], dict(runs=["20990101T000000Z_tiny_standard"], jobs=[], train=[train.name]))
        self.assertEqual(job["elapsed_s"], 5 * 3600)
        self.assertEqual(jobs.job_log_tail(job["id"], root=self.root)["log"], "train")

    def test_log_tail_hides_sentence_lines_and_stays_inside_root(self):
        folder = self.queue / "20990101T000000Z_priv"
        folder.mkdir()
        (folder / "queue.log").write_text('2099-01-01T00:00:00Z stage a: start\n'
                                          '2099-01-01T00:00:01Z {"sentence": "i would like some water please"}\n'
                                          '2099-01-01T00:00:02Z ref: the cat sat on the mat\n'
                                          '2099-01-01T00:00:03Z WER 5.80 (words: 1234)\n')
        tail = jobs.job_log_tail(folder.name, 10, root=self.root)
        self.assertEqual(tail["hidden_lines"], 2)
        self.assertFalse(any("water" in l or "cat" in l for l in tail["lines"]))
        self.assertIn("WER 5.80", tail["lines"][-1])
        outside = Path(self.temp.name) / "outside.log"
        outside.write_text("secret\n")
        status_job = self.queue / f"{stamp()}_leak"
        status_job.mkdir()
        (status_job / "job.json").write_text(json.dumps(dict(id=status_job.name, name="leak", state="done",
                                                                stages=[], stdout=str(outside), runs=[])))
        self.assertEqual(jobs.get_job(status_job.name, self.root)["logs"], [])
        with self.assertRaises(KeyError):
            jobs.job_log_tail("../../outside", root=self.root)
        with self.assertRaises(FileNotFoundError):
            jobs.job_log_tail(folder.name, 5, "stdout", self.root)

    def test_lock_holders_and_gpu_lock_waits(self):
        path = self.root / "locks" / "job.lock"
        path.parent.mkdir(parents=True)
        holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
            import fcntl, sys, time
            h = open({str(path)!r}, "a"); fcntl.flock(h, fcntl.LOCK_EX); print("held", flush=True); time.sleep(30)""")],
            stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.stdout.close)
        self.addCleanup(holder.wait)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        self.assertEqual(jobs.lock_holders(path)[0], [holder.pid])
        waited, got = [], threading.Event()

        def take():
            with jobs.gpu_lock(self.root, on_wait=waited.append):
                got.set()
        thread = threading.Thread(target=take)
        thread.start()
        time.sleep(0.3)
        self.assertFalse(got.is_set())
        self.assertEqual(waited, [[holder.pid]])
        gpu = jobs.gpu_status(self.root, max_age_s=0)
        self.assertTrue(gpu["locks"]["job"]["held"])
        self.assertEqual(gpu["locks"]["job"]["holders"][0]["pid"], holder.pid)
        self.assertEqual(len(gpu["locks"]["job"]["waiters"]), 1)
        holder.kill()
        thread.join(5)
        self.assertTrue(got.is_set())


class JobsServerTests(AioHTTPTestCase):
    async def get_application(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results" / "harness"
        folder = self.root / "queue" / "20990101T000000Z_srv"
        folder.mkdir(parents=True)
        (folder / "queue.log").write_text("".join(f"2099-01-01T00:00:{i:02d}Z line {i}\n" for i in range(30)))
        return create_app(self.root, registry_root=Path(self.temp.name), data_dir=Path(self.temp.name))

    async def test_jobs_log_and_gpu_endpoints(self):
        data = await (await self.client.get("/api/jobs?limit=5")).json()
        self.assertIn("20990101T000000Z_srv", [j["id"] for j in data["jobs"]])
        self.assertIn("checked", data)
        one = await (await self.client.get("/api/jobs/20990101T000000Z_srv")).json()
        self.assertEqual(one["name"], "srv")
        tail = await (await self.client.get("/api/jobs/20990101T000000Z_srv/log?tail=3")).json()
        self.assertEqual(len(tail["lines"]), 3)
        self.assertTrue(tail["lines"][-1].endswith("line 29"))
        self.assertEqual((await self.client.get("/api/jobs/nope/log")).status, 404)
        self.assertEqual((await self.client.get("/api/jobs/..%2F..%2Fetc/log")).status, 404)
        self.assertEqual((await self.client.get("/api/jobs?limit=x")).status, 400)
        gpu = await (await self.client.get("/api/gpu")).json()
        self.assertIn("locks", gpu)
        self.assertIn("gpus", gpu)
        page = await self.client.get("/static/js/views/jobs.js")
        self.assertEqual(page.status, 200)


class ImportTests(unittest.TestCase):
    def test_jobs_does_not_import_torch(self):
        code = "import sys, harness.jobs; harness.jobs.list_jobs(); print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=fx.REPO, check=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
