"""Server-side relay for the live player (no torch): engine subprocess <-> browser WebSocket.

Browser commands are validated here and forwarded to harness/live_engine.py. Engine events are
broadcast to every connected browser. Reference text is read here, from HDF5, and attached to
display events only; it is never sent to the engine.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import sys

from aiohttp import WSMsgType, web

from . import REPO, metrics
from .scopes import DATA_DIR, Trial, list_trials, partition_of, read_labels, trial_info
from .store import utc_stamp

SPLITS = ("train", "val", "test")
TRIAL_KEY = re.compile(r"^trial_\d{4}$")
PIPELINE_KEYS = {"preset", "acoustic", "lm", "decode", "preprocess"}
MAX_ITEMS = 500


class LiveRelay:
    def __init__(self, root, data_dir=None, lm_python=None, engine_command=None):
        self.root = Path(root)
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.lm_python = lm_python
        self.engine_command = engine_command
        self.process = None
        self.reader_task = None
        self.sockets = set()
        self.state = dict(type="state", state="stopped", message="Engine not started")
        self.references = {}
        self.finals = {}
        self.sessions = sorted(p.name for p in self.data_dir.iterdir() if p.is_dir()) if self.data_dir.exists() else []
        self.trial_cache = {}

    def install(self, app):
        app.router.add_get("/api/live", self.websocket)
        app.router.add_get("/api/live/status", self.status)
        app.router.add_get("/api/data/sessions", self.list_sessions)
        app.router.add_get("/api/data/trials", self.list_trials)
        app.on_shutdown.append(self.shutdown)

    # -- engine process ---------------------------------------------------------------------
    async def ensure_engine(self):
        if self.process is not None and self.process.returncode is None:
            return
        command = self.engine_command or [sys.executable, "-u", "-m", "harness.live_engine",
                                          "--root", str(self.root), "--data-dir", str(self.data_dir),
                                          "--lm-python", str(self.lm_python)]
        log_dir = self.root / "live"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stderr = (log_dir / f"engine_stderr_{utc_stamp()}.log").open("ab")
        self.process = await asyncio.create_subprocess_exec(
            *command, cwd=str(REPO), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr, limit=1 << 22)
        self.reader_task = asyncio.create_task(self.read_engine(self.process))

    async def read_engine(self, process):
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            await self.augment(event)
            if event.get("type") == "state":
                self.state = event
            await self.broadcast(event)
        await process.wait()
        self.state = dict(type="state", state="stopped",
                          message=f"Engine exited ({process.returncode}); see {self.stderr.name}")
        await self.broadcast(self.state)

    async def augment(self, event):
        kind = event.get("type")
        if kind == "trial_start":
            item = event["item"]
            trial = Trial(item["session"], item["split"], item["trial_key"],
                          str(self.data_dir / item["session"] / f"data_{item['split']}.hdf5"))
            reference, _ = await asyncio.to_thread(read_labels, trial)
            event["corpus"] = (await asyncio.to_thread(trial_info, trial))["corpus"]
            event["labeled"] = reference is not None
            if reference is not None:
                event["reference"] = reference
                self.references[event["play_id"]] = reference
        elif kind == "final":
            self.finals[event["play_id"]] = event["words"]
        elif kind == "trial_end":
            reference = self.references.pop(event["play_id"], None)
            words = self.finals.pop(event["play_id"], None)
            if reference is not None and words is not None and event.get("status") == "complete":
                edits, ref_words = metrics.word_errors(reference, " ".join(words))
                event["wer"] = dict(edits=edits, ref_words=ref_words,
                                    percent=100.0 * edits / ref_words if ref_words else None)

    async def broadcast(self, event):
        text = json.dumps(event)
        for ws in list(self.sockets):
            try:
                await ws.send_str(text)
            except (ConnectionResetError, RuntimeError):
                self.sockets.discard(ws)

    async def shutdown(self, _app=None):
        for ws in list(self.sockets):
            await ws.close()
        if self.process is not None and self.process.returncode is None:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=60)
            except asyncio.TimeoutError:
                self.process.terminate()
                await self.process.wait()

    # -- validation -------------------------------------------------------------------------
    def validate(self, command):
        op = command.get("op")
        if op in ("stop", "unload"):
            return dict(op=op)
        if op == "load":
            pipeline = command.get("pipeline")
            if not isinstance(pipeline, dict) or not set(pipeline) <= PIPELINE_KEYS:
                raise ValueError(f"pipeline must be an object with keys from {sorted(PIPELINE_KEYS)}")
            for key in ("decode", "preprocess"):
                if pipeline.get(key) is not None and not isinstance(pipeline[key], dict):
                    raise ValueError(f"{key} must be an object")
            return dict(op=op, pipeline=pipeline)
        if op == "play":
            items = command.get("items")
            if not isinstance(items, list) or not 0 < len(items) <= MAX_ITEMS:
                raise ValueError("play needs 1-500 items")
            clean = []
            for item in items:
                if (item.get("split") not in SPLITS or item.get("session") not in self.sessions
                        or not TRIAL_KEY.match(str(item.get("trial_key", "")))):
                    raise ValueError(f"Invalid play item {item!r}")
                clean.append(dict(split=item["split"], session=item["session"], trial_key=item["trial_key"]))
            speed = command.get("speed", 1)
            if speed not in (1, 2, 4, "max"):
                raise ValueError("speed must be 1, 2, 4, or 'max'")
            gap = float(command.get("gap_s", 1.5))
            if not 0 <= gap <= 10:
                raise ValueError("gap_s must be within [0, 10]")
            return dict(op=op, items=clean, speed=speed, gap_s=gap)
        raise ValueError(f"Unknown op {op!r}")

    # -- HTTP -------------------------------------------------------------------------------
    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.sockets.add(ws)
        await ws.send_str(json.dumps(self.state))
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    command = self.validate(json.loads(message.data))
                except (ValueError, TypeError, AttributeError) as exc:
                    await ws.send_str(json.dumps(dict(type="error", message=str(exc))))
                    continue
                if command["op"] == "stop" and (self.process is None or self.process.returncode is not None):
                    continue
                await self.ensure_engine()
                self.process.stdin.write((json.dumps(command) + "\n").encode())
                await self.process.stdin.drain()
        finally:
            self.sockets.discard(ws)
        return ws

    async def status(self, _request):
        return web.json_response(self.state)

    async def list_sessions(self, request):
        split = request.query.get("split", "val")
        if split not in SPLITS:
            raise web.HTTPBadRequest(text="Unknown split")
        key = ("sessions", split)
        if key not in self.trial_cache:
            trials = await asyncio.to_thread(list_trials, self.data_dir, split)
            counts = {}
            for trial in trials:
                counts[trial.session] = counts.get(trial.session, 0) + 1
            self.trial_cache[key] = [dict(session=s, n_trials=n, partition=partition_of(s, split))
                                     for s, n in sorted(counts.items())]
        return web.json_response(self.trial_cache[key])

    async def list_trials(self, request):
        split, session = request.query.get("split", "val"), request.query.get("session")
        if split not in SPLITS or session not in self.sessions:
            raise web.HTTPBadRequest(text="Unknown split or session")
        key = ("trials", split, session)
        if key not in self.trial_cache:
            def load():
                return [dict(trial_key=t.trial_key, **trial_info(t))
                        for t in list_trials(self.data_dir, split, sessions={session})]
            self.trial_cache[key] = await asyncio.to_thread(load)
        return web.json_response(self.trial_cache[key])
