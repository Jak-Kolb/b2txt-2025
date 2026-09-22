"""Local JSON-lines bridge to the Python 3.9 native WFST decoder.

The protocol carries logits and output events only, never reference transcripts.
One request is outstanding at a time; both processes use Linux monotonic_ns.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import select
import subprocess
import sys
import time


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576


def serve(backend, reader, writer, clock=time.monotonic_ns):
    """Run one decoder per worker; a malformed request ends the session."""
    def send(obj):
        writer.write(json.dumps(obj, allow_nan=False) + "\n")
        writer.flush()
    send(dict(type="ready", protocol=PROTOCOL_VERSION, clock="monotonic_ns",
              n_classes=backend.n_classes, python=sys.version.split()[0]))
    expected_id = 0
    active = False
    frame_index = 0
    for line in reader:
        request_id = None
        try:
            if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
                raise ValueError("Oversized worker request")
            request = json.loads(line)
            received = clock()
            request_id = request.get("id")
            if type(request_id) is not int or request_id != expected_id:
                raise ValueError("Requests must have consecutive integer IDs")
            expected_id += 1
            op = request.get("op")
            start = clock()
            words = None
            if op == "reset":
                if active:
                    raise ValueError("Cannot reset before finalization")
                backend.reset()
                frame_index = 0
                active = True
            elif op == "frame":
                if not active or request.get("frame_index") != frame_index:
                    raise ValueError("Unexpected frame or missing reset")
                words = backend.step(request["logits"])
                frame_index += 1
            elif op == "finish":
                if not active:
                    raise ValueError("No active trial to finish")
                words = backend.finish()
                active = False
            elif op == "close":
                if active:
                    raise ValueError("Cannot close an unfinished trial")
            else:
                raise ValueError(f"Unknown operation: {op}")
            ready = clock()
            response = dict(type="response", id=request_id, op=op, worker_received_ns=received,
                            processing_start_ns=start, worker_output_ready_ns=ready,
                            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if words is not None:
                response["words"] = words
            send(response)
            if op == "close":
                return 0
        except Exception as exc:
            send(dict(type="error", id=request_id, error=f"{type(exc).__name__}: {exc}"))
            return 1
    return 1  # EOF without a close acknowledgement is an incomplete session.


class NativeBackend:
    def __init__(self, lm_dir, config, n_classes=41):
        import lm_decoder
        import numpy as np
        self.module = lm_decoder
        self.np = np
        self.n_classes = n_classes
        opts = lm_decoder.DecodeOptions(
            config["max_active"], config["min_active"], config["beam"], config["lattice_beam"],
            config["acoustic_scale"], config["blank_skip_thresh"], config["length_penalty"], 1)
        res = lm_decoder.DecodeResource(str(Path(lm_dir) / "TLG.fst"), "", "",
                                        str(Path(lm_dir) / "words.txt"), "")
        self.decoder = lm_decoder.BrainSpeechDecoder(res, opts)
        self.log_blank = float(np.log(config["blank_penalty"]))

    def reset(self):
        self.decoder.Reset()

    def _words(self):
        result = self.decoder.result()
        return result[0].sentence.strip().split() if result else []

    def step(self, logits):
        row = self.np.asarray(logits, dtype=self.np.float32)
        if row.shape != (self.n_classes,) or not self.np.isfinite(row).all():
            raise ValueError("Worker needs one finite, reordered logit vector")
        self.module.DecodeNumpy(self.decoder, row.reshape(1, -1),
                                self.np.zeros((1, self.n_classes), dtype=self.np.float32), self.log_blank)
        return self._words()

    def finish(self):
        self.decoder.FinishDecoding()
        return self._words()


class LMClient:
    """Bounded synchronous IPC. Native logs go to a separate, fresh file."""

    def __init__(self, command, log_path, timeout_seconds=30.0, startup_seconds=120.0,
                 shutdown_seconds=60.0, clock=time.monotonic_ns):
        self.command = list(command)
        self.log_path = Path(log_path)
        self.timeout_seconds = timeout_seconds
        self.startup_seconds = startup_seconds
        self.shutdown_seconds = shutdown_seconds
        self.clock = clock
        self.process = None
        self.log = None
        self.buffer = b""
        self.next_id = 0
        self.peak_rss_kib = 0

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("xb")
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, bufsize=0)
            self.ready = self._read(self.startup_seconds)
            if (self.ready.get("type") != "ready" or self.ready.get("protocol") != PROTOCOL_VERSION
                    or self.ready.get("clock") != "monotonic_ns"):
                raise RuntimeError("Invalid LM worker handshake")
            return self
        except Exception:
            self._stop()
            raise

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"LM worker timed out; inspect {self.log_path}")
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                raise TimeoutError(f"LM worker timed out; inspect {self.log_path}")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError(f"LM worker closed its output; inspect {self.log_path}")
            self.buffer += block
            if len(self.buffer) > MAX_MESSAGE_BYTES:
                raise ValueError("Oversized LM response")
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("LM response must be an object")
        return response

    def request(self, op, **payload):
        request_id = self.next_id
        self.next_id += 1
        message = json.dumps(dict(id=request_id, op=op, **payload), allow_nan=False).encode() + b"\n"
        if len(message) > MAX_MESSAGE_BYTES:
            raise ValueError("Oversized LM request")
        sent = self.clock()
        self.process.stdin.write(message)
        response = self._read(self.timeout_seconds)
        received = self.clock()
        if response.get("type") == "error":
            raise RuntimeError(f"LM worker rejected request: {response.get('error')}")
        if response.get("id") != request_id or response.get("op") != op:
            raise ValueError("LM response identity mismatch")
        times = [sent, response["worker_received_ns"], response["processing_start_ns"],
                 response["worker_output_ready_ns"], received]
        if any(type(t) is not int for t in times) or times != sorted(times):
            raise RuntimeError("LM/coordinator clocks are inconsistent")
        response["output_ready_ns"] = received
        self.peak_rss_kib = max(self.peak_rss_kib, response["peak_rss_kib"])
        return response

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        if self.log is not None:
            self.log.close()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.request("close")
                if self.process.wait(timeout=self.shutdown_seconds) != 0:
                    raise RuntimeError(f"LM worker exited unsuccessfully; inspect {self.log_path}")
        finally:
            self._stop()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm", required=True)
    parser.add_argument("--config", required=True, help="JSON object of decoder settings")
    parser.add_argument("--n_classes", type=int, default=41)
    args = parser.parse_args()
    # Keep C++/library stdout separate from the protocol, including during FST loading.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        backend = NativeBackend(args.lm, json.loads(args.config), args.n_classes)
        return serve(backend, sys.stdin, protocol)
    finally:
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
