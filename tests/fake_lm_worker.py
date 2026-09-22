"""Synthetic LM process for protocol tests; never a scientific decoder."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_training.benchmark.lm_worker import serve


class Backend:
    n_classes = 3

    def __init__(self, delay, fail):
        self.delay, self.fail = delay, fail

    def reset(self):
        self.words = []

    def step(self, logits):
        if self.fail:
            raise ValueError("synthetic decoder failure")
        if len(logits) != self.n_classes:
            raise ValueError("wrong class count")
        time.sleep(self.delay)
        self.words.append(str(max(range(len(logits)), key=logits.__getitem__)))
        return list(self.words)

    def finish(self):
        return list(self.words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--shutdown_delay", type=float, default=0.0)
    args = parser.parse_args()
    status = serve(Backend(args.delay, args.fail), sys.stdin, sys.stdout)
    time.sleep(args.shutdown_delay)
    raise SystemExit(status)
