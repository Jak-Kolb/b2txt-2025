"""Live player speech (harness/web/js/speech.js): words are spoken when the clock-A sil+D rule commits them.

Random greedy class sequences and partial word lists are replayed through both the browser's WordSpeaker (node)
and learning/live_eval/clock_a.py (word_boundaries + AtTimes + simulate); the committed words and the frames they
are committed at must match, with the final output's remaining words spoken at the end."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if sys.version_info < (3, 10):
    raise unittest.SkipTest("requires the Python 3.10 acoustic stack")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
sys.path.insert(0, str(REPO / "learning/live_eval"))


def node_speaker(frames, final, delay):
    # node reads .js as CommonJS outside an ES-module package; the browser loads it as a module, so import a .mjs copy
    module = Path(tempfile.mkdtemp()) / "speech.mjs"
    shutil.copy(REPO / "harness/web/js/speech.js", module)
    script = f"""
import {{ WordSpeaker }} from {json.dumps(module.as_uri())};
const out = []; let current = null;
const sp = new WordSpeaker((text) => out.push([current, text]), {json.dumps(delay)});
for (const f of {json.dumps(frames)}) {{ current = f.frame; sp.frame(f); }}
current = "final"; sp.final({json.dumps(final)});
console.log(JSON.stringify(out));
"""
    try:
        done = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    finally:
        shutil.rmtree(module.parent)
    return json.loads(done.stdout)


@unittest.skipIf(NODE is None, "node is not installed")
class LiveSpeechTests(unittest.TestCase):
    def test_matches_clock_a_sil_rule(self):
        import clock_a as ca
        rng = np.random.default_rng(0)
        vocab = [f"w{i}" for i in range(30)]
        for trial in range(40):
            n = int(rng.integers(20, 80))
            # greedy classes: mostly blank, some phonemes, some word boundaries (class SIL)
            arg = rng.choice([0, 0, 0, 3, 7, 12, ca.SIL], size=n)
            lp = np.log(np.full((n, ca.SIL + 1), 1e-4, dtype=np.float32))
            lp[np.arange(n), arg] = 0.0
            final = list(rng.choice(vocab, size=int(rng.integers(1, 8))))
            partials, words = [], []
            for f in range(n):                                   # partials grow and get revised now and then
                if rng.random() < 0.15 and len(words) < len(final) + 2:
                    words = words + [str(rng.choice(vocab))]
                if words and rng.random() < 0.05:
                    words = words[:-1] + [final[min(len(words) - 1, len(final) - 1)]]
                partials.append(list(words))
            frames = [dict(frame=f, phone=None if c == 0 else ("|" if c == ca.SIL else f"P{c}"), words=partials[f])
                      for f, c in enumerate(arg)]
            ends = [e for e, _ in ca.word_boundaries(lp)]
            for delay in (1, 13):
                output, commit_frames = ca.simulate(ca.AtTimes("sil", [e + delay for e in ends]), partials, final)
                expected = {}
                for word, f in zip(output, commit_frames):
                    expected.setdefault("final" if f is None else f, []).append(word)
                got = node_speaker(frames, final, delay)
                self.assertEqual([[f, " ".join(ws)] for f, ws in expected.items()], got, msg=f"trial {trial} D={delay}")

    def test_final_mode_speaks_only_the_final_sentence(self):
        frames = [dict(frame=0, phone="P1", words=["a"]), dict(frame=1, phone="|", words=["a"]),
                  dict(frame=5, phone=None, words=["a", "b"])]
        self.assertEqual(node_speaker(frames, ["a", "c"], "final"), [["final", "a c"]])


if __name__ == "__main__":
    unittest.main()
