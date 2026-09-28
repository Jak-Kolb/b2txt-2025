"""Word-commitment policies on synthetic partial sequences (online-only inputs, exact metrics)."""
import sys
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

from harness import commitment
from model_training.benchmark.output_trace import summarize_trial

# partial outputs over frames: "the" appears, "cat" flips to "hat" then back, "sat" appears late
PARTIALS = [[], ["the"], ["the", "cat"], ["the", "hat"], ["the", "cat"], ["the", "cat"],
            ["the", "cat", "sat"], ["the", "cat", "sat"]]
FINAL = ["the", "cat", "sat"]
FRAME_MS = lambda f: 100.0 * f


def run(policy, partials=PARTIALS, final=FINAL):
    return commitment.simulate(policy, partials, final)


class PolicyTests(unittest.TestCase):
    def test_none_commits_only_at_endpoint(self):
        output, frames, displayed = run(commitment.Never())
        self.assertEqual(output, FINAL)
        self.assertEqual(frames, [None, None, None])
        self.assertEqual(displayed[:-1], PARTIALS)

    def test_stable_k_waits_for_an_unchanged_prefix(self):
        output, frames, _ = run(commitment.StableK(2))
        self.assertEqual(output, FINAL)
        self.assertEqual(frames, [2, 5, 7])       # "cat" only after it held for two updates
        early, *_ = run(commitment.StableK(1))
        self.assertEqual(early, ["the", "cat", "sat"])
        _, flip_frames, _ = run(commitment.StableK(1), partials=[["the", "hat"], ["the", "cat"]], final=FINAL)
        self.assertEqual(flip_frames, [0, 0, None])
        wrong, _, _ = run(commitment.StableK(1), partials=[["the", "hat"], ["the", "cat"]], final=FINAL)
        self.assertEqual(wrong, ["the", "hat", "sat"])   # committed words are never retracted

    def test_lag_n_commits_words_behind_the_frontier(self):
        output, frames, _ = run(commitment.LagN(1))
        self.assertEqual(frames, [2, 6, None])
        self.assertEqual(output, FINAL)

    def test_policies_never_see_the_final_hypothesis(self):
        a = run(commitment.StableK(2), final=["x", "y", "z", "w"])
        b = run(commitment.StableK(2), final=FINAL)
        self.assertEqual(a[1][:3], b[1][:3])      # commit decisions identical whatever the final is

    def test_retrospective_reference(self):
        output, frames, _ = commitment.retrospective(PARTIALS, FINAL)
        self.assertEqual(output, FINAL)
        self.assertEqual(frames, [1, 4, 6])       # last time each prefix changed
        _, changed, _ = commitment.retrospective(PARTIALS, ["the", "cat", "sat", "down"])
        self.assertEqual(changed[3], None)        # only at the endpoint

    def test_metrics(self):
        output, frames, displayed = run(commitment.StableK(1), partials=[["the", "hat"], ["the", "cat"]], final=FINAL)
        result = commitment.evaluate_trial(output, frames, displayed, [["the", "hat"], ["the", "cat"]], FINAL,
                                           "The cat sat.", FRAME_MS, 300.0)
        self.assertEqual((result["edits"], result["ref_words"]), (1, 3))
        self.assertEqual(result["commit_errors"], 1)
        self.assertEqual(result["early"], 2)
        self.assertEqual(result["waits"], [0.0, 0.0, 0.0])
        self.assertEqual(result["before_end"], [300.0, 300.0, 0.0])

    def test_no_commitment_revisions_match_the_trace_summary(self):
        output, frames, displayed = run(commitment.Never())
        result = commitment.evaluate_trial(output, frames, displayed, PARTIALS, FINAL, "the cat sat", FRAME_MS, 800.0)
        events = [dict(kind="partial", words=p, output_ready_ns=i) for i, p in enumerate(PARTIALS)]
        events.append(dict(kind="final", words=FINAL, output_ready_ns=len(PARTIALS)))
        self.assertEqual(result["revisions"], summarize_trial(events)["revision_edits"])


if __name__ == "__main__":
    unittest.main()
