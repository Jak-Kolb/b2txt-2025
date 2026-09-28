"""Phoneme-vs-word error attribution (synthetic sequences; no GPU or native decoder)."""
import sys
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

import numpy as np

from harness import errors
from model_training.benchmark.stream_lm import levenshtein, norm_words

ID = {name: i for i, name in enumerate(errors.PHONES)}
LEXICON = {"you": ["Y", "UW"], "can": ["K", "AE", "N"], "tan": ["T", "AE", "N"], "see": ["S", "IY"], "a": ["AH"]}


def ids(text):
    return [ID[p] for p in text.split()]


def tokens(text):
    return [(ID[p], frame) for frame, p in enumerate(text.split())]


class AttributionTests(unittest.TestCase):
    def attribute(self, greedy, hyp, ref="You can see.", ref_phones="Y UW | K AE N | S IY |"):
        return errors.attribute_trial(ref, ids(ref_phones), tokens(greedy), hyp, LEXICON, lambda f: 100.0 * f)

    def categories(self, result):
        return [w["category"] for w in result["words"]]

    def test_phone_table_matches_the_model_code(self):
        from model_training.evaluate_model_helpers import LOGIT_TO_PHONEME
        self.assertEqual(errors.PHONES, [p.strip() or "|" for p in LOGIT_TO_PHONEME])

    def test_categories(self):
        exact = "Y UW | K AE N | S IY |"
        self.assertEqual(self.categories(self.attribute(exact, "you can see")), ["both_correct"] * 3)
        wrong_can = "Y UW | T AE N | S IY |"
        self.assertEqual(self.categories(self.attribute(wrong_can, "you can see")),
                         ["both_correct", "lm_fixed", "both_correct"])
        self.assertEqual(self.categories(self.attribute(wrong_can, "you tan see")),
                         ["both_correct", "both_wrong", "both_correct"])
        result = self.attribute(exact, "you tan see")
        self.assertEqual(self.categories(result), ["both_correct", "lm_introduced", "both_correct"])
        self.assertEqual(result["words"][1]["lm_phones"], ["T", "AE", "N"])

    def test_phoneme_edits_insertions_and_times(self):
        result = self.attribute("Y UW | K AE AE N S IY", "a you can see")
        can = result["words"][1]
        self.assertEqual(can["phone_edits"], 1)
        # which of the two identical AE tokens is "extra" is ambiguous; N must stay inside the word
        self.assertEqual(sorted(g["op"] for g in can["greedy"]), ["ins", "ok", "ok", "ok"])
        self.assertEqual([g["hyp"] for g in can["greedy"]], ["K", "AE", "AE", "N"])
        self.assertEqual(result["words"][2]["phone_edits"], 0)       # a missing boundary is not a phone error
        self.assertEqual(result["insertions"], [dict(word="a", after=-1, lm_phones=["AH"], category="lm_inserted")])
        self.assertEqual(result["words"][0]["t_ms"], 0.0)
        self.assertEqual(result["words"][1]["t_ms"], 300.0)
        deleted = self.attribute("Y UW | K N | S IY |", "you see")
        self.assertEqual(deleted["words"][1]["lm_op"], "del")
        self.assertEqual(deleted["words"][1]["category"], "both_wrong")

    def test_errors_reconcile_with_word_edit_distance(self):
        rng = np.random.default_rng(0)
        vocabulary = ["you", "can", "see", "tan", "a"]
        for _ in range(50):
            hyp = " ".join(rng.choice(vocabulary, size=rng.integers(0, 5)))
            result = self.attribute("Y UW | K AE N | S IY |", hyp)
            c = result["counts"]
            self.assertEqual(c["lm_introduced"] + c["both_wrong"] + c["lm_inserted"],
                             levenshtein(norm_words("You can see."), norm_words(hyp)))

    def test_mismatched_reference_groups_are_reported_not_guessed(self):
        result = self.attribute("Y UW", "you", ref="You can see.", ref_phones="Y UW K AE N")
        self.assertEqual(result["status"], "ref_word_phone_mismatch")
        self.assertEqual(result["words"], [])

    def test_greedy_tokens_collapse_and_reorder(self):
        # WFST order: 0 blank, 1 silence, k>=2 phones (acoustic index k-1)
        frames = [2, 2, 0, 2, 1, 1, 3]
        logits = np.full((len(frames), 41), -5.0, dtype=np.float32)
        logits[np.arange(len(frames)), frames] = 5.0
        self.assertEqual(errors.greedy_tokens(logits), [(1, 0), (1, 3), (40, 4), (2, 6)])

    def test_summary_shares(self):
        trial = dict(status="ok", counts=dict(both_correct=6, lm_fixed=3, lm_introduced=1, both_wrong=1, lm_inserted=0))
        summary = errors.summarize([trial, dict(status="ref_word_phone_mismatch", counts={})])
        self.assertAlmostEqual(summary["pct_acoustic_word_errors_fixed_by_lm"], 75.0)
        self.assertAlmostEqual(summary["pct_word_errors_with_exact_phonemes"], 50.0)
        self.assertEqual(summary["trials_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
