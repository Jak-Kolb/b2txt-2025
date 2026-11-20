"""Out-of-fold split by sentence (dataset.train_fold.by = sentence): stable, normalized, balanced."""
from pathlib import Path
import sys
import unittest

try:
    import h5py  # noqa: F401  (dataset dependency)
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("requires torch/h5py (acoustic environment)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from dataset import sentence_fold


class SentenceFoldTests(unittest.TestCase):
    def test_repeats_and_spelling_variants_share_a_fold(self):
        a = sentence_fold("I'm going home, now.", 5)
        self.assertEqual(a, sentence_fold(b"i'm  going HOME now", 5))
        self.assertIn(a, range(5))

    def test_folds_are_stable_and_roughly_balanced(self):
        word = lambda i: "".join(chr(97 + int(c)) for c in str(i))      # digits are normalized away; use letters
        sentences = [f"sentence {word(i)}" for i in range(5000)]
        folds = [sentence_fold(s, 5) for s in sentences]
        self.assertEqual(folds, [sentence_fold(s, 5) for s in sentences])
        counts = [folds.count(k) for k in range(5)]
        self.assertTrue(all(900 < c < 1100 for c in counts), counts)


if __name__ == "__main__":
    unittest.main()
