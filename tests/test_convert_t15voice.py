"""T15 voice-release -> T15 HDF5 converter on tiny synthetic sessions (no real data)."""
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy as np
    import scipy.io
except ImportError:
    raise unittest.SkipTest("requires h5py and scipy (acoustic environment)")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_prep import convert_t15voice as cv
from test_convert_t12 import fake_phonemizer


def write_session(folder, cues, lengths, rng):
    folder.mkdir(parents=True)
    n = len(cues)
    tx, sp = np.empty((1, n), dtype=object), np.empty((1, n), dtype=object)
    for i, t in enumerate(lengths):
        tx[0, i] = rng.normal(size=(t, 256)).astype(np.float32)
        sp[0, i] = rng.normal(2.0, 1.0, size=(t, 256)).astype(np.float32)
    scipy.io.savemat(folder / f"{folder.name}_neural_data.mat",
                     dict(cues=np.array([c.ljust(30) for c in cues]), threshcross_trials=tx, spikepow_trials=sp))
    return tx, sp


class ConvertT15VoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        (self.root / "b2t" / "t15.2023.08.13").mkdir(parents=True)          # day 27 is a B2T date
        (self.root / "desc.csv").write_text("Date,Post-implant day,Block number,Number of sentences,Corpus,Split\n"
                                            "2023-08-11,25,2,20,50-Word,Train\n2023-08-13,27,1,20,50-Word,Train\n")
        write_session(self.root / "voice" / "t15_day027", ["Skip me."], [40], rng)
        self.tx, self.sp = write_session(self.root / "voice" / "t15_day102",
                                         ["Hello there.", "Banned sentence!", "It\u2019s the last one."], [41, 30, 24], rng)
        (self.root / "exclude.txt").write_text("banned SENTENCE\n")

    def run_main(self, out, *extra):
        return cv.main(["--voice-dir", str(self.root / "voice"), "--b2t-dir", str(self.root / "b2t"),
                        "--b2t-csv", str(self.root / "desc.csv"), "--out-dir", str(out),
                        "--exclude-sentences", str(self.root / "exclude.txt"), *extra])

    def test_skips_b2t_dates_rebins_zscores_and_excludes(self):
        import unittest.mock
        with unittest.mock.patch.object(cv.ct, "repo_phonemizer", lambda: fake_phonemizer):
            summary = self.run_main(self.root / "out")
        self.assertEqual(summary["skipped_b2t_dates"], [dict(day=27, date="2023-08-13")])
        self.assertEqual((summary["n_written"], summary["n_excluded"]), (2, 1))
        path = self.root / "out" / "t15v.2023.10.27" / "data_train.hdf5"           # 2023-07-17 + 102 days
        with h5py.File(path, "r") as f:
            first = f["trial_0000"]
            x = first["input_features"][:]
            self.assertEqual(x.shape, (20, 512))                                   # 41 bins of 10 ms -> 20 of 20 ms
            self.assertEqual(first.attrs["session"], "t15v.2023.10.27")
            self.assertEqual(int(f["trial_0001"].attrs["source_index"]), 2)
            self.assertEqual(f["trial_0001"].attrs["sentence_label"], "It's the last one.")
            features = np.concatenate([f[k]["input_features"][:] for k in f], axis=0)
        raw = np.concatenate([self.tx[0, 0], self.sp[0, 0]], 1).astype(np.float64)[:40].reshape(20, 2, 512).mean(1)
        self.assertGreater(np.corrcoef(raw[:, 3], x[:, 3])[0, 1], 0.99)            # z-scoring is affine per feature
        self.assertLess(abs(features.mean()), 0.2)

    def test_output_loads_through_the_training_dataset(self):
        import unittest.mock
        sys.path.insert(0, str(REPO / "model_training"))
        from dataset import train_test_split_indicies
        with unittest.mock.patch.object(cv.ct, "repo_phonemizer", lambda: fake_phonemizer):
            self.run_main(self.root / "out")
        path = str(self.root / "out" / "t15v.2023.10.27" / "data_train.hdf5")
        train, val = train_test_split_indicies([path], test_percentage=0, seed=1)
        self.assertEqual(len(train[0]["trials"]), 2)

    def test_identity_harmonizer_changes_nothing(self):
        K = 2
        W = np.zeros((256, 2 * (2 * K + 1) + 1, 2))
        W[:, K, 0] = 1                       # tx at offset 0 -> tx
        W[:, 2 * K + 1 + K, 1] = 1           # sbp at offset 0 -> sbp
        v = np.random.default_rng(1).normal(size=(9, 512))
        np.testing.assert_allclose(cv.harmonize(v, W, K), v)


if __name__ == "__main__":
    unittest.main()
