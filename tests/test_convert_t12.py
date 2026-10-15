"""T12 competitionData -> T15 HDF5 converter on tiny synthetic .mat files (no real data)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

try:
    import torch
    import h5py
    import numpy as np
    import scipy.io
except ImportError:
    raise unittest.SkipTest("requires torch, h5py and scipy (acoustic environment)")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model_training"))
from data_prep import convert_t12 as ct
from dataset import BrainToTextDataset, train_test_split_indicies


def fake_phonemizer(sentence):
    """Deterministic stand-in for g2p: 'AA B' per word, SIL after each word (empty -> ['SIL'])."""
    words = ct.normalize_sentence(sentence).split()
    if not words:
        return ["SIL"]
    phones = []
    for word in words:
        phones += ["AA"] * 250 if word == "verylongword" else ["AA", "B"]
        phones.append("SIL")
    return phones


def write_mat(path, sentences, blocks, lengths, rng, outlier=False, dead=None):
    n = len(sentences)
    fields = {f"tx{k}": np.empty((1, n), dtype=object) for k in range(1, 5)}
    fields["spikePow"] = np.empty((1, n), dtype=object)
    for i, t in enumerate(lengths):
        for k in range(1, 5):
            fields[f"tx{k}"][0, i] = rng.poisson(1.0 + k, size=(t, 256)).astype(np.uint8)
        fields["spikePow"][0, i] = (rng.gamma(2.0, 50.0, size=(t, 256)) + 10 * i).astype(np.float32)
        if dead is not None:
            fields["tx2"][0, i][:, dead] = 3
        if outlier and i == 0:
            fields["spikePow"][0, i][0, 5] = 1e7          # far beyond 10 SD of its block
    fields["sentenceText"] = np.array([s.ljust(40) for s in sentences])   # fixed-width, space padded
    fields["blockIdx"] = np.array(blocks, dtype=np.uint8).reshape(-1, 1)
    scipy.io.savemat(path, fields)
    return fields


class ConvertT12Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for split in ("train", "test", "competitionHoldOut"):
            (self.root / "in" / split).mkdir(parents=True)
        self.train_sentences = ["Hello there, world.", "Don't stop!", "Another one here.", "!!!",
                                "verylongword verylongword", "Last train trial."]
        self.train = write_mat(self.root / "in/train/t12.2022.05.05.mat", self.train_sentences,
                               [3, 3, 3, 4, 4, 4], [80, 41, 25, 33, 28, 36], rng, outlier=True, dead=7)
        self.test = write_mat(self.root / "in/test/t12.2022.05.05.mat", ["A test sentence.", "Second test."],
                              [9, 9], [27, 31], rng)
        write_mat(self.root / "in/competitionHoldOut/t12.2022.05.05.mat", ["HELD OUT"] * 2, [1, 1], [20, 20], rng)
        self.classes = ct.class_order()
        self.index = {p: i for i, p in enumerate(self.classes)}

    def convert(self, exclusions=frozenset(), name="out"):
        sessions = ct.find_sessions(self.root / "in")
        path = self.root / name / "t12.2022.05.05" / "data_train.hdf5"
        report = ct.convert_session("t12.2022.05.05", sessions["t12.2022.05.05"], path, fake_phonemizer,
                                    self.index, exclusions)
        return path, report

    def raw(self, fields, i):
        return np.concatenate([fields["tx2"][0, i].astype(np.float64), fields["spikePow"][0, i].astype(np.float64)], 1)

    def test_class_order_matches_the_acoustic_model(self):
        from evaluate_model_helpers import LOGIT_TO_PHONEME
        self.assertEqual(self.classes, ["SIL" if p == " | " else p for p in LOGIT_TO_PHONEME])
        self.assertEqual(len(self.classes), 41)

    def test_layout_dtypes_attrs_and_padding_match_t15(self):
        path, report = self.convert()
        self.assertEqual(report["n_source"], {"train": 6, "test": 2})
        self.assertEqual(report["n_written"], 6)                                   # '!!!' and the long one fail
        self.assertEqual(sorted(f["index"] for f in report["failures"]), [3, 4])
        with h5py.File(path, "r") as f:
            self.assertEqual(list(f.keys()), [f"trial_{i:04d}" for i in range(6)])  # contiguous for the loader
            g = f["trial_0000"]
            self.assertEqual(g["input_features"].dtype, np.float32)
            self.assertEqual(g["input_features"].shape, (80, 512))
            self.assertEqual(g["input_features"].compression, "lzf")
            for name in ("seq_class_ids", "transcription"):
                self.assertEqual(g[name].dtype, np.int32)
                self.assertEqual(g[name].shape, (500,))
            for name in ("block_num", "n_time_steps", "seq_len", "trial_num", "source_index"):
                self.assertIsInstance(g.attrs[name], np.int64)
            for name in ("session", "sentence_label", "source_split"):
                self.assertIsInstance(g.attrs[name], str)
            self.assertEqual(g.attrs["sentence_label"], "Hello there, world.")        # padding stripped
            self.assertEqual(g.attrs["session"], "t12.2022.05.05")
            self.assertEqual(g.attrs["n_time_steps"], 80)
            ids = g["seq_class_ids"][:]
            n = g.attrs["seq_len"]
            self.assertEqual(n, 9)
            self.assertEqual(ids[:n].tolist(), [self.index[p] for p in fake_phonemizer("Hello there, world.")])
            self.assertEqual(ids[n - 1], 40)                                         # trailing SIL
            self.assertFalse(ids[n:].any())
            text = g["transcription"][:]
            self.assertEqual("".join(chr(c) for c in text[:19]), "Hello there, world.")
            self.assertFalse(text[19:].any())
            order = [(int(f[k].attrs["block_num"]), int(f[k].attrs["trial_num"]), f[k].attrs["source_split"])
                     for k in f.keys()]
        self.assertEqual(order, [(3, 0, "train"), (3, 1, "train"), (3, 2, "train"), (4, 2, "train"),
                                 (9, 0, "test"), (9, 1, "test")])

    def test_features_are_block_zscored_tx2_then_spikepow_and_clipped(self):
        path, _ = self.convert()
        with h5py.File(path, "r") as f:
            got = {(f[k].attrs["source_split"], int(f[k].attrs["source_index"])): f[k]["input_features"][:]
                   for k in f.keys()}
        # block 4 (train rows 3-5) statistics include the two trials that failed labelling
        raw = [self.raw(self.train, i) for i in (3, 4, 5)]
        stacked = np.concatenate(raw)
        mean, std = stacked.mean(0), stacked.std(0)
        live = std > 0
        expected = np.zeros_like(raw[2])
        expected[:, live] = (raw[2][:, live] - mean[live]) / std[live]
        np.testing.assert_allclose(got[("train", 5)], np.clip(expected, -10, 10), rtol=1e-5, atol=1e-5)
        self.assertTrue((got[("train", 5)][:, 7] == 0).all())                       # constant tx2 column -> 0
        # block 9 (the test file) is normalized on its own: mean 0, SD 1 over its trials
        block9 = np.concatenate([got[("test", 0)], got[("test", 1)]]).astype(np.float64)
        np.testing.assert_allclose(block9.mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(block9.std(0), 1, atol=1e-4)
        tx_block9 = np.concatenate([self.raw(self.test, 0), self.raw(self.test, 1)])[:, :256]
        np.testing.assert_allclose(got[("test", 0)][:, :256],
                                   (tx_block9[:27] - tx_block9.mean(0)) / tx_block9.std(0), rtol=1e-5, atol=1e-5)
        self.assertEqual(got[("train", 0)][0, 256 + 5], 10.0)                       # the outlier hits the rail
        self.assertLessEqual(max(np.abs(x).max() for x in got.values()), 10.0)

    def test_exclusion_list_drops_normalized_matches_without_renumbering(self):
        exclusions = ct.load_exclusions(self.write_exclusions(["  HELLO there world ", "second   TEST", ""]))
        self.assertEqual(exclusions, {"hello there world", "second test"})
        path, report = self.convert(exclusions)
        self.assertEqual(report["n_excluded"], 2)
        self.assertEqual(report["n_written"], 4)
        with h5py.File(path, "r") as f:
            self.assertEqual(list(f.keys()), [f"trial_{i:04d}" for i in range(4)])
            labels = [f[k].attrs["sentence_label"] for k in f.keys()]
            first = (int(f["trial_0000"].attrs["block_num"]), int(f["trial_0000"].attrs["trial_num"]))
        self.assertNotIn("Hello there, world.", labels)
        self.assertNotIn("Second test.", labels)
        self.assertEqual(first, (3, 1))                                             # trial_num kept from source

    def test_zero_area44_and_drop_blocks(self):
        sessions = ct.find_sessions(self.root / "in")
        path = self.root / "z" / "t12.2022.05.05" / "data_train.hdf5"
        report = ct.convert_session("t12.2022.05.05", sessions["t12.2022.05.05"], path, fake_phonemizer, self.index,
                                    drop_blocks=frozenset({4}), zero_area44=True)
        full, _ = self.convert(name="full")
        self.assertEqual(report["n_dropped_blocks"], 3)
        with h5py.File(path, "r") as f, h5py.File(full, "r") as g:
            self.assertEqual(sorted({int(f[k].attrs["block_num"]) for k in f}), [3, 9])
            x = f["trial_0000"]["input_features"][:]
            self.assertTrue((x[:, 128:256] == 0).all() and (x[:, 384:512] == 0).all())
            # block 3 is normalized without block 4, so area 6v matches the unmodified conversion exactly
            np.testing.assert_array_equal(x[:, :128], g["trial_0000"]["input_features"][:, :128])
            self.assertGreater(np.abs(x[:, :128]).sum(), 0)

    def write_exclusions(self, lines):
        path = self.root / "exclude.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_default_excludes_nothing_and_holdout_is_never_read(self):
        path, report = self.convert()
        self.assertEqual(report["n_excluded"], 0)
        with h5py.File(path, "r") as f:
            labels = {f[k].attrs["sentence_label"] for k in f.keys()}
        self.assertNotIn("HELD OUT", labels)

    def test_output_loads_through_the_training_dataset(self):
        path, _ = self.convert()
        train, _ = train_test_split_indicies([str(path)], test_percentage=0, seed=1)
        self.assertEqual(train[0]["trials"], list(range(6)))
        dataset = BrainToTextDataset(train, n_batches=2, split="train", batch_size=4, days_per_batch=1, random_seed=1)
        batch = dataset[0]
        self.assertEqual(batch["input_features"].shape[0], 4)
        self.assertEqual(batch["input_features"].shape[2], 512)
        self.assertEqual(batch["transcriptions"].shape, (4, 500))
        # a session without data_val.hdf5 yields an empty validation set rather than an error
        _, val = train_test_split_indicies([str(path.with_name("data_val.hdf5"))], test_percentage=1)
        self.assertEqual(val[0]["trials"], [])
        self.assertEqual(len(BrainToTextDataset(val, n_batches=None, split="test", batch_size=4)), 0)

    def test_main_refuses_existing_output_and_links_combined_sessions(self):
        t15 = self.root / "t15"
        for name in ("t15.2023.08.11", "t15.2023.08.13"):
            (t15 / name).mkdir(parents=True)
        out = self.root / "t12"
        with unittest.mock.patch.object(ct, "repo_phonemizer", lambda: fake_phonemizer), \
                contextlib.redirect_stdout(io.StringIO()):
            summary = ct.main(["--input-dir", str(self.root / "in"), "--out-dir", str(out),
                               "--combined-dir", str(self.root / "both"), "--t15-dir", str(t15)])
            self.assertEqual(summary["n_written"], 6)
            self.assertTrue((out / "conversion_summary.json").exists())
            with self.assertRaises(SystemExit):
                ct.main(["--input-dir", str(self.root / "in"), "--out-dir", str(out)])
        links = sorted(p.name for p in (self.root / "both").iterdir())
        self.assertEqual(links, ["t12.2022.05.05", "t15.2023.08.11", "t15.2023.08.13"])
        self.assertTrue((self.root / "both" / "t12.2022.05.05").is_symlink())
        self.assertTrue((self.root / "both" / "t12.2022.05.05" / "data_train.hdf5").exists())
        with self.assertRaises(ValueError):                                         # duplicate session names
            ct.link_combined(self.root / "both2", t15, t15)

    def test_repo_g2p_labels_a_simple_sentence(self):
        try:
            phonemize = ct.repo_phonemizer()
            phones = phonemize("Hello world.")
        except (ImportError, LookupError) as error:                                 # g2p_en / NLTK data missing
            self.skipTest(f"g2p unavailable: {error}")
        self.assertEqual(phones, ["HH", "AH", "L", "OW", "SIL", "W", "ER", "L", "D", "SIL"])
        ids, n = ct.encode_phonemes(phones, self.index)
        self.assertEqual(ids[:n].tolist(), [16, 3, 21, 25, 40, 36, 12, 21, 9, 40])


if __name__ == "__main__":
    unittest.main()
