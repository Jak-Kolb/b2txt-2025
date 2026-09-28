"""Harness registry/scope tests on synthetic checkpoints and HDF5 (no GPU or native decoder)."""
import sys
import tempfile
import unittest
from pathlib import Path

if sys.version_info < (3, 10):
    raise unittest.SkipTest("harness tests require the Python 3.10 acoustic stack")

import harness_fixture as fx
from harness import registry, scopes
from harness.store import HashCache


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.registry = fx.make_registry(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def resolve(self, *args, **kwargs):
        return registry.resolve(*args, root=self.registry, **kwargs)

    def test_preset_resolves_all_decode_keys_and_flags(self):
        spec = self.resolve("tiny_fake")
        self.assertEqual(set(spec["decode"]), set(registry.DECODE_KEYS))
        self.assertIsInstance(spec["decode"]["max_active"], int)
        self.assertTrue(spec["streaming_capable"])
        self.assertEqual(registry.flags(spec, "val-dev"),
                         ["acoustic_selected_on_scope", "decode_tuned_on_scope"])

    def test_identity_is_order_independent_and_tracks_decode(self):
        hasher = HashCache(self.root / "h")
        spec = self.resolve("tiny_fake")
        reordered = dict(spec, decode=dict(reversed(list(spec["decode"].items()))))
        self.assertEqual(registry.identity(spec, hasher)["pipeline_hash"],
                         registry.identity(reordered, hasher)["pipeline_hash"])
        changed = self.resolve("tiny_fake", decode={"beam": 12})
        self.assertNotEqual(registry.identity(spec, hasher)["pipeline_hash"],
                            registry.identity(changed, hasher)["pipeline_hash"])
        self.assertEqual(registry.identity(spec, hasher)["acoustic_key"],
                         registry.identity(changed, hasher)["acoustic_key"])
        self.assertIn("decode_tuning_unrecorded", registry.flags(changed, "val-dev"))

    def test_preprocess_override_is_flagged_and_limited(self):
        spec = self.resolve("tiny_fake", preprocess={"smooth_lookahead": 2})
        self.assertEqual(spec["preprocess"]["effective"]["smooth_lookahead"], 2)
        self.assertEqual(spec["preprocess"]["trained"]["smooth_lookahead"], 0)
        self.assertIn("preprocess_mismatch", registry.flags(spec, "val-dev"))
        with self.assertRaises(ValueError):
            self.resolve("tiny_fake", preprocess={"white_noise_std": 1})

    def test_symmetric_smoothing_is_offline_only(self):
        spec = self.resolve("sym_fake")
        self.assertFalse(spec["streaming_capable"])
        self.assertIn("offline_noncausal", registry.flags(spec, "val-dev"))

    def test_other_acoustic_and_exposed_partition_flags(self):
        spec = self.resolve("tiny_fake", acoustic="tiny_sym")
        self.assertEqual(spec["name"], "tiny_sym+fake")
        flags = registry.flags(spec, "former-val-test", limited=True)
        for flag in ("decode_tuned_for_other_acoustic", "exposed_partition", "smoke_limited"):
            self.assertIn(flag, flags)

    def test_bad_entries_are_rejected(self):
        bad = self.root / "bad_registry"
        (bad / "pipelines").mkdir(parents=True)
        (bad / "acoustic").mkdir()
        (bad / "lm").mkdir()
        (bad / "pipelines" / "p.yaml").write_text("name: p\nacoustic: a\nlm: l\nsurprise: 1\n")
        with self.assertRaisesRegex(ValueError, "keys"):
            registry.load_kind("pipelines", bad)
        (bad / "pipelines" / "p.yaml").write_text("name: q\nacoustic: a\nlm: l\n")
        with self.assertRaisesRegex(ValueError, "file name"):
            registry.load_kind("pipelines", bad)

    def test_preset_decode_override_requires_provenance(self):
        path = self.registry / "pipelines" / "untracked.yaml"
        path.write_text("name: untracked\nacoustic: tiny\nlm: fake\ndecode: {beam: 3}\n")
        try:
            with self.assertRaisesRegex(ValueError, "decode_tuned_on"):
                self.resolve("untracked")
        finally:
            path.unlink()

    def test_real_registry_entries_load(self):
        loaded = registry.load_all()
        self.assertIn("causal_la0", loaded["acoustic"])
        for preset in loaded["pipelines"].values():
            self.assertIn(preset["acoustic"], loaded["acoustic"])
            self.assertIn(preset["lm"], loaded["lm"])


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.data = fx.make_data(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_midpoint_picks(self):
        self.assertEqual(scopes.midpoint_picks(10, 2), [2, 7])
        self.assertEqual(scopes.midpoint_picks(9, 10), list(range(9)))
        self.assertEqual(scopes.midpoint_picks(4, 1), [2])

    def test_val_dev_excludes_former_val_test_and_samples_per_session(self):
        full = scopes.build_scope("val-dev-full", self.data)
        self.assertEqual(full["n_trials"], 8)
        self.assertNotIn("t15.2025.01.10", {t.session for t in full["trials"]})
        sample = scopes.build_scope("val-dev-sample", self.data)
        self.assertEqual([(t.session, t.trial_key) for t in sample["trials"]],
                         [("t15.2099.01.01", "trial_0001"), ("t15.2099.01.01", "trial_0003"),
                          ("t15.2099.01.02", "trial_0001"), ("t15.2099.01.02", "trial_0003")])
        limited = scopes.build_scope("val-dev-full", self.data, limit=3)
        self.assertEqual(limited["n_trials"], 3)
        self.assertNotEqual(limited["hash"], full["hash"])

    def test_exposed_scope_requires_a_reason(self):
        with self.assertRaises(PermissionError):
            scopes.build_scope("former-val-test-full", self.data)
        exposed = scopes.build_scope("former-val-test-full", self.data, allow_exposed="audit")
        self.assertEqual(exposed["partition"], "former-val-test")
        self.assertEqual(exposed["exposed_reason"], "audit")

    def test_hash_tracks_file_content(self):
        root = Path(self.temp.name)
        hasher = HashCache(root / "h")
        first = scopes.build_scope("val-dev-full", self.data, hasher=hasher)
        self.assertEqual(first["hash"], scopes.build_scope("val-dev-full", self.data, hasher=hasher)["hash"])
        other = fx.make_data(root / "other", seed=1)
        self.assertNotEqual(first["hash"], scopes.build_scope("val-dev-full", other, hasher=hasher)["hash"])

    def test_labels_and_seen_set(self):
        trial = scopes.build_scope("val-dev-full", self.data)["trials"][0]
        reference, phones = scopes.read_labels(trial)
        self.assertEqual(reference, "w2 w3.")
        self.assertEqual(phones.tolist(), [1, 2, 3])
        seen = scopes.train_sentence_set(self.data, Path(self.temp.name) / "cache_root")
        self.assertEqual(seen, {"w4", "w9"})
        unlabeled = scopes.list_trials(self.data, "test")[0]
        self.assertEqual(scopes.read_labels(unlabeled), (None, None))

    def test_sentence_source_from_block_description(self):
        trials = scopes.list_trials(self.data, "val")
        by_session = {t.session: scopes.trial_info(t)["corpus"] for t in trials}
        self.assertEqual(by_session, {"t15.2099.01.01": "Switchboard", "t15.2099.01.02": "Random",
                                      "t15.2025.01.10": "Harvard"})
        self.assertIsNone(scopes.corpus_of("t15.2099.01.01", 7, self.data))
        self.assertIsNone(scopes.corpus_of("t15.2099.01.01", 1, Path(self.temp.name) / "nowhere" / "data"))


if __name__ == "__main__":
    unittest.main()
