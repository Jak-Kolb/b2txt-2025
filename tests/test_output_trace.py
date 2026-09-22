"""Synthetic tests: no neural data, checkpoints, GPU, or native decoder required."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from model_training.benchmark import output_trace as trace
from model_training.benchmark import stream_lm


CONFIG = dict(acoustic_scale=1.0, blank_penalty=30.0, beam=17.0,
              lattice_beam=8.0, max_active=7000, min_active=200,
              length_penalty=0.0, blank_skip_thresh=1.0)


class FakeLM:
    def __init__(self, partials, finals):
        self.partials = partials
        self.finals = finals
        self.result_calls = 0
        self.rows = []
        self.resource_calls = 0

    def DecodeOptions(self, *args):
        return args

    def DecodeResource(self, *args):
        self.resource_calls += 1
        return args

    def BrainSpeechDecoder(self, resource, options):
        owner = self

        class Decoder:
            def __init__(self):
                self.trial = -1

            def Reset(self):
                self.trial += 1
                self.frame = -1
                self.finished = False

            def FinishDecoding(self):
                self.finished = True

            def result(self):
                owner.result_calls += 1
                words = owner.finals[self.trial] if self.finished else owner.partials[self.trial][self.frame]
                return [SimpleNamespace(sentence=" ".join(words))] if words else []

        return Decoder()

    def DecodeNumpy(self, decoder, row, zeros, log_penalty):
        self.rows.append(row.copy())
        decoder.frame += 1


class RevisionTests(unittest.TestCase):
    def test_monotone_growth_has_no_revisions(self):
        partials = [[], ["A"], ["A", "B"], ["A", "B"]]
        histories = trace.trace_partials(partials)
        self.assertEqual([t.revisions for t in histories], [0, 0])
        self.assertEqual(trace.word_edits(["A"], ["A", "B"]),
                         dict(appends=1, insertions=0, deletions=0, substitutions=0))

    def test_retracted_word_remains_in_history(self):
        histories = trace.trace_partials([["A", "B"], ["A"]])
        self.assertEqual(len(histories), 2)
        self.assertIsNone(histories[1].final_word)
        self.assertEqual(histories[1].values, ["B", None])
        self.assertEqual(histories[1].revisions, 1)

    def test_disappear_reappear_same_word_is_not_stable(self):
        word = trace.trace_partials([["A"], [], ["A"], ["A"]])[0]
        self.assertEqual(word.values, ["A", None, "A"])
        self.assertEqual((word.revisions, word.first_frame, word.final_frame), (2, 0, 2))

    def test_substitution_and_internal_insertion(self):
        self.assertEqual(trace.word_edits(["A", "B"], ["A", "X", "B"]),
                         dict(appends=0, insertions=1, deletions=0, substitutions=0))
        self.assertEqual(trace.word_edits(["A", "B"], ["X", "B"])["substitutions"], 1)

    def test_deletion_does_not_become_multiple_position_substitutions(self):
        edits = trace.word_edits(["A", "B", "C"], ["B", "C"])
        self.assertEqual(edits, dict(appends=0, insertions=0, deletions=1, substitutions=0))
        self.assertEqual(sum(t.revisions for t in trace.trace_partials(
            [["A", "B", "C"], ["B", "C"]])), 3)

    def test_repeated_words_have_deterministic_alignment(self):
        self.assertEqual(trace.word_edits(["A", "A"], ["A"]),
                         dict(appends=0, insertions=0, deletions=1, substitutions=0))
        self.assertEqual(trace.word_edits(["A", "B"], ["B", "A"])["substitutions"], 2)

    def test_edit_counts_match_independent_distance_for_small_sequences(self):
        sequences = [list(seq) for n in range(4) for seq in itertools.product("AB", repeat=n)]
        for left in sequences:
            for right in sequences:
                with self.subTest(left=left, right=right):
                    counts = trace.word_edits(left, right)
                    self.assertEqual(sum(counts.values()), stream_lm.levenshtein(left, right))

    def test_empty_histories(self):
        self.assertEqual(trace.trace_partials([]), [])
        self.assertEqual(trace.trace_partials([[], []]), [])
        self.assertEqual(sum(trace.word_edits([], []).values()), 0)

    @unittest.skipIf(sys.version_info < (3, 10), "legacy acoustic module uses the Python 3.10 stack")
    def test_legacy_stability_includes_deleted_positions(self):
        from model_training.benchmark.stability import stability_metrics
        report = stability_metrics([trace.trace_partials([["A"], []])])
        self.assertEqual(report["metric_version"], 2)
        self.assertEqual(report["n_words"], 0)
        self.assertEqual(report["n_positions"], 1)
        self.assertEqual(report["total_revisions"], 1)
        self.assertEqual(report["flicker_rate"], 1.0)
        self.assertIsNone(report["ttf_ms"]["p95"])


class TraceFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.jsonl"

    def output(self, writer, kind, frame, start, ready, words):
        writer.output(kind=kind, frame_index=frame, processing_start_ns=start,
                      output_ready_ns=ready, words=words)

    def test_round_trip_and_finalization_changes(self):
        with trace.OutputTraceWriter(self.path, {"timing_boundary": "cached_logits"}) as writer:
            writer.start_trial(12, 4, 2)
            self.output(writer, "partial", 0, 0, 1_000_000, ["A", "B"])
            self.output(writer, "partial", 1, 2_000_000, 3_000_000, ["A"])
            self.output(writer, "final", 1, 4_000_000, 5_000_000, ["A", "C"])
        report = trace.summarize_trace(self.path)
        self.assertEqual((report["n_trials"], report["n_outputs"]), (1, 3))
        trial = report["trials"][0]
        self.assertEqual((trial["cache_index"], trial["day_index"]), (12, 4))
        self.assertEqual(trial["revision_edits"], 1)
        self.assertEqual(trial["position_revisions"], 2)
        self.assertEqual(trial["finalization_edit_counts"]["appends"], 1)
        self.assertEqual(trial["first_output_elapsed_ms"], 1.0)
        self.assertEqual(trial["final_position_stabilization_ms"]["max"], 4.0)
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertIsNone(records[2]["input_available_ns"])

    def test_finalization_substitution_is_counted(self):
        with trace.OutputTraceWriter(self.path, {}) as writer:
            writer.start_trial(0, 0, 1)
            self.output(writer, "partial", 0, 0, 100, ["A"])
            self.output(writer, "final", 0, 100, 200, ["B"])
        report = trace.summarize_trace(self.path)["trials"][0]
        self.assertEqual(report["finalization_edit_counts"]["substitutions"], 1)
        self.assertEqual(report["revision_edits"], 1)

    def test_empty_trial_has_valid_json_and_no_fabricated_word_latency(self):
        with trace.OutputTraceWriter(self.path, {}) as writer:
            writer.start_trial(0, -1, 0)
            self.output(writer, "final", None, 0, 20, [])
        report = trace.summarize_trace(self.path)
        trial = report["trials"][0]
        self.assertIsNone(trial["first_output_elapsed_ms"])
        self.assertIsNone(trial["final_position_stabilization_ms"]["p95"])
        json.dumps(report, allow_nan=False)

    def test_existing_trace_is_never_overwritten(self):
        self.path.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            with trace.OutputTraceWriter(self.path, {}):
                pass
        self.assertEqual(self.path.read_text(), "preserve me")

    def test_failure_preserves_incomplete_trace_and_reader_rejects_it(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with trace.OutputTraceWriter(self.path, {}) as writer:
                writer.start_trial(0, 0, 1)
                self.output(writer, "partial", 0, 0, 10, ["A"])
                raise RuntimeError("synthetic failure")
        self.assertTrue(self.path.exists())
        with self.assertRaisesRegex(ValueError, "Incomplete trace"):
            trace.summarize_trace(self.path)

    def test_rejects_final_before_all_frames(self):
        with self.assertRaisesRegex(ValueError, "all trial frames"):
            with trace.OutputTraceWriter(self.path, {}) as writer:
                writer.start_trial(0, 0, 1)
                self.output(writer, "final", None, 0, 10, [])

    def test_rejects_time_reversal(self):
        with self.assertRaisesRegex(ValueError, "monotone"):
            with trace.OutputTraceWriter(self.path, {}) as writer:
                writer.start_trial(0, 0, 1)
                self.output(writer, "partial", 0, 0, 10, ["A"])
                self.output(writer, "final", 0, 9, 20, ["A"])

    def test_reader_rejects_truncated_or_tampered_completion(self):
        with trace.OutputTraceWriter(self.path, {}) as writer:
            writer.start_trial(0, 0, 0)
            self.output(writer, "final", None, 0, 10, [])
        lines = self.path.read_text().splitlines()
        last = json.loads(lines[-1])
        last["n_outputs"] = 9
        lines[-1] = json.dumps(last)
        self.path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "completion counts"):
            trace.summarize_trace(self.path)

    def test_reader_rejects_content_after_completion(self):
        with trace.OutputTraceWriter(self.path, {}):
            pass
        with self.path.open("a") as handle:
            handle.write('{}\n')
        with self.assertRaisesRegex(ValueError, "after run_end"):
            trace.summarize_trace(self.path)

    def test_partial_at_wrong_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, "each logit frame"):
            with trace.OutputTraceWriter(self.path, {}) as writer:
                writer.start_trial(0, 0, 2)
                self.output(writer, "partial", 1, 0, 10, ["A"])

    def test_noninteger_frame_and_unknown_commitment_are_rejected(self):
        for frame, kind in [(False, "partial"), (0.0, "partial"), (0, "committed")]:
            with self.subTest(frame=frame, kind=kind):
                path = self.path.with_name(str(frame) + "-" + kind + ".jsonl")
                with self.assertRaises(ValueError):
                    with trace.OutputTraceWriter(path, {}) as writer:
                        writer.start_trial(0, 0, 1)
                        self.output(writer, kind, frame, 0, 10, ["A"])

    def test_context_cannot_mark_unfinished_trial_complete(self):
        with self.assertRaisesRegex(ValueError, "before the final"):
            with trace.OutputTraceWriter(self.path, {}) as writer:
                writer.start_trial(0, 0, 1)
        with self.assertRaisesRegex(ValueError, "Incomplete trace"):
            trace.summarize_trace(self.path)

    def test_summary_command_works_without_native_decoder(self):
        with trace.OutputTraceWriter(self.path, {}) as writer:
            writer.start_trial(0, 0, 0)
            self.output(writer, "final", None, 0, 10, [])
        output = self.path.with_suffix(".json")
        args = SimpleNamespace(trace=str(self.path), out=str(output))
        with patch.dict(sys.modules, {"lm_decoder": None}), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(stream_lm.cmd_trace_summary(args), 0)
        self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text()))


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.partials = [[[], ["GO"], ["GO", "LEFT"], ["GO"], ["GO", "RIGHT"]], [[]]]
        self.finals = [["GO", "RIGHT", "NOW"], []]
        self.logits = [np.zeros((len(p), 3), dtype=np.float32) for p in self.partials]

    def run_capture(self, references):
        lm = FakeLM(self.partials, self.finals)
        path = self.base / ("trace-" + str(len(list(self.base.glob("trace-*")))) + ".jsonl")
        clock = itertools.count(0, 100_000)
        with trace.OutputTraceWriter(path, {}) as writer:
            with patch.object(stream_lm.time, "perf_counter_ns", side_effect=lambda: next(clock)):
                report = stream_lm.run_decode(
                    lm, self.base, self.logits, references, **CONFIG, capture_partials=True,
                    trace_writer=writer, trial_ids=[7, 11], day_indices=[2, 3])
        return lm, report, path

    def test_capture_preserves_predictions_and_calls_native_decoder_one_row_at_a_time(self):
        lm, report, path = self.run_capture(["GO RIGHT NOW", ""])
        baseline = stream_lm.run_decode(FakeLM(self.partials, self.finals), self.base,
                                        self.logits, ["GO RIGHT NOW", ""], **CONFIG)
        self.assertEqual(report["hyps"], baseline["hyps"])
        self.assertEqual(report["edits"], baseline["edits"])
        self.assertEqual(report["wer_percent"], 0.0)
        self.assertEqual(len(lm.rows), 6)
        self.assertTrue(all(row.shape == (1, 3) for row in lm.rows))
        summary = trace.summarize_trace(path)
        self.assertEqual(summary["n_outputs"], 8)
        self.assertEqual(summary["trials"][0]["cache_index"], 7)
        self.assertEqual(summary["trials"][1]["day_index"], 3)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        outputs = [e for e in events if e["record"] == "output"]
        self.assertEqual(outputs[2]["words"], ["GO", "LEFT"])
        self.assertEqual(outputs[3]["words"], ["GO"])
        self.assertEqual(outputs[5]["kind"], "final")

    def test_reference_changes_do_not_change_output_events(self):
        _, _, a = self.run_capture(["GO RIGHT NOW", ""])
        _, _, b = self.run_capture(["REFERENCE MUST NOT REACH DECODER", "OTHER"])
        def outputs(path):
            return [json.loads(line) for line in path.read_text().splitlines()
                    if json.loads(line)["record"] == "output"]
        self.assertEqual(outputs(a), outputs(b))
        self.assertNotIn("REFERENCE MUST", b.read_text())

    def test_disabled_capture_only_extracts_final_results(self):
        lm = FakeLM(self.partials, self.finals)
        report = stream_lm.run_decode(lm, self.base, self.logits, ["", ""], **CONFIG)
        self.assertEqual(lm.result_calls, 2)
        self.assertNotIn("partial_extract_ms", report)

    def test_capture_requires_persistence(self):
        with self.assertRaisesRegex(ValueError, "trace writer"):
            stream_lm.run_decode(FakeLM([], []), self.base, self.logits, ["", ""],
                                 **CONFIG, capture_partials=True)

    def test_empty_and_mismatched_inputs_fail_before_native_resource_load(self):
        lm = FakeLM([], [])
        for trials, references in [([], []), (self.logits, ["one"])]:
            with self.subTest(n=len(trials)):
                with self.assertRaises(ValueError):
                    stream_lm.run_decode(lm, self.base, trials, references, **CONFIG)
        self.assertEqual(lm.resource_calls, 0)

    def test_zero_frame_trial_can_finalize(self):
        lm = FakeLM([[]], [[]])
        path = self.base / "empty.jsonl"
        with trace.OutputTraceWriter(path, {}) as writer:
            report = stream_lm.run_decode(lm, self.base, [np.empty((0, 3))], [""],
                                         **CONFIG, capture_partials=True, trace_writer=writer)
        self.assertIsNone(report["per_frame_ms"]["max"])
        json.dumps(report, allow_nan=False)
        self.assertEqual(trace.summarize_trace(path)["n_outputs"], 1)

    def args(self):
        return SimpleNamespace(**CONFIG, cache=str(self.base / "cache.npz"), lm=str(self.base),
                               split="dev", n_trials=1, progress=0, capture_partials=True,
                               trace_out=str(self.base / "cli.jsonl"), out=str(self.base / "metrics.json"),
                               save_hyps=str(self.base / "hyps.txt"))

    def save_cache(self):
        np.savez(self.base / "cache.npz", lengths=np.array([1, 1, 1, 1]),
                 flat=np.zeros((4, 3), dtype=np.float32), days=np.array([39, 0, 41, 1]))
        (self.base / "cache.txt").write_text("SKIP\nGO\nSKIP\nSECOND")

    def test_cli_synthetic_cache_preserves_indices_and_writes_replayable_trace(self):
        self.save_cache()
        args = self.args()
        lm = FakeLM([[["GO"]]], [["GO"]])
        with patch.dict(sys.modules, {"lm_decoder": lm}), contextlib.redirect_stdout(io.StringIO()):
            stream_lm.cmd_decode(args)
        report = json.loads(Path(args.out).read_text())
        self.assertEqual(report["cache_indices"], [1])
        self.assertEqual(report["day_indices"], [0])
        self.assertEqual(report["wer_percent"], 0.0)
        replay = trace.summarize_trace(Path(args.trace_out))
        self.assertEqual(report["output_stability"], replay)
        self.assertEqual(replay["metadata"]["endpoint"], "provided_trial_end")
        self.assertEqual(replay["metadata"]["source_feature_availability"], "not_measured")
        self.assertEqual(len(replay["metadata"]["cache_sha256"]), 64)
        self.assertEqual(Path(args.save_hyps).read_text(), "GO\tGO")

    def test_cli_missing_trace_path_fails_before_cache_load(self):
        args = self.args()
        args.trace_out = ""
        with self.assertRaisesRegex(ValueError, "together"):
            stream_lm.cmd_decode(args)

    def test_cli_existing_outputs_fail_before_cache_load(self):
        args = self.args()
        Path(args.out).write_text("existing")
        with self.assertRaises(FileExistsError):
            stream_lm.cmd_decode(args)
        self.assertEqual(Path(args.out).read_text(), "existing")

    def test_cli_duplicate_paths_fail_before_cache_load(self):
        args = self.args()
        args.trace_out = args.out
        with self.assertRaisesRegex(ValueError, "distinct"):
            stream_lm.cmd_decode(args)

    def test_cli_negative_limit_fails_before_cache_load(self):
        args = self.args()
        args.n_trials = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            stream_lm.cmd_decode(args)

    def test_split_empty_and_unknown_days(self):
        self.assertEqual(stream_lm.split_indices([], "dev"), [])
        with self.assertRaisesRegex(ValueError, "day indices"):
            stream_lm.split_indices([-1, 0], "dev")
        self.assertEqual(stream_lm.split_indices([-1, 0], "all"), [0, 1])


if __name__ == "__main__":
    unittest.main()
