import importlib.util
import gzip
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_mlm_pretraining.py"
)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("mlm_pretraining_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MLMPretainingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_model_specs_include_matched_pretraining_models(self):
        self.assertEqual(set(self.runner.MODEL_SPECS), {"uncased", "cased", "capitalized"})
        self.assertEqual(
            self.runner.MODEL_SPECS["uncased"]["model_name"],
            "bert-base-uncased",
        )
        self.assertEqual(
            self.runner.MODEL_SPECS["cased"]["model_name"],
            "bert-base-cased",
        )
        self.assertEqual(
            self.runner.MODEL_SPECS["capitalized"]["kind"],
            "capitalized",
        )

    def test_flatten_metrics_keeps_numeric_values_only(self):
        flattened = self.runner.flatten_metrics(
            {"eval_loss": 1.2, "epoch": 1, "label": "skip"}
        )

        self.assertEqual(flattened, {"eval_loss": 1.2, "epoch": 1.0})

    def test_write_json_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "metrics.json"

            self.runner.write_json(path, {"model_kind": "uncased"})

            self.assertIn('"model_kind": "uncased"', path.read_text())

    def test_parse_args_accepts_initial_checkpoint(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "uncased",
            "--initial-checkpoint",
            "/tmp/checkpoint",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertEqual(args.initial_checkpoint, "/tmp/checkpoint")

    def test_parse_args_accepts_capitalization_task_mix_corpus(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "capitalized",
            "--corpus",
            "capitalization_task_mix",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertEqual(args.corpus, "capitalization_task_mix")

    def test_parse_args_accepts_augmented_task_mix_and_class_weights(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "capitalized",
            "--corpus",
            "capitalization_task_mix_augmented",
            "--capitalization-loss-weight",
            "0.5",
            "--capitalization-class-weights",
            "1,2,8",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertEqual(args.corpus, "capitalization_task_mix_augmented")
        self.assertEqual(args.capitalization_loss_weight, 0.5)
        self.assertEqual(
            self.runner.parse_class_weights(args.capitalization_class_weights),
            [1.0, 2.0, 8.0],
        )

    def test_parse_args_accepts_mixed_case_capitalization(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "capitalized",
            "--use-mixed-case-capitalization",
            "--capitalization-embedding-dropout",
            "0.1",
            "--capitalization-class-weights",
            "1,2,8,4",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertTrue(args.use_mixed_case_capitalization)
        self.assertEqual(args.capitalization_embedding_dropout, 0.1)
        self.assertEqual(
            self.runner.parse_class_weights(args.capitalization_class_weights),
            [1.0, 2.0, 8.0, 4.0],
        )
        self.assertEqual(
            self.runner.capitalization_config_overrides(args)["capitalization_vocab_size"],
            4,
        )

    def test_parse_args_accepts_real_acronym_mix_corpus(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "capitalized",
            "--corpus",
            "capitalization_real_acronym_mix",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertEqual(args.corpus, "capitalization_real_acronym_mix")

    def test_parse_args_accepts_domain_mix_v2_corpus(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "capitalized",
            "--corpus",
            "capitalization_domain_mix_v2",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertEqual(args.corpus, "capitalization_domain_mix_v2")

    def test_parse_args_accepts_v3_corpora(self):
        import sys
        from unittest.mock import patch

        for corpus in (
            "capitalization_v3_general",
            "capitalization_v3_domain_train",
            "capitalization_v3_mixed_curriculum",
        ):
            argv = [
                "run_mlm_pretraining.py",
                "--model-kind",
                "capitalized",
                "--corpus",
                corpus,
            ]

            with patch.object(sys, "argv", argv):
                args = self.runner.parse_args()

            self.assertEqual(args.corpus, corpus)

    def test_parse_args_accepts_capitalization_freeze_warmup(self):
        import sys
        from unittest.mock import patch

        argv = [
            "run_mlm_pretraining.py",
            "--model-kind",
            "capitalized",
            "--freeze-non-capitalization-parameters",
        ]

        with patch.object(sys, "argv", argv):
            args = self.runner.parse_args()

        self.assertTrue(args.freeze_non_capitalization_parameters)

    def test_augmented_task_mix_adds_case_variants(self):
        rows = ["tom met nasa in paris", "IBM hired alice"]

        augmented = self.runner.augment_capitalization_rows(rows)

        self.assertGreater(len(augmented), len(rows))
        self.assertIn("Tom met nasa in paris", augmented)
        self.assertIn("TOM met nasa in paris", augmented)
        self.assertGreaterEqual(augmented.count("IBM hired alice"), 2)

    def test_acronym_score_prefers_uppercase_terms(self):
        self.assertGreater(
            self.runner.acronym_score("NASA and FDA reviewed DNA data"),
            self.runner.acronym_score("ordinary lowercase sentence"),
        )

    def test_select_acronym_rich_rows_keeps_highest_scoring_rows(self):
        rows = [
            "ordinary lowercase sentence",
            "NASA FDA DNA RNA",
            "The Court reviewed section text",
        ]

        selected = self.runner.select_acronym_rich_rows(rows, max_rows=1)

        self.assertEqual(selected, ["NASA FDA DNA RNA"])

    def test_text_from_value_flattens_nested_dataset_values(self):
        value = {
            "abstract": ["NASA", {"term": "iPhone"}, None],
            "ignored": 3,
        }

        self.assertEqual(
            self.runner.text_from_value(value),
            "NASA iPhone",
        )

    def test_case_signal_score_counts_first_all_and_mixed_case(self):
        self.assertGreater(
            self.runner.case_signal_score("NASA met iPhone at Stanford"),
            self.runner.case_signal_score("ordinary lowercase sentence"),
        )

    def test_capitalization_profile_counts_v3_buckets(self):
        profile = self.runner.capitalization_profile("NASA met iPhone at Stanford")

        self.assertEqual(profile["all_caps_count"], 1)
        self.assertEqual(profile["mixed_case_count"], 1)
        self.assertGreaterEqual(profile["first_cap_count"], 1)
        self.assertEqual(
            self.runner.classify_v3_bucket(profile),
            "mixed_case_rich",
        )

    def test_make_v3_record_adds_manifest_fields(self):
        record = self.runner.make_v3_record(
            "NASA met iPhone at Stanford",
            "unit_source",
            "train",
            7,
        )

        self.assertEqual(record["source"], "unit_source")
        self.assertEqual(record["source_split"], "train")
        self.assertEqual(record["source_index"], 7)
        self.assertEqual(record["selected_bucket"], "mixed_case_rich")
        self.assertEqual(len(record["text_hash"]), 64)

    def test_select_case_rich_rows_deduplicates_and_filters_lowercase(self):
        rows = [
            "ordinary lowercase sentence",
            "NASA met iPhone at Stanford",
            "NASA met iPhone at Stanford",
            "IBM FDA DNA RNA",
        ]

        selected = self.runner.select_case_rich_rows(rows, max_rows=3, min_score=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], "IBM FDA DNA RNA")

    def test_select_v3_records_balances_buckets_and_source_caps(self):
        rows = [
            self.runner.make_v3_record("ordinary lowercase sentence", "a", "train", 0),
            self.runner.make_v3_record("another ordinary lowercase row", "a", "train", 1),
            self.runner.make_v3_record("NASA and FDA reviewed DNA", "b", "train", 0),
            self.runner.make_v3_record("Stanford University Filed Motion", "c", "train", 0),
            self.runner.make_v3_record("iPhone and eBay changed APIs", "d", "train", 0),
            self.runner.make_v3_record("NASA and FDA reviewed DNA", "b", "train", 1),
        ]

        selected = self.runner.select_v3_records(
            rows,
            max_rows=4,
            source_cap=1,
            bucket_fractions={
                "ordinary": 0.25,
                "all_caps_rich": 0.25,
                "first_cap_rich": 0.25,
                "mixed_case_rich": 0.25,
            },
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(len({record["text_hash"] for record in selected}), 4)
        self.assertLessEqual(
            max(
                sum(1 for record in selected if record["source"] == source)
                for source in {record["source"] for record in selected}
            ),
            1,
        )

    def test_write_and_load_cached_v3_rows(self):
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                self.runner.make_v3_record("NASA met iPhone at Stanford", "unit", "train", 0),
                self.runner.make_v3_record("ordinary lowercase sentence", "unit", "train", 1),
            ]

            with patch.dict(os.environ, {"CAP_EMB_WORK_ROOT": tmpdir}):
                self.runner.write_cached_v3_rows(
                    "capitalization_v3_general",
                    [records[0]],
                    [records[1]],
                )
                train_rows, eval_rows = self.runner.load_cached_v3_rows(
                    "capitalization_v3_general",
                )
                manifest_path = self.runner.v3_manifest_path()

            self.assertEqual(train_rows, ["NASA met iPhone at Stanford"])
            self.assertEqual(eval_rows, ["ordinary lowercase sentence"])
            with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
                manifest_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(manifest_rows), 2)
            self.assertNotIn("text", manifest_rows[0])

    def test_stable_shuffle_rows_is_deterministic(self):
        rows = ["a", "b", "c", "d", "e"]

        first = self.runner.stable_shuffle_rows(rows, seed=13)
        second = self.runner.stable_shuffle_rows(rows, seed=13)

        self.assertEqual(first, second)
        self.assertCountEqual(first, rows)
        self.assertNotEqual(first, rows)

    def test_chunk_text_splits_long_documents(self):
        text = " ".join(f"WORD{i}" for i in range(36))

        chunks = self.runner.chunk_text(text, words_per_chunk=12)

        self.assertEqual(len(chunks), 3)

    def test_resolve_resume_checkpoint_prefers_explicit_checkpoint(self):
        args = type(
            "Args",
            (),
            {
                "resume_from_checkpoint": "/tmp/checkpoint-10",
                "no_auto_resume": False,
            },
        )()

        checkpoint = self.runner.resolve_resume_checkpoint(Path("/missing"), args)

        self.assertEqual(checkpoint, "/tmp/checkpoint-10")

    def test_resolve_resume_checkpoint_can_disable_auto_resume(self):
        args = type(
            "Args",
            (),
            {
                "resume_from_checkpoint": "",
                "no_auto_resume": True,
            },
        )()

        checkpoint = self.runner.resolve_resume_checkpoint(Path("/missing"), args)

        self.assertIsNone(checkpoint)


if __name__ == "__main__":
    unittest.main()
