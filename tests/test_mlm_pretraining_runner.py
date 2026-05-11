import importlib.util
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
        self.assertEqual(self.runner.parse_class_weights(args.capitalization_class_weights), [1.0, 2.0, 8.0])

    def test_augmented_task_mix_adds_case_variants(self):
        rows = ["tom met nasa in paris", "IBM hired alice"]

        augmented = self.runner.augment_capitalization_rows(rows)

        self.assertGreater(len(augmented), len(rows))
        self.assertIn("Tom met nasa in paris", augmented)
        self.assertIn("TOM met nasa in paris", augmented)
        self.assertGreaterEqual(augmented.count("IBM hired alice"), 2)

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
