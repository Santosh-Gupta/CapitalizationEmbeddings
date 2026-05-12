import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_sequence_classification_benchmark.py"
)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("sequence_benchmark_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SequenceBenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_model_specs_match_token_runner_keys(self):
        self.assertEqual(
            set(self.runner.MODEL_SPECS),
            {
                "uncased",
                "uncased_pretrained",
                "cased",
                "cased_pretrained",
                "capitalized",
                "capitalized_pretrained",
            },
        )

    def test_merge_text_columns_joins_multiple_fields(self):
        examples = {
            "question_title": ["Title"],
            "question_content": ["Body"],
            "best_answer": ["Answer"],
        }

        merged = self.runner.merge_text_columns(
            examples,
            ("question_title", "question_content", "best_answer"),
            "[SEP]",
        )

        self.assertEqual(merged, ["Title [SEP] Body [SEP] Answer"])

    def test_checkpoint_for_model_requires_expected_cli_arg(self):
        args = type(
            "Args",
            (),
            {
                "uncased_checkpoint": "",
                "cased_checkpoint": "/tmp/cased",
                "capitalized_checkpoint": "",
            },
        )()

        checkpoint = self.runner.checkpoint_for_model(
            "cased_pretrained",
            self.runner.MODEL_SPECS["cased_pretrained"],
            args,
        )

        self.assertEqual(checkpoint, "/tmp/cased")
        with self.assertRaisesRegex(ValueError, "--uncased-checkpoint"):
            self.runner.checkpoint_for_model(
                "uncased_pretrained",
                self.runner.MODEL_SPECS["uncased_pretrained"],
                args,
            )

    def test_hidden_regression_test_labels_use_validation_split(self):
        raw = {
            "validation": {"label": [1.0, 2.0]},
            "test": {"label": [-1.0, -1.0]},
        }

        split = self.runner.labeled_evaluation_split(
            raw,
            "label",
            regression=True,
        )

        self.assertEqual(split, "validation")

    def test_jsonl_and_csv_writers_create_scoreboard_files(self):
        row = {
            "benchmark": "tweet_eval_irony",
            "model_key": "capitalized_pretrained",
            "test_accuracy": 0.7,
            "benchmark_spec": {"not": "csv"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"
            csv_path = Path(tmpdir) / "results.csv"

            self.runner.append_jsonl(jsonl_path, row)
            self.runner.write_csv(csv_path, [row])

            self.assertIn('"model_key": "capitalized_pretrained"', jsonl_path.read_text())
            csv_text = csv_path.read_text()
            self.assertIn("benchmark,model_key,test_accuracy", csv_text)
            self.assertIn("tweet_eval_irony,capitalized_pretrained,0.7", csv_text)

    def test_save_sequence_predictions_writes_paired_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"

            self.runner.save_sequence_predictions(
                path=path,
                model_key="uncased",
                benchmark="tweet_eval_irony",
                evaluation_split="test",
                predictions=[[0.1, 0.9], [0.8, 0.2]],
                labels=[1, 0],
                regression=False,
            )

            text = path.read_text()
            self.assertIn('"prediction": 1', text)
            self.assertIn('"label": 0', text)


if __name__ == "__main__":
    unittest.main()
