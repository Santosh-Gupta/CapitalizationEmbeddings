import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_token_classification_benchmark.py"
)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("benchmark_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_model_specs_include_goal_comparison_models(self):
        self.assertEqual(
            set(self.runner.MODEL_SPECS),
            {"uncased", "cased", "capitalized"},
        )

    def test_label_fallbacks_cover_conll2003_ner(self):
        labels = self.runner.FALLBACK_LABELS[("lhoestq/conll2003", "ner_tags")]

        self.assertEqual(labels[0], "O")
        self.assertEqual(labels[-1], "I-MISC")
        self.assertEqual(len(labels), 9)

    def test_flatten_metrics_keeps_numeric_values_only(self):
        metrics = {
            "test_f1": 0.9,
            "test_samples_per_second": 12,
            "label": "ignored",
            "nested": {"ignored": True},
        }

        flattened = self.runner.flatten_metrics(metrics)

        self.assertEqual(flattened, {"test_f1": 0.9, "test_samples_per_second": 12.0})

    def test_jsonl_and_csv_writers_create_scoreboard_files(self):
        row = {
            "benchmark": "conll2003_ner",
            "model_key": "uncased",
            "test_f1": 0.8,
            "benchmark_spec": {"not": "csv"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "results.jsonl"
            csv_path = Path(tmpdir) / "results.csv"

            self.runner.append_jsonl(jsonl_path, row)
            self.runner.write_csv(csv_path, [row])

            self.assertIn('"model_key": "uncased"', jsonl_path.read_text())
            csv_text = csv_path.read_text()
            self.assertIn("benchmark,model_key,test_f1", csv_text)
            self.assertIn("conll2003_ner,uncased,0.8", csv_text)


if __name__ == "__main__":
    unittest.main()
