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


if __name__ == "__main__":
    unittest.main()
