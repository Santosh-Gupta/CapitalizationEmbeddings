import json
import unittest
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


class NotebookStaticTests(unittest.TestCase):
    def notebook_source(self, notebook_name: str) -> str:
        notebook = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_training_arguments_use_pinned_transformers_api(self):
        for path in NOTEBOOK_DIR.glob("*.ipynb"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "evaluation_strategy",
                text,
                f"{path.name} should use eval_strategy for transformers 4.56.2.",
            )

    def test_trainer_uses_processing_class_not_tokenizer(self):
        for path in NOTEBOOK_DIR.glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue

                source = "".join(cell.get("source", []))
                if "Trainer(" not in source:
                    continue

                trainer_block = source[source.index("Trainer(") :]
                self.assertIn("processing_class=tokenizer", trainer_block, path.name)
                self.assertNotIn("    tokenizer=tokenizer,", trainer_block, path.name)

    def test_notebooks_auto_cd_to_drive_repo_when_available(self):
        expected = "/content/drive/MyDrive/Github/CapitalizationEmbeddings"
        for path in NOTEBOOK_DIR.glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            first_code_cell = next(
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            )
            source = "".join(first_code_cell.get("source", []))
            self.assertIn(expected, source, path.name)
            self.assertIn('drive.mount("/content/drive")', source, path.name)
            self.assertIn("%pip install -q", source, path.name)

    def test_dataset_references_avoid_script_datasets(self):
        conll_ner = self.notebook_source("02_finetune_conll2003_ner.ipynb")
        conll_baseline = self.notebook_source("03_finetune_conll2003_baselines.ipynb")
        mlm = self.notebook_source("01_continue_pretraining_mlm.ipynb")

        self.assertIn('load_dataset("lhoestq/conll2003")', conll_ner)
        self.assertIn('load_dataset("lhoestq/conll2003")', conll_baseline)
        self.assertNotIn('load_dataset("conll2003")', conll_ner)
        self.assertNotIn('load_dataset("conll2003")', conll_baseline)
        self.assertIn('DATASET_NAME = "Salesforce/wikitext"', mlm)
        self.assertIn('DATASET_CONFIG = "wikitext-2-raw-v1"', mlm)
        self.assertIn('hasattr(ner_feature, "names")', conll_ner)
        self.assertIn('hasattr(ner_feature, "names")', conll_baseline)


if __name__ == "__main__":
    unittest.main()
