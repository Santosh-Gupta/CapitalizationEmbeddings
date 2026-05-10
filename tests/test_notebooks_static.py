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

    def test_training_notebooks_use_hf_compat_helpers(self):
        training_notebooks = [
            "01_continue_pretraining_mlm.ipynb",
            "02_finetune_conll2003_ner.ipynb",
            "03_finetune_conll2003_baselines.ipynb",
        ]
        for notebook_name in training_notebooks:
            source = self.notebook_source(notebook_name)
            self.assertIn("make_training_arguments", source, notebook_name)
            self.assertIn("make_trainer", source, notebook_name)
            self.assertNotIn("TrainingArguments(", source, notebook_name)
            self.assertNotIn("Trainer(", source, notebook_name)
            self.assertNotIn("evaluation_strategy", source, notebook_name)
            self.assertIn("eval_strategy", source, notebook_name)
            self.assertIn("processing_class=tokenizer", source, notebook_name)

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
            self.assertIn("configure_huggingface_cache", source, path.name)

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

    def test_training_notebooks_use_runtime_checkpoint_dirs(self):
        mlm = self.notebook_source("01_continue_pretraining_mlm.ipynb")
        conll_ner = self.notebook_source("02_finetune_conll2003_ner.ipynb")
        conll_baseline = self.notebook_source("03_finetune_conll2003_baselines.ipynb")

        self.assertIn('OUTPUT_DIR = checkpoint_dir("mlm")', mlm)
        self.assertIn('OUTPUT_DIR = checkpoint_dir("conll2003_ner")', conll_ner)
        self.assertIn('OUTPUT_ROOT = checkpoint_dir("baselines")', conll_baseline)
        self.assertNotIn("/content/drive/MyDrive/capitalization_embeddings", mlm)
        self.assertNotIn("/content/drive/MyDrive/capitalization_embeddings", conll_ner)
        self.assertNotIn("/content/drive/MyDrive/capitalization_embeddings", conll_baseline)


if __name__ == "__main__":
    unittest.main()
