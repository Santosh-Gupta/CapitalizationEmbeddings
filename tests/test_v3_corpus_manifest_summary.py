import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_v3_corpus_manifest.py"
)


def load_summary_module():
    spec = importlib.util.spec_from_file_location("v3_corpus_summary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V3CorpusManifestSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary_module()

    def test_summarize_rows_counts_sources_buckets_and_means(self):
        rows = [
            {
                "corpus": "capitalization_v3_general",
                "split": "train",
                "source": "wikitext103",
                "selected_bucket": "ordinary",
                "row_length_words": 10,
                "case_signal_score": 0,
                "first_cap_count": 0,
                "all_caps_count": 0,
                "mixed_case_count": 0,
                "lowercase_count": 10,
            },
            {
                "corpus": "capitalization_v3_general",
                "split": "eval",
                "source": "pubmed",
                "selected_bucket": "all_caps_rich",
                "row_length_words": 20,
                "case_signal_score": 8,
                "first_cap_count": 1,
                "all_caps_count": 2,
                "mixed_case_count": 0,
                "lowercase_count": 15,
            },
        ]

        summary = self.summary.summarize_rows(rows)

        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["by_bucket"]["ordinary"], 1)
        self.assertEqual(summary["by_bucket"]["all_caps_rich"], 1)
        self.assertEqual(summary["by_split"], {"eval": 1, "train": 1})
        self.assertEqual(summary["means"]["row_length_words"], 15)

    def test_load_manifest_rows_reads_gzip_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"corpus": "x"}) + "\n")

            rows = self.summary.load_manifest_rows(path)

        self.assertEqual(rows, [{"corpus": "x"}])

    def test_render_markdown_contains_tables(self):
        markdown = self.summary.render_markdown(
            {
                "row_count": 1,
                "by_corpus": {"x": 1},
                "by_split": {"train": 1},
                "by_bucket": {"ordinary": 1},
                "top_sources": {"source": 1},
                "means": {"row_length_words": 12.0},
            }
        )

        self.assertIn("# V3 Corpus Manifest Summary", markdown)
        self.assertIn("| `ordinary` | 1 |", markdown)


if __name__ == "__main__":
    unittest.main()
