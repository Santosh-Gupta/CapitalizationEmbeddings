import unittest

from capitalization_embeddings import BENCHMARKS, benchmark_keys, get_benchmark


class BenchmarkRegistryTests(unittest.TestCase):
    def test_registry_has_initial_capitalization_sensitive_targets(self):
        keys = benchmark_keys()

        self.assertEqual(keys[0], "conll2003_ner")
        self.assertIn("wnut17_ner", keys)
        self.assertIn("ontonotes5_ner", keys)
        self.assertIn("ptb_pos", keys)
        self.assertIn("conll2003_pos", keys)

    def test_get_benchmark_returns_specs_by_key(self):
        spec = get_benchmark("conll2003_ner")

        self.assertEqual(spec.dataset_name, "lhoestq/conll2003")
        self.assertEqual(spec.task_type, "token_classification")
        self.assertEqual(spec.metric, "seqeval_f1")
        self.assertEqual(spec.status, "implemented")

    def test_priorities_are_unique(self):
        priorities = [spec.priority for spec in BENCHMARKS]

        self.assertEqual(len(priorities), len(set(priorities)))

    def test_unknown_benchmark_names_available_keys(self):
        with self.assertRaisesRegex(KeyError, "conll2003_ner"):
            get_benchmark("missing")


if __name__ == "__main__":
    unittest.main()
