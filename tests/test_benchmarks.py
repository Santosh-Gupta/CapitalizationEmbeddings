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
        self.assertIn("tweet_eval_irony", keys)
        self.assertIn("stsb", keys)
        self.assertIn("tweet_eval_emoji", keys)
        self.assertIn("trec_fine", keys)
        self.assertIn("kaggle_walia_ner", keys)
        self.assertIn("isarcasm_eval_en", keys)
        self.assertIn("citation_sentiment_acl", keys)

    def test_get_benchmark_returns_specs_by_key(self):
        spec = get_benchmark("conll2003_ner")

        self.assertEqual(spec.dataset_name, "lhoestq/conll2003")
        self.assertEqual(spec.task_type, "token_classification")
        self.assertEqual(spec.metric, "seqeval_f1")
        self.assertEqual(spec.status, "implemented")

    def test_wnut_uses_script_free_dataset_mirror(self):
        spec = get_benchmark("wnut17_ner")

        self.assertEqual(spec.dataset_name, "flaitenberger/wnut_17")

    def test_ontonotes_uses_script_free_dataset_mirror(self):
        spec = get_benchmark("ontonotes5_ner")

        self.assertEqual(spec.dataset_name, "extraordinarylab/ontonotes5")
        self.assertEqual(spec.label_column, "ner_tags")

    def test_uncased_favored_sequence_benchmarks_are_registered(self):
        spec = get_benchmark("tweet_eval_irony")

        self.assertEqual(spec.task_type, "sequence_classification")
        self.assertEqual(spec.dataset_name, "tweet_eval")
        self.assertEqual(spec.dataset_config, "irony")
        self.assertEqual(spec.metric, "macro_f1")

        stsb = get_benchmark("stsb")
        self.assertEqual(stsb.task_type, "sequence_regression")
        self.assertEqual(stsb.text_columns, ("sentence1", "sentence2"))

    def test_scientific_relation_benchmarks_are_registered(self):
        semeval = get_benchmark("semeval2018_task7")

        self.assertEqual(semeval.dataset_name, "DFKI-SLT/SemEval2018_Task7")
        self.assertEqual(semeval.dataset_config, "Subtask_1_1")
        self.assertEqual(semeval.processor, "semeval2018_task7_relations")

        combined = get_benchmark("scientific_relations_combined")
        self.assertEqual(combined.processor, "combined_scientific_relations")
        self.assertEqual(combined.metric, "accuracy")

    def test_scientbank_variants_are_registered(self):
        three_way = get_benchmark("scientbank_3way_uq")

        self.assertEqual(three_way.dataset_name, "nkazi/SciEntsBank")
        self.assertEqual(three_way.processor, "scientbank_3way_uq")
        self.assertEqual(three_way.metric, "macro_f1")

        five_way = get_benchmark("scientbank_5way_ud")
        self.assertEqual(five_way.processor, "scientbank_5way_ud")
        self.assertEqual(five_way.metric, "accuracy")

    def test_extra_cased_favored_sequence_benchmarks_are_registered(self):
        emoji = get_benchmark("tweet_eval_emoji")

        self.assertEqual(emoji.dataset_name, "tweet_eval")
        self.assertEqual(emoji.dataset_config, "emoji")
        self.assertEqual(emoji.metric, "accuracy")

        trec = get_benchmark("trec_fine")
        self.assertEqual(trec.dataset_name, "lukasgarbas/trec")
        self.assertEqual(trec.label_column, "fine_label")

        isarcasm = get_benchmark("isarcasm_eval_en")
        self.assertEqual(isarcasm.processor, "isarcasm_eval_en_task_a")
        self.assertEqual(isarcasm.metric, "macro_f1")

        citation = get_benchmark("citation_sentiment_acl")
        self.assertEqual(citation.processor, "citation_sentiment_acl")
        self.assertEqual(citation.metric, "macro_f1")

    def test_extra_cased_favored_token_benchmarks_are_registered(self):
        walia = get_benchmark("kaggle_walia_ner")

        self.assertEqual(walia.dataset_name, "rjac/kaggle-entity-annotated-corpus-ner-dataset")
        self.assertEqual(walia.processor, "single_train_token_split")
        self.assertEqual(walia.metric, "seqeval_f1")

    def test_priorities_are_unique(self):
        priorities = [spec.priority for spec in BENCHMARKS]

        self.assertEqual(len(priorities), len(set(priorities)))

    def test_unknown_benchmark_names_available_keys(self):
        with self.assertRaisesRegex(KeyError, "conll2003_ner"):
            get_benchmark("missing")


if __name__ == "__main__":
    unittest.main()
