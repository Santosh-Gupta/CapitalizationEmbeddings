import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import BertTokenizerFast

from capitalization_embeddings import (
    DataCollatorForCapitalizedLanguageModeling,
    DataCollatorForCapitalizedSequenceClassification,
    DataCollatorForCapitalizedTokenClassification,
)


def tiny_tokenizer():
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "tom",
        "met",
        "nasa",
    ]
    directory = tempfile.TemporaryDirectory()
    vocab_path = Path(directory.name) / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer._test_tmpdir = directory
    return tokenizer


class CollatorTests(unittest.TestCase):
    def test_mlm_collator_pads_capitalization_ids(self):
        tokenizer = tiny_tokenizer()
        collator = DataCollatorForCapitalizedLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.50,
        )
        features = [
            {
                "input_ids": [2, 5, 6, 3],
                "attention_mask": [1, 1, 1, 1],
                "special_tokens_mask": [1, 0, 0, 1],
                "capitalization_ids": [0, 1, 0, 0],
            },
            {
                "input_ids": [2, 7, 3],
                "attention_mask": [1, 1, 1],
                "special_tokens_mask": [1, 0, 1],
                "capitalization_ids": [0, 2, 0],
            },
        ]

        torch.manual_seed(0)
        batch = collator(features)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(tuple(batch["capitalization_ids"].shape), (2, 4))
        self.assertEqual(tuple(batch["capitalization_labels"].shape), (2, 4))
        self.assertEqual(batch["capitalization_ids"][1, -1].item(), 0)

    def test_mlm_random_replacements_use_base_vocab_size(self):
        tokenizer = tiny_tokenizer()
        tokenizer.add_tokens(["extra_added_token"])
        collator = DataCollatorForCapitalizedLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=1.0,
        )
        features = [
            {
                "input_ids": [2, 5, 6, 3],
                "attention_mask": [1, 1, 1, 1],
                "special_tokens_mask": [1, 0, 0, 1],
                "capitalization_ids": [0, 1, 0, 0],
            },
        ]

        with patch("torch.randint", return_value=torch.zeros((1, 4), dtype=torch.long)) as randint:
            torch.manual_seed(0)
            collator(features)

        self.assertEqual(randint.call_args.args[0], tokenizer.vocab_size)
        self.assertGreater(len(tokenizer), tokenizer.vocab_size)

    def test_token_classification_collator_pads_capitalization_ids(self):
        tokenizer = tiny_tokenizer()
        collator = DataCollatorForCapitalizedTokenClassification(tokenizer=tokenizer)
        features = [
            {
                "input_ids": [2, 5, 6, 3],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, 1, 0, -100],
                "capitalization_ids": [0, 1, 0, 0],
            },
            {
                "input_ids": [2, 7, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, -100],
                "capitalization_ids": [0, 2, 0],
            },
        ]

        batch = collator(features)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(tuple(batch["labels"].shape), (2, 4))
        self.assertEqual(tuple(batch["capitalization_ids"].shape), (2, 4))
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["capitalization_ids"][1, -1].item(), 0)

    def test_sequence_classification_collator_pads_capitalization_ids(self):
        tokenizer = tiny_tokenizer()
        collator = DataCollatorForCapitalizedSequenceClassification(tokenizer=tokenizer)
        features = [
            {
                "input_ids": [2, 5, 6, 3],
                "attention_mask": [1, 1, 1, 1],
                "labels": 1,
                "capitalization_ids": [0, 1, 0, 0],
            },
            {
                "input_ids": [2, 7, 3],
                "attention_mask": [1, 1, 1],
                "labels": 0,
                "capitalization_ids": [0, 2, 0],
            },
        ]

        batch = collator(features)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(tuple(batch["labels"].shape), (2,))
        self.assertEqual(tuple(batch["capitalization_ids"].shape), (2, 4))
        self.assertEqual(batch["capitalization_ids"][1, -1].item(), 0)


if __name__ == "__main__":
    unittest.main()
