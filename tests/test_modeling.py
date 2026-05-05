import unittest

import torch

from capitalization_embeddings.modeling import (
    CapitalizedBertConfig,
    CapitalizedBertForMaskedLM,
    CapitalizedBertForTokenClassification,
)


def tiny_config(**overrides):
    config = {
        "vocab_size": 101,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 32,
        "num_labels": 5,
    }
    config.update(overrides)
    return CapitalizedBertConfig(**config)


class ModelingTests(unittest.TestCase):
    def test_masked_lm_forward_accepts_capitalization_ids(self):
        model = CapitalizedBertForMaskedLM(tiny_config())
        input_ids = torch.tensor([[2, 10, 11, 12, 3]])
        capitalization_ids = torch.tensor([[0, 1, 0, 2, 0]])
        labels = torch.tensor([[-100, 10, -100, 12, -100]])
        capitalization_labels = torch.tensor([[-100, 1, -100, 2, -100]])

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            capitalization_ids=capitalization_ids,
            labels=labels,
            capitalization_labels=capitalization_labels,
        )

        self.assertEqual(tuple(outputs.logits.shape), (1, 5, 101))
        self.assertEqual(tuple(outputs.capitalization_logits.shape), (1, 5, 3))
        self.assertIsNotNone(outputs.loss)

    def test_token_classifier_forward_accepts_capitalization_ids(self):
        model = CapitalizedBertForTokenClassification(tiny_config())
        input_ids = torch.tensor([[2, 10, 11, 12, 3]])
        capitalization_ids = torch.tensor([[0, 1, 0, 2, 0]])
        labels = torch.tensor([[-100, 1, 0, 2, -100]])

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            capitalization_ids=capitalization_ids,
            labels=labels,
        )

        self.assertEqual(tuple(outputs.logits.shape), (1, 5, 5))
        self.assertIsNotNone(outputs.loss)


if __name__ == "__main__":
    unittest.main()
