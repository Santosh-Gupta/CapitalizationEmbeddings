import unittest

import torch

from capitalization_embeddings.modeling import (
    CapitalizedBertConfig,
    CapitalizedBertForMaskedLM,
    CapitalizedBertModel,
    CapitalizedBertForSequenceClassification,
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
    def test_masked_lm_tied_weight_metadata_is_mapping(self):
        self.assertIsInstance(CapitalizedBertForMaskedLM._tied_weights_keys, dict)
        self.assertEqual(
            CapitalizedBertForMaskedLM._tied_weights_keys[
                "cls.predictions.decoder.weight"
            ],
            "bert.embeddings.word_embeddings.weight",
        )

    def test_base_model_has_local_head_mask_compatibility(self):
        model = CapitalizedBertModel(tiny_config(), add_pooling_layer=False)

        no_mask = model.get_head_mask(None, model.config.num_hidden_layers)
        self.assertEqual(no_mask, [None])

        one_dimensional_mask = torch.ones(model.config.num_attention_heads)
        expanded = model.get_head_mask(
            one_dimensional_mask,
            model.config.num_hidden_layers,
        )

        self.assertEqual(tuple(expanded.shape), (1, 1, 2, 1, 1))

    def test_forward_does_not_depend_on_position_or_token_type_buffers(self):
        model = CapitalizedBertModel(tiny_config(), add_pooling_layer=False)
        model.embeddings.position_ids.fill_(999_999)
        model.embeddings.token_type_ids.fill_(999_999)

        input_ids = torch.tensor([[2, 10, 11, 12, 3]])
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            capitalization_ids=torch.zeros_like(input_ids),
        )

        self.assertEqual(tuple(outputs.last_hidden_state.shape), (1, 5, 16))

    def test_explicit_bad_position_ids_raise_clear_error(self):
        model = CapitalizedBertModel(tiny_config(), add_pooling_layer=False)
        input_ids = torch.tensor([[2, 10, 11, 12, 3]])
        position_ids = torch.tensor([[0, 1, 2, 3, 999_999]])

        with self.assertRaisesRegex(ValueError, "position_ids contains indices"):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                capitalization_ids=torch.zeros_like(input_ids),
                position_ids=position_ids,
            )

    def test_masked_lm_forward_accepts_capitalization_ids(self):
        model = CapitalizedBertForMaskedLM(tiny_config(capitalization_vocab_size=4))
        input_ids = torch.tensor([[2, 10, 11, 12, 3]])
        capitalization_ids = torch.tensor([[0, 1, 3, 2, 0]])
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
        self.assertEqual(tuple(outputs.capitalization_logits.shape), (1, 5, 4))
        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.token_loss)
        self.assertIsNotNone(outputs.capitalization_loss)

    def test_config_keeps_capitalization_embedding_dropout(self):
        config = tiny_config(capitalization_embedding_dropout=0.2)
        model = CapitalizedBertModel(config, add_pooling_layer=False)

        self.assertEqual(model.config.capitalization_embedding_dropout, 0.2)

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

    def test_sequence_classifier_forward_accepts_capitalization_ids(self):
        model = CapitalizedBertForSequenceClassification(tiny_config())
        input_ids = torch.tensor([[2, 10, 11, 12, 3]])
        capitalization_ids = torch.tensor([[0, 1, 0, 2, 0]])
        labels = torch.tensor([1])

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            capitalization_ids=capitalization_ids,
            labels=labels,
        )

        self.assertEqual(tuple(outputs.logits.shape), (1, 5))
        self.assertIsNotNone(outputs.loss)


if __name__ == "__main__":
    unittest.main()
