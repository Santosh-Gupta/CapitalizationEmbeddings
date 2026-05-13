import tempfile
import unittest
from pathlib import Path

import torch

from capitalization_embeddings.modeling import CapitalizedBertConfig, CapitalizedBertForMaskedLM
from scripts.run_mlm_pretraining import restore_overlapping_capitalization_state


def tiny_config(**overrides):
    config = {
        "vocab_size": 101,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 32,
    }
    config.update(overrides)
    return CapitalizedBertConfig(**config)


class MLMPretrainingTests(unittest.TestCase):
    def test_restore_overlapping_capitalization_state_preserves_existing_rows(self):
        source = CapitalizedBertForMaskedLM(tiny_config(capitalization_vocab_size=3))
        target = CapitalizedBertForMaskedLM(tiny_config(capitalization_vocab_size=4))

        with torch.no_grad():
            source.bert.embeddings.capitalization_embeddings.weight.copy_(
                torch.arange(3 * 16, dtype=torch.float32).view(3, 16)
            )
            source.capitalization_classifier.weight.copy_(
                torch.arange(3 * 16, dtype=torch.float32).view(3, 16) + 100
            )
            source.capitalization_classifier.bias.copy_(
                torch.tensor([10.0, 20.0, 30.0])
            )
        original_extra_embedding = (
            target.bert.embeddings.capitalization_embeddings.weight[3].detach().clone()
        )
        original_extra_classifier = (
            target.capitalization_classifier.weight[3].detach().clone()
        )
        original_extra_bias = target.capitalization_classifier.bias[3].detach().clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            source.save_pretrained(checkpoint_dir)
            restore_overlapping_capitalization_state(target, str(checkpoint_dir))

        self.assertTrue(
            torch.equal(
                target.bert.embeddings.capitalization_embeddings.weight[:3],
                source.bert.embeddings.capitalization_embeddings.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                target.capitalization_classifier.weight[:3],
                source.capitalization_classifier.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                target.capitalization_classifier.bias[:3],
                source.capitalization_classifier.bias,
            )
        )
        self.assertTrue(
            torch.equal(
                target.bert.embeddings.capitalization_embeddings.weight[3],
                original_extra_embedding,
            )
        )
        self.assertTrue(
            torch.equal(target.capitalization_classifier.weight[3], original_extra_classifier)
        )
        self.assertTrue(
            torch.equal(target.capitalization_classifier.bias[3], original_extra_bias)
        )


if __name__ == "__main__":
    unittest.main()
