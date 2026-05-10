import unittest

import torch

from capitalization_embeddings.trainer import CapitalizedMLMTrainer


class TrainerMetricTests(unittest.TestCase):
    def test_batch_sums_include_capitalization_counts(self):
        trainer = object.__new__(CapitalizedMLMTrainer)
        trainer.args = type("Args", (), {"device": torch.device("cpu")})()

        outputs = type(
            "Outputs",
            (),
            {
                "logits": torch.randn(1, 4, 7),
                "capitalization_logits": torch.tensor(
                    [
                        [
                            [4.0, 0.0, 0.0],
                            [0.0, 4.0, 0.0],
                            [0.0, 0.0, 4.0],
                            [4.0, 0.0, 0.0],
                        ]
                    ],
                ),
            },
        )()
        inputs = {
            "labels": torch.tensor([[1, 2, 3, -100]]),
            "capitalization_labels": torch.tensor([[0, 1, 2, -100]]),
        }

        sums = trainer._capitalization_batch_sums(outputs, inputs)

        self.assertEqual(tuple(sums.shape), (10,))
        self.assertEqual(sums[2].item(), 3.0)
        self.assertEqual(sums[6].item(), 3.0)
        self.assertEqual(sums[7].item(), 1.0)
        self.assertEqual(sums[8].item(), 1.0)
        self.assertEqual(sums[9].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
