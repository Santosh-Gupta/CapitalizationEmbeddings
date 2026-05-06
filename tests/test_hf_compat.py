import unittest
from unittest.mock import patch

from capitalization_embeddings.hf_compat import make_trainer, make_training_arguments


class HFCompatTests(unittest.TestCase):
    def test_make_training_arguments_uses_new_eval_strategy_name(self):
        class FakeTrainingArguments:
            def __init__(self, *, output_dir, eval_strategy=None):
                self.output_dir = output_dir
                self.eval_strategy = eval_strategy

        with patch("capitalization_embeddings.hf_compat.TrainingArguments", FakeTrainingArguments):
            arguments = make_training_arguments(
                output_dir="/tmp/test",
                eval_strategy="steps",
            )

        self.assertEqual(arguments.output_dir, "/tmp/test")
        self.assertEqual(arguments.eval_strategy, "steps")

    def test_make_training_arguments_uses_old_evaluation_strategy_name(self):
        class FakeTrainingArguments:
            def __init__(self, *, output_dir, evaluation_strategy=None):
                self.output_dir = output_dir
                self.evaluation_strategy = evaluation_strategy

        with patch("capitalization_embeddings.hf_compat.TrainingArguments", FakeTrainingArguments):
            arguments = make_training_arguments(
                output_dir="/tmp/test",
                eval_strategy="steps",
            )

        self.assertEqual(arguments.output_dir, "/tmp/test")
        self.assertEqual(arguments.evaluation_strategy, "steps")

    def test_make_trainer_routes_processing_class_to_new_name(self):
        class FakeTrainer:
            def __init__(self, *, model, args, processing_class=None):
                self.model = model
                self.args = args
                self.processing_class = processing_class

        with patch("capitalization_embeddings.hf_compat.Trainer", FakeTrainer):
            trainer = make_trainer(
                model="model",
                args="args",
                processing_class="tokenizer",
            )

        self.assertEqual(trainer.model, "model")
        self.assertEqual(trainer.args, "args")
        self.assertEqual(trainer.processing_class, "tokenizer")

    def test_make_trainer_routes_processing_class_to_old_tokenizer_name(self):
        class FakeTrainer:
            def __init__(self, *, model, args, tokenizer=None):
                self.model = model
                self.args = args
                self.tokenizer = tokenizer

        with patch("capitalization_embeddings.hf_compat.Trainer", FakeTrainer):
            trainer = make_trainer(
                model="model",
                args="args",
                processing_class="tokenizer",
            )

        self.assertEqual(trainer.model, "model")
        self.assertEqual(trainer.args, "args")
        self.assertEqual(trainer.tokenizer, "tokenizer")

    def test_real_training_arguments_smoke(self):
        arguments = make_training_arguments(
            output_dir="/tmp/capitalization_embeddings_test",
            eval_strategy="no",
        )
        self.assertEqual(
            str(arguments.output_dir),
            "/tmp/capitalization_embeddings_test",
        )


if __name__ == "__main__":
    unittest.main()
