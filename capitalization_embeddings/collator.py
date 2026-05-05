"""Data collators for capitalization-aware pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCapitalizedLanguageModeling:
    """MLM collator that also emits capitalization prediction labels.

    Selected MLM positions have their input `capitalization_ids` zeroed so the
    model must predict case from context instead of reading the answer.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __post_init__(self) -> None:
        if self.tokenizer.mask_token is None:
            raise ValueError("This collator requires a tokenizer with a mask token.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        input_ids, labels = self.torch_mask_tokens(batch["input_ids"], special_tokens_mask)
        batch["input_ids"] = input_ids
        batch["labels"] = labels

        capitalization_labels = batch["capitalization_ids"].clone()
        masked_positions = labels != -100
        capitalization_labels[~masked_positions] = -100
        batch["capitalization_labels"] = capitalization_labels

        batch["capitalization_ids"] = batch["capitalization_ids"].masked_fill(
            masked_positions,
            0,
        )

        return batch

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare masked tokens inputs/labels for masked language modeling."""

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(value, already_has_special_tokens=True)
                for value in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs = inputs.clone()
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token,
        )

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long,
            device=inputs.device,
        )
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
