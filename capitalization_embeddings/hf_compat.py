"""Compatibility helpers for Hugging Face API changes."""

from __future__ import annotations

import inspect
from typing import Any

from transformers import Trainer, TrainingArguments


def make_training_arguments(
    *,
    eval_strategy: str | None = None,
    **kwargs: Any,
) -> TrainingArguments:
    """Create `TrainingArguments` across Transformers API versions.

    Transformers renamed `evaluation_strategy` to `eval_strategy`. Colab images
    can lag or jump versions, so notebooks route through this helper instead of
    hard-coding either spelling.
    """

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if eval_strategy is not None:
        if "eval_strategy" in parameters:
            kwargs["eval_strategy"] = eval_strategy
        elif "evaluation_strategy" in parameters:
            kwargs["evaluation_strategy"] = eval_strategy
        else:
            raise TypeError("TrainingArguments supports neither eval_strategy nor evaluation_strategy.")

    return TrainingArguments(**kwargs)


def make_trainer(
    *,
    processing_class: Any | None = None,
    **kwargs: Any,
) -> Trainer:
    """Create `Trainer` across tokenizer/processing_class API versions."""

    parameters = inspect.signature(Trainer.__init__).parameters
    if processing_class is not None:
        if "processing_class" in parameters:
            kwargs["processing_class"] = processing_class
        elif "tokenizer" in parameters:
            kwargs["tokenizer"] = processing_class
        else:
            raise TypeError("Trainer supports neither processing_class nor tokenizer.")

    return Trainer(**kwargs)
