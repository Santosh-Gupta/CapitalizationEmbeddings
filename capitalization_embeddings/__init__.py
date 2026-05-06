"""Capitalization embedding experiments built on Hugging Face Transformers."""

from .collator import (
    DataCollatorForCapitalizedLanguageModeling,
    DataCollatorForCapitalizedTokenClassification,
)
from .hf_compat import make_trainer, make_training_arguments
from .modeling import (
    CapitalizedBertConfig,
    CapitalizedBertForMaskedLM,
    CapitalizedBertForTokenClassification,
    CapitalizedBertModel,
)
from .tokenization import (
    ALL_CAPS,
    FIRST_CAP,
    NO_CAP,
    capitalization_ids_from_offsets,
    capitalization_ids_from_words,
    classify_capitalization,
    tokenize_with_capitalization,
)

__all__ = [
    "ALL_CAPS",
    "FIRST_CAP",
    "NO_CAP",
    "CapitalizedBertConfig",
    "CapitalizedBertForMaskedLM",
    "CapitalizedBertForTokenClassification",
    "CapitalizedBertModel",
    "DataCollatorForCapitalizedLanguageModeling",
    "DataCollatorForCapitalizedTokenClassification",
    "capitalization_ids_from_offsets",
    "capitalization_ids_from_words",
    "classify_capitalization",
    "make_trainer",
    "make_training_arguments",
    "tokenize_with_capitalization",
]
