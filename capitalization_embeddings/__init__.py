"""Capitalization embedding experiments built on Hugging Face Transformers."""

from .collator import (
    DataCollatorForCapitalizedLanguageModeling,
    DataCollatorForCapitalizedTokenClassification,
)
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
    "tokenize_with_capitalization",
]
