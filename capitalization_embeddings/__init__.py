"""Capitalization embedding experiments built on Hugging Face Transformers."""

from .collator import (
    DataCollatorForCapitalizedLanguageModeling,
    DataCollatorForCapitalizedTokenClassification,
)
from .benchmarks import BENCHMARKS, BenchmarkSpec, benchmark_keys, get_benchmark
from .hf_compat import make_trainer, make_training_arguments
from .modeling import (
    CapitalizedBertConfig,
    CapitalizedBertForMaskedLM,
    CapitalizedBertForTokenClassification,
    CapitalizedBertModel,
)
from .runtime_paths import (
    cache_dir,
    checkpoint_dir,
    configure_huggingface_cache,
    project_root,
    repo_root,
    workspace_root,
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
from .trainer import CapitalizedMLMTrainer

__all__ = [
    "ALL_CAPS",
    "BENCHMARKS",
    "BenchmarkSpec",
    "CapitalizedMLMTrainer",
    "FIRST_CAP",
    "NO_CAP",
    "CapitalizedBertConfig",
    "CapitalizedBertForMaskedLM",
    "CapitalizedBertForTokenClassification",
    "CapitalizedBertModel",
    "DataCollatorForCapitalizedLanguageModeling",
    "DataCollatorForCapitalizedTokenClassification",
    "cache_dir",
    "capitalization_ids_from_offsets",
    "capitalization_ids_from_words",
    "benchmark_keys",
    "classify_capitalization",
    "checkpoint_dir",
    "configure_huggingface_cache",
    "get_benchmark",
    "make_trainer",
    "make_training_arguments",
    "project_root",
    "repo_root",
    "tokenize_with_capitalization",
    "workspace_root",
]
