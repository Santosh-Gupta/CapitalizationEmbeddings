#!/usr/bin/env python
"""Run matched MLM continued pretraining for baseline and capitalized BERT."""

from __future__ import annotations

import argparse
import json
import numbers
import random
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "uncased": {"kind": "baseline", "model_name": "bert-base-uncased"},
    "cased": {"kind": "baseline", "model_name": "bert-base-cased"},
    "capitalized": {"kind": "capitalized", "model_name": "bert-base-uncased"},
}

CORPUS_CHOICES = (
    "wikitext103",
    "conll2003_train",
    "wnut17_train",
    "ontonotes5_train",
    "ptb_pos_train",
    "capitalization_task_mix",
    "capitalization_task_mix_augmented",
    "capitalization_real_acronym_mix",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--initial-checkpoint",
        default="",
        help="Optional checkpoint to continue MLM pretraining from.",
    )
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument(
        "--corpus",
        choices=CORPUS_CHOICES,
        default="wikitext103",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--capitalization-loss-weight",
        type=float,
        default=0.25,
        help="Scalar multiplier for the auxiliary capitalization loss.",
    )
    parser.add_argument(
        "--capitalization-class-weights",
        default="",
        help=(
            "Optional comma-separated class weights for capitalization labels. "
            "Use three values for none,first-cap,all-caps or four values when "
            "--use-mixed-case-capitalization is enabled."
        ),
    )
    parser.add_argument(
        "--use-mixed-case-capitalization",
        action="store_true",
        help="Use a fourth mixed-case capitalization class for iPhone/eBay-style spans.",
    )
    parser.add_argument(
        "--capitalization-embedding-dropout",
        type=float,
        default=0.0,
        help="Dropout applied to capitalization embeddings during training.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from transformers import set_seed

    from capitalization_embeddings import checkpoint_dir, configure_huggingface_cache

    if args.smoke:
        args.max_steps = min(args.max_steps, 2)
        args.max_train_samples = args.max_train_samples or 64
        args.max_eval_samples = args.max_eval_samples or 32

    configure_huggingface_cache()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(
        args.output_root
        or checkpoint_dir("mlm", args.corpus, args.model_kind)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(output_root, args)

    model_spec = MODEL_SPECS[args.model_kind]
    tokenizer, model = load_model_and_tokenizer(args.model_kind, args.initial_checkpoint, args)
    tokenized_train, tokenized_eval = load_and_tokenize_corpus(args, tokenizer)
    data_collator = make_data_collator(args, tokenizer, model_spec["kind"])
    trainer_cls = trainer_class(model_spec["kind"])

    from capitalization_embeddings import make_training_arguments

    training_args = make_training_arguments(
        output_dir=str(output_root),
        overwrite_output_dir=resume_checkpoint is None,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=25 if not args.smoke else 1,
        eval_steps=250 if not args.smoke else 1,
        save_steps=250 if not args.smoke else 1,
        save_total_limit=2,
        eval_strategy="steps",
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    final_dir = output_root / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    row = {
        "model_kind": args.model_kind,
        "base_model_name": model_spec["model_name"],
        "initial_checkpoint": args.initial_checkpoint,
        "resume_from_checkpoint": resume_checkpoint or "",
        "corpus": args.corpus,
        "output_dir": str(output_root),
        "final_dir": str(final_dir),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_loss": train_result.training_loss,
    }
    if args.model_kind == "capitalized":
        row["capitalization_loss_weight"] = float(args.capitalization_loss_weight)
        row["capitalization_class_weights"] = parse_class_weights(
            args.capitalization_class_weights,
        )
        row["use_mixed_case_capitalization"] = bool(args.use_mixed_case_capitalization)
        row["capitalization_embedding_dropout"] = float(
            args.capitalization_embedding_dropout
        )
    row.update(flatten_metrics(eval_metrics))
    write_json(output_root / "pretraining_metrics.json", row)
    print(json.dumps(row, indent=2, sort_keys=True))


def load_model_and_tokenizer(
    model_kind: str,
    initial_checkpoint: str = "",
    args: argparse.Namespace | None = None,
) -> tuple[Any, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from capitalization_embeddings import CapitalizedBertConfig, CapitalizedBertForMaskedLM

    model_spec = MODEL_SPECS[model_kind]
    tokenizer_name = initial_checkpoint or model_spec["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if model_spec["kind"] == "capitalized":
        config_overrides = capitalization_config_overrides(args)
        if initial_checkpoint:
            config = CapitalizedBertConfig.from_pretrained(
                initial_checkpoint,
                **config_overrides,
            )
            model = CapitalizedBertForMaskedLM.from_pretrained(
                initial_checkpoint,
                config=config,
            )
        else:
            model = CapitalizedBertForMaskedLM.from_uncased_pretrained(
                model_spec["model_name"],
                config_kwargs=config_overrides,
            )
    else:
        model = AutoModelForMaskedLM.from_pretrained(initial_checkpoint or model_spec["model_name"])
    return tokenizer, model


def load_and_tokenize_corpus(args: argparse.Namespace, tokenizer: Any) -> tuple[Any, Any]:
    raw_train, raw_eval, text_column = load_raw_corpus(args.corpus)
    raw_train = filter_text(raw_train, text_column)
    raw_eval = filter_text(raw_eval, text_column)

    if args.max_train_samples and len(raw_train) > args.max_train_samples:
        raw_train = raw_train.select(range(args.max_train_samples))
    if args.max_eval_samples and len(raw_eval) > args.max_eval_samples:
        raw_eval = raw_eval.select(range(args.max_eval_samples))

    capitalized = MODEL_SPECS[args.model_kind]["kind"] == "capitalized"

    def preprocess_batch(examples: dict[str, list[str]]) -> dict[str, Any]:
        if capitalized:
            from capitalization_embeddings import tokenize_with_capitalization

            return tokenize_with_capitalization(
                tokenizer,
                examples[text_column],
                truncation=True,
                max_length=args.max_length,
                use_mixed_case=args.use_mixed_case_capitalization,
            )
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train = raw_train.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_train.column_names,
        desc=f"Tokenizing {args.corpus} train for {args.model_kind}",
    )
    tokenized_eval = raw_eval.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_eval.column_names,
        desc=f"Tokenizing {args.corpus} eval for {args.model_kind}",
    )
    return tokenized_train, tokenized_eval


def load_raw_corpus(corpus: str) -> tuple[Any, Any, str]:
    from datasets import Dataset, load_dataset

    if corpus == "wikitext103":
        train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        eval_dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="validation",
        )
        return train, eval_dataset, "text"

    if corpus == "conll2003_train":
        raw = load_dataset("lhoestq/conll2003")
        return token_rows(raw, "tokens")

    if corpus == "wnut17_train":
        raw = load_dataset("flaitenberger/wnut_17")
        return token_rows(raw, "tokens")

    if corpus == "ontonotes5_train":
        raw = load_dataset("extraordinarylab/ontonotes5")
        return token_rows(raw, "tokens")

    if corpus == "ptb_pos_train":
        raw = load_dataset("batterydata/pos_tagging")
        return token_rows(raw, "words", eval_split="test")

    if corpus in {
        "capitalization_task_mix",
        "capitalization_task_mix_augmented",
        "capitalization_real_acronym_mix",
    }:
        train_rows, eval_rows = task_mix_rows()
        if corpus == "capitalization_task_mix_augmented":
            train_rows = augment_capitalization_rows(train_rows)
        if corpus == "capitalization_real_acronym_mix":
            train_rows, eval_rows = add_real_acronym_rows(train_rows, eval_rows)
        return (
            Dataset.from_dict({"text": train_rows}),
            Dataset.from_dict({"text": eval_rows}),
            "text",
        )

    raise ValueError(f"Unsupported corpus {corpus!r}.")


def token_rows(
    raw: Any,
    token_column: str,
    *,
    train_split: str = "train",
    eval_split: str = "validation",
) -> tuple[Any, Any, str]:
    from datasets import Dataset

    def rows(split: str) -> dict[str, list[str]]:
        return {
            "text": [
                " ".join(tokens)
                for tokens in raw[split][token_column]
                if tokens
            ]
        }

    return Dataset.from_dict(rows(train_split)), Dataset.from_dict(rows(eval_split)), "text"


def task_mix_rows() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    train_rows = []
    eval_rows = []
    for dataset, token_column, eval_split in (
        (load_dataset("lhoestq/conll2003"), "tokens", "validation"),
        (load_dataset("flaitenberger/wnut_17"), "tokens", "validation"),
        (load_dataset("extraordinarylab/ontonotes5"), "tokens", "validation"),
        (load_dataset("batterydata/pos_tagging"), "words", "test"),
    ):
        train, eval_dataset, _ = token_rows(
            dataset,
            token_column,
            eval_split=eval_split,
        )
        train_rows.extend(train["text"])
        eval_rows.extend(eval_dataset["text"])
    return train_rows, eval_rows


def add_real_acronym_rows(
    train_rows: list[str],
    eval_rows: list[str],
) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    real_rows = []
    real_rows.extend(
        text_rows_from_dataset(
            load_dataset("ccdv/pubmed-summarization", split="train[:50000]"),
            ("article", "abstract"),
        ),
    )
    real_rows.extend(
        text_rows_from_dataset(
            load_dataset("billsum", split="train"),
            ("text", "summary", "title"),
        ),
    )
    real_rows.extend(
        text_rows_from_dataset(
            load_dataset("lex_glue", "scotus", split="train"),
            ("text",),
        ),
    )

    acronym_rows = select_acronym_rich_rows(real_rows, max_rows=90000)
    eval_acronym_rows = acronym_rows[:5000]
    train_acronym_rows = acronym_rows[5000:]
    return train_rows + train_acronym_rows, eval_rows + eval_acronym_rows


def text_rows_from_dataset(dataset: Any, columns: tuple[str, ...]) -> list[str]:
    rows = []
    for row in dataset:
        parts = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, str):
                parts.append(value)
        text = " ".join(parts)
        if text:
            rows.extend(chunk_text(text))
    return rows


def chunk_text(text: str, *, words_per_chunk: int = 96) -> list[str]:
    words = text.replace("\n", " ").split()
    return [
        " ".join(words[index : index + words_per_chunk])
        for index in range(0, len(words), words_per_chunk)
        if len(words[index : index + words_per_chunk]) >= 12
    ]


def select_acronym_rich_rows(rows: list[str], *, max_rows: int) -> list[str]:
    scored_rows = [
        (acronym_score(row), index, row)
        for index, row in enumerate(rows)
        if acronym_score(row) > 0
    ]
    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored_rows[:max_rows]]


def acronym_score(text: str) -> int:
    score = 0
    for word in text.split():
        letters = [character for character in word if character.isalpha()]
        if len(letters) < 2:
            continue
        if all(character.isupper() for character in letters):
            score += 3 if len(letters) > 2 else 1
        elif word[:1].isupper() and any(character.isupper() for character in word[1:]):
            score += 1
    return score


def filter_text(dataset: Any, text_column: str) -> Any:
    return dataset.filter(
        lambda row: row[text_column] is not None and row[text_column].strip() != ""
    )


def make_data_collator(args: argparse.Namespace, tokenizer: Any, model_kind: str) -> Any:
    if model_kind == "capitalized":
        from capitalization_embeddings import DataCollatorForCapitalizedLanguageModeling

        return DataCollatorForCapitalizedLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=args.mlm_probability,
        )

    from transformers import DataCollatorForLanguageModeling

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )


def trainer_class(model_kind: str) -> type:
    if model_kind == "capitalized":
        from capitalization_embeddings import CapitalizedMLMTrainer

        return CapitalizedMLMTrainer

    from transformers import Trainer

    return Trainer


def resolve_resume_checkpoint(output_root: Path, args: argparse.Namespace) -> str | None:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    if args.no_auto_resume:
        return None

    from transformers.trainer_utils import get_last_checkpoint

    if output_root.exists():
        return get_last_checkpoint(str(output_root))
    return None


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, numbers.Real)
    }


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capitalization_config_overrides(args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return {}
    return {
        "capitalization_vocab_size": 4 if args.use_mixed_case_capitalization else 3,
        "capitalization_loss_weight": args.capitalization_loss_weight,
        "capitalization_embedding_dropout": args.capitalization_embedding_dropout,
        "capitalization_class_weights": parse_class_weights(
            args.capitalization_class_weights,
        ),
    }


def parse_class_weights(value: str) -> list[float] | None:
    if not value:
        return None
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) not in {3, 4}:
        raise ValueError(
            "--capitalization-class-weights must contain three or four values."
        )
    return weights


def augment_capitalization_rows(rows: list[str]) -> list[str]:
    """Add deterministic case-balanced variants for capitalization pretraining."""

    augmented = list(rows)
    for row in rows:
        words = row.split()
        if not words:
            continue

        if contains_all_caps(words):
            augmented.extend([row, row])

        first_cap_variant = transform_case_words(words, mode="first")
        all_caps_variant = transform_case_words(words, mode="all")
        if first_cap_variant != row:
            augmented.append(first_cap_variant)
        if all_caps_variant != row:
            augmented.append(all_caps_variant)
            augmented.append(all_caps_variant)
    return augmented


def contains_all_caps(words: list[str]) -> bool:
    return any(is_case_word(word) and word.upper() == word and len(word) > 1 for word in words)


def transform_case_words(words: list[str], *, mode: str) -> str:
    transformed = []
    eligible_index = 0
    for word in words:
        if is_case_word(word):
            if mode == "first" and eligible_index % 5 == 0:
                transformed.append(word[:1].upper() + word[1:].lower())
            elif mode == "all" and eligible_index % 7 == 0:
                transformed.append(word.upper())
            else:
                transformed.append(word)
            eligible_index += 1
        else:
            transformed.append(word)
    return " ".join(transformed)


def is_case_word(word: str) -> bool:
    letters = [character for character in word if character.isalpha()]
    return len(letters) >= 2


if __name__ == "__main__":
    main()
