#!/usr/bin/env python
"""Run matched MLM continued pretraining for baseline and capitalized BERT."""

from __future__ import annotations

import argparse
import gzip
import json
import numbers
import os
import random
from pathlib import Path
from typing import Any

import torch


MODEL_SPECS = {
    "uncased": {"kind": "baseline", "model_name": "bert-base-uncased"},
    "cased": {"kind": "baseline", "model_name": "bert-base-cased"},
    "capitalized": {"kind": "capitalized", "model_name": "bert-base-uncased"},
}
CAPITALIZATION_STATE_KEYS = (
    "bert.embeddings.capitalization_embeddings.weight",
    "capitalization_classifier.weight",
    "capitalization_classifier.bias",
)

CORPUS_CHOICES = (
    "wikitext103",
    "conll2003_train",
    "wnut17_train",
    "ontonotes5_train",
    "ptb_pos_train",
    "capitalization_task_mix",
    "capitalization_task_mix_augmented",
    "capitalization_real_acronym_mix",
    "capitalization_domain_mix_v2",
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
                ignore_mismatched_sizes=True,
            )
            restore_overlapping_capitalization_state(model, initial_checkpoint)
        else:
            model = CapitalizedBertForMaskedLM.from_uncased_pretrained(
                model_spec["model_name"],
                config_kwargs=config_overrides,
            )
    else:
        model = AutoModelForMaskedLM.from_pretrained(initial_checkpoint or model_spec["model_name"])
    return tokenizer, model


def restore_overlapping_capitalization_state(model: Any, checkpoint: str) -> None:
    state = load_checkpoint_tensors(checkpoint, set(CAPITALIZATION_STATE_KEYS))
    if not state:
        return

    parameters = dict(model.named_parameters())
    restored: dict[str, list[int]] = {}
    with torch.no_grad():
        for key in CAPITALIZATION_STATE_KEYS:
            source = state.get(key)
            target = parameters.get(key)
            if source is None or target is None or source.ndim != target.ndim:
                continue
            common_shape = tuple(
                min(source_dimension, target_dimension)
                for source_dimension, target_dimension in zip(source.shape, target.shape)
            )
            slices = tuple(slice(0, dimension) for dimension in common_shape)
            target.data[slices].copy_(
                source[slices].to(device=target.device, dtype=target.dtype)
            )
            restored[key] = list(common_shape)

    if restored:
        print(
            "Restored overlapping capitalization checkpoint tensors: "
            + json.dumps(restored, sort_keys=True),
            flush=True,
        )


def load_checkpoint_tensors(checkpoint: str, keys: set[str]) -> dict[str, torch.Tensor]:
    path = Path(checkpoint)
    if not path.exists():
        return {}

    if path.is_file():
        return load_checkpoint_file(path, keys)

    simple_files = (
        path / "model.safetensors",
        path / "pytorch_model.bin",
    )
    for checkpoint_file in simple_files:
        if checkpoint_file.exists():
            return load_checkpoint_file(checkpoint_file, keys)

    indexed_files = (
        (path / "model.safetensors.index.json", True),
        (path / "pytorch_model.bin.index.json", False),
    )
    for index_path, is_safetensors in indexed_files:
        if index_path.exists():
            return load_indexed_checkpoint_tensors(
                index_path,
                keys,
                is_safetensors=is_safetensors,
            )

    return {}


def load_indexed_checkpoint_tensors(
    index_path: Path,
    keys: set[str],
    *,
    is_safetensors: bool,
) -> dict[str, torch.Tensor]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    shard_names = sorted({weight_map[key] for key in keys if key in weight_map})

    tensors: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = load_checkpoint_file(
            index_path.parent / shard_name,
            {key for key in keys if weight_map.get(key) == shard_name},
            is_safetensors=is_safetensors,
        )
        tensors.update(shard)
    return tensors


def load_checkpoint_file(
    checkpoint_file: Path,
    keys: set[str],
    *,
    is_safetensors: bool | None = None,
) -> dict[str, torch.Tensor]:
    if is_safetensors is None:
        is_safetensors = checkpoint_file.suffix == ".safetensors"

    if is_safetensors:
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_file), device="cpu")
    else:
        state = torch.load(checkpoint_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        return {}
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if key in keys and isinstance(value, torch.Tensor)
    }


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
        "capitalization_domain_mix_v2",
    }:
        train_rows, eval_rows = task_mix_rows()
        if corpus == "capitalization_task_mix_augmented":
            train_rows = augment_capitalization_rows(train_rows)
        if corpus == "capitalization_real_acronym_mix":
            train_rows, eval_rows = add_real_acronym_rows(train_rows, eval_rows)
        if corpus == "capitalization_domain_mix_v2":
            train_rows, eval_rows = add_domain_mix_v2_rows(train_rows, eval_rows)
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
    cached = load_cached_real_acronym_rows()
    if cached is not None:
        train_acronym_rows, eval_acronym_rows = cached
        return train_rows + train_acronym_rows, eval_rows + eval_acronym_rows

    from datasets import load_dataset

    real_rows = []
    print("Building real-acronym corpus rows from source datasets.", flush=True)
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
    write_cached_real_acronym_rows(train_acronym_rows, eval_acronym_rows)
    return train_rows + train_acronym_rows, eval_rows + eval_acronym_rows


def add_domain_mix_v2_rows(
    train_rows: list[str],
    eval_rows: list[str],
) -> tuple[list[str], list[str]]:
    """Add larger case-rich scientific, legal, and noisy-social text rows.

    This corpus is intended for matched continued pretraining across uncased,
    cased, and capitalized models. It deliberately uses only unlabeled text and
    source training splits for benchmark-like datasets.
    """

    cached = load_cached_domain_mix_v2_rows()
    if cached is not None:
        train_domain_rows, eval_domain_rows = cached
        return train_rows + train_domain_rows, eval_rows + eval_domain_rows

    domain_rows = build_domain_mix_v2_rows()
    selected_rows = select_case_rich_rows(domain_rows, max_rows=180000, min_score=2)
    eval_count = min(10000, max(1, len(selected_rows) // 20)) if selected_rows else 0
    eval_domain_rows = selected_rows[:eval_count]
    train_domain_rows = selected_rows[eval_count:]
    write_cached_domain_mix_v2_rows(train_domain_rows, eval_domain_rows)
    return train_rows + train_domain_rows, eval_rows + eval_domain_rows


def build_domain_mix_v2_rows() -> list[str]:
    from datasets import load_dataset

    rows: list[str] = []
    print("Building capitalization_domain_mix_v2 rows from source datasets.", flush=True)

    rows.extend(
        safe_text_rows_from_dataset(
            "pubmed-summarization",
            lambda: load_dataset("ccdv/pubmed-summarization", split="train[:100000]"),
            ("article", "abstract"),
        )
    )
    rows.extend(
        safe_text_rows_from_dataset(
            "billsum",
            lambda: load_dataset("billsum", split="train"),
            ("text", "summary", "title"),
        )
    )
    rows.extend(
        safe_text_rows_from_dataset(
            "lex_glue/scotus",
            lambda: load_dataset("lex_glue", "scotus", split="train"),
            ("text",),
        )
    )
    rows.extend(
        safe_text_rows_from_dataset(
            "scierc-relations",
            lambda: load_dataset("nsusemiehl/SciERC", split="train"),
            ("text",),
        )
    )
    rows.extend(
        safe_text_rows_from_dataset(
            "scientbank",
            lambda: load_dataset("nkazi/SciEntsBank", split="train"),
            ("question", "reference_answer", "student_answer"),
        )
    )

    for config in ("emoji", "irony", "offensive", "sentiment", "emotion"):
        rows.extend(
            safe_text_rows_from_dataset(
                f"tweet_eval/{config}",
                lambda config=config: load_dataset("tweet_eval", config, split="train"),
                ("text",),
            )
        )

    rows.extend(
        safe_text_rows_from_dataset(
            "trec",
            lambda: load_dataset("lukasgarbas/trec", split="train"),
            ("text",),
        )
    )
    rows.extend(
        safe_text_rows_from_dataset(
            "sst5",
            lambda: load_dataset("SetFit/sst5", split="train"),
            ("text",),
        )
    )
    rows.extend(
        safe_text_rows_from_dataset(
            "20_newsgroups",
            lambda: load_dataset("SetFit/20_newsgroups", split="train"),
            ("text",),
        )
    )
    rows.extend(load_citation_sentiment_text_rows())

    print(f"Built capitalization_domain_mix_v2 candidate rows: {len(rows)}", flush=True)
    return rows


def safe_text_rows_from_dataset(
    source_name: str,
    load_fn: Any,
    columns: tuple[str, ...],
) -> list[str]:
    try:
        dataset = load_fn()
    except Exception as error:
        print(f"Skipping {source_name}: failed to load dataset: {error}", flush=True)
        return []

    rows = text_rows_from_dataset(dataset, columns)
    print(f"Prepared {source_name} case-positive chunks: {len(rows)}", flush=True)
    return rows


def load_citation_sentiment_text_rows() -> list[str]:
    try:
        import pandas as pd
        from datasets import Dataset
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "gaof23/citation_sentiment_corpus",
            "citation_sentiment_corpus.csv",
            repo_type="dataset",
        )
        frame = pd.read_csv(path)
        if "Citation_Text" not in frame:
            print("Skipping citation_sentiment_acl: missing Citation_Text column.", flush=True)
            return []
        dataset = Dataset.from_dict({"text": frame["Citation_Text"].fillna("").astype(str).tolist()})
        return safe_text_rows_from_dataset(
            "citation_sentiment_acl",
            lambda: dataset,
            ("text",),
        )
    except Exception as error:
        print(f"Skipping citation_sentiment_acl: {error}", flush=True)
        return []


def real_acronym_cache_path() -> Path:
    work_root = os.environ.get("CAP_EMB_WORK_ROOT")
    if work_root:
        root = Path(work_root)
    elif Path("/workspace/capitalization_embeddings").exists():
        root = Path("/workspace/capitalization_embeddings")
    else:
        root = Path(".cache") / "capitalization_embeddings"
    return root / "prepared_corpora" / "real_acronym_rows_v1.jsonl.gz"


def domain_mix_v2_cache_path() -> Path:
    work_root = os.environ.get("CAP_EMB_WORK_ROOT")
    if work_root:
        root = Path(work_root)
    elif Path("/workspace/capitalization_embeddings").exists():
        root = Path("/workspace/capitalization_embeddings")
    else:
        root = Path(".cache") / "capitalization_embeddings"
    return root / "prepared_corpora" / "domain_mix_v2_rows.jsonl.gz"


def load_cached_real_acronym_rows() -> tuple[list[str], list[str]] | None:
    return load_cached_text_rows(real_acronym_cache_path(), "real-acronym")


def load_cached_domain_mix_v2_rows() -> tuple[list[str], list[str]] | None:
    return load_cached_text_rows(domain_mix_v2_cache_path(), "domain-mix-v2")


def load_cached_text_rows(
    path: Path,
    cache_name: str,
) -> tuple[list[str], list[str]] | None:
    if not path.exists():
        return None

    train_rows = []
    eval_rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") == "train":
                train_rows.append(str(record["text"]))
            elif record.get("split") == "eval":
                eval_rows.append(str(record["text"]))
    print(
        f"Loaded cached {cache_name} rows from {path}: "
        f"train={len(train_rows)} eval={len(eval_rows)}",
        flush=True,
    )
    return train_rows, eval_rows


def write_cached_real_acronym_rows(
    train_rows: list[str],
    eval_rows: list[str],
) -> None:
    write_cached_text_rows(
        real_acronym_cache_path(),
        train_rows,
        eval_rows,
        "real-acronym",
    )


def write_cached_domain_mix_v2_rows(
    train_rows: list[str],
    eval_rows: list[str],
) -> None:
    write_cached_text_rows(
        domain_mix_v2_cache_path(),
        train_rows,
        eval_rows,
        "domain-mix-v2",
    )


def write_cached_text_rows(
    path: Path,
    train_rows: list[str],
    eval_rows: list[str],
    cache_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps({"split": "train", "text": row}) + "\n")
        for row in eval_rows:
            handle.write(json.dumps({"split": "eval", "text": row}) + "\n")
    temporary_path.replace(path)
    print(
        f"Cached {cache_name} rows to {path}: "
        f"train={len(train_rows)} eval={len(eval_rows)}",
        flush=True,
    )


def text_rows_from_dataset(dataset: Any, columns: tuple[str, ...]) -> list[str]:
    try:
        workers = max(1, min(8, os.cpu_count() or 1))

        def chunk_batch(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            batch_size = max(
                (len(examples.get(column, [])) for column in columns),
                default=0,
            )
            chunked_rows = []
            for row_index in range(batch_size):
                parts = []
                for column in columns:
                    values = examples.get(column, [])
                    value = values[row_index] if row_index < len(values) else None
                    text_value = text_from_value(value)
                    if text_value:
                        parts.append(text_value)
                text = " ".join(parts)
                row_chunks = []
                if text:
                    for chunk in chunk_text(text):
                        if case_signal_score(chunk) > 0:
                            row_chunks.append(chunk)
                chunked_rows.append(row_chunks)
            return {"case_chunks": chunked_rows}

        mapped = dataset.map(
            chunk_batch,
            batched=True,
            batch_size=64,
            remove_columns=dataset.column_names,
            num_proc=workers if workers > 1 else None,
            desc="Chunking acronym source text",
        )
        rows = [
            chunk
            for chunked_row in mapped["case_chunks"]
            for chunk in chunked_row
        ]
        print(
            "Prepared case-positive chunks from source dataset: "
            f"{len(rows)}",
            flush=True,
        )
        return rows
    except Exception as error:
        print(
            "Parallel text chunking failed; falling back to row iteration: "
            f"{error}",
            flush=True,
        )

    rows = []
    for row in dataset:
        parts = []
        for column in columns:
            value = row.get(column)
            text_value = text_from_value(value)
            if text_value:
                parts.append(text_value)
        text = " ".join(parts)
        if text:
            for chunk in chunk_text(text):
                if case_signal_score(chunk) > 0:
                    rows.append(chunk)
    return rows


def text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple)):
        return " ".join(text for item in value if (text := text_from_value(item)))
    if isinstance(value, dict):
        return " ".join(
            text
            for key in sorted(value)
            if (text := text_from_value(value[key]))
        )
    return ""


def chunk_text(text: str, *, words_per_chunk: int = 96) -> list[str]:
    words = text.replace("\n", " ").split()
    return [
        " ".join(words[index : index + words_per_chunk])
        for index in range(0, len(words), words_per_chunk)
        if len(words[index : index + words_per_chunk]) >= 12
    ]


def select_acronym_rich_rows(rows: list[str], *, max_rows: int) -> list[str]:
    scored_rows = []
    for index, row in enumerate(rows):
        score = acronym_score(row)
        if score > 0:
            scored_rows.append((score, index, row))
    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored_rows[:max_rows]]


def select_case_rich_rows(
    rows: list[str],
    *,
    max_rows: int,
    min_score: int = 1,
) -> list[str]:
    scored_rows = []
    seen = set()
    for index, row in enumerate(rows):
        normalized = " ".join(row.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        score = case_signal_score(normalized)
        if score >= min_score:
            scored_rows.append((score, index, normalized))
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


def case_signal_score(text: str) -> int:
    score = 0
    for word in text.split():
        letters = [character for character in word if character.isalpha()]
        if len(letters) < 2:
            continue
        if all(character.isupper() for character in letters):
            score += 4 if len(letters) > 2 else 1
        elif all(character.islower() for character in letters):
            continue
        elif word[:1].isupper() and all(character.islower() for character in letters[1:]):
            score += 1
        elif any(character.isupper() for character in letters[1:]):
            score += 3
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
