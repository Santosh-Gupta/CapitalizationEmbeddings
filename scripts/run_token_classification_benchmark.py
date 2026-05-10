#!/usr/bin/env python
"""Run matched token-classification benchmarks for capitalization experiments."""

from __future__ import annotations

import argparse
import csv
import json
import numbers
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "uncased": {"kind": "baseline", "model_name": "bert-base-uncased"},
    "cased": {"kind": "baseline", "model_name": "bert-base-cased"},
    "capitalized": {"kind": "capitalized", "model_name": "bert-base-uncased"},
    "capitalized_pretrained": {
        "kind": "capitalized",
        "model_name": "bert-base-uncased",
        "requires_checkpoint": True,
    },
}

FALLBACK_LABELS = {
    ("lhoestq/conll2003", "ner_tags"): [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="conll2003_ner")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["uncased", "cased", "capitalized"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument(
        "--capitalized-checkpoint",
        default="",
        help="Optional continued-pretraining checkpoint for the capitalized model.",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--results-file", default="")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from transformers import set_seed

    from capitalization_embeddings import (
        checkpoint_dir,
        configure_huggingface_cache,
        get_benchmark,
    )

    if args.smoke:
        args.epochs = min(args.epochs, 0.05)
        args.max_train_samples = args.max_train_samples or 64
        args.max_eval_samples = args.max_eval_samples or 32
        args.max_test_samples = args.max_test_samples or 32

    configure_huggingface_cache()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    spec = get_benchmark(args.benchmark)
    if spec.task_type != "token_classification":
        raise ValueError(f"{args.benchmark} is not a token-classification benchmark.")

    output_root = Path(args.output_root or checkpoint_dir("benchmarks", args.benchmark))
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = Path(args.results_file or output_root / "results.jsonl")
    csv_file = results_file.with_suffix(".csv")

    raw = load_benchmark_dataset(spec.dataset_name, spec.dataset_config)
    raw = maybe_select_samples(
        raw,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
    )
    label_list = label_names(raw, spec.dataset_name, spec.label_column)
    id2label = {index: label for index, label in enumerate(label_list)}
    label2id = {label: index for index, label in id2label.items()}

    print(f"benchmark: {args.benchmark}")
    print(f"labels: {label_list}")
    print(f"output_root: {output_root}")

    rows = []
    for model_key in args.models:
        row = run_one_model(
            model_key=model_key,
            args=args,
            raw=raw,
            label_list=label_list,
            id2label=id2label,
            label2id=label2id,
            output_root=output_root,
        )
        row.update(
            {
                "benchmark": args.benchmark,
                "benchmark_spec": asdict(spec),
                "seed": args.seed,
                "max_train_samples": args.max_train_samples,
                "max_eval_samples": args.max_eval_samples,
                "max_test_samples": args.max_test_samples,
            }
        )
        append_jsonl(results_file, row)
        rows.append(row)

    write_csv(csv_file, rows)
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"wrote: {results_file}")
    print(f"wrote: {csv_file}")


def load_benchmark_dataset(dataset_name: str, dataset_config: str | None) -> DatasetDict:
    from datasets import load_dataset

    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def maybe_select_samples(
    raw: DatasetDict,
    *,
    max_train_samples: int,
    max_eval_samples: int,
    max_test_samples: int,
) -> DatasetDict:
    from datasets import DatasetDict

    limits = {
        "train": max_train_samples,
        "validation": max_eval_samples,
        "test": max_test_samples,
    }
    selected = {}
    for split, dataset in raw.items():
        limit = limits.get(split, 0)
        if limit and len(dataset) > limit:
            selected[split] = dataset.select(range(limit))
        else:
            selected[split] = dataset
    return DatasetDict(selected)


def label_names(raw: DatasetDict, dataset_name: str, label_column: str) -> list[str]:
    label_feature = raw["train"].features[label_column].feature
    if hasattr(label_feature, "names"):
        return list(label_feature.names)
    fallback = FALLBACK_LABELS.get((dataset_name, label_column))
    if fallback is not None:
        return fallback
    raise ValueError(f"Dataset label column {label_column!r} does not expose label names.")


def run_one_model(
    *,
    model_key: str,
    args: argparse.Namespace,
    raw: DatasetDict,
    label_list: list[str],
    id2label: dict[int, str],
    label2id: dict[str, int],
    output_root: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from capitalization_embeddings import (
        DataCollatorForCapitalizedTokenClassification,
        get_benchmark,
        make_trainer,
        make_training_arguments,
    )

    from transformers import DataCollatorForTokenClassification

    spec = get_benchmark(args.benchmark)
    model_spec = MODEL_SPECS[model_key]
    model_name = model_spec["model_name"]
    is_capitalized = model_spec["kind"] == "capitalized"
    if model_spec.get("requires_checkpoint") and not args.capitalized_checkpoint:
        raise ValueError(f"{model_key} requires --capitalized-checkpoint.")
    checkpoint = (
        args.capitalized_checkpoint
        if model_spec.get("requires_checkpoint")
        else ""
    )
    tokenizer_name = checkpoint or model_name
    output_dir = output_root / model_key

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenized = raw.map(
        lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
            label_column=spec.label_column,
            max_length=args.max_length,
            capitalized=is_capitalized,
        ),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc=f"Tokenizing {model_key}",
    )

    if is_capitalized:
        model = load_capitalized_model(
            checkpoint=checkpoint,
            base_model_name=model_name,
            id2label=id2label,
            label2id=label2id,
        )
        data_collator = DataCollatorForCapitalizedTokenClassification(tokenizer=tokenizer)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    metric = metric_fn(label_list, spec.metric)
    training_args = make_training_arguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=20 if args.smoke else 50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric["main_metric"],
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = make_trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=metric["compute_metrics"],
    )

    train_result = trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    row = {
        "model_key": model_key,
        "model_name": model_name,
        "capitalized_checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "train_loss": train_result.training_loss,
    }
    row.update(flatten_metrics(test_metrics))
    return row


def tokenize_and_align_labels(
    *,
    examples: dict[str, list[Any]],
    tokenizer: Any,
    label_column: str,
    max_length: int,
    capitalized: bool,
) -> dict[str, Any]:
    from capitalization_embeddings import tokenize_with_capitalization

    if capitalized:
        tokenized = tokenize_with_capitalization(
            tokenizer,
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
    else:
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

    aligned_labels = []
    for batch_index, word_labels in enumerate(examples[label_column]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def load_capitalized_model(
    *,
    checkpoint: str,
    base_model_name: str,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> CapitalizedBertForTokenClassification:
    from capitalization_embeddings import (
        CapitalizedBertConfig,
        CapitalizedBertForTokenClassification,
    )

    config_kwargs = {
        "num_labels": len(id2label),
        "id2label": id2label,
        "label2id": label2id,
    }
    if checkpoint:
        config = CapitalizedBertConfig.from_pretrained(checkpoint, **config_kwargs)
        return CapitalizedBertForTokenClassification.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
        )
    return CapitalizedBertForTokenClassification.from_uncased_pretrained(
        base_model_name,
        config_kwargs=config_kwargs,
        ignore_mismatched_sizes=True,
    )


def metric_fn(label_list: list[str], metric_name: str) -> dict[str, Any]:
    if metric_name == "seqeval_f1":
        import evaluate
        import numpy as np

        seqeval = evaluate.load("seqeval")

        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            true_predictions = [
                [label_list[pred] for pred, label in zip(prediction, label_row) if label != -100]
                for prediction, label_row in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[label] for pred, label in zip(prediction, label_row) if label != -100]
                for prediction, label_row in zip(predictions, labels)
            ]
            results = seqeval.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        return {"main_metric": "f1", "compute_metrics": compute_metrics}

    if metric_name == "accuracy":
        import evaluate
        import numpy as np

        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            mask = labels != -100
            results = accuracy.compute(predictions=predictions[mask], references=labels[mask])
            return {"accuracy": results["accuracy"]}

        return {"main_metric": "accuracy", "compute_metrics": compute_metrics}

    raise ValueError(f"Unsupported metric {metric_name!r}.")


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, numbers.Real)
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    metric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (numbers.Real, str))
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in metric_keys})


if __name__ == "__main__":
    main()
