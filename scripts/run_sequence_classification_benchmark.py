#!/usr/bin/env python
"""Run matched sequence-classification benchmarks for capitalization experiments."""

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
    "uncased_pretrained": {
        "kind": "baseline",
        "model_name": "bert-base-uncased",
        "checkpoint_arg": "uncased_checkpoint",
    },
    "cased": {"kind": "baseline", "model_name": "bert-base-cased"},
    "cased_pretrained": {
        "kind": "baseline",
        "model_name": "bert-base-cased",
        "checkpoint_arg": "cased_checkpoint",
    },
    "capitalized": {"kind": "capitalized", "model_name": "bert-base-uncased"},
    "capitalized_pretrained": {
        "kind": "capitalized",
        "model_name": "bert-base-uncased",
        "checkpoint_arg": "capitalized_checkpoint",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="tweet_eval_irony")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["uncased", "cased", "capitalized_pretrained"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument("--capitalized-checkpoint", default="")
    parser.add_argument("--uncased-checkpoint", default="")
    parser.add_argument("--cased-checkpoint", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--results-file", default="")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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
    if spec.task_type not in {"sequence_classification", "sequence_regression"}:
        raise ValueError(f"{args.benchmark} is not a sequence benchmark.")

    output_root = Path(args.output_root or checkpoint_dir("benchmarks", args.benchmark))
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = Path(args.results_file or output_root / "sequence_results.jsonl")
    csv_file = results_file.with_suffix(".csv")

    raw = load_prepared_benchmark_dataset(spec, seed=args.seed)
    raw = ensure_validation_split(raw, seed=args.seed)
    raw = maybe_select_samples(
        raw,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
    )
    labels = labels_for_dataset(raw, spec.label_column, regression=spec.task_type == "sequence_regression")
    label_to_id = None if labels is None else {label: index for index, label in enumerate(labels)}

    print(f"benchmark: {args.benchmark}")
    print(f"output_root: {output_root}")
    if labels is not None:
        print(f"labels: {labels}")

    rows = []
    for model_key in args.models:
        row = run_one_model(
            model_key=model_key,
            args=args,
            raw=raw,
            labels=labels,
            label_to_id=label_to_id,
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


def load_benchmark_dataset(dataset_name: str, dataset_config: str | None):
    from datasets import load_dataset

    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def load_prepared_benchmark_dataset(spec: Any, *, seed: int):
    if spec.processor == "combined_scientific_relations":
        return load_combined_scientific_relations(seed=seed)

    raw = load_benchmark_dataset(spec.dataset_name, spec.dataset_config)
    if spec.processor == "semeval2018_task7_relations":
        return prepare_semeval2018_task7(raw)
    return raw


def prepare_semeval2018_task7(raw):
    from datasets import Dataset, DatasetDict

    prepared = {}
    for split, dataset in raw.items():
        rows = []
        for example in dataset:
            rows.extend(semeval2018_relation_examples(example))
        prepared[split] = Dataset.from_list(rows)
    return DatasetDict(prepared)


def semeval2018_relation_examples(example: dict[str, Any]) -> list[dict[str, Any]]:
    abstract = example.get("abstract") or example.get("text") or ""
    text = abstract
    title = example.get("title")
    if title:
        text = f"{title} [SEP] {abstract}"

    entities = {
        entity["id"]: entity_text_from_offsets(abstract, entity)
        for entity in example.get("entities", [])
        if isinstance(entity, dict) and entity.get("id")
    }

    relations = example.get("relations") or []
    rows = []
    for relation in relations:
        label = relation.get("relation")
        if label is None:
            label = relation.get("label")
        if label is None:
            continue
        marked_text = mark_relation_text(
            text,
            resolve_relation_entity(entities, relation.get("entity_1") or relation.get("e1") or relation.get("arg1")),
            resolve_relation_entity(entities, relation.get("entity_2") or relation.get("e2") or relation.get("arg2")),
        )
        rows.append({"text": marked_text, "label": str(label)})
    return rows


def entity_text_from_offsets(text: str, entity: dict[str, Any]) -> str:
    start = entity.get("char_start")
    end = entity.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        return text[start:end]
    return entity_text(entity)


def resolve_relation_entity(entities: dict[str, str], reference: Any) -> str:
    if isinstance(reference, str) and reference in entities:
        return entities[reference]
    return entity_text(reference)


def mark_relation_text(text: str, entity_1: Any, entity_2: Any) -> str:
    entity_1_text = entity_text(entity_1)
    entity_2_text = entity_text(entity_2)
    if entity_1_text and entity_2_text:
        return f"[E1] {entity_1_text} [/E1] [E2] {entity_2_text} [/E2] {text}"
    return text


def entity_text(entity: Any) -> str:
    if isinstance(entity, str):
        return entity
    if isinstance(entity, dict):
        for key in ("text", "term", "surface", "entity", "mention"):
            value = entity.get(key)
            if value:
                return str(value)
    return ""


def load_combined_scientific_relations(*, seed: int):
    from datasets import DatasetDict, concatenate_datasets, load_dataset

    semeval = prepare_semeval2018_task7(load_dataset("DFKI-SLT/SemEval2018_Task7", "Subtask_1_1"))
    scierc = load_dataset("nsusemiehl/SciERC")
    scierc = DatasetDict(
        {
            split: dataset.select_columns(["text", "label"]).map(
                lambda examples: {"label": [str(label) for label in examples["label"]]},
                batched=True,
            )
            for split, dataset in scierc.items()
        }
    )

    semeval = ensure_validation_split(semeval, seed=seed)
    scierc = ensure_validation_split(scierc, seed=seed)
    splits = {}
    for split in ("train", "validation", "test"):
        datasets = [dataset[split] for dataset in (semeval, scierc) if split in dataset]
        if datasets:
            splits[split] = concatenate_datasets(datasets).shuffle(seed=seed)
    return DatasetDict(splits)


def ensure_validation_split(raw, seed: int):
    if "validation" in raw:
        return raw

    from datasets import DatasetDict

    split = raw["train"].train_test_split(test_size=0.1, seed=seed)
    datasets = {key: value for key, value in raw.items() if key != "train"}
    datasets["train"] = split["train"]
    datasets["validation"] = split["test"]
    return DatasetDict(datasets)


def maybe_select_samples(
    raw,
    *,
    max_train_samples: int,
    max_eval_samples: int,
    max_test_samples: int,
):
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


def labels_for_dataset(raw, label_column: str, *, regression: bool) -> list[str] | None:
    if regression:
        return None
    label_feature = raw["train"].features[label_column]
    if hasattr(label_feature, "names"):
        return list(label_feature.names)
    labels = sorted({row[label_column] for split in raw.values() for row in split})
    return [str(label) for label in labels]


def run_one_model(
    *,
    model_key: str,
    args: argparse.Namespace,
    raw,
    labels: list[str] | None,
    label_to_id: dict[str, int] | None,
    output_root: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from capitalization_embeddings import (
        DataCollatorForCapitalizedSequenceClassification,
        get_benchmark,
        make_trainer,
        make_training_arguments,
    )
    from transformers import DataCollatorWithPadding

    spec = get_benchmark(args.benchmark)
    model_spec = MODEL_SPECS[model_key]
    model_name = model_spec["model_name"]
    is_capitalized = model_spec["kind"] == "capitalized"
    checkpoint = checkpoint_for_model(model_key, model_spec, args)
    tokenizer_name = checkpoint or model_name
    output_dir = output_root / model_key / f"seed_{args.seed}"
    is_regression = spec.task_type == "sequence_regression"
    num_labels = 1 if is_regression else len(labels or [])
    id2label = None if is_regression else {index: label for index, label in enumerate(labels or [])}
    label2id = None if is_regression else {label: index for index, label in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenized = raw.map(
        lambda examples: tokenize_examples(
            examples=examples,
            tokenizer=tokenizer,
            text_columns=spec.text_columns,
            label_column=spec.label_column,
            label_to_id=label_to_id,
            max_length=args.max_length,
            capitalized=is_capitalized,
            regression=is_regression,
        ),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc=f"Tokenizing {model_key}",
    )

    if is_capitalized:
        model = load_capitalized_model(
            checkpoint=checkpoint,
            base_model_name=model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="regression" if is_regression else "single_label_classification",
        )
        data_collator = DataCollatorForCapitalizedSequenceClassification(tokenizer=tokenizer)
    else:
        model_kwargs = {
            "num_labels": num_labels,
            "problem_type": "regression" if is_regression else "single_label_classification",
            "ignore_mismatched_sizes": True,
        }
        if id2label is not None and label2id is not None:
            model_kwargs["id2label"] = id2label
            model_kwargs["label2id"] = label2id
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint or model_name,
            **model_kwargs,
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = metric_fn(spec.metric, regression=is_regression)
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
    eval_split = labeled_evaluation_split(raw, spec.label_column, regression=is_regression)
    prediction_output = trainer.predict(tokenized[eval_split], metric_key_prefix="test")
    test_metrics = prediction_output.metrics
    prediction_file = prediction_path(args, output_root, model_key)
    save_sequence_predictions(
        path=prediction_file,
        model_key=model_key,
        benchmark=args.benchmark,
        evaluation_split=eval_split,
        predictions=prediction_output.predictions,
        labels=prediction_output.label_ids,
        regression=is_regression,
    )
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    row = {
        "model_key": model_key,
        "model_name": model_name,
        "pretraining_checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "train_loss": train_result.training_loss,
        "evaluation_split": eval_split,
        "prediction_file": str(prediction_file),
    }
    row.update(flatten_metrics(test_metrics))
    return row


def prediction_path(args: argparse.Namespace, output_root: Path, model_key: str) -> Path:
    if args.results_file:
        results_file = Path(args.results_file)
        return results_file.with_name(
            f"{results_file.stem}_{model_key}_seed_{args.seed}_predictions.jsonl",
        )
    return output_root / model_key / f"seed_{args.seed}" / "predictions.jsonl"


def save_sequence_predictions(
    *,
    path: Path,
    model_key: str,
    benchmark: str,
    evaluation_split: str,
    predictions: Any,
    labels: Any,
    regression: bool,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(predictions)
    label_array = np.asarray(labels)
    if regression:
        predicted_values = logits.squeeze().tolist()
        label_values = label_array.squeeze().tolist()
    else:
        predicted_values = np.argmax(logits, axis=-1).tolist()
        label_values = label_array.astype(int).tolist()

    with path.open("w", encoding="utf-8") as handle:
        for index, (prediction, label) in enumerate(zip(predicted_values, label_values, strict=True)):
            handle.write(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "model_key": model_key,
                        "evaluation_split": evaluation_split,
                        "index": index,
                        "prediction": prediction,
                        "label": label,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def labeled_evaluation_split(raw, label_column: str, *, regression: bool) -> str:
    if "test" not in raw:
        return "validation"
    labels = raw["test"][label_column]
    if not labels:
        return "validation"
    if regression and all(float(label) < 0 for label in labels):
        return "validation"
    if not regression and all(int(label) < 0 for label in labels):
        return "validation"
    return "test"


def checkpoint_for_model(
    model_key: str,
    model_spec: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    checkpoint_arg = model_spec.get("checkpoint_arg")
    if checkpoint_arg is None:
        return ""

    checkpoint = getattr(args, checkpoint_arg)
    if not checkpoint:
        cli_arg = checkpoint_arg.replace("_", "-")
        raise ValueError(f"{model_key} requires --{cli_arg}.")
    return checkpoint


def tokenize_examples(
    *,
    examples: dict[str, list[Any]],
    tokenizer: Any,
    text_columns: tuple[str, ...],
    label_column: str,
    label_to_id: dict[str, int] | None,
    max_length: int,
    capitalized: bool,
    regression: bool,
) -> dict[str, Any]:
    from capitalization_embeddings import tokenize_with_capitalization

    texts = merge_text_columns(examples, text_columns, tokenizer.sep_token or "[SEP]")
    if capitalized:
        tokenized = tokenize_with_capitalization(
            tokenizer,
            texts,
            truncation=True,
            max_length=max_length,
        )
    else:
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
        )
    if regression:
        tokenized["labels"] = [float(label) for label in examples[label_column]]
    else:
        tokenized["labels"] = [
            label_to_id[str(label)] if label_to_id is not None else int(label)
            for label in examples[label_column]
        ]
    return tokenized


def merge_text_columns(
    examples: dict[str, list[Any]],
    text_columns: tuple[str, ...],
    separator: str,
) -> list[str]:
    row_count = len(examples[text_columns[0]])
    texts = []
    for index in range(row_count):
        parts = []
        for column in text_columns:
            value = examples[column][index]
            if value is None:
                continue
            parts.append(str(value))
        texts.append(f" {separator} ".join(parts))
    return texts


def load_capitalized_model(
    *,
    checkpoint: str,
    base_model_name: str,
    num_labels: int,
    id2label: dict[int, str] | None,
    label2id: dict[str, int] | None,
    problem_type: str,
):
    from capitalization_embeddings import (
        CapitalizedBertConfig,
        CapitalizedBertForSequenceClassification,
    )

    config_kwargs = {
        "num_labels": num_labels,
        "problem_type": problem_type,
    }
    if id2label is not None and label2id is not None:
        config_kwargs["id2label"] = id2label
        config_kwargs["label2id"] = label2id

    if checkpoint:
        config = CapitalizedBertConfig.from_pretrained(checkpoint, **config_kwargs)
        return CapitalizedBertForSequenceClassification.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
        )
    return CapitalizedBertForSequenceClassification.from_uncased_pretrained(
        base_model_name,
        config_kwargs=config_kwargs,
        ignore_mismatched_sizes=True,
    )


def metric_fn(metric_name: str, *, regression: bool) -> dict[str, Any]:
    import evaluate
    import numpy as np

    if regression:
        pearson = evaluate.load("pearsonr")
        spearman = evaluate.load("spearmanr")

        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = logits.squeeze()
            return {
                "pearson": pearson.compute(predictions=predictions, references=labels)["pearsonr"],
                "spearman": spearman.compute(predictions=predictions, references=labels)["spearmanr"],
            }

        return {"main_metric": metric_name, "compute_metrics": compute_metrics}

    if metric_name == "accuracy":
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return accuracy.compute(predictions=predictions, references=labels)

        return {"main_metric": "accuracy", "compute_metrics": compute_metrics}

    if metric_name == "macro_f1":
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return f1_metric.compute(
                predictions=predictions,
                references=labels,
                average="macro",
            )

        return {"main_metric": "f1", "compute_metrics": compute_metrics}

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
