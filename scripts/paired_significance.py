#!/usr/bin/env python
"""Paired bootstrap significance tests for saved benchmark predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Prediction JSONL for model A.")
    parser.add_argument("--b", required=True, help="Prediction JSONL for model B.")
    parser.add_argument(
        "--metric",
        required=True,
        choices=["accuracy", "macro_f1", "seqeval_f1", "pearson"],
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    records_a = load_predictions(Path(args.a))
    records_b = load_predictions(Path(args.b))
    validate_pairs(records_a, records_b)

    observed_a = score(records_a, args.metric)
    observed_b = score(records_b, args.metric)
    observed_delta = observed_a - observed_b
    deltas = []
    n = len(records_a)
    for _ in range(args.samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_a = [records_a[index] for index in indices]
        sample_b = [records_b[index] for index in indices]
        deltas.append(score(sample_a, args.metric) - score(sample_b, args.metric))

    deltas.sort()
    lower = percentile(deltas, 2.5)
    upper = percentile(deltas, 97.5)
    p_value = 2.0 * min(
        sum(delta <= 0 for delta in deltas) / len(deltas),
        sum(delta >= 0 for delta in deltas) / len(deltas),
    )
    p_value = min(1.0, p_value)
    print(
        json.dumps(
            {
                "a": args.a,
                "b": args.b,
                "metric": args.metric,
                "n": n,
                "samples": args.samples,
                "score_a": observed_a,
                "score_b": observed_b,
                "delta_a_minus_b": observed_delta,
                "bootstrap_ci95": [lower, upper],
                "bootstrap_p_two_sided": p_value,
            },
            indent=2,
            sort_keys=True,
        ),
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_pairs(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]]) -> None:
    if len(records_a) != len(records_b):
        raise ValueError(f"Prediction files have different lengths: {len(records_a)} != {len(records_b)}")
    for offset, (record_a, record_b) in enumerate(zip(records_a, records_b, strict=True)):
        if record_a.get("index") != record_b.get("index"):
            raise ValueError(f"Example index mismatch at row {offset}.")
        labels_a = record_a.get("labels", record_a.get("label"))
        labels_b = record_b.get("labels", record_b.get("label"))
        if labels_a != labels_b:
            raise ValueError(f"Label mismatch at row {offset}.")


def score(records: list[dict[str, Any]], metric: str) -> float:
    if metric == "accuracy":
        return accuracy(records)
    if metric == "macro_f1":
        return macro_f1(records)
    if metric == "seqeval_f1":
        return seqeval_f1(records)
    if metric == "pearson":
        return pearson(records)
    raise ValueError(f"Unsupported metric {metric!r}.")


def accuracy(records: list[dict[str, Any]]) -> float:
    if records and "labels" in records[0] and "predictions" in records[0]:
        correct = 0
        total = 0
        for record in records:
            labels = record["labels"]
            predictions = record["predictions"]
            correct += sum(
                prediction == label
                for prediction, label in zip(predictions, labels, strict=True)
            )
            total += len(labels)
        return correct / total if total else 0.0

    correct = sum(record["prediction"] == record["label"] for record in records)
    return correct / len(records)


def macro_f1(records: list[dict[str, Any]]) -> float:
    labels = sorted({record["label"] for record in records} | {record["prediction"] for record in records})
    f1_values = []
    for label in labels:
        tp = sum(record["label"] == label and record["prediction"] == label for record in records)
        fp = sum(record["label"] != label and record["prediction"] == label for record in records)
        fn = sum(record["label"] == label and record["prediction"] != label for record in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        )
    return sum(f1_values) / len(f1_values)


def seqeval_f1(records: list[dict[str, Any]]) -> float:
    predicted_count = 0
    reference_count = 0
    true_positive_count = 0
    for record in records:
        predicted_entities = bio_entities(record["predictions"])
        reference_entities = bio_entities(record["labels"])
        predicted_count += len(predicted_entities)
        reference_count += len(reference_entities)
        true_positive_count += len(predicted_entities & reference_entities)

    precision = true_positive_count / predicted_count if predicted_count else 0.0
    recall = true_positive_count / reference_count if reference_count else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bio_entities(labels: list[str]) -> set[tuple[str, int, int]]:
    """Return BIO entity spans as ``(type, start, end_exclusive)`` tuples."""

    entities: set[tuple[str, int, int]] = set()
    active_type: str | None = None
    active_start = 0

    for index, label in enumerate([*labels, "O"]):
        prefix, entity_type = split_bio_label(label)
        continuing = (
            prefix == "I"
            and active_type is not None
            and entity_type == active_type
        )
        if active_type is not None and not continuing:
            entities.add((active_type, active_start, index))
            active_type = None
        if prefix == "B" or (prefix == "I" and not continuing):
            active_type = entity_type
            active_start = index

    return entities


def split_bio_label(label: str) -> tuple[str, str]:
    if label == "O" or not label:
        return "O", ""
    if "-" not in label:
        return "B", label
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"}:
        return "B", label
    return prefix, entity_type


def pearson(records: list[dict[str, Any]]) -> float:
    predictions = [float(record["prediction"]) for record in records]
    labels = [float(record["label"]) for record in records]
    mean_prediction = sum(predictions) / len(predictions)
    mean_label = sum(labels) / len(labels)
    numerator = sum(
        (prediction - mean_prediction) * (label - mean_label)
        for prediction, label in zip(predictions, labels, strict=True)
    )
    prediction_ss = sum((prediction - mean_prediction) ** 2 for prediction in predictions)
    label_ss = sum((label - mean_label) ** 2 for label in labels)
    denominator = math.sqrt(prediction_ss * label_ss)
    return numerator / denominator if denominator else float("nan")


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    position = (len(values) - 1) * percent / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[int(position)]
    lower = values[lower_index] * (upper_index - position)
    upper = values[upper_index] * (position - lower_index)
    return lower + upper


if __name__ == "__main__":
    main()
