#!/usr/bin/env python
"""Token-level error analysis grouped by source-word capitalization class."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from capitalization_embeddings import get_benchmark
from capitalization_embeddings.tokenization import (
    ALL_CAPS,
    FIRST_CAP,
    MIXED_CASE,
    NO_CAP,
    classify_capitalization,
)
from run_token_classification_benchmark import ensure_validation_split, load_prepared_benchmark_dataset


CASE_LABELS = {
    NO_CAP: "none/lower",
    FIRST_CAP: "first_cap",
    ALL_CAPS: "all_caps",
    MIXED_CASE: "mixed_case",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--results-file", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=Path(""))
    parser.add_argument("--output-md", type=Path, default=Path(""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_benchmark(args.benchmark)
    if spec.task_type != "token_classification":
        raise ValueError(f"{args.benchmark} is not a token-classification benchmark.")

    rows = load_result_rows(args.results_file)
    datasets_by_seed = {}
    aggregates: dict[str, dict[str, dict[str, int]]] = defaultdict(new_case_counts)
    entity_aggregates: dict[str, dict[str, dict[str, int]]] = defaultdict(new_case_counts)

    for row in rows:
        seed = int(row["seed"])
        model_key = str(row["model_key"])
        if seed not in datasets_by_seed:
            raw = load_prepared_benchmark_dataset(spec, seed=seed)
            datasets_by_seed[seed] = ensure_validation_split(raw, seed=seed)["test"]
        dataset = datasets_by_seed[seed]
        prediction_records = load_predictions(Path(row["prediction_file"]))
        accumulate_predictions(
            dataset=dataset,
            token_column=spec.text_columns[0],
            prediction_records=prediction_records,
            aggregate=aggregates[model_key],
            entity_aggregate=entity_aggregates[model_key],
        )

    report = {
        "benchmark": args.benchmark,
        "results_file": str(args.results_file),
        "token_accuracy_by_case": summarize_aggregates(aggregates),
        "entity_token_accuracy_by_case": summarize_aggregates(entity_aggregates),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    markdown = to_markdown(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
    else:
        print(markdown)


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def new_case_counts() -> dict[str, dict[str, int]]:
    return defaultdict(lambda: {"total": 0, "correct": 0})


def accumulate_predictions(
    *,
    dataset: Any,
    token_column: str,
    prediction_records: list[dict[str, Any]],
    aggregate: dict[str, dict[str, int]],
    entity_aggregate: dict[str, dict[str, int]],
) -> None:
    for record in prediction_records:
        words = dataset[int(record["index"])][token_column]
        predictions = record["predictions"]
        labels = record["labels"]
        if len(words) < len(labels):
            raise ValueError(
                f"Prediction record {record['index']} has {len(labels)} labels "
                f"but dataset row has only {len(words)} words."
            )
        for word, prediction, label in zip(words, predictions, labels, strict=False):
            case_label = CASE_LABELS[
                classify_capitalization(str(word), use_mixed_case=True)
            ]
            is_correct = prediction == label
            aggregate[case_label]["total"] += 1
            aggregate[case_label]["correct"] += int(is_correct)
            if label != "O":
                entity_aggregate[case_label]["total"] += 1
                entity_aggregate[case_label]["correct"] += int(is_correct)


def summarize_aggregates(
    aggregates: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, float]]]:
    report = {}
    for model_key, case_counts in sorted(aggregates.items()):
        report[model_key] = {}
        for case_label, counts in sorted(case_counts.items()):
            total = counts["total"]
            correct = counts["correct"]
            report[model_key][case_label] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Error Analysis By Capitalization Class",
        "",
        f"Benchmark: `{report['benchmark']}`",
        f"Results file: `{report['results_file']}`",
        "",
        "## Token Accuracy",
        "",
        *table_lines(report["token_accuracy_by_case"]),
        "",
        "## Entity-Token Accuracy",
        "",
        *table_lines(report["entity_token_accuracy_by_case"]),
        "",
        "Interpretation note: this is token-level correctness grouped by the "
        "source word's capitalization class, not entity-span F1. It is meant for "
        "error analysis, not as a replacement for the benchmark metric.",
        "",
    ]
    return "\n".join(lines)


def table_lines(section: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines = [
        "| Model | Case class | Accuracy | Correct | Total |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model_key, case_rows in sorted(section.items()):
        for case_label, values in sorted(case_rows.items()):
            lines.append(
                "| {} | {} | {:.4f} | {} | {} |".format(
                    model_key,
                    case_label,
                    values["accuracy"],
                    int(values["correct"]),
                    int(values["total"]),
                )
            )
    return lines


if __name__ == "__main__":
    main()
