#!/usr/bin/env python
"""Build paper-ready result tables from generated report JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARIES = (
    "reports/final_token_20seed_ner_bootstrap_1000.json",
    "reports/final_sequence_20seed_macro_f1_bootstrap_1000.json",
    "reports/final_sequence_20seed_accuracy_bootstrap_1000.json",
    "reports/added_cased_favored_5seed_walia_bootstrap_1000.json",
    "reports/added_cased_favored_5seed_isarcasm_bootstrap_1000.json",
)

DEFAULT_HOLM = (
    "reports/final_token_20seed_holm.json",
    "reports/final_sequence_20seed_holm.json",
    "reports/added_cased_favored_5seed_holm_margin005.json",
)

BENCHMARK_LABELS = {
    "conll2003_ner": "CoNLL-2003 NER",
    "wnut17_ner": "WNUT-17 NER",
    "tweet_eval_irony": "TweetEval Irony",
    "tweet_eval_offensive": "TweetEval Offensive",
    "sst5": "SST-5",
    "twenty_newsgroups": "20 Newsgroups",
    "kaggle_walia_ner": "Kaggle/Walia NER",
    "isarcasm_eval_en": "iSarcasmEval EN",
}

METRIC_LABELS = {
    "conll2003_ner": "entity F1",
    "wnut17_ner": "entity F1",
    "kaggle_walia_ner": "entity F1",
    "tweet_eval_irony": "macro-F1",
    "tweet_eval_offensive": "macro-F1",
    "isarcasm_eval_en": "macro-F1",
    "sst5": "accuracy",
    "twenty_newsgroups": "accuracy",
}

BENCHMARK_FAMILY = {
    "conll2003_ner": "token/entity",
    "wnut17_ner": "token/entity",
    "kaggle_walia_ner": "token/entity",
    "tweet_eval_irony": "uncased-favored control",
    "tweet_eval_offensive": "uncased-favored control",
    "sst5": "uncased-favored control",
    "twenty_newsgroups": "uncased-favored control",
    "isarcasm_eval_en": "appendix/neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="*", default=list(DEFAULT_SUMMARIES))
    parser.add_argument("--holm", nargs="*", default=list(DEFAULT_HOLM))
    parser.add_argument("--output-json", type=Path, default=Path("reports/paper_tables.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/paper_tables.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holm_labels = load_holm_labels([Path(path) for path in args.holm])
    rows = []
    for path in [Path(value) for value in args.summaries]:
        rows.extend(load_summary_rows(path, holm_labels))

    rows = sorted(rows, key=sort_key)
    report = {"rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(report))
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


def load_holm_labels(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    labels = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for hypothesis in data["hypotheses"]:
            labels[(hypothesis["benchmark"], hypothesis["comparison"])] = hypothesis
    return labels


def load_summary_rows(path: Path, holm_labels: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for benchmark in data["benchmarks"]:
        key = benchmark["benchmark"]
        models = benchmark["models"]
        cap_uncased = comparison_for(benchmark, "capitalized_pretrained", "uncased_pretrained")
        cap_cased = comparison_for(benchmark, "capitalized_pretrained", "cased_pretrained")
        rows.append(
            {
                "benchmark": key,
                "label": BENCHMARK_LABELS.get(key, key),
                "family": BENCHMARK_FAMILY.get(key, "other"),
                "metric": METRIC_LABELS.get(key, data["metric"]),
                "uncased": models.get("uncased_pretrained"),
                "cased": models.get("cased_pretrained"),
                "capitalized": models.get("capitalized_pretrained"),
                "cap_minus_uncased": cap_uncased["seed_delta_mean"] if cap_uncased else None,
                "cap_minus_cased": cap_cased["seed_delta_mean"] if cap_cased else None,
                "cap_uncased_ci95": cap_uncased["bootstrap_ci95"] if cap_uncased else None,
                "cap_cased_ci95": cap_cased["bootstrap_ci95"] if cap_cased else None,
                "cap_uncased_label": label_for(holm_labels, key, "capitalized_pretrained>uncased_pretrained"),
                "cap_cased_label": label_for(holm_labels, key, "capitalized_pretrained>cased_pretrained"),
            }
        )
    return rows


def comparison_for(benchmark: dict[str, Any], a: str, b: str) -> dict[str, Any] | None:
    for comparison in benchmark["comparisons"]:
        if comparison["a"] == a and comparison["b"] == b:
            return comparison
    return None


def label_for(
    holm_labels: dict[tuple[str, str], dict[str, Any]],
    benchmark: str,
    comparison: str,
) -> str:
    record = holm_labels.get((benchmark, comparison))
    if record is None:
        return ""
    return str(record["label"])


def sort_key(row: dict[str, Any]) -> tuple[int, str]:
    family_order = {
        "token/entity": 0,
        "uncased-favored control": 1,
        "appendix/neutral": 2,
    }
    return (family_order.get(row["family"], 99), row["benchmark"])


def format_mean_std(model_summary: dict[str, float] | None) -> str:
    if model_summary is None:
        return ""
    return f"{model_summary['mean']:.4f} +/- {model_summary['std']:.4f}, n={model_summary['n']}"


def format_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def format_ci(value: list[float] | None) -> str:
    if value is None:
        return ""
    return f"[{value[0]:+.4f}, {value[1]:+.4f}]"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Result Tables",
        "",
        "Generated from report JSON files. Do not hand-edit numeric values here.",
        "",
        "## Main Results",
        "",
        "| Family | Benchmark | Metric | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased | Labels |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        labels = f"vs uncased: {row['cap_uncased_label']}; vs cased: {row['cap_cased_label']}"
        lines.append(
            "| {family} | {label} | {metric} | {uncased} | {cased} | {capitalized} | "
            "{cap_uncased} | {cap_cased} | {labels} |".format(
                family=row["family"],
                label=row["label"],
                metric=row["metric"],
                uncased=format_mean_std(row["uncased"]),
                cased=format_mean_std(row["cased"]),
                capitalized=format_mean_std(row["capitalized"]),
                cap_uncased=format_delta(row["cap_minus_uncased"]),
                cap_cased=format_delta(row["cap_minus_cased"]),
                labels=labels,
            )
        )

    lines.extend(
        [
            "",
            "## Cap-Embedding Comparison Intervals",
            "",
            "| Benchmark | Cap-Uncased delta | Cap-Uncased CI95 | Cap-Cased delta | Cap-Cased CI95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {label} | {cap_uncased} | {cap_uncased_ci} | {cap_cased} | {cap_cased_ci} |".format(
                label=row["label"],
                cap_uncased=format_delta(row["cap_minus_uncased"]),
                cap_uncased_ci=format_ci(row["cap_uncased_ci95"]),
                cap_cased=format_delta(row["cap_minus_cased"]),
                cap_cased_ci=format_ci(row["cap_cased_ci95"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
