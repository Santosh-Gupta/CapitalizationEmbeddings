#!/usr/bin/env python
"""Summarize multi-seed benchmark sweeps and optional paired significance."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from paired_significance import (
    load_predictions,
    score,
    validate_pairs,
    percentile,
)


MODEL_ORDER = ("uncased_pretrained", "cased_pretrained", "capitalized_pretrained")
BASELINE_MODELS = ("uncased_pretrained", "cased_pretrained")
COMPARISONS = (
    ("capitalized_pretrained", "uncased_pretrained"),
    ("capitalized_pretrained", "cased_pretrained"),
    ("cased_pretrained", "uncased_pretrained"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument(
        "--baseline-results-roots",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional sweep roots to read uncased_pretrained/cased_pretrained "
            "rows from when the primary root contains only a capitalized variant."
        ),
    )
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument(
        "--metric",
        choices=["accuracy", "macro_f1", "seqeval_f1", "pearson"],
        required=True,
    )
    parser.add_argument("--output-json", type=Path, default=Path(""))
    parser.add_argument("--output-md", type=Path, default=Path(""))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "results_root": str(args.results_root),
        "metric": args.metric,
        "benchmarks": [
            summarize_benchmark(
                [args.results_root, *args.baseline_results_roots],
                benchmark,
                metric=args.metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for benchmark in args.benchmarks
        ],
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


def summarize_benchmark(
    results_roots: list[Path],
    benchmark: str,
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = load_rows(results_roots, benchmark)
    values = defaultdict(list)
    prediction_files = defaultdict(dict)

    for row in rows:
        model_key = row["model_key"]
        values[model_key].append(metric_value(row, metric))
        prediction_files[model_key][int(row["seed"])] = row["prediction_file"]

    model_summaries = {
        model_key: summarize_values(values[model_key])
        for model_key in MODEL_ORDER
        if values[model_key]
    }
    comparisons = []
    for model_a, model_b in COMPARISONS:
        if model_a not in model_summaries or model_b not in model_summaries:
            continue
        comparisons.append(
            summarize_comparison(
                model_a,
                model_b,
                prediction_files,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )

    return {
        "benchmark": benchmark,
        "results_roots": [str(root) for root in results_roots],
        "rows": len(rows),
        "models": model_summaries,
        "comparisons": comparisons,
    }


def load_rows(results_roots: list[Path], benchmark: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for root_index, results_root in enumerate(results_roots):
        results_file = results_root / benchmark / "results.jsonl"
        if not results_file.exists():
            continue
        for line in results_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model_key = row.get("model_key")
            if root_index > 0 and model_key not in BASELINE_MODELS:
                continue
            key = (model_key, row.get("seed"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        roots = ", ".join(str(root) for root in results_roots)
        raise FileNotFoundError(f"No rows found for benchmark {benchmark!r} in {roots}.")
    return rows


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "seqeval_f1":
        return float(row["test_f1"])
    if metric == "macro_f1":
        return float(row["test_f1"])
    if metric == "accuracy":
        return float(row["test_accuracy"])
    if metric == "pearson":
        return float(row["test_pearson"])
    raise ValueError(f"Unsupported metric {metric!r}.")


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_comparison(
    model_a: str,
    model_b: str,
    prediction_files: dict[str, dict[int, str]],
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    shared_seeds = sorted(set(prediction_files[model_a]) & set(prediction_files[model_b]))
    seed_deltas = []
    bootstrap_deltas = []

    for run_seed in shared_seeds:
        records_a = load_predictions(Path(prediction_files[model_a][run_seed]))
        records_b = load_predictions(Path(prediction_files[model_b][run_seed]))
        validate_pairs(records_a, records_b)
        seed_deltas.append(score(records_a, metric) - score(records_b, metric))
        bootstrap_deltas.extend(
            bootstrap_pair_deltas(
                records_a,
                records_b,
                metric=metric,
                samples=bootstrap_samples,
                seed=seed + run_seed,
            )
        )

    bootstrap_deltas.sort()
    return {
        "a": model_a,
        "b": model_b,
        "n_seeds": len(shared_seeds),
        "seed_delta_mean": sum(seed_deltas) / len(seed_deltas),
        "seed_delta_std": statistics.stdev(seed_deltas) if len(seed_deltas) > 1 else 0.0,
        "bootstrap_ci95": [
            percentile(bootstrap_deltas, 2.5),
            percentile(bootstrap_deltas, 97.5),
        ],
        "bootstrap_p_two_sided": two_sided_p_value(bootstrap_deltas),
    }


def bootstrap_pair_deltas(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> list[float]:
    import random

    rng = random.Random(seed)
    n = len(records_a)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_a = [records_a[index] for index in indices]
        sample_b = [records_b[index] for index in indices]
        deltas.append(score(sample_a, metric) - score(sample_b, metric))
    return deltas


def two_sided_p_value(deltas: list[float]) -> float:
    if not deltas:
        return math.nan
    p_value = 2.0 * min(
        sum(delta <= 0 for delta in deltas) / len(deltas),
        sum(delta >= 0 for delta in deltas) / len(deltas),
    )
    return min(1.0, p_value)


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Sweep Summary",
        "",
        f"Metric: `{report['metric']}`",
        "",
        "| Benchmark | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark in report["benchmarks"]:
        models = benchmark["models"]
        uncased = models.get("uncased_pretrained", {})
        cased = models.get("cased_pretrained", {})
        capitalized = models.get("capitalized_pretrained", {})
        cap_uncased = delta_for(benchmark, "capitalized_pretrained", "uncased_pretrained")
        cap_cased = delta_for(benchmark, "capitalized_pretrained", "cased_pretrained")
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                benchmark["benchmark"],
                fmt_mean(uncased),
                fmt_mean(cased),
                fmt_mean(capitalized),
                fmt_delta(cap_uncased),
                fmt_delta(cap_cased),
            )
        )

    lines.extend(["", "## Comparisons", ""])
    for benchmark in report["benchmarks"]:
        lines.extend([f"### {benchmark['benchmark']}", ""])
        lines.append("| A | B | Seed Delta | Bootstrap CI95 | p |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for comparison in benchmark["comparisons"]:
            lines.append(
                "| {} | {} | {:+.6f} | [{:+.6f}, {:+.6f}] | {:.4g} |".format(
                    comparison["a"],
                    comparison["b"],
                    comparison["seed_delta_mean"],
                    comparison["bootstrap_ci95"][0],
                    comparison["bootstrap_ci95"][1],
                    comparison["bootstrap_p_two_sided"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def delta_for(benchmark: dict[str, Any], model_a: str, model_b: str) -> dict[str, Any] | None:
    for comparison in benchmark["comparisons"]:
        if comparison["a"] == model_a and comparison["b"] == model_b:
            return comparison
    return None


def fmt_mean(summary: dict[str, float]) -> str:
    if not summary:
        return ""
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def fmt_delta(comparison: dict[str, Any] | None) -> str:
    if comparison is None:
        return ""
    return f"{comparison['seed_delta_mean']:+.4f}"


if __name__ == "__main__":
    main()
