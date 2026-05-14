#!/usr/bin/env python
"""Apply Holm-Bonferroni correction to benchmark summary JSON files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Hypothesis:
    family: str
    benchmark: str
    comparison: str
    n_seeds: int
    seed_delta_mean: float
    bootstrap_ci95: tuple[float, float]
    p_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_json",
        nargs="+",
        type=Path,
        help="JSON files produced by scripts/summarize_benchmark_sweep.py.",
    )
    parser.add_argument("--family", default="headline", help="Name for the correction family.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--equivalence-threshold",
        type=float,
        default=0.002,
        help="Absolute metric delta treated as a practical tie when not significant.",
    )
    parser.add_argument(
        "--comparisons",
        nargs="*",
        default=["capitalized_pretrained>uncased_pretrained", "capitalized_pretrained>cased_pretrained"],
        help="Comparisons to include, written as A>B.",
    )
    parser.add_argument("--output-json", type=Path, default=Path(""))
    parser.add_argument("--output-md", type=Path, default=Path(""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = parse_comparisons(args.comparisons)
    hypotheses = []
    for path in args.summary_json:
        hypotheses.extend(load_hypotheses(path, family=args.family, wanted=wanted))

    corrected = holm_bonferroni(hypotheses, alpha=args.alpha)
    report = {
        "family": args.family,
        "alpha": args.alpha,
        "equivalence_threshold": args.equivalence_threshold,
        "hypotheses": [
            {
                **record,
                "label": label_result(
                    record["seed_delta_mean"],
                    record["bootstrap_ci95"],
                    record["holm_reject"],
                    equivalence_threshold=args.equivalence_threshold,
                ),
            }
            for record in corrected
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


def parse_comparisons(values: list[str]) -> set[tuple[str, str]]:
    comparisons = set()
    for value in values:
        if ">" not in value:
            raise ValueError(f"Comparison must be written as A>B, got {value!r}.")
        left, right = value.split(">", 1)
        comparisons.add((left, right))
    return comparisons


def load_hypotheses(
    path: Path,
    *,
    family: str,
    wanted: set[tuple[str, str]],
) -> list[Hypothesis]:
    summary = json.loads(path.read_text())
    hypotheses = []
    for benchmark in summary["benchmarks"]:
        for comparison in benchmark["comparisons"]:
            pair = (comparison["a"], comparison["b"])
            if pair not in wanted:
                continue
            ci_low, ci_high = comparison["bootstrap_ci95"]
            hypotheses.append(
                Hypothesis(
                    family=family,
                    benchmark=benchmark["benchmark"],
                    comparison=f"{comparison['a']}>{comparison['b']}",
                    n_seeds=int(comparison["n_seeds"]),
                    seed_delta_mean=float(comparison["seed_delta_mean"]),
                    bootstrap_ci95=(float(ci_low), float(ci_high)),
                    p_value=float(comparison["bootstrap_p_two_sided"]),
                )
            )
    return hypotheses


def holm_bonferroni(hypotheses: list[Hypothesis], *, alpha: float) -> list[dict[str, Any]]:
    finite = [hypothesis for hypothesis in hypotheses if math.isfinite(hypothesis.p_value)]
    ordered = sorted(finite, key=lambda hypothesis: hypothesis.p_value)
    total = len(ordered)
    raw_records: dict[Hypothesis, dict[str, Any]] = {}
    previous_adjusted = 0.0
    still_rejecting = True

    for rank, hypothesis in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        adjusted = min(1.0, max(previous_adjusted, hypothesis.p_value * multiplier))
        previous_adjusted = adjusted
        threshold = alpha / multiplier
        reject = still_rejecting and hypothesis.p_value <= threshold
        if not reject:
            still_rejecting = False
        raw_records[hypothesis] = {
            "family": hypothesis.family,
            "benchmark": hypothesis.benchmark,
            "comparison": hypothesis.comparison,
            "n_seeds": hypothesis.n_seeds,
            "seed_delta_mean": hypothesis.seed_delta_mean,
            "bootstrap_ci95": list(hypothesis.bootstrap_ci95),
            "p_value": hypothesis.p_value,
            "holm_rank": rank,
            "holm_threshold": threshold,
            "holm_adjusted_p": adjusted,
            "holm_reject": reject,
        }

    records = [raw_records[hypothesis] for hypothesis in ordered]
    missing = [hypothesis for hypothesis in hypotheses if not math.isfinite(hypothesis.p_value)]
    for hypothesis in missing:
        records.append(
            {
                "family": hypothesis.family,
                "benchmark": hypothesis.benchmark,
                "comparison": hypothesis.comparison,
                "n_seeds": hypothesis.n_seeds,
                "seed_delta_mean": hypothesis.seed_delta_mean,
                "bootstrap_ci95": list(hypothesis.bootstrap_ci95),
                "p_value": hypothesis.p_value,
                "holm_rank": None,
                "holm_threshold": None,
                "holm_adjusted_p": hypothesis.p_value,
                "holm_reject": False,
            }
        )
    return sorted(records, key=lambda record: (record["benchmark"], record["comparison"]))


def label_result(
    delta: float,
    ci95: list[float],
    holm_reject: bool,
    *,
    equivalence_threshold: float,
) -> str:
    ci_low, ci_high = ci95
    if holm_reject and ci_low > 0:
        return "win"
    if holm_reject and ci_high < 0:
        return "loss"
    if abs(delta) <= equivalence_threshold or ci_low <= 0 <= ci_high:
        return "tie"
    return "inconclusive_positive" if delta > 0 else "inconclusive_negative"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Holm-Corrected Benchmark Significance",
        "",
        f"Family: `{report['family']}`",
        f"Alpha: `{report['alpha']}`",
        f"Practical equivalence threshold: `{report['equivalence_threshold']}`",
        "",
        "| Benchmark | Comparison | n | Delta | CI95 | raw p | Holm p | Holm reject | Label |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in report["hypotheses"]:
        ci_low, ci_high = record["bootstrap_ci95"]
        lines.append(
            "| {} | {} | {} | {:+.6f} | [{:+.6f}, {:+.6f}] | {} | {} | {} | {} |".format(
                record["benchmark"],
                record["comparison"],
                record["n_seeds"],
                record["seed_delta_mean"],
                ci_low,
                ci_high,
                format_float(record["p_value"]),
                format_float(record["holm_adjusted_p"]),
                "yes" if record["holm_reject"] else "no",
                record["label"],
            )
        )
    return "\n".join(lines) + "\n"


def format_float(value: float) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
