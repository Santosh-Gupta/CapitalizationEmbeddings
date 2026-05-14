#!/usr/bin/env python
"""Seed-level superiority, non-inferiority, and equivalence from summary JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--project-n", type=int, default=35)
    parser.add_argument("--output-md", type=Path, default=Path(""))
    parser.add_argument("--output-json", type=Path, default=Path(""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for path in args.summary_json:
        summary = json.loads(path.read_text())
        for benchmark in summary["benchmarks"]:
            best_baseline = choose_best_baseline(benchmark["models"])
            for comparison in benchmark["comparisons"]:
                if comparison["a"] != "capitalized_pretrained":
                    continue
                records.append(
                    analyze_comparison(
                        benchmark=benchmark["benchmark"],
                        metric=summary["metric"],
                        comparison=comparison,
                        best_baseline=best_baseline,
                        margin=args.margin,
                        alpha=args.alpha,
                        project_n=args.project_n,
                    )
                )

    report = {
        "alpha": args.alpha,
        "margin": args.margin,
        "project_n": args.project_n,
        "records": records,
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


def choose_best_baseline(models: dict[str, dict[str, float]]) -> str:
    cased = models.get("cased_pretrained", {}).get("mean", -math.inf)
    uncased = models.get("uncased_pretrained", {}).get("mean", -math.inf)
    return "cased_pretrained" if cased > uncased else "uncased_pretrained"


def analyze_comparison(
    *,
    benchmark: str,
    metric: str,
    comparison: dict[str, Any],
    best_baseline: str,
    margin: float,
    alpha: float,
    project_n: int,
) -> dict[str, Any]:
    delta = float(comparison["seed_delta_mean"])
    sd = float(comparison["seed_delta_std"])
    n = int(comparison["n_seeds"])
    df = n - 1
    se = sd / math.sqrt(n) if n > 0 else math.nan
    t_value = delta / se if se else math.inf
    superiority_p = 1.0 - stats.t.cdf(t_value, df) if math.isfinite(t_value) else 0.0
    ci95 = t_interval(delta, sd, n, confidence=1 - alpha)

    # Non-inferiority: H0 delta <= -margin, H1 delta > -margin.
    ni_t = (delta + margin) / se if se else math.inf
    noninferiority_p = 1.0 - stats.t.cdf(ni_t, df) if math.isfinite(ni_t) else 0.0
    noninferiority = bool(noninferiority_p < alpha)

    # Equivalence TOST: H0 delta <= -margin or delta >= +margin.
    lower_t = (delta + margin) / se if se else math.inf
    upper_t = (delta - margin) / se if se else -math.inf
    lower_p = 1.0 - stats.t.cdf(lower_t, df) if math.isfinite(lower_t) else 0.0
    upper_p = stats.t.cdf(upper_t, df) if math.isfinite(upper_t) else 0.0
    equivalence_p = max(lower_p, upper_p)
    equivalence = bool(equivalence_p < alpha)
    ci90 = t_interval(delta, sd, n, confidence=1 - 2 * alpha)

    projected_ci95 = t_interval(delta, sd, project_n, confidence=1 - alpha)
    projected_noninferiority = bool(projected_ci95[0] > -margin)
    projected_superiority = bool(projected_ci95[0] > 0)
    needed_superiority = required_n(delta=delta, sd=sd, alpha=alpha, power=0.80, kind="superiority")
    needed_noninferiority = required_n(
        delta=delta + margin,
        sd=sd,
        alpha=alpha,
        power=0.80,
        kind="noninferiority",
    )

    return {
        "benchmark": benchmark,
        "metric": metric,
        "comparison": f"{comparison['a']}>{comparison['b']}",
        "baseline": comparison["b"],
        "is_best_baseline": comparison["b"] == best_baseline,
        "n": n,
        "delta": delta,
        "sd": sd,
        "ci95": ci95,
        "superiority_p_one_sided": superiority_p,
        "superiority": bool(superiority_p < alpha and delta > 0),
        "noninferiority_margin": margin,
        "noninferiority_p": noninferiority_p,
        "noninferiority": noninferiority,
        "equivalence_margin": margin,
        "equivalence_p_tost": equivalence_p,
        "equivalence_ci90": ci90,
        "equivalence": equivalence,
        "project_n": project_n,
        "projected_ci95": projected_ci95,
        "projected_superiority": projected_superiority,
        "projected_noninferiority": projected_noninferiority,
        "approx_n_80_power_superiority": needed_superiority,
        "approx_n_80_power_noninferiority": needed_noninferiority,
    }


def t_interval(delta: float, sd: float, n: int, *, confidence: float) -> list[float]:
    if n <= 1:
        return [math.nan, math.nan]
    critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    half_width = critical * sd / math.sqrt(n)
    return [delta - half_width, delta + half_width]


def required_n(*, delta: float, sd: float, alpha: float, power: float, kind: str) -> float:
    if delta <= 0 or sd <= 0:
        return math.inf
    if kind == "superiority":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    elif kind == "noninferiority":
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError(kind)
    z_power = stats.norm.ppf(power)
    return ((z_alpha + z_power) * sd / delta) ** 2


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Seed-Level Significance",
        "",
        f"Alpha: `{report['alpha']}`",
        f"Non-inferiority/equivalence margin: `{report['margin']}`",
        f"Projected n: `{report['project_n']}`",
        "",
        "This report treats random seed as the replicated unit. Superiority is a",
        "one-sided paired-seed test that the capitalized model is better. Matching is",
        "reported as non-inferiority to the better baseline; strict equivalence uses",
        "TOST and is harder because it also rejects being meaningfully better.",
        "",
        "| Benchmark | Baseline | Best? | n | Delta | CI95 | Sup p | Non-inf p | Non-inf | Equiv p | Equiv | Projected CI95 | Projected sup | Projected non-inf | n80 sup | n80 non-inf |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for record in report["records"]:
        ci_low, ci_high = record["ci95"]
        pci_low, pci_high = record["projected_ci95"]
        lines.append(
            "| {} | {} | {} | {} | {:+.6f} | [{:+.6f}, {:+.6f}] | {:.4g} | {:.4g} | {} | {:.4g} | {} | [{:+.6f}, {:+.6f}] | {} | {} | {} | {} |".format(
                record["benchmark"],
                record["baseline"].replace("_pretrained", ""),
                "yes" if record["is_best_baseline"] else "no",
                record["n"],
                record["delta"],
                ci_low,
                ci_high,
                record["superiority_p_one_sided"],
                record["noninferiority_p"],
                "yes" if record["noninferiority"] else "no",
                record["equivalence_p_tost"],
                "yes" if record["equivalence"] else "no",
                pci_low,
                pci_high,
                "yes" if record["projected_superiority"] else "no",
                "yes" if record["projected_noninferiority"] else "no",
                format_n(record["approx_n_80_power_superiority"]),
                format_n(record["approx_n_80_power_noninferiority"]),
            )
        )
    return "\n".join(lines) + "\n"


def format_n(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return str(math.ceil(value))


if __name__ == "__main__":
    main()
