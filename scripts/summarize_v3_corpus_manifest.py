#!/usr/bin/env python
"""Summarize V3 pretraining corpus manifest balance."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="",
        help=(
            "Path to v3_corpus_manifest.jsonl.gz. Defaults to "
            "$CAP_EMB_WORK_ROOT/prepared_corpora/v3_corpus_manifest.jsonl.gz."
        ),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest) if args.manifest else default_manifest_path()
    rows = load_manifest_rows(manifest)
    summary = summarize_rows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(summary), encoding="utf-8")


def default_manifest_path() -> Path:
    root = os.environ.get("CAP_EMB_WORK_ROOT", "/workspace/capitalization_embeddings")
    return Path(root) / "prepared_corpora" / "v3_corpus_manifest.jsonl.gz"


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_corpus = Counter(row.get("corpus", "") for row in rows)
    by_split = Counter(row.get("split", "") for row in rows)
    by_bucket = Counter(row.get("selected_bucket", "") for row in rows)
    by_source = Counter(row.get("source", "") for row in rows)
    by_corpus_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    by_corpus_source: dict[str, Counter[str]] = defaultdict(Counter)

    totals = {
        "row_length_words": 0,
        "case_signal_score": 0,
        "first_cap_count": 0,
        "all_caps_count": 0,
        "mixed_case_count": 0,
        "lowercase_count": 0,
    }

    for row in rows:
        corpus = str(row.get("corpus", ""))
        by_corpus_bucket[corpus][str(row.get("selected_bucket", ""))] += 1
        by_corpus_source[corpus][str(row.get("source", ""))] += 1
        for key in totals:
            totals[key] += int(row.get(key, 0) or 0)

    count = len(rows)
    means = {
        key: (value / count if count else 0.0)
        for key, value in totals.items()
    }
    return {
        "row_count": count,
        "by_corpus": dict(sorted(by_corpus.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "top_sources": dict(by_source.most_common(25)),
        "by_corpus_bucket": {
            corpus: dict(sorted(counter.items()))
            for corpus, counter in sorted(by_corpus_bucket.items())
        },
        "by_corpus_top_sources": {
            corpus: dict(counter.most_common(15))
            for corpus, counter in sorted(by_corpus_source.items())
        },
        "means": means,
        "totals": totals,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V3 Corpus Manifest Summary",
        "",
        f"Rows: {summary['row_count']}",
        "",
        "## By Corpus",
        "",
        render_table(("Corpus", "Rows"), summary["by_corpus"]),
        "",
        "## By Split",
        "",
        render_table(("Split", "Rows"), summary["by_split"]),
        "",
        "## By Capitalization Bucket",
        "",
        render_table(("Bucket", "Rows"), summary["by_bucket"]),
        "",
        "## Top Sources",
        "",
        render_table(("Source", "Rows"), summary["top_sources"]),
        "",
        "## Means",
        "",
        render_table(
            ("Metric", "Mean"),
            {key: f"{value:.3f}" for key, value in summary["means"].items()},
        ),
        "",
    ]
    return "\n".join(lines)


def render_table(headers: tuple[str, str], values: dict[str, Any]) -> str:
    lines = [
        f"| {headers[0]} | {headers[1]} |",
        "|---|---:|",
    ]
    for key, value in values.items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
