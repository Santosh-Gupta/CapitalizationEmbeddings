#!/usr/bin/env python
"""Analyze case fragmentation in BERT WordPiece vocabularies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import AutoTokenizer

from capitalization_embeddings.tokenization import ALL_CAPS, FIRST_CAP, MIXED_CASE, classify_capitalization


@dataclass(frozen=True)
class TokenCaseRecord:
    token: str
    category: str
    lowercase_token: str
    lowercase_in_cased_vocab: bool
    lowercase_in_uncased_vocab: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cased-model", default="bert-base-cased")
    parser.add_argument("--uncased-model", default="bert-base-uncased")
    parser.add_argument("--output-json", type=Path, default=Path("reports/vocab_fragmentation.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/vocab_fragmentation.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cased_tokenizer = AutoTokenizer.from_pretrained(args.cased_model, use_fast=True)
    uncased_tokenizer = AutoTokenizer.from_pretrained(args.uncased_model, use_fast=True)
    cased_vocab = set(cased_tokenizer.get_vocab())
    uncased_vocab = set(uncased_tokenizer.get_vocab())
    special_tokens = set(cased_tokenizer.all_special_tokens)

    records = [
        analyze_token(token, cased_vocab, uncased_vocab, special_tokens)
        for token in sorted(cased_vocab, key=cased_tokenizer.get_vocab().__getitem__)
    ]
    categories = Counter(record.category for record in records)
    counterpart_counts = summarize_counterparts(records)
    family_summary = summarize_case_families(records)
    examples = illustrative_families(records)

    report = {
        "cased_model": args.cased_model,
        "uncased_model": args.uncased_model,
        "cased_vocab_size": len(cased_vocab),
        "uncased_vocab_size": len(uncased_vocab),
        "category_counts": dict(categories),
        "counterpart_counts": counterpart_counts,
        "case_family_summary": family_summary,
        "illustrative_families": examples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(report))
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


def analyze_token(
    token: str,
    cased_vocab: set[str],
    uncased_vocab: set[str],
    special_tokens: set[str],
) -> TokenCaseRecord:
    if token in special_tokens:
        category = "special"
    else:
        _, core = split_wordpiece_prefix(token)
        if not any(char.isalpha() for char in core):
            category = "no_alpha"
        else:
            cap_id = classify_capitalization(core, use_mixed_case=True)
            if cap_id == FIRST_CAP:
                category = "first_cap"
            elif cap_id == ALL_CAPS:
                category = "all_caps"
            elif cap_id == MIXED_CASE:
                category = "mixed_case"
            else:
                category = "lower_or_uncased"

    lowercase_token = lowercase_wordpiece(token)
    return TokenCaseRecord(
        token=token,
        category=category,
        lowercase_token=lowercase_token,
        lowercase_in_cased_vocab=lowercase_token in cased_vocab,
        lowercase_in_uncased_vocab=lowercase_token in uncased_vocab,
    )


def split_wordpiece_prefix(token: str) -> tuple[str, str]:
    if token.startswith("##"):
        return "##", token[2:]
    return "", token


def lowercase_wordpiece(token: str) -> str:
    prefix, core = split_wordpiece_prefix(token)
    return prefix + core.lower()


def summarize_counterparts(records: list[TokenCaseRecord]) -> dict[str, dict[str, int]]:
    summary = {}
    for category in ["first_cap", "all_caps", "mixed_case"]:
        category_records = [record for record in records if record.category == category]
        summary[category] = {
            "count": len(category_records),
            "lowercase_in_cased_vocab": sum(record.lowercase_in_cased_vocab for record in category_records),
            "lowercase_in_uncased_vocab": sum(
                record.lowercase_in_uncased_vocab for record in category_records
            ),
        }
    return summary


def summarize_case_families(records: list[TokenCaseRecord]) -> dict[str, int]:
    families: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.category in {"special", "no_alpha"}:
            continue
        families[record.lowercase_token].add(record.category)

    return {
        "alphabetic_families": len(families),
        "families_with_multiple_case_forms": sum(len(categories) > 1 for categories in families.values()),
        "families_with_lower_and_first_cap": sum(
            {"lower_or_uncased", "first_cap"}.issubset(categories)
            for categories in families.values()
        ),
        "families_with_lower_and_all_caps": sum(
            {"lower_or_uncased", "all_caps"}.issubset(categories)
            for categories in families.values()
        ),
        "families_with_lower_first_and_all_caps": sum(
            {"lower_or_uncased", "first_cap", "all_caps"}.issubset(categories)
            for categories in families.values()
        ),
        "families_with_mixed_case": sum("mixed_case" in categories for categories in families.values()),
    }


def illustrative_families(records: list[TokenCaseRecord]) -> list[dict[str, object]]:
    families: dict[str, list[TokenCaseRecord]] = defaultdict(list)
    for record in records:
        if record.category in {"special", "no_alpha"}:
            continue
        families[record.lowercase_token].append(record)

    preferred = [
        "tom",
        "john",
        "apple",
        "washington",
        "nasa",
        "ibm",
        "us",
        "##s",
        "##ed",
    ]
    examples = []
    for key in preferred:
        if key in families and len(families[key]) > 1:
            examples.append(family_to_example(key, families[key]))

    if len(examples) < 10:
        rich_families = sorted(
            families.items(),
            key=lambda item: (-len({record.category for record in item[1]}), item[0]),
        )
        seen = {example["lowercase_token"] for example in examples}
        for key, family_records in rich_families:
            if key in seen or len(family_records) < 2:
                continue
            examples.append(family_to_example(key, family_records))
            if len(examples) >= 10:
                break
    return examples


def family_to_example(key: str, records: list[TokenCaseRecord]) -> dict[str, object]:
    return {
        "lowercase_token": key,
        "forms": sorted(record.token for record in records),
        "categories": sorted({record.category for record in records}),
    }


def to_markdown(report: dict[str, object]) -> str:
    category_counts = report["category_counts"]
    counterpart_counts = report["counterpart_counts"]
    family_summary = report["case_family_summary"]
    examples = report["illustrative_families"]

    capitalized_count = sum(
        int(category_counts.get(category, 0))
        for category in ("first_cap", "all_caps", "mixed_case")
    )
    first_all_count = sum(
        int(category_counts.get(category, 0))
        for category in ("first_cap", "all_caps")
    )
    first_all_with_uncased_lower = sum(
        int(counterpart_counts[category]["lowercase_in_uncased_vocab"])
        for category in ("first_cap", "all_caps")
    )

    lines = [
        "# Vocabulary Fragmentation",
        "",
        f"Cased model: `{report['cased_model']}`",
        f"Uncased model: `{report['uncased_model']}`",
        "",
        f"- Cased vocab size: `{int(report['cased_vocab_size']):,}`",
        f"- Uncased vocab size: `{int(report['uncased_vocab_size']):,}`",
        f"- Cased vocab tokens with first-cap/all-caps/mixed-case form: `{capitalized_count:,}`",
        f"- First-cap/all-caps cased tokens: `{first_all_count:,}`",
        f"- First-cap/all-caps cased tokens whose lowercase form exists in the uncased vocab: `{first_all_with_uncased_lower:,}`",
        "",
        "## Cased Vocabulary by Case Class",
        "",
        "| Case class | Count | Share of cased vocab |",
        "| --- | ---: | ---: |",
    ]
    total = int(report["cased_vocab_size"])
    for category in ["special", "no_alpha", "lower_or_uncased", "first_cap", "all_caps", "mixed_case"]:
        count = int(category_counts.get(category, 0))
        lines.append(f"| {category} | {count:,} | {count / total:.2%} |")

    lines.extend(
        [
            "",
            "## Lowercase Counterpart Coverage",
            "",
            "| Case class | Tokens | Lowercase in cased vocab | Lowercase in uncased vocab |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for category in ["first_cap", "all_caps", "mixed_case"]:
        row = counterpart_counts[category]
        lines.append(
            f"| {category} | {int(row['count']):,} | "
            f"{int(row['lowercase_in_cased_vocab']):,} | "
            f"{int(row['lowercase_in_uncased_vocab']):,} |"
        )

    lines.extend(
        [
            "",
            "## Case Families in the Cased Vocab",
            "",
            "| Statistic | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in family_summary.items():
        lines.append(f"| {key} | {int(value):,} |")

    lines.extend(
        [
            "",
            "## Illustrative Case Families",
            "",
            "| Lowercase family | Cased-vocab forms | Classes |",
            "| --- | --- | --- |",
        ]
    )
    for example in examples:
        lines.append(
            "| {} | `{}` | {} |".format(
                example["lowercase_token"],
                "`, `".join(example["forms"]),
                ", ".join(example["categories"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paper-Framing Note",
            "",
            "The cased BERT vocabulary already contains many surface forms that can be "
            "described as a lowercase WordPiece plus a small case feature. This "
            "supports the factorization hypothesis. The mixed-case bucket should be "
            "framed as a pragmatic extension: it lets `iPhone`/`eBay`-style forms "
            "carry a case signal without creating a separate lexical embedding for "
            "each surface form.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
