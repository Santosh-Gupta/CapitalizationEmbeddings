#!/usr/bin/env python
"""Collect benchmark evidence from result roots into a markdown table.

This is intentionally lightweight: it reads the JSONL files produced by the
benchmark runners and prints mean/std/n summaries. It does not run statistical
tests; use ``summarize_benchmark_sweep.py`` for paired bootstrap reports.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHECKPOINT_ROOT = Path("/workspace/capitalization_embeddings/checkpoints")


@dataclass(frozen=True)
class EvidenceRoot:
    name: str
    path: Path
    note: str


DEFAULT_ROOTS = (
    EvidenceRoot("significance_5seed", DEFAULT_CHECKPOINT_ROOT / "significance_5seed", "3-class core 5-seed sweep"),
    EvidenceRoot("cased_sequence_5seed", DEFAULT_CHECKPOINT_ROOT / "cased_sequence_5seed", "cased-favored sequence diagnostics"),
    EvidenceRoot("scientific_5seed", DEFAULT_CHECKPOINT_ROOT / "scientific_5seed", "scientific relation diagnostics"),
    EvidenceRoot(
        "semeval2018_validation_5seed",
        DEFAULT_CHECKPOINT_ROOT / "semeval2018_validation_5seed",
        "corrected SemEval18 validation run",
    ),
    EvidenceRoot("scientbank_5seed", DEFAULT_CHECKPOINT_ROOT / "scientbank_5seed", "SciEntsBank diagnostics"),
    EvidenceRoot("mixed_case_eval_3seed", DEFAULT_CHECKPOINT_ROOT / "mixed_case_eval_3seed", "current best mixed-case token sweep"),
    EvidenceRoot(
        "required_token_baselines_3seed",
        DEFAULT_CHECKPOINT_ROOT / "required_token_baselines_3seed",
        "missing cased/uncased token controls",
    ),
    EvidenceRoot(
        "mixed_case_sequence_5seed",
        DEFAULT_CHECKPOINT_ROOT / "mixed_case_sequence_5seed",
        "current best mixed-case sequence sweep",
    ),
    EvidenceRoot(
        "added_cased_favored_5seed",
        DEFAULT_CHECKPOINT_ROOT / "added_cased_favored_5seed",
        "added cased-favored benchmark diagnostics",
    ),
)


TOKEN_F1_TASKS = {"conll2003_ner", "wnut17_ner", "ontonotes5_ner", "kaggle_walia_ner"}
ACCURACY_TASKS = {
    "ptb_pos",
    "tweet_eval_emoji",
    "trec_fine",
    "sst5",
    "twenty_newsgroups",
    "scientific_relations_combined",
    "scierc_relations",
    "semeval2018_task7",
}
MACRO_F1_TASKS = {
    "tweet_eval_irony",
    "tweet_eval_offensive",
    "tweet_eval_sentiment",
    "tweet_eval_emotion",
    "scientbank_3way_uq",
    "scientbank_3way_ud",
    "isarcasm_eval_en",
    "citation_sentiment_acl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Base checkpoint directory. Defaults to the RunPod workspace path.",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[root.name for root in DEFAULT_ROOTS],
        help="Result root names to scan under --checkpoint-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [
        EvidenceRoot(name, args.checkpoint_root / name, note_for_root(name))
        for name in args.roots
    ]
    print("# Evidence Status\n")
    print(f"Checkpoint root: `{args.checkpoint_root}`\n")
    for root in roots:
        summarize_root(root)


def note_for_root(name: str) -> str:
    for root in DEFAULT_ROOTS:
        if root.name == name:
            return root.note
    return ""


def summarize_root(root: EvidenceRoot) -> None:
    if not root.path.exists():
        return
    task_files = sorted(root.path.glob("*/results.jsonl"))
    if not task_files:
        return

    print(f"## {root.name}\n")
    if root.note:
        print(f"{root.note}.\n")
    print("| Task | Metric | Model | n | Mean | Std | Seeds |")
    print("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for results_file in task_files:
        task = results_file.parent.name
        rows = load_rows(results_file, task)
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        metric_name = metric_for_task(task)
        for row in rows:
            grouped[row["model_key"]].append((int(row["seed"]), row["metric_value"]))
        for model_key in sorted(grouped):
            pairs = sorted(grouped[model_key])
            values = [value for _, value in pairs]
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            seeds = ",".join(str(seed) for seed, _ in pairs)
            print(
                f"| {task} | {metric_name} | {model_key} | {len(values)} | "
                f"{statistics.mean(values):.6f} | {std:.6f} | {seeds} |"
            )
    print()


def load_rows(results_file: Path, task: str) -> list[dict[str, object]]:
    metric_name = metric_for_task(task)
    rows = []
    for line in results_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value_key = metric_key(metric_name)
        if value_key not in row:
            continue
        rows.append(
            {
                "model_key": row["model_key"],
                "seed": row["seed"],
                "metric_value": float(row[value_key]),
            }
        )
    return rows


def metric_for_task(task: str) -> str:
    if task in TOKEN_F1_TASKS:
        return "seqeval_f1"
    if task in MACRO_F1_TASKS:
        return "macro_f1"
    if task in ACCURACY_TASKS:
        return "accuracy"
    return "accuracy"


def metric_key(metric_name: str) -> str:
    if metric_name in {"seqeval_f1", "macro_f1"}:
        return "test_f1"
    if metric_name == "accuracy":
        return "test_accuracy"
    if metric_name == "pearson":
        return "test_pearson"
    raise ValueError(f"Unsupported metric {metric_name!r}")


if __name__ == "__main__":
    main()
