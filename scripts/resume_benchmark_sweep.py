"""Resume a benchmark sweep by running only missing model/seed rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODEL_KEYS = ("uncased_pretrained", "cased_pretrained", "capitalized_pretrained")
TOKEN_TASKS = ("wnut17_ner", "conll2003_ner")
SEQUENCE_TASKS = ("tweet_eval_irony", "tweet_eval_offensive", "sst5", "twenty_newsgroups")


def completed_rows(results_file: Path) -> set[tuple[str, int]]:
    rows: set[tuple[str, int]] = set()
    if not results_file.exists():
        return rows

    for line in results_file.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        model_key = record.get("model_key")
        seed = record.get("seed")
        if isinstance(model_key, str) and isinstance(seed, int):
            rows.add((model_key, seed))
    return rows


def run_command(command: list[str], dry_run: bool) -> None:
    print("RUN", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def run_task(
    *,
    task: str,
    script: str,
    models: list[str],
    seeds: list[int],
    results_root: Path,
    uncased_checkpoint: str,
    cased_checkpoint: str,
    capitalized_checkpoint: str,
    learning_rate: str,
    batch_size: str,
    epochs: str,
    dry_run: bool,
    no_save_model: bool,
) -> None:
    results_file = results_root / task / "results.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_rows(results_file)

    for seed in seeds:
        missing_models = [model for model in models if (model, seed) not in completed]
        if not missing_models:
            print(f"SKIP {task} seed={seed}: already complete", flush=True)
            continue

        command = [
            sys.executable,
            script,
            "--benchmark",
            task,
            "--models",
            *missing_models,
            "--uncased-checkpoint",
            uncased_checkpoint,
            "--cased-checkpoint",
            cased_checkpoint,
            "--capitalized-checkpoint",
            capitalized_checkpoint,
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
            "--learning-rate",
            learning_rate,
            "--seed",
            str(seed),
            "--results-file",
            str(results_file),
        ]
        if no_save_model:
            command.append("--no-save-model")
        run_command(command, dry_run=dry_run)
        completed = completed_rows(results_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--uncased-checkpoint", required=True)
    parser.add_argument("--cased-checkpoint", required=True)
    parser.add_argument("--capitalized-checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 34, 55, 89])
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KEYS),
        choices=MODEL_KEYS,
        help="Subset of pretrained model keys to run.",
    )
    parser.add_argument("--token-tasks", nargs="*", default=list(TOKEN_TASKS))
    parser.add_argument("--sequence-tasks", nargs="*", default=list(SEQUENCE_TASKS))
    parser.add_argument("--token-epochs", default="3")
    parser.add_argument("--sequence-epochs", default="3")
    parser.add_argument("--token-batch-size", default="16")
    parser.add_argument("--sequence-batch-size", default="16")
    parser.add_argument("--token-learning-rate", default="3e-5")
    parser.add_argument("--sequence-learning-rate", default="2e-5")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for task in args.token_tasks:
        run_task(
            task=task,
            script="scripts/run_token_classification_benchmark.py",
            models=args.models,
            seeds=args.seeds,
            results_root=args.results_root,
            uncased_checkpoint=args.uncased_checkpoint,
            cased_checkpoint=args.cased_checkpoint,
            capitalized_checkpoint=args.capitalized_checkpoint,
            learning_rate=args.token_learning_rate,
            batch_size=args.token_batch_size,
            epochs=args.token_epochs,
            dry_run=args.dry_run,
            no_save_model=args.no_save_model,
        )

    for task in args.sequence_tasks:
        run_task(
            task=task,
            script="scripts/run_sequence_classification_benchmark.py",
            models=args.models,
            seeds=args.seeds,
            results_root=args.results_root,
            uncased_checkpoint=args.uncased_checkpoint,
            cased_checkpoint=args.cased_checkpoint,
            capitalized_checkpoint=args.capitalized_checkpoint,
            learning_rate=args.sequence_learning_rate,
            batch_size=args.sequence_batch_size,
            epochs=args.sequence_epochs,
            dry_run=args.dry_run,
            no_save_model=args.no_save_model,
        )


if __name__ == "__main__":
    main()
