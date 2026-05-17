#!/usr/bin/env python
"""Build a transfer manifest for RunPod/Vast/TensorDock migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RELATIVE_PATHS = (
    "checkpoints",
    "prepared_corpora",
    "reports",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        default=os.environ.get("CAP_EMB_WORK_ROOT", "/workspace/capitalization_embeddings"),
        help="Root that contains checkpoints, prepared corpora, reports, and logs.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(DEFAULT_RELATIVE_PATHS),
        help="Relative or absolute paths to include in the manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Where to write transfer_manifest.json/jsonl. Defaults to work_root/transfer_manifest.",
    )
    parser.add_argument(
        "--with-sha256",
        action="store_true",
        help="Hash every file. This is slow for large model checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else work_root / "transfer_manifest"
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [resolve_manifest_path(work_root, path) for path in args.paths]
    files = []
    for root in roots:
        if root.exists():
            files.extend(scan_files(root, work_root, with_sha256=args.with_sha256))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "work_root": str(work_root),
        "repo_commit": repo_commit(),
        "paths": [str(path) for path in roots],
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "total_gib": round(sum(item["size_bytes"] for item in files) / 1024**3, 3),
        "sha256_included": bool(args.with_sha256),
        "by_top_level": summarize_by_top_level(files),
    }

    write_json(output_dir / "transfer_manifest.json", summary)
    write_jsonl(output_dir / "transfer_manifest_files.jsonl", files)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote: {output_dir / 'transfer_manifest.json'}")
    print(f"wrote: {output_dir / 'transfer_manifest_files.jsonl'}")


def resolve_manifest_path(work_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return work_root / candidate


def scan_files(root: Path, work_root: Path, *, with_sha256: bool) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        row: dict[str, Any] = {
            "path": str(path),
            "relative_path": relative_path(path, work_root),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        if with_sha256:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def summarize_by_top_level(files: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for row in files:
        top_level = row["relative_path"].split("/", 1)[0]
        bucket = summary.setdefault(top_level, {"file_count": 0, "bytes": 0, "gib": 0.0})
        bucket["file_count"] = int(bucket["file_count"]) + 1
        bucket["bytes"] = int(bucket["bytes"]) + int(row["size_bytes"])
    for bucket in summary.values():
        bucket["gib"] = round(int(bucket["bytes"]) / 1024**3, 3)
    return dict(sorted(summary.items()))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
