#!/usr/bin/env bash
set -euo pipefail

# Download the compact artifacts needed to preserve this project before deleting
# the RunPod network volume. This intentionally avoids bulk Trainer outputs,
# dataset caches, and most intermediate checkpoints.

ENDPOINT="${RUNPOD_S3_ENDPOINT:-https://s3api-us-il-1.runpod.io}"
BUCKET="${RUNPOD_S3_BUCKET:-s3://c2f2qj9ktg}"
DEST="${RUNPOD_BACKUP_DEST:-$HOME/Downloads/runpod-capitalization-important}"
REGION="${AWS_DEFAULT_REGION:-us-il-1}"
DOWNLOAD_BEST_MODELS="${DOWNLOAD_BEST_MODELS:-1}"

BUCKET="${BUCKET%/}"
export AWS_DEFAULT_REGION="$REGION"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required. Install it first, then rerun this script." >&2
  exit 1
fi

if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
  read -r -p "RunPod S3 access key: " AWS_ACCESS_KEY_ID
  export AWS_ACCESS_KEY_ID
fi

if [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  read -r -s -p "RunPod S3 secret key: " AWS_SECRET_ACCESS_KEY
  echo
  export AWS_SECRET_ACCESS_KEY
fi

echo "Endpoint: $ENDPOINT"
echo "Bucket:   $BUCKET"
echo "Dest:     $DEST"
echo

echo "Bucket size summary:"
aws s3 ls \
  --endpoint-url "$ENDPOINT" \
  --recursive \
  --summarize \
  --human-readable \
  "$BUCKET/" | tail -n 4
echo

mkdir -p "$DEST"

sync_prefix_if_present() {
  local prefix="$1"
  local src="$BUCKET/$prefix/"
  local dst="$DEST/$prefix/"

  if aws s3 ls --endpoint-url "$ENDPOINT" "$src" >/dev/null 2>&1; then
    echo "Syncing $prefix/"
    mkdir -p "$dst"
    aws s3 sync \
      --endpoint-url "$ENDPOINT" \
      "$src" \
      "$dst" \
      --only-show-errors
  else
    echo "Skipping missing prefix: $prefix/"
  fi
}

sync_prefix_if_present "capitalization_embeddings/reports"
sync_prefix_if_present "capitalization_embeddings/logs"

echo "Syncing compact checkpoint evidence files"
mkdir -p "$DEST/capitalization_embeddings/checkpoints"
aws s3 sync \
  --endpoint-url "$ENDPOINT" \
  "$BUCKET/capitalization_embeddings/checkpoints/" \
  "$DEST/capitalization_embeddings/checkpoints/" \
  --exclude "*" \
  --include "*.json" \
  --include "*.jsonl" \
  --include "*.md" \
  --include "*.csv" \
  --include "*.tsv" \
  --include "*.txt" \
  --exclude "benchmarks/**" \
  --only-show-errors

echo "Syncing RunPod-side repo docs/scripts, if present"
mkdir -p "$DEST/repos/CapitalizationEmbeddings"
aws s3 sync \
  --endpoint-url "$ENDPOINT" \
  "$BUCKET/repos/CapitalizationEmbeddings/" \
  "$DEST/repos/CapitalizationEmbeddings/" \
  --exclude "*" \
  --include "*.md" \
  --include "reports/**" \
  --include "logs/**" \
  --include "scripts/*.py" \
  --include "scripts/*.sh" \
  --exclude ".git/**" \
  --only-show-errors || true

if [[ "$DOWNLOAD_BEST_MODELS" == "1" ]]; then
  echo "Syncing current best final model checkpoints"
  sync_prefix_if_present \
    "capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final"
  sync_prefix_if_present \
    "capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final"
  sync_prefix_if_present \
    "capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final"
else
  echo "Skipping model weights because DOWNLOAD_BEST_MODELS=$DOWNLOAD_BEST_MODELS"
fi

echo
echo "Local backup summary:"
du -sh "$DEST"
find "$DEST" -type f | wc -l | awk '{print $1 " files"}'

echo
echo "Backup complete. Inspect $DEST before deleting the RunPod network volume."
