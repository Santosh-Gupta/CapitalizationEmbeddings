# GPU Provider Transfer Runbook

This file is for moving the project from RunPod to another GPU provider such as
TensorDock or Vast.ai without losing checkpoint provenance.

## Current RunPod Storage Layout

Persistent data lives under:

```text
/workspace/capitalization_embeddings
```

Important subdirectories:

```text
checkpoints/       model checkpoints, benchmark JSONL rows, saved predictions
prepared_corpora/  cached continued-pretraining corpora
reports/           paper tables, bootstrap reports, evidence summaries
logs/              launcher logs and process ids
transfer_manifest/ generated transfer inventory files
```

Repo checkout:

```text
/workspace/repos/CapitalizationEmbeddings
```

## Before Leaving RunPod

Generate a lightweight manifest:

```bash
cd /workspace/repos/CapitalizationEmbeddings
python scripts/build_transfer_manifest.py
```

Generate a full checksum manifest only when GPU work is stopped or nearly done,
because hashing model files can take time:

```bash
cd /workspace/repos/CapitalizationEmbeddings
python scripts/build_transfer_manifest.py --with-sha256
```

Manifest outputs:

```text
/workspace/capitalization_embeddings/transfer_manifest/transfer_manifest.json
/workspace/capitalization_embeddings/transfer_manifest/transfer_manifest_files.jsonl
```

## What Must Transfer

Transfer all of:

```text
/workspace/capitalization_embeddings/checkpoints
/workspace/capitalization_embeddings/prepared_corpora
/workspace/capitalization_embeddings/reports
/workspace/capitalization_embeddings/logs
/workspace/capitalization_embeddings/transfer_manifest
```

The Git repo can be recloned from GitHub, but preserving the checkout is also
fine:

```text
/workspace/repos/CapitalizationEmbeddings
```

## Network-To-Network Options

Preferred approach: run the copy from a cloud VM or the destination GPU VM, not
from a laptop.

If the destination supports direct SSH:

```bash
rsync -aH --info=progress2 \
  root@RUNPOD_HOST:/workspace/capitalization_embeddings/ \
  /workspace/capitalization_embeddings/
```

If using RunPod network-volume S3 access, configure `rclone` remotes for RunPod
S3 and the destination object store or VM-mounted storage, then copy server-side
where possible:

```bash
rclone copy runpod-s3:BUCKET_NAME/capitalization_embeddings destination:capitalization_embeddings \
  --progress --transfers 8 --checkers 16
```

Provider-specific names, bucket IDs, endpoints, and credentials should be kept
outside git and outside chat.

## After Transfer

On the destination:

```bash
cd /workspace/repos/CapitalizationEmbeddings
git pull --ff-only origin main
python scripts/build_transfer_manifest.py
```

Compare:

```text
file_count
total_bytes
by_top_level
```

If a `--with-sha256` manifest was generated on both sides, compare
`transfer_manifest_files.jsonl` by `relative_path` and `sha256`.

## Cost Monitoring

Without a provider API token in an environment variable, use elapsed-time
estimates:

```text
remaining_hours = provider_balance_usd / gpu_hourly_price
```

For the current RunPod 4090 price:

```text
$7.36 / $0.69 per hour = about 10.7 GPU-hours
```

Do not paste API tokens into chat. If exact balance polling is needed, put the
provider token in an environment variable on the machine running the monitor and
use a provider-specific script that reads the token from that environment
variable.
