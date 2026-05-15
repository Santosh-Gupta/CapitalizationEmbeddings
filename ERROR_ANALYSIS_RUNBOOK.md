# Error Analysis Runbook

This is the next non-training step once the RunPod network volume is mounted
again. It does not require GPU, but it does require the saved result JSONL files
and prediction JSONL files under `/workspace/capitalization_embeddings`.

## Purpose

Generate token-level correctness tables grouped by source-word capitalization
class:

- `none/lower`
- `first_cap`
- `all_caps`
- `mixed_case`

These tables support the paper's mechanism claim: gains should concentrate on
capitalization-sensitive tokens and entity tokens, not only aggregate F1.

## Important Paths

```text
repo:
/workspace/repos/CapitalizationEmbeddings

checkpoint/results root:
/workspace/capitalization_embeddings/checkpoints

report root:
/workspace/capitalization_embeddings/reports/error_analysis
```

For the headline token/entity tasks:

```text
current-best capitalized CoNLL/WNUT:
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed

matched cased/uncased CoNLL/WNUT:
/workspace/capitalization_embeddings/checkpoints/significance_5seed

added Walia benchmark:
/workspace/capitalization_embeddings/checkpoints/added_cased_favored_5seed
```

## CoNLL-2003

Merge only the current-best capitalized rows from `mixed_case_eval_3seed` and
the matched cased/uncased rows from `significance_5seed`.

```bash
cd /workspace/repos/CapitalizationEmbeddings
mkdir -p /workspace/capitalization_embeddings/reports/error_analysis

python - <<'PY'
import json
from pathlib import Path

rows = []
inputs = [
    (
        Path("/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed/conll2003_ner/results.jsonl"),
        {"capitalized_pretrained"},
    ),
    (
        Path("/workspace/capitalization_embeddings/checkpoints/significance_5seed/conll2003_ner/results.jsonl"),
        {"cased_pretrained", "uncased_pretrained"},
    ),
]
seen = set()
for path, model_keys in inputs:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["model_key"] not in model_keys:
            continue
        key = (row["model_key"], int(row["seed"]))
        if key in seen:
            raise SystemExit(f"duplicate row: {key}")
        seen.add(key)
        if not Path(row["prediction_file"]).exists():
            raise SystemExit(f"missing prediction file: {row['prediction_file']}")
        rows.append(row)

output = Path("/workspace/capitalization_embeddings/reports/error_analysis/conll2003_ner_current_best_results.jsonl")
output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
print(f"wrote {len(rows)} rows to {output}")
PY

python scripts/error_analysis_by_case.py \
  --benchmark conll2003_ner \
  --results-file /workspace/capitalization_embeddings/reports/error_analysis/conll2003_ner_current_best_results.jsonl \
  --output-json /workspace/capitalization_embeddings/reports/error_analysis/conll2003_ner_by_case.json \
  --output-md /workspace/capitalization_embeddings/reports/error_analysis/conll2003_ner_by_case.md
```

Expected row count: `60` rows, from `20` seeds x `3` model families.

## WNUT-17

Use the same merge logic with the WNUT roots:

```bash
cd /workspace/repos/CapitalizationEmbeddings
mkdir -p /workspace/capitalization_embeddings/reports/error_analysis

python - <<'PY'
import json
from pathlib import Path

rows = []
inputs = [
    (
        Path("/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed/wnut17_ner/results.jsonl"),
        {"capitalized_pretrained"},
    ),
    (
        Path("/workspace/capitalization_embeddings/checkpoints/significance_5seed/wnut17_ner/results.jsonl"),
        {"cased_pretrained", "uncased_pretrained"},
    ),
]
seen = set()
for path, model_keys in inputs:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["model_key"] not in model_keys:
            continue
        key = (row["model_key"], int(row["seed"]))
        if key in seen:
            raise SystemExit(f"duplicate row: {key}")
        seen.add(key)
        if not Path(row["prediction_file"]).exists():
            raise SystemExit(f"missing prediction file: {row['prediction_file']}")
        rows.append(row)

output = Path("/workspace/capitalization_embeddings/reports/error_analysis/wnut17_ner_current_best_results.jsonl")
output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
print(f"wrote {len(rows)} rows to {output}")
PY

python scripts/error_analysis_by_case.py \
  --benchmark wnut17_ner \
  --results-file /workspace/capitalization_embeddings/reports/error_analysis/wnut17_ner_current_best_results.jsonl \
  --output-json /workspace/capitalization_embeddings/reports/error_analysis/wnut17_ner_by_case.json \
  --output-md /workspace/capitalization_embeddings/reports/error_analysis/wnut17_ner_by_case.md
```

Expected row count: `60` rows, from `20` seeds x `3` model families.

## Kaggle/Walia NER

The Walia run already has all three model families in one result root.

```bash
cd /workspace/repos/CapitalizationEmbeddings
mkdir -p /workspace/capitalization_embeddings/reports/error_analysis

python scripts/error_analysis_by_case.py \
  --benchmark kaggle_walia_ner \
  --results-file /workspace/capitalization_embeddings/checkpoints/added_cased_favored_5seed/kaggle_walia_ner/results.jsonl \
  --output-json /workspace/capitalization_embeddings/reports/error_analysis/kaggle_walia_ner_by_case.json \
  --output-md /workspace/capitalization_embeddings/reports/error_analysis/kaggle_walia_ner_by_case.md
```

Expected row count: `15` rows, from `5` seeds x `3` model families.

## Paper Interpretation

Use these tables as qualitative mechanism evidence only. They are token-level
accuracy summaries grouped by source case class, not entity-span F1 and not a
replacement for the main benchmark metrics.

The most useful paper pattern would be:

- capitalized improves first-cap and all-caps entity-token accuracy over
  uncased;
- capitalized remains near uncased on lowercase/non-entity tokens;
- cased wins some mixed-case or social-text cases that the small capitalization
  channel does not fully capture.
