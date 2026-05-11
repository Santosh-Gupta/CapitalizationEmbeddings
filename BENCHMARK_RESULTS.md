# Benchmark Results

Results are appended here when a full benchmark run completes. Treat single-seed
wins as promising evidence, not final proof.

## CoNLL-2003 NER

Run date: 2026-05-10

Command:

```bash
python scripts/run_token_classification_benchmark.py \
  --benchmark conll2003_ner \
  --models uncased cased capitalized capitalized_pretrained \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/final \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --results-file /workspace/capitalization_embeddings/checkpoints/benchmarks/conll2003_ner/full_results.jsonl
```

Environment:

```text
RunPod RTX 4090
seed = 13
dataset = lhoestq/conll2003
metric = seqeval test F1
```

| Model | Continued MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904548 | 0.898933 | 0.910234 | 0.980640 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `capitalized_pretrained` initialized from `bert-base-uncased` | yes | 0.913929 | 0.906762 | 0.921211 | 0.982642 |

Current read:

```text
capitalized_pretrained - bert-base-uncased = +0.009381 F1
capitalized_pretrained - bert-base-cased   = +0.000672 F1
```

This satisfies the primary goal on this run and narrowly satisfies the first
stretch goal on this run. Next replication target: repeat with multiple seeds
and add WNUT-17 NER.
