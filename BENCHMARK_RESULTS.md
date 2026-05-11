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

## CoNLL-2003 NER With Matched Domain-Adaptive MLM

Run date: 2026-05-10

Pretraining command pattern:

```bash
python scripts/run_mlm_pretraining.py \
  --model-kind {uncased,cased,capitalized} \
  --corpus conll2003_train \
  --max-steps 1000 \
  --batch-size 32 \
  --gradient-accumulation-steps 2
```

Fine-tuning command:

```bash
python scripts/run_token_classification_benchmark.py \
  --benchmark conll2003_ner \
  --models uncased cased capitalized uncased_pretrained cased_pretrained capitalized_pretrained \
  --uncased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/conll2003_train/uncased_steps1000/final \
  --cased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/conll2003_train/cased_steps1000/final \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/conll2003_train/capitalized_steps1000/final \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --results-file /workspace/capitalization_embeddings/checkpoints/benchmarks/conll2003_ner/domain_pretrain_results.jsonl
```

| Model | Domain MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904259 | 0.898880 | 0.909703 | 0.980575 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `uncased_pretrained` | yes | 0.903907 | 0.898530 | 0.909348 | 0.980467 |
| `cased_pretrained` | yes | 0.915311 | 0.909313 | 0.921388 | 0.982879 |
| `capitalized_pretrained` | yes | 0.916426 | 0.907906 | 0.925106 | 0.982685 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = +0.012518 F1
capitalized_pretrained - cased_pretrained   = +0.001115 F1
```

This is a more useful comparison than the first result because all three model
families received matched MLM adaptation on the downstream domain before
fine-tuning.
