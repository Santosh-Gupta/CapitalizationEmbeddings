# Project Goal

Primary goal: get `capitalized-bert-base-uncased` to beat `bert-base-uncased`
on at least one capitalization-sensitive benchmark.

Stretch goal: get the capitalization-embedding model within range of, or better
than, `bert-base-cased` on the same benchmark. Further stretch: repeat the result
across multiple capitalization-sensitive benchmarks.

## Scoreboard Criteria

Every benchmark run should compare the same downstream training settings across:

```text
bert-base-uncased
bert-base-cased
capitalized-bert-base-uncased initialized from bert-base-uncased
capitalized-bert-base-uncased after continued MLM pretraining
```

The main success criterion is downstream validation/test performance, not MLM
loss by itself. Continued pretraining should still log token MLM loss,
capitalization loss, and first-cap/all-caps accuracy so bad checkpoints can be
filtered before expensive fine-tuning.

## Benchmark Discovery

The best proof-of-concept benchmarks are the ones where `bert-base-cased` has a
clear advantage over `bert-base-uncased`. The initial candidate registry lives in
`capitalization_embeddings/benchmarks.py`.

Initial priority:

1. `conll2003_ner`: first implemented target; clean newswire named entities.
2. `wnut17_ner`: noisy social-media entities, useful for generalization.
3. `ontonotes5_ner`: broader multi-genre NER if the HF dataset path and labels
   are stable enough for automated runs.
4. `conll2003_pos`: cheap auxiliary check for proper-noun tagging.

Before spending major GPU time, run cased-vs-uncased baselines on candidate
tasks and rank by absolute metric gap. Prefer tasks with a large cased advantage,
stable datasets on Hugging Face, and affordable fine-tuning time.

## Iteration Loop

1. Establish matched baselines for cased, uncased, and the capitalization model.
2. Continue-pretrain the capitalization model on capitalization-rich text.
3. Fine-tune all checkpoints with identical downstream settings.
4. Append metrics to a persistent results table.
5. Iterate on capitalization loss weight, base embedding freezing, capitalization
   dropout/noise, and whether the capitalization prediction head is pretraining
   only.

## Current Next Step

Build a benchmark runner that writes a single comparison table for CoNLL-2003
NER, then extend that runner to the benchmark registry after the first full
RunPod run completes.
