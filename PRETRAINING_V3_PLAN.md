# Pretraining V3 Plan

This plan rethinks continued pretraining after the V2 RunPod probe. The goal is
to give the capitalization-embedding model a stronger and cleaner chance while
keeping `bert-base-uncased` and `bert-base-cased` on the same text, update
budget, and downstream evaluation protocol.

## Motivation

V1/V2 continued pretraining was useful but still too shallow to be a final
pretraining story:

- The main capitalized checkpoint was developed through multiple short 3k-step
  continuations.
- V2 made the protocol fairer by giving all three model families matched
  `capitalization_domain_mix_v2` continuation, but it was still only 3k steps.
- V2 improved capitalization-channel metrics but the partial diagnostic showed
  TweetEval Emoji still strongly favored `bert-base-cased` before the pod became
  unreachable.
- The current corpus selector keeps the highest case-signal rows. That trains
  rare case patterns, but it can overemphasize acronym/legal/scientific text and
  underrepresent ordinary natural-casing contexts.

The V3 protocol should answer a stronger question:

> Given the same additional text, update budget, and downstream fine-tuning
> procedure, can capitalization embeddings recover useful casing behavior
> without giving up the lexical sharing benefits of an uncased tokenizer?

## Fairness Rule

Every headline comparison must use a matched triplet:

```text
bert-base-uncased + same V3 text/order/steps
bert-base-cased   + same V3 text/order/steps
cap-embed BERT    + same V3 text/order/steps
```

The capitalized model may use its auxiliary capitalization loss because that is
part of the architecture. To keep the comparison defensible:

- all models see the same raw rows in the same order;
- all models use the same step count, sequence length, batch size, learning-rate
  schedule, random seed, and dynamic masking seed where possible;
- any capitalization-specific warmup must be reported as an ablation, not hidden
  inside the headline checkpoint;
- no validation/test rows from downstream benchmarks are allowed in pretraining
  corpora used for headline results.

## V3 Corpora

Use three separate corpora so the paper can distinguish general pretraining from
domain adaptation.

### V3-G: General Natural Casing

Purpose: improve broad capitalization behavior without benchmark train-text
adaptation.

Candidate source families:

```text
Wikipedia / Wikitext or full Wikipedia dumps
Book-style prose, if a stable HF source is available
news or web text with natural capitalization
scientific abstracts/articles, for acronym and entity casing
legal/government text, for acronym and title casing
social/noisy text, for all-caps emphasis and irregular casing
```

Selection rule:

- do not keep only the highest case-signal rows;
- deduplicate by normalized text hash;
- keep source quotas so one source cannot dominate;
- stratify rows by capitalization profile:
  `low-case-signal`, `first-cap-rich`, `all-caps-rich`, `mixed-case-rich`,
  and `noisy-social`;
- oversample all-caps and mixed-case enough to train the rare channels, but
  keep at least half the corpus ordinary natural text.

### V3-D: Domain-Adaptive Train-Text Corpus

Purpose: test whether matched in-domain MLM helps the capitalized model use
case on the exact benchmark families.

Allowed data:

```text
train split text only from CoNLL-2003, WNUT-17, OntoNotes, PTB, TweetEval,
TREC, scientific relation datasets, SciEntsBank, Walia/Kaggle NER,
iSarcasmEval, SST-5, 20 Newsgroups, STS-B, Yahoo Answers, HASOC/OLID if added
```

Rules:

- labels are never used;
- validation/test text is excluded;
- report V3-D as domain-adaptive pretraining, not pure general pretraining;
- all three model families receive the same V3-D continuation.

### V3-M: Mixed General + Domain Curriculum

Purpose: final best-shot checkpoint.

Recommended order:

```text
Stage 1: V3-G only
Stage 2: V3-G with balanced capitalization oversampling
Stage 3: V3-D train-text adaptation
```

This creates two publishable comparisons:

- general-only, which is cleaner scientifically;
- general-plus-DAPT, which tests maximum practical performance under equal
  additional pretraining.

## Objective Schedule

The current cap model can underuse capitalization embeddings because uncased MLM
can often solve the token prediction task without case. V3 should make the case
channel unavoidable early, then anneal back to normal MLM.

### Capitalized Model

Use four capitalization states:

```text
none
first-cap
all-caps
mixed-case
```

Recommended curriculum:

```text
Warmup A: train cap embeddings + cap classifier strongly; optionally freeze the
          transformer for this warmup only.
Warmup B: unfreeze all parameters, cap loss weight high.
Main:     joint MLM + cap loss, cap loss weight annealed down.
DAPT:     lower learning rate, same objective, short continuation.
```

Initial candidate hyperparameters:

```text
max_length: 128 for diagnostics, 256 for main if memory allows
mlm_probability: 0.15
batch_size: largest stable per GPU
gradient_accumulation: set to keep effective batch matched across providers
capitalization_loss_weight: start 1.0, anneal to 0.25
capitalization_class_weights: derive from corpus frequencies, capped so rare
  classes cannot dominate
capitalization_embedding_dropout: sweep 0.0, 0.05, 0.1
learning_rate: 2e-5 for continuation; consider 1e-5 for DAPT
```

### Baseline Models

`bert-base-uncased` and `bert-base-cased` receive the same MLM-only continuation:

```text
same corpus
same row order
same max steps
same sequence length
same effective batch size
same optimizer/scheduler settings
same checkpoint cadence
```

If the capitalized model uses a capitalization-only warmup, run a matched
baseline-control condition where cased/uncased perform same-step MLM with the
same frozen/unfrozen schedule. This is less useful to the baselines, but it
removes an easy fairness objection.

## Checkpointing And Selection

Do not select the final checkpoint by MLM loss alone.

Checkpoint every fixed interval:

```text
1k, 3k, 5k, 10k, 20k, 30k, 50k if budget permits
```

For every checkpoint, record:

```text
MLM eval loss
overall capitalization accuracy
first-cap accuracy
all-caps accuracy
mixed-case accuracy
per-source MLM loss if feasible
per-source capitalization accuracy if feasible
```

Checkpoint selection should use a small downstream diagnostic panel:

```text
Guardrails:
  CoNLL-2003 NER
  WNUT-17 NER

Cased-favored stress tests:
  TweetEval Emoji
  TREC Fine

Uncased-favored / lexical-sharing checks:
  TweetEval Irony
  scientific relations combined
  20 Newsgroups
```

Use seed 13 only for checkpoint selection. After selecting a checkpoint, run the
full multi-seed evaluation.

## Stop/Go Criteria

Do not spend large multi-seed budget until a V3 checkpoint passes these checks:

```text
all-caps accuracy >= 0.90 on held-out V3-G
mixed-case accuracy >= 0.75 on held-out V3-G
CoNLL/WNUT seed-13 result is not worse than the current best by more than 0.5 F1
TweetEval Emoji or TREC Fine moves meaningfully toward cased
scientific relations does not fall behind matched uncased
```

If the capitalization model only improves MLM/capitalization accuracy but not
downstream diagnostics, treat V3 as evidence that the current architecture needs
more than additive global case embeddings for sequence-level tasks.

## Provider-Agnostic Implementation Requirements

Before launching V3 on TensorDock, Vast.ai, or another provider:

1. Build and cache corpus rows before expensive model training.
2. Write a manifest with row hash, source, split, and capitalization stats.
3. Save the exact selected row order.
4. Save the exact command lines and checkpoint paths in `RUN_LEDGER.md`.
5. Keep compact metric JSONL files separate from disposable Trainer output
   directories.
6. Regenerate the transfer manifest after each long run.

## Implemented Entry Points

The initial V3 scaffolding is implemented in:

```text
scripts/run_mlm_pretraining.py
scripts/run_v3_pretraining.sh
scripts/summarize_v3_corpus_manifest.py
```

Supported V3 corpus names:

```text
capitalization_v3_general
capitalization_v3_domain_train
capitalization_v3_mixed_curriculum
```

The V3 runner writes row caches and a text-free manifest under:

```text
prepared_corpora/v3_general_rows.jsonl.gz
prepared_corpora/v3_domain_train_rows.jsonl.gz
prepared_corpora/v3_mixed_curriculum_rows.jsonl.gz
prepared_corpora/v3_corpus_manifest.jsonl.gz
```

The manifest rows include source, source split, text hash, row length, case
bucket, and capitalization-count statistics. The row cache contains text; the
manifest intentionally omits text so it can be inspected and versioned more
comfortably.

The provider-ready launcher uses three stages:

```text
Stage 0: matched V3-G warmup.
  - baselines: same-step MLM continuation
  - cap model: capitalization-only parameter warmup

Stage 1: matched V3-G general natural-casing continuation.

Stage 2: matched V3-D train-split domain-adaptive continuation.
```

Default launch:

```bash
bash scripts/run_v3_pretraining.sh
```

Default stage budgets:

```text
warmup: 1,000 steps
general: 20,000 steps
domain: 5,000 steps
max_length: 128
batch_size: 32
gradient_accumulation_steps: 2
```

Provider-specific overrides can be supplied with environment variables:

```bash
WARMUP_STEPS=1000 GENERAL_STEPS=20000 DOMAIN_STEPS=5000 \
MAX_LENGTH=128 BATCH_SIZE=32 GRAD_ACCUM=2 \
bash scripts/run_v3_pretraining.sh
```

## Next Implementation Step

Before renting the next GPU, run a CPU corpus-cache build smoke test on the new
provider:

```text
python scripts/run_mlm_pretraining.py --model-kind uncased \
  --corpus capitalization_v3_general \
  --max-train-samples 64 --max-eval-samples 32 --max-steps 1 --smoke
```

Then inspect:

```text
prepared_corpora/v3_corpus_manifest.jsonl.gz
```

After building any V3 corpus cache, summarize the corpus balance with:

```bash
python scripts/summarize_v3_corpus_manifest.py \
  --output-json /workspace/capitalization_embeddings/reports/v3_corpus_manifest_summary.json \
  --output-md /workspace/capitalization_embeddings/reports/v3_corpus_manifest_summary.md
```

The summary reports:

```text
rows by source
rows by selected_bucket
case-count distributions by source
ordinary/case-rich balance
```
