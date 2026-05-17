# Continued Pretraining Corpora

This file records the unlabeled text used for continued MLM pretraining. It is
the source of truth for avoiding accidental compute or data advantages across
model families.

## Model Scale

The current project uses only BERT-base family models:

```text
bert-base-uncased
bert-base-cased
bert-base-uncased + capitalization embeddings
```

No BERT-large runs are part of the current evidence package. If BERT-large is
added later, it needs a separate matched-control protocol.

## Current Paper Checkpoints

The current baseline checkpoints are:

```text
uncased: /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final
cased:   /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
cap:     /workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final
```

The cased and uncased controls received the same first 3k-step
`capitalization_real_acronym_mix` continuation. The current best capitalized
model then received an additional 3k-step mixed-case/dropout continuation while
expanding from three capitalization states to four. That is acceptable as an
architecture-development checkpoint, but any claim about "more pretraining"
should use a fresh matched run where all model families receive the same added
continuation budget.

## Previous Pretraining History

The current evidence package was built from this BERT-base-only chain:

| Stage | Models | Corpus | Steps | Initial checkpoint | Endpoint train loss | Endpoint eval loss |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Wikitext general continuation | uncased | `wikitext103` | 5,000 | `bert-base-uncased` | 1.758 | 1.608 |
| Wikitext general continuation | cased | `wikitext103` | 5,000 | `bert-base-cased` | 1.704 | 1.538 |
| Wikitext general continuation | capitalized | `wikitext103` | 5,000 | `bert-base-uncased` + cap embeddings | 1.798 | 1.640 |
| Task-mix continuation | uncased | `capitalization_task_mix` | 3,000 | wikitext uncased | 2.328 | 2.316 |
| Task-mix continuation | cased | `capitalization_task_mix` | 3,000 | wikitext cased | 2.284 | 2.226 |
| Task-mix continuation | capitalized | `capitalization_task_mix` | 3,000 | wikitext capitalized | 2.363 | 2.344 |
| Real-acronym continuation | uncased | `capitalization_real_acronym_mix` | 3,000 | task-mix uncased | 1.513 | 1.938 |
| Real-acronym continuation | cased | `capitalization_real_acronym_mix` | 3,000 | task-mix cased | 1.501 | 1.913 |
| Real-acronym continuation | capitalized | `capitalization_real_acronym_mix` | 3,000 | task-mix capitalized | 1.559 | 1.992 |
| Mixed-case/dropout development | capitalized | `capitalization_real_acronym_mix` | 3,000 | 3-class real-acronym capitalized | 1.374 | 2.000 |

The mixed-case/dropout row is the current best capitalized checkpoint, but it
is not compute-matched against cased/uncased because those controls did not get
the same extra second real-acronym pass. That is why the V2 launcher first
creates:

```text
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_round2_steps3000_lr2e5/final
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_round2_steps3000_lr2e5/final
```

These round-2 controls should be used only as stepping stones into V2 unless a
separate matched downstream diagnostic is deliberately run.

There was also a 10k-step capitalized-only real-acronym run:

```text
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/capitalized_from_task_mix_steps10000/final
train loss: 1.330
eval loss: 1.891
```

It is useful evidence that longer capitalized pretraining can lower MLM loss,
but it is not a fair headline checkpoint without matched cased/uncased controls.

## `capitalization_real_acronym_mix`

Base rows:

```text
CoNLL-2003 train text
WNUT-17 train text
OntoNotes5 train text
PTB POS train text
```

Additional rows:

```text
PubMed summarization, BillSum, LexGLUE SCOTUS
```

Rows are selected for acronym/case signal and cached under:

```text
/workspace/capitalization_embeddings/prepared_corpora/real_acronym_rows_v1.jsonl.gz
```

Known caveat: the first cache build produced no useful PubMed chunks in the
observed RunPod run, likely because that dataset's fields were not plain strings
in the loaded version. The v2 corpus fixes extraction by recursively flattening
string-like nested values.

Observed v2 note: `ccdv/pubmed-summarization` itself appears lowercased in the
Hugging Face copy used on RunPod, so it can still produce zero case-positive
chunks even with recursive extraction. This is acceptable for a
capitalization-focused corpus, but should be documented rather than treated as
a loader failure.

## `capitalization_domain_mix_v2`

Purpose:

Test whether a larger, more domain-targeted case-rich MLM corpus improves the
capitalization channel, especially on failure slices such as scientific
relations, TweetEval Emoji, and TREC Fine.

Base rows:

```text
CoNLL-2003 train text
WNUT-17 train text
OntoNotes5 train text
PTB POS train text
```

Additional source training text:

```text
Wikitext-103 train text
PubMed summarization
BillSum
LexGLUE SCOTUS
SemEval-2018 Task 7 relation train text
SciERC train text
SciEntsBank train text
TweetEval train text: emoji, irony, offensive, sentiment, emotion
TREC train text
SST-5 train text
20 Newsgroups train text
GLUE STS-B train text
Yahoo Answers Topics train[:100000] text
ACL citation sentiment text
```

Selection:

```text
score rows by first-cap, all-caps, and mixed-case signals
deduplicate normalized rows
keep the top 180,000 rows with score >= 2
shuffle the selected rows with fixed seed 13
hold out up to the first 10,000 selected rows for MLM eval
use the remaining rows for MLM training
```

Cache:

```text
/workspace/capitalization_embeddings/prepared_corpora/domain_mix_v2_rows.jsonl.gz
```

If a source silently drops out, fix it before using the V2 checkpoint as
headline evidence. During the first V2 launch, SemEval18 Task 7 was skipped
because `python scripts/run_mlm_pretraining.py` did not put the repo root on
`sys.path`; this was fixed by adding the repo root before importing helper
modules from `scripts`.

## Matched V2 Protocol

Run:

```bash
bash scripts/run_domain_mix_v2_pretraining.sh
```

The script first compute-matches the cased/uncased controls with a second
3k-step real-acronym continuation, then runs the same 3k-step
`capitalization_domain_mix_v2` continuation for all three model families.

Final checkpoints:

```text
/workspace/capitalization_embeddings/checkpoints/mlm/domain_mix_v2/uncased_from_round2_steps3000_lr2e5/final
/workspace/capitalization_embeddings/checkpoints/mlm/domain_mix_v2/cased_from_round2_steps3000_lr2e5/final
/workspace/capitalization_embeddings/checkpoints/mlm/domain_mix_v2/capitalized_from_mixed_case_current_steps3000_lr2e5/final
```

Only compare these three V2 checkpoints against each other. Do not compare the
V2 capitalized checkpoint against older cased/uncased controls in a headline
table.

## V2 Goals

The V2 probe is not trying to prove that lower MLM loss alone is meaningful.
The decision rule is downstream:

1. Guardrail: preserve or improve the current token-task pattern on
   CoNLL-2003/WNUT-style NER.
2. Failure slices: improve the capitalized model on TREC Fine, TweetEval Emoji,
   and scientific relations relative to its current checkpoint.
3. Fairness: any improvement must survive comparison against cased and uncased
   models that received the same additional V2 pretraining.

If V2 improves the failure slices without damaging token-task guardrails, the
next GPU step is a longer matched V2 run, likely 10k or 20k steps for all three
model families. If V2 mostly improves all models equally or hurts the
capitalized guardrails, keep the current mixed-case/dropout checkpoint as the
paper model and report V2 as a negative/diagnostic result.
