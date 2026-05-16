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
PubMed summarization
BillSum
LexGLUE SCOTUS
SciERC train text
SciEntsBank train text
TweetEval train text: emoji, irony, offensive, sentiment, emotion
TREC train text
SST-5 train text
20 Newsgroups train text
ACL citation sentiment text
```

Selection:

```text
score rows by first-cap, all-caps, and mixed-case signals
deduplicate normalized rows
keep the top 180,000 rows with score >= 2
hold out up to the first 10,000 selected rows for MLM eval
use the remaining rows for MLM training
```

Cache:

```text
/workspace/capitalization_embeddings/prepared_corpora/domain_mix_v2_rows.jsonl.gz
```

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
