# Wrap-Up Plan

The project is now being wrapped as a technical blog post rather than a
conference submission. The goal is to present the idea, implementation, and
current experimental evidence honestly, without spending more GPU time.

## Publishable Claim

Use this claim:

```text
Capitalization embeddings are a lightweight way to add case information to an
uncased BERT-style model. They are most useful on token/entity tasks where case
acts like a reusable feature, while sequence-task results are mixed.
```

Avoid these claims:

```text
Capitalization embeddings beat cased BERT generally.
Capitalization embeddings solve the cased-vs-uncased tradeoff everywhere.
The current results are paper-final.
```

## Main Artifact

Public-facing blog post:

```text
README.md
```

`BLOG_POST.md` is kept as the source copy, but the README now mirrors the blog
post so the repository landing page is the writeup. The draft uses existing
completed results only. It intentionally does not depend on V3 pretraining or
additional GPU experiments.

## Supporting Artifacts

Useful supporting files:

```text
reports/paper_tables.md
reports/parameter_efficiency.md
reports/vocab_fragmentation.md
PAPER_READINESS.md
PRETRAINING_CORPORA.md
PRETRAINING_V3_PLAN.md
```

If publishing externally, the most useful figures/tables are:

1. Parameter overhead table.
2. Cased-vocabulary fragmentation table.
3. Main benchmark table from `reports/paper_tables.md`.
4. A simple architecture diagram showing word + position + segment + case
   embeddings.

## No More GPU Needed For Blog Version

Do not rent another GPU for the blog version unless the goal changes again.

Current completed evidence is enough to tell a useful story:

- CoNLL-2003 and Kaggle/Walia NER show clear cap-over-uncased wins.
- WNUT, SST-5, Irony, Offensive, 20 Newsgroups, and iSarcasmEval show mostly
  tie/neutral behavior.
- TweetEval Emoji, TREC Fine, scientific relations, and SciEntsBank define the
  limitation boundary.

## Final Cleanup Tasks

Before posting:

1. Decide whether to include exact benchmark numbers inline or move some to an
   appendix section.
2. Add a small architecture diagram.
3. Add links to the repo and key source files.
4. Mention that related prior work exists and this is an independent
   implementation/exploration.
5. Remove paper-submission language from the public-facing README if desired.
