# Paper Readiness Plan

This project is now in paper-hardening mode. The next work should improve the
clarity, reproducibility, and reviewer defensibility of the claim before buying
more GPU time.

## Defensible Claim

Do not claim that capitalization embeddings beat cased and uncased BERT on every
task. The current evidence supports a narrower, stronger claim:

```text
Capitalization embeddings factor case away from the lexical WordPiece identity.
They preserve uncased lexical sharing while giving the model a small learned
case channel that is useful on token/entity-heavy tasks and usually ignorable on
tasks where case is noisy or weakly predictive.
```

The paper should distinguish three outcome types:

- `win`: capitalization embeddings beat a baseline with a paired/bootstrap
  interval that excludes zero after the relevant correction.
- `tie`: capitalization embeddings are non-inferior to the better baseline under
  a predeclared practical margin, or the paired interval includes zero.
- `loss`: capitalization embeddings underperform and the result should be
  treated as a limitation or diagnostic boundary.

## Theory Evidence Already Added

CPU-only reports now support the motivation and parameter-efficiency argument:

```text
reports/parameter_efficiency.md
reports/vocab_fragmentation.md
```

Key parameter facts from `reports/parameter_efficiency.md`:

- Four-state capitalization embedding overhead: `3,072` parameters.
- Auxiliary MLM capitalization head: `3,076` parameters.
- A hypothetical lowercase/first-cap/all-caps uncased vocabulary expansion would
  add `46,881,792` word-embedding parameters.
- A four-way expansion with a mixed-case lexical variant would add `70,322,688`
  word-embedding parameters.
- The memory-efficiency claim should be made against explicit case-expanded
  vocabulary designs, not against `bert-base-cased`, because released
  `bert-base-cased` has a smaller vocabulary than `bert-base-uncased`.

Key vocabulary facts from `reports/vocab_fragmentation.md`:

- `bert-base-cased` contains `8,368` first-cap/all-caps/mixed-case WordPieces.
- `8,172` first-cap/all-caps cased WordPieces have lowercase counterparts in
  the `bert-base-uncased` vocabulary.
- `3,819` alphabetic case families in the cased vocabulary have multiple case
  forms.
- This directly supports the factorization hypothesis: many cased WordPieces can
  be represented as a lowercase lexical item plus a small case feature.

## Current Evidence Read

Main positive token/entity evidence:

- CoNLL-2003 NER, 20 seeds: cap beats uncased and ties cased.
- Kaggle/Walia NER, 5 seeds: cap beats uncased and ties cased.
- OntoNotes v5 NER, 5 seeds: cap beats uncased and nearly ties cased.
- PTB POS, 5 seeds: cap beats uncased and nearly ties cased.
- WNUT-17 NER, 20 seeds: cap is positive by mean against both baselines, but
  high variance keeps the statistical label at tie.

Uncased-favored/general-task evidence:

- SST-5 and 20 Newsgroups: cap matches uncased and avoids most of the cased
  penalty.
- TweetEval Irony and Offensive: cap is positive by mean against both baselines,
  but not Holm-significant.

Boundary/negative-control evidence:

- TweetEval Emoji: cased dominates and cap remains near uncased.
- TREC Fine: cap underperforms both baselines.
- Scientific relation and SciEntsBank diagnostics do not support a broad
  "best of cased and uncased everywhere" claim.
- iSarcasmEval is neutral: cap ties both baselines with a small negative mean.

## Main Paper Table Recommendation

Primary table:

| Family | Benchmarks | Purpose |
| --- | --- | --- |
| Token/entity capitalization | CoNLL-2003, WNUT-17, OntoNotes v5, PTB POS, Kaggle/Walia NER | Main evidence for recovering cased-model signal with lexical sharing |
| Uncased-favored controls | SST-5, 20 Newsgroups, TweetEval Irony, TweetEval Offensive | Tests whether the added case channel damages uncased-style behavior |
| Negative controls / appendix | TweetEval Emoji, TREC Fine, SciRel, SciEntsBank, iSarcasmEval | Defines boundary conditions and prevents overclaiming |

## Non-GPU Work Completed

Completed in the current paper-hardening pass:

1. Locked experimental protocol:
   `EXPERIMENTAL_PROTOCOL.md`.
2. GPU ablation matrix:
   `ABLATION_MATRIX.md`.
3. Paper-ready result tables generated from JSON:
   `reports/paper_tables.md`.
4. Parameter-efficiency report:
   `reports/parameter_efficiency.md`.
5. Vocabulary-fragmentation report:
   `reports/vocab_fragmentation.md`.
6. Token error-analysis tool:
   `scripts/error_analysis_by_case.py`.
7. Manuscript outline:
   `MANUSCRIPT_OUTLINE.md`.
8. Appendix limitations and negative-control framing:
   `APPENDIX_LIMITATIONS.md`.
9. First method-section prose draft:
   `METHOD_SECTION_DRAFT.md`.
10. Full manuscript draft scaffold:
    `paper/draft.md`.
11. Starter bibliography and citation caveat tracker:
    `paper/references.bib`, `CITATION_STATUS.md`.
12. Dataset provenance table for main, control, and appendix benchmarks:
    `DATASET_PROVENANCE.md`.

## Non-GPU Work Remaining

These can still be done before restarting the pod:

1. Run error analysis on saved token prediction files after choosing the exact
   main-table prediction roots to inspect.
2. Convert `paper/draft.md` from scaffold prose into the submission manuscript
   after ablations/error analysis are available.

## Next GPU Gate

The non-GPU protocol and ablation matrix are locked. Restart RunPod only when
ready to run the GPU ablations or seed expansions below.

When GPU work resumes, the highest-value runs are:

1. Ablations on CoNLL/Walia:
   - no capitalization embedding
   - three-state first/all only
   - four-state mixed-case
   - four-state mixed-case + capitalization embedding dropout
2. Expand Walia NER from 5 to 20 seeds if it remains a main-table benchmark.
3. Expand OntoNotes and PTB from 5 to 20 seeds only if the paper needs stronger
   token-task replication beyond CoNLL/Walia/WNUT.

Do not spend more GPU on TweetEval Emoji, TREC Fine, scientific relations, or
SciEntsBank unless the paper explicitly needs negative-control replication.
