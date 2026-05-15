# Manuscript Outline

Working title:

```text
Factorized Capitalization Embeddings for Lexically Shared BERT Models
```

## Abstract

Core message:

- Cased tokenizers preserve capitalization but split lexical evidence across
  surface forms.
- Uncased tokenizers share lexical evidence but discard case.
- Capitalization embeddings factor the two signals: an uncased WordPiece ID plus
  a small learned case feature.
- Experiments show strongest gains on token/entity-heavy benchmarks, where the
  method beats uncased BERT and often matches cased BERT, with negligible
  parameter overhead relative to case-expanded vocabularies.
- The method is not universally superior; negative controls define its boundary.

## 1. Introduction

Claims to make:

1. Capitalization is a useful linguistic feature, especially for entities,
   acronyms, and proper nouns.
2. Treating `tom`, `Tom`, and `TOM` as separate lexical embeddings fragments
   training signal.
3. Fully lowercasing removes useful case information.
4. A factorized representation can preserve lexical sharing and expose case as
   a learnable feature.

Avoid:

- Do not claim universal dominance over both cased and uncased BERT.
- Do not claim memory savings against released `bert-base-cased`; the correct
  comparison is against explicit case-expanded vocabulary designs.

## 2. Background And Motivation

Use `reports/vocab_fragmentation.md`:

- `bert-base-cased` has `8,368` first-cap/all-caps/mixed-case WordPieces.
- `8,172` first-cap/all-caps cased WordPieces have lowercase counterparts in
  `bert-base-uncased`.
- `3,819` cased-vocab alphabetic families have multiple case forms.

Argument:

These statistics show that many cased WordPieces are not semantically distinct
lexical items. They are surface-case variants that can be represented by a
shared lexical embedding plus a small case feature.

## 3. Method

Define the input representation:

```text
embedding = word_embedding(lowercase_wordpiece_id)
          + position_embedding
          + token_type_embedding
          + capitalization_embedding(case_id)
```

Case states:

```text
0 none/lower/special
1 first-cap
2 all-caps
3 mixed-case
```

Pretraining objective:

```text
loss = MLM loss + lambda * capitalization prediction loss
```

Current best checkpoint:

```text
lambda = 0.25
class weights = [1, 2, 8, 4]
capitalization embedding dropout = 0.1
real-acronym continuation = 3000 steps at 2e-5
```

## 4. Parameter Efficiency

Use `reports/parameter_efficiency.md`.

Main numbers:

| Design | Extra encoder params vs uncased |
| --- | ---: |
| capitalized BERT, 4 case states | +3,072 |
| hypothetical 3x uncased case-expanded vocab | +46,881,792 |
| hypothetical 4x uncased case-expanded vocab | +70,322,688 |

Key framing:

The factorized method adds almost no downstream parameters. The auxiliary MLM
capitalization head is only a pretraining-time addition and is also tiny
(`3,076` parameters).

## 5. Experimental Setup

Point to `EXPERIMENTAL_PROTOCOL.md`.

Main elements:

- matched continued-pretraining for uncased, cased, and capitalized models;
- fixed current-best capitalized checkpoint before final benchmark tables;
- identical fine-tuning hyperparameters within task family;
- saved predictions for paired bootstrap;
- seed-level replication and paired-bootstrap example-level uncertainty;
- Holm correction within benchmark families;
- non-inferiority framing for "matches" claims.

## 6. Results

Use `reports/paper_tables.md`.

Primary result story:

1. Token/entity benchmarks:
   - CoNLL: cap beats uncased and ties cased at 20 seeds.
   - Walia: cap beats uncased and ties cased at 5 seeds.
   - WNUT: positive mean against both, but high variance.
   - OntoNotes/PTB: supporting 5-seed evidence from `PAPER_EVIDENCE_STATUS.md`.
2. Uncased-favored controls:
   - cap matches uncased on SST-5 and 20 Newsgroups;
   - cap has positive mean on TweetEval Irony/Offensive but not
     Holm-significant.
3. Boundary conditions:
   - TweetEval Emoji and TREC Fine are misses;
   - scientific relation/SciEntsBank diagnostics limit the broad claim.

## 7. Ablations

Use `ABLATION_MATRIX.md`.

Needed GPU work before final submission:

- 4-state with versus without capitalization embedding dropout.
- 3-state first/all only versus 4-state mixed-case.
- with versus without auxiliary capitalization loss.
- no-capitalization-embedding control.

The ablation table is the strongest remaining experimental gap.

## 8. Error Analysis

Use `scripts/error_analysis_by_case.py` once prediction files are selected.

Planned analyses:

- token correctness by source capitalization class;
- entity-token correctness by source capitalization class;
- all-caps acronym behavior;
- mixed-case behavior;
- examples where cased wins but cap-embed fails.

## 9. Limitations

State explicitly:

- The method is not a universal replacement for cased tokenization.
- It can fail when capitalization is highly semantic or task-specific, as in
  TweetEval Emoji.
- It can underperform both baselines on some sequence tasks, as in TREC Fine.
- Some current supporting token results are still 5-seed, not 20-seed.
- The method has only been tested on BERT-base-style encoders so far.

## 10. Conclusion

Final message:

Capitalization embeddings are a simple, parameter-efficient way to factor case
from lexical identity. They are best framed as a targeted architecture for
recovering case-sensitive token/entity behavior while retaining uncased lexical
sharing, not as a universal benchmark winner.
