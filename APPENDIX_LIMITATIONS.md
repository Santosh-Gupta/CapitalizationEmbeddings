# Appendix Limitations And Negative Controls

This appendix material should be included, not hidden. It makes the paper's
claim more credible by showing where capitalization embeddings do not solve the
problem.

## Negative And Boundary Results

| Benchmark | Best observed pattern | Interpretation | Paper placement |
| --- | --- | --- | --- |
| TweetEval Emoji | cased strongly beats both uncased and cap-embed; cap stays near uncased | Casing may encode expressive/social signal that is not recovered by a tiny case channel on top of uncased lexical IDs | negative-control appendix |
| TREC Fine | cap underperforms both matched baselines | Fine-grained question classification can favor cased lexical/tokenization details not captured by the factorized case feature | negative-control appendix |
| SciERC relations | uncased dominates | Scientific relation classification appears to benefit more from lexical unification/domain effects than from explicit case features | appendix limitation |
| Combined scientific relations | uncased dominates | Same family as SciERC/SemEval relation diagnostics; do not count as independent proof against the method, but do not claim success here | appendix limitation |
| SemEval18 Task 7 validation | cased dominates, high variance | The validation-only setup is noisy and should not be used as a headline result | diagnostic only |
| SciEntsBank 3-way UQ | uncased dominates | Short-answer grading is not consistently capitalization-sensitive in the way NER/POS are | appendix limitation |
| SciEntsBank 3-way UD | cased dominates | Results are mixed across SciEntsBank splits, so the family is not clean support for the main claim | appendix limitation |
| iSarcasmEval EN | cap ties both with small negative mean | Social sarcasm signal is noisy; current result is neutral rather than supportive | neutral appendix |

## How To Write The Limitation

Recommended wording:

```text
The factorized case channel is not a universal replacement for cased
tokenization. It is strongest when capitalization behaves as a reusable feature
over shared lexical items, as in token/entity tasks. When capitalization is
itself part of a task-specific social or lexical signal, or when downstream
performance depends on broader tokenizer differences, the method can tie or
underperform the stronger baseline.
```

Avoid:

```text
Capitalization embeddings combine the best of cased and uncased BERT across all
tasks.
```

## Reviewer-Risk Mitigation

Likely reviewer concern:

```text
The method fails on several benchmarks, so the claim is too broad.
```

Response:

```text
The claim is intentionally scoped. The main hypothesis is about factorizing
capitalization from lexical identity, which is most directly tested on
token/entity benchmarks where capitalization is a reusable lexical feature. The
negative controls show that the architecture is not a free universal gain and
help define when cased tokenization still carries useful information.
```

Likely reviewer concern:

```text
The reported memory saving is misleading because bert-base-cased is smaller
than bert-base-uncased.
```

Response:

```text
The paper does not claim parameter savings over the released bert-base-cased
checkpoint. The efficiency comparison is against explicit case-expanded
vocabulary designs that would allocate separate embeddings for lowercase,
first-cap, all-caps, and optionally mixed-case variants.
```

Likely reviewer concern:

```text
Non-significant differences do not prove equality.
```

Response:

```text
The paper uses non-inferiority language with predeclared practical margins for
"matches" claims, and superiority language only when paired/bootstrap intervals
and Holm-corrected tests support a win.
```
