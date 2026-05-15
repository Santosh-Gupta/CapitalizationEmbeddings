# Citation Status

This file tracks bibliography readiness separately from model results so the
paper does not lose citation work across context compaction.

## Verified Starter Bibliography

The starter BibTeX file is:

```text
paper/references.bib
```

It currently covers:

| Area | BibTeX key | Source |
| --- | --- | --- |
| BERT | `devlin-etal-2019-bert` | ACL Anthology |
| Hugging Face Transformers | `wolf-etal-2020-transformers` | ACL Anthology |
| Hugging Face Datasets | `lhoest-etal-2021-datasets` | ACL Anthology |
| WordPiece/subword background | `wu-etal-2016-gnmt` | arXiv |
| Original WordPiece citation | `schuster-nakajima-2012-voice` | Google Research/IEEE |
| CoNLL-2003 | `tjong-kim-sang-de-meulder-2003-introduction` | ACL Anthology |
| WNUT-17 | `derczynski-etal-2017-results` | ACL Anthology |
| OntoNotes v5 | `weischedel-etal-2013-ontonotes` | LDC catalog |
| Penn Treebank | `marcus-etal-1993-penn-treebank` | Computational Linguistics |
| TweetEval | `barbieri-etal-2020-tweeteval` | ACL Anthology |
| SST | `socher-etal-2013-recursive` | ACL Anthology |
| TREC question classification | `li-roth-2002-learning` | ACL Anthology |
| iSarcasmEval | `abu-farha-etal-2022-semeval` | ACL Anthology |
| Kaggle/Walia NER mirror | `rjac-2022-kaggle-entity-annotated-corpus` | Hugging Face dataset card |
| Groningen Meaning Bank source corpus | `bos-etal-2017-groningen` | University of Groningen/Springer |
| SemEval18 scientific relations | `gabor-etal-2018-semeval` | ACL Anthology |
| SciERC scientific relations | `luan-etal-2018-multi` | ACL Anthology |
| SciEntsBank/SemEval13 short-answer grading | `dzikovska-etal-2013-semeval` | ACL Anthology |
| ACL citation sentiment corpus | `athar-2011-sentiment` | ACL Anthology |
| 20 Newsgroups origin | `lang-1995-newsweeder` | ICML/Elsevier |
| Paired bootstrap precedent | `koehn-2004-statistical` | ACL Anthology |
| Holm correction | `holm-1979-simple` | Scandinavian Journal of Statistics |

## Citation Caveats To Resolve

- `citation_sentiment_acl`: only cite and report this if we actually run it.
- `kaggle_walia_ner` should still be described as supporting evidence because
  the exact benchmark is a derived Kaggle/Hugging Face mirror with a
  project-created split, not a canonical peer-reviewed benchmark split.
- Scientific relation and SciEntsBank diagnostics now have source citations, but
  should remain appendix/negative-control evidence unless the claim changes.

## Submission-Readiness Rule

Before submission, every benchmark in the main table must have:

1. a BibTeX entry;
2. a one-sentence dataset provenance note;
3. an explicit statement of whether the split is canonical, Hugging Face
   canonical, or project-created.
