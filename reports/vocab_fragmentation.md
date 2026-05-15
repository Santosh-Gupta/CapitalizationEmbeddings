# Vocabulary Fragmentation

Cased model: `bert-base-cased`
Uncased model: `bert-base-uncased`

- Cased vocab size: `28,996`
- Uncased vocab size: `30,522`
- Cased vocab tokens with first-cap/all-caps/mixed-case form: `8,368`
- First-cap/all-caps cased tokens: `8,321`
- First-cap/all-caps cased tokens whose lowercase form exists in the uncased vocab: `8,172`

## Cased Vocabulary by Case Class

| Case class | Count | Share of cased vocab |
| --- | ---: | ---: |
| special | 5 | 0.02% |
| no_alpha | 1,155 | 3.98% |
| lower_or_uncased | 19,468 | 67.14% |
| first_cap | 7,721 | 26.63% |
| all_caps | 600 | 2.07% |
| mixed_case | 47 | 0.16% |

## Lowercase Counterpart Coverage

| Case class | Tokens | Lowercase in cased vocab | Lowercase in uncased vocab |
| --- | ---: | ---: | ---: |
| first_cap | 7,721 | 3,603 | 7,585 |
| all_caps | 600 | 247 | 587 |
| mixed_case | 47 | 0 | 47 |

## Case Families in the Cased Vocab

| Statistic | Count |
| --- | ---: |
| alphabetic_families | 23,967 |
| families_with_multiple_case_forms | 3,819 |
| families_with_lower_and_first_cap | 3,601 |
| families_with_lower_and_all_caps | 247 |
| families_with_lower_first_and_all_caps | 50 |
| families_with_mixed_case | 47 |

## Illustrative Case Families

| Lowercase family | Cased-vocab forms | Classes |
| --- | --- | --- |
| apple | `Apple`, `apple` | first_cap, lower_or_uncased |
| us | `US`, `Us`, `us` | all_caps, first_cap, lower_or_uncased |
| ##s | `##S`, `##s` | first_cap, lower_or_uncased |
| ##ed | `##ED`, `##ed` | all_caps, lower_or_uncased |
| ##as | `##AS`, `##As`, `##as` | all_caps, first_cap, lower_or_uncased |
| ##cs | `##CS`, `##Cs`, `##cs` | all_caps, first_cap, lower_or_uncased |
| ##ds | `##DS`, `##Ds`, `##ds` | all_caps, first_cap, lower_or_uncased |
| ##es | `##ES`, `##Es`, `##es` | all_caps, first_cap, lower_or_uncased |
| ##ms | `##MS`, `##Ms`, `##ms` | all_caps, first_cap, lower_or_uncased |
| ##os | `##OS`, `##Os`, `##os` | all_caps, first_cap, lower_or_uncased |

## Paper-Framing Note

The cased BERT vocabulary already contains many surface forms that can be described as a lowercase WordPiece plus a small case feature. This supports the factorization hypothesis. The mixed-case bucket should be framed as a pragmatic extension: it lets `iPhone`/`eBay`-style forms carry a case signal without creating a separate lexical embedding for each surface form.
