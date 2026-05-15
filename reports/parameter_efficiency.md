# Parameter Efficiency

All counts are parameter counts from BERT configuration shapes, not loaded weights.
The masked-LM count assumes tied decoder weights, matching standard BERT.

## Model Counts

| Model | Vocab | Encoder+pooler params | MLM params | Extra encoder vs uncased | Extra MLM vs uncased | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bert-base-uncased | 30,522 | 109,482,240 | 109,514,298 | +0 | +0 | standard uncased baseline |
| bert-base-cased | 28,996 | 108,310,272 | 108,340,804 | -1,171,968 | -1,173,494 | standard cased baseline; its vocabulary is not a 3x uncased expansion |
| capitalized-bert, 4 case states | 30,522 | 109,485,312 | 109,520,446 | +3,072 | +6,148 | adds only capitalization embeddings at downstream time; MLM also has an auxiliary capitalization head |
| hypothetical 3x uncased case-expanded vocab | 91,566 | 156,364,032 | 156,457,134 | +46,881,792 | +46,942,836 | separate lowercase/first-cap/all-caps token embeddings |
| hypothetical 4x uncased case-expanded vocab | 122,088 | 179,804,928 | 179,928,552 | +70,322,688 | +70,414,254 | adds a separate mixed-case token variant as well |

## Key Ratios

- Capitalization embedding overhead with four case states: `3,072` parameters.
- Auxiliary MLM capitalization head: `3,076` parameters.
- Three-way case-expanded uncased vocabulary extra word embeddings: `46,881,792` parameters.
- Four-way case-expanded uncased vocabulary extra word embeddings: `70,322,688` parameters.
- 3x expansion is `15,261x` larger than the four-state capitalization embedding table.
- 4x expansion is `22,892x` larger than the four-state capitalization embedding table.

## Paper-Framing Note

The memory-efficiency claim should be made against an explicit case-expanded uncased vocabulary design, not against `bert-base-cased` directly. `bert-base-cased` has a smaller vocabulary than `bert-base-uncased`, so the capitalized model is slightly larger than the released cased baseline while still being dramatically smaller than a vocabulary-tripling design.
