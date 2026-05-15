# Paper Result Tables

Generated from report JSON files. Do not hand-edit numeric values here.

## Main Results

| Family | Benchmark | Metric | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased | Labels |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| token/entity | CoNLL-2003 NER | entity F1 | 0.9040 +/- 0.0025, n=20 | 0.9119 +/- 0.0035, n=20 | 0.9165 +/- 0.0018, n=20 | +0.0125 | +0.0045 | vs uncased: win; vs cased: tie |
| token/entity | Kaggle/Walia NER | entity F1 | 0.8270 +/- 0.0060, n=5 | 0.8437 +/- 0.0061, n=5 | 0.8422 +/- 0.0074, n=5 | +0.0152 | -0.0015 | vs uncased: win; vs cased: tie |
| token/entity | WNUT-17 NER | entity F1 | 0.4424 +/- 0.0152, n=20 | 0.4426 +/- 0.0100, n=20 | 0.4495 +/- 0.0103, n=20 | +0.0071 | +0.0069 | vs uncased: tie; vs cased: tie |
| uncased-favored control | SST-5 | accuracy | 0.5410 +/- 0.0039, n=20 | 0.5283 +/- 0.0084, n=20 | 0.5407 +/- 0.0068, n=20 | -0.0003 | +0.0124 | vs uncased: tie; vs cased: tie |
| uncased-favored control | TweetEval Irony | macro-F1 | 0.6702 +/- 0.0135, n=20 | 0.6591 +/- 0.0144, n=20 | 0.6788 +/- 0.0092, n=20 | +0.0086 | +0.0197 | vs uncased: tie; vs cased: tie |
| uncased-favored control | TweetEval Offensive | macro-F1 | 0.8034 +/- 0.0091, n=20 | 0.7966 +/- 0.0073, n=20 | 0.8099 +/- 0.0065, n=20 | +0.0066 | +0.0133 | vs uncased: tie; vs cased: tie |
| uncased-favored control | 20 Newsgroups | accuracy | 0.7055 +/- 0.0028, n=20 | 0.6939 +/- 0.0023, n=20 | 0.7047 +/- 0.0031, n=20 | -0.0008 | +0.0108 | vs uncased: tie; vs cased: inconclusive_positive |
| appendix/neutral | iSarcasmEval EN | macro-F1 | 0.6076 +/- 0.0175, n=5 | 0.6075 +/- 0.0109, n=5 | 0.6032 +/- 0.0174, n=5 | -0.0044 | -0.0043 | vs uncased: tie; vs cased: tie |

## Cap-Embedding Comparison Intervals

| Benchmark | Cap-Uncased delta | Cap-Uncased CI95 | Cap-Cased delta | Cap-Cased CI95 |
| --- | ---: | ---: | ---: | ---: |
| CoNLL-2003 NER | +0.0125 | [+0.0034, +0.0196] | +0.0045 | [-0.0055, +0.0181] |
| Kaggle/Walia NER | +0.0152 | [+0.0087, +0.0213] | -0.0015 | [-0.0092, +0.0050] |
| WNUT-17 NER | +0.0071 | [-0.0314, +0.0480] | +0.0069 | [-0.0300, +0.0430] |
| SST-5 | -0.0003 | [-0.0213, +0.0181] | +0.0124 | [-0.0158, +0.0398] |
| TweetEval Irony | +0.0086 | [-0.0288, +0.0472] | +0.0197 | [-0.0201, +0.0624] |
| TweetEval Offensive | +0.0066 | [-0.0181, +0.0299] | +0.0133 | [-0.0153, +0.0436] |
| 20 Newsgroups | -0.0008 | [-0.0098, +0.0084] | +0.0108 | [+0.0013, +0.0194] |
| iSarcasmEval EN | -0.0044 | [-0.0439, +0.0450] | -0.0043 | [-0.0505, +0.0494] |
