# Benchmark Sweep Summary

Metric: `seqeval_f1`

| Benchmark | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased |
| --- | ---: | ---: | ---: | ---: | ---: |
| conll2003_ner | 0.9040 +/- 0.0025 | 0.9119 +/- 0.0035 | 0.9165 +/- 0.0018 | +0.0125 | +0.0045 |
| wnut17_ner | 0.4424 +/- 0.0152 | 0.4426 +/- 0.0100 | 0.4495 +/- 0.0103 | +0.0071 | +0.0069 |

## Comparisons

### conll2003_ner

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | +0.012525 | [+0.003364, +0.019591] | 0.0086 |
| capitalized_pretrained | cased_pretrained | +0.004537 | [-0.005488, +0.018059] | 0.2936 |
| cased_pretrained | uncased_pretrained | +0.007988 | [-0.006975, +0.016462] | 0.1235 |

### wnut17_ner

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | +0.007103 | [-0.031437, +0.047961] | 0.7288 |
| capitalized_pretrained | cased_pretrained | +0.006910 | [-0.029998, +0.043024] | 0.7233 |
| cased_pretrained | uncased_pretrained | +0.000193 | [-0.038153, +0.041617] | 0.9941 |
