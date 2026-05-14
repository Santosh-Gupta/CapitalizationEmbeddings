# Holm-Corrected Benchmark Significance

Family: `token_headline`
Alpha: `0.05`
Practical equivalence threshold: `0.002`

| Benchmark | Comparison | n | Delta | CI95 | raw p | Holm p | Holm reject | Label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| conll2003_ner | capitalized_pretrained>cased_pretrained | 20 | +0.004537 | [-0.005488, +0.018059] | 0.2936 | 0.8808 | no | tie |
| conll2003_ner | capitalized_pretrained>uncased_pretrained | 20 | +0.012525 | [+0.003364, +0.019591] | 0.0086 | 0.0344 | yes | win |
| wnut17_ner | capitalized_pretrained>cased_pretrained | 20 | +0.006910 | [-0.029998, +0.043024] | 0.7233 | 1 | no | tie |
| wnut17_ner | capitalized_pretrained>uncased_pretrained | 20 | +0.007103 | [-0.031437, +0.047961] | 0.7288 | 1 | no | tie |
