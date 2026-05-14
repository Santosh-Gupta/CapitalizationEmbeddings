# Holm-Corrected Benchmark Significance

Family: `sequence_headline`
Alpha: `0.05`
Practical equivalence threshold: `0.002`

| Benchmark | Comparison | n | Delta | CI95 | raw p | Holm p | Holm reject | Label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| sst5 | capitalized_pretrained>cased_pretrained | 20 | +0.012353 | [-0.015837, +0.039819] | 0.3839 | 1 | no | tie |
| sst5 | capitalized_pretrained>uncased_pretrained | 20 | -0.000339 | [-0.021267, +0.018100] | 1 | 1 | no | tie |
| tweet_eval_irony | capitalized_pretrained>cased_pretrained | 20 | +0.019679 | [-0.020084, +0.062353] | 0.3834 | 1 | no | tie |
| tweet_eval_irony | capitalized_pretrained>uncased_pretrained | 20 | +0.008566 | [-0.028787, +0.047221] | 0.6442 | 1 | no | tie |
| tweet_eval_offensive | capitalized_pretrained>cased_pretrained | 20 | +0.013305 | [-0.015279, +0.043573] | 0.3844 | 1 | no | tie |
| tweet_eval_offensive | capitalized_pretrained>uncased_pretrained | 20 | +0.006566 | [-0.018079, +0.029941] | 0.5775 | 1 | no | tie |
| twenty_newsgroups | capitalized_pretrained>cased_pretrained | 20 | +0.010761 | [+0.001328, +0.019384] | 0.0285 | 0.228 | no | inconclusive_positive |
| twenty_newsgroups | capitalized_pretrained>uncased_pretrained | 20 | -0.000843 | [-0.009825, +0.008364] | 0.8316 | 1 | no | tie |
