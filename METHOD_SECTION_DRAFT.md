# Method Section Draft

## Factorized Capitalization Embeddings

Standard BERT tokenization exposes a tradeoff between lexical sharing and case
preservation. An uncased tokenizer maps surface forms such as `tom`, `Tom`, and
`TOM` to the same lexical WordPiece ID, which concentrates lexical evidence but
removes capitalization. A cased tokenizer can preserve capitalization, but it
often assigns separate lexical embeddings to case variants that are otherwise
closely related. This splits gradient updates across multiple embeddings even
when the underlying lexical item is the same.

We replace this lexical split with a factorized representation. Each input token
is represented by an uncased WordPiece ID and a parallel capitalization feature
ID. The model uses the standard BERT embedding sum, augmented with a learned
capitalization embedding:

```text
x_i = E_word(w_i) + E_position(i) + E_segment(s_i) + E_case(c_i)
```

where `w_i` is the uncased WordPiece ID and `c_i` is a small discrete case
feature. In the current model, `c_i` takes four values:

```text
0: no capitalization feature, lowercase, punctuation, or special token
1: first-cap
2: all-caps
3: mixed-case
```

The first-cap class covers source spans such as `Tom`, the all-caps class
covers spans such as `NASA`, and the mixed-case class covers spans such as
`iPhone`, `eBay`, and `McDonald`. The lexical WordPiece path remains uncased in
all cases, so case variants share the same word embedding while the model still
receives a learned case signal.

## Tokenization

Capitalization IDs are computed before the uncased tokenizer normalizes the
input. For raw text, we use fast-tokenizer character offsets to recover the
whitespace-delimited source span corresponding to each WordPiece. For
pre-tokenized sequence-labeling datasets, we use the tokenizer's `word_ids`
mapping and assign every WordPiece derived from the same source word the same
capitalization feature. Special tokens, punctuation-only spans, and padding use
the no-capitalization feature.

This design intentionally avoids creating new lexical token IDs for
capitalized variants. The token ID remains inside the original uncased
vocabulary, and the case feature is passed as a parallel tensor
(`capitalization_ids`) to the model.

## Continued Pretraining

We initialize the capitalized model from `bert-base-uncased`. All original BERT
parameters are loaded from the uncased checkpoint where shapes match. The new
capitalization embedding table is randomly initialized. During continued masked
language-model pretraining, we use the standard MLM objective plus an auxiliary
capitalization prediction objective:

```text
L = L_MLM + lambda L_case
```

The auxiliary head predicts the capitalization class at masked positions. At
masked positions, the input capitalization feature is zeroed so that the model
cannot trivially copy the feature into the prediction head. This forces the
model to infer capitalization from context and lexical evidence.

The current best checkpoint uses:

```text
lambda = 0.25
capitalization class weights = [1, 2, 8, 4]
capitalization embedding dropout = 0.1
continued-pretraining steps = 3000
learning rate = 2e-5
```

The class weights compensate for the rarity of all-caps and mixed-case spans.
Capitalization embedding dropout regularizes the case channel so downstream
models can ignore it when case is noisy or uninformative.

## Matched Baselines

To isolate architectural effects from additional pretraining exposure, all main
comparisons use matched continued-pretraining controls. The uncased baseline is
`bert-base-uncased` continued on the same real-acronym corpus. The cased
baseline is `bert-base-cased` continued on the same corpus with the same step
count and learning rate. The proposed model is compared against these matched
controls, not only against the original released checkpoints.

## Downstream Fine-Tuning

For downstream token and sequence classification, the auxiliary capitalization
prediction head is not used. The only architectural difference from BERT is the
additional capitalization embedding in the input representation. Token
classification uses the final hidden state for each token; sequence
classification uses the pooled representation, matching standard BERT
fine-tuning practice.

The downstream tokenizer remains the uncased tokenizer for the proposed model.
The same preprocessing function emits both `input_ids` and `capitalization_ids`,
so inference requires no vocabulary expansion and no routing through enlarged
case-specific token IDs.

## Parameter Overhead

With BERT-base hidden size 768 and four case states, the downstream
capitalization embedding table contains only:

```text
4 * 768 = 3,072 parameters
```

The auxiliary MLM capitalization head adds `3,076` parameters during
pretraining. By contrast, explicitly tripling the uncased vocabulary to allocate
separate lowercase, first-cap, and all-caps embeddings would add `46,881,792`
word-embedding parameters. This is the parameter-efficiency comparison used in
the paper.

The released `bert-base-cased` checkpoint has a smaller vocabulary than
`bert-base-uncased`, so the proposed model is not claimed to be smaller than the
released cased baseline. The efficiency claim is specifically against explicit
case-expanded vocabulary designs.
