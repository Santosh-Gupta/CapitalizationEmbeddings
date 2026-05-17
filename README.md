# CapitalizationEmbeddings

Experiments with factorized capitalization embeddings for BERT-style masked
language models.

The starting hypothesis is that cased lexical identity can be decomposed into:

1. the normal `bert-base-uncased` token ID, and
2. a tiny per-token capitalization feature ID.

The model adds a learned capitalization embedding to the normal BERT embedding
stack:

```text
word_embedding(input_ids)
+ position_embedding(position_ids)
+ token_type_embedding(token_type_ids)
+ capitalization_embedding(capitalization_ids)
```

## Capitalization IDs

Current feature vocabulary:

```text
0 = no capitalization feature / lowercase / special token
1 = first-cap word, e.g. Tom
2 = all-caps word, e.g. TOM, NASA
3 = mixed-case word, e.g. iPhone, eBay, McDonald
```

The first experiment routed mixed-case words such as `iPhone`, `eBay`, and
`McDonald` to `0`. The current best variant adds the fourth `MIXED_CASE`
feature and uses capitalization embedding dropout during continued pretraining.

## Repo Layout

```text
capitalization_embeddings/
  tokenization.py   # Hugging Face tokenizer wrapper that adds capitalization_ids
  modeling.py       # CapitalizedBertConfig/Model/ForMaskedLM/ForTokenClassification
  collator.py       # MLM collator with token labels and capitalization labels
notebooks/
  00_tokenization_and_model_smoke_test.ipynb
  00b_optional_model_smoke_test.ipynb
  01_continue_pretraining_mlm.ipynb
  02_finetune_conll2003_ner.ipynb
  03_finetune_conll2003_baselines.ipynb
tests/
  test_tokenization.py
```

## Experiment Status

The current evidence ledger is maintained in
[PAPER_EVIDENCE_STATUS.md](PAPER_EVIDENCE_STATUS.md). It records the current
best checkpoint family, benchmark roots, completed multi-seed results, negative
results, and the remaining statistical work needed before paper submission.

GPU and CPU experiment batches are tracked in [RUN_LEDGER.md](RUN_LEDGER.md).
Update that file before launching a run and after it completes.

Continued-pretraining corpora and matched-control rules are tracked in
[PRETRAINING_CORPORA.md](PRETRAINING_CORPORA.md). Update it before adding any
new MLM corpus or pretraining checkpoint family.

GPU-provider migration steps are tracked in
[TRANSFER_RUNBOOK.md](TRANSFER_RUNBOOK.md). Generate a transfer manifest before
moving checkpoints off the RunPod network volume.

## Colab Workflow

Clone or upload this repo to Colab, then run from the repo root:

```python
%pip install -q -e . -r requirements-colab.txt
```

If Colab changes installed packages, restart the runtime before continuing so
the notebook imports the pinned versions from `requirements-colab.txt`.

Recommended order:

1. `notebooks/00_tokenization_and_model_smoke_test.ipynb`
2. `notebooks/00b_optional_model_smoke_test.ipynb`
3. `notebooks/01_continue_pretraining_mlm.ipynb`
4. `notebooks/02_finetune_conll2003_ner.ipynb`
5. `notebooks/03_finetune_conll2003_baselines.ipynb`

Notebook `00` is tokenizer-only and does not load BERT weights. Notebook `00b`
is optional and only exists to verify the custom model forward pass.

The notebooks default to saving checkpoints under:

```text
/content/drive/MyDrive/capitalization_embeddings/
```

Model weights are intentionally ignored by git.

## Continued Pretraining Design

The MLM collator masks normal token IDs and also emits
`capitalization_labels`. For masked positions, the input `capitalization_ids`
are zeroed so the model cannot directly read the answer.

The MLM model optimizes:

```text
loss = token_mlm_loss + capitalization_loss_weight * capitalization_loss
```

`capitalization_loss_weight` defaults to `0.25` in `CapitalizedBertConfig`.

## First Evaluation Target

Use CoNLL-2003 NER as the first downstream evaluation because capitalization is
important for named-entity recognition. The key comparison should use matched
fine-tuning settings:

```text
bert-base-uncased
bert-base-cased
bert-base-uncased + capitalization embeddings
bert-base-uncased + capitalization embeddings + continued pretraining
```

## Local Smoke Checks

```bash
python -m compileall capitalization_embeddings tests
python -m unittest discover -s tests
```
