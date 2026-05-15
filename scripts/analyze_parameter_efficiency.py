#!/usr/bin/env python
"""Parameter accounting for capitalization-embedding BERT variants.

This script uses model configs, not downloaded model weights. It is intended to
support paper tables that distinguish the actual capitalization-embedding
overhead from a hypothetical case-expanded vocabulary design.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import BertConfig


@dataclass(frozen=True)
class BertShape:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    type_vocab_size: int


@dataclass(frozen=True)
class CountRow:
    name: str
    vocab_size: int
    encoder_with_pooler_params: int
    masked_lm_params: int
    extra_vs_uncased_encoder: int
    extra_vs_uncased_mlm: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncased-model", default="bert-base-uncased")
    parser.add_argument("--cased-model", default="bert-base-cased")
    parser.add_argument("--capitalization-vocab-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=Path("reports/parameter_efficiency.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/parameter_efficiency.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uncased = shape_from_config(BertConfig.from_pretrained(args.uncased_model))
    cased = shape_from_config(BertConfig.from_pretrained(args.cased_model))
    cap_vocab = args.capitalization_vocab_size

    uncased_encoder = bert_encoder_with_pooler_params(uncased)
    uncased_mlm = bert_masked_lm_params(uncased)
    cap_encoder = bert_encoder_with_pooler_params(uncased) + capitalization_embedding_params(
        uncased, cap_vocab
    )
    cap_mlm = (
        bert_masked_lm_params(uncased)
        + capitalization_embedding_params(uncased, cap_vocab)
        + capitalization_classifier_params(uncased, cap_vocab)
    )

    expanded_3x = BertShape(
        vocab_size=uncased.vocab_size * 3,
        hidden_size=uncased.hidden_size,
        intermediate_size=uncased.intermediate_size,
        num_hidden_layers=uncased.num_hidden_layers,
        max_position_embeddings=uncased.max_position_embeddings,
        type_vocab_size=uncased.type_vocab_size,
    )
    expanded_4x = BertShape(
        vocab_size=uncased.vocab_size * 4,
        hidden_size=uncased.hidden_size,
        intermediate_size=uncased.intermediate_size,
        num_hidden_layers=uncased.num_hidden_layers,
        max_position_embeddings=uncased.max_position_embeddings,
        type_vocab_size=uncased.type_vocab_size,
    )

    rows = [
        CountRow(
            name=args.uncased_model,
            vocab_size=uncased.vocab_size,
            encoder_with_pooler_params=uncased_encoder,
            masked_lm_params=uncased_mlm,
            extra_vs_uncased_encoder=0,
            extra_vs_uncased_mlm=0,
            note="standard uncased baseline",
        ),
        CountRow(
            name=args.cased_model,
            vocab_size=cased.vocab_size,
            encoder_with_pooler_params=bert_encoder_with_pooler_params(cased),
            masked_lm_params=bert_masked_lm_params(cased),
            extra_vs_uncased_encoder=bert_encoder_with_pooler_params(cased) - uncased_encoder,
            extra_vs_uncased_mlm=bert_masked_lm_params(cased) - uncased_mlm,
            note="standard cased baseline; its vocabulary is not a 3x uncased expansion",
        ),
        CountRow(
            name=f"capitalized-bert, {cap_vocab} case states",
            vocab_size=uncased.vocab_size,
            encoder_with_pooler_params=cap_encoder,
            masked_lm_params=cap_mlm,
            extra_vs_uncased_encoder=cap_encoder - uncased_encoder,
            extra_vs_uncased_mlm=cap_mlm - uncased_mlm,
            note="adds only capitalization embeddings at downstream time; MLM also has an auxiliary capitalization head",
        ),
        CountRow(
            name="hypothetical 3x uncased case-expanded vocab",
            vocab_size=expanded_3x.vocab_size,
            encoder_with_pooler_params=bert_encoder_with_pooler_params(expanded_3x),
            masked_lm_params=bert_masked_lm_params(expanded_3x),
            extra_vs_uncased_encoder=bert_encoder_with_pooler_params(expanded_3x) - uncased_encoder,
            extra_vs_uncased_mlm=bert_masked_lm_params(expanded_3x) - uncased_mlm,
            note="separate lowercase/first-cap/all-caps token embeddings",
        ),
        CountRow(
            name="hypothetical 4x uncased case-expanded vocab",
            vocab_size=expanded_4x.vocab_size,
            encoder_with_pooler_params=bert_encoder_with_pooler_params(expanded_4x),
            masked_lm_params=bert_masked_lm_params(expanded_4x),
            extra_vs_uncased_encoder=bert_encoder_with_pooler_params(expanded_4x) - uncased_encoder,
            extra_vs_uncased_mlm=bert_masked_lm_params(expanded_4x) - uncased_mlm,
            note="adds a separate mixed-case token variant as well",
        ),
    ]

    report = {
        "uncased_model": args.uncased_model,
        "cased_model": args.cased_model,
        "capitalization_vocab_size": cap_vocab,
        "shapes": {"uncased": asdict(uncased), "cased": asdict(cased)},
        "component_counts": {
            "capitalization_embedding_params": capitalization_embedding_params(uncased, cap_vocab),
            "capitalization_mlm_classifier_params": capitalization_classifier_params(uncased, cap_vocab),
            "hypothetical_3x_extra_word_embeddings": expanded_vocab_extra_params(uncased, multiplier=3),
            "hypothetical_4x_extra_word_embeddings": expanded_vocab_extra_params(uncased, multiplier=4),
        },
        "rows": [asdict(row) for row in rows],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(report))
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


def shape_from_config(config: BertConfig) -> BertShape:
    return BertShape(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        num_hidden_layers=int(config.num_hidden_layers),
        max_position_embeddings=int(config.max_position_embeddings),
        type_vocab_size=int(config.type_vocab_size),
    )


def bert_embeddings_params(shape: BertShape) -> int:
    hidden = shape.hidden_size
    return (
        shape.vocab_size * hidden
        + shape.max_position_embeddings * hidden
        + shape.type_vocab_size * hidden
        + 2 * hidden
    )


def bert_layer_params(shape: BertShape) -> int:
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    attention = 4 * hidden * hidden + 6 * hidden
    feed_forward = 2 * hidden * intermediate + intermediate + 3 * hidden
    return attention + feed_forward


def bert_encoder_params(shape: BertShape) -> int:
    return shape.num_hidden_layers * bert_layer_params(shape)


def bert_pooler_params(shape: BertShape) -> int:
    return shape.hidden_size * shape.hidden_size + shape.hidden_size


def bert_encoder_with_pooler_params(shape: BertShape) -> int:
    return bert_embeddings_params(shape) + bert_encoder_params(shape) + bert_pooler_params(shape)


def bert_masked_lm_head_params(shape: BertShape) -> int:
    hidden = shape.hidden_size
    transform = hidden * hidden + hidden + 2 * hidden
    decoder_bias = shape.vocab_size
    return transform + decoder_bias


def bert_masked_lm_params(shape: BertShape) -> int:
    # BertForMaskedLM uses BertModel(add_pooling_layer=False), with tied decoder
    # weights and a separate vocabulary-sized decoder bias.
    return bert_embeddings_params(shape) + bert_encoder_params(shape) + bert_masked_lm_head_params(shape)


def capitalization_embedding_params(shape: BertShape, capitalization_vocab_size: int) -> int:
    return capitalization_vocab_size * shape.hidden_size


def capitalization_classifier_params(shape: BertShape, capitalization_vocab_size: int) -> int:
    return shape.hidden_size * capitalization_vocab_size + capitalization_vocab_size


def expanded_vocab_extra_params(shape: BertShape, *, multiplier: int) -> int:
    return (multiplier - 1) * shape.vocab_size * shape.hidden_size


def to_markdown(report: dict[str, object]) -> str:
    component_counts = report["component_counts"]
    rows = report["rows"]

    lines = [
        "# Parameter Efficiency",
        "",
        "All counts are parameter counts from BERT configuration shapes, not loaded weights.",
        "The masked-LM count assumes tied decoder weights, matching standard BERT.",
        "",
        "## Model Counts",
        "",
        "| Model | Vocab | Encoder+pooler params | MLM params | Extra encoder vs uncased | Extra MLM vs uncased | Note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {vocab_size:,} | {encoder_with_pooler_params:,} | "
            "{masked_lm_params:,} | {extra_vs_uncased_encoder:+,} | "
            "{extra_vs_uncased_mlm:+,} | {note} |".format(**row)
        )

    cap_embedding = int(component_counts["capitalization_embedding_params"])
    cap_classifier = int(component_counts["capitalization_mlm_classifier_params"])
    extra_3x = int(component_counts["hypothetical_3x_extra_word_embeddings"])
    extra_4x = int(component_counts["hypothetical_4x_extra_word_embeddings"])

    lines.extend(
        [
            "",
            "## Key Ratios",
            "",
            f"- Capitalization embedding overhead with four case states: `{cap_embedding:,}` parameters.",
            f"- Auxiliary MLM capitalization head: `{cap_classifier:,}` parameters.",
            f"- Three-way case-expanded uncased vocabulary extra word embeddings: `{extra_3x:,}` parameters.",
            f"- Four-way case-expanded uncased vocabulary extra word embeddings: `{extra_4x:,}` parameters.",
            f"- 3x expansion is `{extra_3x / cap_embedding:,.0f}x` larger than the four-state capitalization embedding table.",
            f"- 4x expansion is `{extra_4x / cap_embedding:,.0f}x` larger than the four-state capitalization embedding table.",
            "",
            "## Paper-Framing Note",
            "",
            "The memory-efficiency claim should be made against an explicit case-expanded "
            "uncased vocabulary design, not against `bert-base-cased` directly. "
            "`bert-base-cased` has a smaller vocabulary than `bert-base-uncased`, so "
            "the capitalized model is slightly larger than the released cased baseline "
            "while still being dramatically smaller than a vocabulary-tripling design.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
