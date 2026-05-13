"""Capitalization-aware tokenization helpers.

The tokenizer still emits the base uncased BERT vocabulary IDs. This module
adds a parallel `capitalization_ids` feature computed from the original text
before the uncased tokenizer normalizer discards case information.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

from transformers import BatchEncoding, PreTrainedTokenizerBase


class Capitalization(IntEnum):
    """Small case feature vocabulary added to every token position."""

    NONE = 0
    FIRST_CAP = 1
    ALL_CAPS = 2
    MIXED_CASE = 3


NO_CAP = int(Capitalization.NONE)
FIRST_CAP = int(Capitalization.FIRST_CAP)
ALL_CAPS = int(Capitalization.ALL_CAPS)
MIXED_CASE = int(Capitalization.MIXED_CASE)


def classify_capitalization(text: str, *, use_mixed_case: bool = False) -> int:
    """Classify a source word/span into the capitalization feature vocabulary.

    Mixed-case words such as `iPhone`, `eBay`, and `McDonald` route to `NONE`
    for the original three-class experiment, or to `MIXED_CASE` for the
    four-class variant.
    """

    letters = [char for char in text if char.isalpha()]
    if not letters:
        return NO_CAP

    if len(letters) > 1 and all(char.isupper() for char in letters):
        return ALL_CAPS

    if letters[0].isupper() and all(char.islower() for char in letters[1:]):
        return FIRST_CAP

    if use_mixed_case and any(char.isupper() for char in letters):
        return MIXED_CASE

    return NO_CAP


def _expand_to_nonspace_span(text: str, start: int, end: int) -> str:
    """Return the whitespace-delimited source span enclosing `[start, end)`."""

    left = start
    while left > 0 and not text[left - 1].isspace():
        left -= 1

    right = end
    while right < len(text) and not text[right].isspace():
        right += 1

    return text[left:right]


def capitalization_ids_from_offsets(
    text: str,
    offsets: Sequence[tuple[int, int]] | Sequence[list[int]],
    special_tokens_mask: Sequence[int] | None = None,
    *,
    use_mixed_case: bool = False,
) -> list[int]:
    """Build per-token capitalization IDs from fast-tokenizer offsets."""

    capitalization_ids: list[int] = []
    for index, offset in enumerate(offsets):
        start, end = int(offset[0]), int(offset[1])
        is_special = special_tokens_mask is not None and bool(special_tokens_mask[index])
        if is_special or end <= start:
            capitalization_ids.append(NO_CAP)
            continue

        if not any(char.isalpha() for char in text[start:end]):
            capitalization_ids.append(NO_CAP)
            continue

        source_span = _expand_to_nonspace_span(text, start, end)
        capitalization_ids.append(
            classify_capitalization(source_span, use_mixed_case=use_mixed_case)
        )

    return capitalization_ids


def capitalization_ids_from_words(
    words: Sequence[str],
    word_ids: Sequence[int | None],
    *,
    use_mixed_case: bool = False,
) -> list[int]:
    """Build per-token capitalization IDs from tokenizer `word_ids` output."""

    word_capitalization = [
        classify_capitalization(word, use_mixed_case=use_mixed_case)
        for word in words
    ]
    return [
        NO_CAP if word_id is None else word_capitalization[int(word_id)]
        for word_id in word_ids
    ]


def _is_batched_raw_text(text: Any) -> bool:
    return isinstance(text, list) and (not text or isinstance(text[0], str))


def _is_batched_split_words(text: Any) -> bool:
    return isinstance(text, list) and bool(text) and isinstance(text[0], list)


def tokenize_with_capitalization(
    tokenizer: PreTrainedTokenizerBase,
    text: str | list[str] | list[list[str]],
    *,
    is_split_into_words: bool = False,
    keep_offsets_mapping: bool = False,
    use_mixed_case: bool = False,
    return_tensors: str | None = None,
    **tokenizer_kwargs: Any,
) -> BatchEncoding:
    """Tokenize text and add a parallel `capitalization_ids` feature.

    Args:
        tokenizer: A Hugging Face fast tokenizer. Use
            `AutoTokenizer.from_pretrained(..., use_fast=True)`.
        text: A string, batch of strings, tokenized word list, or batch of
            tokenized word lists.
        is_split_into_words: Pass through to the tokenizer for token
            classification datasets such as CoNLL-2003.
        keep_offsets_mapping: Keep tokenizer offsets in the returned encoding.
        use_mixed_case: Emit a fourth `MIXED_CASE` capitalization ID for spans
            such as `iPhone` and `McDonald`.
        return_tensors: Optional Hugging Face tensor type such as `"pt"`.
        **tokenizer_kwargs: Forwarded to the tokenizer.

    Returns:
        A `BatchEncoding` containing normal tokenizer fields plus
        `capitalization_ids`.
    """

    if not tokenizer.is_fast:
        raise ValueError("Capitalization IDs require a Hugging Face fast tokenizer.")

    tokenizer_kwargs = dict(tokenizer_kwargs)
    tokenizer_kwargs["return_offsets_mapping"] = True
    tokenizer_kwargs["return_special_tokens_mask"] = True

    encoding = tokenizer(
        text,
        is_split_into_words=is_split_into_words,
        return_tensors=None,
        **tokenizer_kwargs,
    )

    offset_mapping = encoding["offset_mapping"]
    special_tokens_mask = encoding.get("special_tokens_mask")

    if is_split_into_words:
        is_batched = _is_batched_split_words(text)
        word_batches = text if is_batched else [text]
        cap_batches = [
            capitalization_ids_from_words(
                words,
                encoding.word_ids(batch_index=batch_index),
                use_mixed_case=use_mixed_case,
            )
            for batch_index, words in enumerate(word_batches)
        ]
    else:
        is_batched = _is_batched_raw_text(text)
        text_batches = text if is_batched else [text]
        offsets_batches = offset_mapping if is_batched else [offset_mapping]
        special_batches = special_tokens_mask if is_batched else [special_tokens_mask]
        cap_batches = [
            capitalization_ids_from_offsets(source_text, offsets, special_mask)
            if not use_mixed_case
            else capitalization_ids_from_offsets(
                source_text,
                offsets,
                special_mask,
                use_mixed_case=True,
            )
            for source_text, offsets, special_mask in zip(
                text_batches,
                offsets_batches,
                special_batches,
                strict=True,
            )
        ]

    encoding["capitalization_ids"] = cap_batches if is_batched else cap_batches[0]

    if not keep_offsets_mapping:
        encoding.pop("offset_mapping", None)

    if return_tensors is not None:
        encoding = encoding.convert_to_tensors(return_tensors)

    return encoding
