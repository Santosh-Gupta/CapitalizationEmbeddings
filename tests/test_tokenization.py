import unittest

from capitalization_embeddings.tokenization import (
    ALL_CAPS,
    FIRST_CAP,
    MIXED_CASE,
    NO_CAP,
    capitalization_ids_from_offsets,
    capitalization_ids_from_words,
    classify_capitalization,
)


class CapitalizationTests(unittest.TestCase):
    def test_classify_capitalization(self):
        self.assertEqual(classify_capitalization("tom"), NO_CAP)
        self.assertEqual(classify_capitalization("Tom"), FIRST_CAP)
        self.assertEqual(classify_capitalization("TOM"), ALL_CAPS)
        self.assertEqual(classify_capitalization("iPhone"), NO_CAP)
        self.assertEqual(
            classify_capitalization("iPhone", use_mixed_case=True),
            MIXED_CASE,
        )
        self.assertEqual(
            classify_capitalization("McDonald", use_mixed_case=True),
            MIXED_CASE,
        )
        self.assertEqual(classify_capitalization("USA"), ALL_CAPS)
        self.assertEqual(classify_capitalization("[CLS]"), ALL_CAPS)
        self.assertEqual(classify_capitalization("123"), NO_CAP)

    def test_offsets_expand_to_whitespace_span(self):
        text = "Tom met TOM and iPhone users."
        offsets = [(0, 0), (0, 3), (8, 11), (16, 22), (28, 29), (0, 0)]
        special_tokens_mask = [1, 0, 0, 0, 0, 1]

        self.assertEqual(
            capitalization_ids_from_offsets(text, offsets, special_tokens_mask),
            [NO_CAP, FIRST_CAP, ALL_CAPS, NO_CAP, NO_CAP, NO_CAP],
        )
        self.assertEqual(
            capitalization_ids_from_offsets(
                text,
                offsets,
                special_tokens_mask,
                use_mixed_case=True,
            ),
            [NO_CAP, FIRST_CAP, ALL_CAPS, MIXED_CASE, NO_CAP, NO_CAP],
        )

    def test_word_ids(self):
        words = ["Tom", "met", "NASA"]
        word_ids = [None, 0, 1, 2, 2, None]

        self.assertEqual(
            capitalization_ids_from_words(words, word_ids),
            [NO_CAP, FIRST_CAP, NO_CAP, ALL_CAPS, ALL_CAPS, NO_CAP],
        )

    def test_word_ids_can_emit_mixed_case(self):
        words = ["iPhone", "met", "eBay"]
        word_ids = [None, 0, 1, 2, None]

        self.assertEqual(
            capitalization_ids_from_words(words, word_ids, use_mixed_case=True),
            [NO_CAP, MIXED_CASE, NO_CAP, MIXED_CASE, NO_CAP],
        )


if __name__ == "__main__":
    unittest.main()
