import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.preprocessing import clean_text, simple_tokenize, build_vocab, encode, decode, turkish_lower

class TestPreprocessing(unittest.TestCase):

    def test_turkish_lower(self):
        self.assertEqual(turkish_lower("Istanbul"), "ıstanbul")
        self.assertEqual(turkish_lower("İzmir"), "izmir")
        self.assertEqual(turkish_lower("HELLO"), "hello")

    def test_clean_text(self):
        text = "Bu [[bir|ornek]] <ref>test</ref> metindir."
        cleaned = clean_text(text)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("<ref>", cleaned)

    def test_simple_tokenize(self):
        text = "bir iki uc"
        tokens = simple_tokenize(text)
        self.assertEqual(tokens, ["bir", "iki", "uc"])

    def test_build_vocab(self):
        texts = ["bir iki uc", "bir iki dort"]
        vocab = build_vocab(texts, min_freq=1)
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)
        self.assertIn("bir", vocab)

    def test_encode_decode(self):
        texts = ["bir iki uc dort"]
        vocab = build_vocab(texts, min_freq=1)
        inv_vocab = {v: k for k, v in vocab.items()}
        ids = encode("bir iki", vocab)
        self.assertIsInstance(ids, list)
        self.assertTrue(len(ids) > 0)
        decoded = decode(ids, inv_vocab)
        self.assertIsInstance(decoded, str)

if __name__ == "__main__":
    unittest.main()
