import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text, tokenize, build_vocab_and_process

class TestPreprocessing(unittest.TestCase):
    
    def test_clean_text(self):
        # HTML removal
        self.assertEqual(clean_text("<div>Hello World</div>"), "hello world")
        # Punctuation
        self.assertEqual(clean_text("Hello, World!"), "hello world")
        # Lowercase
        self.assertEqual(clean_text("HELLO"), "hello")
        # Mixed
        self.assertEqual(clean_text("<p>Hello, World!</p>"), "hello world")
        
    def test_tokenize(self):
        self.assertEqual(tokenize("hello world"), ["hello", "world"])
        self.assertEqual(tokenize(""), [])
        
    def test_build_vocab(self):
        # Small corpus
        plots = ["hello world", "hello python", "world python"]
        # Freqs: hello:2, world:2, python:2. If min_freq=2, all kept.
        vocab, tokens = build_vocab_and_process(plots, min_freq=2)
        
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertIn("python", vocab)
        self.assertIn("<UNK>", vocab)
        self.assertEqual(vocab["<PAD>"], 0)
        
        # Test UNK
        plots_rare = ["rare word appearing once"]
        vocab_rare, tokens_rare = build_vocab_and_process(plots_rare, min_freq=2)
        # "rare", "word", "appearing", "once" all freq=1 -> should be <UNK>
        self.assertNotIn("rare", vocab_rare)
        # Tokens should be all <UNK>
        self.assertEqual(tokens_rare, ["<UNK>", "<UNK>", "<UNK>", "<UNK>"])

if __name__ == '__main__':
    unittest.main()
