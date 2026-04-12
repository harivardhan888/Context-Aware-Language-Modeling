"""
dataset.py — PyTorch Dataset for sliding-window language modeling.

Each sample is a (context_tokens, next_token) pair built from a
preprocessed corpus file.  Vocabulary is built from token frequencies
with an optional minimum-count threshold.
"""

import torch
from torch.utils.data import Dataset
from collections import Counter


class TextDataset(Dataset):
    """
    Sliding-window next-word prediction dataset.

    Given a flat token sequence, produces overlapping windows of
    ``seq_len`` consecutive tokens as inputs and the immediately
    following token as the prediction target.

    Args:
        corpus_path (str): Path to a whitespace-separated token file.
        seq_len     (int): Context window size (default 20).
        vocab       (dict | None): Pre-built word→index mapping.
                         If None, a new vocab is built from the corpus.
        min_freq    (int): Minimum token frequency for vocabulary inclusion.
    """

    def __init__(self, corpus_path: str, seq_len: int = 20,
                 vocab: dict = None, min_freq: int = 5):
        self.seq_len = seq_len

        print(f"[TextDataset] Loading corpus: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
        self.tokens = raw_text.split()
        print(f"  tokens      : {len(self.tokens):>12,}")

        if vocab is None:
            self.word2idx, self.idx2word = self.build_vocab(self.tokens, min_freq)
        else:
            self.word2idx = vocab
            self.idx2word = {i: w for w, i in vocab.items()}

        unk_id = self.word2idx["<UNK>"]
        self.data_indices = [self.word2idx.get(t, unk_id) for t in self.tokens]
        print(f"  vocab size  : {len(self.word2idx):>12,}")
        print(f"  sequences   : {len(self):>12,}  (window = {seq_len})\n")

    # ------------------------------------------------------------------
    def build_vocab(self, tokens: list, min_freq: int) -> tuple:
        """Build word→index and index→word mappings with frequency cutoff."""
        print("  Building vocabulary ...")
        counts = Counter(tokens)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word, cnt in sorted(counts.items()):
            if cnt >= min_freq and word not in word2idx:
                word2idx[word] = len(word2idx)
        idx2word = {i: w for w, i in word2idx.items()}
        return word2idx, idx2word

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_indices) - self.seq_len

    def __getitem__(self, idx: int) -> tuple:
        x = self.data_indices[idx: idx + self.seq_len]
        y = self.data_indices[idx + self.seq_len]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
