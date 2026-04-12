"""
models.py — N-gram and LSTM language model implementations.

Both models expose a ``calculate_perplexity`` interface so that
Notebook 04 can compare them with the same evaluation code.
A standalone ``evaluate_lstm_perplexity`` helper is also provided
for batch-level LSTM evaluation on a DataLoader.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. LSTMModel will not be available.")

from collections import defaultdict, Counter
import math
import random


# ═══════════════════════════════════════════════════════════════════════
# N-gram Language Model
# ═══════════════════════════════════════════════════════════════════════

class NGramModel:
    """
    Statistical N-gram language model with Laplace (add-alpha) smoothing.

    Args:
        n     (int):   Order of the n-gram (2 = bigram, 3 = trigram).
        alpha (float): Smoothing coefficient (1.0 = full Laplace smoothing).
    """

    def __init__(self, n: int = 3, alpha: float = 1.0):
        self.n = n
        self.alpha = alpha
        self.counts = defaultdict(Counter)      # context → Counter of next words
        self.context_counts = Counter()         # context → total occurrences
        self.vocab = set()

    # ------------------------------------------------------------------
    def train(self, tokens: list) -> None:
        """Count n-gram occurrences in *tokens*."""
        print(f"Training {self.n}-gram model on {len(tokens):,} tokens ...")
        self.vocab = set(tokens)
        for i in range(len(tokens) - self.n + 1):
            ngram   = tuple(tokens[i: i + self.n])
            context = ngram[:-1]
            target  = ngram[-1]
            self.counts[context][target] += 1
            self.context_counts[context] += 1
        print(f"  Unique {self.n - 1}-gram contexts: {len(self.context_counts):,}")

    # ------------------------------------------------------------------
    def get_log_prob(self, context: tuple, target: str) -> float:
        """Return log P(target | context) with Laplace smoothing."""
        context = tuple(context[-(self.n - 1):])
        V = len(self.vocab)
        count_wc = self.counts[context][target]
        count_c  = self.context_counts[context]
        prob = (count_wc + self.alpha) / (count_c + self.alpha * V)
        return math.log(prob)

    # ------------------------------------------------------------------
    def calculate_perplexity(self, tokens: list) -> float:
        """Compute perplexity on a token sequence."""
        log_sum, N = 0.0, 0
        for i in range(self.n - 1, len(tokens)):
            ctx     = tuple(tokens[i - (self.n - 1): i])
            log_sum += self.get_log_prob(ctx, tokens[i])
            N       += 1
        return math.exp(-log_sum / N)

    # ------------------------------------------------------------------
    def generate(self, seed_tokens: list, num_words: int = 30,
                 temperature: float = 1.0) -> str:
        """
        Sample *num_words* tokens following *seed_tokens*.

        Args:
            seed_tokens: Starting context (list of word strings).
            num_words:   Number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Generated continuation as a whitespace-joined string.
        """
        tokens     = list(seed_tokens)
        vocab_list = list(self.vocab)

        for _ in range(num_words):
            ctx = tuple(tokens[-(self.n - 1):])
            if ctx in self.counts and self.counts[ctx]:
                candidates = list(self.counts[ctx].keys())
                weights    = [self.counts[ctx][w] ** (1.0 / max(temperature, 1e-8))
                              for w in candidates]
                total      = sum(weights)
                probs      = [w / total for w in weights]
                next_word  = random.choices(candidates, weights=probs, k=1)[0]
            else:
                next_word = random.choice(vocab_list)
            tokens.append(next_word)

        return " ".join(tokens[len(seed_tokens):])


# ═══════════════════════════════════════════════════════════════════════
# LSTM Language Model
# ═══════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        """
        2-layer LSTM language model.

        Architecture:
            Embedding(vocab_size, embed_dim)
            → LSTM(embed_dim, hidden_dim, num_layers, dropout)
            → Dropout
            → Linear(hidden_dim, vocab_size)

        The model takes a batch of token-index sequences [B, seq_len]
        and returns logits [B, vocab_size] based on the *last* LSTM
        output step (next-word prediction).

        Args:
            vocab_size (int): Total number of tokens in the vocabulary.
            embed_dim  (int): Word-embedding dimension (default 128).
            hidden_dim (int): LSTM hidden-state size (default 256).
            num_layers (int): Number of stacked LSTM layers (default 2).
            dropout  (float): Dropout rate between LSTM layers (default 0.3).
        """

        def __init__(self, vocab_size: int, embed_dim: int = 128,
                     hidden_dim: int = 256, num_layers: int = 2,
                     dropout: float = 0.3):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        # --------------------------------------------------------------
        def forward(self, x, hidden=None):
            """
            Args:
                x      : LongTensor [batch, seq_len]
                hidden : Optional (h, c) LSTM state tuple.
            Returns:
                logits [batch, vocab_size], new hidden state.
            """
            emb            = self.embedding(x)          # [B, S, embed_dim]
            out, hidden    = self.lstm(emb, hidden)     # [B, S, hidden_dim]
            last           = self.dropout(out[:, -1, :])# [B, hidden_dim]
            logits         = self.fc(last)              # [B, vocab_size]
            return logits, hidden

        # --------------------------------------------------------------
        @torch.no_grad()
        def generate(self, seed_indices: list, num_words: int,
                     idx2word: dict, word2idx: dict,
                     temperature: float = 0.8,
                     seq_len: int = 20,
                     device=None) -> str:
            """
            Auto-regressively generate *num_words* tokens.

            Args:
                seed_indices: List of starting word indices (int).
                num_words:    Number of tokens to generate.
                idx2word:     Index → word mapping.
                word2idx:     Word → index mapping (used for <PAD>).
                temperature:  Softmax temperature (lower = more greedy).
                seq_len:      Context window size used during training.
                device:       Torch device (inferred from model if None).

            Returns:
                Generated continuation as a whitespace-joined string.
            """
            if device is None:
                device = next(self.parameters()).device
            self.eval()
            pad_id    = word2idx.get("<PAD>", 0)
            generated = list(seed_indices)

            for _ in range(num_words):
                context = generated[-seq_len:]
                if len(context) < seq_len:
                    context = [pad_id] * (seq_len - len(context)) + context
                inp       = torch.tensor([context], dtype=torch.long, device=device)
                logits, _ = self(inp)
                probs     = F.softmax(logits[0] / max(temperature, 1e-8), dim=-1)
                next_idx  = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_idx)

            return " ".join(idx2word.get(i, "<UNK>") for i in generated[len(seed_indices):])

    # ──────────────────────────────────────────────────────────────────
    # Standalone evaluation utility
    # ──────────────────────────────────────────────────────────────────

    def evaluate_lstm_perplexity(model, dataloader, criterion, device) -> float:
        """
        Compute word-level perplexity of *model* on *dataloader*.

        perplexity = exp(mean cross-entropy loss)

        Because nn.CrossEntropyLoss(reduction='mean') equals
        -1/N * sum(log P(correct_word)), exp(avg_loss) gives perplexity
        directly.

        Args:
            model      : Trained LSTMModel.
            dataloader : DataLoader yielding (inputs, targets) batches.
            criterion  : nn.CrossEntropyLoss instance.
            device     : Torch device.

        Returns:
            Scalar perplexity value (float).
        """
        model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _       = model(inputs)
                loss            = criterion(logits, targets)
                total_loss     += loss.item()
                n_batches      += 1
        avg_loss = total_loss / n_batches
        return math.exp(avg_loss)

else:
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Run: pip install torch")

    def evaluate_lstm_perplexity(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Run: pip install torch")
