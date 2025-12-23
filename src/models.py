
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. LSTMModel will not be available.")

from collections import defaultdict, Counter
import math
import random

class NGramModel:
    def __init__(self, n=3, alpha=1.0):
        self.n = n
        self.alpha = alpha  # Laplace smoothing parameter
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, tokens):
        print(f"Training {self.n}-gram model...")
        self.vocab = set(tokens)
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            context = ngram[:-1]
            target = ngram[-1]
            
            self.counts[context][target] += 1
            self.context_counts[context] += 1

    def get_log_prob(self, context, target):
        context = tuple(context[-(self.n-1):]) if len(context) >= self.n-1 else tuple(context)
        
        vocab_size = len(self.vocab)
        
        count_w_context = self.counts[context][target]
        count_context = self.context_counts[context]
        
        prob = (count_w_context + self.alpha) / (count_context + self.alpha * vocab_size)
        return math.log(prob)

    def calculate_perplexity(self, tokens):
        log_prob_sum = 0
        N = 0
        
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - (self.n - 1) : i])
            target = tokens[i]
            log_prob_sum += self.get_log_prob(context, target)
            N += 1
            
        perplexity = math.exp(-log_prob_sum / N)
        return perplexity

if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                                dropout=dropout, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x, hidden=None):
            embed = self.embedding(x)
            out, hidden = self.lstm(embed, hidden)
            last_out = out[:, -1, :] 
            last_out = self.dropout(last_out)
            logits = self.fc(last_out)
            return logits, hidden
else:
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Please install torch to use LSTMModel.")
