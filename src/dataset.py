import torch
from torch.utils.data import Dataset
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, corpus_path, seq_len=20, vocab=None, min_freq=5):
        self.seq_len = seq_len
        
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        self.tokens = raw_text.split()
        
    print(f"Total tokens loaded: {len(self.tokens)}")
        
    if vocab is None:
        self.word2idx, self.idx2word = self.build_vocab(self.tokens, min_freq)
    else:
        self.word2idx = vocab
        self.idx2word = {i: w for w, i in vocab.items()}
        
    print("Converting tokens to indices...")
    self.data_indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in self.tokens]
    
def build_vocab(self, tokens, min_freq):
    unique_tokens = sorted(list(set(tokens)))
    
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    current_idx = 2
    for t in unique_tokens:
        if t not in word2idx:
            word2idx[t] = current_idx
            current_idx += 1
            
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def __len__(self):
    return len(self.data_indices) - self.seq_len

def __getitem__(self, idx):
    in_seq = self.data_indices[idx : idx + self.seq_len]
    target = self.data_indices[idx + self.seq_len]
    
    return torch.tensor(in_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
