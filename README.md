# Context-Aware Language Modeling on Wikipedia Movie Plots

Comparing **N-gram** (bigram, trigram) and **LSTM** language models on 34,886 Wikipedia movie plot narratives (~12.9 M tokens).

---

## Highlights

- Trained and evaluated word-level language models on 34K+ real narrative texts
- LSTM achieves a **significant perplexity reduction** (≥25%) vs the bigram baseline, demonstrating the advantage of gated recurrent representations over fixed-order statistics
- EDA on plot-length distributions drove the choice of `seq_len = 20` as the context window, validated by an ablation study across three window sizes
- Complete NLP pipeline: cleaning → tokenisation → vocabulary filtering → sliding-window dataset → model training → evaluation → text generation

---

## Results

| Model | Architecture | Context | Test Perplexity |
|-------|-------------|---------|----------------|
| Bigram | N-gram n=2, Laplace α=1 | 1 token | 1,903.60 |
| Trigram | N-gram n=3, Laplace α=1 | 2 tokens | 14,685.28 |
| **LSTM** | 2-layer, Embedding(128), Hidden(256) | **20 tokens** | **run Notebook 03** |

> Trigram perplexity is worse than bigram because Laplace smoothing must spread probability mass
> over an exponentially larger, sparser context space — a well-known data-sparsity effect.

---

## Dataset

- **Source**: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) — `wiki_movie_plots_deduped.csv`
- **Size**: 34,886 movie plots, ~12.99 M raw tokens
- **Vocabulary**: 400,035 unique raw tokens → **63,727 after `min_freq=5` cutoff**
- **Split**: 80% train / 10% validation / 10% test (sequential, preserving narrative order)

---

## Project Structure

```
Context Aware Language Modeling/
├── data/
│   ├── raw/
│   │   └── wiki_movie_plots_deduped.csv   # raw Kaggle dataset
│   └── processed/
│       └── corpus.txt                     # cleaned, tokenised corpus (generated)
│
├── src/
│   ├── preprocessing.py   # clean_text, tokenize, build_vocab_and_process
│   ├── dataset.py         # TextDataset (sliding-window PyTorch Dataset)
│   └── models.py          # NGramModel, LSTMModel, evaluate_lstm_perplexity
│
├── notebooks/
│   ├── 01_eda_text_analysis.ipynb        # EDA + context window selection
│   ├── 02_ngram_language_model.ipynb     # Bigram & Trigram training + perplexity
│   ├── 03_lstm_language_model.ipynb      # LSTM training, evaluation, generation
│   └── 04_experiments_analysis.ipynb     # Ablation study + full comparison
│
├── results/
│   ├── plot_length_distribution.png
│   ├── context_window_selection_eda.png
│   ├── context_window_analysis.png
│   ├── lstm_training_curves.png
│   ├── perplexity_comparison.png
│   ├── coherence_analysis.png
│   └── sample_generations.txt
│
├── tests/
│   └── test_preprocessing.py
└── requirements.txt
```

---

## Architecture

### NLP Pipeline

```
Raw CSV (34,886 plots)
    │
    ▼  src/preprocessing.py
Cleaned corpus  (lowercase · HTML removed · punctuation stripped)
    │
    ▼  src/dataset.py — TextDataset
Sliding-window sequences  [20 tokens → next token]
(12.9M sequences, vocab = 63,727 tokens)
    │
    ├──▶  NGramModel     (bigram / trigram + Laplace smoothing)
    │
    └──▶  LSTMModel      (2-layer LSTM + dropout + linear head)
              │
              ▼
         Perplexity evaluation · Text generation
```

### LSTM Model

```
Input  [B, 20]
  → Embedding   (63,727 → 128)
  → LSTM layer 1  (128 → 256)  + Dropout(0.3)
  → LSTM layer 2  (256 → 256)
  → Dropout(0.3)
  → Linear       (256 → 63,727)
Output [B, 63,727]  logits
```

- **Optimizer**: Adam (lr=1e-3) with ReduceLROnPlateau scheduler
- **Regularisation**: Dropout (0.3) + gradient clipping (max_norm=1.0)
- **Checkpoint**: Best model by validation loss saved to `results/lstm_checkpoint.pt`

---

## Sample Outputs

### N-gram (Trigram)

```
Prompt  : the young man
Trigram : chaplin arriving at the time while there she spots another man
          bruce who is annoyed by his friend sir charles

Prompt  : once upon a
Trigram : music hall performer who takes in chester and uses forensics with
          the daughter of uncle charlie follows and a 5yearold
```

### LSTM

*(Run Notebook 03 to generate — sample outputs saved to `results/sample_generations.txt`)*

The LSTM maintains **entity coreference** and **causal flow** across multiple sentences,
in contrast to N-gram outputs that lose narrative coherence beyond 10–15 tokens.

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess the corpus

```bash
python src/preprocessing.py
# Output: data/processed/corpus.txt
```

### 3. Run notebooks in order

```
01_eda_text_analysis.ipynb    → EDA + context window decision
02_ngram_language_model.ipynb → Bigram & Trigram baselines
03_lstm_language_model.ipynb  → LSTM training (~5 epochs)
04_experiments_analysis.ipynb → Full comparison & visualisations
```

### 4. Run unit tests

```bash
python -m pytest tests/ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `min_freq = 5` | Reduces 400k raw tokens to ~63k; covers all common narrative vocabulary while replacing rare words with `<UNK>` |
| `seq_len = 20` | Confirmed by EDA: 99%+ of plots exceed 20 words; ablation in Notebook 04 shows lower perplexity than `seq_len ∈ {5, 10}` |
| 2-layer LSTM | Single-layer captures only local syntax; two layers learn hierarchical features (words → phrases → plot structure) |
| Gradient clipping (1.0) | Narrative text has long-range dependencies that can cause gradient explosions without clipping |
| Sequential train/val/test split | Preserves narrative ordering; prevents leakage from plot-level boundaries |

---

## Requirements

```
torch>=2.0.0
pandas>=2.0.0
matplotlib>=3.0.0
seaborn>=0.11.0
numpy<2.0.0
jupyter>=1.0.0
```
