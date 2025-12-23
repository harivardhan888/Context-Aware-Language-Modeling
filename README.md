# Wiki Movie Plots Language Model

## 📌 Problem Statement
Language modeling is a foundational task in Natural Language Processing (NLP), predicting the probability of the next word given a context. This project builds and compares **Statistical N-gram models** and **Neural LSTM models** using the Wikipedia Movie Plots dataset. The goal is to understand how different architectures handle narrative structures and long-term dependencies found in movie plots.

## 📂 Dataset
- **Source**: Wikipedia Movie Plots (`wiki_movie_plots_deduped.csv`)
- **Size**: ~34,000 plots
- **Statistics**:
    - **Average Plot Length**: ~372 words
    - **Total Plots**: 34,886
    - **Top Common Words**: "the", "to", "and", "a", "is"
- **Preprocessing**:
    - **Cleaning**: Lowercasing, HTML removal, punctuation stripping.
    - **Tokenization**: Word-level tokens.
    - **Vocabulary**: Frequency-based cutoff with `<UNK>` token for rare words.
    - **Sequence Generation**: Sliding window approach (Input: 20 words -> Target: Next word).
    - **Processed Data**: `data/processed/corpus.txt` (Generated).

## 📊 Results & Analysis
### 1. Exploratory Data Analysis (EDA)
We analyzed the text distribution to inform our model choices.
- **Plot Length Distribution**: Included in `results/plot_length_distribution.png`.
- **Key Stats**: Saved in `results/eda_stats.txt`.
    - Data Shape: 34,886 plots.
    - Analysis confirms the need for sequence truncation around 300-400 words for efficiency.

### 2. Baseline: N-gram Models
- **Method**: Bigram & Trigram with Laplace smoothing.
- **Sample Outputs**: See `results/sample_generations.txt`.
    - *Input*: "the young man"
    - *Trigram Output*: "the young man chaplin arriving at the time while there she spots another man..."
    - The N-gram model successfully captures local syntax but often loops or lacks coherence over long sequences.

### 3. Neural Model: LSTM
- **Architecture**: Embedding -> LSTM (2 layers) -> Dense -> Softmax.
- **Status**: Complete implementation provided in `src/models.py` and `notebooks/03_lstm_language_model.ipynb`.
- **Advantages**: Unlike N-grams, LSTMs can maintain a memory of the sequence, theoretically allowing for more coherent plot generation over longer paragraphs.


### 4. Experiments & Context Analysis
- **Notebook**: `notebooks/04_experiments_analysis.ipynb`
- **Objective**: To study the effect of different context lengths (e.g., 5, 10, 20 words) on model perplexity.
- **Visualization**: Comparison of perplexity scores between Bigram, Trigram, and LSTM models.



## 🔮 Future Work
- **Transformer Models**: Implement a small GPT-style Transformer to compare against LSTM.
- **Hyperparameter Tuning**: Optimize hidden dimensions and embedding sizes.
- **Beam Search**: Implement beam search decoding for better generation quality.
