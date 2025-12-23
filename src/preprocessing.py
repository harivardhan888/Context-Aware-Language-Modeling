import pandas as pd
import re
import os
from collections import Counter
import html
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        return pd.DataFrame()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()

def build_vocab_and_process(plots, min_freq=5):
    logger.info("Building vocabulary...")
    counter = Counter()
    for plot in plots:
        cleaned = clean_text(plot)
        tokens = tokenize(cleaned)
        counter.update(tokens)
        
    logger.info(f"Total unique words before cutoff: {len(counter)}")
    
    vocab = {'<PAD>': 0, '<UNK>': 1} 
    valid_words = {w for w, c in counter.items() if c >= min_freq}
    
    for i, w in enumerate(sorted(list(valid_words)), start=2):
        vocab[w] = i
        
    logger.info(f"Vocab size after cutoff ({min_freq}): {len(vocab)}")
    
    logger.info("Processing text...")
    processed_tokens = []
    
    for plot in plots:
        cleaned = clean_text(plot)
        tokens = tokenize(cleaned)
        for t in tokens:
            if t in valid_words:
                processed_tokens.append(t)
            else:
                processed_tokens.append('<UNK>')
                
    return vocab, processed_tokens

if __name__ == "__main__":
    import argparse
    
    # Project paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "wiki_movie_plots_deduped.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    PROCESSED_PATH = os.path.join(PROCESSED_DIR, "corpus.txt")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    df = load_data(RAW_PATH)
    
    if not df.empty and 'Plot' in df.columns:
        logger.info(f"Loaded {len(df)} rows.")
        vocab, tokens = build_vocab_and_process(df['Plot'].dropna(), min_freq=5)
        
        logger.info(f"Saving corpus to {PROCESSED_PATH}...")
        with open(PROCESSED_PATH, 'w', encoding='utf-8') as f:
            f.write(' '.join(tokens))
            
        logger.info("Done.")
    else:
        logger.warning("Dataset empty or 'Plot' column missing.")
