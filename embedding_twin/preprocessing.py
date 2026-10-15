
import re
import numpy as np
from typing import List, Dict, Tuple

class TextPreprocessor:
    """Clean, tokenize, build vocab - shared by Word2Vec & GloVe"""
    def __init__(self, min_freq: int = 1):
        if min_freq < 1:
            raise ValueError("min_freq must be >=1")
        self.min_freq = min_freq
        self.vocab = []
        self.word2id = {}
        self.id2word = {}

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        if not tokens:
            raise ValueError("No tokens found after cleaning")
        return tokens

    def build_vocab(self, tokens: List[str]) -> List[str]:
        if not tokens:
            raise ValueError("tokens empty")
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        self.vocab = sorted([w for w,c in freq.items() if c >= self.min_freq])
        if len(self.vocab) < 2:
            raise ValueError(f"Need >=2 unique tokens, got {self.vocab}")
        self.word2id = {w:i for i,w in enumerate(self.vocab)}
        self.id2word = {i:w for w,i in self.word2id.items()}
        return self.vocab

class CooccurrenceBuilder:
    """Builds X_ij matrix for GloVe"""
    def __init__(self, window_size: int = 2):
        if window_size < 1:
            raise ValueError("window_size must be >=1")
        self.window_size = window_size

    def build(self, tokens: List[str], word2id: Dict[str,int]) -> np.ndarray:
        V = len(word2id)
        X = np.zeros((V,V), dtype=float)
        for idx, word in enumerate(tokens):
            if word not in word2id: continue
            w_id = word2id[word]
            l = max(0, idx - self.window_size)
            r = min(len(tokens), idx + self.window_size + 1)
            for j in range(l, r):
                if j == idx: continue
                ctx = tokens[j]
                if ctx not in word2id: continue
                distance = abs(j-idx)
                X[w_id, word2id[ctx]] += 1.0/distance
        return X
