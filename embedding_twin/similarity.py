
import numpy as np
from typing import List, Tuple

class SimilarityEngine:
    def __init__(self, embeddings: np.ndarray, vocab: List[str], word2id: dict):
        if embeddings.shape[0] != len(vocab):
            raise ValueError("Embeddings rows must match vocab")
        self.emb = embeddings
        self.vocab = vocab
        self.word2id = word2id
        # normalize for fast cosine
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9
        self.norm_emb = self.emb / norms

    def cosine(self, w1: str, w2: str) -> float:
        if w1 not in self.word2id or w2 not in self.word2id:
            raise KeyError(f"OOV: {w1} or {w2} not in vocab")
        return float(np.dot(self.norm_emb[self.word2id[w1]], self.norm_emb[self.word2id[w2]]))

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str,float]]:
        if word not in self.word2id:
            raise KeyError(f"Word '{word}' OOV. Vocab: {self.vocab[:10]}...")
        q = self.norm_emb[self.word2id[word]]
        scores = self.norm_emb @ q
        # argsort descending
        idx = np.argsort(scores)[::-1]
        results = []
        for i in idx:
            if self.vocab[i] == word: continue
            results.append((self.vocab[i], float(scores[i])))
            if len(results) >= top_k: break
        return results

    def analogy(self, a: str, b: str, c: str, top_k=3):
        """a:b :: c:?  -> b - a + c"""
        for w in [a,b,c]:
            if w not in self.word2id:
                raise KeyError(f"OOV word {w}")
        vec = self.emb[self.word2id[b]] - self.emb[self.word2id[a]] + self.emb[self.word2id[c]]
        vec = vec / (np.linalg.norm(vec)+1e-9)
        scores = self.norm_emb @ vec
        idx = np.argsort(scores)[::-1]
        res = []
        for i in idx:
            if self.vocab[i] in [a,b,c]: continue
            res.append((self.vocab[i], float(scores[i])))
            if len(res) >= top_k: break
        return res

    def semantic_search(self, query: str, candidates: List[str], top_k=3):
        """For Digital Twin: search memory with embedding"""
        # simple avg if multi-word query not in vocab
        if query in self.word2id:
            q_vec = self.norm_emb[self.word2id[query]]
        else:
            # average known words
            tokens = query.lower().split()
            known = [self.emb[self.word2id[t]] for t in tokens if t in self.word2id]
            if not known:
                raise ValueError(f"Query '{query}' has no known words")
            q_vec = np.mean(known, axis=0)
            q_vec = q_vec / (np.linalg.norm(q_vec)+1e-9)
        scores = []
        for cand in candidates:
            # candidate embedding as avg
            ct = cand.lower().split()
            known = [self.emb[self.word2id[t]] for t in ct if t in self.word2id]
            if not known: continue
            c_vec = np.mean(known, axis=0)
            c_vec = c_vec / (np.linalg.norm(c_vec)+1e-9)
            scores.append((cand, float(np.dot(q_vec, c_vec))))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
