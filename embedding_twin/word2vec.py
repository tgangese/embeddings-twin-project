
import numpy as np
from typing import List, Tuple

class SkipGram:
    """
    Word2Vec Skip-gram with Negative Sampling from scratch.
    Paper: Mikolov et al. 2013
    """
    def __init__(self, vocab_size: int, embed_dim: int = 50, lr: float = 0.05, neg_samples: int = 5):
        if vocab_size < 2: raise ValueError("vocab_size >=2")
        if embed_dim < 1: raise ValueError("embed_dim >=1")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        self.neg_samples = neg_samples
        # Xavier init
        self.W_in = (np.random.randn(vocab_size, embed_dim) * 0.1).astype(np.float64)
        self.W_out = (np.random.randn(vocab_size, embed_dim) * 0.1).astype(np.float64)
        self.loss_history = []

    def _sigmoid(self, x): 
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def train(self, pairs: List[Tuple[int,int]], epochs: int = 100):
        if not pairs: raise ValueError("pairs empty - need (center, context) ids")
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(pairs)
            for center_id, pos_id in pairs:
                # positive
                total_loss += self._update(center_id, pos_id, 1)
                # negatives
                for _ in range(self.neg_samples):
                    neg_id = np.random.randint(0, self.vocab_size)
                    while neg_id == pos_id:
                        neg_id = np.random.randint(0, self.vocab_size)
                    total_loss += self._update(center_id, neg_id, 0)
            self.loss_history.append(total_loss)
        return self.W_in

    def _update(self, center, context, label):
        v_c = self.W_in[center]
        u_o = self.W_out[context]
        prob = self._sigmoid(np.dot(v_c, u_o))
        grad = prob - label
        # SGD
        self.W_in[center] -= self.lr * grad * u_o
        self.W_out[context] -= self.lr * grad * v_c
        loss = - (label * np.log(prob+1e-9) + (1-label)*np.log(1-prob+1e-9))
        return loss

class CBOW:
    """CBOW: predict center from context - faster for frequent words"""
    def __init__(self, vocab_size: int, embed_dim: int = 50, lr: float = 0.05):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.1
        self.loss_history = []

    def train(self, data: List[Tuple[List[int], int]], epochs=50):
        """data: [([context_ids], center_id)]"""
        for epoch in range(epochs):
            loss = 0
            for ctx_ids, center_id in data:
                if not ctx_ids: continue
                v_ctx = np.mean([self.W_in[i] for i in ctx_ids], axis=0)
                score = np.dot(v_ctx, self.W_out[center_id])
                prob = 1/(1+np.exp(-score))
                grad = prob - 1
                # update
                for cid in ctx_ids:
                    self.W_in[cid] -= self.lr * grad * self.W_out[center_id] / len(ctx_ids)
                self.W_out[center_id] -= self.lr * grad * v_ctx
                loss -= np.log(prob+1e-9)
            self.loss_history.append(loss)
        return self.W_in
