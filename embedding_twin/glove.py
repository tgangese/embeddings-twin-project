
import numpy as np

class GloVe:
    """
    GloVe from scratch: w_i^T w_j + b_i + b_j = log(X_ij)
    Uses weighting function f(X) and AdaGrad.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 50, x_max: int = 100, alpha: float = 0.75, lr: float = 0.05):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.x_max = x_max
        self.alpha = alpha
        self.lr = lr
        self.W = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W_tilde = np.random.randn(vocab_size, embed_dim) * 0.1
        self.b = np.zeros(vocab_size)
        self.b_tilde = np.zeros(vocab_size)
        self.grad_sq_W = np.ones((vocab_size, embed_dim))
        self.grad_sq_Wt = np.ones((vocab_size, embed_dim))
        self.grad_sq_b = np.ones(vocab_size)
        self.grad_sq_bt = np.ones(vocab_size)
        self.loss_history = []

    def _weight(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0

    def train(self, X, epochs=50):
        """X: co-occurrence matrix"""
        if X.shape[0] != self.vocab_size:
            raise ValueError("X shape mismatch vocab_size")
        # Only train on non-zero entries for efficiency
        nonzero = np.argwhere(X > 0)
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(nonzero)
            for i,j in nonzero:
                X_ij = X[i,j]
                w = self._weight(X_ij)
                # prediction
                pred = np.dot(self.W[i], self.W_tilde[j]) + self.b[i] + self.b_tilde[j]
                log_X = np.log(X_ij)
                diff = pred - log_X
                f_diff = w * diff

                total_loss += 0.5 * w * diff**2

                # AdaGrad updates
                grad_W = f_diff * self.W_tilde[j]
                grad_Wt = f_diff * self.W[i]
                grad_b = f_diff
                grad_bt = f_diff

                self.grad_sq_W[i] += grad_W**2
                self.grad_sq_Wt[j] += grad_Wt**2
                self.grad_sq_b[i] += grad_b**2
                self.grad_sq_bt[j] += grad_bt**2

                self.W[i] -= self.lr * grad_W / np.sqrt(self.grad_sq_W[i])
                self.W_tilde[j] -= self.lr * grad_Wt / np.sqrt(self.grad_sq_Wt[j])
                self.b[i] -= self.lr * grad_b / np.sqrt(self.grad_sq_b[i])
                self.b_tilde[j] -= self.lr * grad_bt / np.sqrt(self.grad_sq_bt[j])

            self.loss_history.append(total_loss)
            if epoch % 10 == 0:
                print(f"GloVe epoch {epoch} loss {total_loss:.2f}")
        # Final embeddings = W + W_tilde (paper)
        return self.W + self.W_tilde
