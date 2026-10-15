import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

class EmbeddingVisualizer:
    def __init__(self, embeddings, vocab):
        self.emb = embeddings
        self.vocab = vocab

    def plot_pca(self, save_path=None):
        pca = PCA(n_components=2, random_state=42)
        pts = pca.fit_transform(self.emb)
        plt.figure(figsize=(10,7))
        plt.scatter(pts[:,0], pts[:,1], alpha=0.7, c=range(len(self.vocab)), cmap="tab10")
        for i, word in enumerate(self.vocab):
            plt.annotate(word, (pts[i,0], pts[i,1]), fontsize=11)
        plt.title("Embeddings - PCA Visualization")
        plt.grid(alpha=0.3)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return pts

    def plot_tsne(self, perplexity=5, save_path=None):
        if len(self.vocab) <= perplexity:
            perplexity = max(2, len(self.vocab)-1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        pts = tsne.fit_transform(self.emb)
        plt.figure(figsize=(10,7))
        plt.scatter(pts[:,0], pts[:,1], alpha=0.7, c=range(len(self.vocab)), cmap="tab10")
        for i, word in enumerate(self.vocab):
            plt.annotate(word, (pts[i,0], pts[i,1]), fontsize=11)
        plt.title("Embeddings - t-SNE Visualization")
        plt.grid(alpha=0.3)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return pts

    def plot_similarity_heatmap(self, save_path=None):
        norm = self.emb / (np.linalg.norm(self.emb, axis=1, keepdims=True)+1e-9)
        sim = norm @ norm.T
        plt.figure(figsize=(8,6))
        sns.heatmap