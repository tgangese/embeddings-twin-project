"""Demo that actually learns meaning - repeated corpus for stronger signal"""
from embedding_twin.preprocessing import TextPreprocessor, CooccurrenceBuilder
from embedding_twin.word2vec import SkipGram
from embedding_twin.glove import GloVe
from embedding_twin.similarity import SimilarityEngine
from embedding_twin.visualization import EmbeddingVisualizer
from embedding_twin.digital_twin_integration import TwinMemory
import os
os.makedirs("assets", exist_ok=True)

base = """
king queen man woman prince princess boy girl
king is man queen is woman prince is boy princess is girl
king loves queen man loves woman boy loves girl
king rules queen rules prince rules princess rules
king is king queen is queen man is man woman is woman
apple orange fruit car truck vehicle
apple is fruit orange is fruit car is vehicle truck is vehicle
"""
corpus = (base + "\n") * 15

prep = TextPreprocessor(min_freq=1)
tokens = prep.tokenize(corpus)
vocab = prep.build_vocab(tokens)
print(f"Vocab: {vocab}")
print(f"Tokens: {len(tokens)}")

co_builder = CooccurrenceBuilder(window_size=2)
X = co_builder.build(tokens, prep.word2id)

pairs = []
for i, w in enumerate(tokens):
    center = prep.word2id[w]
    l = max(0, i-2); r = min(len(tokens), i+3)
    for j in range(l,r):
        if i==j: continue
        if tokens[j] in prep.word2id:
            pairs.append((center, prep.word2id[tokens[j]]))

print(f"Training pairs: {len(pairs)}")

w2v = SkipGram(vocab_size=len(vocab), embed_dim=30, lr=0.025, neg_samples=5)
emb_w2v = w2v.train(pairs, epochs=250)

glove = GloVe(vocab_size=len(vocab), embed_dim=30, lr=0.05)
emb_glove = glove.train(X, epochs=80)

sim = SimilarityEngine(emb_w2v, vocab, prep.word2id)
print("\n--- Word2Vec Tests (should make sense now) ---")
print("most similar to king:", sim.most_similar("king", top_k=5))
print("most similar to queen:", sim.most_similar("queen", top_k=5))
print("most similar to man:", sim.most_similar("man", top_k=5))
print("analogy king - man + woman:", sim.analogy("man","king","woman", top_k=3))
print("analogy prince - boy + girl:", sim.analogy("boy","prince","girl", top_k=3))

sim_g = SimilarityEngine(emb_glove, vocab, prep.word2id)
print("\n--- GloVe Tests ---")
print("most similar to queen:", sim_g.most_similar("queen", top_k=5))

viz = EmbeddingVisualizer(emb_w2v, vocab)
viz.plot_pca(save_path="assets/demo_pca.png")
viz.plot_similarity_heatmap(save_path="assets/heatmap.png")
# viz.plot_loss(w2v.loss_history, label="Word2Vec Loss", save_path="assets/loss.png")  # not in visualizer
print("\nPlots saved to assets/")

print("\n--- Digital Twin Memory ---")
twin = TwinMemory(sim)
twin.add_memory("king rules kingdom", {"type": "royalty"})
twin.add_memory("queen rules kingdom", {"type": "royalty"})
twin.add_memory("apple orange are fruit", {"type": "food"})
twin.add_memory("car truck are vehicle", {"type": "vehicle"})
twin.add_memory("man woman love each other", {"type": "family"})

print("Recall 'king':", twin.recall("king", top_k=2))
print("Recall 'queen':", twin.recall("queen", top_k=2))
print("Recall 'fruit':", twin.recall("fruit", top_k=2))
print("Recall 'vehicle':", twin.recall("vehicle", top_k=2))
print("Summary:", twin.summarize())
