
import streamlit as st
import numpy as np
from embedding_twin.preprocessing import TextPreprocessor, CooccurrenceBuilder
from embedding_twin.word2vec import SkipGram
from embedding_twin.similarity import SimilarityEngine
from embedding_twin.visualization import EmbeddingVisualizer

st.set_page_config(page_title="Embeddings Deep Dive - Digital Twin", layout="wide")
st.title("🧠 Week 2.3 Embeddings Deep Dive")
st.markdown("> How computers turn words into numbers to find meaning")

corpus = st.text_area("Training Corpus", "king queen man woman prince princess boy girl king loves queen man loves woman", height=100)
embed_dim = st.slider("Embedding Dim", 5, 50, 20)
epochs = st.slider("Epochs", 20, 300, 100)

if st.button("Train Embeddings"):
    prep = TextPreprocessor()
    tokens = prep.tokenize(corpus)
    vocab = prep.build_vocab(tokens)
    pairs = []
    for i,w in enumerate(tokens):
        if w not in prep.word2id: continue
        c = prep.word2id[w]
        for j in range(max(0,i-2), min(len(tokens), i+3)):
            if i==j: continue
            if tokens[j] in prep.word2id:
                pairs.append((c, prep.word2id[tokens[j]]))
    
    w2v = SkipGram(len(vocab), embed_dim=embed_dim, lr=0.1, neg_samples=3)
    emb = w2v.train(pairs, epochs=epochs)
    sim = SimilarityEngine(emb, vocab, prep.word2id)
    
    st.success(f"Trained on vocab: {vocab}")
    
    col1, col2 = st.columns(2)
    with col1:
        word = st.selectbox("Most similar to:", vocab)
        st.write(sim.most_similar(word, top_k=5))
    with col2:
        st.write("Analogy Test: king - man + woman")
        try:
            st.write(sim.analogy("man","king","woman"))
        except Exception as e:
            st.write(f"Need those words in vocab: {e}")
    
    viz = EmbeddingVisualizer(emb, vocab)
    fig = viz.plot_pca()
    st.pyplot(plt)
