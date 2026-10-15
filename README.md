
# 🧠 Embeddings Deep Dive — Digital Twin Intelligence

> How computers turn words into numbers (Vectors) to find 'meaning'

A production-ready implementation of Word2Vec and GloVe from scratch, built for the Week 2.3 Challenge Project and integrated with a Digital Twin.

### 🚀 Live Demo
```bash
streamlit run app.py
python demo.py

✨ Features
Word2Vec from scratch: Skip-gram with Negative Sampling (NumPy only)
GloVe from scratch: Global co-occurrence factorization with AdaGrad
Semantic Operations: cosine similarity, analogy, semantic search
Visualization: PCA, t-SNE + similarity heatmaps
Digital Twin Integration: Memory layer that uses embeddings for semantic recall
📊 Real Results

Vocab: 17 words | Tokens: 1005 | Pairs: 4014
GloVe: queen -> man 0.95, king 0.93  ✅
Word2Vec: man -> woman 0.98  ✅
Digital Twin: Recall 'king' -> 'king rules kingdom' 0.95

📊 Visuals
Image unavailable. Please retry the request.
Image unavailable. Please retry the request.
Image unavailable. Please retry the request.
Image unavailable. Please retry the request.

🔧 Install & Run
Bash
pip install -r requirements.txt
python demo.py
streamlit run app.py
pytest tests/
Author: tgangese - Week 2.3
MD

