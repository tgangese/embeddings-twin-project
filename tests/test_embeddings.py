
import pytest
import numpy as np
from embedding_twin.preprocessing import TextPreprocessor, CooccurrenceBuilder
from embedding_twin.word2vec import SkipGram
from embedding_twin.similarity import SimilarityEngine

def test_preprocessor():
    p = TextPreprocessor()
    tokens = p.tokenize("Hello World! Hello")
    assert tokens == ["hello","world","hello"]
    vocab = p.build_vocab(tokens)
    assert "hello" in vocab
    with pytest.raises(ValueError):
        p.tokenize("")

def test_cooccurrence():
    p = TextPreprocessor()
    tokens = p.tokenize("king queen king")
    vocab = p.build_vocab(tokens)
    builder = CooccurrenceBuilder(window_size=1)
    X = builder.build(tokens, p.word2id)
    assert X.shape == (len(vocab), len(vocab))
    assert X.sum() > 0

def test_skipgram():
    p = TextPreprocessor()
    tokens = p.tokenize("king queen man woman king queen")
    vocab = p.build_vocab(tokens)
    pairs = [(p.word2id[tokens[i]], p.word2id[tokens[i+1]]) for i in range(len(tokens)-1)]
    model = SkipGram(len(vocab), embed_dim=8, neg_samples=1)
    emb = model.train(pairs, epochs=10)
    assert emb.shape == (len(vocab), 8)

def test_similarity():
    vocab = ["king","queen","man"]
    emb = np.array([[1,0],[0.9,0.1],[0.8,0.2]])
    w2id = {w:i for i,w in enumerate(vocab)}
    sim = SimilarityEngine(emb, vocab, w2id)
    assert sim.cosine("king","queen") > 0.9
    res = sim.most_similar("king", top_k=2)
    assert len(res) == 2

def test_oov():
    vocab = ["king","queen"]
    emb = np.random.randn(2,5)
    w2id = {w:i for i,w in enumerate(vocab)}
    sim = SimilarityEngine(emb, vocab, w2id)
    with pytest.raises(KeyError):
        sim.most_similar("prince")
