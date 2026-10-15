
from typing import List, Dict
import numpy as np

class TwinMemory:
    """
    Production-ready Digital Twin memory that uses embeddings
    for semantic recall - extends Week 2 capabilities
    """
    def __init__(self, similarity_engine):
        self.engine = similarity_engine
        self.memories: List[Dict] = []

    def add_memory(self, text: str, metadata: dict = None):
        if not text or not isinstance(text, str):
            raise ValueError("Memory text must be non-empty string")
        if len(text.strip()) < 2:
            raise ValueError("Memory too short")
        self.memories.append({"text": text, "meta": metadata or {}})

    def recall(self, query: str, top_k=3):
        if not self.memories:
            return []
        candidates = [m["text"] for m in self.memories]
        try:
            results = self.engine.semantic_search(query, candidates, top_k=top_k)
        except Exception as e:
            # fallback to keyword
            results = [(c, 1.0) for c in candidates if query.lower() in c.lower()][:top_k]
        # enrich with metadata
        enriched = []
        for text, score in results:
            mem = next(m for m in self.memories if m["text"]==text)
            enriched.append({"text": text, "score": score, "meta": mem["meta"]})
        return enriched

    def summarize(self):
        return {"total_memories": len(self.memories), "vocab_size": len(self.engine.vocab)}
