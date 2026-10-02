from __future__ import annotations
from typing import Sequence
import numpy as np

class EmbeddingModel:
    """Sentence-transformer embeddings with a lightweight TF-IDF fallback."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        self.backend = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), min_df=1, sublinear_tf=True)
            self.backend = "tfidf-fallback"
    def fit(self, texts: Sequence[str]) -> None:
        if self._tfidf is not None:
            self._tfidf.fit(list(texts))
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0,0), dtype=np.float32)
        if self._model is not None:
            vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)
        if self._tfidf is None:
            raise RuntimeError("Embedding backend is not initialized.")
        return self._tfidf.transform(list(texts)).toarray().astype(np.float32)
    def encode_corpus(self, texts: Sequence[str]) -> np.ndarray:
        if self._tfidf is not None:
            self.fit(texts)
        return self.encode(texts)
