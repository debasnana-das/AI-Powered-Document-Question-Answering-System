from __future__ import annotations
from dataclasses import asdict
from typing import Sequence
import numpy as np
from .document_processor import DocumentChunk

class VectorStore:
    """In-memory vector index with FAISS acceleration when available."""
    def __init__(self) -> None:
        self.chunks = []
        self.vectors = None
        self._faiss = None
        self._index = None
        try:
            import faiss
            self._faiss = faiss
        except Exception:
            self._faiss = None
    @property
    def backend(self) -> str:
        return "FAISS" if self._index is not None else "NumPy cosine similarity"
    def clear(self) -> None:
        self.chunks=[]; self.vectors=None; self._index=None
    def add(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors): raise ValueError("Each document chunk must have one embedding vector.")
        if not chunks: return
        vectors=np.asarray(vectors,dtype=np.float32)
        norms=np.linalg.norm(vectors,axis=1,keepdims=True); norms[norms==0]=1.0
        vectors=vectors/norms
        self.chunks=list(chunks); self.vectors=vectors; self._index=None
        if self._faiss is not None:
            try:
                index=self._faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors); self._index=index
            except Exception: self._index=None
    def search(self, query_vector: np.ndarray, top_k: int=4):
        if not self.chunks or self.vectors is None: return []
        top_k=max(1,min(top_k,len(self.chunks)))
        query=np.asarray(query_vector,dtype=np.float32).reshape(1,-1)
        qnorm=np.linalg.norm(query,axis=1,keepdims=True); qnorm[qnorm==0]=1.0; query=query/qnorm
        if self._index is not None:
            scores,indices=self._index.search(query,top_k)
            return [(self.chunks[int(i)],float(s)) for i,s in zip(indices[0],scores[0]) if i>=0]
        scores=(self.vectors@query[0]).astype(float); indices=np.argsort(-scores)[:top_k]
        return [(self.chunks[int(i)],float(scores[int(i)])) for i in indices]
    def to_dict(self):
        return {"backend":self.backend,"chunks":[asdict(c) for c in self.chunks],"chunk_count":len(self.chunks)}
