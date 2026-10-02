from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .document_processor import DocumentChunk, build_chunks
from .embeddings import EmbeddingModel
from .generator import generate_answer
from .vector_store import VectorStore
@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    retrieved_chunks: list[dict]
    generation_mode: str
class RAGEngine:
    def __init__(self, embedding_model: str="all-MiniLM-L6-v2") -> None:
        self.embedder=EmbeddingModel(embedding_model); self.store=VectorStore(); self.documents=[]
    @property
    def indexed_chunks(self): return len(self.store.chunks)
    @property
    def vector_backend(self): return self.store.backend
    @property
    def embedding_backend(self): return self.embedder.backend
    def reset(self): self.store.clear(); self.documents=[]
    def index_pdfs(self,documents: Sequence[tuple[str,bytes]]) -> int:
        chunks=[]; document_names=[]
        for filename,file_bytes in documents:
            file_chunks=build_chunks(file_bytes,filename)
            if file_chunks: chunks.extend(file_chunks); document_names.append(filename)
        if not chunks: raise ValueError("No extractable text was found in the supplied PDF files.")
        vectors=self.embedder.encode_corpus([c.text for c in chunks]); self.store.add(chunks,vectors); self.documents=document_names
        return len(chunks)
    def ask(self,question:str,top_k:int=4,use_openai:bool=True)->RAGResult:
        question=question.strip()
        if not question: raise ValueError("Question cannot be empty.")
        query_vector=self.embedder.encode([question])[0]; retrieved=self.store.search(query_vector,top_k=top_k)
        if self.embedding_backend=="tfidf-fallback":
            import re
            stop={"the","is","are","a","an","what","which","how","when","where","who","does","do","can","could","please","and","to","of","in","on","for"}
            q_terms={w for w in re.findall(r"[a-zA-Z0-9]+",question.lower()) if w not in stop}
            corpus_terms=set(re.findall(r"[a-zA-Z0-9]+"," ".join(c.text for c,_ in retrieved).lower()))
            if q_terms and not q_terms.intersection(corpus_terms):
                return RAGResult("I could not find the answer in the uploaded documents.",[],[],"guardrail")
        answer,mode=generate_answer(question,retrieved,use_openai=use_openai)
        sources=[]; seen=set()
        for chunk,_ in retrieved:
            if chunk.citation not in seen: seen.add(chunk.citation); sources.append(chunk.citation)
        payload=[{"text":chunk.text,"source":chunk.citation,"similarity":round(score,4)} for chunk,score in retrieved]
        return RAGResult(answer,sources,payload,mode)
