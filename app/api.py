from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGEngine

app = FastAPI(title="AI-Powered Document Question Answering System", version="1.0.0")
engine = RAGEngine()

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4
    use_openai: bool = True

@app.get("/")
def root() -> dict:
    return {"project": "AI-Powered Document Question Answering System", "status": "ready"}

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed_documents": len(engine.documents),
        "indexed_chunks": engine.indexed_chunks,
        "embedding_backend": engine.embedding_backend,
        "vector_backend": engine.vector_backend,
    }

@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required.")
    documents = []
    for uploaded in files:
        if not uploaded.filename or not uploaded.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{uploaded.filename or 'File'} is not a PDF.")
        documents.append((uploaded.filename, await uploaded.read()))
    try:
        count = engine.index_pdfs(documents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"documents": [name for name, _ in documents], "chunks_indexed": count}

@app.post("/documents/process")
def process_documents() -> dict:
    return {"status": "ready", "documents": engine.documents, "chunks_indexed": engine.indexed_chunks}

@app.post("/ask")
def ask_question(request: QuestionRequest) -> dict:
    if not engine.documents:
        raise HTTPException(status_code=400, detail="Upload and process at least one document first.")
    try:
        result = engine.ask(request.question, top_k=request.top_k, use_openai=request.use_openai)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"answer": result.answer, "sources": result.sources, "retrieved_context": result.retrieved_chunks, "generation_mode": result.generation_mode}
