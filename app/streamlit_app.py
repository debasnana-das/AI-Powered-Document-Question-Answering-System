from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGEngine

st.set_page_config(page_title="AI Document Q&A | RAG", page_icon="📄", layout="wide")
st.title("AI-Powered Document Question Answering System")
st.caption("Upload PDFs and ask natural-language questions grounded in their content.")

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    use_openai = st.checkbox("Use OpenAI for answer generation", value=bool(os.getenv("OPENAI_API_KEY")))
    st.markdown("**Embeddings:** " + st.session_state.engine.embedding_backend)
    st.markdown("**Vector search:** " + st.session_state.engine.vector_backend)
    st.info("Without an OpenAI API key, the app uses a grounded extractive fallback and still cites document pages.")

uploaded = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

if st.button("Process documents", type="primary", disabled=not uploaded):
    docs = [(f.name, f.getvalue()) for f in uploaded]
    try:
        with st.spinner("Extracting text, creating chunks, generating embeddings, and indexing..."):
            count = st.session_state.engine.index_pdfs(docs)
        st.session_state.documents_loaded = True
        st.success(f"Indexed {count} text chunks from {len(docs)} document(s).")
    except Exception as exc:
        st.error(f"Document processing failed: {exc}")

if st.session_state.documents_loaded:
    st.metric("Indexed chunks", st.session_state.engine.indexed_chunks)

question = st.text_input("Ask a question", placeholder="What is the work-from-home policy?")

if st.button("Ask", disabled=not st.session_state.documents_loaded):
    try:
        with st.spinner("Searching the document and generating a grounded answer..."):
            result = st.session_state.engine.ask(question, top_k=top_k, use_openai=use_openai)
        st.subheader("Answer")
        st.write(result.answer)
        st.subheader("Sources")
        for source in result.sources:
            st.write(f"- {source}")
        with st.expander("Retrieved context"):
            st.dataframe(result.retrieved_chunks, use_container_width=True)
        st.caption(f"Generation mode: {result.generation_mode}")
    except Exception as exc:
        st.error(str(exc))
