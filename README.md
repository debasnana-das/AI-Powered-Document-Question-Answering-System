# AI-Powered Document Question Answering System

A practical Retrieval-Augmented Generation (RAG) application for answering natural-language questions from uploaded PDF documents. The system extracts document text, splits it into chunks, generates vector representations, performs similarity search, and produces grounded answers with page-level source citations.

## Project Objective

Users upload PDFs such as policies, manuals, reports, or company documents and ask questions in natural language. The application retrieves relevant document sections and uses them as the only context for answer generation.

## RAG Architecture

`PDF -> text extraction -> cleaning -> chunking -> embeddings -> vector index -> semantic retrieval -> relevant context -> answer generation -> sources`

## Components

- PyMuPDF for PDF text extraction.
- Sentence Transformers (`all-MiniLM-L6-v2`) for local embeddings, with TF-IDF fallback.
- FAISS when available for vector similarity search; otherwise NumPy cosine similarity.
- OpenAI for optional LLM-based grounded generation when `OPENAI_API_KEY` is configured.
- Grounded extractive fallback so the application remains usable without an API key.
- Guardrail that declines to answer when retrieval confidence is too weak.
- Streamlit user interface.
- FastAPI endpoints for upload, processing status, and question answering.

## Installation

```bash
python -m venv venv
# Windows CMD
venv\Scripts\activate
# Windows PowerShell
# .\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Streamlit

Main file:

`app/streamlit_app.py`

```bash
python -m streamlit run app/streamlit_app.py
```

Upload one or more PDFs, click **Process documents**, then ask a natural-language question.

## Optional OpenAI Generation

Set `OPENAI_API_KEY` in the environment or Streamlit Cloud secrets. Without the key, the application uses the grounded extractive fallback.

Example:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Never commit a real API key to GitHub.

## Run FastAPI

```bash
uvicorn app.api:app --reload
```

Interactive documentation:

`http://127.0.0.1:8000/docs`

### POST `/documents/upload`

Upload and index one or more PDFs.

### POST `/documents/process`

Return the current processing/indexing status.

### POST `/ask`

Example request:

```json
{
  "question": "What is the work-from-home policy?",
  "top_k": 4,
  "use_openai": false
}
```

## Testing

```bash
pytest -q
```

## Sample Document

`data/sample_documents/Employee_Policy.pdf` is included for a quick demonstration.

## Notes

The vector index is stored in memory for the current process. Restarting the Streamlit app or FastAPI server clears the index, so documents need to be uploaded again.

## Streamlit Cloud

Use branch `main` and set the entrypoint to:

`app/streamlit_app.py`

The root `requirements.txt` contains the application dependencies.
