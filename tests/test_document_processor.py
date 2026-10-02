from src.document_processor import split_text

def test_split_text_returns_chunks():
    text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks=split_text(text,chunk_size=40,overlap=5)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)
