from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import fitz

@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: int
    @property
    def citation(self) -> str:
        return f"{self.source} - Page {self.page}"

def extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc
    pages = []
    try:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text") or "")
            if text:
                pages.append((page_number, text))
    finally:
        doc.close()
    return pages

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) > chunk_size:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end]
                if end < len(paragraph):
                    space = piece.rfind(" ")
                    if space > chunk_size * 0.65:
                        end = start + space
                        piece = paragraph[start:end]
                chunks.append(piece.strip())
                if end >= len(paragraph):
                    break
                start = max(end - overlap, start + 1)
            current = ""
        else:
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]

def build_chunks(file_bytes: bytes, source: str, chunk_size: int = 900, overlap: int = 150) -> list[DocumentChunk]:
    pages = extract_pdf_pages(file_bytes)
    output = []
    chunk_id = 0
    for page_number, page_text in pages:
        for chunk in split_text(page_text, chunk_size=chunk_size, overlap=overlap):
            chunk_id += 1
            output.append(DocumentChunk(text=chunk, source=source, page=page_number, chunk_id=chunk_id))
    return output

def build_chunks_from_pages(pages: Iterable[tuple[int, str]], source: str, chunk_size: int = 900, overlap: int = 150) -> list[DocumentChunk]:
    output = []
    chunk_id = 0
    for page_number, page_text in pages:
        for chunk in split_text(clean_text(page_text), chunk_size=chunk_size, overlap=overlap):
            chunk_id += 1
            output.append(DocumentChunk(text=chunk, source=source, page=page_number, chunk_id=chunk_id))
    return output
