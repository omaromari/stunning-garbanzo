# =============================================================
# Module 01: Document Loading & Chunking — Solution
# =============================================================
# Try the exercise first before reading this file.
# Run with:
#   python solution.py        (Windows)
#   python3 solution.py       (macOS / Linux)
# =============================================================

import os
import re
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()


def extract_text_from_pages(document) -> str:
    text = ""
    for page in document:
        text += page.page_content
    return text


def clean_text(text: str) -> str:
    # Replace PDF line-break artifacts with spaces
    text = text.replace("\n", " ")
    # Collapse runs of whitespace (tabs, multiple spaces) into one space
    text = re.sub(r'\s+', " ", text)
    return text.strip()


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # ~1 000 chars fits comfortably in most embedding models
        chunk_overlap=200,   # 20% overlap reduces information loss at boundaries
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # coarse → fine splitting order
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    docs_dir = Path(__file__).parent.parent / "docs"
    pdf_path = str(docs_dir / "Lipitor.pdf")

    print("=== Loading PDF ===")
    document = load_pdf(pdf_path)
    print(f"Pages loaded: {len(document)}")

    print("\n=== Extracting text ===")
    raw_text = extract_text_from_pages(document)
    print(f"Raw text length: {len(raw_text)} characters")
    print(f"Preview: {raw_text[:150]}...")
    print()

    print("=== Cleaning text ===")
    cleaned = clean_text(raw_text)
    print(f"Cleaned length: {len(cleaned)} characters")
    print(f"Preview: {cleaned[:150]}...")
    print()

    print("=== Chunking text ===")
    chunks = chunk_text(cleaned)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk)
