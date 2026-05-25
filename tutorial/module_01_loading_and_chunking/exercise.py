# =============================================================
# Module 01: Document Loading & Chunking — Exercise
# =============================================================
# Instructions:
#   Replace every _____ with the correct code.
#   Fill in ALL blanks before running:
#       python exercise.py        (Windows)
#       python3 exercise.py       (macOS / Linux)
#   Hints and answers are in README.md
# =============================================================

import os
import re
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =============================================================
# Exercise 1 & 2: Load a PDF and extract its text
# =============================================================
# These two functions show you the exact API used in nymble_chatbot/app.py.
# They are defined here for practice but the main block below uses the
# included sample.txt so you can run without a PDF on hand.
# To test with a real PDF, uncomment the "# PDF:" lines at the bottom.

def load_pdf(pdf_path: str):
    # --- Exercise 1: Instantiate the PDF loader ---
    # FILL IN: Pass the PDF file path to PyMuPDFLoader
    loader = PyMuPDFLoader(_____)
    return loader.load()


def extract_text_from_pages(document) -> str:
    text = ""
    for page in document:
        # --- Exercise 2: Access the text content of each page ---
        # FILL IN: What attribute on a LangChain Document holds the page text?
        text += page._____
    return text


# =============================================================
# Exercise 3 & 4: Clean the extracted text
# =============================================================

def clean_text(text: str) -> str:
    # --- Exercise 3: Replace newline characters ---
    # FILL IN: Replace every "\n" with a single space " "
    text = text.replace(_____, _____)

    # --- Exercise 4: Collapse multiple whitespace characters ---
    # FILL IN: Use re.sub() to collapse runs of whitespace into one space.
    #          What regex pattern matches one-or-more whitespace characters?
    text = re.sub(_____, " ", text)

    return text.strip()


# =============================================================
# Exercise 5 & 6: Chunk the cleaned text
# =============================================================

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        # --- Exercise 5a: Set the maximum characters per chunk ---
        # FILL IN: The original project uses 1 000-character chunks
        chunk_size=_____,

        # --- Exercise 5b: Set the overlap between consecutive chunks ---
        # FILL IN: The original project uses 200-character overlap
        chunk_overlap=_____,

        length_function=len,

        # --- Exercise 5c: Provide the ordered list of separators ---
        # FILL IN: Try paragraph breaks first, then line breaks, then words, then chars
        separators=_____,
    )

    # --- Exercise 6: Split the text ---
    # FILL IN: Call the correct method on `splitter` to split `text` into a list of strings
    return splitter._____(text)


# =============================================================
# Main — runs your implementations and prints the results
# =============================================================
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
