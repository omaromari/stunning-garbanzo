# =============================================================
# Module 02: Generating & Caching Embeddings — Exercise
# =============================================================
# Instructions:
#   Replace every _____ with the correct code.
#   Fill in ALL blanks before running:
#       python exercise.py        (Windows)
#       python3 exercise.py       (macOS / Linux)
#   Hints and answers are in README.md
# =============================================================

import json
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer


# =============================================================
# Exercise 1: Load the embedding model
# =============================================================

# --- Exercise 1: Instantiate SentenceTransformer ---
# FILL IN: Pass the name of the model used in the original project.
#          It outputs 384-dimensional vectors and runs locally without a GPU.
embedding_model = SentenceTransformer(_____)


# =============================================================
# Exercises 2 & 3: Encode text into vectors
# =============================================================

def get_embedding(text: str) -> list:
    # --- Exercise 2: Encode a single string and return a Python list ---
    # FILL IN: Call .encode() with the flag that returns a NumPy array,
    #          then convert the result to a plain Python list.
    embedding = embedding_model.encode(text, _____)._____
    return embedding


def get_embeddings(texts: list) -> list:
    # --- Exercise 3: Encode a list of strings and return a list of lists ---
    # FILL IN: Same pattern as Exercise 2 — encode() accepts a list too.
    embeddings_list = embedding_model.encode(texts, _____)._____
    return embeddings_list


# =============================================================
# Exercises 4 & 5: Save and load the embedding cache
# =============================================================

def save_embeddings(embeddings: list, file_path: str) -> None:
    # --- Exercise 4: Write the embeddings list to a JSON file ---
    # FILL IN: Open the file in write mode with UTF-8 encoding,
    #          then use json.___() to serialise the data.
    with open(file_path, _____, encoding="utf-8") as f:
        json._____(embeddings, f, ensure_ascii=False, indent=4)


def load_embeddings(file_path: str) -> list:
    # --- Exercise 5: Read the embeddings list from a JSON file ---
    # FILL IN: Open the file in read mode with UTF-8 encoding,
    #          then use json.___() to deserialise the data.
    with open(file_path, _____, encoding="utf-8") as f:
        return json._____(f)


# =============================================================
# Main — runs your implementations and prints results
# =============================================================
if __name__ == "__main__":
    cache_path = Path(__file__).parent / "embeddings_cache.json"
    docs_dir = Path(__file__).parent.parent / "docs"

    # Load and chunk PDFs (reusing module 01 logic)
    sys.path.insert(0, str(Path(__file__).parent.parent / "module_01_loading_and_chunking"))
    from solution import load_pdf, extract_text_from_pages, clean_text, chunk_text

    all_chunks = []
    embedded_documents = []
    for pdf_name in ("Lipitor.pdf", "Metformin.pdf"):
        document = load_pdf(str(docs_dir / pdf_name))
        text = clean_text(extract_text_from_pages(document))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        embedded_documents.append({"name": pdf_name, "text": text, "chunks": chunks})

    chunks = all_chunks

    print(f"=== {len(chunks)} chunks to embed ===\n")

    # Generate embeddings per document
    print("Generating embeddings (this may take a few seconds on first run)...")
    for doc in embedded_documents:
        doc["embeddings"] = get_embeddings(doc["chunks"])
    total_vecs = sum(len(d["embeddings"]) for d in embedded_documents)
    print(f"Total vectors: {total_vecs} x {len(embedded_documents[0]['embeddings'][0])} dimensions")
    print(f"First vector (first 8 values): {embedded_documents[0]['embeddings'][0][:8]}")
    print()

    # Test single-text embedding
    query = "What are the side effects of Lipitor?"
    query_embedding = get_embedding(query)
    print(f"Query embedding dimensions: {len(query_embedding)}")
    print()

    # Save to cache
    save_embeddings(embedded_documents, str(cache_path))
    print(f"Embeddings saved to: {cache_path}")

    # Reload and verify
    loaded = load_embeddings(str(cache_path))
    assert len(loaded) == 2, "Expected 2 documents in cache!"
    print(f"Cache loaded successfully — {sum(len(d['chunks']) for d in loaded)} total chunks across {len(loaded)} documents.")
    print()
    print("Module 02 complete! The cache file will be used by Module 03.")
