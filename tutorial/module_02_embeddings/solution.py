# =============================================================
# Module 02: Generating & Caching Embeddings — Solution
# =============================================================
# Try the exercise first before reading this file.
# Run with:
#   python solution.py        (Windows)
#   python3 solution.py       (macOS / Linux)
# =============================================================

import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list:
    return embedding_model.encode(text, convert_to_numpy=True).tolist()


def get_embeddings(texts: list) -> list:
    # Batch encoding is faster than a loop — the model processes all texts at once
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()


def save_embeddings(embeddings: list, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)


def load_embeddings(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    cache_path = Path(__file__).parent / "embeddings_cache.json"
    docs_dir = Path(__file__).parent.parent / "docs"

    sys.path.insert(0, str(Path(__file__).parent.parent / "module_01_loading_and_chunking"))
    from solution import load_pdf, extract_text_from_pages, clean_text, chunk_text

    embedded_documents = []
    for pdf_name in ("Lipitor.pdf", "Metformin.pdf"):
        document = load_pdf(str(docs_dir / pdf_name))
        text = clean_text(extract_text_from_pages(document))
        chunks = chunk_text(text)
        embedded_documents.append({"name": pdf_name, "text": text, "chunks": chunks})

    total_chunks = sum(len(d["chunks"]) for d in embedded_documents)
    print(f"=== {total_chunks} chunks to embed across {len(embedded_documents)} documents ===\n")

    print("Generating embeddings...")
    for doc in embedded_documents:
        doc["embeddings"] = get_embeddings(doc["chunks"])
    total_vecs = sum(len(d["embeddings"]) for d in embedded_documents)
    print(f"Total vectors: {total_vecs} x {len(embedded_documents[0]['embeddings'][0])} dimensions")
    print(f"First vector (first 8 values): {embedded_documents[0]['embeddings'][0][:8]}")
    print()

    query_embedding = get_embedding("What are the side effects of Lipitor?")
    print(f"Query embedding dimensions: {len(query_embedding)}")
    print()

    save_embeddings(embedded_documents, str(cache_path))
    print(f"Embeddings saved to: {cache_path}")

    loaded = load_embeddings(str(cache_path))
    assert len(loaded) == 2
    print(f"Cache loaded successfully — {sum(len(d['chunks']) for d in loaded)} total chunks across {len(loaded)} documents.")
    print()
    print("Module 02 complete! The cache file will be used by Module 03.")
