# =============================================================
# Module 03: Semantic Retrieval (Cosine Similarity) — Exercise
# =============================================================
# Instructions:
#   Replace every _____ with the correct code.
#   Fill in ALL blanks before running:
#       python exercise.py        (Windows)
#       python3 exercise.py       (macOS / Linux)
#   Hints and answers are in README.md
#
# Prerequisites:
#   Run module_02_embeddings/solution.py first to generate embeddings_cache.json.
#   The main block also works standalone with inline sample data if the cache
#   does not exist.
# =============================================================

import json
import sys
from pathlib import Path
import numpy as np


# =============================================================
# Exercises 1, 2, 3: Compute cosine similarity between two vectors
# =============================================================

def get_similarity_score(embedding1: list, embedding2: list) -> float:
    """Return the cosine similarity between two embedding vectors."""
    a = np.array(embedding1)
    b = np.array(embedding2)

    # --- Exercise 1: Compute the dot product ---
    # FILL IN: NumPy function that computes the dot product of two 1-D arrays
    dot_product = np._____(a, b)

    # --- Exercise 2: Compute the L2 norms ---
    # FILL IN: np.linalg function that returns the Euclidean length of a vector
    norm_a = np.linalg._____(a)
    norm_b = np.linalg._____(b)

    # --- Exercise 3: Apply the cosine similarity formula ---
    # FILL IN: dot_product divided by the product of the two norms
    similarity = _____ / (_____ * _____)

    return float(similarity)


# =============================================================
# Exercises 4 & 5: Find the top-k most similar document chunks
# =============================================================

def find_most_similar_documents(
    query_embedding: list,
    embedded_documents: list,
    top_k: int = 5,
) -> list:
    """Return the top_k document records ranked by cosine similarity."""
    results = []

    for doc in embedded_documents:
        # --- Exercise 4: Iterate over each chunk's embedding ---
        # FILL IN: Access the list of per-chunk vectors in each document record.
        #          Each record is a dict with keys "text", "chunks", "embeddings".
        for chunk_embedding in doc[_____]:
            similarity = get_similarity_score(query_embedding, chunk_embedding)
            results.append({"document": doc, "similarity": similarity})

    # --- Exercise 5: Sort descending by similarity and slice to top_k ---
    # FILL IN: Sort `results` so the highest similarity is first,
    #          then return only the first top_k items.
    results = sorted(results, key=lambda x: x[_____], reverse=_____)
    return results[:_____]


# =============================================================
# Main — demonstrate retrieval on sample data
# =============================================================
if __name__ == "__main__":
    cache_path = Path(__file__).parent.parent / "module_02_embeddings" / "embeddings_cache.json"

    # Load the SentenceTransformer model to encode queries
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(text: str) -> list:
        return embed_model.encode(text, convert_to_numpy=True).tolist()

    if cache_path.exists():
        print(f"Loading embeddings from: {cache_path}\n")
        with open(cache_path, "r", encoding="utf-8") as f:
            embedded_documents = json.load(f)
    else:
        print("Cache not found — using inline sample data.")
        print("Run module_02_embeddings/solution.py first for a richer demo.\n")
        sample_chunks = [
            "Lipitor is used to lower cholesterol and triglycerides.",
            "Common side effects of Lipitor include muscle pain and weakness.",
            "Metformin is the first-line medication for type 2 diabetes.",
            "Metformin works by decreasing glucose production in the liver.",
        ]
        chunk_embeddings = embed_model.encode(sample_chunks, convert_to_numpy=True).tolist()
        embedded_documents = [{
            "text": " ".join(sample_chunks),
            "chunks": sample_chunks,
            "embeddings": chunk_embeddings,
        }]

    queries = [
        "What are the side effects of Lipitor?",
        "How does Metformin work?",
        "Tell me about cholesterol medication.",
    ]

    for query in queries:
        query_emb = encode(query)
        top_results = find_most_similar_documents(query_emb, embedded_documents, top_k=3)
        print(f"Query: \"{query}\"")
        for rank, result in enumerate(top_results, start=1):
            snippet = result["document"]["text"][:120].replace("\n", " ")
            print(f"  #{rank}  similarity={result['similarity']:.4f}  |  {snippet}...")
        print()
