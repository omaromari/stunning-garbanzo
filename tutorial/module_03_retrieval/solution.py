# =============================================================
# Module 03: Semantic Retrieval (Cosine Similarity) — Solution
# =============================================================
# Try the exercise first before reading this file.
# Run with:
#   python solution.py        (Windows)
#   python3 solution.py       (macOS / Linux)
# =============================================================

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def get_similarity_score(embedding1: list, embedding2: list) -> float:
    a = np.array(embedding1)
    b = np.array(embedding2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Cosine similarity = dot_product / (|A| * |B|)
    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)


def find_most_similar_documents(
    query_embedding: list,
    embedded_documents: list,
    top_k: int = 5,
) -> list:
    results = []
    for doc in embedded_documents:
        for chunk_embedding in doc["embeddings"]:
            similarity = get_similarity_score(query_embedding, chunk_embedding)
            results.append({"document": doc, "similarity": similarity})
    # Sort highest similarity first, then keep only the top_k
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    cache_path = Path(__file__).parent.parent / "module_02_embeddings" / "embeddings_cache.json"

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(text: str) -> list:
        return embed_model.encode(text, convert_to_numpy=True).tolist()

    if cache_path.exists():
        print(f"Loading embeddings from: {cache_path}\n")
        with open(cache_path, "r", encoding="utf-8") as f:
            embedded_documents = json.load(f)
    else:
        print("Cache not found — using inline sample data.\n")
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
