# Module 03: Semantic Retrieval (Cosine Similarity)

## Concept Introduction

Now that every document chunk has been converted into a vector, you need a way to find the chunks most relevant to a user's question. This is the *retrieval* step in RAG — and it is what separates RAG from naive "put everything in the prompt" approaches.

**Cosine similarity** is the standard metric for comparing text embeddings. Rather than measuring the raw distance between two vectors (Euclidean distance), cosine similarity measures the *angle* between them. Two vectors pointing in the same direction have a cosine similarity of **+1** (maximally similar), two perpendicular vectors score **0** (unrelated), and two pointing in opposite directions score **−1** (maximally dissimilar). In practice, sentence embeddings almost never produce negative similarities, so scores are typically between 0 and 1.

The formula is:

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

where `A · B` is the dot product and `‖A‖` is the L2 norm of vector A.

**Top-k retrieval** runs this calculation for every stored chunk and returns the k highest-scoring ones. With only a few hundred chunks this brute-force approach is fast enough. Production systems use a **vector database** (Pinecone, ChromaDB, Vertex AI Search) which uses approximate nearest-neighbour (ANN) algorithms to search millions of vectors efficiently.

## Key Terms

- **cosine similarity** — angle-based measure of vector similarity; range −1 to +1
- **dot product** — sum of element-wise products: `Σ(aᵢ × bᵢ)`
- **L2 norm** — the Euclidean length of a vector: `√(Σ aᵢ²)` — in NumPy: `np.linalg.norm()`
- **top-k** — returning only the k highest-scoring results instead of all of them
- **brute-force search** — comparing a query against every stored vector; simple but O(n) per query
- **ANN (Approximate Nearest Neighbour)** — fast search algorithms used by vector databases; trades a small accuracy loss for orders-of-magnitude speed gains

---

## Exercises

Open `exercise.py`, fill in every `_____`, then run:

```bash
python exercise.py        (Windows)
python3 exercise.py       (macOS / Linux)
```

---

### Exercise 1 — Compute the dot product

```python
dot_product = np._____(embedding1, embedding2)
```

<details>
<summary>Hint</summary>

NumPy has a function that computes the dot product of two 1-D arrays. Its name is the same as the mathematical symbol.

</details>

<details>
<summary>Answer</summary>

```python
dot_product = np.dot(embedding1, embedding2)
```

</details>

---

### Exercise 2 — Compute the L2 norms

```python
norm_a = np.linalg._____(embedding1)
norm_b = np.linalg._____(embedding2)
```

<details>
<summary>Hint</summary>

The function is in `np.linalg` (linear algebra sub-module). The L2 norm is also called the Euclidean norm or vector length.

</details>

<details>
<summary>Answer</summary>

```python
norm_a = np.linalg.norm(embedding1)
norm_b = np.linalg.norm(embedding2)
```

</details>

---

### Exercise 3 — Compute cosine similarity

```python
similarity = _____ / (_____ * _____)
```

<details>
<summary>Hint</summary>

Apply the formula: dot product divided by the product of the two norms.

</details>

<details>
<summary>Answer</summary>

```python
similarity = dot_product / (norm_a * norm_b)
```

</details>

---

### Exercise 4 — Iterate over stored chunk embeddings

```python
for doc in embedded_documents:
    for chunk_embedding in doc[_____]:
        similarity = get_similarity_score(query_embedding, chunk_embedding)
        results.append({"document": doc, "similarity": similarity})
```

<details>
<summary>Hint</summary>

Look at how the document record is structured in Module 02's `save_embeddings()`. Each record is a dict with three keys: `"text"`, `"chunks"`, and `"embeddings"`.

</details>

<details>
<summary>Answer</summary>

```python
for chunk_embedding in doc["embeddings"]:
```

</details>

---

### Exercise 5 — Sort by similarity and return top-k

```python
results = sorted(results, key=lambda x: x[_____], reverse=_____)
return results[:_____]
```

<details>
<summary>Hint</summary>

You want the highest similarity scores first (descending order). The `key` should pick the `"similarity"` value from each result dict. The `top_k` parameter controls how many to return.

</details>

<details>
<summary>Answer</summary>

```python
results = sorted(results, key=lambda x: x["similarity"], reverse=True)
return results[:top_k]
```

</details>

---

## Further Reading

- [Vector similarity explained (Pinecone)](https://www.pinecope.io/learn/vector-similarity/) — visual walkthrough of dot product, cosine, and Euclidean distance
- [ChromaDB getting started](https://docs.trychroma.com/getting-started) — open-source local vector database that replaces the manual similarity loop
- [Pinecone quickstart](https://docs.pinecone.io/guides/get-started/quickstart) — managed cloud vector database for production workloads
