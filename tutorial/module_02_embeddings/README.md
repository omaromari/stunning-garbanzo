# Module 02: Generating & Caching Embeddings

## Concept Introduction

Text is not natively comparable by a computer — you cannot subtract one sentence from another. **Embeddings** solve this by converting any piece of text into a list of numbers (a *vector*). The model is trained so that semantically similar texts produce vectors that are close together in space, while dissimilar texts produce vectors that are far apart. This is what lets a RAG system answer "what is the side effect of Lipitor?" by finding chunks that mention muscle pain, even if the exact phrase is never used in the query.

The **`sentence-transformers`** library provides pre-trained models that generate embeddings locally — no API key or GPU required for inference at this scale. The model `all-MiniLM-L6-v2` outputs **384-dimensional vectors**: each text becomes a list of 384 floating-point numbers. Bigger models produce larger vectors (OpenAI's `text-embedding-ada-002` produces 1 536 dimensions), but `all-MiniLM-L6-v2` is fast and more than sufficient for small document sets.

**Caching** is important because computing embeddings is slow. Once you have embedded a document, you save the vectors to a JSON file (`embeddings.json`). On subsequent runs the application loads the cached file instead of re-computing, cutting startup time from tens of seconds to milliseconds. The trade-off: if you change your documents you need to delete the cache and regenerate.

## Key Terms

- **embedding** — a fixed-size list of floats representing the semantic content of a text string
- **embedding model** — a neural network trained to map text → vector; different models produce different-sized vectors
- **`SentenceTransformer`** — the class used to load a pre-trained sentence embedding model
- **`.encode()`** — the method that converts a string (or list of strings) into one (or more) embedding vectors
- **`convert_to_numpy=True`** — flag that returns a NumPy array instead of a PyTorch tensor
- **`.tolist()`** — converts a NumPy array to a plain Python list (needed for JSON serialisation)
- **JSON cache** — a file storing pre-computed embeddings so they do not need to be recomputed on every run

---

## Exercises

Open `exercise.py`, fill in every `_____`, then run:

```bash
python exercise.py        (Windows)
python3 exercise.py       (macOS / Linux)
```

---

### Exercise 1 — Load the embedding model

```python
embedding_model = SentenceTransformer(_____)
```

<details>
<summary>Hint</summary>

The model name is a string. The original project uses the lightweight `all-MiniLM-L6-v2` model.

</details>

<details>
<summary>Answer</summary>

```python
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
```

</details>

---

### Exercise 2 — Encode a single text string

```python
embedding = embedding_model.encode(text, _____)._____ 
```

<details>
<summary>Hint</summary>

`encode()` can return either a PyTorch tensor or a NumPy array. Pass the flag that requests a NumPy array. Then convert the array to a plain Python list so it can be stored as JSON.

</details>

<details>
<summary>Answer</summary>

```python
embedding = embedding_model.encode(text, convert_to_numpy=True).tolist()
```

</details>

---

### Exercise 3 — Encode a batch of texts

```python
embeddings_list = embedding_model.encode(texts, _____)._____ 
```

<details>
<summary>Hint</summary>

Exactly the same pattern as Exercise 2 — `encode()` accepts both a single string and a list of strings.

</details>

<details>
<summary>Answer</summary>

```python
embeddings_list = embedding_model.encode(texts, convert_to_numpy=True).tolist()
```

Encoding a batch is much faster than calling `encode()` in a loop because the model processes all texts in a single forward pass.

</details>

---

### Exercise 4 — Save embeddings to a JSON file

```python
with open(file_path, _____, encoding="utf-8") as f:
    json._____(embeddings, f, ensure_ascii=False, indent=4)
```

<details>
<summary>Hint</summary>

To write a file you need the `"w"` mode. `json.dump()` writes a Python object to an open file handle.

</details>

<details>
<summary>Answer</summary>

```python
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=4)
```

`ensure_ascii=False` preserves non-ASCII characters (e.g., medical symbols). `indent=4` makes the file human-readable.

</details>

---

### Exercise 5 — Load embeddings from a JSON file

```python
with open(file_path, _____, encoding="utf-8") as f:
    return json._____(f)
```

<details>
<summary>Hint</summary>

Reading a file uses `"r"` mode. `json.load()` reads from an open file handle and returns the parsed Python object.

</details>

<details>
<summary>Answer</summary>

```python
with open(file_path, "r", encoding="utf-8") as f:
    return json.load(f)
```

</details>

---

## Further Reading

- [Sentence Transformers usage guide](https://www.sbert.net/docs/sentence_transformer/usage/usage.html) — `.encode()` parameters, batch processing, and choosing models
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — architecture details and benchmark scores
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings) — cloud-hosted alternative using `text-embedding-ada-002` (1 536 dims)
