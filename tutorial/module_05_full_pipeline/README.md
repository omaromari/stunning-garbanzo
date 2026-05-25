# Module 05: The Full RAG Pipeline

## Concept Introduction

This final module chains everything from Modules 1–4 into a single working chatbot. Instead of individual scripts that print to a terminal, the pipeline now runs inside **Streamlit** — a Python library that turns a plain `.py` file into an interactive web application.

The complete flow is:

```
PDF / text files
    ↓ (Module 01) PyMuPDFLoader → clean text → RecursiveCharacterTextSplitter
Chunks
    ↓ (Module 02) SentenceTransformer → 384-dim vectors
Embedding cache (embeddings.json)
    ↓ (check cache on startup)
User query
    ↓ (Module 02) SentenceTransformer → query vector
    ↓ (Module 03) cosine similarity → top-5 similar chunks
    ↓ (Module 04) keyword filter → context string → f-string prompt
GPT4All (Llama-3-8B)
    ↓ streaming tokens
Streamlit chat UI
```

**Streamlit session state** (`st.session_state`) is a dictionary that persists across re-runs of the script. Every time the user clicks a button or types input, Streamlit re-executes the entire script from top to bottom. Without session state, the chat history would be lost on every interaction. By storing `chat_history` in `st.session_state`, the history survives each re-run.

The **cache-or-generate pattern** at startup is a common optimisation: check whether a pre-computed result already exists on disk; if it does, load it; if not, compute it and save it. This prevents re-embedding the documents every time the app starts.

## Key Terms

- **Streamlit** — Python library that creates interactive web UIs from scripts; runs with `streamlit run app.py`
- **`st.session_state`** — persistent key-value store that survives Streamlit re-runs within a session
- **`st.rerun()`** — tells Streamlit to re-execute the script immediately (used after updating chat history)
- **`st.chat_message()`** — renders a chat bubble in the Streamlit UI
- **cache-or-generate** — check if result file exists → load it; otherwise compute and save it

---

## Exercises

> **Before running:** set `MODEL_PATH` at the top of `exercise.py`.

Run with Streamlit (not `python`):

```bash
streamlit run exercise.py        (Windows and macOS / Linux)
```

Then open the local URL shown in your terminal (usually http://localhost:8501).

---

### Exercise 1 — Cache check: load or generate embeddings

```python
if _____.exists(embeddings_file):
    embedded_documents = load_embeddings(embeddings_file)
else:
    embedded_documents = load_and_embed_documents(documents)
    save_embeddings(embedded_documents, embeddings_file)
```

<details>
<summary>Hint</summary>

Use the `os.path` module to check whether a file exists. The function takes a path string and returns a boolean.

</details>

<details>
<summary>Answer</summary>

```python
if os.path.exists(embeddings_file):
    embedded_documents = load_embeddings(embeddings_file)
else:
    embedded_documents = load_and_embed_documents(documents)
    save_embeddings(embedded_documents, embeddings_file)
```

</details>

---

### Exercise 2 — Embed the user's query

```python
user_embedding = _____(user_input)
```

<details>
<summary>Hint</summary>

You need a function that takes a string and returns a 384-dimensional vector. It was implemented in Module 02.

</details>

<details>
<summary>Answer</summary>

```python
user_embedding = get_embedding(user_input)
```

</details>

---

### Exercise 3 — Retrieve the most similar document chunks

```python
similar_documents = _____(user_embedding, embedded_documents, top_k=_____)
```

<details>
<summary>Hint</summary>

This is the retrieval function from Module 03. Use `top_k=5` to match the original project.

</details>

<details>
<summary>Answer</summary>

```python
similar_documents = find_most_similar_documents(user_embedding, embedded_documents, top_k=5)
```

</details>

---

### Exercise 4 — Extract text from retrieval results

```python
document_texts = [doc.get(_____, "") for doc in similar_documents]
```

<details>
<summary>Hint</summary>

Each item in `similar_documents` is a dict `{"document": {...}, "similarity": float}`.
The nested document record has a `"text"` key containing the full document text.

</details>

<details>
<summary>Answer</summary>

```python
document_texts = [doc.get("text", "") for doc in similar_documents]
```

Wait — `similar_documents` contains dicts with a `"document"` key. The correct extraction is:

```python
document_texts = [doc["document"].get("text", "") for doc in similar_documents]
```

</details>

---

### Exercise 5 — Initialise Streamlit chat history

```python
if _____ not in st.session_state:
    st.session_state.chat_history = []
```

<details>
<summary>Hint</summary>

`st.session_state` behaves like a Python dict. Use the `in` operator to check whether a key exists.

</details>

<details>
<summary>Answer</summary>

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```

</details>

---

## Further Reading

- [Streamlit docs](https://docs.streamlit.io/) — full API reference including `st.chat_message`, `st.session_state`, and `st.rerun`
- [Streamlit Community Cloud](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app) — deploy your finished chatbot to a public URL for free
- [Docker for Python apps](https://docs.docker.com/language/python/) — containerise the app (a `DockerFile` already exists in `nymble_chatbot/`)
