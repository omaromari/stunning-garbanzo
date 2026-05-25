# Module 04: LLM Inference & Prompt Construction

## Concept Introduction

Retrieval gives you the most relevant chunks; the LLM turns those chunks into a human-readable answer. This module covers loading a local language model, extracting context from retrieved chunks, and constructing a prompt that tells the model exactly how to use that context.

**GPT4All** is a library that runs quantized LLMs locally — no API key, no internet connection, no GPU required for inference (though a GPU speeds things up). The model file is a `.gguf` (GGUF format) containing the model weights compressed to 4-bit integers. The Llama-3-8B model compresses from ~16 GB (full precision) to ~4.7 GB while retaining most of its capability.

**Context extraction** bridges retrieval and prompting. After Module 03 returns the top-k similar document records, you need to extract the actual text strings from those records and filter them down to the portions that are most relevant to the query. The original project uses a simple keyword-overlap filter: it keeps only the retrieved texts that contain the query terms as a substring. This is a lightweight fallback, not a replacement for the semantic search already done.

**Prompt engineering** is the craft of writing the instruction that the LLM receives. A well-structured prompt tells the model its role, its constraints, where the context is, and where the question is. Using clear section headers (like `## Context: ##`) helps the model parse the prompt reliably. The constraint "if the answer is not in the context, say you do not know" is important — without it, the model will hallucinate an answer.

**Streaming** means the model outputs one token at a time rather than waiting to produce the full response. This makes the UI feel responsive and lets users start reading while the model is still generating.

## Key Terms

- **GGUF** — a binary format for storing quantized LLM weights; the standard format for GPT4All models
- **quantization** — compressing model weights to lower precision (e.g., 4-bit integers) to reduce memory and speed up inference
- **`GPT4All(path)`** — loads a local `.gguf` model from the given file path
- **`model.generate()`** — generates a response to a prompt; supports streaming and sampling parameters
- **`top_k` (sampling)** — at each generation step, only the top-k most likely tokens are considered; `top_k=1` is greedy (always pick the most likely token)
- **streaming** — returns tokens one at a time via a generator instead of waiting for the full response
- **context window** — the maximum number of tokens the model can process at once
- **hallucination** — when a model generates plausible-sounding but factually incorrect information not grounded in its context

---

## Exercises

> **Before running:** set `MODEL_PATH` at the top of `exercise.py` to the full path of your `.gguf` file.
> See the main tutorial [README.md](../README.md) for download instructions.

```bash
python exercise.py        (Windows)
python3 exercise.py       (macOS / Linux)
```

---

### Exercise 1 — Load the GPT4All model

```python
model = _____(MODEL_PATH)
```

<details>
<summary>Hint</summary>

`GPT4All` is already imported. Instantiate it with the model file path as the only argument.

</details>

<details>
<summary>Answer</summary>

```python
model = GPT4All(MODEL_PATH)
```

</details>

---

### Exercise 2 — Call `generate()` with streaming

```python
for token in model.generate(prompt=prompt, _____, _____):
    tokens.append(token)
```

<details>
<summary>Hint</summary>

You need two keyword arguments: one to set the sampling parameter to 1 (greedy decoding), and one to enable token-by-token output.

</details>

<details>
<summary>Answer</summary>

```python
for token in model.generate(prompt=prompt, top_k=1, streaming=True):
    tokens.append(token)
```

`top_k=1` makes the output deterministic (always picks the most probable next token). `streaming=True` turns `generate()` into a generator that yields one token string at a time.

</details>

---

### Exercise 3 — Keyword-based context filter

```python
for doc in documents:
    if _____.lower() in _____.lower():
        context.append(doc)
```

<details>
<summary>Hint</summary>

You want to keep documents that contain the query somewhere in their text. The check should be case-insensitive.

</details>

<details>
<summary>Answer</summary>

```python
for doc in documents:
    if query.lower() in doc.lower():
        context.append(doc)
```

This is a simple substring check — if any part of the query appears in the document text, the document is included.

</details>

---

### Exercise 4 — Limit context list to top_k

```python
context = context[:_____]
```

<details>
<summary>Hint</summary>

The function receives a `top_k` parameter that controls the maximum number of context documents to use.

</details>

<details>
<summary>Answer</summary>

```python
context = context[:top_k]
```

</details>

---

### Exercise 5 — Join context documents into a single string

```python
return _____.join(context)
```

<details>
<summary>Hint</summary>

You want each context document to appear on its own line when printed. Use the string method that concatenates a list with a separator.

</details>

<details>
<summary>Answer</summary>

```python
return "\n".join(context)
```

</details>

---

### Exercise 6 — Build the RAG prompt

```python
prompt = f"""
    You are a helpful assistant tasked with answering questions based on the provided context.
    The context is a collection of documents that may contain the answer to the question.

    ## Instructions: ##
    _____

    ## Context: ##
    {context}

    ## Question: ##
    {query}
    """
```

<details>
<summary>Hint</summary>

The instructions section should tell the model to: read the context carefully, give a concise answer, say "I do not know" if the answer is not in the context, and offer to answer more questions.

</details>

<details>
<summary>Answer</summary>

```python
prompt = f"""
    You are a helpful assistant tasked with answering questions based on the provided context.
    The context is a collection of documents that may contain the answer to the question.

    ## Instructions: ##
    - Read the context carefully.
    - Provide a concise and accurate answer to the question.
    - If the answer is not found in the context, state that you do not know the answer.
    - Offer the user to ask another question if they wish.

    ## Context: ##
    {context}

    ## Question: ##
    {query}
    """
```

The section headers (`## Instructions: ##`, `## Context: ##`, `## Question: ##`) help the model distinguish between meta-instructions and actual content.

</details>

---

## Further Reading

- [GPT4All Python docs](https://docs.gpt4all.io/gpt4all_python/home.html) — full `generate()` parameter reference
- [OpenAI Responses API](https://platform.openai.com/docs/guides/responses) — cloud-hosted alternative that replaces the local `GPT4All` model
- [Prompt engineering guide (OpenAI)](https://platform.openai.com/docs/guides/prompt-engineering) — strategies for writing reliable prompts
