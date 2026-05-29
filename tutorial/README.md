# Building a RAG Chatbot — Tutorial

A hands-on, fill-in-the-blank walkthrough of Retrieval-Augmented Generation (RAG),
built around the [`nymble_chatbot`](../nymble_chatbot/) project.

## What You Will Build

By the end of this tutorial you will understand every piece that goes into a working
RAG chatbot: loading documents, splitting them into chunks, converting those chunks
into vector embeddings, retrieving the most relevant chunks for a user query,
constructing a context-aware prompt, and generating a streaming response with a
local LLM.

## Prerequisites

- Python 3.9 or higher
- Basic Python familiarity (functions, lists, dicts)
- ~5 GB free disk space (for the LLM in Modules 4–5)
- [Git](https://git-scm.com/) installed

Clone this repository to get all tutorial files:

```bash
git clone https://github.com/omaromari/stunning-garbanzo.git
cd stunning-garbanzo/tutorial
```

Run the environment check before you start:

```bash
# Windows
python setup.py

# macOS / Linux
python3 setup.py
```

## How to Use This Tutorial

1. Work through modules **in order** — each builds on the previous one.
2. Open `exercise.py` in your editor, read the comments, and replace every `_____` with the correct code.
3. Run your file and check the output:
   ```bash
   python module_0X_.../exercise.py
   ```
4. Compare your answer to `solution.py`.
5. If you get stuck, open the module's `README.md` — each exercise has a collapsible **Hint** and **Answer**.

> **Modules 1–3** run without any LLM. You only need `sentence-transformers` and `langchain`.
> **Modules 4–5** require the GPT4All Llama-3 model (see download instructions below).

---

## Downloading the LLM (Required for Modules 4 and 5)

Modules 1–3 are fully offline. For Modules 4 and 5 you need the
**Meta-Llama-3-8B-Instruct** quantized model (~4.7 GB).

### Option A — GPT4All Desktop App (Easiest)

1. Download the GPT4All app: https://www.nomic.ai/gpt4all
2. Open the app → **Models** tab → search **"Meta Llama 3 8B Instruct"**
3. Click **Download**
4. Default model save location:
   - **Windows:** `C:\Users\<you>\AppData\Local\nomic.ai\GPT4All\`
   - **macOS:** `~/Library/Application Support/nomic.ai/GPT4All/`
   - **Linux:** `~/.local/share/nomic.ai/GPT4All/`

### Option B — Direct Download (Hugging Face)

Download `Meta-Llama-3-8B-Instruct.Q4_0.gguf` from:
https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF

Place the `.gguf` file anywhere on your machine and note the full path.

### Setting `MODEL_PATH` in the exercises

At the top of `module_04_llm_and_prompting/exercise.py` and
`module_05_full_pipeline/exercise.py`, replace the placeholder:

```python
MODEL_PATH = "path/to/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
```

with your actual path:

```python
# Windows
MODEL_PATH = r"C:\Users\you\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# macOS / Linux
MODEL_PATH = "/home/you/.local/share/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
```

---

## Sample Documents

The `docs/` folder contains `Lipitor.pdf` and `Metformin.pdf` — the same source
documents used by the original `nymble_chatbot` project. All exercises load these
PDFs directly; no extra setup is needed.

---

## Module Index

| Module | Topic | LLM needed? |
|--------|-------|:-----------:|
| [01 — Loading & Chunking](module_01_loading_and_chunking/) | Load PDFs, clean text, split into chunks | No |
| [02 — Embeddings](module_02_embeddings/) | Convert chunks to vectors, cache to JSON | No |
| [03 — Retrieval](module_03_retrieval/) | Cosine similarity, top-k search | No |
| [04 — LLM & Prompting](module_04_llm_and_prompting/) | Load GPT4All, build context-aware prompt | Yes |
| [05 — Full Pipeline](module_05_full_pipeline/) | Chain everything into a Streamlit chatbot | Yes |

---

## Further Reading

See [resources.md](resources.md) for a curated list of vector databases, cloud LLM APIs, and deployment guides that extend what you build here.
