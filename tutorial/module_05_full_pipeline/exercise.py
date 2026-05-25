# =============================================================
# Module 05: The Full RAG Pipeline — Exercise
# =============================================================
# Instructions:
#   1. Set MODEL_PATH below to the full path of your .gguf file.
#      See ../README.md for download instructions.
#   2. Replace every _____ with the correct code.
#   3. Run with Streamlit (NOT python):
#       streamlit run exercise.py
#      Then open the URL shown in your terminal (http://localhost:8501).
#   Hints and answers are in README.md
# =============================================================

# NOTE: requires GPT4All model at MODEL_PATH.
# Examples:
#   Windows:  r"C:\Users\you\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#   macOS:    "/Users/you/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#   Linux:    "/home/you/.local/share/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
MODEL_PATH = "path/to/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# ── Embedding model (loaded once at module level) ──────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="RAG Chatbot Tutorial", page_icon="🤖", layout="wide")


# ── Utility functions (from Modules 01–04) ────────────────────────────────

def load_embeddings(file_path: str) -> list:
    import json
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_embeddings(embeddings: list, file_path: str) -> None:
    import json
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)

def get_embedding(text: str) -> list:
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

def get_embeddings(texts: list) -> list:
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def get_similarity_score(embedding1: list, embedding2: list) -> float:
    import numpy as np
    a, b = map(np.array, [embedding1, embedding2])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_most_similar_documents(query_embedding: list, embedded_documents: list, top_k: int = 5) -> list:
    results = []
    for doc in embedded_documents:
        for chunk_embedding in doc["embeddings"]:
            similarity = get_similarity_score(query_embedding, chunk_embedding)
            results.append({"document": doc, "similarity": similarity})
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def get_context(query: str, documents: list, top_k: int = 5) -> str:
    context = [doc for doc in documents if query.lower() in doc.lower()]
    return "\n".join(context[:top_k])

def get_response(model, prompt: str) -> str:
    tokens = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for token in model.generate(prompt=prompt, top_k=1, streaming=True):
            tokens.append(token)
            placeholder.markdown("".join(tokens))
    return "".join(tokens)

def get_answer_with_context(model, query: str, documents: list) -> str:
    context = get_context(query, documents)
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
    return get_response(model, prompt)


# ── Document loading and embedding ────────────────────────────────────────

def load_and_embed_documents(doc_paths: list) -> list:
    """Load PDFs, clean text, chunk, and embed."""
    embedded = []
    for path in doc_paths:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        raw_text = "".join(p.page_content for p in pages)
        text = re.sub(r'\s+', " ", raw_text.replace("\n", " ")).strip()
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        embedded.append({"text": text, "chunks": chunks, "embeddings": embeddings})
    return embedded


# ── Startup: load or generate embeddings ──────────────────────────────────

docs_dir = Path(__file__).parent.parent / "docs"
documents = [str(docs_dir / "Lipitor.pdf"), str(docs_dir / "Metformin.pdf")]

embeddings_file = str(Path(__file__).parent / "embeddings.json")

# --- Exercise 1: Load embeddings from cache if it exists; otherwise generate ---
# FILL IN: Use os.path to check whether `embeddings_file` exists on disk
if _____.exists(embeddings_file):
    embedded_documents = load_embeddings(embeddings_file)
else:
    embedded_documents = load_and_embed_documents(documents)
    save_embeddings(embedded_documents, embeddings_file)


# ── Streamlit app ─────────────────────────────────────────────────────────

def handle_user_input(model, user_input: str) -> str:
    # --- Exercise 2: Embed the user's question ---
    # FILL IN: Call the function that converts a string to a 384-dim vector
    user_embedding = _____(user_input)

    # --- Exercise 3: Retrieve the top-5 most similar document chunks ---
    # FILL IN: Call find_most_similar_documents with the query embedding,
    #          the embedded_documents store, and top_k=5
    similar_documents = _____(user_embedding, embedded_documents, top_k=_____)

    # --- Exercise 4: Extract text from each retrieval result ---
    # FILL IN: Each item in similar_documents is {"document": {...}, "similarity": float}.
    #          Build a list of text strings from the nested "document" dict.
    document_texts = [doc["document"].get(_____, "") for doc in similar_documents]

    return get_answer_with_context(model, user_input, document_texts)


@st.cache_resource
def load_model():
    return GPT4All(MODEL_PATH)


def main():
    st.title("RAG Chatbot Tutorial")
    st.write("Ask me anything about the loaded documents!")

    # --- Exercise 5: Initialise chat history in session state ---
    # FILL IN: Check if "chat_history" is not yet in st.session_state
    if _____ not in st.session_state:
        st.session_state.chat_history = []

    model = load_model()

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["bot"])

    # User input
    user_input = st.text_input("Your question:", "")
    if st.button("Ask") and user_input:
        answer = handle_user_input(model, user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": answer})
        st.rerun()


if __name__ == "__main__":
    main()
