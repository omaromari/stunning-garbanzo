# =============================================================
# Module 05: The Full RAG Pipeline — Solution
# =============================================================
# NOTE: requires GPT4All model at MODEL_PATH.
# See ../README.md for download instructions.
# Try the exercise first before reading this file.
# Run with:
#   streamlit run solution.py
# =============================================================

MODEL_PATH = "path/to/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

import os
import re
from pathlib import Path

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="RAG Chatbot Tutorial", page_icon="🤖", layout="wide")


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

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def get_similarity_score(e1: list, e2: list) -> float:
    import numpy as np
    a, b = map(np.array, [e1, e2])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_most_similar_documents(query_emb: list, docs: list, top_k: int = 5) -> list:
    results = []
    for doc in docs:
        for chunk_emb in doc["embeddings"]:
            results.append({"document": doc, "similarity": get_similarity_score(query_emb, chunk_emb)})
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]

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

def load_and_embed_documents(doc_paths: list) -> list:
    embedded = []
    for path in doc_paths:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        raw_text = "".join(p.page_content for p in pages)
        text = re.sub(r'\s+', " ", raw_text.replace("\n", " ")).strip()
        chunks = chunk_text(text)
        embedded.append({"text": text, "chunks": chunks, "embeddings": get_embeddings(chunks)})
    return embedded


# ── Startup ───────────────────────────────────────────────────────────────

docs_dir = Path(__file__).parent.parent / "docs"
documents = [str(docs_dir / "Lipitor.pdf"), str(docs_dir / "Metformin.pdf")]
embeddings_file = str(Path(__file__).parent / "embeddings.json")

# Load from cache or generate and cache
if os.path.exists(embeddings_file):
    embedded_documents = load_embeddings(embeddings_file)
else:
    embedded_documents = load_and_embed_documents(documents)
    save_embeddings(embedded_documents, embeddings_file)


# ── App ───────────────────────────────────────────────────────────────────

def handle_user_input(model, user_input: str) -> str:
    user_embedding = get_embedding(user_input)
    similar_documents = find_most_similar_documents(user_embedding, embedded_documents, top_k=5)
    document_texts = [doc["document"].get("text", "") for doc in similar_documents]
    return get_answer_with_context(model, user_input, document_texts)


@st.cache_resource
def load_model():
    return GPT4All(MODEL_PATH)


def main():
    st.title("RAG Chatbot Tutorial")
    st.write("Ask me anything about the loaded documents!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    model = load_model()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["bot"])

    user_input = st.text_input("Your question:", "")
    if st.button("Ask") and user_input:
        answer = handle_user_input(model, user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": answer})
        st.rerun()


if __name__ == "__main__":
    main()
