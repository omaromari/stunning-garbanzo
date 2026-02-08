## The purpose of this file is to launch the chatbot application using Streamlit, referencing the utility functions defined in utils.py, the rag_agent.py to initialize the chatbot, and providing a user interface for interaction.
# ## It also includes functions for handling user input, displaying chat history, and managing the chatbot's state.

import os
import json
import re
import numpy as np
import tiktoken
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from utils import load_embeddings, save_embeddings, save_json, chunk_text, get_embedding, get_embeddings, get_similarity_score, get_most_similar_documents, load_documents_from_directory, find_most_similar_documents
from rag_agent import load_model, get_response, get_context, get_answer, get_answer_with_context
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader

# Set the page configuration for Streamlit
st.set_page_config(
    page_title="Nymble Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and embed the PDF documents Lipitor and Metformin
def load_and_embed_documents(documents: List[str]) -> List[Dict[str, Any]]:
    """Load and embed the documents."""
    embedded_documents = []
    for doc in documents:
        # Load the document
        loader = PyMuPDFLoader(doc)
        document = loader.load()
        # Extract text from the document
        text = ""
        ## print(document)
        print(document)
        for page in document:
            text += page.page_content  # Access the page content directly
        # Clean and process the text
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        # Chunk the text into smaller pieces
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        embedded_documents.append({
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings
        })
    return embedded_documents

# Path to the embeddings file
embeddings_file = "embeddings.json"

if os.path.exists(embeddings_file):
    # Load embeddings from file
    embedded_documents = load_embeddings(embeddings_file)
else:
    # Generate embeddings and save them
    documents = [
        "docs/lipitor.pdf",
        "docs/metformin.pdf"
    ]
    embedded_documents = load_and_embed_documents(documents)
    save_embeddings(embedded_documents, embeddings_file)

# # Initialize the vector store
# vectorstore = Chroma.from_documents(
#     documents=embedded_documents,
#     embedding_function=None,  # No need to recompute embeddings
#     persist_directory="chroma_db"
# )

# Function to handle user input and display chat history
def handle_user_input(model, user_input: str, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Handle user input and update chat history."""
    # Get the embedding for the user input
    user_embedding = get_embedding(user_input)

    # Retrieve the most similar documents from the vector store
    similar_documents = find_most_similar_documents(user_embedding, embedded_documents, top_k=5)

    # Debugging: Print the structure of similar_documents
    print("Similar Documents:", similar_documents)

    # Extract the text content from the similar documents
    document_texts = [doc.get("text", "") for doc in similar_documents]

    # Get the answer with context
    answer = get_answer_with_context(model, user_input, document_texts)

    # Update chat history
    chat_history.append({"user": user_input, "bot": answer})
    return chat_history

# Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.title("Nymble Chatbot")
    st.write("Ask me anything about Lipitor and Metformin!")
    model = load_model()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"**User:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

    # User input
    user_input = st.text_input("Your question:", "")

    # Handle user input when the button is clicked
    if st.button("Ask"):
        if user_input:
            st.session_state.chat_history = handle_user_input(model, user_input, st.session_state.chat_history)
            # Clear the input field
            st.text_input("Your question:", "", key="user_input")
            st.rerun()
        else:
            st.warning("Please enter a question.")
            st.rerun()

if __name__ == "__main__":
    main()
