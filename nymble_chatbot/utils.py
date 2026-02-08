## The purpose of this file is to provide utility functions for the chatbot.
## It includes functions for chunking text from PDF files in docs/, generating embeddings, context retreival or RAG, and other helper functions.
## It also includes functions for loading and saving data, and for handling file paths.
import os
import json
import re
import numpy as np
import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import *

# Load the Sentence Transformer model (e.g., "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_embeddings(file_path: str) -> List[Dict[str, Any]]:
    """Load embeddings from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_embeddings(embeddings: List[Dict[str, Any]], file_path: str) -> None:
    """Save embeddings to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save a dictionary as a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Chunk text into smaller pieces for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding(text: str, model_name: str = "text-embedding-ada-002") -> List[float]:
    """Generate an embedding for the given text using OpenAI's API."""
    # embeddings = OpenAIEmbeddings(model=model_name)
    ## We dont have access to OpenAI API, so we will use the local model instead
    embedding = embedding_model.encode(text, convert_to_numpy=True).tolist()
    return embedding

def get_embeddings(texts: List[str], model_name: str = "text-embedding-ada-002") -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    # embeddings = OpenAIEmbeddings(model=model_name)
    embeddings_list =  embedding_model.encode(texts, convert_to_numpy=True).tolist()
    return embeddings_list

def get_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate the cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_most_similar_documents(query_embedding: List[float], vectorstore: Chroma, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve the most similar documents from the vector store."""
    results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    return results

def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    """Load documents from a directory based on file extension."""
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            ext = get_file_extension(file_path)
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                continue  # Skip unsupported file types
            
            documents.extend(loader.load())
    return documents

def find_most_similar_documents(query_embedding: List[float], embedded_documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Find the most similar documents manually using cosine similarity."""
    similarities = []
    for doc in embedded_documents:
        for chunk_embedding in doc["embeddings"]:
            similarity = get_similarity_score(query_embedding, chunk_embedding)
            similarities.append({"document": doc, "similarity": similarity})
    # Sort by similarity and return the top_k results
    similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]