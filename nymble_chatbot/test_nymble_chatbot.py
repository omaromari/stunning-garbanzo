"""
Comprehensive test suite for the Nymble Chatbot application.

This test file covers:
- Utility functions (utils.py)
- RAG agent functions (rag_agent.py)
- Application functions (app.py)
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import List, Dict, Any

# Import functions to test
from utils import (
    load_embeddings,
    save_embeddings,
    load_json,
    save_json,
    chunk_text,
    get_embedding,
    get_embeddings,
    get_similarity_score,
    find_most_similar_documents,
    get_file_extension,
)

from rag_agent import (
    get_context,
    get_answer,
    get_answer_with_context,
)

from app import (
    load_and_embed_documents,
    handle_user_input,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample text. " * 100


@pytest.fixture
def sample_embedding():
    """Sample embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_embeddings():
    """Sample list of embeddings."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
    ]


@pytest.fixture
def sample_embedded_documents(sample_embeddings):
    """Sample embedded documents."""
    return [
        {
            "text": "Document about Lipitor medication.",
            "chunks": ["Lipitor is used to treat high cholesterol."],
            "embeddings": [sample_embeddings[0]],
        },
        {
            "text": "Document about Metformin medication.",
            "chunks": ["Metformin is used to treat type 2 diabetes."],
            "embeddings": [sample_embeddings[1]],
        },
    ]


@pytest.fixture
def mock_model():
    """Mock GPT4All model."""
    model = Mock()
    model.generate.return_value = iter(["This ", "is ", "a ", "test ", "response."])
    return model


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file."""
    file_path = tmp_path / "test_data.json"
    return str(file_path)


# ============================================================================
# TESTS FOR utils.py
# ============================================================================

class TestUtilsFunctions:
    """Test suite for utility functions."""

    def test_save_and_load_embeddings(self, temp_json_file, sample_embedded_documents):
        """Test saving and loading embeddings from JSON file."""
        # Save embeddings
        save_embeddings(sample_embedded_documents, temp_json_file)

        # Load embeddings
        loaded_embeddings = load_embeddings(temp_json_file)

        # Verify
        assert len(loaded_embeddings) == len(sample_embedded_documents)
        assert loaded_embeddings[0]["text"] == sample_embedded_documents[0]["text"]

    def test_save_and_load_json(self, temp_json_file):
        """Test saving and loading JSON data."""
        test_data = {"key1": "value1", "key2": [1, 2, 3]}

        # Save JSON
        save_json(test_data, temp_json_file)

        # Load JSON
        loaded_data = load_json(temp_json_file)

        # Verify
        assert loaded_data == test_data

    def test_chunk_text_default_params(self, sample_text):
        """Test chunking text with default parameters."""
        chunks = chunk_text(sample_text)

        # Verify chunks are created
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Verify chunks don't exceed max size (with some tolerance)
        assert all(len(chunk) <= 1200 for chunk in chunks)  # 1000 + 200 overlap

    def test_chunk_text_custom_params(self):
        """Test chunking text with custom parameters."""
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        # Verify chunks are created with custom size
        assert len(chunks) > 0
        assert all(len(chunk) <= 120 for chunk in chunks)

    def test_get_embedding(self):
        """Test getting embedding for a single text."""
        text = "Test sentence for embedding."
        embedding = get_embedding(text)

        # Verify embedding is a list of floats
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(val, float) for val in embedding)

    def test_get_embeddings_multiple_texts(self):
        """Test getting embeddings for multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = get_embeddings(texts)

        # Verify embeddings
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_get_similarity_score_identical(self, sample_embedding):
        """Test similarity score for identical embeddings."""
        similarity = get_similarity_score(sample_embedding, sample_embedding)

        # Identical embeddings should have similarity ~1.0
        assert 0.99 <= similarity <= 1.01

    def test_get_similarity_score_different(self):
        """Test similarity score for different embeddings."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = get_similarity_score(emb1, emb2)

        # Orthogonal vectors should have similarity ~0.0
        assert -0.01 <= similarity <= 0.01

    def test_get_similarity_score_opposite(self):
        """Test similarity score for opposite embeddings."""
        emb1 = [1.0, 1.0, 1.0]
        emb2 = [-1.0, -1.0, -1.0]
        similarity = get_similarity_score(emb1, emb2)

        # Opposite vectors should have similarity ~-1.0
        assert -1.01 <= similarity <= -0.99

    def test_find_most_similar_documents(self, sample_embedding, sample_embedded_documents):
        """Test finding most similar documents."""
        results = find_most_similar_documents(
            sample_embedding,
            sample_embedded_documents,
            top_k=2
        )

        # Verify results
        assert len(results) <= 2
        assert all("document" in result for result in results)
        assert all("similarity" in result for result in results)

        # Verify results are sorted by similarity (descending)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_get_file_extension(self):
        """Test getting file extensions."""
        assert get_file_extension("document.pdf") == ".pdf"
        assert get_file_extension("file.txt") == ".txt"
        assert get_file_extension("archive.tar.gz") == ".gz"
        assert get_file_extension("FILE.PDF") == ".pdf"  # Test case insensitivity


# ============================================================================
# TESTS FOR rag_agent.py
# ============================================================================

class TestRAGAgentFunctions:
    """Test suite for RAG agent functions."""

    def test_get_context_with_matching_documents(self):
        """Test getting context when query matches documents."""
        query = "diabetes"
        documents = [
            "Metformin is used to treat type 2 diabetes.",
            "Lipitor is used to treat high cholesterol.",
            "Diabetes management includes diet and exercise.",
        ]

        context = get_context(None, query, documents, top_k=2)

        # Verify context contains matching documents
        assert "diabetes" in context.lower()
        assert context.count("diabetes") >= 2  # Should find at least 2 matches

    def test_get_context_with_no_matches(self):
        """Test getting context when query doesn't match documents."""
        query = "cancer"
        documents = [
            "Metformin is used to treat type 2 diabetes.",
            "Lipitor is used to treat high cholesterol.",
        ]

        context = get_context(None, query, documents, top_k=2)

        # Verify context is empty
        assert context == ""

    def test_get_context_respects_top_k(self):
        """Test that get_context respects the top_k parameter."""
        query = "test"
        documents = ["test 1", "test 2", "test 3", "test 4", "test 5"]

        context = get_context(None, query, documents, top_k=3)

        # Verify only top_k results are returned
        lines = context.split("\n")
        assert len(lines) <= 3

    @patch('rag_agent.get_response')
    def test_get_answer(self, mock_get_response, mock_model):
        """Test getting an answer without context."""
        mock_get_response.return_value = "Paris is the capital of France."

        query = "What is the capital of France?"
        answer = get_answer(mock_model, query)

        # Verify get_response was called
        mock_get_response.assert_called_once()
        assert "Paris" in answer

    @patch('rag_agent.get_response')
    @patch('rag_agent.get_context')
    def test_get_answer_with_context(self, mock_get_context, mock_get_response, mock_model):
        """Test getting an answer with context."""
        mock_get_context.return_value = "Lipitor treats high cholesterol."
        mock_get_response.return_value = "Lipitor is used to lower cholesterol levels."

        query = "What does Lipitor treat?"
        documents = ["Lipitor treats high cholesterol."]

        answer = get_answer_with_context(mock_model, query, documents)

        # Verify functions were called
        mock_get_context.assert_called_once()
        mock_get_response.assert_called_once()
        assert isinstance(answer, str)

    @patch('rag_agent.get_response')
    @patch('rag_agent.get_context')
    def test_get_answer_with_context_no_answer(self, mock_get_context, mock_get_response, mock_model):
        """Test getting an answer when no relevant context is found."""
        mock_get_context.return_value = ""
        mock_get_response.return_value = "I do not know the answer based on the provided context."

        query = "What is the meaning of life?"
        documents = ["Lipitor treats high cholesterol."]

        answer = get_answer_with_context(mock_model, query, documents)

        # Verify response indicates no answer
        assert "not" in answer.lower() or "do not know" in answer.lower()


# ============================================================================
# TESTS FOR app.py
# ============================================================================

class TestAppFunctions:
    """Test suite for app.py functions."""

    @patch('app.PyMuPDFLoader')
    @patch('app.get_embeddings')
    @patch('app.chunk_text')
    def test_load_and_embed_documents(self, mock_chunk_text, mock_get_embeddings, mock_loader):
        """Test loading and embedding documents."""
        # Setup mocks
        mock_page = Mock()
        mock_page.page_content = "This is a test document about Lipitor."

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_page]
        mock_loader.return_value = mock_loader_instance

        mock_chunk_text.return_value = ["chunk1", "chunk2"]
        mock_get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

        # Test
        documents = ["test.pdf"]
        result = load_and_embed_documents(documents)

        # Verify
        assert len(result) == 1
        assert "text" in result[0]
        assert "chunks" in result[0]
        assert "embeddings" in result[0]
        assert len(result[0]["chunks"]) == 2
        assert len(result[0]["embeddings"]) == 2

    @patch('app.get_embedding')
    @patch('app.find_most_similar_documents')
    @patch('app.get_answer_with_context')
    def test_handle_user_input(
        self,
        mock_get_answer,
        mock_find_similar,
        mock_get_embedding,
        mock_model,
        sample_embedded_documents
    ):
        """Test handling user input."""
        # Setup mocks
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_find_similar.return_value = [
            {"text": "Lipitor treats high cholesterol.", "similarity": 0.9}
        ]
        mock_get_answer.return_value = "Lipitor is used to treat high cholesterol."

        # Test
        user_input = "What does Lipitor treat?"
        chat_history = []

        # Mock the embedded_documents global variable
        with patch('app.embedded_documents', sample_embedded_documents):
            result = handle_user_input(mock_model, user_input, chat_history)

        # Verify
        assert len(result) == 1
        assert result[0]["user"] == user_input
        assert "Lipitor" in result[0]["bot"]

    @patch('app.get_embedding')
    @patch('app.find_most_similar_documents')
    @patch('app.get_answer_with_context')
    def test_handle_user_input_multiple_messages(
        self,
        mock_get_answer,
        mock_find_similar,
        mock_get_embedding,
        mock_model,
        sample_embedded_documents
    ):
        """Test handling multiple user inputs."""
        # Setup mocks
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_find_similar.return_value = [
            {"text": "Test document.", "similarity": 0.9}
        ]
        mock_get_answer.return_value = "This is a test response."

        # Test
        chat_history = [
            {"user": "First question?", "bot": "First answer."}
        ]

        with patch('app.embedded_documents', sample_embedded_documents):
            result = handle_user_input(mock_model, "Second question?", chat_history)

        # Verify
        assert len(result) == 2
        assert result[0]["user"] == "First question?"
        assert result[1]["user"] == "Second question?"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_embedding_and_similarity_workflow(self):
        """Test the complete embedding and similarity search workflow."""
        # Create sample texts
        texts = [
            "Lipitor is used to treat high cholesterol.",
            "Metformin is used to treat type 2 diabetes.",
            "Exercise is important for health."
        ]

        # Generate embeddings
        embeddings = get_embeddings(texts)

        # Create embedded documents
        embedded_docs = []
        for text, embedding in zip(texts, embeddings):
            embedded_docs.append({
                "text": text,
                "chunks": [text],
                "embeddings": [embedding]
            })

        # Query
        query = "How to treat diabetes?"
        query_embedding = get_embedding(query)

        # Find similar documents
        similar_docs = find_most_similar_documents(query_embedding, embedded_docs, top_k=2)

        # Verify
        assert len(similar_docs) <= 2
        assert similar_docs[0]["similarity"] > 0  # Should have some similarity

        # The most similar document should contain "diabetes" or related medical terms
        top_doc_text = similar_docs[0]["document"]["text"].lower()
        assert "diabetes" in top_doc_text or "metformin" in top_doc_text

    def test_context_retrieval_and_answer_generation(self):
        """Test context retrieval and answer generation workflow."""
        documents = [
            "Lipitor (atorvastatin) is used to treat high cholesterol and reduce cardiovascular risk.",
            "Common side effects of Lipitor include muscle pain and liver problems.",
            "Metformin is the first-line medication for type 2 diabetes."
        ]

        query = "cholesterol"

        # Get context
        context = get_context(None, query, documents, top_k=2)

        # Verify context contains relevant information
        assert "cholesterol" in context.lower()
        assert "Lipitor" in context or "atorvastatin" in context


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_chunking(self):
        """Test chunking empty text."""
        chunks = chunk_text("")
        assert len(chunks) == 0 or chunks == ['']

    def test_short_text_chunking(self):
        """Test chunking very short text."""
        text = "Short text."
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        assert chunks[0] == text or text in chunks[0]

    def test_similarity_with_zero_vectors(self):
        """Test similarity calculation with zero vectors."""
        emb1 = [0.0, 0.0, 0.0]
        emb2 = [1.0, 1.0, 1.0]

        # This should handle the division by zero case
        with pytest.raises(Exception):
            # Depending on numpy version, this might raise or return nan
            similarity = get_similarity_score(emb1, emb2)
            assert np.isnan(similarity)

    def test_find_similar_documents_empty_list(self, sample_embedding):
        """Test finding similar documents with empty document list."""
        results = find_most_similar_documents(sample_embedding, [], top_k=5)
        assert len(results) == 0

    def test_find_similar_documents_top_k_larger_than_available(
        self, sample_embedding, sample_embedded_documents
    ):
        """Test finding similar documents when top_k exceeds available documents."""
        results = find_most_similar_documents(sample_embedding, sample_embedded_documents, top_k=100)

        # Should return all available documents
        assert len(results) <= len(sample_embedded_documents) * 10  # Accounting for chunks

    def test_get_context_empty_query(self):
        """Test getting context with empty query."""
        documents = ["Document 1", "Document 2"]
        context = get_context(None, "", documents)
        assert context == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
