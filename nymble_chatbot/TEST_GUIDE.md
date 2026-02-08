# Test Guide for Nymble Chatbot

This guide explains how to run and understand the test suite for the Nymble Chatbot application.

## Test File

- **Location**: `test_nymble_chatbot.py`
- **Framework**: pytest
- **Coverage**: utils.py, rag_agent.py, app.py

## Prerequisites

Ensure pytest is installed:

```bash
pip install pytest pytest-cov
```

## Running Tests

### Run All Tests

```bash
# From the nymble_chatbot directory
pytest test_nymble_chatbot.py -v
```

### Run Specific Test Classes

```bash
# Test only utility functions
pytest test_nymble_chatbot.py::TestUtilsFunctions -v

# Test only RAG agent functions
pytest test_nymble_chatbot.py::TestRAGAgentFunctions -v

# Test only app functions
pytest test_nymble_chatbot.py::TestAppFunctions -v

# Test integration scenarios
pytest test_nymble_chatbot.py::TestIntegration -v

# Test edge cases
pytest test_nymble_chatbot.py::TestEdgeCases -v
```

### Run Specific Tests

```bash
# Run a specific test function
pytest test_nymble_chatbot.py::TestUtilsFunctions::test_chunk_text_default_params -v
```

### Run with Coverage Report

```bash
# Generate coverage report
pytest test_nymble_chatbot.py --cov=. --cov-report=html

# View coverage report
# Open htmlcov/index.html in a browser
```

### Run with Detailed Output

```bash
# Show print statements and detailed output
pytest test_nymble_chatbot.py -v -s

# Show test durations
pytest test_nymble_chatbot.py -v --durations=10
```

## Test Structure

### 1. TestUtilsFunctions
Tests for `utils.py` functions:
- ✅ `test_save_and_load_embeddings` - Saving/loading embeddings to/from JSON
- ✅ `test_save_and_load_json` - JSON file operations
- ✅ `test_chunk_text_default_params` - Text chunking with default parameters
- ✅ `test_chunk_text_custom_params` - Text chunking with custom parameters
- ✅ `test_get_embedding` - Single text embedding generation
- ✅ `test_get_embeddings_multiple_texts` - Multiple text embeddings
- ✅ `test_get_similarity_score_identical` - Similarity of identical vectors
- ✅ `test_get_similarity_score_different` - Similarity of different vectors
- ✅ `test_get_similarity_score_opposite` - Similarity of opposite vectors
- ✅ `test_find_most_similar_documents` - Document similarity search
- ✅ `test_get_file_extension` - File extension extraction

### 2. TestRAGAgentFunctions
Tests for `rag_agent.py` functions:
- ✅ `test_get_context_with_matching_documents` - Context retrieval with matches
- ✅ `test_get_context_with_no_matches` - Context retrieval without matches
- ✅ `test_get_context_respects_top_k` - Top-k limiting
- ✅ `test_get_answer` - Basic answer generation
- ✅ `test_get_answer_with_context` - Context-aware answer generation
- ✅ `test_get_answer_with_context_no_answer` - Handling no relevant context

### 3. TestAppFunctions
Tests for `app.py` functions:
- ✅ `test_load_and_embed_documents` - Document loading and embedding
- ✅ `test_handle_user_input` - Single user interaction
- ✅ `test_handle_user_input_multiple_messages` - Multiple user interactions

### 4. TestIntegration
End-to-end integration tests:
- ✅ `test_embedding_and_similarity_workflow` - Complete embedding workflow
- ✅ `test_context_retrieval_and_answer_generation` - Complete Q&A workflow

### 5. TestEdgeCases
Edge cases and error handling:
- ✅ `test_empty_text_chunking` - Empty text handling
- ✅ `test_short_text_chunking` - Short text handling
- ✅ `test_similarity_with_zero_vectors` - Zero vector handling
- ✅ `test_find_similar_documents_empty_list` - Empty document list
- ✅ `test_find_similar_documents_top_k_larger_than_available` - Large top_k
- ✅ `test_get_context_empty_query` - Empty query handling

## Fixtures

The test suite uses several fixtures to provide consistent test data:

- `sample_text` - Long sample text for chunking tests
- `sample_embedding` - Single embedding vector
- `sample_embeddings` - Multiple embedding vectors
- `sample_embedded_documents` - Complete document structures
- `mock_model` - Mocked GPT4All model
- `temp_json_file` - Temporary file for I/O tests

## Mocking Strategy

Tests use mocking to avoid dependencies on:
- **GPT4All model**: Mocked to avoid loading large model files
- **PDF loading**: Mocked to avoid requiring actual PDF files
- **Streamlit**: Mocked to enable testing without UI
- **File I/O**: Uses temporary files or mocks where appropriate

## Common Issues and Solutions

### Issue: `ModuleNotFoundError`
**Solution**: Ensure you're running tests from the project root or nymble_chatbot directory

```bash
cd nymble_chatbot
pytest test_nymble_chatbot.py -v
```

### Issue: `ImportError` for dependencies
**Solution**: Install required dependencies

```bash
pip install -r ../general_requirements.txt
```

### Issue: Tests fail due to missing model
**Solution**: Tests use mocks, so the actual model shouldn't be needed. If errors persist, check that mocking is working correctly.

### Issue: Slow test execution
**Solution**: The embedding tests use real models by default. For faster tests, you can run specific test classes that don't require embeddings:

```bash
pytest test_nymble_chatbot.py::TestRAGAgentFunctions -v
pytest test_nymble_chatbot.py::TestEdgeCases -v
```

## Continuous Integration

To run tests in CI/CD pipelines:

```bash
# Install dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest test_nymble_chatbot.py --cov=. --cov-report=xml --cov-report=term

# Tests will fail CI if any test fails (exit code != 0)
```

## Adding New Tests

When adding new functionality:

1. Add corresponding test cases to the appropriate test class
2. Use fixtures for common test data
3. Mock external dependencies (models, file I/O, APIs)
4. Test both success and failure scenarios
5. Include edge cases

Example:

```python
def test_new_feature(self, sample_data, mock_model):
    """Test description."""
    # Arrange
    input_data = sample_data

    # Act
    result = your_function(input_data)

    # Assert
    assert result is not None
    assert expected_condition
```

## Test Coverage Goals

- **Target**: >80% code coverage
- **Critical paths**: 100% coverage for core functions
- **Edge cases**: All known edge cases covered

Check current coverage:

```bash
pytest test_nymble_chatbot.py --cov=. --cov-report=term-missing
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
