# Further Reading & Resources

A curated list of tools, APIs, and tutorials that extend what you built in this series.

---

## Retrieval-Augmented Generation (RAG)

### LangChain RAG Tutorial
**URL:** https://python.langchain.com/docs/tutorials/rag/  
**Description:** Official LangChain walkthrough covering document loading, splitting, embedding, vector stores, and retrieval chains — the same concepts from this tutorial, now using LangChain's higher-level managed abstractions instead of manual code.

### OpenAI RAG Cookbook — File Search with the Responses API
**URL:** https://cookbook.openai.com/examples/file_search_responses  
**Description:** OpenAI's practical guide for building RAG with the Responses API's built-in file search. Shows how to replace the manual cosine similarity loop with a managed cloud retrieval backend. Great next step after finishing this tutorial.

---

## Vector Databases

### Pinecone — Managed Vector Database
**URL:** https://docs.pinecone.io/guides/get-started/quickstart  
**Description:** Serverless vector database with a Python SDK. A production-grade replacement for the `embeddings.json` cache built in Module 2. Supports metadata filtering, namespaces, and hybrid search (keyword + vector). Free tier available.

### ChromaDB — Local / Embedded Vector Database
**URL:** https://docs.trychroma.com/getting-started  
**Description:** Lightweight open-source embedding database that runs locally (or as a server). Drop-in upgrade from manual JSON storage; integrates directly with LangChain via `Chroma.from_documents()`. The `nymble_chatbot` project already imports Chroma — this is the natural next upgrade from the flat JSON cache.

### GCP Vertex AI Search
**URL:** https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction  
**Description:** Google Cloud's fully managed RAG infrastructure. Handles document ingestion, chunking, embedding, and retrieval as a service. Relevant for production deployments that need Google Cloud scale and compliance guarantees.

---

## Language Models & Inference

### OpenAI Responses API
**URL:** https://platform.openai.com/docs/guides/responses  
**Description:** OpenAI's latest inference API with built-in tool-use, file search, and conversation threading. Replaces the local GPT4All model with a cloud-hosted LLM — no `.gguf` file required. Ideal when you want to focus on application logic rather than local model management.

### GPT4All Python Documentation
**URL:** https://docs.gpt4all.io/gpt4all_python/home.html  
**Description:** Reference documentation for the Python bindings used in Modules 4 and 5. Explains `model.generate()` parameters (`top_k`, `streaming`, `max_tokens`, context window) and how to list available models.

### Sentence Transformers
**URL:** https://www.sbert.net/docs/sentence_transformer/usage/usage.html  
**Description:** Documentation for the `sentence-transformers` library and the `all-MiniLM-L6-v2` model used in Module 2. Covers `.encode()` parameters, batch processing, choosing between embedding models, and fine-tuning.

---

## Embeddings Deep Dive

### Vector Similarity Explained (Pinecone)
**URL:** https://www.pinecone.io/learn/vector-similarity/  
**Description:** Visual explainer of dot product, cosine similarity, and Euclidean distance — the math behind Module 3's `get_similarity_score()` function. Includes interactive diagrams and Python examples.

### OpenAI Text Embeddings Guide
**URL:** https://platform.openai.com/docs/guides/embeddings  
**Description:** Explains embedding dimensions, how to choose between models, and best practices for chunking. Useful context for understanding why `all-MiniLM-L6-v2` outputs 384-dimensional vectors compared to 1 536 for OpenAI's `text-embedding-ada-002`.

---

## Deployment & Production

### Streamlit Community Cloud
**URL:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app  
**Description:** How to deploy a `streamlit run app.py` application to a public URL for free — the simplest path from local demo to a shareable link your team can access.

### Docker for Python Applications
**URL:** https://docs.docker.com/language/python/  
**Description:** Official Docker guide for containerising Python applications. Directly relevant to the `DockerFile` already present in the `nymble_chatbot` project — follow this guide to build and run the chatbot as a container.
