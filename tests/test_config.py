"""Test configurations for RAG-related tests."""

from typing import Dict, Any, List

# Test-specific RAG configuration
TEST_RAG_CONFIG = {
    "name": "test-rag-agent",
    "description": "Test RAG agent for unit testing",
    "args_schema": {
        "query": {
            "type": "string",
            "description": "Test query for RAG agent",
        }
    },
    "model_provider": "test",
    "model_name": "test-model",
    "max_tokens": 100,
    "temperature": 0.1,
    "context_size": 100,
    "tools_names": [],
    "system_prompt": "You are a test RAG agent. Answer the following question based on the context: {question}\n\nContext: {context}",
    "default_queries": [],
    "retriever": {
        "num_retrieved_docs": 2,
        "num_docs_final": 1,
        "embeddings_model_name": "thenlper/gte-small",
        "reranker_name": "colbert-ir/colbertv2.0",
        "input_data_folders": [],
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
        "multi_process": False,
    },
}

# Test documents for RAG testing
TEST_DOCUMENTS = [
    {
        "page_content": "This is a test document about French law.",
        "metadata": {"source": "test1.pdf", "page": 1},
    },
    {
        "page_content": "Another test document with legal content.",
        "metadata": {"source": "test2.pdf", "page": 1},
    },
    {
        "page_content": "This document contains information about legal procedures.",
        "metadata": {"source": "test3.pdf", "page": 1},
    },
]
