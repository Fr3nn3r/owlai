import pytest
from typing import List, Optional, Tuple, Any, Callable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
import tempfile
import os
from tests.test_config import TEST_RAG_CONFIG, TEST_DOCUMENTS


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [LangchainDocument(**doc) for doc in TEST_DOCUMENTS]


@pytest.fixture
def embedding_model():
    """Create a test embedding model"""
    return HuggingFaceEmbeddings(
        model_name=TEST_RAG_CONFIG["retriever"]["embeddings_model_name"],
        model_kwargs=TEST_RAG_CONFIG["retriever"]["model_kwargs"],
        encode_kwargs=TEST_RAG_CONFIG["retriever"]["encode_kwargs"],
    )


@pytest.fixture
def rag_agent(embedding_model):
    """Create a RAG agent for testing"""
    from owlai.rag import RAGOwlAgent

    return RAGOwlAgent(**TEST_RAG_CONFIG)


def test_rag_search_basic(rag_agent, sample_documents, tmp_path):
    """Test basic search functionality"""
    # Load documents into vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    # Perform search
    results = rag_agent.search(query="French law", k=2, vector_store=vector_store)

    assert len(results) == 2
    assert all(isinstance(doc, LangchainDocument) for doc in results)
    assert any("French law" in doc.page_content.lower() for doc in results)


def test_rag_search_empty_query(rag_agent, sample_documents, tmp_path):
    """Test search with empty query"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    with pytest.raises(ValueError):
        rag_agent.search(query="", k=1, vector_store=vector_store)


def test_rag_search_invalid_k(rag_agent, sample_documents, tmp_path):
    """Test search with invalid k parameter"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    with pytest.raises(ValueError):
        rag_agent.search(query="test", k=0, vector_store=vector_store)


def test_rag_search_k_larger_than_docs(rag_agent, sample_documents, tmp_path):
    """Test search with k larger than number of documents"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    results = rag_agent.search(
        query="test", k=10, vector_store=vector_store  # Larger than number of documents
    )

    assert len(results) == len(sample_documents)


def test_rag_search_no_matches(rag_agent, sample_documents, tmp_path):
    """Test search with query that has no matches"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    results = rag_agent.search(
        query="nonexistent content", k=2, vector_store=vector_store
    )

    assert (
        len(results) > 0
    )  # Should still return some results due to semantic similarity


def test_rag_search_metadata_preservation(rag_agent, sample_documents, tmp_path):
    """Test that document metadata is preserved in search results"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)

    assert "source" in results[0].metadata
    assert "page" in results[0].metadata


def test_rag_search_similarity_scores(rag_agent, sample_documents, tmp_path):
    """Test that search results are ordered by similarity"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    results = rag_agent.search(query="French law", k=2, vector_store=vector_store)

    # First result should be more relevant than second
    assert "French law" in results[0].page_content.lower()
    assert any("French law" not in doc.page_content.lower() for doc in results[1:])


def test_rag_search_empty_store(rag_agent, tmp_path):
    """Test search with empty vector store"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=[],
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    results = rag_agent.search(query="test", k=1, vector_store=vector_store)

    assert len(results) == 0
