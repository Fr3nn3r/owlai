import pytest
from typing import List, Optional, Tuple, Any, Callable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
import tempfile
import os


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        LangchainDocument(
            page_content="This is a test document about French law.",
            metadata={"source": "test1.pdf", "page": 1},
        ),
        LangchainDocument(
            page_content="Another test document with legal content.",
            metadata={"source": "test2.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def embedding_model():
    """Create a test embedding model"""
    return HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture
def vector_store(embedding_model, sample_documents):
    """Create a test vector store"""
    from owlai.vector_store import VectorStore

    store = VectorStore(embedding_model=embedding_model)
    store.add_documents(sample_documents)
    return store


def test_vector_store_creation(embedding_model):
    """Test creating a vector store"""
    from owlai.vector_store import VectorStore

    store = VectorStore(embedding_model=embedding_model)
    assert store is not None
    assert store.embedding_model == embedding_model
    assert len(store.docstore.docs) == 0


def test_vector_store_add_documents(vector_store, sample_documents):
    """Test adding documents to vector store"""
    assert len(vector_store.docstore.docs) == len(sample_documents)
    assert all(
        isinstance(doc, LangchainDocument)
        for doc in vector_store.docstore.docs.values()
    )


def test_vector_store_search(vector_store):
    """Test searching the vector store"""
    results = vector_store.similarity_search(query="French law", k=1)

    assert len(results) > 0
    assert isinstance(results[0], LangchainDocument)
    assert "French law" in results[0].page_content.lower()


def test_vector_store_empty_search(vector_store):
    """Test searching with empty query"""
    with pytest.raises(ValueError):
        vector_store.similarity_search(query="", k=1)


def test_vector_store_invalid_k(vector_store):
    """Test searching with invalid k parameter"""
    with pytest.raises(ValueError):
        vector_store.similarity_search(query="test", k=0)


def test_vector_store_save_load(vector_store, tmp_path):
    """Test saving and loading vector store"""
    # Save vector store
    save_path = os.path.join(tmp_path, "vector_store")
    vector_store.save_local(save_path)

    # Create new vector store and load
    from owlai.vector_store import VectorStore

    new_store = VectorStore(embedding_model=vector_store.embedding_model)
    new_store.load_local(save_path)

    assert len(new_store.docstore.docs) == len(vector_store.docstore.docs)


def test_vector_store_merge_documents(vector_store, sample_documents):
    """Test merging documents into vector store"""
    # Create additional documents
    new_docs = [
        LangchainDocument(
            page_content="New test document.",
            metadata={"source": "test3.pdf", "page": 1},
        )
    ]

    # Merge documents
    vector_store.add_documents(new_docs)

    assert len(vector_store.docstore.docs) == len(sample_documents) + len(new_docs)


def test_vector_store_metadata_preservation(vector_store, sample_documents):
    """Test that document metadata is preserved"""
    for doc in vector_store.docstore.docs.values():
        assert "source" in doc.metadata
        assert "page" in doc.metadata


def test_vector_store_duplicate_documents(vector_store, sample_documents):
    """Test handling of duplicate documents"""
    # Add same documents again
    vector_store.add_documents(sample_documents)

    # Should not add duplicates
    assert len(vector_store.docstore.docs) == len(sample_documents)
