import pytest
from typing import List, Optional, Tuple, Any, Callable
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import tempfile
import os
from langchain_huggingface import HuggingFaceEndpoint
from owlai.rag_agent import RAGAgent
from owlai.vector_store import VectorStore
from owlai.document_parser import FrenchLawParser


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
def rag_agent(embedding_model):
    """Create a RAG agent for testing"""
    from owlai.rag import RAGOwlAgent

    return RAGOwlAgent(embedding_model=embedding_model)


def test_rag_agent_creation(rag_agent):
    """Test that a RAG agent can be created"""
    assert rag_agent is not None
    assert rag_agent.embedding_model is not None


def test_rag_agent_load_dataset(rag_agent, sample_documents, tmp_path):
    """Test loading a dataset into the vector store"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    assert vector_store is not None
    assert len(vector_store.docstore.docs) == len(sample_documents)


def test_rag_agent_search(rag_agent, sample_documents, tmp_path):
    """Test searching the vector store"""
    # First load the documents
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    # Perform a search
    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)

    assert len(results) > 0
    assert isinstance(results[0], LangchainDocument)
    assert "French law" in results[0].page_content.lower()


def test_rag_agent_empty_search(rag_agent, sample_documents, tmp_path):
    """Test searching with an empty query"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    with pytest.raises(ValueError):
        rag_agent.search(query="", k=1, vector_store=vector_store)


def test_rag_agent_invalid_k(rag_agent, sample_documents, tmp_path):
    """Test searching with invalid k parameter"""
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    with pytest.raises(ValueError):
        rag_agent.search(query="test", k=0, vector_store=vector_store)


def test_rag_agent_save_load_vector_store(rag_agent, sample_documents, tmp_path):
    """Test saving and loading the vector store"""
    # Create and save vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=sample_documents,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    # Load the vector store
    loaded_store = rag_agent.load_dataset_from_split_docs(
        split_docs=[],  # Empty list since we're loading existing store
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    assert loaded_store is not None
    assert len(loaded_store.docstore.docs) == len(sample_documents)


def test_rag_agent_merge_documents(rag_agent, sample_documents, tmp_path):
    """Test merging new documents into existing vector store"""
    # Create initial vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=[sample_documents[0]],
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    # Merge additional document
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=[sample_documents[1]],
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    assert len(vector_store.docstore.docs) == 2


def test_rag_context_consistency():
    """Test that the documents used in the chain's context match the retrieved documents."""
    # Initialize models
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    # Create RAG agent
    agent = RAGAgent(embedding_model=embedding_model, llm=llm)

    # Create test documents with known content
    test_docs = [
        LangchainDocument(
            page_content="This is test document 1 about French law.",
            metadata={"source": "test1.txt", "title": "Test 1"},
        ),
        LangchainDocument(
            page_content="This is test document 2 about French law.",
            metadata={"source": "test2.txt", "title": "Test 2"},
        ),
        LangchainDocument(
            page_content="This is test document 3 about French law.",
            metadata={"source": "test3.txt", "title": "Test 3"},
        ),
    ]

    # Add documents to vector store
    agent.vector_store.add_documents(test_docs)

    # Make a query
    query = "What is French law?"
    result = agent.query(query)

    # Get the chain's source documents
    chain_docs = result.get("source_documents", [])

    # Get the documents we retrieved directly
    retrieved_docs = agent.vector_store.similarity_search(query, k=4)

    # Verify that the documents match
    assert len(chain_docs) == len(retrieved_docs), "Number of documents should match"

    # Compare document contents
    for chain_doc, retrieved_doc in zip(chain_docs, retrieved_docs):
        assert (
            chain_doc.page_content == retrieved_doc.page_content
        ), "Document contents should match"
        assert (
            chain_doc.metadata == retrieved_doc.metadata
        ), "Document metadata should match"

    # Verify the answer is not empty
    assert result["answer"], "Answer should not be empty"

    # Verify metadata contains correct number of sources
    assert result["metadata"]["num_docs_retrieved"] == len(
        retrieved_docs
    ), "Number of retrieved documents should match metadata"
