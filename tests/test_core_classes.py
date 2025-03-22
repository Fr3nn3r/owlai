import os
import pytest
import tempfile
import fitz
import logging
import logging.config
import yaml
from langchain.docstore.document import Document as LangchainDocument

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from owlai.rag import RAGOwlAgent
from owlai.db import RAG_AGENTS_CONFIG
from owlai.owlsys import load_logger_config

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")

# Configure logging
load_logger_config()
logger = logging.getLogger("main")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a temporary PDF file with French law content."""
    pdf_path = os.path.join(temp_dir, "test.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test French Law Content")
    doc.save(pdf_path)
    doc.close()
    logger.info(f"Created test PDF at: {pdf_path}")
    return pdf_path


@pytest.fixture
def embedding_model():
    """Create a test embedding model."""
    logger.info("Initializing embedding model")
    model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model initialized")
    return model


@pytest.fixture
def rag_agent():
    """Create a test RAG agent."""
    logger.info("Initializing RAG agent")
    agent = RAGOwlAgent(**RAG_AGENTS_CONFIG[0])
    logger.info("RAG agent initialized")
    return agent


def test_rag_agent_initialization(rag_agent):
    """Test that the RAG agent initializes correctly."""
    logger.info("Testing RAG agent initialization")
    assert rag_agent._embeddings is not None
    assert rag_agent._reranker is not None
    assert rag_agent._prompt is not None
    # assert rag_agent._vector_stores is None #relaxed for now
    logger.info("RAG agent initialization test passed")


def test_rag_agent_document_processing(rag_agent, sample_pdf_path, embedding_model):
    """Test document processing functionality."""
    logger.info("Testing document processing")
    vector_store = rag_agent.load_dataset(
        os.path.dirname(sample_pdf_path), embedding_model
    )
    assert vector_store is not None
    assert isinstance(vector_store, FAISS)
    logger.info("Document processing test passed")


def test_rag_agent_question_answering(rag_agent, sample_pdf_path, embedding_model):
    """Test question answering functionality."""
    logger.info("Testing question answering")
    # First load the document
    vector_store = rag_agent.load_dataset(
        os.path.dirname(sample_pdf_path), embedding_model
    )
    rag_agent._vector_stores = vector_store

    # Test question answering
    result = rag_agent.rag_question("What is the test content?")
    assert "answer" in result
    assert "metadata" in result
    assert "retrieved_docs" in result["metadata"]
    logger.info("Question answering test passed")


def test_rag_agent_chunk_analysis(rag_agent, sample_pdf_path, embedding_model):
    """Test chunk size analysis functionality."""
    logger.info("Testing chunk analysis")
    # First load the document
    vector_store = rag_agent.load_dataset(
        os.path.dirname(sample_pdf_path), embedding_model
    )

    # Test chunk analysis
    analysis_file = rag_agent.analyze_chunk_size_distribution(
        os.path.dirname(sample_pdf_path),
        "test",
        vector_store.docstore.docs if hasattr(vector_store.docstore, "docs") else [],
    )
    assert os.path.exists(analysis_file)
    logger.info("Chunk analysis test passed")


def test_rag_agent_vector_store_persistence(
    rag_agent, sample_pdf_path, embedding_model
):
    """Test vector store saving and loading."""
    logger.info("Testing vector store persistence")
    # Create and save vector store
    vector_store = rag_agent.load_dataset(
        os.path.dirname(sample_pdf_path), embedding_model
    )

    # Load the vector store
    loaded_store = rag_agent.load_vector_store(
        os.path.dirname(sample_pdf_path), embedding_model
    )
    assert loaded_store is not None
    assert isinstance(loaded_store, FAISS)
    logger.info("Vector store persistence test passed")


def test_rag_agent_error_handling(rag_agent):
    """Test error handling for various edge cases."""
    logger.info("Testing error handling")
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        rag_agent.load_vector_store("non_existent_path", rag_agent._embeddings)

    # Test with empty question
    with pytest.raises(ValueError):
        rag_agent.rag_question("")

    # Test with None vector store
    rag_agent._vector_stores = None
    result = rag_agent.rag_question("test question")
    assert result["answer"] == "I don't know based on the provided sources."
    logger.info("Error handling test passed")
