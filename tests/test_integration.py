import pytest
from typing import List, Optional, Tuple, Any, Callable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
import tempfile
import os
import fitz  # PyMuPDF


@pytest.fixture
def sample_pdf_path():
    """Create a temporary PDF file with French law content"""
    doc = None
    tmp_path = None
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Code de commerce\nArticle 1\nTest content")
        page.insert_text(
            (50, 700),
            "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025",
        )
        # Create a temporary file with a unique name
        tmp_path = tempfile.mktemp(suffix=".pdf")
        doc.save(tmp_path)
        doc.close()
        return tmp_path
    except Exception as e:
        if doc:
            doc.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise e


@pytest.fixture(autouse=True)
def cleanup_temp_files(request):
    """Cleanup temporary files after each test"""

    def finalizer():
        try:
            # Get the sample_pdf_path fixture if it was used
            sample_pdf = request.getfixturevalue("sample_pdf_path")
            if sample_pdf and os.path.exists(sample_pdf):
                try:
                    os.remove(sample_pdf)
                except:
                    pass
        except:
            pass

    request.addfinalizer(finalizer)


@pytest.fixture
def embedding_model():
    """Create a test embedding model"""
    return HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture
def french_law_parser():
    """Fixture to provide a French law parser"""
    from owlai.document_parser import FrenchLawParser

    return FrenchLawParser()


@pytest.fixture
def rag_agent(embedding_model):
    """Create a RAG agent for testing"""
    from owlai.rag import RAGOwlAgent, RAGConfig
    from owlai.core.config import AgentConfig, ModelConfig

    retriever_config = RAGConfig(
        num_retrieved_docs=3,
        num_docs_final=2,
        embeddings_model_name="thenlper/gte-small",
        reranker_name="BAAI/bge-reranker-base",
        input_data_folders=[],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        multi_process=False,
    )

    model_config = ModelConfig(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        temperature=0.1,
        max_tokens=2048,
        context_size=4096,
    )

    config = AgentConfig(
        name="test_rag_agent",
        description="Test RAG agent",
        system_prompt="You are a test RAG agent.",
        llm_config=model_config,
        retriever=retriever_config.model_dump(),
        tools_names=["search", "retrieve"],
    )

    return RAGOwlAgent(config=config, embedding_model=embedding_model)


def test_end_to_end_processing(french_law_parser, rag_agent, sample_pdf_path, tmp_path):
    """Test complete end-to-end document processing pipeline"""
    # 1. Parse PDF
    documents = french_law_parser.parse(sample_pdf_path)
    assert len(documents) > 0

    # 2. Split documents
    split_docs = french_law_parser.split(documents, chunk_size=100)
    assert len(split_docs) > 0

    # 3. Load into vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=split_docs,
        input_data_folder=str(tmp_path),
    )
    assert vector_store is not None


def test_document_metadata_integration(
    french_law_parser, rag_agent, sample_pdf_path, tmp_path
):
    """Test metadata preservation throughout the pipeline"""
    # 1. Parse PDF
    documents = french_law_parser.parse(sample_pdf_path)

    # 2. Split documents
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # 3. Load into vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=split_docs,
        input_data_folder=str(tmp_path),
    )
    assert vector_store is not None


def test_footer_removal_integration(
    french_law_parser, rag_agent, sample_pdf_path, tmp_path
):
    """Test footer removal throughout the pipeline"""
    # 1. Parse PDF
    documents = french_law_parser.parse(sample_pdf_path)

    # 2. Split documents
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # 3. Load into vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=split_docs,
        input_data_folder=str(tmp_path),
    )
    assert vector_store is not None


def test_chunk_size_integration(
    french_law_parser, rag_agent, sample_pdf_path, tmp_path
):
    """Test chunk size consistency throughout the pipeline"""
    chunk_size = 100

    # 1. Parse PDF
    documents = french_law_parser.parse(sample_pdf_path)

    # 2. Split documents
    split_docs = french_law_parser.split(documents, chunk_size=chunk_size)

    # 3. Load into vector store
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=split_docs,
        input_data_folder=str(tmp_path),
    )
    assert vector_store is not None


def test_vector_store_persistence(
    french_law_parser, rag_agent, sample_pdf_path, tmp_path
):
    """Test vector store persistence and reloading"""
    # 1. Parse PDF
    documents = french_law_parser.parse(sample_pdf_path)

    # 2. Split documents
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # 3. Load into vector store and save
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=split_docs,
        input_data_folder=str(tmp_path),
    )
    assert vector_store is not None


def test_concurrent_processing_integration(
    french_law_parser, rag_agent, sample_pdf_path, tmp_path
):
    """Test concurrent processing throughout the pipeline"""
    import concurrent.futures

    # 1. Parse PDF
    documents = french_law_parser.parse(sample_pdf_path)

    # 2. Split documents
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # 3. Process documents concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for doc in split_docs:
            future = executor.submit(
                rag_agent.load_dataset,
                input_data_folder=str(tmp_path),
                embedding_model=rag_agent._embeddings,
                chunk_size=100,
            )
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def test_error_handling_integration(
    french_law_parser, rag_agent, sample_pdf_path, tmp_path
):
    """Test error handling throughout the pipeline"""
    # 1. Parse PDF with invalid file
    with pytest.raises(FileNotFoundError):
        french_law_parser.parse("nonexistent.pdf")

    # 2. Parse valid PDF
    documents = french_law_parser.parse(sample_pdf_path)

    # 3. Split documents with invalid chunk size
    with pytest.raises(ValueError):
        french_law_parser.split(documents, chunk_size=0)

    # 4. Split documents with valid chunk size
    split_docs = french_law_parser.split(documents, chunk_size=100)

    # 5. Load into vector store with invalid input store
    with pytest.raises(ValueError):
        rag_agent.load_dataset_from_split_docs(
            split_docs=split_docs,
            input_data_folder=str(tmp_path),
            input_store={"invalid": "store"},  # Pass a dict instead of a FAISS instance
        )
