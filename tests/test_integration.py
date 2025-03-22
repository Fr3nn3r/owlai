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
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create a PDF with French law content
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Code de commerce\nArticle 1\nTest content")
        page.insert_text(
            (50, 700),
            "Code de commerce - Dernière modification le 01 mars 2025 - Document généré le 12 mars 2025",
        )
        doc.save(tmp.name)
        doc.close()
        return tmp.name


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
    from owlai.rag import RAGOwlAgent

    return RAGOwlAgent(embedding_model=embedding_model)


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
        embedding_model=rag_agent.embedding_model,
    )
    assert vector_store is not None

    # 4. Search in vector store
    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)
    assert len(results) > 0


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
        embedding_model=rag_agent.embedding_model,
    )

    # 4. Search and verify metadata
    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)

    # Check metadata preservation
    assert "source" in results[0].metadata
    assert "page" in results[0].metadata
    assert "title" in results[0].metadata


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
        embedding_model=rag_agent.embedding_model,
    )

    # 4. Search and verify footer removal
    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)

    # Check footer removal
    assert "Document généré" not in results[0].page_content
    assert "Dernière modification" not in results[0].page_content


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
        embedding_model=rag_agent.embedding_model,
    )

    # 4. Search and verify chunk size
    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)

    # Check chunk size
    assert len(results[0].page_content) <= chunk_size


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
        embedding_model=rag_agent.embedding_model,
    )

    # 4. Create new agent and load vector store
    new_rag_agent = RAGOwlAgent(embedding_model=rag_agent.embedding_model)
    new_vector_store = new_rag_agent.load_dataset_from_split_docs(
        split_docs=[],  # Empty list since we're loading existing store
        input_data_folder=str(tmp_path),
        embedding_model=new_rag_agent.embedding_model,
    )

    # 5. Verify persistence
    results = new_rag_agent.search(
        query="French law", k=1, vector_store=new_vector_store
    )
    assert len(results) > 0


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
                rag_agent.load_dataset_from_split_docs,
                split_docs=[doc],
                input_data_folder=str(tmp_path),
                embedding_model=rag_agent.embedding_model,
            )
            futures.append(future)

        # Wait for all tasks to complete
        vector_stores = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    # 4. Merge vector stores
    final_store = vector_stores[0]
    for store in vector_stores[1:]:
        final_store.merge(store)

    # 5. Verify results
    results = rag_agent.search(query="French law", k=1, vector_store=final_store)
    assert len(results) > 0


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

    # 5. Load into vector store with invalid embedding model
    with pytest.raises(ValueError):
        rag_agent.load_dataset_from_split_docs(
            split_docs=split_docs, input_data_folder=str(tmp_path), embedding_model=None
        )

    # 6. Load into vector store with valid embedding model
    vector_store = rag_agent.load_dataset_from_split_docs(
        split_docs=split_docs,
        input_data_folder=str(tmp_path),
        embedding_model=rag_agent.embedding_model,
    )

    # 7. Search with invalid query
    with pytest.raises(ValueError):
        rag_agent.search(query="", k=1, vector_store=vector_store)

    # 8. Search with valid query
    results = rag_agent.search(query="French law", k=1, vector_store=vector_store)
    assert len(results) > 0
