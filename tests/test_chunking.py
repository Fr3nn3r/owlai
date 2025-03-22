import pytest
from typing import List
from langchain.docstore.document import Document
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
def french_law_parser():
    """Fixture to provide a French law parser"""
    from owlai.document_parser import FrenchLawParser

    return FrenchLawParser()


def test_split_documents(french_law_parser, sample_pdf_path):
    """Test splitting documents into chunks"""
    # Load document
    documents = french_law_parser.parse(sample_pdf_path)

    # Split into chunks
    chunk_size = 50  # Small chunk size for testing
    split_docs = french_law_parser.split(documents, chunk_size=chunk_size)

    assert len(split_docs) > 0
    assert all(isinstance(doc, Document) for doc in split_docs)
    assert all(len(doc.page_content) <= chunk_size for doc in split_docs)


def test_split_documents_empty(french_law_parser):
    """Test splitting empty document list"""
    split_docs = french_law_parser.split([], chunk_size=50)
    assert len(split_docs) == 0


def test_split_documents_single_chunk(french_law_parser, sample_pdf_path):
    """Test splitting documents with large chunk size"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=1000)

    assert len(split_docs) == 1
    assert len(split_docs[0].page_content) <= 1000


def test_split_documents_metadata_preservation(french_law_parser, sample_pdf_path):
    """Test that metadata is preserved in chunks"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=50)

    for doc in split_docs:
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert "title" in doc.metadata


def test_split_documents_overlap(french_law_parser, sample_pdf_path):
    """Test splitting documents with overlap"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=50, chunk_overlap=10)

    # Check that chunks overlap
    for i in range(len(split_docs) - 1):
        current_chunk = split_docs[i].page_content
        next_chunk = split_docs[i + 1].page_content
        overlap = set(current_chunk.split()) & set(next_chunk.split())
        assert len(overlap) > 0


def test_split_documents_no_overlap(french_law_parser, sample_pdf_path):
    """Test splitting documents without overlap"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=50, chunk_overlap=0)

    # Check that chunks don't overlap
    for i in range(len(split_docs) - 1):
        current_chunk = split_docs[i].page_content
        next_chunk = split_docs[i + 1].page_content
        overlap = set(current_chunk.split()) & set(next_chunk.split())
        assert len(overlap) == 0


def test_split_documents_sentence_boundary(french_law_parser, sample_pdf_path):
    """Test that chunks respect sentence boundaries"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=50, chunk_overlap=0)

    for doc in split_docs:
        content = doc.page_content
        # Check that chunks don't break in the middle of sentences
        if content.endswith((".", "!", "?")):
            assert True
        else:
            # If not ending with sentence boundary, should be last chunk
            assert doc == split_docs[-1]


def test_split_documents_paragraph_boundary(french_law_parser, sample_pdf_path):
    """Test that chunks respect paragraph boundaries"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=50, chunk_overlap=0)

    for doc in split_docs:
        content = doc.page_content
        # Check that chunks don't break in the middle of paragraphs
        if content.endswith("\n"):
            assert True
        else:
            # If not ending with newline, should be last chunk
            assert doc == split_docs[-1]


def test_split_documents_chunk_size_distribution(
    french_law_parser, sample_pdf_path, tmp_path
):
    """Test analyzing chunk size distribution"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=50)

    distribution_file = french_law_parser.analyze_chunk_size_distribution(
        str(tmp_path), "test.pdf", split_docs, "thenlper/gte-small"
    )

    assert os.path.exists(distribution_file)
    assert distribution_file.endswith(".png")
