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


def test_load_fr_law_pdf(sample_pdf_path):
    """Test loading a French law PDF file"""
    from fr_law_load_docs import load_fr_law_pdf

    documents = load_fr_law_pdf(sample_pdf_path)

    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)
    assert "Code de commerce" in documents[0].page_content
    assert "Article 1" in documents[0].page_content


def test_load_fr_law_pdf_invalid_file():
    """Test loading with invalid file path"""
    from fr_law_load_docs import load_fr_law_pdf

    with pytest.raises(FileNotFoundError):
        load_fr_law_pdf("nonexistent.pdf")


def test_load_fr_law_pdf_empty_file():
    """Test loading an empty PDF file"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create an empty PDF
        doc = fitz.open()
        doc.new_page()
        doc.save(tmp.name)
        doc.close()

        from fr_law_load_docs import load_fr_law_pdf

        with pytest.raises(ValueError):
            load_fr_law_pdf(tmp.name)


def test_analyze_chunk_size_distribution(sample_pdf_path, tmp_path):
    """Test analyzing chunk size distribution"""
    from fr_law_load_docs import load_fr_law_pdf, analyze_chunk_size_distribution

    # Load and split documents
    documents = load_fr_law_pdf(sample_pdf_path)
    split_docs = [
        Document(
            page_content=doc.page_content[:100],  # Create chunks of max 100 chars
            metadata=doc.metadata,
        )
        for doc in documents
    ]

    # Analyze distribution
    distribution_file = analyze_chunk_size_distribution(
        str(tmp_path), "test.pdf", split_docs, "thenlper/gte-small"
    )

    assert os.path.exists(distribution_file)
    assert distribution_file.endswith(".png")


def test_document_curator(sample_pdf_path):
    """Test document curator function"""
    from fr_law_load_docs import load_fr_law_pdf, document_curator

    # Load document
    documents = load_fr_law_pdf(sample_pdf_path)

    # Curate document
    curated_content = document_curator(documents[0].page_content, sample_pdf_path)

    # Check that footer is removed
    assert "Document généré" not in curated_content
    assert "Code de commerce" in curated_content
    assert "Article 1" in curated_content


def test_document_curator_empty_content():
    """Test document curator with empty content"""
    from fr_law_load_docs import document_curator

    curated_content = document_curator("", "test.pdf")
    assert curated_content == ""


def test_document_curator_single_line():
    """Test document curator with single line content"""
    from fr_law_load_docs import document_curator

    curated_content = document_curator("Single line content", "test.pdf")
    assert curated_content == "Single line content"


def test_document_curator_no_footer():
    """Test document curator with content without footer"""
    from fr_law_load_docs import document_curator

    content = "Line 1\nLine 2\nLine 3"
    curated_content = document_curator(content, "test.pdf")
    assert curated_content == content
