import pytest
from typing import List
from langchain.docstore.document import Document
import os
import tempfile
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


def test_french_law_parser_creation(french_law_parser):
    """Test that a French law parser can be created"""
    assert french_law_parser is not None


def test_french_law_parser_parse(sample_pdf_path, french_law_parser):
    """Test parsing a French law document"""
    documents = french_law_parser.parse(sample_pdf_path)
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)

    # Check metadata
    assert "title" in documents[0].metadata
    assert "last_modification_fr" in documents[0].metadata
    assert "doc_generated_on_fr" in documents[0].metadata


def test_french_law_parser_split(french_law_parser, sample_pdf_path):
    """Test splitting a French law document"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=100)

    assert len(split_docs) > 0
    assert all(isinstance(doc, Document) for doc in split_docs)
    assert all(len(doc.page_content) <= 100 for doc in split_docs)


def test_french_law_parser_footer_extraction(sample_pdf_path, french_law_parser):
    """Test footer extraction from French law document"""
    footer = french_law_parser.extract_footer(sample_pdf_path)
    assert "Code de commerce" in footer
    assert "Dernière modification" in footer
    assert "Document généré" in footer


def test_french_law_parser_metadata_extraction(sample_pdf_path, french_law_parser):
    """Test metadata extraction from French law document"""
    footer = french_law_parser.extract_footer(sample_pdf_path)
    metadata = french_law_parser.extract_metadata_fr_law(footer)

    assert metadata["title"] == "Code de commerce"
    assert metadata["last_modification_fr"] == "01 mars 2025"
    assert metadata["doc_generated_on_fr"] == "12 mars 2025"


def test_french_law_parser_invalid_file(french_law_parser):
    """Test parser behavior with invalid file"""
    with pytest.raises(FileNotFoundError):
        french_law_parser.parse("nonexistent.pdf")


def test_french_law_parser_empty_file():
    """Test parser behavior with empty file"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create an empty PDF
        doc = fitz.open()
        doc.new_page()
        doc.save(tmp.name)
        doc.close()

        parser = FrenchLawParser()
        with pytest.raises(ValueError):
            parser.parse(tmp.name)


def test_french_law_parser_chunk_size_distribution(
    french_law_parser, sample_pdf_path, tmp_path
):
    """Test chunk size distribution analysis"""
    documents = french_law_parser.parse(sample_pdf_path)
    split_docs = french_law_parser.split(documents, chunk_size=100)

    distribution_file = french_law_parser.analyze_chunk_size_distribution(
        str(tmp_path), "test.pdf", split_docs, "thenlper/gte-small"
    )

    assert os.path.exists(distribution_file)
    assert distribution_file.endswith(".png")
